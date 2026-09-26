#include <SDL3/SDL.h>
#include <function/renderer/MaterialDescriptor.h>
#include <function/renderer/rhi/RhiTexture.h>
#include <function/renderer/shader/ShaderProgram.h>
#include <function/renderer/vk/VkDeviceContext.h>
#include <function/renderer/vk/VulkanRhiDevice.h>
#include <function/resources/InxMaterial/InxMaterial.h>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <array>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <unordered_map>

using namespace infernux;

static std::vector<char> ReadSpirv(const char *path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    assert(input);
    const auto size = static_cast<size_t>(input.tellg());
    assert(size > 0 && size % sizeof(uint32_t) == 0);
    std::vector<char> bytes(size);
    input.seekg(0);
    input.read(bytes.data(), static_cast<std::streamsize>(size));
    assert(input);
    return bytes;
}

int main(int argc, char **argv)
{
    assert(argc == 3);
    assert(SDL_Init(SDL_INIT_VIDEO));
    auto *window = SDL_CreateWindow("Material texture readiness", 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
    assert(window);
    vk::VkDeviceContext context;
    vk::DeviceConfig config;
    config.enableValidationLayers = true;
    assert(context.Initialize(window, config));
    auto &device = context.GetRhiDevice();

    ShaderProgram program;
    ShaderProgramVariantKey key;
    key.program.stages = {"Tests/MaterialTextureReadiness.vert", "Tests/MaterialTextureReadiness.frag"};
    key.program.revision = 1;
    key.target = ShaderCompileTarget::Forward;
    assert(program.Create(context.GetDevice(), ReadSpirv(argv[1]), ReadSpirv(argv[2]), key));
    std::unordered_map<std::string, uint32_t> slots;
    for (const auto &binding : program.GetDescriptorBindings())
        if (binding.set == 0 && binding.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
            slots.emplace(binding.name, binding.binding);
    assert(slots.size() == 2 && slots.find("texSampler") != slots.end() && slots.find("detailTex") != slots.end());

    rhi::TextureDesc textureDesc;
    textureDesc.format = rhi::PixelFormat::RGBA8UNorm;
    textureDesc.usage = rhi::TextureUsageFlags::Sampled;
    std::array<rhi::TextureHandle, 3> textures;
    std::array<rhi::TextureViewHandle, 3> views;
    for (size_t i = 0; i < textures.size(); ++i) {
        textures[i] = device.CreateTexture(textureDesc);
        rhi::TextureViewDesc viewDesc;
        viewDesc.texture = textures[i];
        viewDesc.format = textureDesc.format;
        views[i] = device.CreateTextureView(viewDesc);
        assert(textures[i].IsValid() && views[i].IsValid());
    }
    const auto sampler = device.CreateSampler({});
    assert(sampler.IsValid());
    const auto publication = [&](size_t index) {
        return std::make_shared<rhi::TextureGpuView>("test-texture-" + std::to_string(index), 1, textures[index],
                                                     views[index], sampler, 4, std::make_shared<int>(1),
                                                     rhi::PixelFormat::RGBA8UNorm);
    };
    std::array<std::shared_ptr<const rhi::TextureGpuView>, 3> gpuViews = {publication(0), publication(1),
                                                                          publication(2)};

    vk::VkDescriptorManager descriptorAllocator(context.GetDevice(), context.GetDeviceId());
    GpuRetirementQueue retirement;
    rhi::SubmissionSerial serial = 1;
    retirement.BindSerialSource([&] { return serial; });
    MaterialDescriptorManager descriptors;
    descriptors.Initialize(context.GetVmaAllocator(), context.GetDevice(), context.GetPhysicalDevice(),
                           &descriptorAllocator);
    descriptors.SetRetirementQueue(&retirement);
    descriptors.SetDefaultTexture(device.Resolve(views[0]), device.Resolve(sampler), gpuViews[0]);

    std::unordered_map<std::string, TextureResolveStatus> statuses = {{"icon-guid", TextureResolveStatus::Pending},
                                                                      {"detail-guid", TextureResolveStatus::Ready}};
    descriptors.SetTextureResolver([&](const std::string &guid, const std::string &, const MaterialTextureSampler *) {
        TextureResolveResult result;
        result.status = statuses.at(guid);
        if (result.status == TextureResolveStatus::Ready) {
            const size_t index = guid == "icon-guid" ? 1 : 2;
            result.binding.imageView = device.Resolve(views[index]);
            result.binding.sampler = device.Resolve(sampler);
            result.binding.gpuView = gpuViews[index];
        }
        return result;
    });

    InxMaterial material("texture-readiness", "Gizmo Icon");
    auto document = material.SerializeDocument();
    document["properties"]["texSampler"] = {{"type", static_cast<int>(MaterialPropertyType::Texture2D)},
                                            {"guid", "icon-guid"}};
    document["properties"]["detailTex"] = {{"type", static_cast<int>(MaterialPropertyType::Texture2D)},
                                           {"guid", "detail-guid"}};
    assert(material.DeserializeDocument(document));
    const auto materialKey = material.GetMaterialKey();
    auto *descriptor = descriptors.GetOrCreateDescriptorSet(material, program);
    assert(descriptor && descriptor->isValid);
    const auto iconSlot = slots.at("texSampler");
    const auto detailSlot = slots.at("detailTex");
    const auto check = [&](bool unresolved, bool pending, size_t iconView, size_t detailView) {
        assert(descriptors.HasUnresolvedExplicitTextureProperties(materialKey) == unresolved);
        assert(descriptors.HasPendingTextureProperties(materialKey) == pending);
        assert(descriptor->textureBindings.at(iconSlot).gpuView == gpuViews[iconView]);
        assert(descriptor->textureBindings.at(detailSlot).gpuView == gpuViews[detailView]);
        assert(descriptor->textureBindings.at(iconSlot).resolvedExplicitTexture == (iconView != 0));
        assert(descriptor->textureBindings.at(detailSlot).resolvedExplicitTexture == (detailView != 0));
    };

    // The dedicated icon draw must wait while its first explicit texture is missing.
    check(true, true, 0, 2);
    statuses["icon-guid"] = TextureResolveStatus::Failed;
    InxMaterial failedMaterial("failed-texture-readiness", "Gizmo Icon");
    document["name"] = failedMaterial.GetName();
    assert(failedMaterial.DeserializeDocument(document));
    auto *failedDescriptor = descriptors.GetOrCreateDescriptorSet(failedMaterial, program);
    assert(failedDescriptor && failedDescriptor->isValid);
    assert(descriptors.HasUnresolvedExplicitTextureProperties(failedMaterial.GetMaterialKey()));
    assert(!descriptors.HasPendingTextureProperties(failedMaterial.GetMaterialKey()));
    assert(failedDescriptor->textureBindings.at(iconSlot).gpuView == gpuViews[0]);
    assert(failedDescriptor->textureBindings.at(detailSlot).gpuView == gpuViews[2]);

    statuses["detail-guid"] = TextureResolveStatus::Pending;
    descriptors.ResolveTextureProperties(materialKey, material, program);
    check(true, true, 0, 2); // A bad icon cannot demote the other good texture.

    statuses["icon-guid"] = TextureResolveStatus::Ready;
    statuses["detail-guid"] = TextureResolveStatus::Ready;
    descriptors.ResolveTextureProperties(materialKey, material, program);
    check(false, false, 1, 2);

    statuses["icon-guid"] = TextureResolveStatus::Pending;
    statuses["detail-guid"] = TextureResolveStatus::Failed;
    descriptors.ResolveTextureProperties(materialKey, material, program);
    check(false, true, 1, 2); // Hot replacement retains both published GPU views.

    descriptors.Shutdown();
    retirement.Collect((std::numeric_limits<rhi::SubmissionSerial>::max)());
    descriptorAllocator.Destroy();
    program.Destroy();
    for (const auto view : views)
        device.Release(view);
    for (const auto texture : textures)
        device.Release(texture);
    device.Release(sampler);
    context.Destroy();
    SDL_DestroyWindow(window);
    SDL_Quit();
    std::cout << "Material texture readiness regression passed\n";
    return 0;
}
