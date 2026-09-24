#include <SDL3/SDL.h>
#include <function/renderer/InxVkCoreModular.h>
#include <function/renderer/VkShaderCache.h>
#include <function/renderer/gui/InxScreenUIRenderer.h>
#include <function/renderer/gui/InxTextLayout.h>
#include <function/renderer/rhi/RhiRenderTexture.h>
#include <function/renderer/vk/RenderGraph.h>
#include <function/renderer/vk/VkDeviceContext.h>
#include <function/renderer/vk/VulkanRhiDevice.h>
#include <function/resources/InxFileLoader/InxShaderLoader.hpp>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/ShaderAsset/ShaderStageLinker.h>
#include <function/scene/SceneManager.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <unordered_map>

using namespace infernux;

namespace infernux
{
struct ScreenUIVulkanTestAccess
{
    static VkPipeline ResolveMaterialPipeline(InxScreenUIRenderer &renderer, const UIShaderMaterialBinding &binding,
                                              ShaderProgramDomain domain, const rhi::GraphicsRenderingSignature *target)
    {
        return renderer.GetMaterialPipeline(binding, domain, target);
    }

    static VkDescriptorSet ResolveMaterialDescriptor(InxScreenUIRenderer &renderer,
                                                     const UIShaderMaterialBinding &binding,
                                                     const ShaderProgramArtifact &artifact)
    {
        return renderer.GetMaterialDescriptor(binding, artifact);
    }
};
} // namespace infernux

static std::shared_ptr<const ShaderProgramArtifact> CompileUiProgram(bool world, bool alternate, bool reload = false,
                                                                     bool materialProperties = false)
{
    const std::string domain = world ? "WorldUI" : "ScreenUI";
    const std::string vertex = "ShaderInfo { Name \"Tests/" + domain + "Vertex\" Capabilities [" + domain + "] }\n" +
                               (world ? R"glsl(
#version 450
layout(location=0) in vec3 aPosition;
layout(location=1) in vec2 aUV;
layout(location=2) in vec4 aColor;
layout(location=3) in vec2 aLocalPosition;
layout(location=4) in vec3 aAnchor;
layout(location=5) in vec2 aLocalOffset;
layout(location=6) in float aPolicy;
layout(push_constant) uniform WorldUIConstants {
    mat4 viewProjection;
    vec4 materialColor;
    float alphaClipThreshold;
    float alphaClipEnabled;
    vec2 screenScale;
    vec4 cameraRight;
    vec4 cameraUp;
} pc;
layout(location=0) out vec4 outColor;
layout(location=1) out vec2 outUV;
layout(location=2) out vec2 outLocalPosition;
void main() {
    vec3 position = aPosition;
    if (aPolicy > 0.5)
        position = aAnchor + pc.cameraRight.xyz * aLocalOffset.x + pc.cameraUp.xyz * aLocalOffset.y +
                   vec3(aLocalPosition * pc.screenScale, 0.0);
    gl_Position = pc.viewProjection * vec4(position, 1.0);
    outColor = aColor;
    outUV = aUV;
    outLocalPosition = aLocalPosition;
}
)glsl"
                                      : R"glsl(
#version 450
layout(location=0) in vec2 aPosition;
layout(location=1) in vec2 aUV;
layout(location=2) in vec4 aColor;
layout(push_constant) uniform ScreenUIConstants {
    vec2 scale;
    vec2 translate;
    float encodeSample;
    float _pad0;
    float _pad1;
    float _pad2;
    vec4 materialColor;
    float alphaClipThreshold;
    float alphaClipEnabled;
    vec2 _tailPadding;
} pc;
layout(location=0) out vec4 outColor;
layout(location=1) out vec2 outUV;
void main() {
    gl_Position = vec4(aPosition * pc.scale + pc.translate, 0.0, 1.0);
    outColor = aColor;
    outUV = aUV;
}
)glsl");
    const std::string fragment = "ShaderInfo { Name \"Tests/" + domain +
                                 (alternate && !reload ? "Alternate" : "Default") + "Fragment\" Capabilities [" +
                                 domain + "] }\n" +
                                 (world ? R"glsl(
#version 450
layout(location=0) in vec4 inColor;
layout(location=1) in vec2 inUV;
layout(location=2) in vec2 inLocalPosition;
layout(set=0,binding=0) uniform sampler2D uiTexture;
layout(push_constant) uniform WorldUIConstants {
    mat4 viewProjection;
    vec4 materialColor;
    float alphaClipThreshold;
    float alphaClipEnabled;
    vec2 screenScale;
    vec4 cameraRight;
    vec4 cameraUp;
} pc;
layout(location=0) out vec4 outColor;
void main() {
    outColor = texture(uiTexture, inUV) * inColor * pc.materialColor * vec4(MULTIPLIER, 1.0)
               + vec4(inLocalPosition, 0.0, 0.0) * 0.000001;
    if ((pc.alphaClipEnabled > 0.5 && outColor.a < pc.alphaClipThreshold) || outColor.a <= 0.0)
        discard;
}
)glsl"
                                        : R"glsl(
#version 450
layout(location=0) in vec4 inColor;
layout(location=1) in vec2 inUV;
layout(set=0,binding=0) uniform sampler2D uiTexture;
layout(push_constant) uniform ScreenUIConstants {
    vec2 scale;
    vec2 translate;
    float encodeSample;
    float _pad0;
    float _pad1;
    float _pad2;
    vec4 materialColor;
    float alphaClipThreshold;
    float alphaClipEnabled;
    vec2 _tailPadding;
} pc;
layout(location=0) out vec4 outColor;
void main() {
    outColor = texture(uiTexture, inUV) * inColor * pc.materialColor * vec4(MULTIPLIER, 1.0);
    if (pc.alphaClipEnabled > 0.5 && outColor.a < pc.alphaClipThreshold)
        discard;
}
)glsl");
    std::string authoredFragment = fragment;
    if (materialProperties) {
        const std::string capability = "Capabilities [" + domain + "] }";
        const size_t info = authoredFragment.find(capability);
        assert(info != std::string::npos);
        authoredFragment.replace(info, capability.size(), "Capabilities [" + domain + R"(]
    Properties {
        Float gain = 1.0
        Float2 offset = [0.0, 0.0]
        Float3 axis = [0.0, 0.0, 0.0]
        Float4 channel = [0.0, 0.0, 0.0, 0.0]
        Color tint = [1.0, 1.0, 1.0, 1.0]
        Int mode = 1
        Mat4 matrix = [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]
        Texture2D detailTex = white
    }
})");
        const std::string plainSample = "texture(uiTexture, inUV)";
        const size_t sample = authoredFragment.find(plainSample);
        assert(sample != std::string::npos);
        authoredFragment.replace(sample, plainSample.size(), "texture(uiTexture, inUV) * texture(detailTex, inUV)");
        const std::string plainMultiplier = "vec4(MULTIPLIER, 1.0)";
        const size_t factor = authoredFragment.find(plainMultiplier);
        assert(factor != std::string::npos);
        authoredFragment.replace(factor, plainMultiplier.size(),
                                 plainMultiplier + " * vec4(vec3((material.gain + material.offset.x + "
                                                   "material.axis.x + material.channel.x + material.tint.r + "
                                                   "float(material.mode) + material.matrix[0][0]) / 7.0), 1.0)");
    }
    const size_t multiplier = authoredFragment.find("MULTIPLIER");
    assert(multiplier != std::string::npos);
    authoredFragment.replace(multiplier, sizeof("MULTIPLIER") - 1, alternate ? "0.125, 0.5, 1.0" : "1.0, 1.0, 1.0");
    InxShaderLoader compiler(true, false, false, false, false, true, false, false, false, false);
    const auto vertexDescriptor = compiler.ParseShaderSource(vertex, "UITest.vert");
    const auto fragmentDescriptor = compiler.ParseShaderSource(authoredFragment, "UITest.frag");
    assert(!ShaderStageLinker::ShouldPrewarmSceneMaterial(vertexDescriptor, fragmentDescriptor));
    const auto compiled = compiler.CompileLinkedProgramArtifact(vertex, "UITest.vert", authoredFragment, "UITest.frag");
    for (const auto &error : compiled.errors)
        std::cerr << error << '\n';
    assert(compiled.IsValid());
    auto artifact = std::make_shared<ShaderProgramArtifact>(compiled.CreateRuntimeArtifact());
    assert(artifact->IsValid());
    assert(artifact->domain == (world ? ShaderProgramDomain::WorldUI : ShaderProgramDomain::ScreenUI));
    if (materialProperties) {
        assert(artifact->properties.size() == 8);
        assert(artifact->materialBufferSize >= 128);
        assert(std::count_if(artifact->properties.begin(), artifact->properties.end(),
                             [](const auto &property) { return property.textureSlot.has_value(); }) == 1);
    }
    assert(!ShaderStageLinker::ShouldPublishScenePrewarmArtifact(*artifact));
    if (!alternate) {
        auto conflictingFragment = authoredFragment;
        const std::string domainCapability = "Capabilities [" + domain + "]";
        const auto capabilityPosition = conflictingFragment.find(domainCapability);
        assert(capabilityPosition != std::string::npos);
        conflictingFragment.replace(capabilityPosition, domainCapability.size(),
                                    "Capabilities [" + domain + ", BindlessTextures]");
        const auto conflicting =
            compiler.CompileLinkedProgramArtifact(vertex, "UITest.vert", conflictingFragment, "UITest.frag");
        assert(!conflicting.IsValid());
        assert(std::any_of(conflicting.errors.begin(), conflicting.errors.end(),
                           [](const auto &error) { return error.find("cannot combine") != std::string::npos; }));
    }
    if (!world && !alternate) {
        auto badFragment = authoredFragment;
        const size_t descriptor = badFragment.find("layout(set=0,binding=0)");
        assert(descriptor != std::string::npos);
        badFragment.replace(descriptor, sizeof("layout(set=0,binding=0)") - 1, "layout(set=1,binding=0)");
        const auto wrongDescriptor =
            compiler.CompileLinkedProgramArtifact(vertex, "UITest.vert", badFragment, "UITest.frag");
        assert(!wrongDescriptor.IsValid());
        assert(std::any_of(wrongDescriptor.errors.begin(), wrongDescriptor.errors.end(),
                           [](const auto &error) { return error.find("set 0 binding 0") != std::string::npos; }));

        badFragment = authoredFragment;
        const size_t image = badFragment.find("layout(set=0,binding=0) uniform sampler2D uiTexture;");
        const size_t sample = badFragment.find("texture(uiTexture, inUV)");
        assert(image != std::string::npos && sample != std::string::npos);
        badFragment.replace(sample, sizeof("texture(uiTexture, inUV)") - 1,
                            "texture(uiTexture, inUV) + subpassLoad(extraInput)");
        badFragment.insert(image,
                           "layout(input_attachment_index=0,set=0,binding=1) uniform subpassInput extraInput;\n");
        const auto inputAttachment =
            compiler.CompileLinkedProgramArtifact(vertex, "UITest.vert", badFragment, "UITest.frag");
        assert(!inputAttachment.IsValid());
        assert(std::any_of(inputAttachment.errors.begin(), inputAttachment.errors.end(), [](const auto &error) {
            return error.find("descriptors outside the UI material ABI") != std::string::npos;
        }));

        badFragment = authoredFragment;
        const size_t imageDecl = badFragment.find("uniform sampler2D uiTexture;");
        assert(imageDecl != std::string::npos);
        badFragment.replace(imageDecl, sizeof("uniform sampler2D uiTexture;") - 1, "uniform sampler2D uiTexture[2];");
        const size_t imageSample = badFragment.find("texture(uiTexture, inUV)");
        assert(imageSample != std::string::npos);
        badFragment.replace(imageSample, sizeof("texture(uiTexture, inUV)") - 1, "texture(uiTexture[0], inUV)");
        const auto arrayImage =
            compiler.CompileLinkedProgramArtifact(vertex, "UITest.vert", badFragment, "UITest.frag");
        assert(!arrayImage.IsValid());
        assert(std::any_of(arrayImage.errors.begin(), arrayImage.errors.end(),
                           [](const auto &error) { return error.find("one non-arrayed 2D") != std::string::npos; }));

        badFragment = authoredFragment;
        const size_t nestedImageDecl = badFragment.find("uniform sampler2D uiTexture;");
        const size_t nestedImageSample = badFragment.find("texture(uiTexture, inUV)");
        assert(nestedImageDecl != std::string::npos && nestedImageSample != std::string::npos);
        badFragment.replace(nestedImageSample, sizeof("texture(uiTexture, inUV)") - 1,
                            "texture(uiTexture[0][0], inUV)");
        badFragment.replace(nestedImageDecl, sizeof("uniform sampler2D uiTexture;") - 1,
                            "uniform sampler2D uiTexture[1][2];");
        const auto nestedArrayImage =
            compiler.CompileLinkedProgramArtifact(vertex, "UITest.vert", badFragment, "UITest.frag");
        assert(!nestedArrayImage.IsValid());
        assert(std::any_of(nestedArrayImage.errors.begin(), nestedArrayImage.errors.end(),
                           [](const auto &error) { return error.find("one non-arrayed 2D") != std::string::npos; }));

        badFragment = authoredFragment;
        const size_t msDecl = badFragment.find("uniform sampler2D uiTexture;");
        const size_t msSample = badFragment.find("texture(uiTexture, inUV)");
        assert(msDecl != std::string::npos && msSample != std::string::npos);
        badFragment.replace(msSample, sizeof("texture(uiTexture, inUV)") - 1,
                            "texelFetch(uiTexture, ivec2(inUV * 128.0), 0)");
        badFragment.replace(msDecl, sizeof("uniform sampler2D uiTexture;") - 1, "uniform sampler2DMS uiTexture;");
        const auto msImage = compiler.CompileLinkedProgramArtifact(vertex, "UITest.vert", badFragment, "UITest.frag");
        assert(!msImage.IsValid());
        assert(std::any_of(msImage.errors.begin(), msImage.errors.end(),
                           [](const auto &error) { return error.find("one non-arrayed 2D") != std::string::npos; }));

        for (const auto *samplerType : {"samplerCube", "sampler3D", "sampler2DArray"}) {
            badFragment = authoredFragment;
            const size_t typePosition = badFragment.find("sampler2D uiTexture");
            const size_t samplePosition = badFragment.find("texture(uiTexture, inUV)");
            assert(typePosition != std::string::npos && samplePosition != std::string::npos);
            badFragment.replace(samplePosition, sizeof("texture(uiTexture, inUV)") - 1,
                                "texture(uiTexture, vec3(inUV, 0.0))");
            badFragment.replace(typePosition, sizeof("sampler2D uiTexture") - 1,
                                std::string(samplerType) + " uiTexture");
            const auto wrongDimension =
                compiler.CompileLinkedProgramArtifact(vertex, "UITest.vert", badFragment, "UITest.frag");
            assert(!wrongDimension.IsValid());
            assert(std::any_of(wrongDimension.errors.begin(), wrongDimension.errors.end(),
                               [](const auto &error) { return error.find("non-arrayed 2D") != std::string::npos; }));
        }

        auto badVertex = vertex;
        const size_t vertexUV = badVertex.find("layout(location=1) in vec2 aUV");
        assert(vertexUV != std::string::npos);
        badVertex.replace(vertexUV, sizeof("layout(location=1) in vec2 aUV") - 1, "layout(location=4) in vec2 aUV");
        const auto wrongVertex =
            compiler.CompileLinkedProgramArtifact(badVertex, "UITest.vert", authoredFragment, "UITest.frag");
        assert(!wrongVertex.IsValid());
        assert(std::any_of(wrongVertex.errors.begin(), wrongVertex.errors.end(),
                           [](const auto &error) { return error.find("input location 1") != std::string::npos; }));

        badFragment = authoredFragment;
        const size_t capability = badFragment.find("Capabilities [ScreenUI]");
        assert(capability != std::string::npos);
        badFragment.replace(capability, sizeof("Capabilities [ScreenUI]") - 1, "Capabilities [WorldUI]");
        const auto wrongDomain =
            compiler.CompileLinkedProgramArtifact(vertex, "UITest.vert", badFragment, "UITest.frag");
        assert(!wrongDomain.IsValid());
        assert(std::any_of(wrongDomain.errors.begin(), wrongDomain.errors.end(),
                           [](const auto &error) { return error.find("same program domain") != std::string::npos; }));
    }
    if (world && !alternate) {
        auto minimalFragment = authoredFragment;
        const size_t optionalVarying = minimalFragment.find("layout(location=2) in vec2 inLocalPosition");
        const size_t optionalUse =
            minimalFragment.find("\n               + vec4(inLocalPosition, 0.0, 0.0) * 0.000001");
        assert(optionalVarying != std::string::npos && optionalUse != std::string::npos);
        const size_t varyingLineEnd = minimalFragment.find('\n', optionalVarying);
        const size_t expressionEnd = minimalFragment.find(';', optionalUse);
        assert(varyingLineEnd != std::string::npos && expressionEnd != std::string::npos);
        minimalFragment.erase(optionalUse, expressionEnd - optionalUse);
        minimalFragment.erase(optionalVarying, varyingLineEnd + 1 - optionalVarying);
        const auto optionalVaryingOmitted =
            compiler.CompileLinkedProgramArtifact(vertex, "UITest.vert", minimalFragment, "UITest.frag");
        for (const auto &error : optionalVaryingOmitted.errors)
            std::cerr << "Optional WorldUI varying validation: " << error << std::endl;
        assert(optionalVaryingOmitted.IsValid());

        auto badFragment = authoredFragment;
        const size_t localVarying = badFragment.find("layout(location=2) in vec2 inLocalPosition");
        assert(localVarying != std::string::npos);
        badFragment.replace(localVarying, sizeof("layout(location=2) in vec2 inLocalPosition") - 1,
                            "layout(location=2) in vec3 inLocalPosition");
        const size_t localUse = badFragment.find("vec4(inLocalPosition, 0.0, 0.0)");
        assert(localUse != std::string::npos);
        badFragment.replace(localUse, sizeof("vec4(inLocalPosition, 0.0, 0.0)") - 1, "vec4(inLocalPosition, 0.0)");
        const auto wrongVarying =
            compiler.CompileLinkedProgramArtifact(vertex, "UITest.vert", badFragment, "UITest.frag");
        assert(!wrongVarying.IsValid());
        assert(std::any_of(wrongVarying.errors.begin(), wrongVarying.errors.end(),
                           [](const auto &error) { return error.find("input location 2") != std::string::npos; }));
    }
    return artifact;
}

int main(int argc, char **argv)
{
    assert(SDL_Init(SDL_INIT_VIDEO));
    auto *window = SDL_CreateWindow("UI Vulkan regression", 128, 128, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
    vk::VkDeviceContext context;
    vk::DeviceConfig config;
    config.appName = "UI Vulkan regression";
    // Correctness runs retain validation; performance runs opt out explicitly
    // so layer bookkeeping cannot be mistaken for engine/driver work.
    assert(argc == 1 || (argc == 2 && std::string(argv[1]) == "--performance"));
    config.enableValidationLayers = argc == 1;
    std::cout << "UI_BENCH_VALIDATION requested=" << config.enableValidationLayers << std::endl;
    assert(window && context.Initialize(window, config));
    auto &device = context.GetRhiDevice();
    rhi::SubmissionSerial epoch = 1;
    device.UseSubmissionSerials([&] { return epoch; });
    GpuRetirementQueue retirement;
    retirement.BindSerialSource([&] { return epoch; });

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = context.GetQueueIndices().graphicsFamily.value();
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool pool;
    assert(vkCreateCommandPool(context.GetDevice(), &poolInfo, nullptr, &pool) == VK_SUCCESS);
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    VkCommandBuffer command;
    assert(vkAllocateCommandBuffers(context.GetDevice(), &allocate, &command) == VK_SUCCESS);
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    assert(vkCreateFence(context.GetDevice(), &fenceInfo, nullptr, &fence) == VK_SUCCESS);
    auto begin = [&] {
        assert(vkResetCommandBuffer(command, 0) == VK_SUCCESS);
        VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        assert(vkBeginCommandBuffer(command, &info) == VK_SUCCESS);
    };
    auto finish = [&] {
        assert(vkEndCommandBuffer(command) == VK_SUCCESS);
        assert(vkResetFences(context.GetDevice(), 1, &fence) == VK_SUCCESS);
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        assert(vkQueueSubmit(context.GetGraphicsQueue(), 1, &submit, fence) == VK_SUCCESS);
        assert(vkWaitForFences(context.GetDevice(), 1, &fence, VK_TRUE, 5'000'000'000ull) == VK_SUCCESS);
        retirement.Collect(epoch++);
    };

    // One white texel makes solid-quad/color readback independent of font art.
    // The real ImGui atlas still generates the text benchmark's glyph geometry.
    rhi::TextureDesc whiteDesc;
    whiteDesc.format = rhi::PixelFormat::RGBA8UNorm;
    whiteDesc.usage = rhi::TextureUsageFlags::Sampled | rhi::TextureUsageFlags::TransferDestination;
    const auto white = device.CreateTexture(whiteDesc);
    rhi::TextureViewDesc viewDesc;
    viewDesc.texture = white;
    viewDesc.format = whiteDesc.format;
    const auto view = device.CreateTextureView(viewDesc);
    const auto sampler = device.CreateSampler({});
    rhi::BindingLayoutDesc layoutDesc;
    layoutDesc.entryCount = 1;
    layoutDesc.entries[0] = {0, rhi::BindingType::CombinedTextureSampler, rhi::ShaderStage::Fragment, 1};
    const auto layout = device.CreateBindingLayout(layoutDesc);
    rhi::BindGroupDesc groupDesc;
    groupDesc.layout = layout;
    groupDesc.textureCount = 1;
    groupDesc.textures[0] = {0, rhi::BindingType::CombinedTextureSampler, view, sampler};
    const auto group = device.CreateBindGroup(groupDesc);
    assert(white.IsValid() && view.IsValid() && sampler.IsValid() && group.IsValid());
    begin();
    VkImageMemoryBarrier image{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    image.srcQueueFamilyIndex = image.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image.image = device.Resolve(white);
    image.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    image.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    image.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &image);
    const VkClearColorValue whiteValue{{1, 1, 1, 1}};
    vkCmdClearColorImage(command, image.image, image.newLayout, &whiteValue, 1, &image.subresourceRange);
    image.oldLayout = image.newLayout;
    image.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    image.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    image.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &image);
    finish();

    ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = ImVec2(128, 128);
    io.DeltaTime = 1.f / 60;
    io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;
    io.Fonts->AddFontDefault();
    assert(io.Fonts->Build());
    io.Fonts->SetTexID(static_cast<ImTextureID>(reinterpret_cast<uintptr_t>(device.Resolve(group))));
    ImGui::NewFrame();
    InxScreenUIRenderer renderer;
    assert(renderer.Initialize(context.GetDevice(), context.GetVmaAllocator(), device.GetDescriptorManager(),
                               VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D32_SFLOAT, VK_SAMPLE_COUNT_1_BIT, 4));
    renderer.SetRetirementQueue(&retirement);
    std::vector<ShaderProgramKey> releasedProgramKeys;
    renderer.SetMaterialProgramRelease([&](const ShaderProgramKey &key) { releasedProgramKeys.push_back(key); });
    std::shared_ptr<InxScreenUIRenderer::CommandPacket> shutdownPacket;
    {
        rhi::RenderTextureDesc desc;
        desc.width = desc.height = 128;
        desc.colorFormat = rhi::PixelFormat::RGBA8UNorm;
        desc.depthFormat = rhi::PixelFormat::D32SFloat;
        rhi::RenderTexture target(device, "UI readback", desc);
        rhi::GraphicsRenderingSignature signature;
        signature.colorFormatCount = 1;
        signature.colorFormats[0] = desc.colorFormat;
        signature.depthFormat = desc.depthFormat;
        rhi::BufferDesc readback;
        readback.byteSize = 128 * 128 * 4;
        readback.usage = rhi::BufferUsageFlags::TransferDestination;
        readback.memory = rhi::BufferMemory::Readback;
        const auto output = device.CreateBuffer(readback);
        ScreenUIList list = ScreenUIList::Overlay;
        const auto screenProgram = CompileUiProgram(false, false);
        const auto screenAlternate = CompileUiProgram(false, true);
        const auto worldProgram = CompileUiProgram(true, false);
        const auto worldAlternate = CompileUiProgram(true, true);
        const auto screenProperties = CompileUiProgram(false, false, false, true);
        const auto worldProperties = CompileUiProgram(true, false, false, true);
        auto screenPropertyMaterial = std::make_shared<InxMaterial>("UIScreenProperties");
        auto worldPropertyMaterial = std::make_shared<InxMaterial>("UIWorldProperties");
        screenPropertyMaterial->SetGuid("ui-screen-properties-guid");
        worldPropertyMaterial->SetGuid("ui-world-properties-guid");
        const auto initializePropertyMaterial = [](InxMaterial &material, const ShaderProgramArtifact &program) {
            assert(material.SynchronizeShaderPropertyDefaults(program));
            material.SetFloat("gain", 0.25f);
            material.SetVector2("offset", glm::vec2(1.0f, 0.0f));
            material.SetVector3("axis", glm::vec3(1.0f, 0.0f, 0.0f));
            material.SetVector4("channel", glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
            material.SetColor("tint", glm::vec4(1.0f));
            material.SetInt("mode", 1);
            material.SetMatrix("matrix", glm::mat4(1.0f));
            material.SetTextureGuid("detailTex", "white");
        };
        initializePropertyMaterial(*screenPropertyMaterial, *screenProperties);
        initializePropertyMaterial(*worldPropertyMaterial, *worldProperties);
        auto whitePublication = std::make_shared<rhi::TextureGpuView>(
            "ui-white-guid", 1, white, view, sampler, 4, std::make_shared<int>(1), rhi::PixelFormat::RGBA8UNorm);
        assert(whitePublication->IsValid());
        uint64_t textureGeneration = 1;
        size_t textureResolutions = 0;
        size_t pendingTextureResolutions = 0;
        bool texturePending = false;
        renderer.SetMaterialAssetResolver([&](const std::string &guid, uint64_t generation) {
            const auto material = guid == "ui-screen-properties-guid" ? screenPropertyMaterial : worldPropertyMaterial;
            assert(material->GetVersion() == generation);
            return std::shared_ptr<const InxMaterial>(material);
        });
        renderer.SetMaterialTextureResolver(
            [&](const std::string &guid, const std::string &name, const MaterialTextureSampler *) {
                assert(guid == "white" && name == "detailTex");
                ++textureResolutions;
                TextureResolveResult result;
                if (texturePending) {
                    ++pendingTextureResolutions;
                    result.status = TextureResolveStatus::Pending;
                    return result;
                }
                result.status = TextureResolveStatus::Ready;
                result.binding.imageView = device.Resolve(view);
                result.binding.sampler = device.Resolve(sampler);
                result.binding.gpuView = whitePublication;
                return result;
            });
        renderer.SetMaterialTextureGenerationResolver([&](const std::string &guid) {
            assert(guid == "white");
            return textureGeneration;
        });
        // Both cross-domain first draws reject before publication and before
        // an artifact can enter the material resolver's retained cache.
        for (const auto &declaredProgram : {screenProgram, worldProgram}) {
            VkShaderCache crossDomainCache;
            crossDomainCache.GetProgramCache().Initialize(context.GetDevice());
            renderer.SetMaterialProgramResolver(
                [&](const std::string &, uint64_t,
                    ShaderProgramDomain requestedDomain) -> std::shared_ptr<const ShaderProgramArtifact> {
                    if (requestedDomain != declaredProgram->domain)
                        throw std::runtime_error("UI material shader domain mismatch before publication");
                    assert(crossDomainCache.PublishProgramArtifact(*declaredProgram).accepted);
                    return crossDomainCache.ShareProgramArtifact(declaredProgram->key.stages);
                });
            UIShaderMaterialBinding crossDomainBinding;
            crossDomainBinding.materialGuid = declaredProgram->domain == ShaderProgramDomain::ScreenUI
                                                  ? "screen-material-in-world"
                                                  : "world-material-in-screen";
            crossDomainBinding.generation = 1;
            crossDomainBinding.pipelineKey = crossDomainBinding.materialGuid;
            const auto requestedDomain = declaredProgram->domain == ShaderProgramDomain::ScreenUI
                                             ? ShaderProgramDomain::WorldUI
                                             : ShaderProgramDomain::ScreenUI;
            bool rejectedCrossDomain = false;
            try {
                (void)ScreenUIVulkanTestAccess::ResolveMaterialPipeline(renderer, crossDomainBinding, requestedDomain,
                                                                        &signature);
            } catch (const std::runtime_error &) {
                rejectedCrossDomain = true;
            }
            assert(rejectedCrossDomain);
            assert(!crossDomainCache.ShareProgramArtifact(declaredProgram->key.stages));
            assert(!crossDomainCache.GetProgramCache().HasProgram(declaredProgram->key));
            crossDomainCache.GetProgramCache().Shutdown();
            crossDomainCache.Clear();
        }
        VkShaderCache scenePrewarmCache;
        scenePrewarmCache.GetProgramCache().Initialize(context.GetDevice());
        for (const auto &uiProgram : {screenProgram, worldProgram}) {
            if (ShaderStageLinker::ShouldPublishScenePrewarmArtifact(*uiProgram))
                assert(scenePrewarmCache.PublishProgramArtifact(*uiProgram).accepted);
            assert(!scenePrewarmCache.ShareProgramArtifact(uiProgram->key.stages));
            assert(!scenePrewarmCache.GetProgramCache().HasProgram(uiProgram->key));
        }
        scenePrewarmCache.GetProgramCache().Shutdown();
        scenePrewarmCache.Clear();
        // Published UI stage pairs have an explicit owner-release path, not
        // an unbounded process-lifetime artifact/Forward-program cache.
        VkShaderCache ownershipCache;
        ownershipCache.GetProgramCache().Initialize(context.GetDevice());
        size_t retiredOwnedPrograms = 0;
        for (int index = 0; index < 24; ++index) {
            ShaderProgramArtifact candidate = *screenProgram;
            candidate.key.stages.vertexShaderId += "/owned-" + std::to_string(index);
            candidate.key.stages.fragmentShaderId += "/owned-" + std::to_string(index);
            candidate.key.revision = ComputeShaderProgramArtifactRevision(candidate);
            assert(candidate.IsValid());
            assert(ownershipCache.PublishProgramArtifact(candidate).accepted);
            assert(ownershipCache.ShareProgramArtifact(candidate.key.stages));
            assert(ownershipCache.GetProgramCache().HasProgram(candidate.key));
            auto released = ownershipCache.TakeUIProgramArtifact(candidate.key);
            assert(released && !ownershipCache.ShareProgramArtifact(candidate.key.stages));
            auto programs = ownershipCache.GetProgramCache().TakePrograms(candidate.key);
            assert(!programs.empty() && !ownershipCache.GetProgramCache().HasProgram(candidate.key));
            for (auto &program : programs) {
                retirement.Retire([owned = std::move(program), &retiredOwnedPrograms]() mutable {
                    owned.reset();
                    ++retiredOwnedPrograms;
                });
            }
            retirement.Retire([owned = std::move(released)]() mutable { owned.reset(); });
        }
        assert(retirement.Collect(epoch - 1) == 0);
        assert(retirement.GetStats().pending >= 48);
        begin();
        finish();
        assert(retiredOwnedPrograms == 24);

        for (const auto domain : {ShaderProgramDomain::Mesh, ShaderProgramDomain::ParticleSprite}) {
            ShaderProgramArtifact shared = *screenProgram;
            shared.domain = domain;
            shared.key.stages.vertexShaderId += domain == ShaderProgramDomain::Mesh ? "/mesh" : "/particle";
            shared.key.stages.fragmentShaderId += domain == ShaderProgramDomain::Mesh ? "/mesh" : "/particle";
            shared.key.revision = ComputeShaderProgramArtifactRevision(shared);
            assert(ownershipCache.PublishProgramArtifact(shared).accepted);
            assert(!ownershipCache.TakeUIProgramArtifact(shared.key));
            assert(ownershipCache.ShareProgramArtifact(shared.key.stages));
            assert(ownershipCache.GetProgramCache().HasProgram(shared.key));
        }
        ownershipCache.GetProgramCache().Shutdown();
        ownershipCache.Clear();
        // Core owns a program across UI renderer generations. Only the last
        // renderer owner may evict its artifact and Forward program.
        auto ownerCore = std::make_unique<InxVkCoreModular>();
        ownerCore->GetRetirementQueue().BindSerialSource([&] { return epoch; });
        auto &coreShaderCache = ownerCore->GetShaderCache();
        coreShaderCache.GetProgramCache().Initialize(context.GetDevice());
        // The real mesh-filter and particle-output entry points both use the
        // Core resolver with an expected non-UI domain. A rejected UI source
        // must not publish either an artifact or a Forward program.
        auto wrongDomainMaterial = std::make_shared<InxMaterial>("UIOnNonUIDraw");
        wrongDomainMaterial->SetVertShader(screenProgram->key.stages.vertexShaderId);
        wrongDomainMaterial->SetFragShader(screenProgram->key.stages.fragmentShaderId);
        size_t rejectedNonUIResolutions = 0;
        ownerCore->SetShaderProgramArtifactResolver(
            [&](const std::shared_ptr<InxMaterial> &, std::optional<ShaderProgramDomain> expected) {
                assert(expected &&
                       (*expected == ShaderProgramDomain::Mesh || *expected == ShaderProgramDomain::ParticleSprite));
                ++rejectedNonUIResolutions;
                throw std::runtime_error("UI shader cannot be published for non-UI draw");
            });
        for (const auto expected : {ShaderProgramDomain::Mesh, ShaderProgramDomain::ParticleSprite}) {
            bool rejected = false;
            try {
                (void)ownerCore->ResolveShaderProgramArtifact(wrongDomainMaterial, screenProgram->key.stages, expected);
            } catch (const std::runtime_error &) {
                rejected = true;
            }
            assert(rejected);
            assert(!coreShaderCache.ShareProgramArtifact(screenProgram->key.stages));
            assert(!coreShaderCache.GetProgramCache().HasProgram(screenProgram->key));
        }
        assert(rejectedNonUIResolutions == 2);
        assert(coreShaderCache.PublishProgramArtifact(*screenProgram).accepted);
        bool rejectedCachedUI = false;
        try {
            (void)ownerCore->ResolveShaderProgramArtifact(wrongDomainMaterial, screenProgram->key.stages,
                                                          ShaderProgramDomain::Mesh);
        } catch (const std::runtime_error &) {
            rejectedCachedUI = true;
        }
        assert(rejectedCachedUI);
        assert(coreShaderCache.GetProgramCache().HasProgram(screenProgram->key));
        ownerCore->AcquireUIShaderProgramOwner(screenProgram->key);
        ownerCore->AcquireUIShaderProgramOwner(screenProgram->key);
        assert(!ownerCore->ReleaseUIShaderProgramArtifact(screenProgram->key));
        ownerCore->ReleaseUIShaderProgramOwner(screenProgram->key);
        assert(coreShaderCache.ShareProgramArtifact(screenProgram->key.stages));
        ownerCore->ReleaseUIShaderProgramOwner(screenProgram->key);
        assert(!coreShaderCache.ShareProgramArtifact(screenProgram->key.stages));
        assert(!coreShaderCache.GetProgramCache().HasProgram(screenProgram->key));
        // A reload of a never-drawn UI material has no owner; even if an
        // artifact was published before this policy, invalidation evicts it.
        assert(coreShaderCache.PublishProgramArtifact(*worldProgram).accepted);
        assert(ownerCore->ReleaseUIShaderProgramArtifact(worldProgram->key));
        assert(!coreShaderCache.ShareProgramArtifact(worldProgram->key.stages));
        assert(!coreShaderCache.GetProgramCache().HasProgram(worldProgram->key));
        // A drawn UI material releases the old revision before the next draw
        // publishes and acquires the edited revision.
        const auto reloadedOwnedProgram = CompileUiProgram(false, true, true);
        assert(reloadedOwnedProgram->key.stages == screenProgram->key.stages);
        assert(coreShaderCache.PublishProgramArtifact(*reloadedOwnedProgram).accepted);
        ownerCore->AcquireUIShaderProgramOwner(reloadedOwnedProgram->key);
        assert(!ownerCore->ReleaseUIShaderProgramArtifact(reloadedOwnedProgram->key));
        ownerCore->ReleaseUIShaderProgramOwner(reloadedOwnedProgram->key);
        assert(!coreShaderCache.ShareProgramArtifact(reloadedOwnedProgram->key.stages));
        assert(!coreShaderCache.GetProgramCache().HasProgram(reloadedOwnedProgram->key));
        assert(ownerCore->GetRetirementQueue().Collect(epoch - 1) == 0);
        assert(ownerCore->GetRetirementQueue().Collect(epoch) > 0);
        coreShaderCache.GetProgramCache().Shutdown();
        coreShaderCache.Clear();
        ownerCore.reset();
        auto activeScreenProgram = screenProgram;
        size_t materialResolutions = 0;
        const auto resolveMaterialProgram =
            [&](const std::string &guid, uint64_t generation,
                ShaderProgramDomain requestedDomain) -> std::shared_ptr<const ShaderProgramArtifact> {
            assert(!guid.empty() && generation != 0);
            ++materialResolutions;
            if (guid == "ui-ordinary-guid")
                return std::shared_ptr<const ShaderProgramArtifact>{};
            if (guid == "ui-screen-properties-guid")
                return screenProperties;
            if (guid == "ui-world-properties-guid")
                return worldProperties;
            const bool alternate = guid == "ui-alternate-guid" || guid == "world-ui-alternate-guid";
            if (requestedDomain == ShaderProgramDomain::WorldUI)
                return alternate ? worldAlternate : worldProgram;
            return alternate ? screenAlternate : activeScreenProgram;
        };
        renderer.SetMaterialProgramResolver(resolveMaterialProgram);
        glm::mat4 camera(1.f);
        glm::mat4 cameraView(1.f);
        glm::mat4 cameraProjection(1.f);
        uint32_t frameSlot = 0;
        uint32_t cullingMask = 0xffffffffu;
        bool isolateSlots = false;
        float clearDepth = 1.0f;
        double renderMs = 0;
        InxScreenUIRenderer *activeRenderer = &renderer;
        vk::RenderGraph graph;
        vk::ResourceHandle color;
        auto buildGraph = [&] {
            graph.Destroy();
            graph.Initialize(&context);
            const auto attachments = graph.ImportRenderTexture("UI readback", target.Acquire());
            graph.AddPass("draw-ui", [&](vk::PassBuilder &builder) {
                color = builder.WriteColor(attachments.color);
                if (list == ScreenUIList::World)
                    builder.WriteDepth(attachments.depth);
                builder.SetClearColor(0, 0, 0, 0);
                builder.SetClearDepth(clearDepth, 0);
                builder.SetDepthTest(true);
                builder.SetRenderArea(128, 128);
                return [&](vk::RenderContext &ctx) {
                    const auto start = std::chrono::steady_clock::now();
                    if (isolateSlots) {
                        // Both draws are recorded before either executes on the GPU.
                        // A shared writable buffer makes the red quad disappear.
                        activeRenderer->BeginFrame(128, 128);
                        activeRenderer->AddFilledRect(list, 8, 8, 56, 120, 1, 0, 0, 1);
                        activeRenderer->Render(ctx.GetCommandBuffer(), list, 128, 128, 0);
                        activeRenderer->BeginFrame(128, 128);
                        activeRenderer->AddFilledRect(list, 72, 8, 120, 120, 0, 1, 0, 1);
                        activeRenderer->Render(ctx.GetCommandBuffer(), list, 128, 128, 3);
                        return;
                    }
                    if (list == ScreenUIList::World)
                        activeRenderer->RenderWorld(ctx.GetCommandBuffer(), 128, 128, camera, signature, frameSlot,
                                                    cullingMask, cameraView, cameraProjection);
                    else
                        activeRenderer->Render(ctx.GetCommandBuffer(), list, 128, 128, frameSlot);
                    renderMs =
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                };
            });
            graph.AddTransferPass("read-ui", [&](vk::PassBuilder &builder) {
                builder.TransferRead(color);
                builder.TransferWrite(builder.ImportBuffer("pixels", output, readback.byteSize));
                builder.SetQueueRole(rhi::QueueRole::Graphics);
                builder.SetSideEffect();
                return [&](vk::RenderContext &ctx) {
                    VkBufferImageCopy copy{};
                    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    copy.imageExtent = {128, 128, 1};
                    vkCmdCopyImageToBuffer(ctx.GetCommandBuffer(), device.Resolve(ctx.GetTextureHandle(color)),
                                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, device.Resolve(output), 1, &copy);
                };
            });
            graph.AddPass("export-ui", [&](vk::PassBuilder &builder) {
                builder.Read(color);
                builder.SetSideEffect();
                return [](vk::RenderContext &) {};
            });
            assert(graph.Compile());
        };
        auto frame = [&] {
            begin();
            graph.Execute(command);
            VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = device.Resolve(output);
            barrier.size = readback.byteSize;
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                                 &barrier, 0, nullptr);
            finish();
            frameSlot = (frameSlot + 1) % 4;
        };
        for (auto kind : {ScreenUIList::Camera, ScreenUIList::Overlay, ScreenUIList::World}) {
            list = kind;
            buildGraph();
            const uint64_t contentRevision = 100 + static_cast<uint64_t>(kind);
            assert(!renderer.BeginFrameCached(128, 128, contentRevision));
            for (int i = 0; i < 1000; ++i) {
                if (list == ScreenUIList::World) {
                    std::array<float, 16> pose{};
                    glm::mat4 matrix(1.f);
                    matrix[3].z = .5f;
                    std::copy_n(glm::value_ptr(matrix), 16, pose.begin());
                    renderer.BeginWorldElement(pose, 50, 20);
                }
                renderer.AddText(list, 0, 0, 100, 40, "UI 041", 1, 1, 1, 1, 0, 0, 18);
                if (list == ScreenUIList::World)
                    renderer.EndWorldElement();
            }
            std::vector<double> samples;
            for (int i = 0; i < 24; ++i) {
                assert(renderer.BeginFrameCached(128, 128, contentRevision));
                frame();
                if (i >= 4)
                    samples.push_back(renderMs);
            }
            std::sort(samples.begin(), samples.end());
            std::cout << "UI_NATIVE_BENCH list=" << int(list)
                      << " count=1000 render_record_p50_ms=" << samples[samples.size() / 2]
                      << " draws=" << renderer.GetLastSubmittedDrawCount(list) << std::endl;
            assert(renderer.GetLastSubmittedIndexCount(list) > 0);
            assert(renderer.GetLastSubmittedDrawCount(list) == 1);
            const auto stats = renderer.GetGeometryStats(list);
            assert(stats.preparations == 1 && stats.uploads == 4 && stats.uploadedBytes > 0);
        }

        // The opt-in world policy bypasses scene depth without changing the
        // default pipeline. Both variants are exercised by actual readback.
        list = ScreenUIList::World;
        glm::mat4 topPose(1.0f);
        topPose[3].z = 0.5f;
        std::array<float, 16> topMatrix{};
        std::copy_n(glm::value_ptr(topPose), 16, topMatrix.begin());
        constexpr size_t policySample = (64 * 128 + 64) * 4;
        const auto drawWorldPolicy = [&](bool alwaysOnTop) {
            renderer.BeginFrame(128, 128);
            renderer.BeginWorldElement(topMatrix, 50, 50, 0xffffffffu, alwaysOnTop);
            renderer.AddFilledRect(list, 0, 0, 100, 100, alwaysOnTop ? 0.f : 1.f, alwaysOnTop ? 1.f : 0.f, 0.f, 1.f);
            renderer.EndWorldElement();
            frame();
            std::vector<uint8_t> readback(128 * 128 * 4);
            assert(device.ReadBuffer(output, 0, readback.data(), readback.size()));
            return std::array<uint8_t, 4>{readback[policySample], readback[policySample + 1],
                                          readback[policySample + 2], readback[policySample + 3]};
        };
        clearDepth = 0.0f;
        buildGraph();
        const auto hiddenByScene = drawWorldPolicy(false);
        const auto overScene = drawWorldPolicy(true);
        assert(hiddenByScene[3] == 0);
        assert(overScene[0] == 0 && overScene[1] == 255 && overScene[3] == 255);
        clearDepth = 1.0f;
        buildGraph();
        const auto defaultAgain = drawWorldPolicy(false);
        assert(defaultAgain[0] == 255 && defaultAgain[1] == 0 && defaultAgain[3] == 255);
        // Submission order and a nearer default element cannot cover the
        // special element; its policy defines a separate final UI stratum.
        renderer.BeginFrame(128, 128);
        renderer.BeginWorldElement(topMatrix, 50, 50, 0xffffffffu, true);
        renderer.AddFilledRect(list, 0, 0, 100, 100, 0, 1, 0, 1);
        renderer.EndWorldElement();
        renderer.BeginWorldElement(topMatrix, 50, 50);
        renderer.AddFilledRect(list, 0, 0, 100, 100, 1, 0, 0, 1);
        renderer.EndWorldElement();
        frame();
        std::vector<uint8_t> topOverDefault(128 * 128 * 4);
        assert(device.ReadBuffer(output, 0, topOverDefault.data(), topOverDefault.size()));
        assert(topOverDefault[policySample] == 0 && topOverDefault[policySample + 1] == 255);

        // Camera policies are opt-in, evaluated per camera, and preserve the
        // same pixel footprint at different perspective depths.
        const auto worldBounds = [&](const std::vector<uint8_t> &image) {
            std::array<int, 4> result{128, 128, -1, -1};
            for (int y = 0; y < 128; ++y)
                for (int x = 0; x < 128; ++x)
                    if (image[static_cast<size_t>((y * 128 + x) * 4 + 3)] != 0) {
                        result[0] = std::min(result[0], x);
                        result[1] = std::min(result[1], y);
                        result[2] = std::max(result[2], x);
                        result[3] = std::max(result[3], y);
                    }
            return result;
        };
        const auto cameraPolicyBounds = [&](float depth, bool billboard, bool constantSize, bool rotate) {
            glm::mat4 pose =
                rotate ? glm::rotate(glm::mat4(1.f), glm::radians(90.f), glm::vec3(0, 1, 0)) : glm::mat4(1.f);
            pose[3].z = -depth;
            std::array<float, 16> matrix{};
            std::copy_n(glm::value_ptr(pose), 16, matrix.begin());
            renderer.BeginFrame(128, 128);
            renderer.BeginWorldElement(matrix, 10, 10, 0xffffffffu, false, billboard, constantSize);
            renderer.AddFilledRect(list, 0, 0, 20, 20, 1, 1, 1, 1);
            renderer.EndWorldElement();
            frame();
            std::vector<uint8_t> image(128 * 128 * 4);
            assert(device.ReadBuffer(output, 0, image.data(), image.size()));
            return worldBounds(image);
        };
        cameraProjection = glm::perspectiveRH_ZO(glm::radians(90.f), 1.f, .1f, 10.f);
        camera = cameraProjection * cameraView;
        buildGraph();
        const auto nearPixels = cameraPolicyBounds(2.f, true, true, true);
        const auto farPixels = cameraPolicyBounds(4.f, true, true, true);
        assert(nearPixels == farPixels && nearPixels[2] - nearPixels[0] >= 18);
        const auto authoredEdge = cameraPolicyBounds(2.f, false, false, true);
        assert(authoredEdge[2] - authoredEdge[0] < nearPixels[2] - nearPixels[0]);
        const auto perspectiveWorld = cameraPolicyBounds(2.f, false, false, false);
        const auto perspectiveFixed = cameraPolicyBounds(2.f, false, true, false);
        assert(perspectiveFixed[2] - perspectiveFixed[0] > perspectiveWorld[2] - perspectiveWorld[0]);
        cameraView = glm::lookAtRH(glm::vec3(2.f, 0.f, -2.f), glm::vec3(0.f, 0.f, -2.f), glm::vec3(0.f, 1.f, 0.f));
        camera = cameraProjection * cameraView;
        buildGraph();
        const auto sideCameraPixels = cameraPolicyBounds(2.f, true, true, true);
        assert(sideCameraPixels == nearPixels);
        cameraView = glm::mat4(1.f);
        cameraProjection = glm::orthoRH_ZO(-1.f, 1.f, -1.f, 1.f, .1f, 10.f);
        camera = cameraProjection * cameraView;
        buildGraph();
        assert(cameraPolicyBounds(2.f, true, true, true) == cameraPolicyBounds(4.f, true, true, true));
        cameraProjection = camera = glm::mat4(1.f);
        buildGraph();

        // A retained UI packet carries an asset identity contract alongside
        // each draw command.  This is intentionally independent of the fixed
        // Screen/World depth pipelines and must not retain a path or pointer.
        list = ScreenUIList::Overlay;
        renderer.BeginFrame(128, 128);
        renderer.BeginCommandPacket();
        renderer.SetMaterialBinding(list, "ui-material-guid", 7, "ui|shader=Ui.vert:Ui.frag|state=1,6,7,0,0,0,7,0");
        renderer.AddFilledRect(list, 8, 8, 56, 56, 1, 1, 1, 1);
        renderer.SetMaterialBinding(list, "ui-text-guid", 11,
                                    "ui|shader=UiText.vert:UiText.frag|state=1,6,7,0,0,0,7,0");
        renderer.AddFilledRect(list, 60, 8, 112, 56, 1, 1, 1, 1);
        const auto contractPacket = renderer.EndCommandPacket();
        renderer.AppendCommandPackets({contractPacket});
        const auto &contracts = renderer.GetCommandBindings(list);
        assert(std::any_of(contracts.begin(), contracts.end(), [](const UIShaderMaterialBinding &binding) {
            return binding.IsValid() && binding.materialGuid == "ui-material-guid" && binding.generation == 7 &&
                   binding.pipelineKey.rfind("ui|shader=", 0) == 0;
        }));
        assert(std::any_of(contracts.begin(), contracts.end(), [](const UIShaderMaterialBinding &binding) {
            return binding.IsValid() && binding.materialGuid == "ui-text-guid" && binding.generation == 11;
        }));

        // One button packet can contain an authored background followed by
        // default label geometry. Clearing the binding must isolate the label.
        renderer.BeginFrame(128, 128);
        renderer.BeginCommandPacket();
        renderer.SetMaterialBinding(list, "ui-background-guid", 2, "ui-background", {0.25f, 0.5f, 0.75f, 1.0f});
        renderer.AddFilledRect(list, 0, 0, 48, 48, 1, 1, 1, 1);
        renderer.SetMaterialBinding(list, "", 0, "");
        renderer.AddFilledRect(list, 64, 0, 112, 48, 1, 1, 1, 1);
        renderer.AppendCommandPackets({renderer.EndCommandPacket()});
        const auto &resetContracts = renderer.GetCommandBindings(list);
        assert(std::any_of(resetContracts.begin(), resetContracts.end(), [](const UIShaderMaterialBinding &binding) {
            return binding.materialGuid == "ui-background-guid";
        }));
        assert(std::count_if(resetContracts.begin(), resetContracts.end(),
                             [](const UIShaderMaterialBinding &binding) { return binding.IsValid(); }) == 1);
        buildGraph();
        frame();
        std::array<uint8_t, 128 * 128 * 4> resetObserved{};
        assert(device.ReadBuffer(output, 0, resetObserved.data(), resetObserved.size()));
        const size_t defaultSample = (24 * 128 + 88) * 4;
        assert(resetObserved[defaultSample] == 255 && resetObserved[defaultSample + 1] == 255 &&
               resetObserved[defaultSample + 2] == 255 && resetObserved[defaultSample + 3] == 255);

        // Authored values must reach the fragment stage, not merely survive in
        // packet metadata. White geometry isolates material modulation; an
        // alpha-clipped material must leave the cleared target untouched.
        for (auto kind : {ScreenUIList::Camera, ScreenUIList::Overlay, ScreenUIList::World}) {
            list = kind;
            buildGraph();
            for (bool clip : {false, true}) {
                renderer.BeginFrame(128, 128);
                renderer.BeginCommandPacket();
                renderer.SetMaterialBinding(list,
                                            list == ScreenUIList::World ? "world-ui-authored-guid" : "ui-authored-guid",
                                            1, "ui-authored", {0.25f, 0.75f, 0.5f, clip ? 0.25f : 1.f}, clip, 0.5f);
                if (list == ScreenUIList::World) {
                    glm::mat4 matrix(1.f);
                    matrix[3].z = .5f;
                    std::array<float, 16> pose{};
                    std::copy_n(glm::value_ptr(matrix), 16, pose.begin());
                    renderer.BeginWorldElement(pose, 50, 50);
                }
                renderer.AddFilledRect(list, 0, 0, 100, 100, 1, 1, 1, 1);
                if (list == ScreenUIList::World)
                    renderer.EndWorldElement();
                const auto packet = renderer.EndCommandPacket();
                renderer.AppendCommandPackets({packet});
                frame();
                std::array<uint8_t, 128 * 128 * 4> observed{};
                assert(device.ReadBuffer(output, 0, observed.data(), observed.size()));
                const size_t sample = (64 * 128 + 64) * 4;
                if (clip) {
                    assert(observed[sample + 3] == 0);
                } else {
                    assert(observed[sample] >= 63 && observed[sample] <= 64);
                    assert(observed[sample + 1] >= 191 && observed[sample + 1] <= 192);
                    assert(observed[sample + 2] >= 127 && observed[sample + 2] <= 128);
                    assert(observed[sample + 3] == 255);
                }
            }
        }

        // A normal Standard/Unlit material still has a GUID and tint, but no
        // UI ShaderInfo domain. It must draw with the fixed pipeline on all
        // three UI passes instead of being mistaken for a custom program.
        for (auto kind : {ScreenUIList::Camera, ScreenUIList::Overlay, ScreenUIList::World}) {
            list = kind;
            buildGraph();
            renderer.BeginFrame(128, 128);
            renderer.BeginCommandPacket();
            renderer.SetMaterialBinding(list, "ui-ordinary-guid", 1, "ordinary-material", {0.5f, 0.25f, 0.75f, 1.f});
            if (list == ScreenUIList::World) {
                glm::mat4 matrix(1.f);
                matrix[3].z = .5f;
                std::array<float, 16> pose{};
                std::copy_n(glm::value_ptr(matrix), 16, pose.begin());
                renderer.BeginWorldElement(pose, 50, 50);
            }
            renderer.AddFilledRect(list, 0, 0, 100, 100, 1, 1, 1, 1);
            if (list == ScreenUIList::World)
                renderer.EndWorldElement();
            renderer.AppendCommandPackets({renderer.EndCommandPacket()});
            frame();
            std::vector<uint8_t> pixels(128 * 128 * 4);
            assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
            const size_t sample = (64 * 128 + 64) * 4;
            assert(pixels[sample] >= 127 && pixels[sample] <= 128);
            assert(pixels[sample + 1] >= 63 && pixels[sample + 1] <= 64);
            assert(pixels[sample + 2] >= 191 && pixels[sample + 2] <= 192);
        }

        // The same GUID-bound command must execute authored SPIR-V. The two
        // programs share material color and the image descriptor but produce
        // different pixels in Camera, Overlay and World UI.
        for (auto kind : {ScreenUIList::Camera, ScreenUIList::Overlay, ScreenUIList::World}) {
            list = kind;
            buildGraph();
            for (bool alternate : {false, true}) {
                renderer.BeginFrame(128, 128);
                renderer.BeginCommandPacket();
                const char *materialGuid = list == ScreenUIList::World
                                               ? (alternate ? "world-ui-alternate-guid" : "world-ui-authored-guid")
                                               : (alternate ? "ui-alternate-guid" : "ui-authored-guid");
                renderer.SetMaterialBinding(list, materialGuid, 1, "test-ui-program", {1.f, 1.f, 1.f, 1.f});
                if (list == ScreenUIList::World) {
                    glm::mat4 matrix(1.f);
                    matrix[3].z = .5f;
                    std::array<float, 16> pose{};
                    std::copy_n(glm::value_ptr(matrix), 16, pose.begin());
                    renderer.BeginWorldElement(pose, 50, 50);
                }
                renderer.AddFilledRect(list, 0, 0, 100, 100, 1, 1, 1, 1);
                if (list == ScreenUIList::World)
                    renderer.EndWorldElement();
                renderer.AppendCommandPackets({renderer.EndCommandPacket()});
                frame();
                std::array<uint8_t, 128 * 128 * 4> pixels{};
                assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
                const size_t sample = (64 * 128 + 64) * 4;
                if (alternate) {
                    assert(pixels[sample] >= 31 && pixels[sample] <= 32);
                    assert(pixels[sample + 1] >= 127 && pixels[sample + 1] <= 128);
                    assert(pixels[sample + 2] == 255);
                } else {
                    assert(pixels[sample] == 255 && pixels[sample + 1] == 255 && pixels[sample + 2] == 255);
                }
            }
        }

        // ShaderInfo values and a GUID-resolved Texture2D must reach both UI
        // domains through set 1. The readback depends on all seven UBO types.
        const auto renderPropertyList = [&](ScreenUIList kind) {
            list = kind;
            buildGraph();
            renderer.BeginFrame(128, 128);
            renderer.BeginCommandPacket();
            const auto &material = list == ScreenUIList::World ? worldPropertyMaterial : screenPropertyMaterial;
            renderer.SetMaterialBinding(list, material->GetGuid(), material->GetVersion(), "ui-property-program");
            if (list == ScreenUIList::World) {
                glm::mat4 matrix(1.f);
                matrix[3].z = .5f;
                std::array<float, 16> pose{};
                std::copy_n(glm::value_ptr(matrix), 16, pose.begin());
                renderer.BeginWorldElement(pose, 50, 50);
            }
            renderer.AddFilledRect(list, 0, 0, 100, 100, 1, 1, 1, 1);
            if (list == ScreenUIList::World)
                renderer.EndWorldElement();
            renderer.AppendCommandPackets({renderer.EndCommandPacket()});
            frame();
            std::vector<uint8_t> pixels(128 * 128 * 4);
            assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
            const size_t sample = (64 * 128 + 64) * 4;
            assert(pixels[sample] >= 226 && pixels[sample] <= 229);
            assert(pixels[sample + 1] >= 226 && pixels[sample + 1] <= 229);
            assert(pixels[sample + 2] >= 226 && pixels[sample + 2] <= 229);
            assert(pixels[sample + 3] == 255);
        };
        for (auto kind : {ScreenUIList::Camera, ScreenUIList::Overlay, ScreenUIList::World})
            renderPropertyList(kind);
        // Stable generations reuse the descriptor without repeating texture
        // resolution. A texture reimport changes its GPU publication even
        // while the owning material generation and retained command stay put.
        assert(textureResolutions == 2);
        const size_t retiredBeforeReimport = retirement.GetStats().pushed;
        UIShaderMaterialBinding propertyBinding;
        propertyBinding.materialGuid = screenPropertyMaterial->GetGuid();
        propertyBinding.generation = screenPropertyMaterial->GetVersion();
        propertyBinding.pipelineKey = "ui-property-program";
        const VkDescriptorSet descriptorBeforeReimport =
            ScreenUIVulkanTestAccess::ResolveMaterialDescriptor(renderer, propertyBinding, *screenProperties);
        assert(descriptorBeforeReimport != VK_NULL_HANDLE);
        ++textureGeneration;
        texturePending = true;
        renderPropertyList(ScreenUIList::Overlay);
        const VkDescriptorSet descriptorWhilePending =
            ScreenUIVulkanTestAccess::ResolveMaterialDescriptor(renderer, propertyBinding, *screenProperties);
        assert(descriptorWhilePending == descriptorBeforeReimport);
        assert(pendingTextureResolutions == 1);
        const size_t resolutionsWhilePending = textureResolutions;
        texturePending = false;
        whitePublication =
            std::make_shared<rhi::TextureGpuView>("ui-white-guid", textureGeneration, white, view, sampler, 4,
                                                  std::make_shared<int>(1), rhi::PixelFormat::RGBA8UNorm);
        renderPropertyList(ScreenUIList::Overlay);
        assert(textureResolutions == resolutionsWhilePending + 1);
        assert(retirement.GetStats().pushed > retiredBeforeReimport);

        // A retained command resolves once per GUID/generation, not per draw.
        list = ScreenUIList::Overlay;
        buildGraph();
        renderer.BeginFrame(128, 128);
        renderer.BeginCommandPacket();
        renderer.SetMaterialBinding(list, "ui-authored-guid", 1, "test-ui-program");
        renderer.AddFilledRect(list, 0, 0, 100, 100, 1, 1, 1, 1);
        const auto reloadPacket = renderer.EndCommandPacket();
        renderer.AppendCommandPackets({reloadPacket});
        frame();
        const size_t resolvedBeforeRetained = materialResolutions;
        frame();
        assert(materialResolutions == resolvedBeforeRetained);

        // A new shader revision retires obsolete Vulkan pipelines by GPU
        // submission serial and re-resolves the retained command exactly once.
        const auto reloadedScreen = CompileUiProgram(false, true, true);
        assert(reloadedScreen->key.stages == screenProgram->key.stages);
        assert(reloadedScreen->key != screenProgram->key);
        activeScreenProgram = reloadedScreen;
        const auto variantsBeforeReload = renderer.GetMaterialPipelineVariantCount();
        const auto retirementsBeforeReload = retirement.GetStats().pushed;
        renderer.InvalidateMaterialProgram(screenProgram->key.stages);
        assert(renderer.GetMaterialPipelineVariantCount() < variantsBeforeReload);
        assert(retirement.GetStats().pushed > retirementsBeforeReload);
        frame();
        assert(materialResolutions == resolvedBeforeRetained + 1);

        assert(renderer.GetMaterialPipelineVariantCount() <= variantsBeforeReload);
        std::array<uint8_t, 128 * 128 * 4> hotPixels{};
        assert(device.ReadBuffer(output, 0, hotPixels.data(), hotPixels.size()));
        const size_t hotSample = (64 * 128 + 64) * 4;
        assert(hotPixels[hotSample] >= 31 && hotPixels[hotSample] <= 32);
        assert(hotPixels[hotSample + 2] == 255);
        frame();
        assert(materialResolutions == resolvedBeforeRetained + 1);

        // Removing the last GUID reference retires its pipeline after the
        // current submission serial. An earlier completed serial cannot free
        // a pipeline that a recorded/in-flight frame could still reference.
        const auto retirementsBeforeRemoval = retirement.GetStats().pushed;
        renderer.BeginFrame(128, 128);
        begin();
        graph.Execute(command);
        assert(renderer.GetMaterialPipelineVariantCount() == 0);
        assert(retirement.GetStats().pushed > retirementsBeforeRemoval);
        assert(std::find(releasedProgramKeys.begin(), releasedProgramKeys.end(), reloadedScreen->key) !=
               releasedProgramKeys.end());
        const auto pendingBeforeCompletion = retirement.GetStats().pending;
        assert(pendingBeforeCompletion > 0);
        assert(retirement.Collect(epoch - 1) == 0);
        assert(retirement.GetStats().pending == pendingBeforeCompletion);
        finish();
        assert(retirement.GetStats().pending == 0);

        // Changed content forces a real rebuild, unlike the static benchmark.
        // Distinct descriptor sets deliberately prevent texture batching.
        std::vector<rhi::BindGroupHandle> imageGroups;
        for (int i = 0; i < 64; ++i) {
            imageGroups.push_back(device.CreateBindGroup(groupDesc));
            assert(imageGroups.back().IsValid());
        }
        for (auto kind : {ScreenUIList::Overlay, ScreenUIList::World}) {
            list = kind;
            buildGraph();
            for (bool textured : {false, true}) {
                std::vector<std::array<uint8_t, 128 * 128 * 4>> expectedFrames;
                for (bool retained : {false, true}) {
                    std::vector<std::shared_ptr<InxScreenUIRenderer::CommandPacket>> packets(1000);
                    const auto capturesBefore = renderer.GetGeometryStats(list).packetCaptures;
                    std::vector<double> buildSamples, renderSamples;
                    for (int iteration = 0; iteration < 20; ++iteration) {
                        const auto started = std::chrono::steady_clock::now();
                        renderer.BeginFrame(128, 128);
                        for (int i = 0; i < 1000; ++i) {
                            if (retained && packets[i] && i != iteration && i != iteration - 1)
                                continue;
                            if (retained)
                                renderer.BeginCommandPacket();
                            if (list == ScreenUIList::World) {
                                glm::mat4 matrix(1.f);
                                matrix[3].z = .5f;
                                std::array<float, 16> pose{};
                                std::copy_n(glm::value_ptr(matrix), 16, pose.begin());
                                renderer.BeginWorldElement(pose, 50, 20);
                            }
                            if (textured) {
                                const auto textureId = static_cast<uint64_t>(
                                    reinterpret_cast<uintptr_t>(device.Resolve(imageGroups[i % imageGroups.size()])));
                                renderer.AddImage(list, textureId, 0, 0, 100, 40);
                            } else {
                                renderer.AddText(list, 0, 0, 100, 40, i == iteration ? "updated UI" : "UI 041", 1, 1, 1,
                                                 1, 0, 0, 18);
                            }
                            if (list == ScreenUIList::World)
                                renderer.EndWorldElement();
                            if (retained)
                                packets[i] = renderer.EndCommandPacket();
                        }
                        if (retained)
                            renderer.AppendCommandPackets(packets);
                        const double buildMs =
                            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
                                .count();
                        frame();
                        std::array<uint8_t, 128 * 128 * 4> observed{};
                        assert(device.ReadBuffer(output, 0, observed.data(), observed.size()));
                        if (retained)
                            assert(observed == expectedFrames[iteration]);
                        else
                            expectedFrames.push_back(observed);
                        if (iteration >= 4) {
                            buildSamples.push_back(buildMs);
                            renderSamples.push_back(renderMs);
                        }
                    }
                    std::sort(buildSamples.begin(), buildSamples.end());
                    std::sort(renderSamples.begin(), renderSamples.end());
                    std::cout << "UI_DYNAMIC_NATIVE_BENCH list=" << int(list) << " textured=" << textured
                              << " retained=" << retained
                              << " count=1000 build_p50_ms=" << buildSamples[buildSamples.size() / 2]
                              << " record_p50_ms=" << renderSamples[renderSamples.size() / 2]
                              << " draws=" << renderer.GetLastSubmittedDrawCount(list) << std::endl;
                    assert(renderer.GetLastSubmittedDrawCount(list) == (textured ? 1000u : 1u));
                    assert(renderer.GetGeometryStats(list).packetCaptures - capturesBefore == (retained ? 1038u : 0u));
                }
            }
        }
        for (auto imageGroup : imageGroups)
            device.Release(imageGroup);
        list = ScreenUIList::Overlay;
        buildGraph();
        renderer.BeginFrame(128, 128);
        renderer.AddFilledRect(list, 16, 16, 112, 112, 1, 0, 0, 1);
        frame();
        std::array<uint8_t, 128 * 128 * 4> pixels{};
        assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
        const size_t center = (64 * 128 + 64) * 4;
        assert(pixels[center] == 255 && pixels[center + 1] == 0 && pixels[center + 3] == 255);
        renderer.BeginFrame(128, 128);
        frame();
        assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
        assert(pixels[center] == 0 && pixels[center + 3] == 0);
        isolateSlots = true;
        frame();
        isolateSlots = false;
        assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
        const size_t left = (64 * 128 + 32) * 4;
        const size_t right = (64 * 128 + 96) * 4;
        assert(pixels[left] == 255 && pixels[left + 1] == 0);
        assert(pixels[right] == 0 && pixels[right + 1] == 255);

        list = ScreenUIList::World;
        buildGraph();
        renderer.BeginFrame(128, 128);
        auto worldQuad = [&](float z, float red, float blue, float alpha, uint32_t layer) {
            glm::mat4 matrix(1.f);
            matrix[3].z = z;
            std::array<float, 16> pose{};
            std::copy_n(glm::value_ptr(matrix), 16, pose.begin());
            renderer.BeginWorldElement(pose, 50, 50, layer);
            renderer.AddFilledRect(list, 0, 0, 100, 100, red, 0, blue, alpha);
            renderer.EndWorldElement();
        };
        worldQuad(.2f, 1, 0, .5f, 1); // submitted near before far; sort must reverse it
        worldQuad(.8f, 0, 1, 1, 2);
        frame();
        assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
        assert(pixels[center] >= 127 && pixels[center] <= 129 && pixels[center + 2] >= 126);
        const auto prepared = renderer.GetGeometryStats(list).preparations;
        // Reverse camera depth without changing the world geometry.
        camera[2][2] = -1;
        camera[3][2] = 1;
        frame();
        assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
        assert(pixels[center] == 0 && pixels[center + 2] == 255);
        cullingMask = 1;
        frame();
        assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
        assert(pixels[center] >= 127 && pixels[center + 2] == 0);
        assert(renderer.GetGeometryStats(list).preparations == prepared);
        cullingMask = 0;
        frame();
        assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
        assert(pixels[center + 3] == 0 && renderer.GetLastSubmittedDrawCount(list) == 0);
        camera = glm::mat4(1.f);
        cullingMask = 0xffffffffu;
        renderer.BeginFrame(128, 128);
        renderer.PushClipRect(list, 0, 0, 50, 100);
        worldQuad(.5f, 1, 0, 1, 1);
        renderer.PopClipRect(list);
        renderer.PushClipRect(list, 50, 0, 100, 100);
        worldQuad(.5f, 0, 1, .5f, 1);
        renderer.PopClipRect(list);
        frame();
        assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
        const size_t innerLeft = (64 * 128 + 48) * 4;
        const size_t innerRight = (64 * 128 + 80) * 4;
        // World UI is not constrained by screen/canvas clip rectangles. Both
        // elements therefore remain visible even though their authored ImGui
        // clip regions are disjoint; only scene scissor/depth apply.
        for (const size_t sample : {innerLeft, innerRight}) {
            assert(pixels[sample] >= 126 && pixels[sample] <= 129);
            assert(pixels[sample + 2] >= 126 && pixels[sample + 2] <= 129);
            assert(pixels[sample + 3] == 255);
        }
        // Ignored canvas clips cannot split otherwise compatible world draws.
        assert(renderer.GetLastSubmittedDrawCount(list) == 1);
        // Stable equal-depth order and alpha blending must survive batching.
        renderer.BeginFrame(128, 128);
        worldQuad(.5f, 1, 0, 1, 1);
        worldQuad(.5f, 0, 1, .5f, 1);
        frame();
        assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
        assert(pixels[center] >= 126 && pixels[center + 2] >= 127 && pixels[center + 3] == 255);
        assert(renderer.GetLastSubmittedDrawCount(list) == 1);
        renderer.SetEnabled(false);
        frame();
        assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
        assert(pixels[center + 3] == 0);
        renderer.SetEnabled(true);
        frame();
        assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
        assert(pixels[center + 3] == 255);

        // A different texture descriptor is a hard batch boundary.
        const auto secondGroup = device.CreateBindGroup(groupDesc);
        renderer.BeginFrame(128, 128);
        worldQuad(.5f, 1, 0, 1, 1);
        glm::mat4 poseMatrix(1.f);
        poseMatrix[3].z = .5f;
        std::array<float, 16> pose{};
        std::copy_n(glm::value_ptr(poseMatrix), 16, pose.begin());
        renderer.BeginWorldElement(pose, 50, 50);
        renderer.AddImage(list, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(device.Resolve(secondGroup))), 0, 0,
                          100, 100, 0, 0, 1, 1, 0, 0, 1, .5f);
        renderer.EndWorldElement();
        frame();
        assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
        assert(pixels[center] >= 126 && pixels[center + 2] >= 127);
        assert(renderer.GetLastSubmittedDrawCount(list) == 2);
        renderer.BeginFrame(128, 128);
        device.Release(secondGroup);

        // Crossing ImGui's 16-bit index range requires distinct base vertices.
        for (int i = 0; i < 4000; ++i) {
            renderer.BeginWorldElement(pose, 50, 50);
            renderer.AddText(list, 0, 0, 100, 100, "UI 041", 1, 1, 1, 1, 0, 0, 18);
            renderer.EndWorldElement();
        }
        frame();
        assert(renderer.GetLastSubmittedDrawCount(list) == (sizeof(ImDrawIdx) == 2 ? 2 : 1));

        // Compare retained and fresh geometry byte-for-byte through real GPU
        // readback: clip, rotation/mirror, HDR tint, textures, world layers,
        // alpha order, mixed direct/retained submission and large index ranges.
        using Packet = std::shared_ptr<InxScreenUIRenderer::CommandPacket>;
        for (auto kind : {ScreenUIList::Camera, ScreenUIList::Overlay, ScreenUIList::World}) {
            list = kind;
            buildGraph();
            const auto decorate = [&](int item, bool large = false) {
                glm::mat4 matrix(1.f);
                matrix[3] = glm::vec4(float(item) * .05f, .1f, item == 1 ? .7f : .4f, 1.f);
                std::array<float, 16> transform{};
                std::copy_n(glm::value_ptr(matrix), 16, transform.begin());
                if (list == ScreenUIList::World)
                    renderer.BeginWorldElement(transform, 64, 64, 1u << item);
                renderer.PushClipRect(list, 12, 17, 116, 107);
                renderer.AddFilledRect(list, 8, 8, 112, 114, 1.7f, .2f, .4f, .4f, 8, 23, false, true);
                renderer.AddImage(list, static_cast<uint64_t>(io.Fonts->TexRef.GetTexID()), 10, 21, 82, 101, 0, 0, 1, 1,
                                  .2f, .6f, 1, .6f, -18, true, false, 5);
                const int repeats = large ? 3500 : 1;
                for (int n = 0; n < repeats; ++n)
                    renderer.AddText(list, 13, 20, 109, 106, "Alpha\nBeta UI", .2f, 1.4f, .6f, .2f, .5f, .5f, 18, 72,
                                     13, true, false, "", 1.2f, 2, true);
                renderer.PopClipRect(list);
                if (list == ScreenUIList::World)
                    renderer.EndWorldElement();
            };
            for (bool large : {false, true}) {
                renderer.BeginFrame(128, 128);
                decorate(0);
                decorate(1, large);
                decorate(2);
                frame();
                std::array<uint8_t, 128 * 128 * 4> expected{};
                assert(device.ReadBuffer(output, 0, expected.data(), expected.size()));
                const auto indices = renderer.GetLastSubmittedIndexCount(list);
                renderer.BeginFrame(128, 128);
                renderer.BeginCommandPacket();
                decorate(1, large);
                Packet retained = renderer.EndCommandPacket();
                shutdownPacket = retained;
                decorate(0);
                renderer.AppendCommandPackets({retained});
                decorate(2);
                frame();
                assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
                assert(pixels == expected);
                assert(renderer.GetLastSubmittedIndexCount(list) == indices);
                if (list == ScreenUIList::World) {
                    cullingMask = 2;
                    frame();
                    std::array<uint8_t, 128 * 128 * 4> masked{};
                    assert(device.ReadBuffer(output, 0, masked.data(), masked.size()));
                    renderer.BeginFrame(128, 128);
                    decorate(0);
                    decorate(1, large);
                    decorate(2);
                    frame();
                    assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
                    assert(pixels == masked);
                    cullingMask = 0xffffffffu;
                }
                renderer.BeginFrame(128, 128);
                decorate(0);
                renderer.AppendCommandPackets({retained});
                decorate(2);
                frame();
                assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
                assert(pixels == expected);
            }
        }
        // Scene-bound packets keep local glyph/shape geometry immutable.
        // Pose/layer is sampled at publication, including uncommitted frame
        // cache writes. Compare with explicit fresh geometry through Vulkan.
        list = ScreenUIList::World;
        buildGraph();
        auto *uiScene = SceneManager::Instance().CreateScene("Retained world UI");
        auto *parent = uiScene->CreateGameObject("Parent");
        auto *object = uiScene->CreateGameObject("Animated UI");
        object->SetParent(parent, false);
        object->GetTransform()->SetLocalPosition(.1f, .1f, .4f);
        renderer.BeginFrame(128, 128);
        assert(!renderer.HasSelectiveWorldOcclusion());
        renderer.BeginCommandPacket();
        renderer.BeginWorldObject(object, 50, 50, false, false, false, parent->GetID());
        renderer.AddFilledRect(list, 0, 0, 10, 10, 1, 1, 1, 1);
        renderer.EndWorldElement();
        auto selectivePacket = renderer.EndCommandPacket();
        assert(!renderer.HasSelectiveWorldOcclusion()); // Capture alone does not draw.
        renderer.AppendCommandPackets({selectivePacket});
        assert(renderer.HasSelectiveWorldOcclusion());
        assert(!renderer.HasSelectiveWorldOcclusion(1u << 30));
        const auto selectiveRuns = renderer.GetWorldDepthRuns(glm::mat4(1.0f));
        assert(selectiveRuns.size() == 1 && selectiveRuns[0].ignoredOccluderId == parent->GetID());
        assert(!renderer.BeginFrameCached(128, 128, 0x57u)); // Appending invalidated the cached frame.
        assert(renderer.BeginFrameCached(128, 128, 0x57u));  // Reusing the empty frame keeps it empty.
        assert(!renderer.HasSelectiveWorldOcclusion());
        const auto drawLocal = [&] {
            renderer.AddFilledRect(list, 10, 20, 85, 70, .3f, .8f, 1.2f, .7f, 7);
            renderer.AddText(list, 10, 20, 85, 70, "UI", 1, 1, 1, .8f, .5f, .5f, 18);
        };
        renderer.BeginFrame(128, 128);
        renderer.BeginCommandPacket();
        renderer.BeginWorldObject(object, 50, 50);
        drawLocal();
        renderer.EndWorldElement();
        auto animated = renderer.EndCommandPacket();
        const auto captureCount = renderer.GetGeometryStats(list).packetCaptures;
        auto &transforms = TransformECSStore::Instance();
        std::array<uint8_t, 128 * 128 * 4> previousPose{};
        for (int step = 0; step < 4; ++step) {
            parent->GetTransform()->SetLocalEulerAngles(8, float(step * 13), 7);
            parent->GetTransform()->SetLocalScale(1.5f, .7f, 1.f);
            // Its own scale must not stretch UI, but parent scale affects its position.
            object->GetTransform()->SetLocalScale(3.f, .2f, 2.f);
            transforms.BeginFrameCache();
            object->GetTransform()->SetPosition(float(step) * .12f, .05f, .4f);
            object->SetLayer(step);
            for (bool visible : {true, false}) {
                cullingMask = visible ? (1u << step) : (1u << (step + 1));
                renderer.BeginFrame(128, 128);
                renderer.AppendCommandPackets({animated});
                frame();
                std::array<uint8_t, 128 * 128 * 4> retained{};
                assert(device.ReadBuffer(output, 0, retained.data(), retained.size()));
                glm::mat4 current = glm::mat4_cast(object->GetTransform()->GetRotation());
                current[3] = glm::vec4(object->GetTransform()->GetPosition(), 1);
                std::array<float, 16> explicitPose{};
                std::copy_n(glm::value_ptr(current), 16, explicitPose.begin());
                renderer.BeginFrame(128, 128);
                renderer.BeginWorldElement(explicitPose, 50, 50, 1u << step);
                drawLocal();
                renderer.EndWorldElement();
                frame();
                assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
                assert(pixels == retained);
                assert(renderer.GetGeometryStats(list).packetCaptures == captureCount);
                if (visible) {
                    assert(pixels != previousPose);
                    previousPose = pixels;
                } else {
                    assert(std::all_of(pixels.begin(), pixels.end(), [](uint8_t value) { return value == 0; }));
                }
            }
            (void)transforms.EndFrameCache();
        }
        cullingMask = 0xffffffffu;
        // Screen-bound packets resolve Transform pose at publication too.
        // A pose-only change must reach the GPU even when geometry revision
        // is unchanged.
        list = ScreenUIList::Overlay;
        buildGraph();
        auto *screenParent = uiScene->CreateGameObject("Screen Canvas Parent");
        auto *screenObject = uiScene->CreateGameObject("Animated screen UI");
        screenObject->SetParent(screenParent, false);
        renderer.BeginFrame(128, 128);
        renderer.BeginCommandPacket();
        renderer.BeginScreenObject(screenObject, list, 40, 40);
        renderer.AddFilledRect(list, 8, 8, 72, 56, 1, 0, 0, 1);
        renderer.EndScreenObject();
        auto screenPacket = renderer.EndCommandPacket();
        renderer.BeginFrame(128, 128);
        renderer.AppendCommandPackets({screenPacket});
        frame();
        // The first Vulkan submission may only populate the newly-created
        // upload allocation; use one settled frame as the geometry baseline.
        renderer.BeginFrame(128, 128);
        renderer.AppendCommandPackets({screenPacket});
        frame();
        std::array<uint8_t, 128 * 128 * 4> screenBefore{};
        assert(device.ReadBuffer(output, 0, screenBefore.data(), screenBefore.size()));
        const auto bounds = [](const auto &pixels) {
            std::array<int, 4> result{128, 128, -1, -1};
            for (int y = 0; y < 128; ++y) {
                for (int x = 0; x < 128; ++x) {
                    if (pixels[static_cast<size_t>((y * 128 + x) * 4 + 3)] == 0)
                        continue;
                    result[0] = std::min(result[0], x);
                    result[1] = std::min(result[1], y);
                    result[2] = std::max(result[2], x);
                    result[3] = std::max(result[3], y);
                }
            }
            return result;
        };
        const auto screenBeforeBounds = bounds(screenBefore);
        // Screen-space geometry is independent of the Canvas/parent world
        // pose.  Moving, rotating or scaling that parent must not move the
        // retained packet; changing the element's own local pose still does.
        auto assertParentPoseIgnored = [&](const char *label) {
            renderer.BeginFrame(128, 128);
            renderer.AppendCommandPackets({screenPacket});
            frame();
            std::array<uint8_t, 128 * 128 * 4> current{};
            assert(device.ReadBuffer(output, 0, current.data(), current.size()));
            const auto currentBounds = bounds(current);
            std::cout << "SCREEN_PARENT_POSE " << label << " bounds=" << currentBounds[0] << "," << currentBounds[1]
                      << "," << currentBounds[2] << "," << currentBounds[3] << " baseline=" << screenBeforeBounds[0]
                      << "," << screenBeforeBounds[1] << "," << screenBeforeBounds[2] << "," << screenBeforeBounds[3]
                      << std::endl;
            assert(currentBounds == screenBeforeBounds);
        };
        screenParent->GetTransform()->SetPosition(.5f, 0.25f, 3.0f);
        assertParentPoseIgnored("position");
        screenParent->GetTransform()->SetLocalEulerAngles(15.0f, 25.0f, 35.0f);
        assertParentPoseIgnored("rotation");
        screenParent->GetTransform()->SetLocalScale(2.0f, .5f, 4.0f);
        assertParentPoseIgnored("scale");
        screenObject->GetTransform()->SetLocalPosition(20.0f, 0.0f, 0.0f);
        renderer.BeginFrame(128, 128);
        renderer.AppendCommandPackets({screenPacket});
        frame();
        std::array<uint8_t, 128 * 128 * 4> screenAfter{};
        assert(device.ReadBuffer(output, 0, screenAfter.data(), screenAfter.size()));
        assert(bounds(screenAfter) != screenBeforeBounds);
        screenObject->GetTransform()->SetLocalEulerAngles(0.0f, 0.0f, 25.0f);
        screenObject->GetTransform()->SetLocalScale(1.5f, 0.75f, 1.0f);
        renderer.BeginFrame(128, 128);
        renderer.AppendCommandPackets({screenPacket});
        frame();
        std::array<uint8_t, 128 * 128 * 4> screenAfterRotateScale{};
        assert(device.ReadBuffer(output, 0, screenAfterRotateScale.data(), screenAfterRotateScale.size()));
        assert(screenAfterRotateScale != screenAfter);
        screenPacket.reset();

        // No owning scene pointer is retained; recycling a Transform slot
        // must reject an expired packet rather than attach it to a new object.
        uiScene->DestroyGameObject(object);
        uiScene->ProcessPendingDestroys();
        auto *replacement = uiScene->CreateGameObject("Replacement");
        (void)replacement;
        renderer.BeginFrame(128, 128);
        bool expired = false;
        try {
            renderer.AppendCommandPackets({animated});
        } catch (const std::runtime_error &) {
            expired = true;
        }
        assert(expired);
        animated.reset();

        // Moving 1000 scene objects must not regenerate 1000 text packets.
        // Both policies use the same native backend and render identical frames.
        auto *crowdRoot = uiScene->CreateGameObject("Animated UI crowd");
        std::vector<GameObject *> crowd;
        for (int index = 0; index < 1000; ++index) {
            auto *label = uiScene->CreateGameObject("Label");
            label->SetParent(crowdRoot, false);
            label->GetTransform()->SetLocalPosition(0, 0, .5f);
            crowd.push_back(label);
        }
        std::vector<std::array<uint8_t, 128 * 128 * 4>> poseFrames;
        for (bool retain : {false, true}) {
            std::vector<Packet> packets(1000);
            std::vector<double> buildTimes, recordTimes;
            const auto captures = renderer.GetGeometryStats(list).packetCaptures;
            for (int iteration = 0; iteration < 12; ++iteration) {
                crowdRoot->GetTransform()->SetLocalEulerAngles(0, float(iteration * 2), 0);
                const auto started = std::chrono::steady_clock::now();
                renderer.BeginFrame(128, 128);
                if (!retain || iteration == 0) {
                    for (int index = 0; index < 1000; ++index) {
                        renderer.BeginCommandPacket();
                        renderer.BeginWorldObject(crowd[index], 50, 20);
                        renderer.AddText(list, 0, 0, 100, 40, "UI 041", 1, 1, 1, 1, 0, 0, 18);
                        renderer.EndWorldElement();
                        packets[index] = renderer.EndCommandPacket();
                    }
                }
                renderer.AppendCommandPackets(packets);
                const double buildTime =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
                frame();
                assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
                if (retain)
                    assert(pixels == poseFrames[iteration]);
                else
                    poseFrames.push_back(pixels);
                if (iteration >= 4) {
                    buildTimes.push_back(buildTime);
                    recordTimes.push_back(renderMs);
                }
            }
            std::sort(buildTimes.begin(), buildTimes.end());
            std::sort(recordTimes.begin(), recordTimes.end());
            std::cout << "UI_WORLD_POSE_NATIVE_BENCH count=1000 retained=" << retain
                      << " build_p50_ms=" << buildTimes[buildTimes.size() / 2]
                      << " record_p50_ms=" << recordTimes[recordTimes.size() / 2] << std::endl;
            assert(renderer.GetGeometryStats(list).packetCaptures - captures == (retain ? 1000u : 12000u));
        }
        renderer.BeginFrame(128, 128);
        renderer.BeginCommandPacket();
        renderer.BeginWorldElement(pose, 50, 50);
        renderer.AbortCommandPacket();
        renderer.BeginCommandPacket();
        const auto empty = renderer.EndCommandPacket();
        renderer.AppendCommandPackets({empty});
        assert(!renderer.HasCommands(ScreenUIList::World));
        assert(!renderer.BeginFrameCached(128, 128, 7788));
        assert(renderer.BeginFrameCached(128, 128, 7788));
        const auto fontEpoch = renderer.GetCommandPacketEpoch();
        textlayout::ClearFontCache();
        assert(renderer.GetCommandPacketEpoch() != fontEpoch);
        assert(!renderer.BeginFrameCached(128, 128, 7788));
        renderer.InvalidateMaterialProgram(activeScreenProgram->key.stages);
        renderer.InvalidateMaterialProgram(worldProgram->key.stages);
        assert(renderer.GetMaterialPipelineVariantCount() == 0);
        // Mirror an MSAA renderer generation cutover: the replacement and the
        // retired renderer may own the same key until the old submission ends.
        std::unordered_map<ShaderProgramKey, size_t, ShaderProgramKeyHash> uiOwners;
        std::vector<ShaderProgramKey> lastOwnerReleased;
        auto acquireUI = [&](const ShaderProgramKey &key) { ++uiOwners[key]; };
        auto releaseUI = [&](const ShaderProgramKey &key) {
            const auto owner = uiOwners.find(key);
            assert(owner != uiOwners.end() && owner->second != 0);
            if (--owner->second == 0) {
                uiOwners.erase(owner);
                lastOwnerReleased.push_back(key);
            }
        };
        renderer.SetMaterialProgramAcquire(acquireUI);
        renderer.SetMaterialProgramRelease(releaseUI);
        list = ScreenUIList::Overlay;
        buildGraph();
        renderer.BeginFrame(128, 128);
        renderer.BeginCommandPacket();
        renderer.SetMaterialBinding(list, "ui-authored-guid", 1, "test-ui-program");
        renderer.AddFilledRect(list, 0, 0, 100, 100, 1, 1, 1, 1);
        renderer.AppendCommandPackets({renderer.EndCommandPacket()});
        frame();
        const auto sharedKey = activeScreenProgram->key;
        assert(uiOwners.at(sharedKey) == 1);

        auto replacementUI = std::make_unique<InxScreenUIRenderer>();
        assert(replacementUI->Initialize(context.GetDevice(), context.GetVmaAllocator(), device.GetDescriptorManager(),
                                         VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D32_SFLOAT, VK_SAMPLE_COUNT_1_BIT, 4));
        replacementUI->SetRetirementQueue(&retirement);
        replacementUI->SetMaterialProgramAcquire(acquireUI);
        replacementUI->SetMaterialProgramRelease(releaseUI);
        replacementUI->SetMaterialProgramResolver([&](const std::string &guid, uint64_t generation,
                                                      ShaderProgramDomain requestedDomain) {
            assert(generation == 1);
            assert(requestedDomain == ShaderProgramDomain::ScreenUI || requestedDomain == ShaderProgramDomain::WorldUI);
            return guid == "world-ui-authored-guid" ? worldProgram : activeScreenProgram;
        });
        activeRenderer = replacementUI.get();
        replacementUI->BeginFrame(128, 128);
        replacementUI->BeginCommandPacket();
        replacementUI->SetMaterialBinding(list, "ui-authored-guid", 1, "test-ui-program");
        replacementUI->AddFilledRect(list, 0, 0, 100, 100, 1, 1, 1, 1);
        replacementUI->AppendCommandPackets({replacementUI->EndCommandPacket()});
        frame();
        assert(uiOwners.at(sharedKey) == 2);
        const auto pendingBeforeOldDestroy = retirement.GetStats().pending;
        renderer.Destroy();
        assert(uiOwners.at(sharedKey) == 1);
        assert(lastOwnerReleased.empty());
        assert(retirement.GetStats().pending > pendingBeforeOldDestroy);
        assert(retirement.Collect(epoch - 1) == 0);
        begin();
        finish();

        list = ScreenUIList::World;
        buildGraph();
        replacementUI->BeginFrame(128, 128);
        replacementUI->BeginCommandPacket();
        replacementUI->SetMaterialBinding(ScreenUIList::Overlay, "ui-authored-guid", 1, "test-ui-program");
        replacementUI->AddFilledRect(ScreenUIList::Overlay, 0, 0, 100, 100, 1, 1, 1, 1);
        replacementUI->AppendCommandPackets({replacementUI->EndCommandPacket()});
        replacementUI->BeginCommandPacket();
        replacementUI->SetMaterialBinding(list, "world-ui-authored-guid", 1, "test-ui-program");
        replacementUI->BeginWorldElement(pose, 50, 50);
        replacementUI->AddFilledRect(list, 0, 0, 100, 100, 1, 1, 1, 1);
        replacementUI->EndWorldElement();
        replacementUI->AppendCommandPackets({replacementUI->EndCommandPacket()});
        frame();
        assert(uiOwners.at(worldProgram->key) == 1);
        assert(uiOwners.at(sharedKey) == 1 && lastOwnerReleased.empty());
        replacementUI->Destroy();
        assert(uiOwners.empty());
        assert(std::find(lastOwnerReleased.begin(), lastOwnerReleased.end(), sharedKey) != lastOwnerReleased.end());
        assert(std::find(lastOwnerReleased.begin(), lastOwnerReleased.end(), worldProgram->key) !=
               lastOwnerReleased.end());
        begin();
        finish();
        graph.Destroy();
        device.Release(output);
    }
    renderer.Destroy();
    ImGui::EndFrame();
    ImGui::DestroyContext();
    shutdownPacket.reset(); // CPU packets have no dangling ImGui/GPU owners.
    device.Release(group);
    device.Release(layout);
    device.Release(sampler);
    device.Release(view);
    device.Release(white);
    device.CollectDescriptorRetirements(epoch);
    device.CollectResourceRetirements(epoch);
    retirement.FlushAll();
    vkDestroyFence(context.GetDevice(), fence, nullptr);
    vkDestroyCommandPool(context.GetDevice(), pool, nullptr);
    context.Destroy();
    SDL_DestroyWindow(window);
    SDL_Quit();
    std::cout << "PASS UI readback, retained geometry, frame-slot isolation and camera depth/layer order\n";
}
