#include <SDL3/SDL.h>
#include <function/renderer/vk/RhiVulkanTypes.h>
#include <function/renderer/vk/VkDeviceContext.h>
#include <function/renderer/vk/VulkanRhiDevice.h>
#include <function/resources/ShaderAsset/SlangCompiler.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace infernux;

namespace
{
void Require(bool result, const char *message)
{
    if (!result)
        throw std::runtime_error(message);
}

constexpr const char *kProgram = R"(
struct Parameters { float4 tint; uint objectId; };
ConstantBuffer<Parameters> params;
RWStructuredBuffer<float4> positions;
Texture2D<float4> albedo;
SamplerState albedoSampler;

[shader("compute")]
[numthreads(4, 1, 1)]
void prepare_vertices(uint3 id : SV_DispatchThreadID)
{
    positions[id.x] = float4(float(id.x & 1) * 2.0 - 1.0, float(id.x >> 1) * 2.0 - 1.0, 0.0, 1.0);
}

struct VertexInput { float3 position : POSITION; float2 uv : TEXCOORD0; };
struct Varyings { float4 position : SV_Position; float2 uv : TEXCOORD0; };
[shader("vertex")]
Varyings mesh_vertex(VertexInput input)
{
    Varyings result;
    result.position = float4(input.position, 1.0);
    result.uv = input.uv;
    return result;
}

struct Targets { float4 color : SV_Target0; uint2 identity : SV_Target1; };
[shader("fragment")]
Targets mesh_fragment(Varyings input)
{
    Targets result;
    result.color = albedo.Sample(albedoSampler, input.uv) * params.tint;
    result.identity = uint2(params.objectId, 0xfedcba98u);
    return result;
}
)";

struct Fixture
{
    SDL_Window *window = nullptr;
    vk::VkDeviceContext context;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;

    ~Fixture()
    {
        if (context.IsValid()) {
            context.WaitIdleForShutdown();
            const auto device = context.GetDevice();
            if (framebuffer)
                vkDestroyFramebuffer(device, framebuffer, nullptr);
            if (renderPass)
                vkDestroyRenderPass(device, renderPass, nullptr);
            if (fence)
                vkDestroyFence(device, fence, nullptr);
            if (pool)
                vkDestroyCommandPool(device, pool, nullptr);
            context.Destroy();
        }
        if (window)
            SDL_DestroyWindow(window);
        SDL_Quit();
    }
};

void ImageBarrier(VkCommandBuffer commands, VkImage image, VkImageLayout before, VkImageLayout after,
                  VkPipelineStageFlags sourceStage, VkPipelineStageFlags destinationStage, VkAccessFlags sourceAccess,
                  VkAccessFlags destinationAccess)
{
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = sourceAccess;
    barrier.dstAccessMask = destinationAccess;
    barrier.oldLayout = before;
    barrier.newLayout = after;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(commands, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void Run()
{
    Fixture fixture;
    Require(SDL_Init(SDL_INIT_VIDEO), SDL_GetError());
    fixture.window = SDL_CreateWindow("Slang RHI mesh", 32, 32, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
    Require(fixture.window != nullptr, SDL_GetError());
    vk::DeviceConfig config;
    config.enableValidationLayers = true;
    Require(fixture.context.Initialize(fixture.window, config), "Vulkan initialization failed");
    auto &device = fixture.context.GetRhiDevice();
    const auto nativeDevice = fixture.context.GetDevice();
    const auto compiled = SlangCompiler::Get().Compile({"mesh", "mesh.compute.py", kProgram}, {},
                                                       {ShaderCodeTarget::SpirV, ShaderCodeTarget::Wgsl});
    const auto &program = compiled.targets[0];
    const auto reflection = nlohmann::json::parse(program.reflection);
    std::array<rhi::ShaderModuleHandle, 3> shaders{};
    std::array<std::string, 3> entryNames{};
    for (size_t index = 0; index < shaders.size(); ++index) {
        const auto &entry = program.entries[index];
        shaders[index] =
            device.CreateShaderModule({rhi::ShaderSourceLanguage::SpirV, entry.code.data(), entry.code.size()});
        entryNames[index] = entry.name;
        Require(shaders[index].IsValid(), "Slang shader module creation failed");
    }
    auto buffer = [&](uint64_t bytes, rhi::BufferUsageFlags usage, rhi::BufferMemory memory,
                      const void *initial = nullptr) {
        rhi::BufferDesc desc;
        desc.byteSize = bytes;
        desc.usage = usage;
        desc.memory = memory;
        desc.initialData = initial;
        desc.initialDataBytes = initial ? bytes : 0;
        const auto result = device.CreateBuffer(desc);
        Require(result.IsValid(), "RHI buffer allocation failed");
        return result;
    };
    const auto vertices =
        buffer(4 * 16, rhi::BufferUsageFlags::Storage | rhi::BufferUsageFlags::Vertex, rhi::BufferMemory::DeviceLocal);
    const std::array<float, 8> uvs{0, 0, 1, 0, 0, 1, 1, 1};
    const auto uvBuffer = buffer(sizeof(uvs), rhi::BufferUsageFlags::Vertex, rhi::BufferMemory::Upload, uvs.data());
    const std::array<uint16_t, 6> indices{0, 1, 2, 2, 1, 3};
    const auto indexBuffer =
        buffer(sizeof(indices), rhi::BufferUsageFlags::Index, rhi::BufferMemory::Upload, indices.data());

    const auto &uniformLayout = reflection["parameters"][0]["type"]["elementVarLayout"]["type"];
    const auto &fields = uniformLayout["fields"];
    std::array<char, 32> constants{};
    const std::array<float, 4> tint{0.5f, 0.75f, 1.0f, 1.0f};
    const uint32_t objectId = 0x1020304;
    std::memcpy(constants.data() + fields[0]["binding"]["offset"].get<size_t>(), tint.data(), sizeof(tint));
    std::memcpy(constants.data() + fields[1]["binding"]["offset"].get<size_t>(), &objectId, sizeof(objectId));
    const auto uniform =
        buffer(constants.size(), rhi::BufferUsageFlags::Uniform, rhi::BufferMemory::Upload, constants.data());

    rhi::TextureDesc sourceDesc;
    sourceDesc.width = sourceDesc.height = 2;
    sourceDesc.format = rhi::PixelFormat::RGBA8UNorm;
    sourceDesc.mutableFormat = true;
    sourceDesc.usage =
        rhi::TextureUsageFlags::Sampled | rhi::TextureUsageFlags::Storage | rhi::TextureUsageFlags::TransferDestination;
    const auto source = device.CreateTexture(sourceDesc);
    rhi::TextureViewDesc sourceViewDesc;
    sourceViewDesc.texture = source;
    sourceViewDesc.format = rhi::PixelFormat::RGBA8Srgb;
    sourceViewDesc.usage = rhi::TextureUsageFlags::Sampled;
    const auto sourceView = device.CreateTextureView(sourceViewDesc);
    Require(source.IsValid() && sourceView.IsValid(), "UNORM storage + sampled sRGB view failed");
    rhi::SamplerDesc samplerDesc;
    samplerDesc.magFilter = samplerDesc.minFilter = rhi::FilterMode::Nearest;
    samplerDesc.addressU = samplerDesc.addressV = rhi::AddressMode::ClampToEdge;
    const auto sampler = device.CreateSampler(samplerDesc);
    std::array<uint8_t, 512> texels{};
    const std::array<uint8_t, 8> top{128, 64, 192, 255, 255, 128, 64, 255};
    const std::array<uint8_t, 8> bottom{64, 255, 128, 255, 192, 128, 255, 255};
    std::memcpy(texels.data(), top.data(), top.size());
    std::memcpy(texels.data() + 256, bottom.data(), bottom.size());
    const auto upload =
        buffer(texels.size(), rhi::BufferUsageFlags::TransferSource, rhi::BufferMemory::Upload, texels.data());

    // Binding use comes from Slang's post-codegen metadata. In particular,
    // a writable Compute output cannot remain in a Draw bind group even when
    // its binding has Compute-only visibility: WebGPU counts the whole group
    // in the render-pass usage scope.
    auto bindingsFor = [&](bool graphics) {
        rhi::BindingLayoutDesc bindingDesc;
        rhi::BindGroupDesc groupDesc;
        const std::array<rhi::BindingType, 4> types{rhi::BindingType::UniformBuffer, rhi::BindingType::StorageBuffer,
                                                    rhi::BindingType::SampledTexture, rhi::BindingType::Sampler};
        for (size_t index = 0; index < types.size(); ++index) {
            const uint32_t binding = reflection["parameters"][index]["binding"]["index"];
            rhi::ShaderStage visibility = rhi::ShaderStage::None;
            const size_t first = graphics ? 1 : 0;
            const size_t end = graphics ? 3 : 1;
            for (size_t entryIndex = first; entryIndex < end; ++entryIndex) {
                const auto &entry = program.entries[entryIndex];
                const std::array<uint32_t, 2> key{0, binding};
                if (std::find(entry.bindings.begin(), entry.bindings.end(), key) != entry.bindings.end())
                    visibility = visibility | entry.stage;
            }
            if (visibility == rhi::ShaderStage::None)
                continue;
            bindingDesc.entries[bindingDesc.entryCount++] = {binding, types[index], visibility, 1};
            if (index < 2)
                groupDesc.buffers[groupDesc.bufferCount++] = {binding, types[index], index == 0 ? uniform : vertices};
            else
                groupDesc.textures[groupDesc.textureCount++] = {binding, types[index],
                                                                index == 2 ? sourceView : rhi::TextureViewHandle{},
                                                                index == 3 ? sampler : rhi::SamplerHandle{}};
        }
        groupDesc.layout = device.CreateBindingLayout(bindingDesc);
        const auto group = device.CreateBindGroup(groupDesc);
        Require(groupDesc.layout.IsValid() && group.IsValid(), "Reflected RHI bindings failed");
        return std::pair{groupDesc.layout, group};
    };
    const auto [computeLayout, computeGroup] = bindingsFor(false);
    const auto [graphicsLayout, graphicsGroup] = bindingsFor(true);
    rhi::ComputePipelineDesc computeDesc;
    computeDesc.computeShader = shaders[0];
    computeDesc.entryPoint = entryNames[0].c_str();
    computeDesc.bindingLayouts[0] = computeLayout;
    computeDesc.bindingLayoutCount = 1;
    const auto compute = device.CreateComputePipeline(computeDesc);
    Require(compute.IsValid(), "Slang compute pipeline failed");

    constexpr uint32_t size = 16;
    std::array<rhi::TextureHandle, 2> targets{};
    std::array<rhi::TextureViewHandle, 2> views{};
    const std::array<rhi::PixelFormat, 2> formats{rhi::PixelFormat::RGBA8UNorm, rhi::PixelFormat::RG32UInt};
    std::array<VkAttachmentDescription, 2> attachments{};
    std::array<VkAttachmentReference, 2> references{};
    std::array<VkImageView, 2> nativeViews{};
    for (size_t index = 0; index < targets.size(); ++index) {
        rhi::TextureDesc desc;
        desc.width = desc.height = size;
        desc.format = formats[index];
        desc.usage = rhi::TextureUsageFlags::ColorAttachment | rhi::TextureUsageFlags::TransferSource;
        targets[index] = device.CreateTexture(desc);
        rhi::TextureViewDesc view;
        view.texture = targets[index];
        views[index] = device.CreateTextureView(view);
        nativeViews[index] = device.Resolve(views[index]);
        attachments[index].format = rhi::ToVkFormat(formats[index]);
        attachments[index].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[index].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[index].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[index].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[index].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        references[index] = {static_cast<uint32_t>(index), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    }
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 2;
    subpass.pColorAttachments = references.data();
    VkRenderPassCreateInfo renderPass{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    renderPass.attachmentCount = 2;
    renderPass.pAttachments = attachments.data();
    renderPass.subpassCount = 1;
    renderPass.pSubpasses = &subpass;
    Require(vkCreateRenderPass(nativeDevice, &renderPass, nullptr, &fixture.renderPass) == VK_SUCCESS,
            "Render pass failed");
    VkFramebufferCreateInfo framebuffer{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    framebuffer.renderPass = fixture.renderPass;
    framebuffer.attachmentCount = 2;
    framebuffer.pAttachments = nativeViews.data();
    framebuffer.width = framebuffer.height = size;
    framebuffer.layers = 1;
    Require(vkCreateFramebuffer(nativeDevice, &framebuffer, nullptr, &fixture.framebuffer) == VK_SUCCESS,
            "Framebuffer failed");
    rhi::GraphicsPipelineDesc graphicsDesc;
    graphicsDesc.vertexShader = shaders[1];
    graphicsDesc.fragmentShader = shaders[2];
    graphicsDesc.vertexEntryPoint = entryNames[1].c_str();
    graphicsDesc.fragmentEntryPoint = entryNames[2].c_str();
    graphicsDesc.renderTargetLayout = device.RegisterRenderTargetLayout(fixture.renderPass);
    graphicsDesc.raster.cullMode = rhi::CullMode::None;
    graphicsDesc.bindingLayouts[0] = graphicsLayout;
    graphicsDesc.bindingLayoutCount = 1;
    graphicsDesc.colorTargetCount = 2;
    graphicsDesc.colorTargets[0].format = formats[0];
    graphicsDesc.colorTargets[1].format = formats[1];
    graphicsDesc.vertexInput.bufferCount = 2;
    graphicsDesc.vertexInput.buffers[0] = {16, rhi::VertexStepMode::Vertex};
    graphicsDesc.vertexInput.buffers[1] = {8, rhi::VertexStepMode::Vertex};
    graphicsDesc.vertexInput.attributeCount = 2;
    graphicsDesc.vertexInput.attributes[0] = {0, 0, rhi::VertexFormat::Float32x3, 0};
    graphicsDesc.vertexInput.attributes[1] = {1, 1, rhi::VertexFormat::Float32x2, 0};
    const auto graphics = device.CreateGraphicsPipeline(graphicsDesc);
    Require(graphics.IsValid(), "Slang mesh graphics pipeline failed");
    const auto colorReadback =
        buffer(size * 256, rhi::BufferUsageFlags::TransferDestination, rhi::BufferMemory::Readback);
    const auto idReadback = buffer(size * 256, rhi::BufferUsageFlags::TransferDestination, rhi::BufferMemory::Readback);
    VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool.queueFamilyIndex = fixture.context.GetQueueIndices().graphicsFamily.value();
    Require(vkCreateCommandPool(nativeDevice, &pool, nullptr, &fixture.pool) == VK_SUCCESS, "Command pool failed");
    VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation.commandPool = fixture.pool;
    allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation.commandBufferCount = 1;
    VkCommandBuffer commands;
    Require(vkAllocateCommandBuffers(nativeDevice, &allocation, &commands) == VK_SUCCESS, "Command allocation failed");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    Require(vkBeginCommandBuffer(commands, &begin) == VK_SUCCESS, "Command begin failed");
    vk::VulkanTransferCommandContext transferContext;
    const auto transfer = device.MakeTransferCommandEncoder(transferContext, commands);
    ImageBarrier(commands, device.Resolve(source), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    rhi::TextureBufferCopyRegion uploadRegion;
    uploadRegion.width = uploadRegion.height = 2;
    uploadRegion.bytesPerRow = 256;
    Require(transfer.CopyBufferToTexture(upload, source, uploadRegion), "Texture upload failed");
    ImageBarrier(commands, device.Resolve(source), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    vk::VulkanComputeCommandContext computeContext;
    const auto computeCommands = device.MakeComputeCommandEncoder(computeContext, commands);
    computeCommands.BindPipeline(compute);
    computeCommands.BindGroup(compute, 0, computeGroup);
    computeCommands.Dispatch(1);
    VkMemoryBarrier vertexReady{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    vertexReady.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    vertexReady.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 1,
                         &vertexReady, 0, nullptr, 0, nullptr);
    std::array<VkClearValue, 2> clears{};
    VkRenderPassBeginInfo render{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    render.renderPass = fixture.renderPass;
    render.framebuffer = fixture.framebuffer;
    render.renderArea.extent = {size, size};
    render.clearValueCount = 2;
    render.pClearValues = clears.data();
    vkCmdBeginRenderPass(commands, &render, VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport viewport{0, 0, float(size), float(size), 0, 1};
    const VkRect2D scissor{{0, 0}, {size, size}};
    vkCmdSetViewport(commands, 0, 1, &viewport);
    vkCmdSetScissor(commands, 0, 1, &scissor);
    vk::VulkanGraphicsCommandContext graphicsContext;
    const auto draw = device.MakeGraphicsCommandEncoder(graphicsContext, commands);
    Require(draw.HasCompleteDispatch(), "Mesh command table is incomplete");
    draw.BindPipeline(graphics);
    draw.BindGroup(graphics, 0, graphicsGroup);
    draw.BindVertexBuffer(0, vertices);
    draw.BindVertexBuffer(1, uvBuffer);
    draw.BindIndexBuffer(indexBuffer, rhi::IndexFormat::Uint16);
    draw.DrawIndexed(6);
    vkCmdEndRenderPass(commands);
    for (const auto target : targets)
        ImageBarrier(commands, device.Resolve(target), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    rhi::TextureBufferCopyRegion readRegion;
    readRegion.width = readRegion.height = size;
    readRegion.bytesPerRow = 256;
    Require(transfer.CopyTextureToBuffer(targets[0], colorReadback, readRegion), "Color readback record failed");
    Require(transfer.CopyTextureToBuffer(targets[1], idReadback, readRegion), "Integer readback record failed");
    Require(vkEndCommandBuffer(commands) == VK_SUCCESS, "Command end failed");
    VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    Require(vkCreateFence(nativeDevice, &fence, nullptr, &fixture.fence) == VK_SUCCESS, "Fence failed");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &commands;
    Require(vkQueueSubmit(fixture.context.GetGraphicsQueue(), 1, &submit, fixture.fence) == VK_SUCCESS,
            "Submit failed");
    Require(vkWaitForFences(nativeDevice, 1, &fixture.fence, VK_TRUE, 10000000000ull) == VK_SUCCESS, "GPU timeout");
    std::array<uint8_t, size * 256> colorBytes{};
    std::array<uint8_t, size * 256> idBytes{};
    Require(device.ReadBuffer(colorReadback, 0, colorBytes.data(), colorBytes.size()), "Color readback failed");
    Require(device.ReadBuffer(idReadback, 0, idBytes.data(), idBytes.size()), "Integer readback failed");
    for (uint32_t y = 0; y < size; ++y) {
        for (uint32_t x = 0; x < size; ++x) {
            const uint8_t *encoded = texels.data() + ((size - 1 - y) / 8) * 256 + (x / 8) * 4;
            for (size_t channel = 0; channel < 3; ++channel) {
                const auto srgb = encoded[channel] / 255.0f;
                const auto linear = srgb <= 0.04045f ? srgb / 12.92f : std::pow((srgb + 0.055f) / 1.055f, 2.4f);
                const auto expected = std::lround(linear * tint[channel] * 255.0f);
                Require(std::abs(int(colorBytes[y * 256 + x * 4 + channel]) - expected) <= 1,
                        "sRGB sampling/varying/color mismatch");
            }
            uint32_t identity[2];
            std::memcpy(identity, idBytes.data() + y * 256 + x * 8, sizeof(identity));
            Require(identity[0] == objectId && identity[1] == 0xfedcba98u, "Integer MRT was changed or averaged");
        }
    }
    std::cout << "Slang indexed mesh, varying, reflected uniform/texture/sampler, UNORM-storage+sRGB-sampled view, "
                 "integer MRT, padded transfer and Compute-to-Vertex sharing passed\n";
}
} // namespace

int main()
{
    try {
        Run();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
