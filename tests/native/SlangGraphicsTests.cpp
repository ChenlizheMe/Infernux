#include <function/renderer/shader/DeviceShaderProgram.h>
#include <function/renderer/shader/ShaderValueCopy.h>
#include <function/renderer/vk/RhiVulkanTypes.h>
#include <function/renderer/vk/VkDeviceContext.h>
#include <function/renderer/vk/VulkanRhiDevice.h>
#include <function/resources/ShaderAsset/SlangCompiler.h>

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
struct Material {
    float4 tint; uint objectId;
    float3x3 channelTransform;
    float weights[2];
    Texture2D<float4> albedo;
    SamplerState albedoSampler;
    Texture2D<float4> generatedImage;
};
struct Scene { float exposure; ParameterBlock<Material> material; };
ParameterBlock<Scene> scene;
RWStructuredBuffer<float4> positions;
[format("rgba8")] WTexture2D<float4> generated;

[Differentiable]
float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

[shader("compute")]
[numthreads(4, 1, 1)]
void prepare_vertices(uint3 id : SV_DispatchThreadID)
{
    positions[id.x] = float4(float(id.x & 1) * 2.0 - 1.0, float(id.x >> 1) * 2.0 - 1.0, 0.0, 1.0);
    DifferentialPair<float> value = diffPair(float(id.x) - 1.5, 0.0);
    bwd_diff(sigmoid)(value, 1.0);
    generated.Store(int2(id.x & 1, id.x >> 1), float4(value.d, sigmoid(value.p), 1.0 - value.d, 1.0));
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
    result.color = scene.material.albedo.Sample(scene.material.albedoSampler, input.uv) * scene.material.tint
        * scene.material.generatedImage.Sample(scene.material.albedoSampler, input.uv) * scene.exposure;
    result.color.rgb = mul(scene.material.channelTransform, result.color.rgb) * scene.material.weights[1];
    result.identity = uint2(scene.material.objectId, 0xfedcba98u);
    return result;
}
)";

struct Fixture
{
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
    vk::DeviceConfig config;
    config.enableValidationLayers = true;
    config.presentation = false;
    Require(fixture.context.InitializeInstance(config) && fixture.context.InitializeDevice(VK_NULL_HANDLE, config),
            "Vulkan initialization failed");
    auto &device = fixture.context.GetRhiDevice();
    const auto nativeDevice = fixture.context.GetDevice();
    const auto compiled = SlangCompiler::Get().Compile({"mesh", "mesh.compute.py", kProgram}, {},
                                                       {ShaderCodeTarget::SpirV, ShaderCodeTarget::Wgsl});
    const auto &program = compiled.targets[0];
    DeviceShaderProgram computeProgram(device, program, {0});
    DeviceShaderProgram graphicsProgram(device, program, {1, 2});
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

    const auto &uniformLayout =
        std::find_if(program.resources.begin(), program.resources.end(), [](const auto &resource) {
            return resource.name == "scene.material";
        })->value;
    const auto &fields = uniformLayout.fields;
    std::vector<char> constants(uniformLayout.byteSize);
    const std::array<float, 4> tint{0.5f, 0.75f, 1.0f, 1.0f};
    const std::array<float, 9> channelTransform{1.0f, 0.2f, 0.0f, 0.0f, 0.9f, 0.1f, 0.1f, 0.0f, 0.8f};
    const std::array<float, 2> weights{0.25f, 0.8f};
    const uint32_t objectId = 0x1020304;
    ShaderValueCopy(fields[0]).Write(constants.data(), tint.data());
    ShaderValueCopy(fields[1]).Write(constants.data(), &objectId);
    ShaderValueCopy(fields[2]).Write(constants.data(), channelTransform.data());
    ShaderValueCopy(fields[3]).Write(constants.data(), weights.data());
    const auto uniform =
        buffer(constants.size(), rhi::BufferUsageFlags::Uniform, rhi::BufferMemory::Upload, constants.data());
    const float exposure = 1.0f;
    const auto sceneUniform = buffer(16, rhi::BufferUsageFlags::Uniform, rhi::BufferMemory::Upload);
    Require(device.WriteBuffer(sceneUniform, 0, &exposure, sizeof(exposure)), "Scene uniform write failed");

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
    rhi::TextureDesc generatedDesc;
    generatedDesc.width = generatedDesc.height = 2;
    generatedDesc.format = rhi::PixelFormat::RGBA8UNorm;
    generatedDesc.usage = rhi::TextureUsageFlags::Storage | rhi::TextureUsageFlags::Sampled;
    const auto generated = device.CreateTexture(generatedDesc);
    rhi::TextureViewDesc generatedViewDesc;
    generatedViewDesc.texture = generated;
    generatedViewDesc.usage = rhi::TextureUsageFlags::Storage;
    const auto generatedStorage = device.CreateTextureView(generatedViewDesc);
    generatedViewDesc.usage = rhi::TextureUsageFlags::Sampled;
    const auto generatedSampled = device.CreateTextureView(generatedViewDesc);
    Require(generatedStorage.IsValid() && generatedSampled.IsValid(), "Compute-to-sampled image views failed");
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
    auto bindingsFor = [&](const DeviceShaderProgram &selected) {
        std::vector<rhi::BindGroupHandle> groups;
        for (uint32_t set = 0; set < selected.GetBindingLayouts().size(); ++set) {
            rhi::BindGroupDesc groupDesc;
            for (const auto &resource : selected.GetResources()) {
                if (resource.set != set)
                    continue;
                const auto type = resource.binding.type;
                const uint32_t binding = resource.binding.binding;
                if (type == rhi::BindingType::UniformBuffer || type == rhi::BindingType::StorageBuffer) {
                    const auto value = resource.name == "scene.material" ? uniform
                                       : resource.name == "scene"        ? sceneUniform
                                                                         : vertices;
                    groupDesc.buffers[groupDesc.bufferCount++] = {binding, type, value};
                } else {
                    const auto image = resource.name == "generated"                       ? generatedStorage
                                       : resource.name == "scene.material.generatedImage" ? generatedSampled
                                                                                          : sourceView;
                    groupDesc.textures[groupDesc.textureCount++] = {
                        binding, type, type != rhi::BindingType::Sampler ? image : rhi::TextureViewHandle{},
                        type == rhi::BindingType::Sampler ? sampler : rhi::SamplerHandle{}};
                }
            }
            groupDesc.layout = selected.GetBindingLayouts()[set];
            const auto group = device.CreateBindGroup(groupDesc);
            Require(groupDesc.layout.IsValid() && group.IsValid(), "Reflected RHI bindings failed");
            groups.push_back(group);
        }
        return groups;
    };
    const auto computeGroup = bindingsFor(computeProgram);
    const auto graphicsGroup = bindingsFor(graphicsProgram);
    rhi::ComputePipelineDesc computeDesc;
    computeProgram.Apply(computeDesc);
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
    graphicsProgram.Apply(graphicsDesc);
    graphicsDesc.renderTargetLayout = device.RegisterRenderTargetLayout(fixture.renderPass);
    graphicsDesc.raster.cullMode = rhi::CullMode::None;
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
    ImageBarrier(commands, device.Resolve(generated), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                 VK_ACCESS_SHADER_WRITE_BIT);
    computeCommands.BindPipeline(compute);
    for (uint32_t set = 0; set < computeGroup.size(); ++set)
        computeCommands.BindGroup(compute, set, computeGroup[set]);
    computeCommands.Dispatch(1);
    ImageBarrier(commands, device.Resolve(generated), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                 VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
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
    for (uint32_t set = 0; set < graphicsGroup.size(); ++set)
        draw.BindGroup(graphics, set, graphicsGroup[set]);
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
            std::array<float, 3> input{};
            for (size_t channel = 0; channel < 3; ++channel) {
                const auto srgb = encoded[channel] / 255.0f;
                const auto linear = srgb <= 0.04045f ? srgb / 12.92f : std::pow((srgb + 0.055f) / 1.055f, 2.4f);
                const auto texel = float((x / 8) + ((size - 1 - y) / 8) * 2);
                const auto sigmoid = 1.0f / (1.0f + std::exp(-(texel - 1.5f)));
                const auto derivative = sigmoid * (1.0f - sigmoid);
                const std::array<float, 3> generatedValue{std::round(derivative * 255.0f) / 255.0f,
                                                          std::round(sigmoid * 255.0f) / 255.0f,
                                                          std::round((1.0f - derivative) * 255.0f) / 255.0f};
                input[channel] = linear * tint[channel] * generatedValue[channel];
            }
            for (size_t channel = 0; channel < 3; ++channel) {
                float value = 0;
                for (size_t source = 0; source < 3; ++source)
                    value += channelTransform[channel * 3 + source] * input[source];
                const auto expected = std::lround(value * weights[1] * 255.0f);
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
