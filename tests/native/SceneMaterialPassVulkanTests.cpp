#include <SDL3/SDL.h>
#include <function/renderer/SceneMaterialPass.h>
#include <function/renderer/vk/VkDeviceContext.h>
#include <function/renderer/vk/VulkanRhiDevice.h>
#include <function/resources/ShaderAsset/ShaderAuthorMaterial.h>
#include <function/resources/ShaderAsset/SlangCompiler.h>

#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace infernux;

namespace
{
void Require(bool result, const char *message)
{
    if (!result)
        throw std::runtime_error(message);
}

void Barrier(VkCommandBuffer commands, VkImage image, VkImageLayout before, VkImageLayout after,
             VkPipelineStageFlags source, VkPipelineStageFlags destination, VkAccessFlags reads, VkAccessFlags writes)
{
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = before;
    barrier.newLayout = after;
    barrier.srcAccessMask = reads;
    barrier.dstAccessMask = writes;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(commands, source, destination, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void Run()
{
    Require(SDL_Init(SDL_INIT_VIDEO), "SDL initialization failed");
    auto *window = SDL_CreateWindow("Shared scene material", 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
    vk::VkDeviceContext context;
    vk::DeviceConfig configuration;
    configuration.enableValidationLayers = true;
    Require(window && context.Initialize(window, configuration), "Vulkan initialization failed");
    auto &device = context.GetRhiDevice();
    const auto native = context.GetDevice();
    rhi::SubmissionSerial serial = 1;
    device.UseSubmissionSerials([&] { return serial; });

    ShaderAuthorSource source;
    source.source = {"scene_material", "<scene material>", R"(
struct Parameters { float4 signal; };
[[vk::binding(0, 0)]] ConstantBuffer<Parameters> inx_parameters;
[shader("vertex")] float4 vertex_main(uint id : SV_VertexID) : SV_Position {
    return float4(float2((id << 1) & 2, id & 2) * 2.0 - 1.0, 0.5, 1.0);
}
[shader("fragment")] float4 fragment_main() : SV_Target0 { return inx_parameters.signal; }
)"};
    source.schema = R"({"domain":"mesh", "inputs":[], "parameters":[
        {"name":"signal","kind":"vec4","default":[0,0,0,1],"hdr":false,"range":null,"stages":["fragment"]}]})";
    const auto compiled = std::make_shared<const ShaderCompilation>(SlangCompiler::Get().CompileAuthorProgram(source));
    auto artifact = std::make_shared<const ShaderProgramArtifact>(
        CreateShaderAuthorMaterial("test/scene", compiled, source.schema, "infernux/standard"));
    InxMaterial material;
    material.SynchronizeShaderPropertyDefaults(*artifact);
    RenderState state;
    state.cullMode = MaterialCullMode::None;
    state.depthTestEnable = state.depthWriteEnable = false;
    MaterialPassPipelineDescriptor contract;
    contract.colorFormats = {rhi::PixelFormat::RGBA16SFloat};
    constexpr uint32_t size = 8, count = 3;
    const std::array<glm::vec4, count> signals{{{.25f, .5f, .75f, 1}, {.75f, .25f, .5f, 1}, {.5f, .75f, .25f, 1}}};
    const std::array<std::array<uint16_t, 4>, count> expected{
        {{0x3400, 0x3800, 0x3a00, 0x3c00}, {0x3a00, 0x3400, 0x3800, 0x3c00}, {0x3800, 0x3a00, 0x3400, 0x3c00}}};
    std::array<rhi::TextureHandle, count> images;
    std::array<rhi::TextureViewHandle, count> views;
    std::array<rhi::BufferHandle, count> readbacks;
    VkCommandPool pool;
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = context.GetQueueIndices().graphicsFamily.value();
    Require(vkCreateCommandPool(native, &poolInfo, nullptr, &pool) == VK_SUCCESS, "Command pool failed");
    std::array<VkCommandBuffer, count> commandBuffers;
    VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation.commandPool = pool;
    allocation.commandBufferCount = count;
    allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    Require(vkAllocateCommandBuffers(native, &allocation, commandBuffers.data()) == VK_SUCCESS, "Commands failed");
    {
        SceneMaterialPass pass(device, artifact, state, contract);
        for (uint32_t frame = 0; frame < count; ++frame) {
            serial = frame + 1;
            material.SetVector4("signal", signals[frame]);
            auto values = material.CaptureRenderPublication();
            // CaptureRenderPublication deliberately has its own property version.
            // Successive immutable publications must still update the GPU value.
            Require(pass.Prepare(
                        values, {},
                        [](const auto &, auto &) -> bool { throw std::runtime_error("Unexpected external resource"); }),
                    "Material preparation failed");
            rhi::TextureDesc image;
            image.width = image.height = size;
            image.format = contract.colorFormats.front();
            image.usage = rhi::TextureUsageFlags::ColorAttachment | rhi::TextureUsageFlags::TransferSource;
            images[frame] = device.CreateTexture(image);
            rhi::TextureViewDesc view;
            view.texture = images[frame];
            view.format = image.format;
            views[frame] = device.CreateTextureView(view);
            rhi::BufferDesc readback;
            readback.byteSize = size * 256;
            readback.memory = rhi::BufferMemory::Readback;
            readback.usage = rhi::BufferUsageFlags::TransferDestination;
            readbacks[frame] = device.CreateBuffer(readback);
            const auto commands = commandBuffers[frame];
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            Require(vkBeginCommandBuffer(commands, &begin) == VK_SUCCESS, "Begin failed");
            Barrier(commands, device.Resolve(images[frame]), VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
            rhi::RasterPassDescription raster;
            raster.width = raster.height = size;
            raster.colorCount = 1;
            raster.colors[0].view = views[frame];
            raster.colors[0].format = image.format;
            raster.colors[0].load = rhi::AttachmentLoad::Clear;
            vk::VulkanRasterPass prepared;
            device.PrepareRasterPass(raster, prepared);
            vkCmdBeginRendering(commands, &prepared.info);
            const VkViewport viewport{0, 0, float(size), float(size), 0, 1};
            const VkRect2D scissor{{0, 0}, {size, size}};
            vkCmdSetViewport(commands, 0, 1, &viewport);
            vkCmdSetScissor(commands, 0, 1, &scissor);
            vk::VulkanGraphicsCommandContext graphics;
            const auto draw = device.MakeGraphicsCommandEncoder(graphics, commands);
            pass.Bind(draw);
            draw.Draw(3);
            vkCmdEndRendering(commands);
            Barrier(commands, device.Resolve(images[frame]), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            vk::VulkanTransferCommandContext transfer;
            rhi::TextureBufferCopyRegion region;
            region.width = region.height = size;
            region.bytesPerRow = 256;
            Require(device.MakeTransferCommandEncoder(transfer, commands)
                        .CopyTextureToBuffer(images[frame], readbacks[frame], region),
                    "Copy failed");
            Require(vkEndCommandBuffer(commands) == VK_SUCCESS, "End failed");
        }
        // All updates happen before any GPU work, so overwriting a shared
        // uniform allocation makes every image incorrectly use the final value.
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = count;
        submit.pCommandBuffers = commandBuffers.data();
        Require(vkQueueSubmit(context.GetGraphicsQueue(), 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS, "Submit failed");
        Require(vkQueueWaitIdle(context.GetGraphicsQueue()) == VK_SUCCESS, "Completion failed");
        for (uint32_t frame = 0; frame < count; ++frame) {
            std::array<std::byte, size * 256> pixels;
            Require(device.ReadBuffer(readbacks[frame], 0, pixels.data(), pixels.size()), "Readback failed");
            for (uint32_t y = 0; y < size; ++y)
                for (uint32_t x = 0; x < size; ++x) {
                    std::array<uint16_t, 4> pixel;
                    std::memcpy(pixel.data(), pixels.data() + y * 256 + x * 8, 8);
                    Require(pixel == expected[frame], "Queued material publication changed its recorded value");
                }
        }
    }
    for (uint32_t frame = 0; frame < count; ++frame) {
        device.Release(readbacks[frame]);
        device.Release(views[frame]);
        device.Release(images[frame]);
    }
    device.CollectDescriptorRetirements(serial);
    device.CollectResourceRetirements(serial);
    vkDestroyCommandPool(native, pool, nullptr);
    context.Destroy();
    SDL_DestroyWindow(window);
    SDL_Quit();
    std::cout << "Shared scene material: three queued Vulkan frames preserve their exact values\n";
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
