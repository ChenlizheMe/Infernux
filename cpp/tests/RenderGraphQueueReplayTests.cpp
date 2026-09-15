#include <SDL3/SDL.h>
#include <function/renderer/vk/RenderGraph.h>
#include <function/renderer/vk/VkDeviceContext.h>
#include <function/renderer/vk/VulkanFrameSubmission.h>
#include <function/renderer/vk/VulkanQueueManager.h>
#include <function/renderer/vk/VulkanRhiDevice.h>
#include <function/renderer/vk/VulkanSubmissionExecutor.h>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <array>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>

using namespace infernux;

enum class ResourceKind
{
    Color,
    Depth,
    Buffer
};

static void CheckReplay(vk::VkDeviceContext &context, vk::VulkanQueueManager &queues,
                        vk::VulkanSubmissionExecutor &executor, VkFence fence, ResourceKind kind,
                        rhi::QueueRole producer, rhi::QueueRole consumer, rhi::BindingLayoutHandle layout,
                        rhi::ComputePipelineHandle pipeline, rhi::SamplerHandle sampler)
{
    auto &device = context.GetRhiDevice();
    rhi::BufferDesc desc;
    desc.byteSize = 8 * 4 * 4 * (kind == ResourceKind::Depth ? 4 : 1);
    desc.memory = rhi::BufferMemory::Readback;
    desc.usage =
        kind == ResourceKind::Depth ? rhi::BufferUsageFlags::Storage : rhi::BufferUsageFlags::TransferDestination;
    desc.queueAccess = consumer == rhi::QueueRole::Compute    ? rhi::QueueAccessFlags::Compute
                       : consumer == rhi::QueueRole::Transfer ? rhi::QueueAccessFlags::Transfer
                                                              : rhi::QueueAccessFlags::Graphics;
    const auto readback = device.CreateBuffer(desc);
    assert(readback.IsValid());
    rhi::BufferHandle upload;
    if (kind == ResourceKind::Color && producer == rhi::QueueRole::Transfer) {
        auto uploadDesc = desc;
        uploadDesc.memory = rhi::BufferMemory::Upload;
        uploadDesc.usage = rhi::BufferUsageFlags::TransferSource;
        uploadDesc.queueAccess = rhi::QueueAccessFlags::Transfer;
        upload = device.CreateBuffer(uploadDesc);
        assert(upload.IsValid());
    }
    {
        vk::RenderGraph graph;
        graph.Initialize(&context, nullptr, &queues);
        auto image =
            kind == ResourceKind::Buffer
                ? graph.RegisterTransientBuffer("ReplayBuffer", desc.byteSize,
                                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)
                : graph.RegisterTransientTexture("ReplayImage", 8, 4,
                                                 kind == ResourceKind::Depth ? VK_FORMAT_D32_SFLOAT
                                                                             : VK_FORMAT_R8G8B8A8_UNORM);
        uint32_t frame = 0;
        rhi::BindGroupHandle group;
        graph.AddTransferPass("Produce", [&](vk::PassBuilder &builder) {
            builder.SetQueueRole(producer);
            if (upload.IsValid())
                builder.TransferRead(builder.ImportBuffer("ReplayUpload", upload, desc.byteSize));
            image = builder.TransferWrite(image);
            return [&](vk::RenderContext &commands) {
                if (kind == ResourceKind::Buffer) {
                    vkCmdFillBuffer(commands.GetCommandBuffer(), commands.GetBuffer(image), 0, desc.byteSize,
                                    0x01020300u + frame);
                    return;
                }
                if (kind == ResourceKind::Depth) {
                    VkClearDepthStencilValue depth{(frame + 1) / 8.0f, 0};
                    VkImageSubresourceRange range{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
                    vkCmdClearDepthStencilImage(commands.GetCommandBuffer(),
                                                device.Resolve(commands.GetTextureHandle(image)),
                                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &depth, 1, &range);
                    return;
                }
                if (upload.IsValid()) {
                    VkBufferImageCopy copy{};
                    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    copy.imageExtent = {8, 4, 1};
                    vkCmdCopyBufferToImage(commands.GetCommandBuffer(), device.Resolve(upload),
                                           device.Resolve(commands.GetTextureHandle(image)),
                                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
                    return;
                }
                VkClearColorValue color{};
                color.float32[frame % 3] = 1.0f;
                color.float32[3] = 1.0f;
                VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdClearColorImage(commands.GetCommandBuffer(), device.Resolve(commands.GetTextureHandle(image)),
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);
            };
        });
        if (kind == ResourceKind::Depth) {
            // Depth clear/copy requires Graphics without optional maintenance
            // features. Sample it with a real compute shader to exercise the
            // Graphics -> Compute -> next-frame Graphics ownership cycle.
            assert(producer == rhi::QueueRole::Graphics && consumer != rhi::QueueRole::Transfer);
            graph.AddComputePass("SampleDepth", [&](vk::PassBuilder &builder) {
                builder.SetQueueRole(consumer);
                builder.Read(image, rhi::PipelineStage::ComputeShader);
                builder.WriteStorageBuffer(builder.ImportBuffer("DepthReadback", readback, desc.byteSize));
                builder.SetSideEffect();
                return [&](vk::RenderContext &commands) {
                    auto &encoder = commands.GetComputeCommandEncoder();
                    encoder.BindPipeline(pipeline);
                    encoder.BindGroup(pipeline, 0, group);
                    encoder.Dispatch(1, 1, 1);
                    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
                    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    barrier.buffer = device.Resolve(readback);
                    barrier.size = desc.byteSize;
                    vkCmdPipelineBarrier(commands.GetCommandBuffer(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
                };
            });
        } else
            graph.AddTransferPass("Consume", [&](vk::PassBuilder &builder) {
                builder.SetQueueRole(consumer);
                builder.TransferRead(image);
                builder.TransferWrite(builder.ImportBuffer("ReplayReadback", readback, desc.byteSize));
                builder.SetSideEffect();
                return [&](vk::RenderContext &commands) {
                    if (kind == ResourceKind::Buffer) {
                        VkBufferCopy copy{0, 0, desc.byteSize};
                        vkCmdCopyBuffer(commands.GetCommandBuffer(), commands.GetBuffer(image),
                                        device.Resolve(readback), 1, &copy);
                    } else {
                        VkBufferImageCopy copy{};
                        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                        copy.imageExtent = {8, 4, 1};
                        vkCmdCopyImageToBuffer(
                            commands.GetCommandBuffer(), device.Resolve(commands.GetTextureHandle(image)),
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, device.Resolve(readback), 1, &copy);
                    }
                    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
                    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    barrier.buffer = device.Resolve(readback);
                    barrier.size = desc.byteSize;
                    vkCmdPipelineBarrier(commands.GetCommandBuffer(), VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
                };
            });
        assert(graph.Compile());
        if (kind == ResourceKind::Depth) {
            rhi::BindGroupDesc groupDesc;
            groupDesc.layout = layout;
            groupDesc.textures[0] = {0, rhi::BindingType::CombinedTextureSampler, graph.ResolveRhiTextureView(image),
                                     sampler};
            groupDesc.textureCount = 1;
            groupDesc.buffers[0] = {1, rhi::BindingType::StorageBuffer, readback, 0, desc.byteSize};
            groupDesc.bufferCount = 1;
            group = device.CreateBindGroup(groupDesc);
            assert(group.IsValid());
        }
        assert(graph.GetSubmissionPlan().batches.front().queue == producer);
        assert(graph.GetSubmissionPlan().batches.back().queue == consumer);
        const bool crossFamily = queues.GetSnapshot(producer).family != queues.GetSnapshot(consumer).family;
        assert(!graph.HasExternalQueueOwnershipReleases(consumer));
        for (; frame < 6; ++frame) {
            if (upload.IsValid()) {
                std::array<uint32_t, 8 * 4> source;
                source.fill(0xff000000u | (0xffu << (8 * (frame % 3))));
                assert(device.WriteBuffer(upload, 0, source.data(), sizeof(source)));
            }
            if (frame == 2) {
                vk::RenderGraph moved(std::move(graph));
                assert(!graph.HasExternalQueueOwnershipReleases(consumer));
                assert(moved.HasExternalQueueOwnershipReleases(consumer) == crossFamily);
                graph = std::move(moved);
                assert(!moved.HasExternalQueueOwnershipReleases(consumer));
            }
            if (frame == 3)
                assert(graph.Compile()); // Recompile without replacing the physical allocation.
            const bool pendingRelease = graph.HasExternalQueueOwnershipReleases(consumer);
            if (pendingRelease != (crossFamily && frame != 0))
                std::cerr << "Release mismatch kind=" << static_cast<int>(kind)
                          << " producer=" << static_cast<int>(producer) << " consumer=" << static_cast<int>(consumer)
                          << " frame=" << frame << " transfers=" << graph.GetQueueOwnershipTransfers().size()
                          << " pending=" << pendingRelease << std::endl;
            assert(pendingRelease == (crossFamily && frame != 0));
            vk::VulkanFrameSubmission submission;
            submission.Reset();
            std::vector<uint32_t> releases;
            for (auto role : {rhi::QueueRole::Graphics, rhi::QueueRole::Compute, rhi::QueueRole::Transfer}) {
                if (graph.HasExternalQueueOwnershipReleases(role))
                    releases.push_back(submission.AddWork(
                        context.GetDeviceId(), role, rhi::SubmissionDomain::Frame, rhi::InvalidRenderViewId,
                        rhi::PipelineStage::AllCommands, {},
                        [&, role](VkCommandBuffer commands) {
                            return graph.RecordExternalQueueOwnershipReleases(role, commands);
                        },
                        "ReplayRelease"));
            }
            assert(!submission.AppendRenderGraph(graph, releases).Empty());
            rhi::SubmissionPlan plan;
            std::string error;
            assert(submission.Build(plan, error));
            assert(vkResetFences(context.GetDevice(), 1, &fence) == VK_SUCCESS);
            vk::VulkanSubmissionExecutor::ExternalSync sync{};
            sync.completionFence = fence;
            sync.completionEpoch = queues.ReserveCompletionEpoch();
            const uint32_t slot = frame % 2;
            const auto result = executor.Execute(
                slot, plan,
                [&](uint32_t batch, VkCommandBuffer commands) { return submission.RecordBatch(plan, batch, commands); },
                sync);
            assert(result.Succeeded());
            assert(vkWaitForFences(context.GetDevice(), 1, &fence, VK_TRUE, 5'000'000'000ull) == VK_SUCCESS);
            executor.CompleteFrame(slot);
            queues.CompleteCompletionEpoch(sync.completionEpoch);
            std::vector<uint32_t> pixels(desc.byteSize / sizeof(uint32_t));
            assert(device.ReadBuffer(readback, 0, pixels.data(), desc.byteSize));
            uint32_t expected = 0x01020300u + frame;
            if (kind == ResourceKind::Depth) {
                const float depth = (frame + 1) / 8.0f;
                std::memcpy(&expected, &depth, sizeof(expected));
            } else if (kind == ResourceKind::Color) {
                std::array<uint8_t, 4> color{};
                color[frame % 3] = color[3] = 255;
                std::memcpy(&expected, color.data(), sizeof(expected));
            }
            for (size_t i = 0; i < pixels.size(); ++i) {
                uint32_t channelExpected = expected;
                if (kind == ResourceKind::Depth && i % 4 != 0)
                    channelExpected = i % 4 == 3 ? 0x3f800000u : 0;
                const auto pixel = pixels[i];
                if (pixel != channelExpected)
                    std::cerr << "Readback mismatch kind=" << static_cast<int>(kind)
                              << " producer=" << static_cast<int>(producer)
                              << " consumer=" << static_cast<int>(consumer) << " frame=" << frame << " actual=" << pixel
                              << " expected=" << channelExpected << std::endl;
                assert(pixel == channelExpected);
            }
        }
        graph.Reset();
        assert(!graph.HasExternalQueueOwnershipReleases(consumer));
        if (group.IsValid())
            device.Release(group);
    }
    device.Release(readback);
    if (upload.IsValid())
        device.Release(upload);
}

int main(int argc, char **argv)
{
    assert(argc == 2);
    std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
    assert(input);
    const auto bytes = static_cast<size_t>(input.tellg());
    assert(bytes > 0 && bytes % sizeof(uint32_t) == 0);
    std::vector<uint32_t> code(bytes / sizeof(uint32_t));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(code.data()), bytes);
    assert(input);
    assert(SDL_Init(SDL_INIT_VIDEO));
    auto *window = SDL_CreateWindow("RenderGraph queue replay", 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
    assert(window);
    vk::VkDeviceContext context;
    vk::DeviceConfig config;
    config.enableValidationLayers = true;
    assert(context.Initialize(window, config));
    auto &device = context.GetRhiDevice();
    const auto shader = device.CreateShaderModule(rhi::ShaderModuleDesc::FromSpirV(code.data(), code.size()));
    rhi::BindingLayoutDesc layoutDesc;
    layoutDesc.entries[0] = {0, rhi::BindingType::CombinedTextureSampler, rhi::ShaderStage::Compute, 1};
    layoutDesc.entries[1] = {1, rhi::BindingType::StorageBuffer, rhi::ShaderStage::Compute, 1};
    layoutDesc.entryCount = 2;
    const auto layout = device.CreateBindingLayout(layoutDesc);
    rhi::ComputePipelineDesc pipelineDesc;
    pipelineDesc.computeShader = shader;
    pipelineDesc.bindingLayouts[0] = layout;
    pipelineDesc.bindingLayoutCount = 1;
    const auto pipeline = device.CreateComputePipeline(pipelineDesc);
    rhi::SamplerDesc samplerDesc;
    samplerDesc.minFilter = samplerDesc.magFilter = samplerDesc.mipFilter = rhi::FilterMode::Nearest;
    const auto sampler = device.CreateSampler(samplerDesc);
    assert(shader.IsValid() && layout.IsValid() && pipeline.IsValid() && sampler.IsValid());
    vk::VulkanQueueManager queues;
    assert(queues.Initialize(context, 2));
    vk::VulkanSubmissionExecutor executor;
    assert(executor.Initialize(context, queues, 2));
    std::cout << "Replay queue families: graphics=" << queues.GetSnapshot(rhi::QueueRole::Graphics).family
              << " compute=" << queues.GetSnapshot(rhi::QueueRole::Compute).family
              << " transfer=" << queues.GetSnapshot(rhi::QueueRole::Transfer).family << std::endl;
    VkFenceCreateInfo info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence{};
    assert(vkCreateFence(context.GetDevice(), &info, nullptr, &fence) == VK_SUCCESS);
    for (auto kind : {ResourceKind::Color, ResourceKind::Buffer}) {
        for (auto producer : {rhi::QueueRole::Graphics, rhi::QueueRole::Compute, rhi::QueueRole::Transfer}) {
            for (auto consumer : {rhi::QueueRole::Graphics, rhi::QueueRole::Compute, rhi::QueueRole::Transfer})
                CheckReplay(context, queues, executor, fence, kind, producer, consumer, layout, pipeline, sampler);
        }
    }
    for (auto consumer : {rhi::QueueRole::Graphics, rhi::QueueRole::Compute})
        CheckReplay(context, queues, executor, fence, ResourceKind::Depth, rhi::QueueRole::Graphics, consumer, layout,
                    pipeline, sampler);
    std::cout << "Replay verified: color/buffer x 9 queue pairs + depth x 2 sampling queues; 120 executions"
              << std::endl;
    vkDestroyFence(context.GetDevice(), fence, nullptr);
    executor.Destroy();
    queues.Destroy();
    device.Release(sampler);
    device.Release(pipeline);
    device.Release(layout);
    device.Release(shader);
    context.Destroy();
    SDL_DestroyWindow(window);
    SDL_Quit();
}
