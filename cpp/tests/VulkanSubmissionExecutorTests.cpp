#include <function/renderer/rhi/RenderSubmissionPlan.h>
#include <function/renderer/vk/DescriptorBindTrace.h>
#include <function/renderer/vk/GpuTimestampQueries.h>
#include <function/renderer/vk/VkDeviceContext.h>
#include <function/renderer/vk/VkTypes.h>
#include <function/renderer/vk/VulkanQueueManager.h>
#include <function/renderer/vk/VulkanRhiDevice.h>
#include <function/renderer/vk/VulkanSubmissionExecutor.h>

#include <SDL3/SDL.h>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <array>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernux;

int main()
{
    assert(SDL_Init(SDL_INIT_VIDEO));
    SDL_Window *window =
        SDL_CreateWindow("Infernux Vulkan Submission Executor Test", 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
    assert(window != nullptr);

    vk::VkDeviceContext context;
    vk::DeviceConfig config;
    config.appName = "Infernux Vulkan Submission Executor Test";
    config.enableValidationLayers = true;
    assert(context.Initialize(window, config));

    vk::VulkanQueueManager queues;
    assert(queues.Initialize(context, 2));
    assert(queues.GetLastReservedCompletionEpoch() == 0);
    assert(queues.GetCompletedCompletionEpoch() == 0);
    assert(queues.ResetGraphicsFrameFence(0));
    const auto abandonedEpoch = queues.GetFrameCompletionEpoch(0);
    assert(abandonedEpoch != rhi::InvalidSubmissionSerial);
    assert(queues.GetLastReservedCompletionEpoch() == abandonedEpoch);
    assert(!queues.ResetGraphicsFrameFence(0));
    assert(queues.AbandonGraphicsFrameSlot(0));
    assert(queues.GetCompletedCompletionEpoch() == abandonedEpoch);
    assert(queues.GetFrameCompletionEpoch(0) == rhi::InvalidSubmissionSerial);
    const auto earlierEpoch = queues.ReserveCompletionEpoch();
    const auto laterEpoch = queues.ReserveCompletionEpoch();
    assert(!queues.IsCompletionEpochComplete(earlierEpoch));
    assert(!queues.IsCompletionEpochComplete(laterEpoch));
    queues.CompleteCompletionEpoch(laterEpoch);
    assert(queues.GetCompletedCompletionEpoch() == abandonedEpoch);
    assert(!queues.IsCompletionEpochComplete(earlierEpoch));
    assert(queues.IsCompletionEpochComplete(laterEpoch));
    queues.CompleteCompletionEpoch(earlierEpoch);
    assert(queues.GetCompletedCompletionEpoch() == laterEpoch);
    assert(queues.IsCompletionEpochComplete(earlierEpoch));
    const auto graphicsLane = queues.GetSnapshot(rhi::QueueRole::Graphics).nativeLane;
    const auto computeLane = queues.GetSnapshot(rhi::QueueRole::Compute).nativeLane;
    const auto transferLane = queues.GetSnapshot(rhi::QueueRole::Transfer).nativeLane;
    assert((graphicsLane != computeLane) == context.HasIndependentComputeQueue());

    vk::VulkanSubmissionExecutor executor;
    assert(executor.Initialize(context, queues, 2));
    assert((executor.GetDependencyTimeline(rhi::QueueRole::Graphics) !=
            executor.GetDependencyTimeline(rhi::QueueRole::Compute)) == context.HasIndependentComputeQueue());
    if (transferLane != graphicsLane) {
        assert(executor.GetDependencyTimeline(rhi::QueueRole::Transfer) !=
               executor.GetDependencyTimeline(rhi::QueueRole::Graphics));
    }

    std::vector<rhi::SubmissionWorkItem> work(3);
    work[0] = {0,
               context.GetDeviceId(),
               rhi::QueueRole::Graphics,
               rhi::SubmissionDomain::Frame,
               rhi::InvalidRenderViewId,
               rhi::PipelineStage::AllCommands,
               {}};
    work[1] = {1,
               context.GetDeviceId(),
               rhi::QueueRole::Compute,
               rhi::SubmissionDomain::Frame,
               rhi::InvalidRenderViewId,
               rhi::PipelineStage::ComputeShader,
               {0}};
    work[2] = {2,
               context.GetDeviceId(),
               rhi::QueueRole::Graphics,
               rhi::SubmissionDomain::Frame,
               rhi::InvalidRenderViewId,
               rhi::PipelineStage::DrawIndirect,
               {1}};

    rhi::SubmissionPlan plan;
    std::string error;
    assert(rhi::BuildSubmissionPlan(work, plan, error));
    assert(plan.batches.size() == 3);

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence completionFence = VK_NULL_HANDLE;
    assert(vkCreateFence(context.GetDevice(), &fenceInfo, nullptr, &completionFence) == VK_SUCCESS);

    vk::VulkanSubmissionExecutor::ExternalSync sync{};
    assert(queues.ResetGraphicsFrameFence(0));
    sync.completionFence = queues.GetGraphicsFrameFence(0);
    sync.completionEpoch = queues.GetFrameCompletionEpoch(0);
    uint32_t recorded = 0;
    const auto result = executor.Execute(
        0, plan,
        [&](uint32_t batchIndex, VkCommandBuffer commandBuffer) {
            assert(batchIndex == recorded);
            assert(commandBuffer != VK_NULL_HANDLE);
            assert(vkdebug::GetDescriptorRecordingSubmissionSerial() == sync.completionEpoch);
            ++recorded;
            return true;
        },
        sync);
    assert(result.Succeeded());
    assert(vkdebug::GetDescriptorRecordingSubmissionSerial() == rhi::InvalidSubmissionSerial);
    assert(queues.AssociateFrameSlot(0, result.completionTicket));
    assert(recorded == plan.batches.size());
    assert(queues.WaitForGraphicsFrameSlot(0));

    executor.CompleteFrame(0);
    assert(queues.CompleteFrameSlot(0).IsValid());
    assert(queues.GetCompletedCompletionEpoch() == sync.completionEpoch);
    assert(queues.GetCompletedSerial(rhi::QueueRole::Graphics) == result.completionTicket.serial);
    assert(queues.GetCompletedSerial(rhi::QueueRole::Compute) > 0);

    sync.completionFence = completionFence;

    if (transferLane != graphicsLane) {
        assert(vkResetFences(context.GetDevice(), 1, &completionFence) == VK_SUCCESS);
        sync.completionEpoch = queues.ReserveCompletionEpoch();
        std::vector<rhi::SubmissionWorkItem> independentWork(2);
        independentWork[0] = {10,
                              context.GetDeviceId(),
                              rhi::QueueRole::Transfer,
                              rhi::SubmissionDomain::Frame,
                              rhi::InvalidRenderViewId,
                              rhi::PipelineStage::Transfer,
                              {}};
        independentWork[1] = {11,
                              context.GetDeviceId(),
                              rhi::QueueRole::Graphics,
                              rhi::SubmissionDomain::Frame,
                              rhi::InvalidRenderViewId,
                              rhi::PipelineStage::AllGraphics,
                              {}};
        rhi::SubmissionPlan independentPlan;
        assert(rhi::BuildSubmissionPlan(independentWork, independentPlan, error));
        const auto joined = executor.Execute(
            1, independentPlan, [](uint32_t, VkCommandBuffer commandBuffer) { return commandBuffer != VK_NULL_HANDLE; },
            sync);
        assert(joined.Succeeded());
        assert(vkWaitForFences(context.GetDevice(), 1, &completionFence, VK_TRUE, 5'000'000'000ull) == VK_SUCCESS);
        executor.CompleteFrame(1);
        queues.CompleteCompletionEpoch(sync.completionEpoch);
        assert(queues.GetCompletedSerial(rhi::QueueRole::Transfer) > 0);
    }

    if (computeLane != graphicsLane) {
        assert(vkResetFences(context.GetDevice(), 1, &completionFence) == VK_SUCCESS);
        sync.completionEpoch = queues.ReserveCompletionEpoch();
        const std::vector<rhi::SubmissionWorkItem> overlappedWork = {
            {20,
             context.GetDeviceId(),
             rhi::QueueRole::Compute,
             rhi::SubmissionDomain::Frame,
             rhi::InvalidRenderViewId,
             rhi::PipelineStage::ComputeShader,
             {}},
            {21,
             context.GetDeviceId(),
             rhi::QueueRole::Graphics,
             rhi::SubmissionDomain::Frame,
             rhi::InvalidRenderViewId,
             rhi::PipelineStage::AllGraphics,
             {}},
            {22,
             context.GetDeviceId(),
             rhi::QueueRole::Compute,
             rhi::SubmissionDomain::Frame,
             rhi::InvalidRenderViewId,
             rhi::PipelineStage::ComputeShader,
             {20, 21}},
        };
        rhi::SubmissionPlan overlappedPlan;
        assert(rhi::BuildSubmissionPlan(overlappedWork, overlappedPlan, error));
        assert(overlappedPlan.batches.back().queue == rhi::QueueRole::Compute);
        const auto overlapped = executor.Execute(
            0, overlappedPlan, [](uint32_t, VkCommandBuffer commandBuffer) { return commandBuffer != VK_NULL_HANDLE; },
            sync);
        assert(overlapped.Succeeded());
        assert(overlapped.completionTicket.queue == rhi::QueueRole::Compute);
        assert(overlapped.completionTimeline != VK_NULL_HANDLE && overlapped.completionTimelineValue != 0);
        assert(vkWaitForFences(context.GetDevice(), 1, &completionFence, VK_TRUE, 5'000'000'000ull) == VK_SUCCESS);
        executor.CompleteFrame(0);
        queues.CompleteCompletionEpoch(sync.completionEpoch);

        assert(vkResetFences(context.GetDevice(), 1, &completionFence) == VK_SUCCESS);
        sync.completionEpoch = queues.ReserveCompletionEpoch();
        rhi::SubmissionPlan nextFramePlan;
        assert(rhi::BuildSubmissionPlan({{23,
                                          context.GetDeviceId(),
                                          rhi::QueueRole::Graphics,
                                          rhi::SubmissionDomain::Frame,
                                          rhi::InvalidRenderViewId,
                                          rhi::PipelineStage::AllGraphics,
                                          {}}},
                                        nextFramePlan, error));
        sync.previousFrameTimeline = overlapped.completionTimeline;
        sync.previousFrameTimelineValue = overlapped.completionTimelineValue;
        const auto nextFrame = executor.Execute(
            1, nextFramePlan, [](uint32_t, VkCommandBuffer commandBuffer) { return commandBuffer != VK_NULL_HANDLE; },
            sync);
        assert(nextFrame.Succeeded());
        assert(vkWaitForFences(context.GetDevice(), 1, &completionFence, VK_TRUE, 5'000'000'000ull) == VK_SUCCESS);
        executor.CompleteFrame(1);
        queues.CompleteCompletionEpoch(sync.completionEpoch);
        sync.previousFrameTimeline = VK_NULL_HANDLE;
        sync.previousFrameTimelineValue = 0;

        // A graph-generation prime starts on Compute while preserving GPU
        // resident state from the previous generation. Verify that the
        // previous generation dependency gates that first Compute batch.
        VkSemaphoreTypeCreateInfo gateType{};
        gateType.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        gateType.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        gateType.initialValue = 0;
        VkSemaphoreCreateInfo gateInfo{};
        gateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        gateInfo.pNext = &gateType;
        VkSemaphore generationGate = VK_NULL_HANDLE;
        assert(vkCreateSemaphore(context.GetDevice(), &gateInfo, nullptr, &generationGate) == VK_SUCCESS);

        assert(vkResetFences(context.GetDevice(), 1, &completionFence) == VK_SUCCESS);
        sync.completionEpoch = queues.ReserveCompletionEpoch();
        sync.previousFrameTimeline = generationGate;
        sync.previousFrameTimelineValue = 1;
        sync.previousFrameWaitAtFirstBatch = true;
        const auto generationPrime = executor.Execute(
            0, overlappedPlan, [](uint32_t, VkCommandBuffer commandBuffer) { return commandBuffer != VK_NULL_HANDLE; },
            sync);
        assert(generationPrime.Succeeded());
        assert(vkWaitForFences(context.GetDevice(), 1, &completionFence, VK_TRUE, 1'000'000ull) == VK_TIMEOUT);

        VkSemaphoreSignalInfo signalInfo{};
        signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
        signalInfo.semaphore = generationGate;
        signalInfo.value = 1;
        assert(vkSignalSemaphore(context.GetDevice(), &signalInfo) == VK_SUCCESS);
        assert(vkWaitForFences(context.GetDevice(), 1, &completionFence, VK_TRUE, 5'000'000'000ull) == VK_SUCCESS);
        executor.CompleteFrame(0);
        queues.CompleteCompletionEpoch(sync.completionEpoch);
        vkDestroySemaphore(context.GetDevice(), generationGate, nullptr);
        sync.previousFrameTimeline = VK_NULL_HANDLE;
        sync.previousFrameTimelineValue = 0;
        sync.previousFrameWaitAtFirstBatch = false;
    }

    // Compute/transfer work has no camera, render view, or presentation sync.
    // It must use the same executor and completion accounting without adding
    // an empty Graphics pass to satisfy a frame-only precondition.
    VkSemaphoreTypeCreateInfo uploadGateType{};
    uploadGateType.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    uploadGateType.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo uploadGateInfo{};
    uploadGateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    uploadGateInfo.pNext = &uploadGateType;
    VkSemaphore uploadGate = VK_NULL_HANDLE;
    assert(vkCreateSemaphore(context.GetDevice(), &uploadGateInfo, nullptr, &uploadGate) == VK_SUCCESS);
    uint64_t uploadValue = 0;
    for (const auto role : {rhi::QueueRole::Graphics, rhi::QueueRole::Compute, rhi::QueueRole::Transfer}) {
        // Use production RHI allocation/encoding/readback, not raw Vulkan
        // allocations or a second plugin-owned device/queue. Transfer commands
        // are legal on both logical queues and need no Graphics batch.
        auto &device = context.GetRhiDevice();
        rhi::BufferDesc bufferDesc;
        bufferDesc.byteSize = 64;
        bufferDesc.usage = rhi::BufferUsageFlags::TransferSource | rhi::BufferUsageFlags::TransferDestination;
        bufferDesc.queueAccess = role == rhi::QueueRole::Graphics  ? rhi::QueueAccessFlags::Graphics
                                 : role == rhi::QueueRole::Compute ? rhi::QueueAccessFlags::Compute
                                                                   : rhi::QueueAccessFlags::Transfer;
        const auto buffer = device.CreateBuffer(bufferDesc);
        assert(buffer.IsValid());
        bufferDesc.memory = rhi::BufferMemory::Readback;
        bufferDesc.usage = rhi::BufferUsageFlags::TransferDestination;
        const auto readback = device.CreateBuffer(bufferDesc);
        assert(readback.IsValid());
        bufferDesc.memory = rhi::BufferMemory::Upload;
        bufferDesc.usage = rhi::BufferUsageFlags::TransferSource;
        const auto upload = device.CreateBuffer(bufferDesc);
        assert(upload.IsValid());
        using MapAccess = rhi::BufferMapAccess;
        assert(!device.MapBuffer(buffer, 0, 4, MapAccess::Write));
        assert(!device.MapBuffer(upload, 0, 4, MapAccess::Read));
        assert(!device.MapBuffer(upload, 0, 4, MapAccess::ReadWrite));
        assert(!device.MapBuffer(upload, 0, 0, MapAccess::Write));
        assert(!device.MapBuffer(upload, 60, 8, MapAccess::Write));
        assert(!device.MapBuffer(upload, std::numeric_limits<uint64_t>::max(), 4, MapAccess::Write));
        assert(!device.MapBuffer(upload, 0, 4, static_cast<MapAccess>(255)));
        assert(!device.UnmapBuffer(upload, 0, 4, static_cast<MapAccess>(255)));
        // Non-zero mapped offset must select only that byte range; the copy
        // later reads this region through the actual GPU transfer command.
        auto *mappedUpload = static_cast<uint32_t *>(device.MapBuffer(upload, 8, 8, MapAccess::Write));
        assert(mappedUpload);
        mappedUpload[0] = 0x01020304u;
        mappedUpload[1] = 0x50607080u;
        assert(device.UnmapBuffer(upload, 8, 8, MapAccess::Write));
        rhi::SubmissionPlan standalonePlan;
        assert(rhi::BuildSubmissionPlan({{30,
                                          context.GetDeviceId(),
                                          role,
                                          rhi::SubmissionDomain::Background,
                                          rhi::InvalidRenderViewId,
                                          rhi::PipelineStage::Transfer,
                                          {}},
                                         {31,
                                          context.GetDeviceId(),
                                          role,
                                          rhi::SubmissionDomain::Background,
                                          rhi::InvalidRenderViewId,
                                          rhi::PipelineStage::Transfer,
                                          {30},
                                          true},
                                         {32,
                                          context.GetDeviceId(),
                                          role,
                                          rhi::SubmissionDomain::Background,
                                          rhi::InvalidRenderViewId,
                                          rhi::PipelineStage::Transfer,
                                          {31},
                                          true}},
                                        standalonePlan, error));
        assert(standalonePlan.batches.size() == 3);
        assert(vkResetFences(context.GetDevice(), 1, &completionFence) == VK_SUCCESS);
        sync.completionEpoch = queues.ReserveCompletionEpoch();
        sync.uploadTimeline = uploadGate;
        sync.uploadTimelineValue = ++uploadValue;
        const auto reservedBefore = queues.GetSnapshot(role).lastReserved;
        const auto standalone = executor.Execute(
            0, standalonePlan,
            [&](uint32_t batchIndex, VkCommandBuffer commandBuffer) {
                vk::VulkanTransferCommandContext commands;
                const auto transfer = device.MakeTransferCommandEncoder(commands, commandBuffer);
                if (batchIndex == 0) {
                    assert(!transfer.FillBuffer(buffer, 64, 4));
                    assert(!transfer.FillBuffer(buffer, 60, 8));
                    assert(!transfer.FillBuffer(buffer, std::numeric_limits<uint64_t>::max() - 3, 8));
                    assert(transfer.FillBuffer(buffer, 0, 64, 0x12345678u));
                    return true;
                }
                VkBufferMemoryBarrier barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.buffer = device.Resolve(buffer);
                barrier.size = 64;
                if (batchIndex == 1) {
                    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         0, 0, nullptr, 1, &barrier, 0, nullptr);
                    transfer.CopyBuffer(upload, buffer, {8, 0, 8});
                    assert(transfer.FillBuffer(buffer, 16, 16));
                    assert(transfer.FillBuffer(buffer, 60, 4, 0xfedcba98u));
                    return true;
                }
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                     0, nullptr, 1, &barrier, 0, nullptr);
                transfer.CopyBuffer(buffer, readback, {0, 0, 64});
                barrier.buffer = device.Resolve(readback);
                barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0,
                                     nullptr, 1, &barrier, 0, nullptr);
                return true;
            },
            sync);
        assert(standalone.Succeeded());
        assert(standalone.completionTicket.queue == role);
        assert(standalone.completionTicket.serial == reservedBefore + 3);
        assert(queues.GetSnapshot(role).lastSubmitted == standalone.completionTicket.serial);
        // One host submit must retain the first batch's external wait and the
        // last batch's fence. It must not finish before the upload gate opens.
        assert(vkWaitForFences(context.GetDevice(), 1, &completionFence, VK_TRUE, 1'000'000ull) == VK_TIMEOUT);
        VkSemaphoreSignalInfo uploadSignal{};
        uploadSignal.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
        uploadSignal.semaphore = uploadGate;
        uploadSignal.value = uploadValue;
        assert(vkSignalSemaphore(context.GetDevice(), &uploadSignal) == VK_SUCCESS);
        assert(vkWaitForFences(context.GetDevice(), 1, &completionFence, VK_TRUE, 5'000'000'000ull) == VK_SUCCESS);
        executor.CompleteFrame(0);
        queues.CompleteCompletionEpoch(sync.completionEpoch);
        assert(queues.GetCompletedSerial(role) == standalone.completionTicket.serial);
        std::array<uint32_t, 16> actual{};
        assert(device.ReadBuffer(readback, 0, actual.data(), sizeof(actual)));
        for (size_t index = 0; index < actual.size(); ++index) {
            const uint32_t expected = index == 0    ? 0x01020304u
                                      : index == 1  ? 0x50607080u
                                      : index == 15 ? 0xfedcba98u
                                                    : (index >= 4 && index < 8 ? 0u : 0x12345678u);
            assert(actual[index] == expected);
        }
        auto *mappedReadback = static_cast<uint32_t *>(device.MapBuffer(readback, 4, 8, MapAccess::ReadWrite));
        assert(mappedReadback && mappedReadback[0] == 0x50607080u && mappedReadback[1] == 0x12345678u);
        mappedReadback[1] = 99;
        assert(device.UnmapBuffer(readback, 4, 8, MapAccess::ReadWrite));
        uint32_t updated = 0;
        assert(device.ReadBuffer(readback, 8, &updated, sizeof(updated)) && updated == 99);
        assert(!device.UnmapBuffer(readback, 60, 8, MapAccess::Read));
        device.Release(buffer);
        device.Release(readback);
        device.Release(upload);
        assert(!device.MapBuffer(upload, 8, 8, MapAccess::Write));
        assert(!device.UnmapBuffer(upload, 8, 8, MapAccess::Write));
        device.CollectResourceRetirements(sync.completionEpoch);
    }
    vkDestroySemaphore(context.GetDevice(), uploadGate, nullptr);
    sync.uploadTimeline = VK_NULL_HANDLE;
    sync.uploadTimelineValue = 0;

    // Independent background lanes must both contribute to completion. Gate
    // the Transfer root; Compute may finish first, but the final fence cannot.
    VkSemaphoreTypeCreateInfo backgroundGateType{};
    backgroundGateType.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    backgroundGateType.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo backgroundGateInfo{};
    backgroundGateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    backgroundGateInfo.pNext = &backgroundGateType;
    VkSemaphore backgroundGate = VK_NULL_HANDLE;
    assert(vkCreateSemaphore(context.GetDevice(), &backgroundGateInfo, nullptr, &backgroundGate) == VK_SUCCESS);
    rhi::SubmissionPlan backgroundPlan;
    assert(rhi::BuildSubmissionPlan({{40,
                                      context.GetDeviceId(),
                                      rhi::QueueRole::Transfer,
                                      rhi::SubmissionDomain::Background,
                                      rhi::InvalidRenderViewId,
                                      rhi::PipelineStage::Transfer,
                                      {}},
                                     {41,
                                      context.GetDeviceId(),
                                      rhi::QueueRole::Compute,
                                      rhi::SubmissionDomain::Background,
                                      rhi::InvalidRenderViewId,
                                      rhi::PipelineStage::ComputeShader,
                                      {}}},
                                    backgroundPlan, error));
    assert(vkResetFences(context.GetDevice(), 1, &completionFence) == VK_SUCCESS);
    sync.completionEpoch = queues.ReserveCompletionEpoch();
    sync.previousFrameTimeline = backgroundGate;
    sync.previousFrameTimelineValue = 1;
    // No Graphics batch exists: the external dependency gates the first work
    // even with the normal (false) previousFrameWaitAtFirstBatch setting.
    const auto background = executor.Execute(
        0, backgroundPlan, [](uint32_t, VkCommandBuffer commandBuffer) { return commandBuffer != VK_NULL_HANDLE; },
        sync);
    assert(background.Succeeded());
    assert(background.completionTicket.queue == rhi::QueueRole::Compute);
    assert(vkWaitForFences(context.GetDevice(), 1, &completionFence, VK_TRUE, 1'000'000ull) == VK_TIMEOUT);
    VkSemaphoreSignalInfo backgroundSignal{};
    backgroundSignal.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    backgroundSignal.semaphore = backgroundGate;
    backgroundSignal.value = 1;
    assert(vkSignalSemaphore(context.GetDevice(), &backgroundSignal) == VK_SUCCESS);
    assert(vkWaitForFences(context.GetDevice(), 1, &completionFence, VK_TRUE, 5'000'000'000ull) == VK_SUCCESS);
    executor.CompleteFrame(0);
    queues.CompleteCompletionEpoch(sync.completionEpoch);
    vkDestroySemaphore(context.GetDevice(), backgroundGate, nullptr);
    sync.previousFrameTimeline = VK_NULL_HANDLE;
    sync.previousFrameTimelineValue = 0;

    // Reject nonsensical presentation sync before recording or reserving work.
    VkSemaphoreCreateInfo presentationInfo{};
    presentationInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkSemaphore presentationSignal = VK_NULL_HANDLE;
    assert(vkCreateSemaphore(context.GetDevice(), &presentationInfo, nullptr, &presentationSignal) == VK_SUCCESS);
    for (bool waitForImage : {false, true}) {
        sync.imageAvailable = waitForImage ? presentationSignal : VK_NULL_HANDLE;
        sync.renderFinished = waitForImage ? VK_NULL_HANDLE : presentationSignal;
        const auto reservedBefore = queues.GetSnapshot(rhi::QueueRole::Compute).lastReserved;
        const auto rejected = executor.Execute(
            0, backgroundPlan,
            [](uint32_t, VkCommandBuffer) {
                assert(false && "invalid presentation sync reached command recording");
                return true;
            },
            sync);
        assert(!rejected.Succeeded() && !rejected.submittedAny);
        assert(queues.GetSnapshot(rhi::QueueRole::Compute).lastReserved == reservedBefore);
    }
    sync.imageAvailable = VK_NULL_HANDLE;
    sync.renderFinished = VK_NULL_HANDLE;
    vkDestroySemaphore(context.GetDevice(), presentationSignal, nullptr);

    rhi::SubmissionPlan failedPlan;
    assert(rhi::BuildSubmissionPlan({work.front()}, failedPlan, error));
    sync.completionEpoch = queues.ReserveCompletionEpoch();
    const auto failed = executor.Execute(
        1, failedPlan,
        [&](uint32_t, VkCommandBuffer) {
            assert(vkdebug::GetDescriptorRecordingSubmissionSerial() == sync.completionEpoch);
            return false;
        },
        sync);
    assert(!failed.Succeeded());
    assert(vkdebug::GetDescriptorRecordingSubmissionSerial() == rhi::InvalidSubmissionSerial);
    queues.CompleteCompletionEpoch(sync.completionEpoch);
    const auto afterFailure = queues.Reserve(rhi::QueueRole::Graphics);
    assert(afterFailure.IsValid());
    assert(queues.CancelReservation(afterFailure));

    context.WaitIdle();
    // Completion can be collected out of slot order. An older query must not
    // overwrite the newest submission shown by the profiler.
    vk::GpuTimestampQueries timestamps;
    if (timestamps.Initialize(context, 2, 1)) {
        for (uint32_t slot = 0; slot < 2; ++slot) {
            assert(vkResetFences(context.GetDevice(), 1, &completionFence) == VK_SUCCESS);
            vk::VulkanSubmissionExecutor::ExternalSync timingSync;
            timingSync.completionFence = completionFence;
            timingSync.completionEpoch = queues.ReserveCompletionEpoch();
            const auto timed = executor.Execute(
                slot, failedPlan,
                [&](uint32_t, VkCommandBuffer commands) {
                    timestamps.BeginFrame(commands, slot);
                    const auto region =
                        timestamps.BeginRegion(commands, "ordered_sample", VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
                    timestamps.EndRegion(commands, region, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
                    timestamps.FinishFrame(slot);
                    return true;
                },
                timingSync);
            assert(timed.Succeeded());
            timestamps.MarkSubmitted(slot);
            assert(vkWaitForFences(context.GetDevice(), 1, &completionFence, VK_TRUE, 5'000'000'000ull) == VK_SUCCESS);
            executor.CompleteFrame(slot);
            queues.CompleteCompletionEpoch(timingSync.completionEpoch);
        }
        assert(timestamps.CollectCompletedFrame(1));
        assert(timestamps.LatestFrame().serial == 2 && timestamps.LatestFrame().Find("ordered_sample"));
        assert(!timestamps.CollectCompletedFrame(0));
        assert(timestamps.LatestFrame().serial == 2);
    }
    timestamps.Destroy();
    vkDestroyFence(context.GetDevice(), completionFence, nullptr);
    executor.Destroy();
    queues.Destroy();
    context.Destroy();
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
