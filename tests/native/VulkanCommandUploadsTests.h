// Uses the production executor, including its recording storage and slot
// lifetime. The timeline gate keeps both frames in flight while host data dies.
void VerifyExecutorCommandUploads(vk::VkDeviceContext &context, vk::VulkanQueueManager &queues,
                                  vk::VulkanSubmissionExecutor &executor)
{
    auto &device = context.GetRhiDevice();
    constexpr uint64_t byteSize = 256 * 1024;
    for (const auto role : {rhi::QueueRole::Graphics, rhi::QueueRole::Compute, rhi::QueueRole::Transfer}) {
        rhi::BufferDesc desc;
        desc.byteSize = byteSize;
        desc.usage = rhi::BufferUsageFlags::TransferSource | rhi::BufferUsageFlags::TransferDestination;
        desc.queueAccess = role == rhi::QueueRole::Graphics  ? rhi::QueueAccessFlags::Graphics
                           : role == rhi::QueueRole::Compute ? rhi::QueueAccessFlags::Compute
                                                             : rhi::QueueAccessFlags::Transfer;
        const auto target = device.CreateBuffer(desc);
        desc.byteSize *= 2;
        desc.memory = rhi::BufferMemory::Readback;
        desc.usage = rhi::BufferUsageFlags::TransferDestination;
        const auto readback = device.CreateBuffer(desc);
        assert(target.IsValid() && readback.IsValid());
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        semaphoreInfo.pNext = &type;
        VkSemaphore gate = VK_NULL_HANDLE;
        assert(vkCreateSemaphore(context.GetDevice(), &semaphoreInfo, nullptr, &gate) == VK_SUCCESS);
        std::array<VkFence, 2> fences{};
        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        for (auto &fence : fences)
            assert(vkCreateFence(context.GetDevice(), &fenceInfo, nullptr, &fence) == VK_SUCCESS);
        rhi::SubmissionPlan plan;
        std::string error;
        assert(rhi::BuildSubmissionPlan({{100,
                                          context.GetDeviceId(),
                                          role,
                                          rhi::SubmissionDomain::Background,
                                          rhi::InvalidRenderViewId,
                                          rhi::PipelineStage::Transfer,
                                          {}}},
                                        plan, error));
        for (uint32_t cycle = 0; cycle < 2; ++cycle) {
            std::array<rhi::SubmissionSerial, 2> epochs{};
            std::vector<uint32_t> source(byteSize / 4);
            for (uint32_t frame = 0; frame < 2; ++frame) {
                assert(vkResetFences(context.GetDevice(), 1, &fences[frame]) == VK_SUCCESS);
                const uint32_t value = 0x12340000u + cycle * 256 + frame;
                std::fill(source.begin(), source.end(), value);
                vk::VulkanSubmissionExecutor::ExternalSync sync;
                sync.completionEpoch = epochs[frame] = queues.ReserveCompletionEpoch();
                sync.completionFence = fences[frame];
                sync.uploadTimeline = gate;
                sync.uploadTimelineValue = cycle + 1;
                const auto submitted = executor.Execute(
                    frame, plan,
                    [&](uint32_t, VkCommandBuffer commands) {
                        auto *storage = vk::VulkanCommandUploads::Current(context.GetVmaAllocator(), commands);
                        assert(storage);
                        VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
                        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                        barrier.buffer = device.Resolve(target);
                        barrier.size = byteSize;
                        vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                             0, 0, nullptr, 1, &barrier, 0, nullptr);
                        vk::VulkanTransferCommandContext transferContext;
                        const auto transfer = device.MakeTransferCommandEncoder(transferContext, commands);
                        assert(transfer.UpdateBuffer(target, 0, source.data(), byteSize));
                        assert(storage->PageCount() == 1); // Includes the second slot-reuse cycle.
                        std::fill(source.begin(), source.end(), 0xBAD0BAD0u);
                        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                        vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                             0, 0, nullptr, 1, &barrier, 0, nullptr);
                        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
                        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                        barrier.buffer = device.Resolve(readback);
                        barrier.offset = frame * byteSize;
                        vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
                        transfer.CopyBuffer(target, readback, {0, frame * byteSize, byteSize});
                        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                        barrier.buffer = device.Resolve(readback);
                        barrier.offset = frame * byteSize;
                        vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0,
                                             nullptr, 1, &barrier, 0, nullptr);
                        return true;
                    },
                    sync);
                assert(submitted.Succeeded());
            }
            source.clear();
            source.shrink_to_fit();
            assert(vkWaitForFences(context.GetDevice(), 2, fences.data(), VK_TRUE, 1'000'000) == VK_TIMEOUT);
            uint32_t unexpectedRecordings = 0;
            vk::VulkanSubmissionExecutor::ExternalSync rejectedSync;
            rejectedSync.completionFence = fences[0];
            rejectedSync.completionEpoch = epochs[0];
            const auto rejected = executor.Execute(
                0, plan,
                [&](uint32_t, VkCommandBuffer) {
                    ++unexpectedRecordings;
                    return true;
                },
                rejectedSync);
            assert(!rejected.Succeeded() && !rejected.submittedAny && unexpectedRecordings == 0);
            VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
            signal.semaphore = gate;
            signal.value = cycle + 1;
            assert(vkSignalSemaphore(context.GetDevice(), &signal) == VK_SUCCESS);
            assert(vkWaitForFences(context.GetDevice(), 2, fences.data(), VK_TRUE, 5'000'000'000) == VK_SUCCESS);
            for (uint32_t frame = 0; frame < 2; ++frame) {
                executor.CompleteFrame(frame);
                queues.CompleteCompletionEpoch(epochs[frame]);
            }
            const auto *result =
                static_cast<const uint32_t *>(device.MapBuffer(readback, 0, byteSize * 2, rhi::BufferMapAccess::Read));
            assert(result);
            for (uint32_t frame = 0; frame < 2; ++frame)
                for (uint32_t word = 0; word < byteSize / 4; ++word)
                    assert(result[frame * byteSize / 4 + word] == 0x12340000u + cycle * 256 + frame);
            assert(device.UnmapBuffer(readback, 0, byteSize * 2, rhi::BufferMapAccess::Read));
        }
        device.Release(readback);
        device.Release(target);
        for (auto fence : fences)
            vkDestroyFence(context.GetDevice(), fence, nullptr);
        vkDestroySemaphore(context.GetDevice(), gate, nullptr);
    }
}
