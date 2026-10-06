// The caller owns the Vulkan fixture; this verifies immutable command payloads
// across multiple recordings made before either submission completes.
bool VerifyRhiBufferUpdateSnapshots(TestResources &resources)
{
    using namespace infernux;
    auto &device = resources.context.GetRhiDevice();
    const VkDevice native = resources.context.GetDevice();
    for (const uint64_t payloadBytes : {4u, 256u, 65532u, 65536u, 65540u, 262144u, 1048580u}) {
        constexpr uint64_t prefixBytes = 32;
        const uint64_t totalBytes = payloadBytes + 2 * prefixBytes;
        struct Owner
        {
            vk::VulkanRhiDevice &device;
            VkDevice native;
            rhi::BufferHandle target;
            VkCommandPool pool = VK_NULL_HANDLE;
            ~Owner()
            {
                if (pool)
                    vkDestroyCommandPool(native, pool, nullptr);
                device.Release(target);
            }
        } owner{device, native};
        vk::VulkanCommandUploads uploads(resources.context.GetVmaAllocator());
        rhi::BufferDesc desc;
        desc.byteSize = totalBytes;
        desc.usage = rhi::BufferUsageFlags::TransferSource | rhi::BufferUsageFlags::TransferDestination;
        owner.target = device.CreateBuffer(desc);
        if (!Require(owner.target.IsValid(), "Command update target allocation failed"))
            return false;
        BufferReadback readback;
        if (!Require(readback.Create(resources.context.GetVmaAllocator(), totalBytes * 2),
                     "Command update readback allocation failed"))
            return false;
        VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolInfo.queueFamilyIndex = resources.context.GetQueueIndices().graphicsFamily.value();
        if (!Require(vkCreateCommandPool(native, &poolInfo, nullptr, &owner.pool) == VK_SUCCESS,
                     "Command update pool allocation failed"))
            return false;
        std::array<VkCommandBuffer, 2> commands{};
        VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate.commandPool = owner.pool;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = static_cast<uint32_t>(commands.size());
        if (!Require(vkAllocateCommandBuffers(native, &allocate, commands.data()) == VK_SUCCESS,
                     "Command update buffer allocation failed"))
            return false;
        for (uint32_t replay = 0; replay < 2; ++replay) {
            if (!Require(vkResetCommandPool(native, owner.pool, 0) == VK_SUCCESS, "Command update pool reset failed"))
                return false;
            uploads.Reset();
            std::vector<uint32_t> source(payloadBytes / 4);
            for (uint32_t frame = 0; frame < commands.size(); ++frame) {
                for (size_t word = 0; word < source.size(); ++word)
                    source[word] = (frame == 0 ? 0x12340000u : 0x76540000u) + static_cast<uint32_t>(word);
                const auto command = commands[frame];
                VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                if (!Require(vkBeginCommandBuffer(command, &begin) == VK_SUCCESS, "Command update begin failed"))
                    return false;
                vk::VulkanTransferCommandContext transferContext;
                const auto transfer = device.MakeTransferCommandEncoder(transferContext, command);
                if (payloadBytes > 65536 &&
                    !Require(!transfer.UpdateBuffer(owner.target, prefixBytes, source.data(), payloadBytes),
                             "Large upload accepted without recording storage"))
                    return false;
                vk::VulkanCommandUploads::Scope uploadScope(uploads, command);
                VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
                barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.buffer = device.Resolve(owner.target);
                barrier.size = VK_WHOLE_SIZE;
                if (frame == 0 && !Require(transfer.FillBuffer(owner.target, 0, totalBytes, 0xDEADBEEFu),
                                           "Command update guard fill failed"))
                    return false;
                barrier.srcAccessMask = frame == 0 ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_TRANSFER_READ_BIT;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                     nullptr, 1, &barrier, 0, nullptr);
                if (!Require(!transfer.UpdateBuffer(owner.target, totalBytes, source.data(), 4) &&
                                 !transfer.UpdateBuffer(owner.target, UINT64_MAX - 3, source.data(), 4) &&
                                 !transfer.UpdateBuffer(owner.target, prefixBytes, source.data(), totalBytes) &&
                                 !transfer.UpdateBuffer({owner.target.index, owner.target.generation + 1}, 0,
                                                        source.data(), 4),
                             "Invalid command update range or handle was accepted"))
                    return false;
                if (!Require(transfer.UpdateBuffer(owner.target, prefixBytes, source.data(), payloadBytes),
                             "Valid command update was rejected"))
                    return false;
                std::fill(source.begin(), source.end(), 0xBAD0BAD0u);
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                     nullptr, 1, &barrier, 0, nullptr);
                VkBufferCopy copy{0, totalBytes * frame, totalBytes};
                vkCmdCopyBuffer(command, barrier.buffer, readback.buffer, 1, &copy);
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                barrier.buffer = readback.buffer;
                barrier.offset = totalBytes * frame;
                barrier.size = totalBytes;
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr,
                                     1, &barrier, 0, nullptr);
                if (!Require(vkEndCommandBuffer(command) == VK_SUCCESS, "Command update end failed"))
                    return false;
            }
            source.clear();
            source.shrink_to_fit();
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit.commandBufferCount = static_cast<uint32_t>(commands.size());
            submit.pCommandBuffers = commands.data();
            const auto queue = resources.context.GetGraphicsQueue();
            const auto submitted = vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
            const auto waited = vkQueueWaitIdle(queue);
            if (!Require(submitted == VK_SUCCESS && waited == VK_SUCCESS, "Command update submission failed") ||
                !Require(vmaInvalidateAllocation(readback.allocator, readback.allocation, 0, totalBytes * 2) ==
                             VK_SUCCESS,
                         "Command update readback invalidation failed"))
                return false;
            const auto *actual = static_cast<const uint32_t *>(readback.mapped);
            for (uint32_t frame = 0; frame < commands.size(); ++frame) {
                for (uint64_t word = 0; word < totalBytes / 4; ++word) {
                    const bool payload = word >= prefixBytes / 4 && word < (prefixBytes + payloadBytes) / 4;
                    const uint32_t expected = payload ? (frame == 0 ? 0x12340000u : 0x76540000u) +
                                                            static_cast<uint32_t>(word - prefixBytes / 4)
                                                      : 0xDEADBEEFu;
                    if (!Require(actual[frame * totalBytes / 4 + word] == expected,
                                 "Recorded upload lost its immutable payload or overwrote guards"))
                        return false;
                }
            }
            const size_t expectedPages = payloadBytes <= 65536 ? 0 : payloadBytes <= 262144 ? 1 : 2;
            if (!Require(uploads.PageCount() == expectedPages, "Command upload pages were not reused"))
                return false;
            std::cout << "Immutable command update bytes=" << payloadBytes << " recordings=2 replay=" << replay
                      << " pages=" << uploads.PageCount() << '\n';
        }
        if (!Require(vkResetCommandPool(native, owner.pool, 0) == VK_SUCCESS, "Idle upload pool reset failed"))
            return false;
        uploads.Reset();
        uploads.Reset(); // A completed recording with no large updates retains no idle pages.
        if (!Require(uploads.PageCount() == 0, "Idle command uploads retained unused pages"))
            return false;
    }
    return true;
}
