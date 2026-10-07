// Independent graph branches must own disjoint memory even when their linear
// lifetimes do not overlap. Submission order alone does not synchronize aliases.
bool VerifyTransientBufferIsolation(TestResources &resources)
{
    using namespace infernux;
    struct Owner
    {
        vk::VkDeviceContext &context;
        vk::VulkanQueueManager queues;
        vk::VulkanSubmissionExecutor executor;
        RenderGraph graph;
        std::array<rhi::BufferHandle, 3> readbacks{};
        VkFence fence = VK_NULL_HANDLE;
        ~Owner()
        {
            context.WaitIdle();
            graph.Destroy();
            executor.Destroy();
            queues.Destroy();
            for (const auto buffer : readbacks)
                context.GetRhiDevice().Release(buffer);
            if (fence)
                vkDestroyFence(context.GetDevice(), fence, nullptr);
        }
    } owner{resources.context};
    if (!Require(owner.queues.Initialize(resources.context, 1) &&
                     owner.executor.Initialize(resources.context, owner.queues, 1),
                 "Transient buffer executor initialization failed"))
        return false;
    owner.graph.Initialize(&resources.context, nullptr, &owner.queues);
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (!Require(vkCreateFence(resources.context.GetDevice(), &fenceInfo, nullptr, &owner.fence) == VK_SUCCESS,
                 "Transient buffer fence creation failed"))
        return false;
    auto &device = resources.context.GetRhiDevice();
    const std::array<rhi::QueueRole, 3> roles = {rhi::QueueRole::Graphics, rhi::QueueRole::Compute,
                                                 rhi::QueueRole::Transfer};
    for (const auto role : roles) {
        const auto queue = owner.queues.GetSnapshot(role);
        std::cout << "Transient buffer queue role=" << static_cast<int>(role) << " family=" << queue.family
                  << " lane=" << queue.nativeLane << '\n';
    }
    for (const uint64_t bytes : {16u, 256u, 65540u, 1048580u}) {
        for (uint32_t mixedQueues = 0; mixedQueues < 2; ++mixedQueues) {
            owner.graph.Reset();
            for (uint32_t branch = 0; branch < 3; ++branch) {
                device.Release(owner.readbacks[branch]);
                rhi::BufferDesc desc;
                desc.byteSize = bytes;
                desc.memory = rhi::BufferMemory::Readback;
                desc.usage = rhi::BufferUsageFlags::TransferDestination;
                desc.queueAccess = !mixedQueues || branch == 0 ? rhi::QueueAccessFlags::Graphics
                                   : branch == 1               ? rhi::QueueAccessFlags::Compute
                                                               : rhi::QueueAccessFlags::Transfer;
                owner.readbacks[branch] = device.CreateBuffer(desc);
                if (!Require(owner.readbacks[branch].IsValid(), "Transient branch readback allocation failed"))
                    return false;
            }
            for (uint32_t rebuild = 0; rebuild < 2; ++rebuild) {
                owner.graph.Reset();
                std::array<ResourceHandle, 3> buffers{};
                std::vector<std::string> expectedOrder;
                const uint32_t patternBase = 0x12340000u + rebuild * 0x10000u;
                for (uint32_t branch = 0; branch < 3; ++branch) {
                    const auto role = mixedQueues ? roles[branch] : roles[0];
                    const auto suffix = std::to_string(branch);
                    expectedOrder.push_back("Fill" + suffix);
                    expectedOrder.push_back("Read" + suffix);
                    owner.graph.AddTransferPass("Fill" + suffix, [&, branch, role, suffix](PassBuilder &builder) {
                        builder.SetQueueRole(role);
                        buffers[branch] = builder.TransferWrite(
                            builder.CreateBuffer("Branch" + suffix, bytes,
                                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT));
                        return [&, branch, patternBase](RenderContext &context) {
                            if (!context.GetTransferCommandEncoder().FillBuffer(
                                    context.GetBufferHandle(buffers[branch]), 0, bytes, patternBase + branch))
                                throw std::runtime_error("Transient branch fill failed");
                        };
                    });
                    owner.graph.AddTransferPass("Read" + suffix, [&, branch, role, suffix](PassBuilder &builder) {
                        builder.SetQueueRole(role);
                        builder.TransferRead(buffers[branch]);
                        const auto output = builder.TransferWrite(
                            builder.ImportBuffer("Result" + suffix, owner.readbacks[branch], bytes));
                        return [&, branch, output](RenderContext &context) {
                            context.GetTransferCommandEncoder().CopyBuffer(context.GetBufferHandle(buffers[branch]),
                                                                           context.GetBufferHandle(output),
                                                                           {0, 0, bytes});
                        };
                    });
                }
                ResourceHandle unused;
                owner.graph.AddTransferPass("Unreachable", [&](PassBuilder &builder) {
                    unused =
                        builder.TransferWrite(builder.CreateBuffer("Unused", bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT));
                    return [](RenderContext &) { throw std::runtime_error("Culled branch executed"); };
                });
                if (!Require(owner.graph.Compile() && owner.graph.GetExecutionPassNames() == expectedOrder,
                             "Independent branch lifetimes are not disjoint in the compiled order"))
                    return false;
                const auto allocations = owner.graph.GetTransientAllocationCount();
                if (allocations != 3)
                    std::cerr << "Transient branch allocation count=" << allocations << " expected=3\n";
                if (!Require(allocations == 3 && owner.graph.GetTransientResidentBytes() >= 3 * bytes &&
                                 owner.graph.ResolveBuffer(unused) == VK_NULL_HANDLE,
                             "Independent transient buffers share an allocation or allocate a culled resource"))
                    return false;
                std::array<VkBuffer, 3> resident{};
                for (uint32_t branch = 0; branch < 3; ++branch)
                    resident[branch] = owner.graph.ResolveBuffer(buffers[branch]);
                if (!Require(owner.graph.Compile() && owner.graph.GetTransientAllocationCount() == 3,
                             "Compiling resident buffers again allocated duplicate storage"))
                    return false;
                for (uint32_t branch = 0; branch < 3; ++branch)
                    if (!Require(resident[branch] == owner.graph.ResolveBuffer(buffers[branch]),
                                 "Recompilation replaced a resident buffer"))
                        return false;

                if (!Require(vkResetFences(resources.context.GetDevice(), 1, &owner.fence) == VK_SUCCESS,
                             "Transient buffer fence reset failed"))
                    return false;
                owner.graph.BeginExecution();
                vk::VulkanSubmissionExecutor::ExternalSync sync;
                sync.completionFence = owner.fence;
                sync.completionEpoch = owner.queues.ReserveCompletionEpoch();
                const auto submitted = owner.executor.Execute(
                    0, owner.graph.GetSubmissionPlan(),
                    [&](uint32_t batch, VkCommandBuffer commands) {
                        return owner.graph.RecordSubmissionBatch(batch, commands);
                    },
                    sync);
                if (!Require(submitted.Succeeded() && vkWaitForFences(resources.context.GetDevice(), 1, &owner.fence,
                                                                      VK_TRUE, 5'000'000'000ull) == VK_SUCCESS,
                             "Transient buffer GPU submission failed"))
                    return false;
                owner.executor.CompleteFrame(0);
                owner.queues.CompleteCompletionEpoch(sync.completionEpoch);
                std::vector<uint32_t> actual(bytes / sizeof(uint32_t));
                for (uint32_t branch = 0; branch < 3; ++branch) {
                    if (!Require(device.ReadBuffer(owner.readbacks[branch], 0, actual.data(), bytes) &&
                                     std::all_of(actual.begin(), actual.end(),
                                                 [&](uint32_t word) { return word == patternBase + branch; }),
                                 "Transient branch GPU data was overwritten by an independent branch"))
                        return false;
                }
            }
        }
    }
    owner.graph.Reset();
    return Require(owner.graph.GetTransientAllocationCount() == 0 && owner.graph.GetTransientResidentBytes() == 0,
                   "Transient buffer allocations remained resident after graph reset");
}
