bool VerifyParticleSpawnInputSnapshots(TestResources &resources, infernux::InxShaderLoader &compiler,
                                       const infernux::particle::GpuParticleSpawnProgram &spawnProgram)
{
    using namespace infernux;
    constexpr uint32_t wordCount = 48;
    constexpr uint64_t bytes = wordCount * sizeof(uint32_t);
    auto &device = resources.context.GetRhiDevice();
    const auto code = SpirvWords(compiler.CompileComputeGlsl(R"glsl(
#version 450
layout(local_size_x=1) in;
layout(std430, set=0, binding=0) writeonly buffer Output { uint result[]; };
layout(std430, set=3, binding=2) buffer Parameters { uvec4 parameters[]; };
layout(std430, set=3, binding=3) buffer Requests { uint requests[]; };
layout(std430, set=3, binding=4) readonly buffer Playing { uint playing[]; };
void main() {
    for (uint i=0u; i<12u; ++i) result[i] = parameters[i / 4u][i % 4u];
    for (uint i=0u; i<3u; ++i) {
        result[12u + i] = playing[i];
        result[15u + i] = requests[i];
    }
    parameters[1] += uvec4(100u);
    requests[0] = 2u;
    requests[1] = 1u;
}
)glsl",
                                                             "Tests/SpawnInputSnapshot.comp"));
    const auto poisonCode = SpirvWords(compiler.CompileComputeGlsl(R"glsl(
#version 450
layout(local_size_x=1) in;
layout(std430, set=0, binding=0) buffer Metadata { uint metadata[]; };
void main() {
    for (uint word=0u; word<24u; ++word) metadata[word] = 0x12340000u + word;
}
)glsl",
                                                                   "Tests/SpawnMetadataPoison.comp"));
    if (!Require(!code.empty() && !poisonCode.empty(), "Spawn input shader failed"))
        return false;
    struct Owner
    {
        vk::VkDeviceContext &context;
        vk::VulkanRhiDevice &device;
        vk::VulkanQueueManager queues;
        vk::VulkanSubmissionExecutor executor;
        vk::RenderGraph graph;
        particle::ParticleGpuRuntime runtime;
        particle::ParticleGpuGraphSpawnDomain spawn;
        rhi::BufferHandle readback;
        rhi::BindingLayoutHandle poisonLayout;
        rhi::ShaderModuleHandle poisonShader;
        rhi::ComputePipelineHandle poisonPipeline;
        rhi::BindGroupHandle poisonGroup;
        VkSemaphore gate = VK_NULL_HANDLE;
        std::array<VkFence, 2> fences{};
        ~Owner()
        {
            if (gate) {
                VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
                signal.semaphore = gate;
                signal.value = 99;
                vkSignalSemaphore(context.GetDevice(), &signal);
            }
            context.WaitIdle();
            executor.Destroy();
            queues.Destroy();
            graph.Destroy();
            device.Release(poisonGroup);
            device.Release(poisonPipeline);
            device.Release(poisonShader);
            device.Release(poisonLayout);
            spawn.Destroy();
            runtime.Destroy();
            device.Release(readback);
            for (auto fence : fences)
                if (fence)
                    vkDestroyFence(context.GetDevice(), fence, nullptr);
            if (gate)
                vkDestroySemaphore(context.GetDevice(), gate, nullptr);
        }
    } owner{resources.context, device};
    rhi::BufferDesc readback;
    readback.byteSize = bytes * 2;
    readback.usage = rhi::BufferUsageFlags::TransferDestination;
    readback.memory = rhi::BufferMemory::Readback;
    readback.queueAccess = rhi::QueueAccessFlags::Compute;
    owner.readback = device.CreateBuffer(readback);
    particle::GpuEmitterDesc desc;
    desc.capacity = 1;
    desc.stateStride = 128;
    for (auto &kernel : desc.kernels)
        kernel = {code.data(), code.size()};
    desc.kernels[static_cast<size_t>(particle::GpuKernelStage::UpdateRenderingFused)] = {};
    const std::vector<uint32_t> initial{10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30};
    if (!Require(owner.readback.IsValid() && owner.runtime.Create(device, desc) &&
                     owner.spawn.Create(device, 99601, 3, spawnProgram, initial) &&
                     owner.spawn.RegisterEmitter(0, owner.runtime) && owner.queues.Initialize(resources.context, 2) &&
                     owner.executor.Initialize(resources.context, owner.queues, 2),
                 "Spawn input fixture creation failed"))
        return false;
    rhi::BindingLayoutDesc poisonLayout;
    poisonLayout.entryCount = 1;
    poisonLayout.entries[0] = {0, rhi::BindingType::StorageBuffer, rhi::ShaderStage::Compute, 1};
    owner.poisonLayout = device.CreateBindingLayout(poisonLayout);
    owner.poisonShader =
        device.CreateShaderModule(rhi::ShaderModuleDesc::FromSpirV(poisonCode.data(), poisonCode.size()));
    rhi::ComputePipelineDesc poisonPipeline;
    poisonPipeline.computeShader = owner.poisonShader;
    poisonPipeline.bindingLayoutCount = 1;
    poisonPipeline.bindingLayouts[0] = owner.poisonLayout;
    owner.poisonPipeline = device.CreateComputePipeline(poisonPipeline);
    rhi::BindGroupDesc poisonGroup;
    poisonGroup.layout = owner.poisonLayout;
    poisonGroup.bufferCount = 1;
    poisonGroup.buffers[0] = {0, rhi::BindingType::StorageBuffer, owner.spawn.MetadataBuffer()};
    owner.poisonGroup = device.CreateBindGroup(poisonGroup);
    if (!Require(owner.poisonPipeline.IsValid() && owner.poisonGroup.IsValid(), "Spawn poison fixture failed"))
        return false;
    for (uint32_t slot = 0; slot < 3; ++slot)
        if (!Require(owner.spawn.SetEmitterAcceptingBurstRequests(slot, true), "Spawn acceptance setup failed"))
            return false;
    owner.graph.Initialize(&resources.context, nullptr);
    particle::ParticleGpuGraphSpawnDomain::GraphResources spawnResources;
    if (!Require(owner.spawn.Attach(owner.graph, "SpawnSnapshots", spawnResources), "Spawn advance attach failed"))
        return false;
    if (!Require(owner.graph.Compile(), "Spawn advance compilation failed"))
        return false;
    VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    semaphore.pNext = &type;
    if (!Require(vkCreateSemaphore(resources.context.GetDevice(), &semaphore, nullptr, &owner.gate) == VK_SUCCESS,
                 "Spawn gate creation failed"))
        return false;
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    for (auto &fence : owner.fences)
        if (!Require(vkCreateFence(resources.context.GetDevice(), &fenceInfo, nullptr, &fence) == VK_SUCCESS,
                     "Spawn fence creation failed"))
            return false;
    rhi::SubmissionPlan plan;
    std::string error;
    if (!Require(rhi::BuildSubmissionPlan({{100,
                                            resources.context.GetDeviceId(),
                                            rhi::QueueRole::Compute,
                                            rhi::SubmissionDomain::Background,
                                            rhi::InvalidRenderViewId,
                                            rhi::PipelineStage::Transfer | rhi::PipelineStage::ComputeShader,
                                            {}}},
                                          plan, error),
                 "Spawn input plan failed"))
        return false;
    const std::array<std::array<uint32_t, 3>, 4> expectedParameters{
        {{10, 20, 30}, {10, 120, 99}, {10, 20, 99}, {40, 120, 99}}};
    for (uint32_t cycle = 0; cycle < 2; ++cycle) {
        std::array<rhi::SubmissionSerial, 2> epochs{};
        for (uint32_t frame = 0; frame < 2; ++frame) {
            const uint32_t step = cycle * 2 + frame;
            if (step == 1 && !Require(owner.spawn.UpdateParameters({{0, {10, 10, 10, 10}}, {8, {99, 99, 99, 99}}}),
                                      "Spawn disjoint parameter patch failed"))
                return false;
            if (step == 2 && !Require(owner.spawn.UpdateParameters({{4, {20, 20, 20, 20}}}),
                                      "Spawn repeated CPU value patch failed"))
                return false;
            if (step == 3 &&
                !Require(owner.spawn.UpdateParameters({{0, {40, 40, 40, 40}}}), "Spawn final patch failed"))
                return false;
            if (!Require(owner.spawn.SetEmitterPlaying(0, step == 2) &&
                             owner.spawn.SetEmitterAcceptingBurstRequests(2, frame == 0),
                         "Spawn targeted control update failed"))
                return false;
            owner.runtime.RequestBootstrap();
            owner.spawn.MarkFramePending();
            if (!Require(vkResetFences(resources.context.GetDevice(), 1, &owner.fences[frame]) == VK_SUCCESS,
                         "Spawn fence reset failed"))
                return false;
            vk::VulkanSubmissionExecutor::ExternalSync sync;
            sync.completionEpoch = epochs[frame] = owner.queues.ReserveCompletionEpoch();
            sync.completionFence = owner.fences[frame];
            sync.uploadTimeline = owner.gate;
            sync.uploadTimelineValue = cycle + 1;
            const auto submitted = owner.executor.Execute(
                frame, plan,
                [&](uint32_t, VkCommandBuffer command) {
                    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
                    vk::VulkanTransferCommandContext transferContext;
                    const auto transfer = device.MakeTransferCommandEncoder(transferContext, command);
                    if (!owner.spawn.RecordPendingUploads(transfer))
                        return false;
                    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         0, 1, &barrier, 0, nullptr, 0, nullptr);
                    vk::VulkanComputeCommandContext computeContext;
                    const auto compute = device.MakeComputeCommandEncoder(computeContext, command);
                    const auto poisonMetadata = [&] {
                        compute.BindPipeline(owner.poisonPipeline);
                        compute.BindGroup(owner.poisonPipeline, 0, owner.poisonGroup);
                        compute.Dispatch(1, 1, 1);
                    };
                    // Make first-use correctness independent of allocator contents.
                    if (step == 0) {
                        poisonMetadata();
                        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0,
                                             nullptr);
                    }
                    owner.graph.Execute(command, rhi::QueueRole::Compute);
                    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
                    if (!owner.runtime.RecordBootstrap(compute, 0, owner.spawn.RuntimeGroup(0)))
                        return false;
                    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
                    transfer.CopyBuffer(owner.runtime.StateBuffer(), owner.readback,
                                        {0, frame * bytes, 18 * sizeof(uint32_t)});
                    transfer.CopyBuffer(owner.spawn.BurstRequestAcceptanceBuffer(), owner.readback,
                                        {0, frame * bytes + 18 * sizeof(uint32_t), 3 * sizeof(uint32_t)});
                    transfer.CopyBuffer(owner.spawn.MetadataBuffer(), owner.readback,
                                        {0, frame * bytes + 24 * sizeof(uint32_t), 24 * sizeof(uint32_t)});
                    if (step == 0) {
                        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0,
                                             nullptr);
                        // Later Advance calls must preserve established metadata.
                        poisonMetadata();
                    }
                    // Preroll may author another acceptance value during the
                    // same recording. The first copy must retain its snapshot.
                    barrier.srcAccessMask =
                        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
                    if (!owner.spawn.SetEmitterAcceptingBurstRequests(2, frame != 0) ||
                        !owner.spawn.RecordPendingUploads(transfer))
                        return false;
                    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                                         &barrier, 0, nullptr, 0, nullptr);
                    transfer.CopyBuffer(owner.spawn.BurstRequestAcceptanceBuffer(), owner.readback,
                                        {0, frame * bytes + 21 * sizeof(uint32_t), 3 * sizeof(uint32_t)});
                    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                                         &barrier, 0, nullptr, 0, nullptr);
                    return true;
                },
                sync);
            owner.spawn.NotifySubmission(submitted.Succeeded());
            if (!Require(submitted.Succeeded() && !owner.spawn.HasPendingUploads(), "Spawn input submit failed"))
                return false;
        }
        if (!Require(vkWaitForFences(resources.context.GetDevice(), 2, owner.fences.data(), VK_TRUE, 1'000'000) ==
                         VK_TIMEOUT,
                     "Spawn gate did not retain two in-flight frames"))
            return false;
        VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
        signal.semaphore = owner.gate;
        signal.value = cycle + 1;
        if (!Require(vkSignalSemaphore(resources.context.GetDevice(), &signal) == VK_SUCCESS &&
                         vkWaitForFences(resources.context.GetDevice(), 2, owner.fences.data(), VK_TRUE,
                                         5'000'000'000) == VK_SUCCESS,
                     "Spawn input completion failed"))
            return false;
        for (uint32_t frame = 0; frame < 2; ++frame) {
            owner.executor.CompleteFrame(frame);
            owner.queues.CompleteCompletionEpoch(epochs[frame]);
        }
        const auto *data =
            static_cast<const uint32_t *>(device.MapBuffer(owner.readback, 0, bytes * 2, rhi::BufferMapAccess::Read));
        if (!Require(data != nullptr, "Spawn input readback failed"))
            return false;
        bool matched = true;
        for (uint32_t frame = 0; frame < 2; ++frame) {
            const uint32_t step = cycle * 2 + frame;
            const auto *words = data + frame * wordCount;
            for (uint32_t word = 0; word < 12; ++word)
                matched &= words[word] == expectedParameters[step][word / 4];
            matched &= words[12] == (step == 2 ? 1u : 0u) && words[13] == (step == 0 ? 1u : 0u) && words[14] == 1;
            matched &= words[15] == 0 && words[16] == 0 && words[17] == 0;
            matched &= words[18] == 1 && words[19] == 1 && words[20] == (frame == 0 ? 1u : 0u);
            matched &= words[21] == 1 && words[22] == 1 && words[23] == (frame == 0 ? 0u : 1u);
            for (uint32_t word = 0; word < 24; ++word) {
                const uint32_t expected = step == 0 ? ((word % 8 == 5 || word % 8 == 6) ? 1u : 0u) : 0x12340000u + word;
                if (words[24 + word] != expected) {
                    std::cerr << "Spawn metadata step=" << step << " word=" << word << " actual=" << words[24 + word]
                              << " expected=" << expected << '\n';
                    matched = false;
                }
            }
        }
        const bool unmapped = device.UnmapBuffer(owner.readback, 0, bytes * 2, rhi::BufferMapAccess::Read);
        if (!Require(matched && unmapped, "Spawn inputs crossed frames or overwrote GPU-owned slots"))
            return false;
        std::cout << "Particle spawn input snapshots cycle=" << cycle
                  << " frames=2 GPU parameter/playback mutations preserved, metadata initialized once\n";
    }
    return true;
}
