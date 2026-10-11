bool VerifyParticleRuntimeInputSnapshots(TestResources &resources, infernux::InxShaderLoader &compiler,
                                         const infernux::particle::GpuParticleSpawnProgram &spawnProgram)
{
    using namespace infernux;
    constexpr uint32_t boneCount = 1025;
    constexpr uint32_t wordCount = 64 + 32 + boneCount * 16;
    constexpr uint64_t bytes = wordCount * 4;
    auto &device = resources.context.GetRhiDevice();
    const auto code = SpirvWords(compiler.CompileComputeGlsl(R"glsl(
#version 450
layout(local_size_x=256) in;
layout(std430, set=0, binding=0) writeonly buffer Output { uint result[]; };
layout(std140, set=0, binding=5) uniform Transforms { uvec4 matrices[16]; };
layout(std430, set=1, binding=0) readonly buffer Metadata { uint metadata[]; };
layout(std430, set=1, binding=4) readonly buffer Palette { uint palette[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < 64u) result[i] = matrices[i / 4u][i % 4u];
    else if (i < 96u) result[i] = metadata[i - 64u];
    else if (i < 96u + 1025u * 16u) result[i] = palette[i - 96u];
}
)glsl",
                                                             "Tests/ParticleRuntimeInputs.comp"));
    if (!Require(!code.empty(), "Runtime input shader failed"))
        return false;
    struct Owner
    {
        vk::VkDeviceContext &context;
        vk::VulkanRhiDevice &device;
        vk::VulkanQueueManager queues;
        vk::VulkanSubmissionExecutor executor;
        particle::ParticleGpuRuntime runtime;
        particle::ParticleGpuRuntime replacement;
        particle::ParticleGpuGraphSpawnDomain spawn;
        rhi::BufferHandle geometry, readback;
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
            spawn.Destroy();
            replacement.Destroy();
            runtime.Destroy();
            device.Release(geometry);
            device.Release(readback);
            for (auto fence : fences)
                if (fence)
                    vkDestroyFence(context.GetDevice(), fence, nullptr);
            if (gate)
                vkDestroySemaphore(context.GetDevice(), gate, nullptr);
        }
    } owner{resources.context, device};
    rhi::BufferDesc buffer;
    buffer.byteSize = 256;
    buffer.usage = rhi::BufferUsageFlags::Storage;
    owner.geometry = device.CreateBuffer(buffer);
    buffer.byteSize = bytes * 2;
    buffer.usage = rhi::BufferUsageFlags::TransferDestination;
    buffer.memory = rhi::BufferMemory::Readback;
    buffer.queueAccess = rhi::QueueAccessFlags::Compute;
    owner.readback = device.CreateBuffer(buffer);
    if (!Require(owner.geometry.IsValid() && owner.readback.IsValid() &&
                     owner.queues.Initialize(resources.context, 2) &&
                     owner.executor.Initialize(resources.context, owner.queues, 2),
                 "Runtime input fixture allocation failed"))
        return false;
    particle::GpuEmitterDesc desc;
    desc.capacity = 32768;
    desc.stateStride = 16;
    for (auto &kernel : desc.kernels)
        kernel = {code.data(), code.size()};
    desc.kernels[static_cast<size_t>(particle::GpuKernelStage::UpdateRenderingFused)] = {};
    particle::GpuMeshInterfaceDesc mesh;
    mesh.stableId = "large-skinned-source";
    mesh.vertexBinding = 1;
    mesh.triangleBinding = 2;
    mesh.influenceBinding = 3;
    mesh.paletteBinding = 4;
    mesh.vertexCount = 3;
    mesh.triangleCount = 1;
    mesh.edgeCount = 3;
    mesh.boneCount = boneCount;
    mesh.worldSpace = true;
    mesh.vertices = mesh.triangles = mesh.influences = owner.geometry;
    mesh.vertexBufferBytes = mesh.triangleBufferBytes = mesh.influenceBufferBytes = 256;
    mesh.keepAlive = std::make_shared<int>(1);
    mesh.initialPalette.assign(boneCount, glm::mat4(1));
    desc.meshInterfaces = {mesh};
    if (!Require(owner.runtime.Create(device, desc) && owner.spawn.Create(device, 99502, 1, spawnProgram, {}) &&
                     owner.spawn.RegisterEmitter(0, owner.runtime),
                 "Runtime input creation failed"))
        return false;
    VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    semaphore.pNext = &type;
    if (!Require(vkCreateSemaphore(resources.context.GetDevice(), &semaphore, nullptr, &owner.gate) == VK_SUCCESS,
                 "Runtime input gate failed"))
        return false;
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    for (auto &fence : owner.fences)
        if (!Require(vkCreateFence(resources.context.GetDevice(), &fenceInfo, nullptr, &fence) == VK_SUCCESS,
                     "Runtime input fence failed"))
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
                 "Runtime input plan failed"))
        return false;
    const std::array<float, 16> identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    for (uint32_t cycle = 0; cycle < 2; ++cycle) {
        if (cycle == 1 && !Require(owner.replacement.CreateCompatible(device, desc, owner.runtime) &&
                                       owner.runtime.AdoptCompatibleRevision(owner.replacement),
                                   "Runtime input replacement failed"))
            return false;
        std::array<std::vector<glm::mat4>, 2> expectedPalette;
        std::array<particle::GpuParticleTransforms, 2> expectedTransforms;
        std::array<rhi::SubmissionSerial, 2> epochs{};
        for (uint32_t frame = 0; frame < 2; ++frame) {
            const float value = float(1 + cycle * 2 + frame);
            auto palette = std::make_shared<std::vector<glm::mat4>>(boneCount, glm::mat4(1));
            for (uint32_t bone = 0; bone < boneCount; ++bone)
                (*palette)[bone][3][0] = value * 2000 + float(bone);
            auto source = identity;
            source[3] = value * 10;
            auto &transforms = expectedTransforms[frame];
            transforms = {identity, identity, identity, identity};
            transforms.emitterToWorld[12] = value;
            transforms.worldToEmitter[12] = -value;
            transforms.simulationToWorld[12] = value * 2;
            transforms.worldToSimulation[12] = -value * 2;
            if (!Require(owner.runtime.UpdateSkinnedMeshSources({{0, uint64_t(value), source, palette}}) &&
                             owner.runtime.UpdateTransforms(transforms),
                         "Runtime input preparation failed"))
                return false;
            expectedPalette[frame] = *palette;
            palette.reset();
            owner.runtime.RequestBootstrap();
            if (!Require(vkResetFences(resources.context.GetDevice(), 1, &owner.fences[frame]) == VK_SUCCESS,
                         "Runtime input fence reset failed"))
                return false;
            vk::VulkanSubmissionExecutor::ExternalSync sync;
            sync.completionEpoch = epochs[frame] = owner.queues.ReserveCompletionEpoch();
            sync.completionFence = owner.fences[frame];
            sync.uploadTimeline = owner.gate;
            sync.uploadTimelineValue = cycle + 1;
            const auto submitted = owner.executor.Execute(
                frame, plan,
                [&](uint32_t, VkCommandBuffer commands) {
                    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT |
                                            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                    vkCmdPipelineBarrier(commands,
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                                         &barrier, 0, nullptr, 0, nullptr);
                    vk::VulkanTransferCommandContext transferContext;
                    const auto transfer = device.MakeTransferCommandEncoder(transferContext, commands);
                    if (!owner.runtime.RecordPendingUploads(transfer) || !owner.spawn.RecordPendingUploads(transfer))
                        return false;
                    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
                    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         0, 1, &barrier, 0, nullptr, 0, nullptr);
                    vk::VulkanComputeCommandContext computeContext;
                    const auto compute = device.MakeComputeCommandEncoder(computeContext, commands);
                    if (!owner.runtime.RecordBootstrap(compute, 0, owner.spawn.RuntimeGroup(0)))
                        return false;
                    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         0, 1, &barrier, 0, nullptr, 0, nullptr);
                    transfer.CopyBuffer(owner.runtime.StateBuffer(), owner.readback, {0, frame * bytes, bytes});
                    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                                         &barrier, 0, nullptr, 0, nullptr);
                    return true;
                },
                sync);
            owner.runtime.NotifySubmission(submitted.Succeeded());
            owner.spawn.NotifySubmission(submitted.Succeeded());
            if (!Require(submitted.Succeeded() && !owner.runtime.HasPendingUploads(),
                         "Runtime input submission failed"))
                return false;
        }
        if (!Require(vkWaitForFences(resources.context.GetDevice(), 2, owner.fences.data(), VK_TRUE, 1'000'000) ==
                         VK_TIMEOUT,
                     "Runtime gate failed to keep two frames in flight"))
            return false;
        VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
        signal.semaphore = owner.gate;
        signal.value = cycle + 1;
        if (!Require(vkSignalSemaphore(resources.context.GetDevice(), &signal) == VK_SUCCESS &&
                         vkWaitForFences(resources.context.GetDevice(), 2, owner.fences.data(), VK_TRUE,
                                         5'000'000'000) == VK_SUCCESS,
                     "Runtime input completion failed"))
            return false;
        for (uint32_t frame = 0; frame < 2; ++frame) {
            owner.executor.CompleteFrame(frame);
            owner.queues.CompleteCompletionEpoch(epochs[frame]);
        }
        const auto *data =
            static_cast<const uint32_t *>(device.MapBuffer(owner.readback, 0, bytes * 2, rhi::BufferMapAccess::Read));
        if (!Require(data != nullptr, "Runtime input readback failed"))
            return false;
        bool matched = true;
        for (uint32_t frame = 0; frame < 2; ++frame) {
            const auto *words = data + frame * wordCount;
            matched &= std::memcmp(words, &expectedTransforms[frame], sizeof(particle::GpuParticleTransforms)) == 0;
            matched &= std::memcmp(words + 96, expectedPalette[frame].data(), boneCount * sizeof(glm::mat4)) == 0;
            float sourceX;
            std::memcpy(&sourceX, words + 64 + 16, sizeof(float));
            matched &= sourceX == float(1 + cycle * 2 + frame) * 8;
        }
        const bool unmapped = device.UnmapBuffer(owner.readback, 0, bytes * 2, rhi::BufferMapAccess::Read);
        if (!Require(matched && unmapped,
                     "Runtime inputs were overwritten by a later frame or retained old descriptors"))
            return false;
        std::cout << "Particle runtime input snapshots cycle=" << cycle << " frames=2 bones=1025\n";
    }
    return true;
}
