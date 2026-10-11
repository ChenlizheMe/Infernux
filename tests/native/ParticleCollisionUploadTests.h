// The GPU is held behind a timeline while the next CPU snapshot replaces the
// prior one. Read the actual collision descriptors, including the bulk table.
bool VerifyParticleCollisionUploads(TestResources &resources, infernux::InxShaderLoader &compiler)
{
    using namespace infernux;
    constexpr uint32_t staticCount = 513, colliderCount = staticCount + 1;
    constexpr uint32_t colliderWords = colliderCount * sizeof(particle::GpuParticleColliderRecord) / 4;
    constexpr uint32_t wordCount = 16 + colliderWords + 2 + colliderCount + 12 + 3 + 12;
    constexpr uint64_t byteSize = wordCount * 4;
    auto &device = resources.context.GetRhiDevice();
    particle::ParticleGpuCollisionScene scene;
    if (!Require(scene.Create(device, colliderCount, 3, 3, 1), "Collision upload scene creation failed"))
        return false;
    struct Owner
    {
        vk::VkDeviceContext &context;
        vk::VulkanRhiDevice &device;
        vk::VulkanQueueManager queues;
        vk::VulkanSubmissionExecutor executor;
        rhi::BufferHandle output, readback;
        rhi::BindingLayoutHandle layout;
        rhi::ShaderModuleHandle shader;
        rhi::ComputePipelineHandle pipeline;
        rhi::BindGroupHandle group;
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
            device.Release(group);
            device.Release(pipeline);
            device.Release(shader);
            device.Release(layout);
            device.Release(output);
            device.Release(readback);
            for (auto fence : fences)
                if (fence)
                    vkDestroyFence(context.GetDevice(), fence, nullptr);
            if (gate)
                vkDestroySemaphore(context.GetDevice(), gate, nullptr);
        }
    } owner{resources.context, device};
    if (!Require(owner.queues.Initialize(resources.context, 2) &&
                     owner.executor.Initialize(resources.context, owner.queues, 2),
                 "Collision executor creation failed"))
        return false;
    const auto code = SpirvWords(compiler.CompileComputeGlsl(R"glsl(
#version 450
layout(local_size_x = 64) in;
layout(std430, set=0, binding=0) readonly buffer Header { uint header[]; };
layout(std430, set=0, binding=1) readonly buffer Colliders { uint colliders[]; };
layout(std430, set=0, binding=2) readonly buffer Offsets { uint offsets[]; };
layout(std430, set=0, binding=3) readonly buffer Grid { uint grid[]; };
layout(std430, set=0, binding=4) readonly buffer Vertices { uint vertices[]; };
layout(std430, set=0, binding=5) readonly buffer Indices { uint indices[]; };
layout(std430, set=0, binding=6) readonly buffer Bvh { uint bvh[]; };
layout(std430, set=0, binding=7) writeonly buffer Output { uint result[]; };
void main() {
    uint outIndex = gl_GlobalInvocationID.x, i = outIndex;
    if (i < 16u) { result[outIndex] = header[i]; return; } i -= 16u;
    if (i < 514u * 76u) { result[outIndex] = colliders[i]; return; } i -= 514u * 76u;
    if (i < 2u) { result[outIndex] = offsets[i]; return; } i -= 2u;
    if (i < 514u) { result[outIndex] = grid[i]; return; } i -= 514u;
    if (i < 12u) { result[outIndex] = vertices[i]; return; } i -= 12u;
    if (i < 3u) { result[outIndex] = indices[i]; return; } i -= 3u;
    if (i < 12u) result[outIndex] = bvh[i];
}
)glsl",
                                                             "Tests/ParticleCollisionUpload.comp"));
    if (!Require(!code.empty(), "Collision probe compilation failed"))
        return false;
    rhi::BufferDesc desc;
    desc.byteSize = byteSize;
    desc.queueAccess = rhi::QueueAccessFlags::Compute;
    desc.usage = rhi::BufferUsageFlags::Storage | rhi::BufferUsageFlags::TransferSource;
    owner.output = device.CreateBuffer(desc);
    desc.byteSize *= 2;
    desc.usage = rhi::BufferUsageFlags::TransferDestination;
    desc.memory = rhi::BufferMemory::Readback;
    owner.readback = device.CreateBuffer(desc);
    rhi::BindingLayoutDesc layout;
    layout.entryCount = 8;
    for (uint32_t binding = 0; binding < 8; ++binding)
        layout.entries[binding] = {binding, rhi::BindingType::StorageBuffer, rhi::ShaderStage::Compute, 1};
    owner.layout = device.CreateBindingLayout(layout);
    owner.shader = device.CreateShaderModule(rhi::ShaderModuleDesc::FromSpirV(code.data(), code.size()));
    rhi::ComputePipelineDesc pipeline;
    pipeline.computeShader = owner.shader;
    pipeline.bindingLayouts[0] = owner.layout;
    pipeline.bindingLayoutCount = 1;
    owner.pipeline = device.CreateComputePipeline(pipeline);
    rhi::BindGroupDesc group;
    group.layout = owner.layout;
    group.bufferCount = 8;
    const std::array<rhi::BufferHandle, 8> buffers{
        scene.HeaderBuffer(),     scene.ColliderBuffer(),  scene.GridOffsetBuffer(), scene.GridColliderIndexBuffer(),
        scene.MeshVertexBuffer(), scene.MeshIndexBuffer(), scene.MeshBvhBuffer(),    owner.output};
    const std::array<uint64_t, 8> sizes{64, colliderWords * 4, 8, colliderCount * 4, 48, 12, 48, byteSize};
    for (uint32_t binding = 0; binding < 8; ++binding)
        group.buffers[binding] = {binding, rhi::BindingType::StorageBuffer, buffers[binding], 0, sizes[binding]};
    owner.group = device.CreateBindGroup(group);
    if (!Require(owner.output.IsValid() && owner.readback.IsValid() && owner.pipeline.IsValid() &&
                     owner.group.IsValid(),
                 "Collision probe resources failed"))
        return false;
    VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    semaphore.pNext = &type;
    if (!Require(vkCreateSemaphore(resources.context.GetDevice(), &semaphore, nullptr, &owner.gate) == VK_SUCCESS,
                 "Collision gate creation failed"))
        return false;
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    for (auto &fence : owner.fences)
        if (!Require(vkCreateFence(resources.context.GetDevice(), &fenceInfo, nullptr, &fence) == VK_SUCCESS,
                     "Collision fence creation failed"))
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
                 "Collision submission plan failed"))
        return false;
    for (uint32_t cycle = 0; cycle < 2; ++cycle) {
        std::array<std::vector<uint32_t>, 2> expected;
        std::array<rhi::SubmissionSerial, 2> epochs{};
        for (uint32_t frame = 0; frame < 2; ++frame) {
            particle::GpuParticleCollisionSceneSnapshot snapshot;
            snapshot.revision = 2 + cycle * 4 + frame * 2;
            snapshot.topologyRevision = cycle + 2;
            snapshot.replaceMeshTopology = frame == 0;
            snapshot.staticColliders.resize(staticCount);
            for (uint32_t index = 0; index < staticCount; ++index) {
                auto &record = snapshot.staticColliders[index];
                record.identity[0] = index + 1;
                record.metadata[0] = static_cast<uint32_t>(particle::GpuParticleColliderType::Sphere);
                record.material[0] = float(cycle * 1000 + frame * 100 + index);
            }
            snapshot.staticColliders[0].metadata[0] = static_cast<uint32_t>(particle::GpuParticleColliderType::Mesh);
            particle::GpuParticleCollisionMeshGeometry mesh;
            mesh.identity = 1;
            mesh.positions = {{{-1, float(cycle), -1, 0}}, {{1, float(cycle), -1, 0}}, {{0, float(cycle), 1, 0}}};
            mesh.indices = {0, 1, 2};
            if (snapshot.replaceMeshTopology)
                snapshot.meshGeometries = {mesh};
            if (!Require(scene.Publish(snapshot, &error), "Collision static publication failed"))
                return false;
            snapshot.replaceMeshTopology = false;
            snapshot.meshGeometries.clear();
            ++snapshot.revision;
            snapshot.dynamicColliders.resize(1);
            snapshot.dynamicColliders[0].identity[0] = colliderCount;
            snapshot.dynamicColliders[0].material[0] = float(snapshot.revision);
            if (!Require(scene.Publish(snapshot, &error), "Collision coalesced publication failed"))
                return false;
            // Compare every word of the table, not only its first/last element.
            snapshot.staticColliders[0].geometry = {0, 0, 0, 1};
            expected[frame].resize(colliderWords);
            std::memcpy(expected[frame].data(), snapshot.staticColliders.data(),
                        staticCount * sizeof(particle::GpuParticleColliderRecord));
            std::memcpy(expected[frame].data() + staticCount * 76, snapshot.dynamicColliders.data(),
                        sizeof(particle::GpuParticleColliderRecord));
            if (!Require(vkResetFences(resources.context.GetDevice(), 1, &owner.fences[frame]) == VK_SUCCESS,
                         "Collision fence reset failed"))
                return false;
            vk::VulkanSubmissionExecutor::ExternalSync sync;
            sync.completionEpoch = epochs[frame] = owner.queues.ReserveCompletionEpoch();
            sync.completionFence = owner.fences[frame];
            sync.uploadTimeline = owner.gate;
            sync.uploadTimelineValue = cycle + 1;
            const auto result = owner.executor.Execute(
                frame, plan,
                [&](uint32_t, VkCommandBuffer commands) {
                    VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    before.srcAccessMask =
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
                    before.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                    vkCmdPipelineBarrier(commands,
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                                         &before, 0, nullptr, 0, nullptr);
                    vk::VulkanTransferCommandContext transferContext;
                    const auto transfer = device.MakeTransferCommandEncoder(transferContext, commands);
                    if (!scene.RecordPendingUpload(transfer))
                        return false;
                    VkMemoryBarrier ready{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    ready.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    ready.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         0, 1, &ready, 0, nullptr, 0, nullptr);
                    vk::VulkanComputeCommandContext computeContext;
                    const auto compute = device.MakeComputeCommandEncoder(computeContext, commands);
                    compute.BindPipeline(owner.pipeline);
                    compute.BindGroup(owner.pipeline, 0, owner.group);
                    compute.Dispatch((wordCount + 63) / 64, 1, 1);
                    ready.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                    ready.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         0, 1, &ready, 0, nullptr, 0, nullptr);
                    transfer.CopyBuffer(owner.output, owner.readback, {0, frame * byteSize, byteSize});
                    ready.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    ready.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                                         &ready, 0, nullptr, 0, nullptr);
                    return true;
                },
                sync);
            scene.NotifySubmission(result.Succeeded());
            if (!Require(result.Succeeded() && !scene.HasPendingUpload() &&
                             scene.PublishedRevision() == snapshot.revision,
                         "Collision submission acknowledgement failed"))
                return false;
        }
        if (!Require(vkWaitForFences(resources.context.GetDevice(), 2, owner.fences.data(), VK_TRUE, 1'000'000) ==
                         VK_TIMEOUT,
                     "Collision GPU gate did not hold both frames"))
            return false;
        VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
        signal.semaphore = owner.gate;
        signal.value = cycle + 1;
        if (!Require(vkSignalSemaphore(resources.context.GetDevice(), &signal) == VK_SUCCESS &&
                         vkWaitForFences(resources.context.GetDevice(), 2, owner.fences.data(), VK_TRUE,
                                         5'000'000'000) == VK_SUCCESS,
                     "Collision GPU completion failed"))
            return false;
        for (uint32_t frame = 0; frame < 2; ++frame) {
            owner.executor.CompleteFrame(frame);
            owner.queues.CompleteCompletionEpoch(epochs[frame]);
        }
        const auto *words = static_cast<const uint32_t *>(
            device.MapBuffer(owner.readback, 0, byteSize * 2, rhi::BufferMapAccess::Read));
        if (!Require(words != nullptr, "Collision readback mapping failed"))
            return false;
        bool matched = true;
        for (uint32_t frame = 0; frame < 2; ++frame) {
            const auto *data = words + wordCount * frame;
            matched &= data[0] == colliderCount && data[1] == staticCount && data[2] == 3 + cycle * 4 + frame * 2;
            matched &= data[12] == 3 && data[13] == 3 && data[14] == 1 && data[15] == cycle + 2;
            matched &= std::equal(expected[frame].begin(), expected[frame].end(), data + 16);
            const auto *grid = data + 16 + colliderWords;
            matched &= grid[0] == 0 && grid[1] == colliderCount;
            for (uint32_t index = 0; index < colliderCount; ++index)
                matched &= grid[index + 2] == index;
            const std::array<float, 12> vertices{-1, float(cycle), -1, 0, 1, float(cycle), -1, 0,
                                                 0,  float(cycle), 1,  0};
            const auto *topology = grid + 2 + colliderCount;
            matched &= std::memcmp(topology, vertices.data(), sizeof(vertices)) == 0;
            matched &= topology[12] == 0 && topology[13] == 1 && topology[14] == 2;
            matched &= topology[15 + 10] == 0 && topology[15 + 11] == 1;
        }
        const bool unmapped = device.UnmapBuffer(owner.readback, 0, byteSize * 2, rhi::BufferMapAccess::Read);
        if (!Require(matched && unmapped, "GPU collision descriptors contained stale or overwritten data"))
            return false;
        std::cout << "Particle collision immutable uploads cycle=" << cycle << " frames=2 colliders=514\n";
    }
    return true;
}
