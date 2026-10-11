bool VerifyParticleDomainRetention(TestResources &resources, infernux::InxShaderLoader &compiler,
                                   const infernux::particle::GpuParticleEmitterProgram &prototype,
                                   const infernux::particle::GpuParticleSortProgram &sortProgram,
                                   const infernux::particle::GpuParticleCullProgram &cullProgram,
                                   const infernux::particle::GpuParticleBoundsProgram &boundsProgram,
                                   const infernux::particle::GpuParticleMigrationProgram &migrationProgram,
                                   const infernux::particle::GpuParticleSpawnProgram &spawnProgram)
{
    using namespace infernux;
    using namespace infernux::particle;
    const auto noop = SpirvWords(compiler.CompileComputeGlsl(
        "#version 450\nlayout(local_size_x=256) in; void main() {}", "Tests/DomainRetentionNoop.comp"));
    const auto bootstrap = SpirvWords(compiler.CompileComputeGlsl(R"glsl(
#version 450
layout(local_size_x=256) in;
layout(std430, set=0, binding=2) buffer Counters { uint counters[]; };
layout(std430, set=0, binding=4) buffer Indirect { uvec4 indirectArgs; };
layout(std430, set=0, binding=9) buffer Dispatch { uvec4 dispatches[]; };
void main() {
    if (gl_GlobalInvocationID.x != 0u) return;
    counters[0] = 32u;
    indirectArgs = uvec4(0u);
    dispatches[0] = dispatches[1] = uvec4(1u, 1u, 1u, 0u);
}
)glsl",
                                                                  "Tests/DomainRetentionBootstrap.comp"));
    const auto update = SpirvWords(compiler.CompileComputeGlsl(R"glsl(
#version 450
layout(local_size_x=256) in;
layout(std430, set=0, binding=0) buffer States { uvec4 states[]; };
layout(std430, set=3, binding=0) buffer Burst { uint bursts[]; };
layout(std430, set=3, binding=1) readonly buffer Metadata { uint metadata[]; };
layout(std430, set=3, binding=2) buffer Parameters { uvec4 parameters[]; };
layout(std430, set=3, binding=3) buffer Requests { uint requests[]; };
layout(std430, set=3, binding=4) readonly buffer Playing { uint playing[]; };
void main() {
    if (gl_GlobalInvocationID.x != 0u) return;
    uint previous = parameters[0].x;
    states[0] = uvec4(previous, playing[0], metadata[0], metadata[7]);
    parameters[0].x += 10u;
    requests[0] = previous == 100u ? 1u : (previous == 110u ? 0u : 2u);
    if (bursts[0] != 0xffffffffu) bursts[0] += previous == 100u ? 7u : 5u;
}
)glsl",
                                                               "Tests/DomainRetentionUpdate.comp"));
    const auto rendering = SpirvWords(compiler.CompileComputeGlsl(R"glsl(
#version 450
layout(local_size_x=256) in;
layout(std430, set=0, binding=0) readonly buffer States { uvec4 states[]; };
layout(std430, set=0, binding=3) writeonly buffer Instances { uvec4 snapshots[]; };
void main() {
    if (gl_GlobalInvocationID.x == 0u) snapshots[0] = states[0];
}
)glsl",
                                                                  "Tests/DomainRetentionRendering.comp"));
    const auto readbackCode = SpirvWords(compiler.CompileComputeGlsl(R"glsl(
#version 450
layout(local_size_x=1) in;
layout(std430, set=0, binding=0) readonly buffer Input { uvec4 snapshot; };
layout(std430, set=0, binding=1) writeonly buffer Output { uvec4 result; };
void main() { result = snapshot; }
)glsl",
                                                                     "Tests/DomainRetentionReadback.comp"));
    if (!Require(!noop.empty() && !bootstrap.empty() && !update.empty() && !rendering.empty() && !readbackCode.empty(),
                 "Particle domain retention shaders failed"))
        return false;

    struct Owner
    {
        vk::VkDeviceContext &context;
        GpuRetirementQueue retirement;
        ParticleGpuDrawRegistry draws;
        ParticleGpuSystemManager manager;
        rhi::BufferHandle readback;
        rhi::BindingLayoutHandle readbackLayout;
        rhi::ShaderModuleHandle readbackShader;
        rhi::ComputePipelineHandle readbackPipeline;
        rhi::BindGroupHandle readbackGroup;
        VkCommandPool pool = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        ~Owner()
        {
            context.WaitIdle();
            manager.Shutdown();
            retirement.FlushAll();
            context.GetRhiDevice().Release(readbackGroup);
            context.GetRhiDevice().Release(readbackPipeline);
            context.GetRhiDevice().Release(readbackShader);
            context.GetRhiDevice().Release(readbackLayout);
            context.GetRhiDevice().Release(readback);
            if (fence)
                vkDestroyFence(context.GetDevice(), fence, nullptr);
            if (pool)
                vkDestroyCommandPool(context.GetDevice(), pool, nullptr);
        }
    } owner{resources.context};
    owner.retirement.BindSerialSource([] { return rhi::SubmissionSerial{1}; });
    if (!Require(owner.manager.Initialize(resources.context, resources.pipelines, resources.resources, owner.retirement,
                                          owner.draws, {}, {}, {}, sortProgram, cullProgram, boundsProgram,
                                          migrationProgram, spawnProgram),
                 "Particle domain retention manager failed"))
        return false;
    auto a = prototype;
    a.id = 99601;
    a.graphInstanceId = 9961;
    a.stableId = "retained";
    a.parameterWords = {100, 0, 0, 0};
    a.outputs.front().id = 996011;
    for (auto &kernel : a.kernels)
        kernel = noop;
    a.kernels[static_cast<size_t>(GpuKernelStage::Bootstrap)] = bootstrap;
    a.kernels[static_cast<size_t>(GpuKernelStage::Update)] = update;
    a.kernels[static_cast<size_t>(GpuKernelStage::Rendering)] = rendering;
    a.kernels[static_cast<size_t>(GpuKernelStage::UpdateRenderingFused)].clear();
    auto b = a;
    b.id = 99602;
    b.graphInstanceId = 9962;
    b.stableId = "unrelated";
    b.outputs.front().id = 996021;
    std::string error;
    const auto publish = [&](const GpuParticleEmitterProgram &program) {
        return owner.manager.ApplyGraph({program.graphInstanceId, {program}, {}}, &error);
    };
    const auto removeB = [&] { return owner.manager.ApplyGraph({b.graphInstanceId, {}, {b.id}}, &error); };
    if (!Require(publish(a), error.c_str()))
        return false;
    const auto entries = owner.draws.SnapshotShared(0, 5000);
    const auto instances = entries->front().instances;
    auto &device = resources.context.GetRhiDevice();
    rhi::BufferDesc readback;
    readback.byteSize = 16;
    readback.usage = rhi::BufferUsageFlags::Storage;
    readback.memory = rhi::BufferMemory::Readback;
    owner.readback = device.CreateBuffer(readback);
    rhi::BindingLayoutDesc readbackLayout;
    readbackLayout.entryCount = 2;
    for (uint32_t binding = 0; binding < 2; ++binding)
        readbackLayout.entries[binding] = {binding, rhi::BindingType::StorageBuffer, rhi::ShaderStage::Compute, 1};
    owner.readbackLayout = device.CreateBindingLayout(readbackLayout);
    owner.readbackShader =
        device.CreateShaderModule(rhi::ShaderModuleDesc::FromSpirV(readbackCode.data(), readbackCode.size()));
    rhi::ComputePipelineDesc readbackPipeline;
    readbackPipeline.computeShader = owner.readbackShader;
    readbackPipeline.bindingLayoutCount = 1;
    readbackPipeline.bindingLayouts[0] = owner.readbackLayout;
    owner.readbackPipeline = device.CreateComputePipeline(readbackPipeline);
    rhi::BindGroupDesc readbackGroup;
    readbackGroup.layout = owner.readbackLayout;
    readbackGroup.bufferCount = 2;
    readbackGroup.buffers[0] = {0, rhi::BindingType::StorageBuffer, instances, 0, 16};
    readbackGroup.buffers[1] = {1, rhi::BindingType::StorageBuffer, owner.readback, 0, 16};
    owner.readbackGroup = device.CreateBindGroup(readbackGroup);
    VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool.queueFamilyIndex = resources.context.GetQueueIndices().graphicsFamily.value();
    pool.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (!Require(owner.readback.IsValid() && owner.readbackPipeline.IsValid() && owner.readbackGroup.IsValid() &&
                     vkCreateCommandPool(resources.context.GetDevice(), &pool, nullptr, &owner.pool) == VK_SUCCESS &&
                     vkCreateFence(resources.context.GetDevice(), &fence, nullptr, &owner.fence) == VK_SUCCESS,
                 "Particle domain retention submission resources failed"))
        return false;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = owner.pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    if (!Require(vkAllocateCommandBuffers(resources.context.GetDevice(), &allocate, &command) == VK_SUCCESS,
                 "Particle domain retention command allocation failed"))
        return false;
    const auto execute = [&](const std::array<uint32_t, 4> &expected, const char *label) {
        const auto vkDevice = resources.context.GetDevice();
        if (!Require(vkResetCommandPool(vkDevice, owner.pool, 0) == VK_SUCCESS &&
                         vkResetFences(vkDevice, 1, &owner.fence) == VK_SUCCESS,
                     "Particle domain retention submission reset failed"))
            return false;
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (!Require(vkBeginCommandBuffer(command, &begin) == VK_SUCCESS,
                     "Particle domain retention command begin failed"))
            return false;
        // Serialize the test readback against the next simulation. No host
        // waits or direct inspection of manager internals enter production.
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &barrier, 0, nullptr, 0, nullptr);
        owner.manager.Execute(command);
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &barrier, 0, nullptr, 0, nullptr);
        vk::VulkanComputeCommandContext computeContext;
        const auto compute = device.MakeComputeCommandEncoder(computeContext, command);
        compute.BindPipeline(owner.readbackPipeline);
        compute.BindGroup(owner.readbackPipeline, 0, owner.readbackGroup);
        compute.Dispatch(1, 1, 1);
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier,
                             0, nullptr, 0, nullptr);
        if (!Require(vkEndCommandBuffer(command) == VK_SUCCESS, "Particle domain retention command end failed"))
            return false;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        const auto submitted = vkQueueSubmit(resources.context.GetGraphicsQueue(), 1, &submit, owner.fence);
        owner.manager.NotifySubmission(submitted == VK_SUCCESS);
        if (!Require(submitted == VK_SUCCESS &&
                         vkWaitForFences(vkDevice, 1, &owner.fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS,
                     "Particle domain retention GPU submission failed"))
            return false;
        owner.retirement.FlushAll();
        std::array<uint32_t, 4> actual{};
        if (!Require(device.ReadBuffer(owner.readback, 0, actual.data(), sizeof(actual)),
                     "Particle domain retention readback failed"))
            return false;
        if (actual != expected)
            std::cerr << label << ": actual=" << actual[0] << ',' << actual[1] << ',' << actual[2] << ',' << actual[3]
                      << " expected=" << expected[0] << ',' << expected[1] << ',' << expected[2] << ',' << expected[3]
                      << '\n';
        return Require(actual == expected, label);
    };
    GpuParticleTransforms transforms;
    GpuParticleFrameRequest frame;
    frame.frameIndex = 100;
    frame.boundsMode = GpuParticleBoundsMode::Manual;
    frame.manualBoundsLower = {-1, -1, -1};
    frame.manualBoundsUpper = {1, 1, 1};
    const auto beginNext = [&] {
        ++frame.frameIndex;
        return owner.manager.BeginFrame(a.id, frame, transforms);
    };
    if (!Require(owner.manager.BeginFrame(a.id, frame, transforms), "Initial retained frame rejected") ||
        !execute({100, 1, 0, 0}, "Initial particle control state incorrect"))
        return false;
    if (!Require(beginNext() && publish(b), "Unrelated graph creation failed") ||
        !execute({110, 0, 0, 0}, "Creation lost pending frame or GPU parameter/stop state") ||
        !Require(!owner.manager.BeginFrame(a.id, frame, transforms), "Rebuild lost consumed frame history"))
        return false;
    ++b.artifactRevision;
    if (!Require(publish(b) && !owner.manager.BeginFrame(a.id, frame, transforms),
                 "Idle rebuild allowed an already consumed frame to replay") ||
        !Require(beginNext(), "Retained stopped frame rejected") ||
        !execute({120, 0, 0, 0}, "Idle rebuild lost GPU playing state without a pending request"))
        return false;
    if (!Require(owner.manager.UpdateGraphParameters(a.graphInstanceId, {{0, {500, 0, 0, 0}}}) && beginNext() &&
                     removeB(),
                 "Unrelated removal setup failed") ||
        !execute({500, 1, 0, 0}, "Removal lost pending parameter upload or GPU play request"))
        return false;
    if (!Require(beginNext() && publish(b), "Unrelated recreation failed") ||
        !execute({510, 1, 5, 5}, "Recreation lost queued GPU burst"))
        return false;
    ++b.artifactRevision;
    if (!Require(beginNext() && publish(b), "Unrelated compatible update failed") ||
        !execute({520, 1, 5, 10}, "Unrelated update reset accumulated spawn metadata"))
        return false;
    auto invalid = b;
    ++invalid.artifactRevision;
    invalid.outputs.front().id = a.outputs.front().id;
    const auto registryRevision = owner.draws.Revision();
    if (!Require(beginNext() && !publish(invalid) && error == "failed to publish GPU particle draw entries" &&
                     owner.draws.Revision() == registryRevision,
                 "Draw collision did not reject the built candidate graph atomically") ||
        !execute({530, 1, 5, 15}, "Rejected graph publication damaged resident graph state"))
        return false;

    ++frame.frameIndex;
    auto first = frame;
    first.render = false;
    auto second = first;
    second.substepIndex = 1;
    frame.substepIndex = 2;
    if (!Require(owner.manager.BeginFrameBatch(a.graphInstanceId, {{a.id, {first, second}, frame, transforms}}),
                 "Retained preroll batch rejected") ||
        !execute({530, 1, 5, 15}, "Preroll unexpectedly rendered") ||
        !Require(removeB(), "Unrelated removal between queued substeps failed") ||
        !execute({530, 1, 5, 15}, "Rebuilt preroll unexpectedly rendered") ||
        !execute({560, 1, 5, 30}, "Rebuild lost a queued simulation step") ||
        !Require(!owner.manager.BeginFrame(a.id, first, transforms), "Queued rebuild allowed frame replay"))
        return false;
    frame.substepIndex = 0;
    if (!Require(owner.manager.Reset(a.id) && publish(b) && beginNext(), "Retained reset setup failed") ||
        !execute({570, 1, 0, 0}, "Unrelated rebuild lost pending reset or reset persistent parameters"))
        return false;
    std::cout << "Particle domain retention: create/remove/update/rejected publication, pending uploads, "
                 "GPU play/bursts, queued substeps and reset passed\n";
    return true;
}
