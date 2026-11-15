// Production manager -> frame composer -> two graphics consumers, with CPU
// authoring of frame N+1 while both submissions are held behind a GPU gate.
bool VerifyParticleFrameSnapshots(TestResources &resources,
                                  const infernux::particle::GpuParticleEmitterProgram &prototype,
                                  const infernux::particle::GpuParticleSortProgram &sortProgram,
                                  const infernux::particle::GpuParticleCullProgram &cullProgram,
                                  const infernux::particle::GpuParticleBoundsProgram &boundsProgram,
                                  const infernux::particle::GpuParticleMigrationProgram &migrationProgram,
                                  const infernux::particle::GpuParticleSpawnProgram &spawnProgram)
{
    using namespace infernux;
    using namespace infernux::particle;
    auto &registry = AssetRegistry::Instance();
    registry.Initialize(std::make_unique<AssetDatabase>());
    registry.RegisterLoader(ResourceType::Mesh, std::make_unique<MeshLoader>());
    auto &scenes = SceneManager::Instance();
    scenes.Stop();
    scenes.UnloadAllScenes();
    struct SceneOwner
    {
        ~SceneOwner()
        {
            SceneManager::Instance().Stop();
            SceneManager::Instance().UnloadAllScenes();
            AssetRegistry::Instance().Shutdown();
        }
    } sceneOwner;
    auto model = std::make_shared<InxSkinnedMesh>();
    model->scaleFactor = 1.0f;
    model->baseVertices.resize(3);
    model->baseVertices[0].pos = {0, 0, 0};
    model->baseVertices[1].pos = {1, 0, 0};
    model->baseVertices[2].pos = {0, 1, 0};
    model->indices = {0, 1, 2};
    model->influences.resize(3);
    for (auto &influence : model->influences) {
        influence.boneIndex[0] = 0;
        influence.weight[0] = 1.0f;
    }
    SkinnedRuntimeNode node;
    node.name = "Root";
    model->skeleton.nodes.push_back(node);
    model->skeleton.nodeByName.emplace("Root", 0);
    SkinnedRuntimeBone bone;
    bone.name = "Root";
    bone.nodeIndex = 0;
    model->skeleton.bones.push_back(bone);
    model->skeleton.boneByName.emplace("Root", 0);
    SkinnedRuntimeTrack track;
    track.nodeIndex = 0;
    track.positions = {{0.0, {0, 0, 0}}, {2.0, {2, 0, 0}}};
    SkinnedRuntimeAnimation animation;
    animation.name = "Move";
    animation.durationTicks = 2.0;
    animation.ticksPerSecond = 1.0;
    animation.trackByNodeIndex = {0};
    animation.tracks.push_back(track);
    model->animations.push_back(animation);
    auto mesh = registry.CreateRuntimeMesh("FrameSnapshotSource");
    InxMesh replacement("FrameSnapshotSource");
    replacement.SetData(model->baseVertices, model->indices, {});
    replacement.SetSkinnedData(model);
    registry.PublishMesh(mesh->GetGuid(), std::move(replacement));
    auto *activeScene = scenes.CreateScene("SnapshotActive");
    auto *sourceScene = scenes.CreateScene("SnapshotSource");
    auto *object = sourceScene->CreateGameObject("MovingSkin");
    auto *skin = object->AddComponent<SkinnedMeshRenderer>();
    skin->SetSourceModelGuid(mesh->GetGuid());
    skin->SetActiveTakeName("Move");
    scenes.SetActiveScene(activeScene);

    const auto noop = CompileTestCompute("[shader(\"compute\")] [numthreads(256,1,1)] void main() {}",
                                         "Tests/FrameSnapshotNoop.comp.slang");
    const auto bootstrap = CompileTestCompute(R"slang(

[[vk::binding(0, 0)]] RWStructuredBuffer<uint4> states;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> counters;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint4> indirectArgs;
[[vk::binding(9, 0)]] RWStructuredBuffer<uint4> dispatches;
[shader("compute")] [numthreads(256, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    if (dispatch_id.x < 32u) states[dispatch_id.x] = uint4(0u);
    if (dispatch_id.x != 0u) return;
    counters[0] = 32u;
    indirectArgs[0] = uint4(0u);
    dispatches[0] = dispatches[1] = uint4(1u, 1u, 1u, 0u);
}
)slang",
                                              "Tests/FrameSnapshotBootstrap.comp.slang");
    const auto rendering = CompileTestCompute(R"slang(

[[vk::binding(3, 0)]] RWStructuredBuffer<float4> snapshots;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint4> indirectArgs;
struct Transforms {  float4x4 transforms[4];  };
[[vk::binding(5, 0)]] ConstantBuffer<Transforms> test_transforms;
[[vk::binding(0, 1)]] StructuredBuffer<uint> metadata;
[[vk::binding(4, 1)]] StructuredBuffer<float4x4> bones;
[[vk::binding(2, 3)]] StructuredBuffer<uint4> parameters;
[shader("compute")] [numthreads(256, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    if (dispatch_id.x != 0u) return;
    snapshots[0] = float4(bones[0][3].x, asfloat(metadata[16]),
                        asfloat(parameters[0].x), test_transforms.transforms[0][3].x);
    indirectArgs[0] = uint4(6u, 1u, 0u, 0u);
}
)slang",
                                              "Tests/FrameSnapshotRendering.comp.slang");
    const auto geometry = std::make_shared<const ShaderCompilation>(SlangCompiler::Get().Compile(
        {"particle_snapshot_geometry", "Tests/ParticleFrameSnapshot.slang", R"slang(import inx_particle;
[[vk::binding(1, 2)]] StructuredBuffer<float4> instances;
[shader("vertex")]
InxParticleInput vertex_main(uint vertex : SV_VulkanVertexID, uint instance : SV_VulkanInstanceID)
{
    float2 positions[3] = {float2(-1,-1), float2(3,-1), float2(-1,3)};
    InxParticleInput result = (InxParticleInput)0;
    result.position = float4(positions[vertex % 3], 0.5, 1.0);
    float4 snapshot = instances[instance * 7] + float4(inx_particle_draw().camera_right.x, 0, 0, 0);
    result.color = snapshot.rgb;
    result.alpha = snapshot.a;
    return result;
}
)slang"},
        {}, {ShaderCodeTarget::SpirV, ShaderCodeTarget::Wgsl}));
    auto graphicsPrograms = particle_test::GeometryPrograms();
    graphicsPrograms[0] = geometry;
    if (!Require(!noop.empty() && !bootstrap.empty() && !rendering.empty(), "Particle frame snapshot shaders failed"))
        return false;
    auto &device = resources.context.GetRhiDevice();
    struct Owner
    {
        vk::VkDeviceContext &context;
        vk::VulkanQueueManager queues;
        vk::VulkanSubmissionExecutor executor;
        GpuRetirementQueue retirement;
        ParticleGpuDrawRegistry draws;
        ParticleGpuSystemManager manager;
        std::unique_ptr<particle_test::ViewResources> perView;
        std::array<rhi::TextureHandle, 4> targets{};
        std::array<rhi::TextureViewHandle, 4> views{};
        rhi::BufferHandle readback;
        VkSemaphore gate = VK_NULL_HANDLE;
        std::array<VkFence, 2> fences{};
        ~Owner()
        {
            if (gate) {
                VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
                signal.semaphore = gate;
                signal.value = 999;
                vkSignalSemaphore(context.GetDevice(), &signal);
            }
            context.WaitIdle();
            executor.Destroy();
            manager.Shutdown();
            retirement.FlushAll();
            queues.Destroy();
            auto &device = context.GetRhiDevice();
            perView.reset();
            for (auto view : views)
                device.Release(view);
            for (auto target : targets)
                device.Release(target);
            device.Release(readback);
            for (auto fence : fences)
                if (fence)
                    vkDestroyFence(context.GetDevice(), fence, nullptr);
            if (gate)
                vkDestroySemaphore(context.GetDevice(), gate, nullptr);
        }
    } owner{resources.context};
    if (!Require(owner.queues.Initialize(resources.context, 2) &&
                     owner.executor.Initialize(resources.context, owner.queues, 2) &&
                     owner.manager.Initialize(resources.context, resources.pipelines, resources.resources,
                                              owner.retirement, owner.draws, {}, {}, ResolveSceneSkinnedMeshSource,
                                              sortProgram, cullProgram, boundsProgram, migrationProgram, spawnProgram,
                                              {}, {}, graphicsPrograms),
                 "Particle frame snapshot manager initialization failed"))
        return false;
    owner.retirement.BindSerialSource([&] { return owner.queues.GetLastReservedCompletionEpoch(); });
    owner.perView = std::make_unique<particle_test::ViewResources>(device);
    rhi::BufferDesc readback;
    readback.byteSize = 4 * sizeof(glm::vec4);
    readback.usage = rhi::BufferUsageFlags::TransferDestination;
    readback.memory = rhi::BufferMemory::Readback;
    owner.readback = device.CreateBuffer(readback);
    for (size_t index = 0; index < owner.targets.size(); ++index) {
        rhi::TextureDesc target;
        target.format = rhi::PixelFormat::RGBA32SFloat;
        target.usage = rhi::TextureUsageFlags::ColorAttachment | rhi::TextureUsageFlags::TransferSource;
        owner.targets[index] = device.CreateTexture(target);
        rhi::TextureViewDesc view;
        view.texture = owner.targets[index];
        view.format = target.format;
        owner.views[index] = device.CreateTextureView(view);
        if (!Require(owner.targets[index].IsValid() && owner.views[index].IsValid(), "Snapshot target failed"))
            return false;
    }
    VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    semaphore.pNext = &type;
    if (!Require(owner.readback.IsValid() && owner.perView->bindings.group.IsValid() &&
                     vkCreateSemaphore(resources.context.GetDevice(), &semaphore, nullptr, &owner.gate) == VK_SUCCESS,
                 "Snapshot gate/resources failed"))
        return false;
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    for (auto &fence : owner.fences)
        if (!Require(vkCreateFence(resources.context.GetDevice(), &fenceInfo, nullptr, &fence) == VK_SUCCESS,
                     "Snapshot fence failed"))
            return false;
    auto material = std::make_shared<InxMaterial>("Frame snapshots");
    auto materialState = material->GetRenderState();
    materialState.blendEnable = materialState.depthWriteEnable = false;
    materialState.cullMode = MaterialCullMode::None;
    material->SetRenderState(materialState);
    material->SetVector4("color", glm::vec4(0));
    const auto artifact = particle_test::Surface();
    auto program = prototype;
    program.id = 99801;
    program.graphInstanceId = 9981;
    program.outputs.resize(1);
    program.outputs[0].id = 998011;
    program.outputs[0].shaderProgram = artifact;
    program.outputs[0].material = material;
    program.outputs[0].semantics.sortMode = ParticleSortMode::None;
    program.parameterWords = {0, 0, 0, 0};
    for (auto &kernel : program.kernels)
        kernel = noop;
    program.kernels[static_cast<size_t>(GpuKernelStage::Bootstrap)] = bootstrap;
    program.kernels[static_cast<size_t>(GpuKernelStage::Rendering)] = rendering;
    program.kernels[static_cast<size_t>(GpuKernelStage::UpdateRenderingFused)].clear();
    GpuParticleMeshInterfaceProgram meshInterface;
    meshInterface.stableId = "moving-source";
    meshInterface.mesh = mesh;
    meshInterface.skinnedRendererHandle = BindSceneSkinnedMeshSource(*skin);
    program.meshInterfaces = {meshInterface};
    GpuParticleGraphProgram graph;
    graph.graphInstanceId = program.graphInstanceId;
    graph.emitters = {program};
    std::string error;
    if (!resources.PublishParticleGraph(owner.manager, graph, &error)) {
        std::cerr << error << '\n';
        return Require(false, "Particle frame snapshot publication failed");
    }
    const auto entries = owner.draws.SnapshotShared(0, 5000);
    if (!Require(entries->size() == 1, "Particle frame snapshot draw not published"))
        return false;
    const auto &entry = entries->front();
    MaterialPassPipelineDescriptor pass;
    pass.colorFormats = {rhi::PixelFormat::RGBA32SFloat};
    const auto dynamicRendering = rhi::ResolveDynamicRenderingCommands(resources.context.GetDevice());
    if (!Require(dynamicRendering.IsValid(), "Particle snapshot dynamic rendering unavailable"))
        return false;
    vk::VulkanSubmissionExecutor::ExecuteResult previous;
    const std::array<float, 16> identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    const bool independent = resources.context.HasIndependentComputeQueue();
    // Inline Graphics, split compute prime, then steady asynchronous export.
    const uint32_t cycles = independent ? 3 : 1;
    for (uint32_t cycle = 0; cycle < cycles; ++cycle) {
        std::array<glm::vec4, 4> expected{};
        std::array<rhi::SubmissionSerial, 2> epochs{};
        for (uint32_t frame = 0; frame < 2; ++frame) {
            const uint32_t sequence = cycle * 2 + frame + 1;
            const float pose = float(sequence) * 0.125f;
            skin->SetRuntimeAnimationTime(pose);
            object->GetTransform()->SetPosition({10.0f + float(sequence), 0, 0});
            GpuParticleTransforms transforms{identity, identity, identity, identity};
            transforms.emitterToWorld[12] = float(sequence);
            transforms.worldToEmitter[12] = -float(sequence);
            GpuParticleFrameRequest request;
            request.frameIndex = sequence;
            request.simulationStep = sequence;
            request.deltaTime = 1.0f / 60.0f;
            request.boundsMode = GpuParticleBoundsMode::Manual;
            request.manualBoundsLower = {-1, -1, -1};
            request.manualBoundsUpper = {1, 1, 1};
            const float parameter = float(sequence) * 3.0f;
            uint32_t parameterWord;
            std::memcpy(&parameterWord, &parameter, sizeof(parameter));
            if (!Require(
                    owner.manager.UpdateGraphParameters(program.graphInstanceId, {{0, {parameterWord, 0, 0, 0}}}) &&
                        owner.manager.BeginFrame(program.id, request, transforms) &&
                        (cycle == 0 || owner.manager.CanRecordPartitioned()),
                    "Snapshot frame preparation failed"))
                return false;
            const glm::vec4 tint{pose * 0.5f, 0.25f, 0.5f, 1.0f};
            material->SetVector4("color", tint);
            const uint32_t renderedSequence = cycle == 2 ? sequence - 1 : sequence;
            const glm::vec4 snapshot{float(renderedSequence) * 0.125f, 10.0f + float(renderedSequence),
                                     float(renderedSequence) * 3.0f, float(renderedSequence)};
            vk::VulkanFrameSubmission submission;
            uint32_t simulation = 0;
            if (cycle != 2) {
                simulation = submission.AddWork(
                    resources.context.GetDeviceId(), cycle == 0 ? rhi::QueueRole::Graphics : rhi::QueueRole::Compute,
                    rhi::SubmissionDomain::Frame, rhi::InvalidRenderViewId, rhi::PipelineStage::AllCommands, {},
                    [&](VkCommandBuffer commands) {
                        if (cycle != 0)
                            return owner.manager.RecordAsyncSimulation(commands);
                        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                        barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
                        barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
                        vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0,
                                             nullptr);
                        owner.manager.Execute(commands);
                        return true;
                    },
                    "Snapshot/Simulation");
                if (cycle == 1)
                    simulation = submission.AddWork(
                        resources.context.GetDeviceId(), rhi::QueueRole::Compute, rhi::SubmissionDomain::Frame,
                        rhi::InvalidRenderViewId, rhi::PipelineStage::AllCommands, {simulation},
                        [&](VkCommandBuffer commands) { return owner.manager.RecordAsyncExport(commands); },
                        "Snapshot/PrimeExport");
            }
            uint32_t lastView = 0;
            for (uint32_t viewIndex = 0; viewIndex < 2; ++viewIndex) {
                const size_t index = frame * 2 + viewIndex;
                const float viewMarker = 100.0f * float(viewIndex + 1);
                expected[index] = snapshot + tint + glm::vec4(viewMarker, 0, 0, 0);
                lastView = submission.AddWork(
                    resources.context.GetDeviceId(), rhi::QueueRole::Graphics, rhi::SubmissionDomain::Frame,
                    viewIndex + 1, rhi::PipelineStage::AllCommands,
                    simulation ? std::vector<uint32_t>{simulation} : std::vector<uint32_t>{},
                    [&, index, viewMarker](VkCommandBuffer commands) {
                        VkMemoryBarrier read{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                        read.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                        read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
                        vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                                             0, 1, &read, 0, nullptr, 0, nullptr);
                        VkImageMemoryBarrier image{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                        image.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                        image.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                        image.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                        image.srcQueueFamilyIndex = image.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                        image.image = device.Resolve(owner.targets[index]);
                        image.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                        vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr,
                                             1, &image);
                        VkRenderingAttachmentInfo attachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
                        attachment.imageView = device.Resolve(owner.views[index]);
                        attachment.imageLayout = image.newLayout;
                        attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                        attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                        VkRenderingInfo info{VK_STRUCTURE_TYPE_RENDERING_INFO};
                        info.renderArea.extent = {1, 1};
                        info.layerCount = 1;
                        info.colorAttachmentCount = 1;
                        info.pColorAttachments = &attachment;
                        dynamicRendering.begin(commands, &info);
                        VkViewport viewport{0, 0, 1, 1, 0, 1};
                        VkRect2D scissor{{0, 0}, {1, 1}};
                        vkCmdSetViewport(commands, 0, 1, &viewport);
                        vkCmdSetScissor(commands, 0, 1, &scissor);
                        vk::VulkanGraphicsCommandContext context;
                        const auto graphics = device.MakeGraphicsCommandEncoder(context, commands);
                        GpuParticleViewConstants view;
                        view.cameraRight[0] = viewMarker;
                        const bool drawn = entry.renderer->RecordDraw(graphics, pass, entry.indirectArguments, view,
                                                                      owner.perView->bindings);
                        dynamicRendering.end(commands);
                        if (!drawn)
                            return false;
                        image.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                        image.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                        image.oldLayout = image.newLayout;
                        image.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                        vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &image);
                        VkBufferImageCopy copy{};
                        copy.bufferOffset = index * sizeof(glm::vec4);
                        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                        copy.imageExtent = {1, 1, 1};
                        vkCmdCopyImageToBuffer(commands, image.image, image.newLayout, device.Resolve(owner.readback),
                                               1, &copy);
                        read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                        read.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                        vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                                             &read, 0, nullptr, 0, nullptr);
                        return true;
                    },
                    "Snapshot/View" + std::to_string(viewIndex));
            }
            if (cycle == 2) {
                simulation = submission.AddWork(
                    resources.context.GetDeviceId(), rhi::QueueRole::Compute, rhi::SubmissionDomain::Frame,
                    rhi::InvalidRenderViewId, rhi::PipelineStage::AllCommands, {},
                    [&](VkCommandBuffer commands) { return owner.manager.RecordAsyncSimulation(commands); },
                    "Snapshot/AsyncSimulation");
                (void)submission.AddWork(
                    resources.context.GetDeviceId(), rhi::QueueRole::Compute, rhi::SubmissionDomain::Frame,
                    rhi::InvalidRenderViewId, rhi::PipelineStage::AllCommands, {simulation, lastView},
                    [&](VkCommandBuffer commands) { return owner.manager.RecordAsyncExport(commands); },
                    "Snapshot/AsyncExport");
            }
            rhi::SubmissionPlan plan;
            if (!Require(submission.Build(plan, error), "Snapshot frame composition failed"))
                return false;
            vk::VulkanSubmissionExecutor::ExternalSync sync;
            sync.completionEpoch = epochs[frame] = owner.queues.ReserveCompletionEpoch();
            sync.completionFence = owner.fences[frame];
            sync.producers = {{owner.gate, cycle + 1}};
            sync.previousFrameTimeline = previous.completionTimeline;
            sync.previousFrameTimelineValue = previous.completionTimelineValue;
            sync.previousFrameWaitAtFirstBatch = cycle != 2;
            if (!Require(vkResetFences(resources.context.GetDevice(), 1, &owner.fences[frame]) == VK_SUCCESS,
                         "Snapshot fence reset failed"))
                return false;
            previous = owner.executor.Execute(
                frame, plan,
                [&](uint32_t batch, VkCommandBuffer commands) { return submission.RecordBatch(plan, batch, commands); },
                sync);
            owner.manager.NotifySubmission(previous.Succeeded());
            if (!Require(previous.Succeeded(), "Snapshot frame submission failed"))
                return false;
        }
        for (auto fence : owner.fences)
            if (!Require(vkGetFenceStatus(resources.context.GetDevice(), fence) == VK_NOT_READY,
                         "Snapshot gate did not retain each frame"))
                return false;
        VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
        signal.semaphore = owner.gate;
        signal.value = cycle + 1;
        if (!Require(vkSignalSemaphore(resources.context.GetDevice(), &signal) == VK_SUCCESS &&
                         vkWaitForFences(resources.context.GetDevice(), 2, owner.fences.data(), VK_TRUE,
                                         5'000'000'000) == VK_SUCCESS,
                     "Snapshot frame completion failed"))
            return false;
        for (uint32_t frame = 0; frame < 2; ++frame) {
            owner.executor.CompleteFrame(frame);
            owner.queues.CompleteCompletionEpoch(epochs[frame]);
        }
        std::array<glm::vec4, 4> actual{};
        if (!Require(device.ReadBuffer(owner.readback, 0, actual.data(), sizeof(actual)),
                     "Snapshot image readback failed"))
            return false;
        for (size_t index = 0; index < actual.size(); ++index) {
            for (uint32_t channel = 0; channel < 4; ++channel) {
                if (!std::isfinite(actual[index][channel]) ||
                    std::abs(actual[index][channel] - expected[index][channel]) > 1e-5f) {
                    std::cerr << "Snapshot cycle=" << cycle << " image=" << index << " channel=" << channel
                              << " actual=" << actual[index][channel] << " expected=" << expected[index][channel]
                              << '\n';
                    return Require(false, "A particle view consumed another frame's inputs");
                }
            }
        }
        (void)owner.retirement.Collect(epochs[1]);
        if (!Require(owner.retirement.PendingCount() == 0 && scenes.GetActiveScene() == activeScene,
                     "Snapshot versions leaked or source lookup changed the active scene"))
            return false;
        std::cout << "Particle frame snapshots: phase=" << cycle
                  << " frames=2 views=2 delayed=true independent_compute=" << independent
                  << " transforms/palette/metadata/parameters/material/pixels matched\n";
    }
    return true;
}
