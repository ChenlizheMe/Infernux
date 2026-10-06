bool VerifyParticleSceneSources(TestResources &resources, infernux::InxShaderLoader &compiler,
                                const infernux::particle::GpuParticleEmitterProgram &prototype,
                                const infernux::particle::GpuParticleSortProgram &sortProgram,
                                const infernux::particle::GpuParticleCullProgram &cullProgram,
                                const infernux::particle::GpuParticleBoundsProgram &boundsProgram,
                                const infernux::particle::GpuParticleMigrationProgram &migrationProgram,
                                const infernux::particle::GpuParticleSpawnProgram &spawnProgram)
try {
    using namespace infernux;
    using namespace infernux::particle;
    auto &registry = AssetRegistry::Instance();
    registry.Initialize(std::make_unique<AssetDatabase>());
    struct RegistryOwner
    {
        AssetRegistry &registry;
        ~RegistryOwner()
        {
            registry.Shutdown();
        }
    } registryOwner{registry};
    registry.RegisterLoader(ResourceType::Mesh, std::make_unique<MeshLoader>());
    auto &scenes = SceneManager::Instance();
    scenes.Stop();
    scenes.UnloadAllScenes();
    struct SceneOwner
    {
        SceneManager &scenes;
        ~SceneOwner()
        {
            scenes.Stop();
            scenes.UnloadAllScenes();
        }
    } sceneOwner{scenes};
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
    auto mesh = registry.CreateRuntimeMesh("ParticleSceneSource");
    InxMesh replacement("ParticleSceneSource");
    replacement.SetData(model->baseVertices, model->indices, {});
    replacement.SetSkinnedData(model);
    registry.PublishMesh(mesh->GetGuid(), std::move(replacement));
    Scene *sceneA = scenes.CreateScene("ParticleSourceA");
    Scene *sceneB = scenes.CreateScene("ParticleSourceB");
    auto *object = sceneB->CreateGameObject("SkinnedSource");
    auto *source = object->AddComponent<SkinnedMeshRenderer>();
    source->SetSourceModelGuid(mesh->GetGuid());
    source->SetActiveTakeName("Move");
    source->SetRuntimeAnimationTime(0.25f);
    object->GetTransform()->SetPosition({10, 0, 0});
    scenes.SetActiveScene(sceneB);
    const ObjectHandle initialHandle = source->GetHandle();
    const auto active = ResolveSceneSkinnedMeshSource(initialHandle);
    if (!Require(active && active->currentPalette && active->currentPalette->size() == 1,
                 "Active-scene skin source control failed"))
        return false;
    scenes.SetActiveScene(sceneA);
    const auto inactive = ResolveSceneSkinnedMeshSource(initialHandle);
    if (!Require(inactive && inactive->mesh == mesh && inactive->revision == active->revision &&
                     inactive->sourceToWorld[3] == 10.0f && scenes.GetActiveScene() == sceneA,
                 "Live skin source in a nonactive loaded scene could not be resolved"))
        return false;

    const auto noop = SpirvWords(compiler.CompileComputeGlsl(
        "#version 450\nlayout(local_size_x=256) in; void main() {}", "Tests/SceneSourceNoop.comp"));
    const auto bootstrap = SpirvWords(compiler.CompileComputeGlsl(R"glsl(
#version 450
layout(local_size_x=256) in;
layout(std430, set=0, binding=0) buffer States { uvec4 states[]; };
layout(std430, set=0, binding=2) buffer Counters { uint counters[]; };
layout(std430, set=0, binding=4) buffer Indirect { uvec4 indirectArgs; };
layout(std430, set=0, binding=9) buffer Dispatch { uvec4 dispatches[]; };
void main() {
    if (gl_GlobalInvocationID.x < 32u) states[gl_GlobalInvocationID.x] = uvec4(0u);
    if (gl_GlobalInvocationID.x != 0u) return;
    counters[0] = 32u;
    indirectArgs = uvec4(0u);
    dispatches[0] = dispatches[1] = uvec4(1u, 1u, 1u, 0u);
}
)glsl",
                                                                  "Tests/SceneSourceBootstrap.comp"));
    const auto rendering = SpirvWords(compiler.CompileComputeGlsl(R"glsl(
#version 450
layout(local_size_x=256) in;
layout(std430, set=0, binding=3) writeonly buffer Instances { vec4 snapshots[]; };
layout(std430, set=1, binding=0) readonly buffer Metadata { uint metadata[]; };
layout(std430, set=1, binding=4) readonly buffer Palette { mat4 bones[]; };
void main() {
    if (gl_GlobalInvocationID.x == 0u)
        snapshots[0] = vec4(bones[0][3].x, uintBitsToFloat(metadata[16]), float(metadata[3]), 1.0);
}
)glsl",
                                                                  "Tests/SceneSourceRendering.comp"));
    const auto readbackCode = SpirvWords(compiler.CompileComputeGlsl(R"glsl(
#version 450
layout(local_size_x=1) in;
layout(std430, set=0, binding=0) readonly buffer Input { vec4 snapshot; };
layout(std430, set=0, binding=1) writeonly buffer Output { vec4 result; };
void main() { result = snapshot; }
)glsl",
                                                                     "Tests/SceneSourceReadback.comp"));
    if (!Require(!noop.empty() && !bootstrap.empty() && !rendering.empty() && !readbackCode.empty(),
                 "Scene source probe shaders failed"))
        return false;
    struct Owner
    {
        vk::VkDeviceContext &context;
        GpuRetirementQueue retirement;
        ParticleGpuDrawRegistry draws;
        ParticleGpuSystemManager manager;
        rhi::BufferHandle readback;
        rhi::BindingLayoutHandle layout;
        rhi::ShaderModuleHandle shader;
        rhi::ComputePipelineHandle pipeline;
        rhi::BindGroupHandle group;
        VkCommandPool pool = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        ~Owner()
        {
            context.WaitIdle();
            manager.Shutdown();
            retirement.FlushAll();
            auto &device = context.GetRhiDevice();
            device.Release(group);
            device.Release(pipeline);
            device.Release(shader);
            device.Release(layout);
            device.Release(readback);
            if (fence)
                vkDestroyFence(context.GetDevice(), fence, nullptr);
            if (pool)
                vkDestroyCommandPool(context.GetDevice(), pool, nullptr);
        }
    } owner{resources.context};
    owner.retirement.BindSerialSource([] { return rhi::SubmissionSerial{1}; });
    if (!Require(owner.manager.Initialize(resources.context, resources.pipelines, resources.resources, owner.retirement,
                                          owner.draws, {}, {}, ResolveSceneSkinnedMeshSource, sortProgram, cullProgram,
                                          boundsProgram, migrationProgram, spawnProgram),
                 "Scene source particle manager initialization failed"))
        return false;
    auto program = prototype;
    program.id = 99701;
    program.graphInstanceId = 9971;
    program.outputs.front().id = 997011;
    for (auto &kernel : program.kernels)
        kernel = noop;
    program.kernels[static_cast<size_t>(GpuKernelStage::Bootstrap)] = bootstrap;
    program.kernels[static_cast<size_t>(GpuKernelStage::Rendering)] = rendering;
    program.kernels[static_cast<size_t>(GpuKernelStage::UpdateRenderingFused)].clear();
    GpuParticleMeshInterfaceProgram meshInterface;
    meshInterface.stableId = "scene-source";
    meshInterface.mesh = mesh;
    meshInterface.skinnedRendererHandle = BindSceneSkinnedMeshSource(*source);
    program.meshInterfaces = {meshInterface};
    const auto publish = [&] {
        GpuParticleGraphProgram graph;
        graph.graphInstanceId = program.graphInstanceId;
        graph.emitters = {program};
        std::string error;
        if (owner.manager.ApplyGraph(graph, &error))
            return true;
        std::cerr << error << '\n';
        return Require(false, "A valid nonactive-scene skin source could not be published");
    };
    if (!publish())
        return false;
    auto &device = resources.context.GetRhiDevice();
    rhi::BufferDesc readback;
    readback.byteSize = 16;
    readback.usage = rhi::BufferUsageFlags::Storage;
    readback.memory = rhi::BufferMemory::Readback;
    owner.readback = device.CreateBuffer(readback);
    rhi::BindingLayoutDesc layout;
    layout.entryCount = 2;
    for (uint32_t binding = 0; binding < 2; ++binding)
        layout.entries[binding] = {binding, rhi::BindingType::StorageBuffer, rhi::ShaderStage::Compute, 1};
    owner.layout = device.CreateBindingLayout(layout);
    owner.shader =
        device.CreateShaderModule(rhi::ShaderModuleDesc::FromSpirV(readbackCode.data(), readbackCode.size()));
    rhi::ComputePipelineDesc pipeline;
    pipeline.computeShader = owner.shader;
    pipeline.bindingLayoutCount = 1;
    pipeline.bindingLayouts[0] = owner.layout;
    owner.pipeline = device.CreateComputePipeline(pipeline);
    VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool.queueFamilyIndex = resources.context.GetQueueIndices().graphicsFamily.value();
    VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (!Require(owner.readback.IsValid() && owner.pipeline.IsValid() &&
                     vkCreateCommandPool(resources.context.GetDevice(), &pool, nullptr, &owner.pool) == VK_SUCCESS &&
                     vkCreateFence(resources.context.GetDevice(), &fence, nullptr, &owner.fence) == VK_SUCCESS,
                 "Scene source readback resources failed"))
        return false;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = owner.pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    if (!Require(vkAllocateCommandBuffers(resources.context.GetDevice(), &allocate, &command) == VK_SUCCESS,
                 "Scene source command allocation failed"))
        return false;
    uint64_t frameIndex = 200;
    const auto sample = [&](float poseX, float worldX) {
        GpuParticleFrameRequest frame;
        frame.frameIndex = frameIndex++;
        frame.deltaTime = 1.0f / 60.0f;
        frame.boundsMode = GpuParticleBoundsMode::Manual;
        frame.manualBoundsLower = {-1, -1, -1};
        frame.manualBoundsUpper = {1, 1, 1};
        GpuParticleTransforms transforms;
        for (size_t axis = 0; axis < 4; ++axis) {
            transforms.emitterToWorld[axis * 5] = 1.0f;
            transforms.worldToEmitter[axis * 5] = 1.0f;
            transforms.simulationToWorld[axis * 5] = 1.0f;
            transforms.worldToSimulation[axis * 5] = 1.0f;
        }
        if (!Require(owner.manager.BeginFrame(program.id, frame, transforms), "Scene source BeginFrame failed"))
            return false;
        const auto entries = owner.draws.SnapshotShared(0, 5000);
        if (!Require(entries && entries->size() == 1 && entries->front().id == program.outputs.front().id,
                     "Scene source draw output missing"))
            return false;
        device.Release(owner.group);
        rhi::BindGroupDesc group;
        group.layout = owner.layout;
        group.bufferCount = 2;
        group.buffers[0] = {0, rhi::BindingType::StorageBuffer, entries->front().instances, 0, 16};
        group.buffers[1] = {1, rhi::BindingType::StorageBuffer, owner.readback, 0, 16};
        owner.group = device.CreateBindGroup(group);
        if (!Require(owner.group.IsValid() &&
                         vkResetCommandPool(resources.context.GetDevice(), owner.pool, 0) == VK_SUCCESS &&
                         vkResetFences(resources.context.GetDevice(), 1, &owner.fence) == VK_SUCCESS,
                     "Scene source recording reset failed"))
            return false;
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        if (!Require(vkBeginCommandBuffer(command, &begin) == VK_SUCCESS, "Scene source command begin failed"))
            return false;
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0,
                             nullptr, 0, nullptr);
        owner.manager.Execute(command);
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &barrier, 0, nullptr, 0, nullptr);
        vk::VulkanComputeCommandContext computeContext;
        auto compute = device.MakeComputeCommandEncoder(computeContext, command);
        compute.BindPipeline(owner.pipeline);
        compute.BindGroup(owner.pipeline, 0, owner.group);
        compute.Dispatch(1, 1, 1);
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier,
                             0, nullptr, 0, nullptr);
        if (!Require(vkEndCommandBuffer(command) == VK_SUCCESS, "Scene source command end failed"))
            return false;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        const auto result = vkQueueSubmit(resources.context.GetGraphicsQueue(), 1, &submit, owner.fence);
        owner.manager.NotifySubmission(result == VK_SUCCESS);
        if (!Require(result == VK_SUCCESS && vkWaitForFences(resources.context.GetDevice(), 1, &owner.fence, VK_TRUE,
                                                             5'000'000'000ull) == VK_SUCCESS,
                     "Scene source GPU submission failed"))
            return false;
        std::array<float, 4> actual{};
        if (!Require(device.ReadBuffer(owner.readback, 0, actual.data(), sizeof(actual)),
                     "Scene source readback failed"))
            return false;
        if (std::abs(actual[0] - poseX) > 1.0e-5f || std::abs(actual[1] - worldX) > 1.0e-5f || actual[2] != 1.0f ||
            actual[3] != 1.0f) {
            std::cerr << "Skin source actual=" << actual[0] << ',' << actual[1] << ',' << actual[2] << ',' << actual[3]
                      << " expected=" << poseX << ',' << worldX << ",1,1\n";
            return Require(false, "GPU skin source froze or resolved another scene");
        }
        owner.retirement.FlushAll();
        return true;
    };
    if (!sample(0.25f, 10.0f))
        return false;
    scenes.SetActiveScene(sceneB);
    source->SetRuntimeAnimationTime(0.5f);
    object->GetTransform()->SetPosition({20, 0, 0});
    if (!sample(0.5f, 20.0f))
        return false;
    scenes.SetActiveScene(sceneA);
    source->SetRuntimeAnimationTime(0.75f);
    object->GetTransform()->SetPosition({30, 0, 0});
    if (!sample(0.75f, 30.0f))
        return false;

    auto wrongWorld = initialHandle;
    wrongWorld.worldId = sceneA->GetWorldId();
    auto expired = initialHandle;
    ++expired.generation;
    if (!Require(!ResolveSceneSkinnedMeshSource(wrongWorld) && !ResolveSceneSkinnedMeshSource(expired) &&
                     !ResolveSceneSkinnedMeshSource({}) &&
                     !ResolveSceneSkinnedMeshSource(object->GetTransform()->GetHandle()),
                 "Invalid world, generation or component type bypassed source validation"))
        return false;
    scenes.MoveGameObjectToScene(object, sceneA);
    source->SetRuntimeAnimationTime(0.875f);
    object->GetTransform()->SetPosition({35, 0, 0});
    if (!Require(meshInterface.skinnedRendererHandle() == source->GetHandle() &&
                     !ResolveSceneSkinnedMeshSource(initialHandle),
                 "Bound component did not follow an explicit Scene move") ||
        !sample(0.875f, 35.0f))
        return false;
    scenes.Play();
    scenes.DontDestroyOnLoad(object);
    scenes.PrepareActiveSceneReplacement();
    if (!Require(object->GetScene() == scenes.GetRuntimePersistentScene() &&
                     !ResolveSceneSkinnedMeshSource(initialHandle) &&
                     ResolveSceneSkinnedMeshSource(source->GetHandle()),
                 "Persistent migration did not enforce the new owning world identity"))
        return false;
    // Residency changes must not require a graph publication or reset.
    source->SetRuntimeAnimationTime(1.0f);
    if (!sample(1.0f, 35.0f))
        return false;
    const auto persistentHandle = source->GetHandle();
    scenes.UnloadAllScenes();
    source->SetRuntimeAnimationTime(1.25f);
    if (!sample(1.25f, 35.0f))
        return false;
    scenes.Stop();
    if (!Require(!ResolveSceneSkinnedMeshSource(persistentHandle) && !meshInterface.skinnedRendererHandle().IsValid(),
                 "Destroyed persistent source remained bound"))
        return false;
    // Reusing an authored ID must never revive a binding to a destroyed native
    // instance, even while the old graph and callback copies remain resident.
    auto *replacementScene = scenes.CreateScene("ReplacementSkin");
    auto *replacementObject = replacementScene->CreateGameObject("Replacement");
    auto *replacementSource = replacementObject->AddComponent<SkinnedMeshRenderer>();
    replacementSource->SetComponentID(persistentHandle.id);
    replacementSource->SetSourceModelGuid(mesh->GetGuid());
    replacementSource->SetActiveTakeName("Move");
    replacementSource->SetRuntimeAnimationTime(1.5f);
    if (!Require(!meshInterface.skinnedRendererHandle().IsValid(), "Retired binding attached to a replacement"))
        return false;
    // Also release the graph's borrower before its source, the opposite order
    // to the persistent-source destruction above.
    auto temporaryBinding = BindSceneSkinnedMeshSource(*replacementSource);
    if (!Require(temporaryBinding() == replacementSource->GetHandle(), "Replacement could not be bound explicitly"))
        return false;
    temporaryBinding = {};
    replacementScene->DestroyGameObject(replacementObject);
    std::cout << "Particle scene source: samples=6 publications=1 explicit_move=1 persistent_move=1 retired=1\n";
    return true;
} catch (const std::exception &error) {
    std::cerr << "FAILED: Scene source fixture: " << error.what() << std::endl;
    return false;
}
