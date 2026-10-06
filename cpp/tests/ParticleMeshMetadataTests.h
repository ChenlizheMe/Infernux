// Uses the enclosing real Vulkan fixture; the probe reads the runtime-owned
// descriptor set, rather than a separately prepared copy of its metadata.
bool VerifyParticleMeshMetadata(TestResources &resources, infernux::InxShaderLoader &compiler,
                                const infernux::particle::GpuParticleSpawnProgram &spawnProgram)
{
    using namespace infernux;
    const auto code = SpirvWords(compiler.CompileComputeGlsl(R"glsl(
#version 450
layout(local_size_x = 256) in;
layout(std430, set = 0, binding = 0) buffer Output { uint output_words[]; };
layout(std430, set = 1, binding = 0) readonly buffer Metadata { uint metadata_words[]; };
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index < 32u) output_words[index] = metadata_words[index];
}
)glsl", "Tests/ParticleMeshMetadata.comp"));
    if (!Require(!code.empty(), "Particle metadata probe compilation failed")) return false;
    auto &device = resources.context.GetRhiDevice();
    struct Owners {
        vk::VulkanRhiDevice &device;
        VkDevice native;
        rhi::BufferHandle geometry;
        VkCommandPool pool = VK_NULL_HANDLE;
        ~Owners() {
            if (pool) vkDestroyCommandPool(native, pool, nullptr);
            device.Release(geometry);
        }
    } owners{device, resources.context.GetDevice()};
    rhi::BufferDesc geometryDesc;
    geometryDesc.byteSize = 256;
    geometryDesc.usage = rhi::BufferUsageFlags::Storage;
    owners.geometry = device.CreateBuffer(geometryDesc);
    if (!Require(owners.geometry.IsValid(), "Particle metadata geometry allocation failed")) return false;

    particle::GpuEmitterDesc desc;
    desc.capacity = 32;
    desc.stateStride = 16;
    for (auto &kernel : desc.kernels) kernel = {code.data(), code.size()};
    desc.kernels[static_cast<size_t>(particle::GpuKernelStage::UpdateRenderingFused)] = {};
    particle::GpuMeshInterfaceDesc mesh;
    mesh.stableId = "moving-skinned-source";
    mesh.vertexBinding = 1;
    mesh.triangleBinding = 2;
    mesh.influenceBinding = 3;
    mesh.paletteBinding = 4;
    mesh.vertexCount = 3;
    mesh.triangleCount = 1;
    mesh.edgeCount = 3;
    mesh.boneCount = 1;
    mesh.worldSpace = true;
    mesh.vertices = mesh.triangles = mesh.influences = owners.geometry;
    mesh.keepAlive = std::make_shared<int>(1);
    mesh.initialPalette = {glm::mat4(1.0f)};
    desc.meshInterfaces = {mesh};
    particle::ParticleGpuRuntime runtime;
    particle::ParticleGpuGraphSpawnDomain spawn;
    if (!Require(runtime.Create(device, desc) && spawn.Create(device, 99501, 1, spawnProgram, {}) &&
                 spawn.RegisterEmitter(0, runtime), "Particle metadata runtime creation failed")) return false;
    RenderGraph graph;
    graph.Initialize(&resources.context);
    graph.AddComputePass("Metadata/ReadDescriptor", [&](PassBuilder &builder) {
        auto output = builder.ImportBuffer("State", runtime.StateBuffer(), 32 * 16);
        graph.SetResourceInitialState(output, rhi::TextureLayout::Undefined, rhi::Access::TransferRead,
                                       rhi::PipelineStage::Transfer);
        builder.WriteStorageBuffer(output);
        builder.SetSideEffect();
        return [&](RenderContext &context) {
            if (!runtime.RecordBootstrap(context.GetComputeCommandEncoder(), 0, spawn.RuntimeGroup(0)))
                throw std::runtime_error("Particle metadata probe recording failed");
        };
    });
    if (!Require(graph.Compile(), "Particle metadata graph compilation failed")) return false;
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = resources.context.GetQueueIndices().graphicsFamily.value();
    if (!Require(vkCreateCommandPool(owners.native, &poolInfo, nullptr, &owners.pool) == VK_SUCCESS,
                 "Particle metadata command pool failed")) return false;
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = owners.pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    VkCommandBuffer command;
    if (!Require(vkAllocateCommandBuffers(owners.native, &allocate, &command) == VK_SUCCESS,
                 "Particle metadata command allocation failed")) return false;
    BufferReadback readback;
    if (!Require(readback.Create(resources.context.GetVmaAllocator(), 128), "Particle metadata readback failed"))
        return false;
    const std::array<float, 16> identity{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    particle::GpuParticleTransforms transforms{identity, identity, identity, identity};
    auto source = identity;
    const auto palette = std::make_shared<const std::vector<glm::mat4>>(mesh.initialPalette);
    for (uint32_t step = 0; step < 5; ++step) {
        if (step == 1) source[3] = 8;
        if (step == 2) source = {0,-3,0,8, 2,0,0,0, 0,0,4,0, 0,0,0,1};
        if (step == 4) {
            transforms.emitterToWorld[12] = 10;
            transforms.worldToEmitter[12] = -10;
            transforms.simulationToWorld[12] = 4;
            transforms.worldToSimulation[12] = -4;
        }
        if (!Require(runtime.UpdateSkinnedMeshSources({{0u, step >= 3 ? 2u : 1u, source, palette}}) &&
                     runtime.UpdateTransforms(transforms), "Particle metadata frame update failed")) return false;
        runtime.RequestBootstrap();
        if (!Require(vkResetCommandBuffer(command, 0) == VK_SUCCESS, "Particle metadata command reset failed"))
            return false;
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        if (!Require(vkBeginCommandBuffer(command, &begin) == VK_SUCCESS, "Particle metadata begin failed")) return false;
        VkMemoryBarrier inputs{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        inputs.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT |
                               VK_ACCESS_TRANSFER_WRITE_BIT;
        inputs.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &inputs, 0, nullptr, 0, nullptr);
        vk::VulkanTransferCommandContext transferContext;
        const auto transfer = device.MakeTransferCommandEncoder(transferContext, command);
        if (!Require(runtime.RecordPendingUploads(transfer) && spawn.RecordPendingUploads(transfer),
                     "Particle metadata input recording failed"))
            return false;
        inputs.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        inputs.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &inputs, 0, nullptr, 0, nullptr);
        graph.Execute(command);
        VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = device.Resolve(runtime.StateBuffer());
        barrier.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 1, &barrier, 0, nullptr);
        VkBufferCopy copy{0, 0, 128};
        vkCmdCopyBuffer(command, barrier.buffer, readback.buffer, 1, &copy);
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        barrier.buffer = readback.buffer;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                             0, 0, nullptr, 1, &barrier, 0, nullptr);
        if (!Require(vkEndCommandBuffer(command) == VK_SUCCESS, "Particle metadata end failed")) return false;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        const auto queue = resources.context.GetGraphicsQueue();
        const auto submitted = vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
        runtime.NotifySubmission(submitted == VK_SUCCESS);
        spawn.NotifySubmission(submitted == VK_SUCCESS);
        const auto waited = vkQueueWaitIdle(queue);
        if (!Require(submitted == VK_SUCCESS && waited == VK_SUCCESS, "Particle metadata submit failed") ||
            !Require(vmaInvalidateAllocation(readback.allocator, readback.allocation, 0, 128) == VK_SUCCESS,
                     "Particle metadata invalidate failed")) return false;
        std::array<float, 32> values;
        std::memcpy(values.data(), readback.mapped, sizeof(values));
        const float expectedX = step == 0 ? 0.0f : step == 4 ? 4.0f : 8.0f;
        if (!Require(std::abs(values[16] - expectedX) < 1e-5f &&
                     std::abs(values[21] - (step >= 2 ? 0.5f : 0.0f)) < 1e-5f &&
                     std::abs(values[24] - (step >= 2 ? -1.0f / 3.0f : 0.0f)) < 1e-5f,
                     "GPU descriptor retained a stale source position or normal")) return false;
        std::cout << "Particle mesh metadata step=" << step << " source_x=" << values[16] << '\n';
    }
    return true;
}
