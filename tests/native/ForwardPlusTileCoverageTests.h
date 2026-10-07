// Exercise the actual grid owner, push constants, per-frame bindings and GPU
// dispatch. The broader projection/ray oracle lives in the Python GPU suite.
bool VerifyForwardPlusTileCoverage(TestResources &resources, infernux::InxShaderLoader &compiler)
{
    using namespace infernux;
    using namespace infernux::lighting;
    const auto code = SpirvWords(compiler.CompileComputeGlsl(std::string(ForwardPlusLightGrid::ShaderSource()),
                                                             "Tests/ForwardPlusCoverage.comp"));
    if (!Require(!code.empty(), "Forward+ coverage shader compilation failed"))
        return false;
    auto &device = resources.context.GetRhiDevice();
    const auto native = resources.context.GetDevice();
    struct Owner
    {
        vk::VkDeviceContext &context;
        ForwardPlusLightGrid grid;
        rhi::BufferHandle canonical;
        rhi::BufferHandle output;
        rhi::BindingLayoutHandle layout;
        rhi::ShaderModuleHandle shader;
        rhi::ComputePipelineHandle pipeline;
        std::array<rhi::BindGroupHandle, 2> groups{};
        BufferReadback readback;
        VkCommandPool pool = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        ~Owner()
        {
            context.WaitIdle();
            auto &device = context.GetRhiDevice();
            for (auto group : groups)
                device.Release(group);
            device.Release(pipeline);
            device.Release(shader);
            device.Release(layout);
            device.Release(output);
            grid.Shutdown();
            device.Release(canonical);
            if (pool)
                vkDestroyCommandPool(context.GetDevice(), pool, nullptr);
            if (fence)
                vkDestroyFence(context.GetDevice(), fence, nullptr);
        }
    } owner{resources.context};
    struct alignas(16) Lights
    {
        glm::uvec4 header{0, 2, 2, 1};
        std::array<CanonicalLightData, 2> values{};
    } lights;
    lights.values[0].positionRange = {3, 0, 5, 1};
    lights.values[1].positionRange = {-3, 0, 5, 1};
    lights.values[0].metadata = {1, ~0u, 0, CanonicalLightAffectsGeometry};
    lights.values[1].metadata = {1, ~0u, 0, CanonicalLightAffectsParticles};
    owner.canonical = device.CreateBuffer(
        {sizeof(lights), rhi::BufferUsageFlags::Storage | rhi::BufferUsageFlags::TransferDestination});
    if (!Require(owner.canonical.IsValid() && owner.grid.Initialize(device, 2, {code.data(), code.size()}),
                 "Forward+ grid initialization failed"))
        return false;
    for (uint32_t frame = 0; frame < 2; ++frame) {
        if (!Require(owner.grid.PrepareFrame(frame, 2048, 2048, 2, 1u << frame, owner.canonical),
                     "Forward+ grid frame preparation failed"))
            return false;
    }
    const auto maskBytes = owner.grid.Frame(0).config.maskBytes;
    // Production masks are Storage-only. Consume them through a shader as
    // the renderer does, without adding test-only transfer usage to the grid.
    const auto copyCode = SpirvWords(compiler.CompileComputeGlsl(R"glsl(
#version 450
layout(local_size_x=64) in;
layout(std430, set=0, binding=0) readonly buffer Masks { uint masks[]; };
layout(std430, set=0, binding=1) writeonly buffer Output { uint outputWords[]; };
layout(push_constant) uniform Copy { uint offset; } pc;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < 128u * 128u) outputWords[pc.offset + i] = masks[i];
}
)glsl",
                                                                 "Tests/ForwardPlusCoverageReadback.comp"));
    if (!Require(!copyCode.empty(), "Forward+ mask consumer compilation failed"))
        return false;
    owner.output =
        device.CreateBuffer({maskBytes * 2, rhi::BufferUsageFlags::Storage | rhi::BufferUsageFlags::TransferSource});
    rhi::BindingLayoutDesc layout;
    layout.entryCount = 2;
    for (uint32_t binding = 0; binding < 2; ++binding)
        layout.entries[binding] = {binding, rhi::BindingType::StorageBuffer, rhi::ShaderStage::Compute, 1};
    owner.layout = device.CreateBindingLayout(layout);
    owner.shader = device.CreateShaderModule(rhi::ShaderModuleDesc::FromSpirV(copyCode.data(), copyCode.size()));
    rhi::ComputePipelineDesc copyPipeline;
    copyPipeline.computeShader = owner.shader;
    copyPipeline.bindingLayouts[0] = owner.layout;
    copyPipeline.bindingLayoutCount = 1;
    copyPipeline.pushConstantBytes = sizeof(uint32_t);
    owner.pipeline = device.CreateComputePipeline(copyPipeline);
    if (!Require(owner.output.IsValid() && owner.layout.IsValid() && owner.shader.IsValid() && owner.pipeline.IsValid(),
                 "Forward+ mask consumer resources failed"))
        return false;
    for (uint32_t frame = 0; frame < 2; ++frame) {
        rhi::BindGroupDesc group;
        group.layout = owner.layout;
        group.bufferCount = 2;
        group.buffers[0] = {0, rhi::BindingType::StorageBuffer, owner.grid.Frame(frame).lightMasks, 0, maskBytes};
        group.buffers[1] = {1, rhi::BindingType::StorageBuffer, owner.output, 0, maskBytes * 2};
        owner.groups[frame] = device.CreateBindGroup(group);
        if (!Require(owner.groups[frame].IsValid(), "Forward+ mask consumer binding failed"))
            return false;
    }
    if (!Require(owner.readback.Create(resources.context.GetVmaAllocator(), maskBytes * 2),
                 "Forward+ readback allocation failed"))
        return false;
    VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool.queueFamilyIndex = resources.context.GetQueueIndices().graphicsFamily.value();
    if (!Require(vkCreateCommandPool(native, &pool, nullptr, &owner.pool) == VK_SUCCESS,
                 "Forward+ command pool allocation failed"))
        return false;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = owner.pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    if (!Require(vkAllocateCommandBuffers(native, &allocate, &command) == VK_SUCCESS,
                 "Forward+ command allocation failed"))
        return false;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    if (!Require(vkBeginCommandBuffer(command, &begin) == VK_SUCCESS, "Forward+ command begin failed"))
        return false;
    vkCmdUpdateBuffer(command, device.Resolve(owner.canonical), 0, sizeof(lights), &lights);
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = device.Resolve(owner.canonical);
    barrier.size = VK_WHOLE_SIZE;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                         1, &barrier, 0, nullptr);
    ForwardPlusGridConstants constants{};
    glm::mat4 projection{0.0f};
    projection[0][0] = 1;
    projection[1][1] = -1;
    projection[2][2] = 100.0f / 99.9f;
    projection[2][3] = 1;
    projection[3][2] = -10.0f / 99.9f;
    std::memcpy(constants.viewProjection, &projection[0][0], sizeof(projection));
    // The record entry point must supply the actual viewport/grid/domain.
    constants.viewportAndProjectionScale[0] = constants.viewportAndProjectionScale[1] = 9;
    constants.viewportAndProjectionScale[2] = constants.viewportAndProjectionScale[3] = 1;
    vk::VulkanComputeCommandContext computeContext;
    auto encoder = device.MakeComputeCommandEncoder(computeContext, command);
    for (uint32_t frame = 0; frame < 2; ++frame) {
        owner.grid.Record(frame, encoder, constants);
        barrier.buffer = device.Resolve(owner.grid.Frame(frame).lightMasks);
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                             nullptr, 1, &barrier, 0, nullptr);
        encoder.BindPipeline(owner.pipeline);
        encoder.BindGroup(owner.pipeline, 0, owner.groups[frame]);
        const uint32_t offset = frame * static_cast<uint32_t>(maskBytes / sizeof(uint32_t));
        encoder.PushConstants(owner.pipeline, sizeof(offset), &offset);
        encoder.Dispatch(128 * 128 / 64, 1, 1);
        // The second consumer writes another range of the same descriptor;
        // explicitly order the writes for validation and all Vulkan devices.
        barrier.buffer = device.Resolve(owner.output);
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                             nullptr, 1, &barrier, 0, nullptr);
    }
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                         1, &barrier, 0, nullptr);
    const VkBufferCopy copy{0, 0, maskBytes * 2};
    vkCmdCopyBuffer(command, barrier.buffer, owner.readback.buffer, 1, &copy);
    if (!Require(vkEndCommandBuffer(command) == VK_SUCCESS, "Forward+ command end failed"))
        return false;
    VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (!Require(vkCreateFence(native, &fence, nullptr, &owner.fence) == VK_SUCCESS, "Forward+ fence creation failed"))
        return false;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    if (!Require(vkQueueSubmit(resources.context.GetGraphicsQueue(), 1, &submit, owner.fence) == VK_SUCCESS &&
                     vkWaitForFences(native, 1, &owner.fence, VK_TRUE, 5000000000ull) == VK_SUCCESS,
                 "Forward+ submission failed"))
        return false;
    if (!Require(vmaInvalidateAllocation(owner.readback.allocator, owner.readback.allocation, 0, maskBytes * 2) ==
                     VK_SUCCESS,
                 "Forward+ readback invalidate failed"))
        return false;
    const auto *masks = static_cast<const uint32_t *>(owner.readback.mapped);
    for (uint32_t frame = 0; frame < 2; ++frame) {
        const auto *view = masks + frame * maskBytes / sizeof(uint32_t);
        const uint32_t x = frame == 0 ? 119 : 8;
        if (!Require(view[64 * 128 + x] == (1u << frame), "Forward+ missed an off-axis lit tile"))
            return false;
        for (size_t tile = 0; tile < maskBytes / sizeof(uint32_t); ++tile) {
            if (!Require((view[tile] & ~(1u << frame)) == 0, "Forward+ mixed per-frame light domains"))
                return false;
        }
        if (!Require(view[64 * 128 + (127 - x)] == 0, "Forward+ retained a disjoint tile"))
            return false;
    }
    std::cout << "Forward+ native grid coverage: two frame/domain readbacks passed\n";
    return true;
}
