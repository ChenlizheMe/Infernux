// Included beside the real Vulkan fixture and its BufferReadback helper.
bool VerifyParticleCullConservation(TestResources &resources,
                                    const infernux::particle::GpuParticleCullProgram &program)
{
    using namespace infernux;
    constexpr uint32_t capacity = 1024 * 1024;
    auto &device = resources.context.GetRhiDevice();
    const VkDevice nativeDevice = resources.context.GetDevice();
    struct Inputs
    {
        vk::VulkanRhiDevice &device;
        std::vector<rhi::BufferHandle> buffers;
        ~Inputs() { for (auto buffer : buffers) device.Release(buffer); }
        rhi::BufferHandle Add(uint64_t bytes)
        {
            rhi::BufferDesc desc;
            desc.byteSize = bytes;
            desc.usage = rhi::BufferUsageFlags::Storage;
            desc.memory = rhi::BufferMemory::Upload;
            desc.queueAccess = rhi::QueueAccessFlags::Graphics | rhi::QueueAccessFlags::Compute;
            auto buffer = device.CreateBuffer(desc);
            buffers.push_back(buffer);
            return buffer;
        }
    } inputs{device};
    const auto visibility = inputs.Add(uint64_t(capacity) * 16);
    const auto source = inputs.Add(16);
    const auto indices = inputs.Add(uint64_t(capacity) * 4);
    const auto bounds = inputs.Add(32);
    const auto control = inputs.Add(16);
    const auto metadata = inputs.Add(uint64_t(capacity) * particle::ParticleGpuRuntime::RenderInstanceStride);
    for (auto buffer : inputs.buffers)
        if (!Require(buffer.IsValid(), "Cull conservation input allocation failed")) return false;
    std::vector<uint32_t> sourceIndices(capacity);
    for (uint32_t index = 0; index < capacity; ++index) sourceIndices[index] = capacity - index - 1;
    std::vector<uint32_t> ribbonMetadata(uint64_t(capacity) * 28, 0);
    const std::array<uint32_t, 4> controlValues{0, 1, 0, 0};
    if (!Require(device.WriteBuffer(indices, 0, sourceIndices.data(), sourceIndices.size() * sizeof(uint32_t)) &&
                 device.WriteBuffer(metadata, 0, ribbonMetadata.data(), ribbonMetadata.size() * sizeof(uint32_t)) &&
                 device.WriteBuffer(control, 0, controlValues.data(), sizeof(controlValues)),
                 "Cull conservation initial upload failed")) return false;

    struct Pool
    {
        VkDevice device;
        VkCommandPool handle = VK_NULL_HANDLE;
        ~Pool() { if (handle) vkDestroyCommandPool(device, handle, nullptr); }
    } pool{nativeDevice};
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = resources.context.GetQueueIndices().graphicsFamily.value();
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (!Require(vkCreateCommandPool(nativeDevice, &poolInfo, nullptr, &pool.handle) == VK_SUCCESS,
                 "Cull conservation command pool failed")) return false;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = pool.handle;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    if (!Require(vkAllocateCommandBuffers(nativeDevice, &allocate, &command) == VK_SUCCESS,
                 "Cull conservation command allocation failed")) return false;
    BufferReadback readback;
    if (!Require(readback.Create(resources.context.GetVmaAllocator(), 36),
                 "Cull conservation readback failed")) return false;
    const std::array<float, 24> planes{1,0,0,1, -1,0,0,1, 0,1,0,1, 0,-1,0,1, 0,0,1,1, 0,0,-1,1};
    const auto ordered = [](float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
    };
    struct Case { uint32_t count; int coarse; }; // -1 outside, 0 unknown, 1 inside
    const std::array<Case, 7> cases{{{0, 1}, {1, 0}, {257, 0}, {4096, 1},
                                   {capacity, 0}, {capacity, 1}, {capacity, -1}}};
    for (bool ribbon : {false, true}) {
        particle::ParticleGpuCuller culler;
        particle::GpuParticleCullerDesc desc;
        desc.capacity = capacity;
        desc.vertexCount = 6;
        desc.visibility = visibility;
        desc.sourceIndices = indices;
        desc.sourceIndirectArguments = source;
        desc.bounds = bounds;
        desc.simulationControl = control;
        desc.ribbonInstances = metadata;
        desc.mode = ribbon ? particle::GpuParticleCullMode::RibbonSegments : particle::GpuParticleCullMode::Instances;
        desc.program = program;
        if (!Require(culler.Create(device, desc), "Cull conservation culler creation failed")) return false;
        RenderGraph graph;
        graph.Initialize(&resources.context);
        ResourceHandle graphSource, graphBounds, graphControl, graphVisibility, graphIndices, graphMetadata;
        ResourceHandle graphVisible, graphDraw, graphDispatch;
        graph.AddComputePass("Conservation/Reset", [&](PassBuilder &builder) {
            graphSource = builder.ImportBuffer("Source", source, 16);
            graphBounds = builder.ImportBuffer("Bounds", bounds, 32);
            graphControl = builder.ImportBuffer("Control", control, 16);
            graphVisibility = builder.ImportBuffer("Visibility", visibility, uint64_t(capacity) * 16);
            graphIndices = builder.ImportBuffer("SourceIndices", indices, uint64_t(capacity) * 4);
            graphMetadata = builder.ImportBuffer("Metadata", metadata, uint64_t(capacity) * 112);
            graphVisible = builder.ImportBuffer("Visible", culler.VisibleIndexBuffer(), uint64_t(capacity) * 4);
            graphDraw = builder.ImportBuffer("Draw", culler.DrawIndirectBuffer(), 16);
            graphDispatch = builder.ImportBuffer("Dispatch", culler.SortDispatchBuffer(), 20);
            for (auto handle : {graphSource, graphBounds, graphVisibility, graphIndices, graphMetadata})
                graph.SetResourceInitialState(handle, rhi::TextureLayout::Undefined, rhi::Access::HostWrite,
                                               rhi::PipelineStage::Host);
            for (auto handle : {graphDraw, graphDispatch})
                graph.SetResourceInitialState(handle, rhi::TextureLayout::Undefined, rhi::Access::TransferRead,
                                               rhi::PipelineStage::Transfer);
            graph.SetResourceInitialState(graphControl, rhi::TextureLayout::Undefined, rhi::Access::ShaderWrite,
                                           rhi::PipelineStage::ComputeShader);
            graph.SetResourceInitialState(graphVisible, rhi::TextureLayout::Undefined, rhi::Access::ShaderWrite,
                                           rhi::PipelineStage::ComputeShader);
            builder.ReadStorageBuffer(graphSource);
            builder.ReadStorageBuffer(graphBounds);
            graphControl = builder.ReadWrite(graphControl, rhi::PipelineStage::ComputeShader);
            graphDraw = builder.WriteStorageBuffer(graphDraw);
            graphDispatch = builder.WriteStorageBuffer(graphDispatch);
            return [&](RenderContext &context) { culler.RecordReset(context.GetComputeCommandEncoder(), planes); };
        });
        graph.AddComputePass("Conservation/Cull", [&](PassBuilder &builder) {
            builder.ReadStorageBuffer(graphSource);
            builder.ReadStorageBuffer(graphVisibility);
            builder.ReadStorageBuffer(graphIndices);
            builder.ReadStorageBuffer(graphMetadata);
            // Exercise both declaration orders for the shared indirect/SSBO input.
            if (ribbon) builder.ReadStorageBuffer(graphDispatch);
            builder.ReadIndirectBuffer(graphDispatch);
            if (!ribbon) builder.ReadStorageBuffer(graphDispatch);
            graphVisible = builder.WriteStorageBuffer(graphVisible);
            graphDraw = builder.ReadWrite(graphDraw, rhi::PipelineStage::ComputeShader);
            return [&](RenderContext &context) { culler.RecordCull(context.GetComputeCommandEncoder(), planes); };
        });
        graph.AddComputePass("Conservation/Finalize", [&](PassBuilder &builder) {
            graphDraw = builder.ReadWrite(graphDraw, rhi::PipelineStage::ComputeShader);
            graphDispatch = builder.WriteStorageBuffer(graphDispatch);
            builder.SetSideEffect();
            return [&](RenderContext &context) { culler.RecordFinalize(context.GetComputeCommandEncoder()); };
        });
        if (!Require(graph.Compile(), "Cull conservation graph compile failed")) return false;
        std::vector<float> spheres(uint64_t(capacity) * 4, 0.0f);
        for (const auto test : cases) {
            for (uint32_t index = 0; index < capacity; ++index) {
                spheres[index * 4] = test.coarse == -1 ? 4.0f : 0.0f;
                spheres[index * 4 + 3] = 0.25f;
            }
            const uint32_t sourceCount = ribbon ? (test.count ? test.count - 1 : 0) : test.count;
            const uint32_t expected = test.coarse == -1 ? 0 : sourceCount;
            const std::array<uint32_t, 4> arguments{ribbon ? sourceCount * 6 : 6, test.count, 7, 9};
            const float center = test.coarse == -1 ? 4.0f : 0.0f;
            const std::array<uint32_t, 8> boundsValues{ordered(center - .25f), ordered(-.25f), ordered(-.25f),
                ordered(center + .25f), ordered(.25f), ordered(.25f), test.coarse != 0 ? 1u : 0u, 0};
            if (!Require(device.WriteBuffer(source, 0, arguments.data(), sizeof(arguments)) &&
                         device.WriteBuffer(bounds, 0, boundsValues.data(), sizeof(boundsValues)) &&
                         device.WriteBuffer(visibility, 0, spheres.data(), spheres.size() * sizeof(float)),
                         "Cull conservation frame upload failed")) return false;
            for (int repeat = 0; repeat < 3; ++repeat) {
                if (!Require(vkResetCommandBuffer(command, 0) == VK_SUCCESS, "Cull command reset failed")) return false;
                VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                if (!Require(vkBeginCommandBuffer(command, &begin) == VK_SUCCESS, "Cull command begin failed")) return false;
                graph.Execute(command);
                std::array<VkBufferMemoryBarrier, 2> barriers{};
                const std::array<rhi::BufferHandle, 2> outputs{culler.DrawIndirectBuffer(), culler.SortDispatchBuffer()};
                for (size_t index = 0; index < barriers.size(); ++index) {
                    auto &barrier = barriers[index];
                    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    barrier.buffer = device.Resolve(outputs[index]);
                    barrier.size = VK_WHOLE_SIZE;
                }
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     0, 0, nullptr, 2, barriers.data(), 0, nullptr);
                for (uint32_t index = 0; index < 2; ++index) {
                    VkBufferCopy copy{0, index == 0 ? 0u : 16u, index == 0 ? 16u : 20u};
                    vkCmdCopyBuffer(command, barriers[index].buffer, readback.buffer, 1, &copy);
                }
                VkBufferMemoryBarrier hostBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
                hostBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                hostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                hostBarrier.srcQueueFamilyIndex = hostBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                hostBarrier.buffer = readback.buffer;
                hostBarrier.size = VK_WHOLE_SIZE;
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                                     0, 0, nullptr, 1, &hostBarrier, 0, nullptr);
                if (!Require(vkEndCommandBuffer(command) == VK_SUCCESS, "Cull command end failed")) return false;
                VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                submit.commandBufferCount = 1;
                submit.pCommandBuffers = &command;
                const VkQueue queue = resources.context.GetGraphicsQueue();
                const auto submitted = vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
                const auto waited = vkQueueWaitIdle(queue);
                if (!Require(submitted == VK_SUCCESS && waited == VK_SUCCESS, "Cull GPU submission failed")) return false;
                if (!Require(vmaInvalidateAllocation(readback.allocator, readback.allocation, 0, 36) == VK_SUCCESS,
                             "Cull readback invalidation failed")) return false;
                const auto *words = static_cast<const uint32_t *>(readback.mapped);
                if (!Require(words[0] == 6 && words[1] == expected && words[2] == 7 && words[3] == 9 &&
                             words[4] == (expected + 255) / 256 && words[5] == 1 && words[6] == 1 && words[7] == sourceCount,
                             "Production indirect Cull lost visible instances or dispatch arguments")) return false;
            }
            std::cout << "Cull conservation mode=" << (ribbon ? "ribbon" : "instances")
                      << " count=" << test.count << " coarse=" << test.coarse << " visible=" << expected << '\n';
        }
    }
    return true;
}
