#include <SDL3/SDL.h>
#include <function/renderer/CommandBuffer.h>
#include <function/renderer/MaterialDescriptor.h>
#include <function/renderer/rhi/RhiComputeBuffer.h>
#include <function/renderer/rhi/RhiComputeHost.h>
#include <function/renderer/vk/RenderGraph.h>
#include <function/renderer/vk/VkDeviceContext.h>
#include <function/renderer/vk/VulkanComputeQueue.h>
#include <function/renderer/vk/VulkanFrameSubmission.h>
#include <function/renderer/vk/VulkanQueueManager.h>
#include <function/renderer/vk/VulkanRhiDevice.h>
#include <function/renderer/vk/VulkanSubmissionExecutor.h>
#include <function/resources/InxMaterial/InxMaterial.h>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>

using namespace infernux;

enum class ResourceKind
{
    Color,
    Depth,
    Buffer
};

static std::vector<uint32_t> ReadSpirv(const char *path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    assert(input);
    const auto bytes = static_cast<size_t>(input.tellg());
    assert(bytes > 0 && bytes % sizeof(uint32_t) == 0);
    std::vector<uint32_t> code(bytes / sizeof(uint32_t));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(code.data()), static_cast<std::streamsize>(bytes));
    assert(input);
    return code;
}

static std::vector<char> SpirvBytes(const std::vector<uint32_t> &words)
{
    std::vector<char> bytes(words.size() * sizeof(uint32_t));
    std::memcpy(bytes.data(), words.data(), bytes.size());
    return bytes;
}

static VkPipeline CreateFullscreenPipeline(VkDevice device, VkShaderModule vertex, VkShaderModule fragment,
                                           VkPipelineLayout layout)
{
    std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertex;
    stages[0].pName = "main";
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragment;
    stages[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState colorAttachment{};
    colorAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments = &colorAttachment;
    constexpr std::array<VkDynamicState, 2> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamic.pDynamicStates = dynamicStates.data();
    const VkFormat colorFormat = VK_FORMAT_R8G8B8A8_UNORM;
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &colorFormat;
    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    info.pNext = &rendering;
    info.stageCount = static_cast<uint32_t>(stages.size());
    info.pStages = stages.data();
    info.pVertexInputState = &vertexInput;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pDepthStencilState = &depth;
    info.pColorBlendState = &blend;
    info.pDynamicState = &dynamic;
    info.layout = layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    assert(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) == VK_SUCCESS);
    return pipeline;
}

static void CheckRendererParameterBuffer(vk::VkDeviceContext &context, vk::VulkanQueueManager &queues,
                                         vk::VulkanSubmissionExecutor &executor, VkFence fence,
                                         const std::vector<uint32_t> &computeCode,
                                         const std::vector<uint32_t> &vertexCode,
                                         const std::vector<uint32_t> &fragmentCode)
{
    auto &device = context.GetRhiDevice();

    // Use the public inx.buffer backing object. The second vec4 is authored on
    // the host, copied into device-local storage by the engine compute queue,
    // and then copied to the first vec4 by a real compute dispatch below.
    vk::VulkanComputeQueue computeQueue;
    assert(computeQueue.Initialize(context, queues, 2));
    rhi::ComputeHost computeHost(device, computeQueue);
    auto parameters = std::make_shared<rhi::ComputeBuffer>(
        computeHost, rhi::ComputeBufferDesc{2, rhi::ComputeScalarType::Float32, 4});
    const std::array<float, 8> authored = {
        1.0f, 0.0f, 0.0f, 1.0f, 32.0f / 255.0f, 128.0f / 255.0f, 224.0f / 255.0f, 1.0f,
    };
    parameters->SetData(0, authored.data(), sizeof(authored));
    computeQueue.Wait(parameters->GetLastWriteSubmission());
    const auto hostRoundTrip = parameters->GetData(0, sizeof(authored));
    assert(hostRoundTrip.size() == sizeof(authored));
    assert(std::memcmp(hostRoundTrip.data(), authored.data(), sizeof(authored)) == 0);

    // Draw parameters expose one reflected binding namespace. A field cannot
    // silently exist as both a value and a storage buffer; callers must remove
    // the old binding before changing its kind.
    DrawParameterBlock bufferFirst;
    bufferFirst.SetBuffer("RendererParameters", parameters);
    bool rejectedConflictingBinding = false;
    try {
        bufferFirst.SetFloat("RendererParameters", 1.0f);
    } catch (const std::invalid_argument &) {
        rejectedConflictingBinding = true;
    }
    assert(rejectedConflictingBinding);
    DrawParameterBlock valueFirst;
    valueFirst.SetFloat("RendererParameters", 1.0f);
    rejectedConflictingBinding = false;
    try {
        valueFirst.SetBuffer("RendererParameters", parameters);
    } catch (const std::invalid_argument &) {
        rejectedConflictingBinding = true;
    }
    assert(rejectedConflictingBinding);

    // Authoring order must not depend on the renderer having already
    // published the material's first Forward program. Reflection validation
    // happens when that real pass is resolved below.
    auto material = std::make_shared<InxMaterial>("renderer-parameter-buffer");
    material->SetBuffer("RendererParameters", parameters);
    DrawParameterBlock prepublishedParameters;
    prepublishedParameters.SetBuffer("RendererParameters", parameters);
    auto publication = prepublishedParameters.Capture(*material);
    assert(publication && publication->buffers.at("RendererParameters") == parameters);

    rhi::BindingLayoutDesc computeLayoutDesc;
    computeLayoutDesc.entries[0] = {0, rhi::BindingType::StorageBuffer, rhi::ShaderStage::Compute, 1};
    computeLayoutDesc.entryCount = 1;
    const auto computeLayout = device.CreateBindingLayout(computeLayoutDesc);
    rhi::BindGroupDesc groupDesc;
    groupDesc.layout = computeLayout;
    groupDesc.buffers[0] = {0, rhi::BindingType::StorageBuffer, parameters->GetBuffer(), 0, parameters->GetByteSize()};
    groupDesc.bufferCount = 1;
    const auto group = device.CreateBindGroup(groupDesc);
    const auto computeShader =
        device.CreateShaderModule(rhi::ShaderModuleDesc::FromSpirV(computeCode.data(), computeCode.size()));
    auto program = std::make_shared<ShaderProgram>();
    ShaderProgramVariantKey programKey;
    programKey.program.stages = {"Tests/RendererParameterBuffer.vert", "Tests/RendererParameterBuffer.frag"};
    programKey.program.revision = 1;
    programKey.target = ShaderCompileTarget::Forward;
    assert(program->Create(context.GetDevice(), SpirvBytes(vertexCode), SpirvBytes(fragmentCode), programKey));
    const auto storageBinding = std::find_if(
        program->GetDescriptorBindings().begin(), program->GetDescriptorBindings().end(),
        [](const MergedDescriptorBinding &binding) {
            return binding.set == 0 && binding.binding == 0 && binding.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        });
    assert(storageBinding != program->GetDescriptorBindings().end() && storageBinding->name == "RendererParameters");
    rhi::ComputePipelineDesc computeDesc;
    computeDesc.computeShader = computeShader;
    computeDesc.bindingLayouts[0] = computeLayout;
    computeDesc.bindingLayoutCount = 1;
    const auto computePipeline = device.CreateComputePipeline(computeDesc);
    const VkPipeline nativeGraphicsPipeline = CreateFullscreenPipeline(
        context.GetDevice(), program->GetVertexModule(), program->GetFragmentModule(), program->GetPipelineLayout());
    const auto graphicsPipeline = device.RegisterGraphicsPipeline(nativeGraphicsPipeline, program->GetPipelineLayout());
    assert(computeLayout.IsValid() && group.IsValid() && computeShader.IsValid() && computePipeline.IsValid() &&
           graphicsPipeline.IsValid());

    // Build the renderer descriptor through the production reflected set-0
    // path. The graphics pipeline's independent RHI layout is Vulkan-compatible
    // with this reflected layout, so the draw below consumes the exact
    // descriptor publication used by material/per-draw renderer parameters.
    material->SetPassShaderProgram(ShaderCompileTarget::Forward, program);
    vk::VkDescriptorManager descriptorAllocator(context.GetDevice(), context.GetDeviceId());
    MaterialDescriptorManager descriptors;
    rhi::SubmissionSerial rendererRetirementSerial = 40;
    GpuRetirementQueue rendererRetirement;
    rendererRetirement.BindSerialSource([&] { return rendererRetirementSerial; });
    descriptors.Initialize(context.GetVmaAllocator(), context.GetDevice(), context.GetPhysicalDevice(),
                           &descriptorAllocator);
    descriptors.SetRetirementQueue(&rendererRetirement);
    descriptors.SetBufferResolver([&device](const std::shared_ptr<rhi::ComputeBuffer> &buffer) {
        if (!buffer || buffer->GetBuffer().Device() != device.GetDeviceId())
            return VkDescriptorBufferInfo{};
        return VkDescriptorBufferInfo{device.Resolve(buffer->GetBuffer()), 0, buffer->GetByteSize()};
    });
    MaterialDescriptorSet *rendererDescriptor =
        descriptors.GetOrCreateRendererDescriptorSet(*material, *program, publication);
    assert(rendererDescriptor && rendererDescriptor->isValid && !rendererDescriptor->hasUnboundRequiredBuffers &&
           rendererDescriptor->descriptorSet != VK_NULL_HANDLE &&
           rendererDescriptor->storageBufferBindings.at(storageBinding->binding) == parameters);
    assert(program->GetPipelineLayout() != VK_NULL_HANDLE);

    // Semantic passes consume the Forward material set-0 ABI. Calling the
    // renderer publication path with another compatible program must neither
    // replace the Forward base descriptor nor create a pass-local override
    // for the same immutable parameter block.
    auto semanticProgram = std::make_shared<ShaderProgram>();
    auto semanticKey = programKey;
    semanticKey.target = ShaderCompileTarget::GBuffer;
    assert(semanticProgram->Create(context.GetDevice(), SpirvBytes(vertexCode), SpirvBytes(fragmentCode), semanticKey));
    material->SetPassShaderProgram(ShaderCompileTarget::GBuffer, semanticProgram);
    MaterialDescriptorSet *forwardBase = descriptors.GetOrCreateDescriptorSet(*material, *program);
    auto emptyPublication = std::make_shared<RendererParameterBlock>();
    assert(descriptors.GetOrCreateRendererDescriptorSet(*material, *semanticProgram, emptyPublication) == forwardBase);
    assert(descriptors.GetOrCreateRendererDescriptorSet(*material, *semanticProgram, publication) ==
           rendererDescriptor);

    rhi::BufferDesc readbackDesc;
    readbackDesc.byteSize = 4;
    readbackDesc.usage = rhi::BufferUsageFlags::TransferDestination;
    readbackDesc.memory = rhi::BufferMemory::Readback;
    readbackDesc.queueAccess = rhi::QueueAccessFlags::Graphics;
    const auto readback = device.CreateBuffer(readbackDesc);
    assert(readback.IsValid());

    vk::RenderGraph graph;
    graph.Initialize(&context, nullptr, &queues);
    vk::ResourceHandle residentParameters;
    vk::ResourceHandle color;
    graph.AddComputePass("Write inx.buffer", [&](vk::PassBuilder &builder) {
        residentParameters =
            builder.ImportBuffer("RendererParameters", parameters->GetBuffer(), parameters->GetByteSize());
        graph.SetResourceInitialState(residentParameters, rhi::TextureLayout::Undefined,
                                      rhi::Access::ShaderRead | rhi::Access::ShaderWrite,
                                      rhi::PipelineStage::ComputeShader, rhi::QueueRole::Compute);
        residentParameters = builder.WriteStorageBuffer(residentParameters);
        return [&](vk::RenderContext &commands) {
            auto &encoder = commands.GetComputeCommandEncoder();
            encoder.BindPipeline(computePipeline);
            encoder.BindGroup(computePipeline, 0, group);
            encoder.Dispatch(1, 1, 1);
        };
    });
    graph.AddPass("Consume set0 storage buffer", [&](vk::PassBuilder &builder) {
        builder.ReadStorageBuffer(residentParameters, rhi::PipelineStage::FragmentShader);
        color = builder.CreateTexture("RendererParameterPixel", 1, 1, VK_FORMAT_R8G8B8A8_UNORM);
        color = builder.WriteColor(color);
        builder.SetClearColor(1.0f, 0.0f, 1.0f, 1.0f);
        builder.SetRenderArea(1, 1);
        return [&](vk::RenderContext &commands) {
            auto &encoder = commands.GetGraphicsCommandEncoder();
            // Raw command-buffer access deliberately invalidates the RHI
            // encoder's binding cache. Record every raw command first, then
            // bind the registered pipeline immediately before the RHI draw so
            // the encoder cannot correctly elide the draw as unbound.
            const VkCommandBuffer commandBuffer = commands.GetCommandBuffer();
            const VkViewport viewport{0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
            const VkRect2D scissor{{0, 0}, {1, 1}};
            vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
            vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
            const VkDescriptorSet set = rendererDescriptor->descriptorSet;
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, program->GetPipelineLayout(), 0, 1,
                                    &set, 0, nullptr);
            encoder.BindPipeline(graphicsPipeline);
            encoder.Draw(3);
        };
    });
    graph.AddTransferPass("Read renderer parameter pixel", [&](vk::PassBuilder &builder) {
        builder.SetQueueRole(rhi::QueueRole::Graphics);
        builder.TransferRead(color);
        builder.TransferWrite(builder.ImportBuffer("RendererParameterReadback", readback, 4));
        builder.SetSideEffect();
        return [&](vk::RenderContext &commands) {
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {1, 1, 1};
            vkCmdCopyImageToBuffer(commands.GetCommandBuffer(), device.Resolve(commands.GetTextureHandle(color)),
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, device.Resolve(readback), 1, &copy);
            VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = device.Resolve(readback);
            barrier.size = 4;
            vkCmdPipelineBarrier(commands.GetCommandBuffer(), VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
        };
    });
    assert(graph.Compile());

    const auto &graphPlan = graph.GetSubmissionPlan();
    assert(graphPlan.batches.size() >= 2);
    assert(graphPlan.batches.front().queue == rhi::QueueRole::Compute);
    assert(graphPlan.batches.back().queue == rhi::QueueRole::Graphics);
    // inx.buffer is created for concurrent graphics/compute/transfer access:
    // lane changes require a timeline dependency and memory barrier, but must
    // not emit illegal exclusive-family ownership transfers.
    assert(graph.GetQueueOwnershipTransfers().empty());
    const auto computeLane = queues.GetSnapshot(rhi::QueueRole::Compute).nativeLane;
    const auto graphicsLane = queues.GetSnapshot(rhi::QueueRole::Graphics).nativeLane;
    if (computeLane != graphicsLane) {
        const auto graphicsBatch =
            std::find_if(graphPlan.batches.begin(), graphPlan.batches.end(),
                         [](const rhi::SubmissionBatch &batch) { return batch.queue == rhi::QueueRole::Graphics; });
        assert(graphicsBatch != graphPlan.batches.end());
        const auto &waits = graphicsBatch->waitsFor;
        assert(std::any_of(waits.begin(), waits.end(), [](const rhi::SubmissionBatchDependency &dependency) {
            return dependency.sourceBatch == 0 &&
                   rhi::HasAny(dependency.waitStages, rhi::PipelineStage::FragmentShader);
        }));
    }

    vk::VulkanFrameSubmission submission;
    submission.Reset();
    assert(!submission.AppendRenderGraph(graph).Empty());
    rhi::SubmissionPlan plan;
    std::string error;
    assert(submission.Build(plan, error));
    assert(vkResetFences(context.GetDevice(), 1, &fence) == VK_SUCCESS);
    vk::VulkanSubmissionExecutor::ExternalSync sync{};
    sync.completionFence = fence;
    sync.completionEpoch = queues.ReserveCompletionEpoch();
    const auto result = executor.Execute(
        0, plan,
        [&](uint32_t batch, VkCommandBuffer commands) { return submission.RecordBatch(plan, batch, commands); }, sync);
    assert(result.Succeeded());
    assert(vkWaitForFences(context.GetDevice(), 1, &fence, VK_TRUE, 5'000'000'000ull) == VK_SUCCESS);
    executor.CompleteFrame(0);
    queues.CompleteCompletionEpoch(sync.completionEpoch);
    const auto computedRoundTrip = parameters->GetData(0, sizeof(authored));
    assert(computedRoundTrip.size() == sizeof(authored));
    std::array<float, 8> computed{};
    std::memcpy(computed.data(), computedRoundTrip.data(), computedRoundTrip.size());
    assert(std::equal(computed.begin(), computed.begin() + 4, authored.begin() + 4));
    std::array<uint8_t, 4> pixel{};
    assert(device.ReadBuffer(readback, 0, pixel.data(), pixel.size()));
    if (pixel != std::array<uint8_t, 4>{32, 128, 224, 255}) {
        std::cerr << "Renderer parameter pixel mismatch: " << static_cast<unsigned>(pixel[0]) << ','
                  << static_cast<unsigned>(pixel[1]) << ',' << static_cast<unsigned>(pixel[2]) << ','
                  << static_cast<unsigned>(pixel[3]) << '\n';
    }
    assert((pixel == std::array<uint8_t, 4>{32, 128, 224, 255}));

    // Renderer parameter publications are immutable submission owners. A
    // texture hot-reload/delete follows this same publication boundary: the
    // replacement descriptor may become live immediately, while the old one
    // remains owned until both the captured CPU publication and its exact GPU
    // submission serial have retired.
    auto replacementPublication = std::make_shared<RendererParameterBlock>();
    replacementPublication->buffers.emplace(storageBinding->name, parameters);
    replacementPublication->revision = 2;
    MaterialDescriptorSet *replacementDescriptor =
        descriptors.GetOrCreateRendererDescriptorSet(*material, *program, replacementPublication);
    assert(replacementDescriptor && replacementDescriptor != rendererDescriptor);
    assert(descriptors.CollectExpiredRendererDescriptorSets() == 0);
    publication.reset();
    assert(descriptors.CollectExpiredRendererDescriptorSets() == 1);
    assert(descriptors.GetRetiredDescriptorSetCount() == 1 && rendererRetirement.PendingCount() == 1);
    assert(rendererRetirement.Collect(rendererRetirementSerial - 1) == 0);
    assert(descriptors.GetRetiredDescriptorSetCount() == 1);
    assert(rendererRetirement.Collect(rendererRetirementSerial) == 1);
    assert(descriptors.GetRetiredDescriptorSetCount() == 0);
    replacementPublication.reset();
    assert(descriptors.CollectExpiredRendererDescriptorSets() == 1);
    assert(rendererRetirement.Collect(rendererRetirementSerial) == 1);
    assert(descriptors.GetRetiredDescriptorSetCount() == 0);

    graph.Destroy();
    device.Release(readback);
    device.Release(graphicsPipeline);
    vkDestroyPipeline(context.GetDevice(), nativeGraphicsPipeline, nullptr);
    device.Release(computePipeline);
    device.Release(computeShader);
    device.Release(group);
    device.Release(computeLayout);
    descriptors.Shutdown();
    descriptorAllocator.Destroy();
    material.reset();
    semanticProgram.reset();
    program.reset();
    parameters.reset();
    computeQueue.Destroy();
}

static void CheckReplay(vk::VkDeviceContext &context, vk::VulkanQueueManager &queues,
                        vk::VulkanSubmissionExecutor &executor, VkFence fence, ResourceKind kind,
                        rhi::QueueRole producer, rhi::QueueRole consumer, rhi::BindingLayoutHandle layout,
                        rhi::ComputePipelineHandle pipeline, rhi::SamplerHandle sampler)
{
    auto &device = context.GetRhiDevice();
    rhi::BufferDesc desc;
    desc.byteSize = 8 * 4 * 4 * (kind == ResourceKind::Depth ? 4 : 1);
    desc.memory = rhi::BufferMemory::Readback;
    desc.usage =
        kind == ResourceKind::Depth ? rhi::BufferUsageFlags::Storage : rhi::BufferUsageFlags::TransferDestination;
    desc.queueAccess = consumer == rhi::QueueRole::Compute    ? rhi::QueueAccessFlags::Compute
                       : consumer == rhi::QueueRole::Transfer ? rhi::QueueAccessFlags::Transfer
                                                              : rhi::QueueAccessFlags::Graphics;
    const auto readback = device.CreateBuffer(desc);
    assert(readback.IsValid());
    rhi::BufferHandle upload;
    if (kind == ResourceKind::Color && producer == rhi::QueueRole::Transfer) {
        auto uploadDesc = desc;
        uploadDesc.memory = rhi::BufferMemory::Upload;
        uploadDesc.usage = rhi::BufferUsageFlags::TransferSource;
        uploadDesc.queueAccess = rhi::QueueAccessFlags::Transfer;
        upload = device.CreateBuffer(uploadDesc);
        assert(upload.IsValid());
    }
    {
        vk::RenderGraph graph;
        graph.Initialize(&context, nullptr, &queues);
        auto image =
            kind == ResourceKind::Buffer
                ? graph.RegisterTransientBuffer("ReplayBuffer", desc.byteSize,
                                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)
                : graph.RegisterTransientTexture("ReplayImage", 8, 4,
                                                 kind == ResourceKind::Depth ? VK_FORMAT_D32_SFLOAT
                                                                             : VK_FORMAT_R8G8B8A8_UNORM);
        uint32_t frame = 0;
        rhi::BindGroupHandle group;
        graph.AddTransferPass("Produce", [&](vk::PassBuilder &builder) {
            builder.SetQueueRole(producer);
            if (upload.IsValid())
                builder.TransferRead(builder.ImportBuffer("ReplayUpload", upload, desc.byteSize));
            image = builder.TransferWrite(image);
            return [&](vk::RenderContext &commands) {
                if (kind == ResourceKind::Buffer) {
                    vkCmdFillBuffer(commands.GetCommandBuffer(), commands.GetBuffer(image), 0, desc.byteSize,
                                    0x01020300u + frame);
                    return;
                }
                if (kind == ResourceKind::Depth) {
                    VkClearDepthStencilValue depth{(frame + 1) / 8.0f, 0};
                    VkImageSubresourceRange range{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
                    vkCmdClearDepthStencilImage(commands.GetCommandBuffer(),
                                                device.Resolve(commands.GetTextureHandle(image)),
                                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &depth, 1, &range);
                    return;
                }
                if (upload.IsValid()) {
                    VkBufferImageCopy copy{};
                    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    copy.imageExtent = {8, 4, 1};
                    vkCmdCopyBufferToImage(commands.GetCommandBuffer(), device.Resolve(upload),
                                           device.Resolve(commands.GetTextureHandle(image)),
                                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
                    return;
                }
                VkClearColorValue color{};
                color.float32[frame % 3] = 1.0f;
                color.float32[3] = 1.0f;
                VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdClearColorImage(commands.GetCommandBuffer(), device.Resolve(commands.GetTextureHandle(image)),
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);
            };
        });
        if (kind == ResourceKind::Depth) {
            // Depth clear/copy requires Graphics without optional maintenance
            // features. Sample it with a real compute shader to exercise the
            // Graphics -> Compute -> next-frame Graphics ownership cycle.
            assert(producer == rhi::QueueRole::Graphics && consumer != rhi::QueueRole::Transfer);
            graph.AddComputePass("SampleDepth", [&](vk::PassBuilder &builder) {
                builder.SetQueueRole(consumer);
                builder.Read(image, rhi::PipelineStage::ComputeShader);
                builder.WriteStorageBuffer(builder.ImportBuffer("DepthReadback", readback, desc.byteSize));
                builder.SetSideEffect();
                return [&](vk::RenderContext &commands) {
                    auto &encoder = commands.GetComputeCommandEncoder();
                    encoder.BindPipeline(pipeline);
                    encoder.BindGroup(pipeline, 0, group);
                    encoder.Dispatch(1, 1, 1);
                    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
                    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    barrier.buffer = device.Resolve(readback);
                    barrier.size = desc.byteSize;
                    vkCmdPipelineBarrier(commands.GetCommandBuffer(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
                };
            });
        } else
            graph.AddTransferPass("Consume", [&](vk::PassBuilder &builder) {
                builder.SetQueueRole(consumer);
                builder.TransferRead(image);
                builder.TransferWrite(builder.ImportBuffer("ReplayReadback", readback, desc.byteSize));
                builder.SetSideEffect();
                return [&](vk::RenderContext &commands) {
                    if (kind == ResourceKind::Buffer) {
                        VkBufferCopy copy{0, 0, desc.byteSize};
                        vkCmdCopyBuffer(commands.GetCommandBuffer(), commands.GetBuffer(image),
                                        device.Resolve(readback), 1, &copy);
                    } else {
                        VkBufferImageCopy copy{};
                        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                        copy.imageExtent = {8, 4, 1};
                        vkCmdCopyImageToBuffer(
                            commands.GetCommandBuffer(), device.Resolve(commands.GetTextureHandle(image)),
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, device.Resolve(readback), 1, &copy);
                    }
                    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
                    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    barrier.buffer = device.Resolve(readback);
                    barrier.size = desc.byteSize;
                    vkCmdPipelineBarrier(commands.GetCommandBuffer(), VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
                };
            });
        assert(graph.Compile());
        if (kind == ResourceKind::Depth) {
            rhi::BindGroupDesc groupDesc;
            groupDesc.layout = layout;
            groupDesc.textures[0] = {0, rhi::BindingType::CombinedTextureSampler, graph.ResolveRhiTextureView(image),
                                     sampler};
            groupDesc.textureCount = 1;
            groupDesc.buffers[0] = {1, rhi::BindingType::StorageBuffer, readback, 0, desc.byteSize};
            groupDesc.bufferCount = 1;
            group = device.CreateBindGroup(groupDesc);
            assert(group.IsValid());
        }
        assert(graph.GetSubmissionPlan().batches.front().queue == producer);
        assert(graph.GetSubmissionPlan().batches.back().queue == consumer);
        const bool crossFamily = queues.GetSnapshot(producer).family != queues.GetSnapshot(consumer).family;
        assert(!graph.HasExternalQueueOwnershipReleases(consumer));
        for (; frame < 6; ++frame) {
            if (upload.IsValid()) {
                std::array<uint32_t, 8 * 4> source;
                source.fill(0xff000000u | (0xffu << (8 * (frame % 3))));
                assert(device.WriteBuffer(upload, 0, source.data(), sizeof(source)));
            }
            if (frame == 2) {
                vk::RenderGraph moved(std::move(graph));
                assert(!graph.HasExternalQueueOwnershipReleases(consumer));
                assert(moved.HasExternalQueueOwnershipReleases(consumer) == crossFamily);
                graph = std::move(moved);
                assert(!moved.HasExternalQueueOwnershipReleases(consumer));
            }
            if (frame == 3)
                assert(graph.Compile()); // Recompile without replacing the physical allocation.
            const bool pendingRelease = graph.HasExternalQueueOwnershipReleases(consumer);
            if (pendingRelease != (crossFamily && frame != 0))
                std::cerr << "Release mismatch kind=" << static_cast<int>(kind)
                          << " producer=" << static_cast<int>(producer) << " consumer=" << static_cast<int>(consumer)
                          << " frame=" << frame << " transfers=" << graph.GetQueueOwnershipTransfers().size()
                          << " pending=" << pendingRelease << std::endl;
            assert(pendingRelease == (crossFamily && frame != 0));
            vk::VulkanFrameSubmission submission;
            submission.Reset();
            std::vector<uint32_t> releases;
            for (auto role : {rhi::QueueRole::Graphics, rhi::QueueRole::Compute, rhi::QueueRole::Transfer}) {
                if (graph.HasExternalQueueOwnershipReleases(role))
                    releases.push_back(submission.AddWork(
                        context.GetDeviceId(), role, rhi::SubmissionDomain::Frame, rhi::InvalidRenderViewId,
                        rhi::PipelineStage::AllCommands, {},
                        [&, role](VkCommandBuffer commands) {
                            return graph.RecordExternalQueueOwnershipReleases(role, commands);
                        },
                        "ReplayRelease"));
            }
            assert(!submission.AppendRenderGraph(graph, releases).Empty());
            rhi::SubmissionPlan plan;
            std::string error;
            assert(submission.Build(plan, error));
            assert(vkResetFences(context.GetDevice(), 1, &fence) == VK_SUCCESS);
            vk::VulkanSubmissionExecutor::ExternalSync sync{};
            sync.completionFence = fence;
            sync.completionEpoch = queues.ReserveCompletionEpoch();
            const uint32_t slot = frame % 2;
            const auto result = executor.Execute(
                slot, plan,
                [&](uint32_t batch, VkCommandBuffer commands) { return submission.RecordBatch(plan, batch, commands); },
                sync);
            assert(result.Succeeded());
            assert(vkWaitForFences(context.GetDevice(), 1, &fence, VK_TRUE, 5'000'000'000ull) == VK_SUCCESS);
            executor.CompleteFrame(slot);
            queues.CompleteCompletionEpoch(sync.completionEpoch);
            std::vector<uint32_t> pixels(desc.byteSize / sizeof(uint32_t));
            assert(device.ReadBuffer(readback, 0, pixels.data(), desc.byteSize));
            uint32_t expected = 0x01020300u + frame;
            if (kind == ResourceKind::Depth) {
                const float depth = (frame + 1) / 8.0f;
                std::memcpy(&expected, &depth, sizeof(expected));
            } else if (kind == ResourceKind::Color) {
                std::array<uint8_t, 4> color{};
                color[frame % 3] = color[3] = 255;
                std::memcpy(&expected, color.data(), sizeof(expected));
            }
            for (size_t i = 0; i < pixels.size(); ++i) {
                uint32_t channelExpected = expected;
                if (kind == ResourceKind::Depth && i % 4 != 0)
                    channelExpected = i % 4 == 3 ? 0x3f800000u : 0;
                const auto pixel = pixels[i];
                if (pixel != channelExpected)
                    std::cerr << "Readback mismatch kind=" << static_cast<int>(kind)
                              << " producer=" << static_cast<int>(producer)
                              << " consumer=" << static_cast<int>(consumer) << " frame=" << frame << " actual=" << pixel
                              << " expected=" << channelExpected << std::endl;
                assert(pixel == channelExpected);
            }
        }
        graph.Reset();
        assert(!graph.HasExternalQueueOwnershipReleases(consumer));
        if (group.IsValid())
            device.Release(group);
    }
    device.Release(readback);
    if (upload.IsValid())
        device.Release(upload);
}

int main(int argc, char **argv)
{
    assert(argc == 5);
    const auto code = ReadSpirv(argv[1]);
    const auto rendererParameterCompute = ReadSpirv(argv[2]);
    const auto rendererParameterVertex = ReadSpirv(argv[3]);
    const auto rendererParameterFragment = ReadSpirv(argv[4]);
    assert(SDL_Init(SDL_INIT_VIDEO));
    auto *window = SDL_CreateWindow("RenderGraph queue replay", 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
    assert(window);
    vk::VkDeviceContext context;
    vk::DeviceConfig config;
    config.enableValidationLayers = true;
    assert(context.Initialize(window, config));
    auto &device = context.GetRhiDevice();
    const auto shader = device.CreateShaderModule(rhi::ShaderModuleDesc::FromSpirV(code.data(), code.size()));
    rhi::BindingLayoutDesc layoutDesc;
    layoutDesc.entries[0] = {0, rhi::BindingType::CombinedTextureSampler, rhi::ShaderStage::Compute, 1};
    layoutDesc.entries[1] = {1, rhi::BindingType::StorageBuffer, rhi::ShaderStage::Compute, 1};
    layoutDesc.entryCount = 2;
    const auto layout = device.CreateBindingLayout(layoutDesc);
    rhi::ComputePipelineDesc pipelineDesc;
    pipelineDesc.computeShader = shader;
    pipelineDesc.bindingLayouts[0] = layout;
    pipelineDesc.bindingLayoutCount = 1;
    const auto pipeline = device.CreateComputePipeline(pipelineDesc);
    rhi::SamplerDesc samplerDesc;
    samplerDesc.minFilter = samplerDesc.magFilter = samplerDesc.mipFilter = rhi::FilterMode::Nearest;
    const auto sampler = device.CreateSampler(samplerDesc);
    assert(shader.IsValid() && layout.IsValid() && pipeline.IsValid() && sampler.IsValid());
    vk::VulkanQueueManager queues;
    assert(queues.Initialize(context, 2));
    vk::VulkanSubmissionExecutor executor;
    assert(executor.Initialize(context, queues, 2));
    std::cout << "Replay queue families: graphics=" << queues.GetSnapshot(rhi::QueueRole::Graphics).family
              << " compute=" << queues.GetSnapshot(rhi::QueueRole::Compute).family
              << " transfer=" << queues.GetSnapshot(rhi::QueueRole::Transfer).family << std::endl;
    VkFenceCreateInfo info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence{};
    assert(vkCreateFence(context.GetDevice(), &info, nullptr, &fence) == VK_SUCCESS);
    CheckRendererParameterBuffer(context, queues, executor, fence, rendererParameterCompute, rendererParameterVertex,
                                 rendererParameterFragment);
    for (auto kind : {ResourceKind::Color, ResourceKind::Buffer}) {
        for (auto producer : {rhi::QueueRole::Graphics, rhi::QueueRole::Compute, rhi::QueueRole::Transfer}) {
            for (auto consumer : {rhi::QueueRole::Graphics, rhi::QueueRole::Compute, rhi::QueueRole::Transfer})
                CheckReplay(context, queues, executor, fence, kind, producer, consumer, layout, pipeline, sampler);
        }
    }
    for (auto consumer : {rhi::QueueRole::Graphics, rhi::QueueRole::Compute})
        CheckReplay(context, queues, executor, fence, ResourceKind::Depth, rhi::QueueRole::Graphics, consumer, layout,
                    pipeline, sampler);
    std::cout << "Replay verified: color/buffer x 9 queue pairs + depth x 2 sampling queues; 120 executions"
              << std::endl;
    vkDestroyFence(context.GetDevice(), fence, nullptr);
    executor.Destroy();
    queues.Destroy();
    device.Release(sampler);
    device.Release(pipeline);
    device.Release(layout);
    device.Release(shader);
    context.Destroy();
    SDL_DestroyWindow(window);
    SDL_Quit();
}
