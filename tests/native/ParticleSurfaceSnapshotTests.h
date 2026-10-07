bool VerifyParticleSurfaceSnapshots(TestResources &resources, infernux::InxShaderLoader &compiler)
{
    using namespace infernux;
    auto &device = resources.context.GetRhiDevice();
    const auto vertex = SpirvWords(compiler.CompileVertexGlsl(R"glsl(
#version 450
void main() {
    vec2 positions[3] = vec2[](vec2(-1,-1), vec2(3,-1), vec2(-1,3));
    gl_Position = vec4(positions[gl_VertexIndex], 0, 1);
}
)glsl",
                                                              "Tests/SurfaceSnapshot.vert"));
    const auto fragment = SpirvWords(compiler.CompileFragmentGlsl(R"glsl(
#version 450
layout(std140, set=0, binding=14) uniform Material { vec4 color; };
layout(location=0) out vec4 output_color;
void main() { output_color = color; }
)glsl",
                                                                  "Tests/SurfaceSnapshot.frag"));
    if (!Require(!vertex.empty() && !fragment.empty(), "Surface snapshot shader compilation failed"))
        return false;
    struct Owner
    {
        vk::VkDeviceContext &context;
        vk::VulkanRhiDevice &device;
        vk::VulkanQueueManager queues;
        vk::VulkanSubmissionExecutor executor;
        GpuRetirementQueue retirement;
        particle::ParticleGpuSurfaceBinding surface;
        rhi::TextureHandle target;
        rhi::TextureViewHandle view;
        rhi::BufferHandle readback;
        rhi::ShaderModuleHandle vertex, fragment;
        rhi::GraphicsPipelineHandle pipeline;
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
            surface.Destroy();
            retirement.FlushAll();
            queues.Destroy();
            device.Release(pipeline);
            device.Release(vertex);
            device.Release(fragment);
            device.Release(view);
            device.Release(target);
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
                 "Surface snapshot executor creation failed"))
        return false;
    owner.retirement.BindSerialSource([&] { return owner.queues.GetLastReservedCompletionEpoch(); });
    auto material = std::make_shared<InxMaterial>("GPU surface snapshot");
    material->SetVector4("color", glm::vec4(1));
    auto artifact = std::make_shared<ShaderProgramArtifact>();
    artifact->key = {{"Tests/SurfaceSnapshot", "Tests/SurfaceSnapshot"}, 1};
    artifact->domain = ShaderProgramDomain::ParticleSprite;
    artifact->compatibilitySignature = 79;
    artifact->materialBufferSize = 16;
    artifact->properties = {{"color", "Float4", "[1,1,1,1]", "", ShaderProgramStageMask::Fragment, false, std::nullopt,
                             0, std::nullopt, 16, 16}};
    ShaderProgramArtifact::PassVariant variant;
    variant.compatibilitySignature = 79;
    variant.vertexSpirv = SpirvBytes(vertex);
    variant.fragmentSpirv = SpirvBytes(fragment);
    artifact->variants.push_back(std::move(variant));
    if (!Require(owner.surface.Create(device, artifact, material, {}, {}, {}, &owner.retirement),
                 "Surface snapshot creation failed"))
        return false;
    owner.vertex = device.CreateShaderModule(rhi::ShaderModuleDesc::FromSpirV(vertex.data(), vertex.size()));
    owner.fragment = device.CreateShaderModule(rhi::ShaderModuleDesc::FromSpirV(fragment.data(), fragment.size()));
    rhi::GraphicsPipelineDesc pipeline;
    pipeline.vertexShader = owner.vertex;
    pipeline.fragmentShader = owner.fragment;
    pipeline.raster.cullMode = rhi::CullMode::None;
    pipeline.bindingLayoutCount = 1;
    pipeline.bindingLayouts[0] = owner.surface.Layout();
    pipeline.useDynamicRendering = true;
    pipeline.renderingSignature.colorFormatCount = 1;
    pipeline.renderingSignature.colorFormats[0] = rhi::PixelFormat::RGBA32SFloat;
    pipeline.colorTargetCount = 1;
    pipeline.colorTargets[0].format = rhi::PixelFormat::RGBA32SFloat;
    owner.pipeline = device.CreateGraphicsPipeline(pipeline);
    rhi::TextureDesc target;
    target.format = rhi::PixelFormat::RGBA32SFloat;
    target.usage = rhi::TextureUsageFlags::ColorAttachment | rhi::TextureUsageFlags::TransferSource;
    owner.target = device.CreateTexture(target);
    rhi::TextureViewDesc view;
    view.texture = owner.target;
    view.format = target.format;
    owner.view = device.CreateTextureView(view);
    rhi::BufferDesc readback;
    readback.byteSize = 32;
    readback.usage = rhi::BufferUsageFlags::TransferDestination;
    readback.memory = rhi::BufferMemory::Readback;
    owner.readback = device.CreateBuffer(readback);
    const auto rendering = rhi::ResolveDynamicRenderingCommands(resources.context.GetDevice());
    if (!Require(owner.pipeline.IsValid() && owner.view.IsValid() && owner.readback.IsValid() && rendering.IsValid(),
                 "Surface snapshot render resources failed"))
        return false;
    VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    semaphore.pNext = &type;
    if (!Require(vkCreateSemaphore(resources.context.GetDevice(), &semaphore, nullptr, &owner.gate) == VK_SUCCESS,
                 "Surface snapshot gate failed"))
        return false;
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    for (auto &fence : owner.fences)
        if (!Require(vkCreateFence(resources.context.GetDevice(), &fenceInfo, nullptr, &fence) == VK_SUCCESS,
                     "Surface snapshot fence failed"))
            return false;
    rhi::SubmissionPlan plan;
    std::string error;
    if (!Require(rhi::BuildSubmissionPlan({{100,
                                            resources.context.GetDeviceId(),
                                            rhi::QueueRole::Graphics,
                                            rhi::SubmissionDomain::Background,
                                            rhi::InvalidRenderViewId,
                                            rhi::PipelineStage::AllGraphics | rhi::PipelineStage::Transfer,
                                            {}}},
                                          plan, error),
                 "Surface snapshot submission plan failed"))
        return false;
    for (uint32_t cycle = 0; cycle < 2; ++cycle) {
        std::array<glm::vec4, 2> expected;
        std::array<rhi::SubmissionSerial, 2> epochs{};
        for (uint32_t frame = 0; frame < 2; ++frame) {
            expected[frame] = glm::vec4(float(1 + cycle * 2 + frame), 0.5f, 0.25f, 1);
            epochs[frame] = owner.queues.ReserveCompletionEpoch();
            material->SetVector4("color", expected[frame]);
            if (!Require(owner.surface.RefreshMaterialBuffer(false) && owner.surface.RefreshTextureBindings(false),
                         "Surface snapshot refresh failed"))
                return false;
            const auto group = owner.surface.ResolveBindGroup();
            const auto pending = owner.retirement.GetStats().pending;
            if (!Require(owner.surface.RefreshMaterialBuffer(false) && owner.surface.ResolveBindGroup() == group &&
                             owner.retirement.GetStats().pending == pending,
                         "Unchanged surface allocated another version"))
                return false;
            if (!Require(vkResetFences(resources.context.GetDevice(), 1, &owner.fences[frame]) == VK_SUCCESS,
                         "Surface snapshot fence reset failed"))
                return false;
            vk::VulkanSubmissionExecutor::ExternalSync sync;
            sync.completionEpoch = epochs[frame];
            sync.completionFence = owner.fences[frame];
            sync.uploadTimeline = owner.gate;
            sync.uploadTimelineValue = cycle + 1;
            const auto submitted = owner.executor.Execute(
                frame, plan,
                [&](uint32_t, VkCommandBuffer commands) {
                    VkImageMemoryBarrier imageBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                    imageBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                    imageBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                    imageBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    imageBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                    imageBarrier.srcQueueFamilyIndex = imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    imageBarrier.image = device.Resolve(owner.target);
                    imageBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1,
                                         &imageBarrier);
                    VkRenderingAttachmentInfo attachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
                    attachment.imageView = device.Resolve(owner.view);
                    attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                    VkRenderingInfo info{VK_STRUCTURE_TYPE_RENDERING_INFO};
                    info.renderArea.extent = {1, 1};
                    info.layerCount = 1;
                    info.colorAttachmentCount = 1;
                    info.pColorAttachments = &attachment;
                    rendering.begin(commands, &info);
                    VkViewport viewport{0, 0, 1, 1, 0, 1};
                    VkRect2D scissor{{0, 0}, {1, 1}};
                    vkCmdSetViewport(commands, 0, 1, &viewport);
                    vkCmdSetScissor(commands, 0, 1, &scissor);
                    vk::VulkanGraphicsCommandContext graphicsContext;
                    const auto graphics = device.MakeGraphicsCommandEncoder(graphicsContext, commands);
                    graphics.BindPipeline(owner.pipeline);
                    graphics.BindGroup(owner.pipeline, 0, group);
                    graphics.Draw(3);
                    rendering.end(commands);
                    imageBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                    imageBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                    imageBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                    imageBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageBarrier);
                    VkBufferMemoryBarrier bufferBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
                    bufferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
                    bufferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    bufferBarrier.srcQueueFamilyIndex = bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    bufferBarrier.buffer = device.Resolve(owner.readback);
                    bufferBarrier.offset = frame * 16;
                    bufferBarrier.size = 16;
                    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &bufferBarrier, 0, nullptr);
                    VkBufferImageCopy copy{};
                    copy.bufferOffset = frame * 16;
                    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    copy.imageExtent = {1, 1, 1};
                    vkCmdCopyImageToBuffer(commands, imageBarrier.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                           bufferBarrier.buffer, 1, &copy);
                    bufferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    bufferBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0,
                                         nullptr, 1, &bufferBarrier, 0, nullptr);
                    return true;
                },
                sync);
            if (!Require(submitted.Succeeded(), "Surface snapshot submission failed"))
                return false;
        }
        if (!Require(owner.retirement.Collect(0) == 0 &&
                         vkWaitForFences(resources.context.GetDevice(), 2, owner.fences.data(), VK_TRUE, 1'000'000) ==
                             VK_TIMEOUT,
                     "Surface snapshot gate did not retain in-flight resources"))
            return false;
        VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
        signal.semaphore = owner.gate;
        signal.value = cycle + 1;
        if (!Require(vkSignalSemaphore(resources.context.GetDevice(), &signal) == VK_SUCCESS &&
                         vkWaitForFences(resources.context.GetDevice(), 2, owner.fences.data(), VK_TRUE,
                                         5'000'000'000) == VK_SUCCESS,
                     "Surface snapshot GPU completion failed"))
            return false;
        for (uint32_t frame = 0; frame < 2; ++frame) {
            owner.executor.CompleteFrame(frame);
            owner.queues.CompleteCompletionEpoch(epochs[frame]);
        }
        const void *data = device.MapBuffer(owner.readback, 0, 32, rhi::BufferMapAccess::Read);
        if (!Require(data != nullptr, "Surface snapshot readback failed"))
            return false;
        const bool matched = std::memcmp(data, expected.data(), 32) == 0;
        const bool unmapped = device.UnmapBuffer(owner.readback, 0, 32, rhi::BufferMapAccess::Read);
        if (!Require(matched && unmapped, "Earlier draw used a later material version"))
            return false;
        (void)owner.retirement.Collect(epochs[1]);
        if (!Require(owner.retirement.GetStats().pending == 0, "Surface snapshot retirement leaked versions"))
            return false;
        std::cout << "Particle surface snapshots cycle=" << cycle << " frames=2 distinct_colors=2\n";
    }
    return true;
}
