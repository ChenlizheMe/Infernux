// Material publications must survive both delayed recording and queued frames.
#include <SDL3/SDL.h>
#include <function/renderer/MaterialDescriptor.h>
#include <function/renderer/vk/VkDeviceContext.h>
#include <function/renderer/vk/VulkanRhiDevice.h>
#include <function/resources/InxMaterial/InxMaterial.h>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <array>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>

using namespace infernux;

static std::vector<char> ReadSpirv(const char *path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    assert(file);
    std::vector<char> bytes(static_cast<size_t>(file.tellg()));
    file.seekg(0);
    file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    assert(file && !bytes.empty());
    return bytes;
}

int main(int argc, char **argv)
{
    assert(argc == 5);
    const uint32_t count = static_cast<uint32_t>(std::stoi(argv[3]));
    const bool queued = std::string(argv[4]) == "queued";
    assert(count >= 1 && count <= 4);
    assert(SDL_Init(SDL_INIT_VIDEO));
    auto *window = SDL_CreateWindow("Immutable material publications", 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
    vk::VkDeviceContext context;
    vk::DeviceConfig config;
    config.enableValidationLayers = true;
    assert(window && context.Initialize(window, config));
    auto &device = context.GetRhiDevice();
    const auto vkDevice = context.GetDevice();
    rhi::SubmissionSerial serial = 1;
    device.UseSubmissionSerials([&] { return serial; });
    GpuRetirementQueue retirement;
    retirement.BindSerialSource([&] { return serial; });
    vk::VkDescriptorManager allocator(vkDevice, context.GetDeviceId());
    allocator.UseSubmissionSerials([&] { return serial; });
    MaterialDescriptorManager descriptors;
    descriptors.Initialize(context.GetVmaAllocator(), vkDevice, context.GetPhysicalDevice(), &allocator);
    descriptors.SetRetirementQueue(&retirement);

    ShaderProgram program;
    ShaderProgramVariantKey key;
    key.program.stages = {"Tests/MaterialPublication.vert", "Tests/MaterialPublication.frag"};
    key.program.revision = 1;
    key.target = ShaderCompileTarget::Forward;
    assert(program.Create(vkDevice, ReadSpirv(argv[1]), ReadSpirv(argv[2]), key));

    VkAttachmentDescription attachment{};
    attachment.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachment.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    VkAttachmentReference color{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color;
    VkSubpassDependency dependency{};
    dependency.srcSubpass = 0;
    dependency.dstSubpass = VK_SUBPASS_EXTERNAL;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependency.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    VkRenderPassCreateInfo passInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    passInfo.attachmentCount = 1;
    passInfo.pAttachments = &attachment;
    passInfo.subpassCount = 1;
    passInfo.pSubpasses = &subpass;
    passInfo.dependencyCount = 1;
    passInfo.pDependencies = &dependency;
    VkRenderPass pass;
    assert(vkCreateRenderPass(vkDevice, &passInfo, nullptr, &pass) == VK_SUCCESS);

    std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
    for (auto &stage : stages) {
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.pName = "main";
    }
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = program.GetVertexModule();
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = program.GetFragmentModule();
    VkPipelineVertexInputStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo samples{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    samples.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState blend{};
    blend.colorWriteMask = 15;
    VkPipelineColorBlendStateCreateInfo blending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blending.attachmentCount = 1;
    blending.pAttachments = &blend;
    const std::array<VkDynamicState, 2> dynamicStates{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamic.pDynamicStates = dynamicStates.data();
    VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
    pipelineInfo.pStages = stages.data();
    pipelineInfo.pVertexInputState = &input;
    pipelineInfo.pInputAssemblyState = &assembly;
    pipelineInfo.pViewportState = &viewport;
    pipelineInfo.pRasterizationState = &raster;
    pipelineInfo.pMultisampleState = &samples;
    pipelineInfo.pColorBlendState = &blending;
    pipelineInfo.pDynamicState = &dynamic;
    pipelineInfo.layout = program.GetPipelineLayout();
    pipelineInfo.renderPass = pass;
    VkPipeline pipeline;
    assert(vkCreateGraphicsPipelines(vkDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) == VK_SUCCESS);

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = context.GetQueueIndices().graphicsFamily.value();
    VkCommandPool pool;
    assert(vkCreateCommandPool(vkDevice, &poolInfo, nullptr, &pool) == VK_SUCCESS);
    std::vector<VkCommandBuffer> commands(count);
    VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation.commandPool = pool;
    allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation.commandBufferCount = count;
    assert(vkAllocateCommandBuffers(vkDevice, &allocation, commands.data()) == VK_SUCCESS);
    VkSemaphoreTypeCreateInfo timelineType{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    timelineType.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    semaphoreInfo.pNext = &timelineType;
    VkSemaphore gate;
    assert(vkCreateSemaphore(vkDevice, &semaphoreInfo, nullptr, &gate) == VK_SUCCESS);

    struct Frame
    {
        rhi::TextureHandle texture;
        rhi::TextureViewHandle view;
        rhi::BufferHandle readback;
        VkFramebuffer framebuffer;
        VkFence fence;
        std::array<float, 4> expected{};
    };
    std::vector<Frame> frames(count);
    const auto submit = [&](uint32_t index) {
        const uint64_t value = 1;
        VkTimelineSemaphoreSubmitInfo wait{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        wait.waitSemaphoreValueCount = 1;
        wait.pWaitSemaphoreValues = &value;
        const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        info.pNext = &wait;
        info.waitSemaphoreCount = 1;
        info.pWaitSemaphores = &gate;
        info.pWaitDstStageMask = &stage;
        info.commandBufferCount = 1;
        info.pCommandBuffers = &commands[index];
        assert(vkQueueSubmit(context.GetGraphicsQueue(), 1, &info, frames[index].fence) == VK_SUCCESS);
    };

    InxMaterial material("publication-test", "publication-test");
    MaterialDescriptorSet *publication = nullptr;
    VkDescriptorSet previous = VK_NULL_HANDLE;
    VkBuffer previousFragment = VK_NULL_HANDLE, previousVertex = VK_NULL_HANDLE;
    for (uint32_t index = 0; index < count; ++index) {
        serial = index + 1;
        glm::vec4 tint(0, 0, 0, 1);
        tint[index % 3] = 1;
        const float fragmentGain = 0.25f * (index + 1);
        const float vertexGain = index % 2 ? 0.5f : 1.0f;
        material.SetColor("tint", tint);
        material.SetFloat("fragmentGain", fragmentGain);
        material.SetFloat("vertexGain", vertexGain);
        if (!publication)
            publication = descriptors.GetOrCreateDescriptorSet(material, program);
        else
            assert(descriptors.UpdateMaterialUBO(material.GetMaterialKey(), material));
        assert(publication && publication->materialUBO && publication->vertexMaterialUBO);
        assert(publication->descriptorSet != previous);
        assert(publication->materialUBO->GetBuffer() != previousFragment);
        assert(publication->vertexMaterialUBO->GetBuffer() != previousVertex);
        previous = publication->descriptorSet;
        previousFragment = publication->materialUBO->GetBuffer();
        previousVertex = publication->vertexMaterialUBO->GetBuffer();
        allocator.MarkUsed(publication->descriptorLease, serial);
        // Repeated views and pending-texture polling must reuse this publication.
        const auto allocated = allocator.GetStats().liveSets;
        for (int repeat = 0; repeat < 8; ++repeat) {
            assert(descriptors.UpdateMaterialUBO(material.GetMaterialKey(), material));
            assert(publication->descriptorSet == previous);
        }
        assert(allocator.GetStats().liveSets == allocated);
        assert(retirement.Collect(0) == 0 && allocator.Collect(0) == 0);

        auto &frame = frames[index];
        frame.expected[index % 3] = fragmentGain * vertexGain;
        frame.expected[3] = 1;
        rhi::TextureDesc texture;
        texture.width = 2;
        texture.height = 1;
        texture.format = rhi::PixelFormat::RGBA32SFloat;
        texture.usage = rhi::TextureUsageFlags::ColorAttachment | rhi::TextureUsageFlags::TransferSource;
        frame.texture = device.CreateTexture(texture);
        rhi::TextureViewDesc view;
        view.texture = frame.texture;
        view.format = texture.format;
        frame.view = device.CreateTextureView(view);
        const VkImageView imageView = device.Resolve(frame.view);
        VkFramebufferCreateInfo framebuffer{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        framebuffer.renderPass = pass;
        framebuffer.attachmentCount = 1;
        framebuffer.pAttachments = &imageView;
        framebuffer.width = 2;
        framebuffer.height = framebuffer.layers = 1;
        assert(vkCreateFramebuffer(vkDevice, &framebuffer, nullptr, &frame.framebuffer) == VK_SUCCESS);
        rhi::BufferDesc buffer;
        buffer.byteSize = 2 * 4 * sizeof(float);
        buffer.usage = rhi::BufferUsageFlags::TransferDestination;
        buffer.memory = rhi::BufferMemory::Readback;
        frame.readback = device.CreateBuffer(buffer);
        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        assert(vkCreateFence(vkDevice, &fenceInfo, nullptr, &frame.fence) == VK_SUCCESS);
        const auto command = commands[index];
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        assert(vkBeginCommandBuffer(command, &begin) == VK_SUCCESS);
        VkClearValue clear{};
        VkRenderPassBeginInfo render{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        render.renderPass = pass;
        render.framebuffer = frame.framebuffer;
        render.renderArea.extent = {2, 1};
        render.clearValueCount = 1;
        render.pClearValues = &clear;
        vkCmdBeginRenderPass(command, &render, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, program.GetPipelineLayout(), 0, 1,
                                &publication->descriptorSet, 0, nullptr);
        for (int viewIndex = 0; viewIndex < 2; ++viewIndex) {
            VkViewport viewport{static_cast<float>(viewIndex), 0, 1, 1, 0, 1};
            VkRect2D scissor{{viewIndex, 0}, {1, 1}};
            vkCmdSetViewport(command, 0, 1, &viewport);
            vkCmdSetScissor(command, 0, 1, &scissor);
            vkCmdDraw(command, 3, 1, 0, 0);
        }
        vkCmdEndRenderPass(command);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {2, 1, 1};
        vkCmdCopyImageToBuffer(command, device.Resolve(frame.texture), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               device.Resolve(frame.readback), 1, &copy);
        VkBufferMemoryBarrier host{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        host.srcQueueFamilyIndex = host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        host.buffer = device.Resolve(frame.readback);
        host.size = buffer.byteSize;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                             &host, 0, nullptr);
        assert(vkEndCommandBuffer(command) == VK_SUCCESS);
        if (queued)
            submit(index);
    }
    // Change the live material once more while every older publication is held.
    material.SetFloat("fragmentGain", 0);
    assert(descriptors.UpdateMaterialUBO(material.GetMaterialKey(), material));
    assert(publication->descriptorSet != previous);
    if (!queued)
        for (uint32_t index = 0; index < count; ++index)
            submit(index);
    for (const auto &frame : frames)
        assert(vkGetFenceStatus(vkDevice, frame.fence) == VK_NOT_READY);
    VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
    signal.semaphore = gate;
    signal.value = 1;
    assert(vkSignalSemaphore(vkDevice, &signal) == VK_SUCCESS);
    for (auto &frame : frames) {
        assert(vkWaitForFences(vkDevice, 1, &frame.fence, VK_TRUE, 10'000'000'000ULL) == VK_SUCCESS);
        std::array<float, 8> pixels{};
        assert(device.ReadBuffer(frame.readback, 0, pixels.data(), sizeof(pixels)));
        for (size_t view = 0; view < 2; ++view)
            for (size_t channel = 0; channel < 4; ++channel)
                assert(std::abs(pixels[view * 4 + channel] - frame.expected[channel]) < 0.00001f);
        vkDestroyFence(vkDevice, frame.fence, nullptr);
        vkDestroyFramebuffer(vkDevice, frame.framebuffer, nullptr);
        device.Release(frame.readback);
        device.Release(frame.view);
        device.Release(frame.texture);
    }
    vkDestroyCommandPool(vkDevice, pool, nullptr);
    vkDestroySemaphore(vkDevice, gate, nullptr);
    assert(retirement.Collect(serial) == count * 2);
    assert(allocator.Collect(serial) == count);
    descriptors.Shutdown();
    allocator.Destroy();
    device.CollectResourceRetirements(serial);
    vkDestroyPipeline(vkDevice, pipeline, nullptr);
    vkDestroyRenderPass(vkDevice, pass, nullptr);
    program.Destroy();
    context.Destroy();
    SDL_DestroyWindow(window);
    SDL_Quit();
    std::cout << "PASS " << count << ' ' << argv[4] << " frames, two views, immutable fragment/vertex UBOs\n";
}
