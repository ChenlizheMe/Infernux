#include <function/renderer/vk/AsyncTransferContext.h>
#include <function/renderer/vk/VkDeviceContext.h>
#include <function/renderer/vk/VkResourceManager.h>
#include <function/renderer/vk/VulkanQueueManager.h>
#include <function/renderer/vk/VulkanRhiDevice.h>
#include <SDL3/SDL.h>

#include <array>
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#ifdef _WIN32
#include <Windows.h>
#endif

using namespace infernux;

static bool Run(vk::VkDeviceContext &context, vk::VulkanQueueManager &queues,
                vk::AsyncTransferContext &uploads, const std::vector<uint32_t> &spirv, const std::string &scenario)
{
    const bool async = scenario.rfind("async", 0) == 0;
    const bool compute = scenario.find("compute") != std::string::npos;
    vk::VkResourceManager resources;
    assert(resources.Initialize(context, &queues));
    if (async)
        resources.SetAsyncTransferContext(&uploads);
    auto &device = context.GetRhiDevice();
    const auto role = compute ? rhi::QueueRole::Compute : rhi::QueueRole::Graphics;
    const auto family = queues.GetSnapshot(role).family;
    const auto graphicsFamily = queues.GetSnapshot(rhi::QueueRole::Graphics).family;
    const auto transferFamily = uploads.GetQueueFamily();
    const bool transferUpload = async && uploads.IsAsyncCapable();
    const bool concurrent = family != graphicsFamily || (transferUpload && transferFamily != graphicsFamily);
    std::array<uint32_t, 256> input{};
    for (uint32_t index = 0; index < input.size(); ++index)
        input[index] = index * 123 + 9;
    rhi::BufferUploadRequest request{input.data(), sizeof(input), rhi::BufferUsage::Storage};
    if (compute)
        request.queueAccess = rhi::QueueAccessFlags::Graphics | rhi::QueueAccessFlags::Compute;
    if (scenario.rfind("invalid", 0) == 0) {
        request.queueAccess = scenario == "invalid_none" ? rhi::QueueAccessFlags::None
                                                         : static_cast<rhi::QueueAccessFlags>(128);
        bool rejected = false;
        try {
            (void)resources.BeginBufferUpload(request);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        assert(rejected && resources.GetStagingAllocationCount() == 0 &&
               resources.GetBufferUploadSubmissionCount() == 0);
        return true;
    }
    const auto upload = resources.BeginBufferUpload(request);
    assert(resources.TryPublishBufferUpload(upload));
    const auto source = resources.GetPublishedRhiBuffer(upload);
    assert(source && source->GetBuffer().IsValid());
    if (device.UsesConcurrentQueueSharing(source->GetBuffer()) != concurrent) {
        // Check the allocation contract before dispatching any possibly illegal consumer.
        resources.DrainBufferUploads();
        std::cerr << "Published upload sharing does not match its Graphics/Compute/Transfer consumers\n";
        return false;
    }

    rhi::BufferDesc outputDesc;
    outputDesc.byteSize = sizeof(input);
    outputDesc.usage = rhi::BufferUsageFlags::Storage;
    outputDesc.memory = rhi::BufferMemory::Readback;
    outputDesc.queueAccess = compute ? rhi::QueueAccessFlags::Compute : rhi::QueueAccessFlags::Graphics;
    const auto output = device.CreateBuffer(outputDesc);
    const auto shader = device.CreateShaderModule(rhi::ShaderModuleDesc::FromSpirV(spirv.data(), spirv.size()));
    rhi::BindingLayoutDesc layoutDesc;
    layoutDesc.entryCount = 2;
    layoutDesc.entries[0] = {0, rhi::BindingType::StorageBuffer, rhi::ShaderStage::Compute, 1};
    layoutDesc.entries[1] = {1, rhi::BindingType::StorageBuffer, rhi::ShaderStage::Compute, 1};
    const auto layout = device.CreateBindingLayout(layoutDesc);
    rhi::BindGroupDesc groupDesc;
    groupDesc.layout = layout;
    groupDesc.bufferCount = 2;
    groupDesc.buffers[0] = {0, rhi::BindingType::StorageBuffer, source->GetBuffer()};
    groupDesc.buffers[1] = {1, rhi::BindingType::StorageBuffer, output};
    const auto group = device.CreateBindGroup(groupDesc);
    rhi::ComputePipelineDesc pipelineDesc;
    pipelineDesc.computeShader = shader;
    pipelineDesc.bindingLayouts[0] = layout;
    pipelineDesc.bindingLayoutCount = 1;
    const auto pipeline = device.CreateComputePipeline(pipelineDesc);
    assert(output.IsValid() && shader.IsValid() && layout.IsValid() && group.IsValid() && pipeline.IsValid());

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = family;
    VkCommandPool pool = VK_NULL_HANDLE;
    assert(vkCreateCommandPool(context.GetDevice(), &poolInfo, nullptr, &pool) == VK_SUCCESS);
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    assert(vkAllocateCommandBuffers(context.GetDevice(), &allocate, &commands) == VK_SUCCESS);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    assert(vkBeginCommandBuffer(commands, &begin) == VK_SUCCESS);
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = device.Resolve(source->GetBuffer());
    barrier.size = sizeof(input);
    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 1, &barrier, 0, nullptr);
    vk::VulkanComputeCommandContext encoderContext;
    const auto encoder = device.MakeComputeCommandEncoder(encoderContext, commands);
    encoder.BindPipeline(pipeline);
    encoder.BindGroup(pipeline, 0, group);
    encoder.Dispatch(4, 1, 1);
    barrier.buffer = device.Resolve(output);
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 0, nullptr, 1, &barrier, 0, nullptr);
    assert(vkEndCommandBuffer(commands) == VK_SUCCESS);

    const uint64_t uploadValue = resources.GetRequiredUploadTimelineValue();
    const auto uploadSemaphore = resources.GetUploadTimelineSemaphore();
    VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    timeline.waitSemaphoreValueCount = uploadValue ? 1 : 0;
    timeline.pWaitSemaphoreValues = &uploadValue;
    const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.pNext = &timeline;
    submit.waitSemaphoreCount = uploadValue ? 1 : 0;
    submit.pWaitSemaphores = &uploadSemaphore;
    submit.pWaitDstStageMask = &waitStage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &commands;
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    assert(vkCreateFence(context.GetDevice(), &fenceInfo, nullptr, &fence) == VK_SUCCESS);
    const auto ticket = queues.Reserve(role);
    assert(queues.SubmitReserved(ticket, submit, fence) == VK_SUCCESS);
    assert(vkWaitForFences(context.GetDevice(), 1, &fence, VK_TRUE, 10'000'000'000ULL) == VK_SUCCESS);
    queues.MarkCompleted(ticket);
    std::array<uint32_t, 256> actual{};
    assert(device.ReadBuffer(output, 0, actual.data(), sizeof(actual)));
    for (size_t index = 0; index < input.size(); ++index)
        assert(actual[index] == input[index] * 7 + 13);
    resources.DrainBufferUploads();
    device.Release(group);
    device.Release(pipeline);
    device.Release(layout);
    device.Release(shader);
    device.Release(output);
    vkDestroyFence(context.GetDevice(), fence, nullptr);
    vkDestroyCommandPool(context.GetDevice(), pool, nullptr);
    std::cout << "GPU words verified: 256; graphics=" << graphicsFamily << " compute="
              << queues.GetSnapshot(rhi::QueueRole::Compute).family << " transfer=" << transferFamily
              << " consumer=" << family << " concurrent=" << concurrent << '\n';
    return true;
}

int main(int argc, char **argv)
{
#ifdef _WIN32
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
#endif
    assert(argc == 3);
    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    assert(file);
    const auto bytes = static_cast<size_t>(file.tellg());
    assert(bytes && bytes % 4 == 0);
    std::vector<uint32_t> spirv(bytes / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char *>(spirv.data()), bytes);
    assert(file && SDL_Init(SDL_INIT_VIDEO));
    const std::string scenario = argv[2];
    auto *window = SDL_CreateWindow("Buffer upload consumers", 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
    assert(window);
    vk::VkDeviceContext context;
    vk::DeviceConfig config;
    config.enableValidationLayers = true;
    assert(context.Initialize(window, config));
    vk::VulkanQueueManager queues;
    assert(queues.Initialize(context, 2));
    vk::AsyncTransferContext uploads;
    const auto family = context.GetQueueIndices().transferFamily.value();
    assert(uploads.Initialize(context.GetDevice(), family, context.HasDedicatedTransferQueue(),
                              context.IsTimelineSemaphoreEnabled(), queues,
                              context.HasDedicatedTransferQueue() ? rhi::QueueRole::Transfer : rhi::QueueRole::Graphics));
    const bool passed = Run(context, queues, uploads, spirv, scenario);
    uploads.Destroy();
    queues.Destroy();
    context.Destroy();
    SDL_DestroyWindow(window);
    SDL_Quit();
    return passed ? 0 : 1;
}
