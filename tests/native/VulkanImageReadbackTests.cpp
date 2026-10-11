#include <function/renderer/vk/VkDeviceContext.h>
#include <function/renderer/vk/VkResourceManager.h>
#include <function/renderer/vk/VulkanQueueManager.h>
#include <SDL3/SDL.h>

#include <cassert>
#include <iostream>
#include <string>
#ifdef _WIN32
#include <Windows.h>
#endif

using namespace infernux;

static bool HasNoData(const std::shared_ptr<vk::ImageReadbackTicket> &ticket)
{
    try {
        (void)ticket->GetData();
        return false;
    } catch (const std::exception &) {
        return true;
    }
}

static bool Run(vk::VkDeviceContext &device, vk::VulkanQueueManager &queues,
                vk::VkResourceManager &resources, const std::string &scenario)
{
    using Status = vk::ImageReadbackStatus;
    const std::vector<uint8_t> blue{0, 0, 255, 255};
    if (scenario == "recorder_abandon") {
        {
            auto recorder = resources.BeginGraphicsImageReadback(1, 1, VK_FORMAT_R8G8B8A8_UNORM);
            assert(recorder.GetCommandBuffer() != VK_NULL_HANDLE);
        }
        assert(resources.GetPendingImageReadbackCount() == 0);
        assert(resources.GetStagingPoolBufferCount() == 1);
        return true;
    }
    if (scenario.rfind("graphics", 0) == 0) {
        auto recorder = resources.BeginGraphicsImageReadback(1, 1, VK_FORMAT_R8G8B8A8_UNORM);
        vkCmdFillBuffer(recorder.GetCommandBuffer(), recorder.GetStagingBuffer(), 0, 4, 0xffff0000);
        unsigned released = 0;
        auto ticket = recorder.Submit([&released] { ++released; });
        const bool cancelled = scenario == "graphics_cancelled";
        if (cancelled)
            ticket->Cancel();
        resources.PollImageReadbacks();
        assert(ticket->GetStatus() == (cancelled ? Status::Cancelled : Status::Pending));
        assert(HasNoData(ticket) && resources.GetPendingImageReadbackCount() == 1);
        assert(resources.GetStagingPoolBufferCount() == 0 && released == 0);
        if (scenario == "graphics_drain") {
            resources.DrainImageReadbacks();
        } else {
            resources.DrainAsyncGraphicsSubmissions();
            resources.PollImageReadbacks();
        }
        assert(released == 1 && resources.GetPendingImageReadbackCount() == 0);
        assert(resources.GetStagingPoolBufferCount() == 1);
        assert(ticket->GetStatus() == (cancelled ? Status::Cancelled : Status::Completed));
        assert(cancelled ? HasNoData(ticket) : ticket->GetData() == blue);
        return true;
    }

    auto source = resources.CreateImage(1, 1, VK_FORMAT_R8G8B8A8_UNORM,
                                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    assert(source && source->IsValid());
    const auto allocation = resources.AllocatePrimaryCommandBuffer();
    assert(allocation.cmdBuffer != VK_NULL_HANDLE);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    assert(vkBeginCommandBuffer(allocation.cmdBuffer, &begin) == VK_SUCCESS);
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = source->GetImage();
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(allocation.cmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);
    const VkClearColorValue color{{0, 0, 1, 1}};
    vkCmdClearColorImage(allocation.cmdBuffer, source->GetImage(), barrier.newLayout, &color, 1, &barrier.subresourceRange);
    const bool invalid = scenario == "invalid_epoch";
    const auto epoch = invalid ? rhi::InvalidSubmissionSerial : queues.ReserveCompletionEpoch();
    std::shared_ptr<vk::ImageReadbackTicket> ticket;
    try {
        ticket = resources.RecordFrameImageReadback(allocation.cmdBuffer, source->GetImage(), barrier.newLayout,
            VK_IMAGE_ASPECT_COLOR_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            1, 1, VK_FORMAT_R8G8B8A8_UNORM, epoch);
        assert(!invalid);
    } catch (const std::invalid_argument &) {
        assert(invalid && resources.GetPendingImageReadbackCount() == 0);
        assert(resources.GetStagingPoolBufferCount() == 0);
        assert(vkResetCommandBuffer(allocation.cmdBuffer, 0) == VK_SUCCESS);
        resources.FreeCommandBuffer(allocation);
        return true;
    }
    const bool cancelled = scenario == "frame_cancelled";
    if (cancelled)
        ticket->Cancel();
    if (scenario != "frame_completed") {
        resources.PollImageReadbacks();
        const bool premature = resources.GetPendingImageReadbackCount() != 1 ||
                               resources.GetStagingPoolBufferCount() != 0 ||
                               ticket->GetStatus() != (cancelled ? Status::Cancelled : Status::Pending);
        if (premature) {
            // Never submit commands referencing a prematurely recycled staging allocation.
            assert(vkResetCommandBuffer(allocation.cmdBuffer, 0) == VK_SUCCESS);
            resources.FreeCommandBuffer(allocation);
            queues.CompleteCompletionEpoch(epoch);
            std::cerr << "Readback finalized before its frame completion epoch\n";
            return false;
        }
        assert(!queues.IsCompletionEpochComplete(epoch) && HasNoData(ticket));
    }
    assert(vkEndCommandBuffer(allocation.cmdBuffer) == VK_SUCCESS);
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    assert(vkCreateFence(device.GetDevice(), &fenceInfo, nullptr, &fence) == VK_SUCCESS);
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &allocation.cmdBuffer;
    const auto submission = queues.Reserve(rhi::QueueRole::Graphics);
    assert(queues.SubmitReserved(submission, submit, fence) == VK_SUCCESS);
    assert(vkWaitForFences(device.GetDevice(), 1, &fence, VK_TRUE, 10'000'000'000ULL) == VK_SUCCESS);
    queues.MarkCompleted(submission);
    queues.CompleteCompletionEpoch(epoch);
    const bool drain = scenario == "frame_drain";
    if (drain)
        resources.DrainImageReadbacks();
    else
        resources.PollImageReadbacks();
    assert(ticket->GetStatus() == (cancelled || drain ? Status::Cancelled : Status::Completed));
    assert(cancelled || drain ? HasNoData(ticket) : ticket->GetData() == blue);
    assert(resources.GetPendingImageReadbackCount() == 0 && resources.GetPendingImageReadbackBytes() == 0);
    assert(resources.GetStagingPoolBufferCount() == 1);
    vkDestroyFence(device.GetDevice(), fence, nullptr);
    resources.FreeCommandBuffer(allocation);
    return true;
}

int main(int argc, char **argv)
{
#ifdef _WIN32
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
#endif
    assert(argc == 2 && SDL_Init(SDL_INIT_VIDEO));
    auto *window = SDL_CreateWindow("Image readback completion", 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
    assert(window);
    vk::VkDeviceContext device;
    vk::DeviceConfig config;
    config.enableValidationLayers = true;
    assert(device.Initialize(window, config));
    vk::VulkanQueueManager queues;
    assert(queues.Initialize(device, 2));
    bool passed;
    {
        vk::VkResourceManager resources;
        assert(resources.Initialize(device, &queues));
        passed = Run(device, queues, resources, argv[1]);
    }
    queues.Destroy();
    device.Destroy();
    SDL_DestroyWindow(window);
    SDL_Quit();
    std::cout << "cleanup_complete " << argv[1] << '\n';
    return passed ? 0 : 1;
}
