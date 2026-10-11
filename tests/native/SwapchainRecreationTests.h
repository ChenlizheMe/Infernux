#pragma once

#include <function/renderer/vk/VkSwapchainManager.h>
#include <function/renderer/vk/VulkanQueueManager.h>
#include <SDL3/SDL.h>

#include <cassert>
#include <iostream>
#include <unordered_set>
#include <vector>

namespace swapchain_recreation_test
{
using namespace infernux;

enum class Fault { None, Create, ImageCount, Images, ImageView, Semaphore };

struct Recorder
{
    static inline Recorder *current = nullptr;
    Fault fault = Fault::None;
    unsigned viewsThisAttempt = 0, semaphoresThisAttempt = 0;
    std::vector<VkSwapchainKHR> createOld, acquired;
    std::unordered_set<VkSwapchainKHR> live, retired;
    std::unordered_set<VkImageView> views;
    std::unordered_set<VkSemaphore> semaphores;
    VkSwapchainKHR expectedCommitOld = VK_NULL_HANDLE;
    std::unordered_set<VkImageView> expectedRetireViews;
    bool committed = false;

    Recorder() { assert(!current); current = this; }
    ~Recorder() { assert(live.empty() && views.empty() && semaphores.empty()); current = nullptr; }

    static VKAPI_ATTR VkResult VKAPI_CALL Create(VkDevice device, const VkSwapchainCreateInfoKHR *info,
                                                 const VkAllocationCallbacks *allocator, VkSwapchainKHR *result)
    {
        auto &r = *current;
        r.createOld.push_back(info->oldSwapchain);
        assert(info->oldSwapchain == VK_NULL_HANDLE || !r.retired.count(info->oldSwapchain));
        r.viewsThisAttempt = r.semaphoresThisAttempt = 0;
        const auto status = vkCreateSwapchainKHR(device, info, allocator, result);
        if (status != VK_SUCCESS)
            std::cerr << "fixture_create_failed fault=" << int(r.fault) << " call=" << r.createOld.size()
                      << " old=" << info->oldSwapchain << " result=" << int(status) << std::endl;
        assert(status == VK_SUCCESS); // Test fixture, not a claim that the driver failed.
        if (info->oldSwapchain != VK_NULL_HANDLE)
            r.retired.insert(info->oldSwapchain);
        r.retired.erase(*result);
        if (r.fault == Fault::Create) {
            // The real call has retired oldSwapchain. Discard its unpublished
            // result and report a controlled failure at the production boundary.
            vkDestroySwapchainKHR(device, *result, allocator);
            *result = VK_NULL_HANDLE;
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        assert(r.live.insert(*result).second);
        return status;
    }

    static VKAPI_ATTR VkResult VKAPI_CALL Images(VkDevice device, VkSwapchainKHR swapchain,
                                                  uint32_t *count, VkImage *images)
    {
        auto &r = *current;
        if ((r.fault == Fault::ImageCount && !images) || (r.fault == Fault::Images && images))
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        return vkGetSwapchainImagesKHR(device, swapchain, count, images);
    }

    static VKAPI_ATTR VkResult VKAPI_CALL ImageView(VkDevice device, const VkImageViewCreateInfo *info,
                                                     const VkAllocationCallbacks *allocator, VkImageView *result)
    {
        auto &r = *current;
        if (++r.viewsThisAttempt == 2 && r.fault == Fault::ImageView)
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        const auto status = vkCreateImageView(device, info, allocator, result);
        assert(status == VK_SUCCESS && r.views.insert(*result).second);
        return status;
    }

    static VKAPI_ATTR VkResult VKAPI_CALL Semaphore(VkDevice device, const VkSemaphoreCreateInfo *info,
                                                     const VkAllocationCallbacks *allocator, VkSemaphore *result)
    {
        auto &r = *current;
        if (++r.semaphoresThisAttempt == 2 && r.fault == Fault::Semaphore)
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        const auto status = vkCreateSemaphore(device, info, allocator, result);
        assert(status == VK_SUCCESS && r.semaphores.insert(*result).second);
        return status;
    }

    static VKAPI_ATTR VkResult VKAPI_CALL Acquire(VkDevice, VkSwapchainKHR swapchain, uint64_t,
                                                  VkSemaphore, VkFence, uint32_t *index)
    {
        auto &r = *current;
        r.acquired.push_back(swapchain);
        // Record only. Never acquire a known retired handle on the real GPU.
        if (r.retired.count(swapchain))
            return VK_ERROR_OUT_OF_DATE_KHR;
        *index = 0;
        return VK_SUCCESS;
    }

    static VKAPI_ATTR void VKAPI_CALL DestroySwapchain(VkDevice device, VkSwapchainKHR swapchain,
                                                        const VkAllocationCallbacks *allocator)
    {
        auto &r = *current;
        assert(r.live.erase(swapchain) == 1);
        if (swapchain == r.expectedCommitOld)
            assert(r.committed); // External aliases retire before the old views.
        vkDestroySwapchainKHR(device, swapchain, allocator);
    }
    static VKAPI_ATTR void VKAPI_CALL DestroyView(VkDevice device, VkImageView view,
                                                   const VkAllocationCallbacks *allocator)
    {
        if (current->expectedRetireViews.count(view))
            assert(current->committed);
        assert(current->views.erase(view) == 1);
        vkDestroyImageView(device, view, allocator);
    }
    static VKAPI_ATTR void VKAPI_CALL DestroySemaphore(VkDevice device, VkSemaphore semaphore,
                                                        const VkAllocationCallbacks *allocator)
    {
        assert(current->semaphores.erase(semaphore) == 1);
        vkDestroySemaphore(device, semaphore, allocator);
    }
    static vk::VkSwapchainManager::Dispatch Calls()
    {
        return {Create, Images, ImageView, Semaphore, Acquire, DestroySwapchain, DestroyView, DestroySemaphore};
    }
};

inline void RenderResizedWindows(vk::VkDeviceContext &context, vk::VulkanQueueManager &queues, SDL_Window *window)
{
    assert(SDL_ShowWindow(window));
    vk::VkSwapchainManager presentation;
    assert(presentation.Create(context, 64, 64)); // Unmodified production Vulkan dispatch.
    const auto device = context.GetDevice();
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = context.GetQueueIndices().graphicsFamily.value();
    VkCommandPool pool = VK_NULL_HANDLE;
    assert(vkCreateCommandPool(device, &poolInfo, nullptr, &pool) == VK_SUCCESS);
    VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation.commandPool = pool;
    allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    assert(vkAllocateCommandBuffers(device, &allocation, &command) == VK_SUCCESS);
    unsigned frames = 0;
    for (auto size : {VkExtent2D{64, 64}, VkExtent2D{96, 72}, VkExtent2D{128, 96}, VkExtent2D{80, 64}, VkExtent2D{64, 64}}) {
        assert(SDL_SetWindowSize(window, int(size.width), int(size.height)));
        assert(SDL_SyncWindow(window));
        SDL_PumpEvents();
        assert(presentation.Recreate(context, queues, size.width, size.height, {}));
        const auto extent = presentation.GetExtent();
        assert(extent.width == size.width && extent.height == size.height);
        uint32_t index = UINT32_MAX;
        assert(presentation.AcquireNextImage(0, index) == vk::SwapchainResult::Success);

        VkAttachmentDescription attachment{};
        attachment.format = presentation.GetImageFormat();
        attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkAttachmentReference reference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &reference;
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        VkRenderPassCreateInfo passInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        passInfo.attachmentCount = passInfo.subpassCount = passInfo.dependencyCount = 1;
        passInfo.pAttachments = &attachment;
        passInfo.pSubpasses = &subpass;
        passInfo.pDependencies = &dependency;
        VkRenderPass pass = VK_NULL_HANDLE;
        assert(vkCreateRenderPass(device, &passInfo, nullptr, &pass) == VK_SUCCESS);
        const auto view = presentation.GetImageView(index);
        VkFramebufferCreateInfo framebufferInfo{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        framebufferInfo.renderPass = pass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = &view;
        framebufferInfo.width = extent.width;
        framebufferInfo.height = extent.height;
        framebufferInfo.layers = 1;
        VkFramebuffer framebuffer = VK_NULL_HANDLE;
        assert(vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffer) == VK_SUCCESS);
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        assert(vkBeginCommandBuffer(command, &begin) == VK_SUCCESS);
        VkClearValue clear{};
        clear.color = {{0.2f, 0.1f * float(frames + 1), 0.4f, 1.0f}};
        VkRenderPassBeginInfo render{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        render.renderPass = pass;
        render.framebuffer = framebuffer;
        render.renderArea.extent = extent;
        render.clearValueCount = 1;
        render.pClearValues = &clear;
        vkCmdBeginRenderPass(command, &render, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdEndRenderPass(command);
        assert(vkEndCommandBuffer(command) == VK_SUCCESS);
        const auto available = presentation.GetImageAvailableSemaphore(0);
        const auto finished = presentation.GetRenderFinishedSemaphore(index);
        const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.waitSemaphoreCount = submit.signalSemaphoreCount = submit.commandBufferCount = 1;
        submit.pWaitSemaphores = &available;
        submit.pWaitDstStageMask = &waitStage;
        submit.pSignalSemaphores = &finished;
        submit.pCommandBuffers = &command;
        assert(queues.SubmitReserved(queues.Reserve(rhi::QueueRole::Graphics), submit, VK_NULL_HANDLE) == VK_SUCCESS);
        assert(presentation.Present(queues, index) == vk::SwapchainResult::Success);
        assert(queues.WaitIdleForPresentation() == VK_SUCCESS);
        vkDestroyFramebuffer(device, framebuffer, nullptr);
        vkDestroyRenderPass(device, pass, nullptr);
        assert(vkResetCommandPool(device, pool, 0) == VK_SUCCESS);
        ++frames;
    }
    vkDestroyCommandPool(device, pool, nullptr);
    std::cout << "real_swapchain_resize_present frames=" << frames << " cleanup_complete=true\n";
    assert(SDL_HideWindow(window));
}

inline void Run(vk::VkDeviceContext &context, SDL_Window *window)
{
    vk::VulkanQueueManager queues;
    assert(queues.Initialize(context, 2));
    for (auto fault : {Fault::None, Fault::Create, Fault::ImageCount, Fault::Images, Fault::ImageView, Fault::Semaphore}) {
        Recorder recorder;
        vk::VkSwapchainManager presentation;
        assert(presentation.Create(context, 64, 64, Recorder::Calls()));
        const auto original = presentation.GetSwapchain();
        const auto originalViews = presentation.GetImageViews();
        assert(presentation.IsValid() && recorder.live.size() == 1);
        uint32_t image = UINT32_MAX;
        assert(presentation.AcquireNextImage(0, image) == vk::SwapchainResult::Success);
        assert(recorder.acquired.back() == original);

        vk::VulkanQueueManager unavailableQueues;
        assert(!presentation.Recreate(context, unavailableQueues, 64, 64, {}));
        assert(recorder.createOld.size() == 1 && presentation.IsValid());

        // Rejection before vkCreateSwapchainKHR must not retire the old owner.
        vk::VkDeviceContext uninitialized;
        assert(!presentation.Recreate(uninitialized, queues, 64, 64, {}));
        assert(recorder.createOld.size() == 1 && presentation.IsValid());
        assert(presentation.AcquireNextImage(1, image) == vk::SwapchainResult::Success);
        assert(recorder.acquired.back() == original);

        unsigned commits = 0;
        auto commit = [&]() {
            ++commits;
            assert(recorder.live.count(original));
            for (auto view : originalViews)
                assert(recorder.views.count(view));
            recorder.committed = true;
        };
        recorder.expectedCommitOld = original;
        recorder.expectedRetireViews.insert(originalViews.begin(), originalViews.end());
        recorder.fault = fault;
        const bool replaced = presentation.Recreate(context, queues, 64, 64, commit);
        assert(replaced == (fault == Fault::None));
        assert(recorder.createOld.back() == original && recorder.retired.count(original));
        if (fault != Fault::None) {
            assert(commits == 1 && presentation.GetSwapchain() == VK_NULL_HANDLE);
            assert(presentation.GetImageViews().empty());
            assert(recorder.live.empty() && recorder.views.empty());
            assert(recorder.semaphores.size() == vk::VkSwapchainManager::MAX_FRAMES_IN_FLIGHT);
            const auto before = recorder.acquired.size();
            presentation.AcquireNextImage(0, image);
            assert(recorder.acquired.size() == before); // Regression: old code called the retired chain.
            assert(!presentation.IsValid());
            // Move both ways preserves the empty generation and its dispatch owner.
            vk::VkSwapchainManager moved(std::move(presentation));
            presentation = std::move(moved);
            assert(!moved.IsValid() && !presentation.IsValid());
            assert(presentation.AcquireNextImage(1, image) == vk::SwapchainResult::Error);
            assert(recorder.acquired.size() == before);
            recorder.fault = Fault::None;
            assert(presentation.Recreate(context, queues, 64, 64, commit));
            assert(recorder.createOld.back() == VK_NULL_HANDLE); // Never pass a retired oldSwapchain again.
        }
        assert(commits == 1 && presentation.IsValid());
        // A native handle may be reused once its previous object is destroyed.
        assert(recorder.live.size() == 1 && recorder.live.count(presentation.GetSwapchain()));
        assert(presentation.AcquireNextImage(0, image) == vk::SwapchainResult::Success);
        assert(recorder.acquired.back() == presentation.GetSwapchain());
        presentation.Destroy();
        presentation.Destroy();
        const auto before = recorder.acquired.size();
        assert(presentation.AcquireNextImage(0, image) == vk::SwapchainResult::Error);
        assert(recorder.acquired.size() == before);
        std::cout << "swapchain_recreation case=" << int(fault) << " cleanup_complete=true\n";
    }
    RenderResizedWindows(context, queues, window);
    queues.Destroy();
}
} // namespace swapchain_recreation_test
