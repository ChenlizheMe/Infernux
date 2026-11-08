/**
 * @file AsyncTransferContext.cpp
 * @brief Implementation of the pooled async-upload pipeline.
 */

#include "AsyncTransferContext.h"
#include "VulkanQueueManager.h"
#include <algorithm>
#include <core/log/InxLog.h>
#include <stdexcept>

namespace infernux
{
namespace vk
{

namespace
{

constexpr uint64_t kPollTimeoutNs = 50'000'000; // 50 ms

VkResult WaitFence(VkDevice device, VkFence fence)
{
    // Transfers signal in milliseconds; a fence still unsignaled after this
    // budget means the device is lost or the driver is wedged. Bounding the
    // wait keeps Destroy() (engine shutdown) from hanging forever on it.
    constexpr int kMaxPolls = 200; // 10 s total at 50 ms per poll
    for (int poll = 0; poll < kMaxPolls; ++poll) {
        VkResult result = vkWaitForFences(device, 1, &fence, VK_TRUE, kPollTimeoutNs);
        if (result == VK_SUCCESS) {
            return result;
        }
        if (result != VK_TIMEOUT) {
            INXLOG_ERROR("AsyncTransferContext::WaitFence vkWaitForFences failed: ", VkResultToString(result));
            return result;
        }
        // Pumping events here is intentionally omitted — async transfer
        // rarely takes anywhere near 50 ms; if it does, the OS already has
        // the main thread's loop pumping events on its own cadence.
    }
    INXLOG_ERROR("AsyncTransferContext::WaitFence abandoned after 10 s (device lost or driver hang)");
    return VK_TIMEOUT;
}

} // namespace

AsyncTransferContext::~AsyncTransferContext()
{
    Destroy();
}

bool AsyncTransferContext::Initialize(VkDevice device, VulkanQueueManager &queueManager, rhi::QueueRole submissionRole,
                                      bool publishInFlight)
{
    const auto queue = queueManager.GetSnapshot(submissionRole);
    if (device == VK_NULL_HANDLE || queue.queue == VK_NULL_HANDLE)
        return false;

    // Pool flags:
    //   - TRANSIENT_BIT: hint that command buffers are short-lived (driver
    //     can choose a faster allocation strategy)
    //   - RESET_COMMAND_BUFFER_BIT: required for vkResetCommandBuffer in
    //     the recycling path (otherwise we'd have to reset the entire pool
    //     on every reuse)
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queue.family;

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &m_pool) != VK_SUCCESS) {
        INXLOG_ERROR("AsyncTransferContext::Initialize: vkCreateCommandPool failed (family ", queue.family, ")");
        return false;
    }

    m_device = device;
    m_queueFamily = queue.family;
    m_queueManager = &queueManager;
    m_submissionRole = submissionRole;
    m_publicationTimeline = publishInFlight ? queueManager.GetDependencyTimeline(submissionRole) : VK_NULL_HANDLE;
    INXLOG_INFO("AsyncTransferContext initialized on queue family ", queue.family);
    return true;
}

void AsyncTransferContext::Destroy() noexcept
{
    if (m_device == VK_NULL_HANDLE) {
        return;
    }

    // Teardown requires completed submissions before destroying pool objects.
    {
        std::lock_guard<std::mutex> guard(m_mutex);
        for (const auto &up : m_inFlight) {
            if (up.fence != VK_NULL_HANDLE) {
                const VkResult result = WaitFence(m_device, up.fence);
                if (result != VK_SUCCESS && result != VK_ERROR_DEVICE_LOST) {
                    INXLOG_ERROR("AsyncTransferContext teardown retained pending Vulkan objects after wait failure");
                    return;
                }
                if (result == VK_SUCCESS)
                    m_queueManager->MarkCompleted(up.ticket);
            }
        }
        m_inFlight.clear();

        for (VkFence fence : m_allFences) {
            if (fence != VK_NULL_HANDLE) {
                vkDestroyFence(m_device, fence, nullptr);
            }
        }
        m_allFences.clear();
        m_freeFences.clear();

        // Command buffers are owned by the pool; destroying the pool below
        // frees them implicitly.
        m_allCmdBuffers.clear();
        m_freeCmdBuffers.clear();
    }

    if (m_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(m_device, m_pool, nullptr);
        m_pool = VK_NULL_HANDLE;
    }
    m_device = VK_NULL_HANDLE;
    m_queueFamily = 0;
    m_queueManager = nullptr;
    m_submissionRole = rhi::QueueRole::Transfer;
    m_publicationTimeline = VK_NULL_HANDLE;
}

VkCommandBuffer AsyncTransferContext::AcquireCommandBufferLocked()
{
    if (!m_freeCmdBuffers.empty()) {
        VkCommandBuffer cmd = m_freeCmdBuffers.back();
        m_freeCmdBuffers.pop_back();
        if (vkResetCommandBuffer(cmd, 0) != VK_SUCCESS)
            throw std::runtime_error("Async upload command buffer reset failed");
        return cmd;
    }

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(m_device, &allocInfo, &cmd) != VK_SUCCESS) {
        INXLOG_ERROR("AsyncTransferContext: vkAllocateCommandBuffers failed");
        return VK_NULL_HANDLE;
    }

    m_allCmdBuffers.push_back(cmd);
    return cmd;
}

VkFence AsyncTransferContext::AcquireFenceLocked()
{
    if (!m_freeFences.empty()) {
        VkFence fence = m_freeFences.back();
        m_freeFences.pop_back();
        if (vkResetFences(m_device, 1, &fence) != VK_SUCCESS)
            throw std::runtime_error("Async upload fence reset failed");
        return fence;
    }

    VkFence fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    if (vkCreateFence(m_device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
        INXLOG_ERROR("AsyncTransferContext: vkCreateFence failed");
        return VK_NULL_HANDLE;
    }

    m_allFences.push_back(fence);
    return fence;
}

VkCommandBuffer AsyncTransferContext::Begin()
{
    if (m_device == VK_NULL_HANDLE) {
        INXLOG_ERROR("AsyncTransferContext::Begin called on uninitialised context");
        return VK_NULL_HANDLE;
    }

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    {
        std::lock_guard<std::mutex> guard(m_mutex);
        // Reap completed uploads opportunistically — most frames there will
        // be a handful of small uploads in flight, and reaping here keeps
        // the free-list warm without a dedicated background thread.
        ReapCompletedLocked();
        cmd = AcquireCommandBufferLocked();
    }

    if (cmd == VK_NULL_HANDLE) {
        return VK_NULL_HANDLE;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS) {
        INXLOG_ERROR("AsyncTransferContext::Begin: vkBeginCommandBuffer failed");
        std::lock_guard<std::mutex> guard(m_mutex);
        m_freeCmdBuffers.push_back(cmd);
        return VK_NULL_HANDLE;
    }

    return cmd;
}

rhi::SubmissionTicket AsyncTransferContext::EndAsync(VkCommandBuffer cmd)
{
    if (cmd == VK_NULL_HANDLE)
        return {};

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        std::lock_guard<std::mutex> guard(m_mutex);
        m_freeCmdBuffers.push_back(cmd);
        throw std::runtime_error("Async upload command recording could not finish");
    }

    VkFence fence = VK_NULL_HANDLE;
    {
        std::lock_guard<std::mutex> guard(m_mutex);
        fence = AcquireFenceLocked();
    }
    if (fence == VK_NULL_HANDLE) {
        std::lock_guard<std::mutex> guard(m_mutex);
        m_freeCmdBuffers.push_back(cmd);
        return {};
    }

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;

    // Reserve fence tracking storage before the queue can accept work.
    std::lock_guard<std::mutex> guard(m_mutex);
    try {
        if (m_inFlight.size() == m_inFlight.capacity())
            m_inFlight.reserve((std::max)(size_t{1}, m_inFlight.capacity() * 2));
    } catch (...) {
        m_freeFences.push_back(fence);
        m_freeCmdBuffers.push_back(cmd);
        throw;
    }
    const auto ticket = m_queueManager->Reserve(m_submissionRole);
    const VkResult submitResult = m_queueManager->SubmitReserved(ticket, submit, fence);
    if (submitResult != VK_SUCCESS) {
        INXLOG_ERROR("AsyncTransferContext::EndAsync: vkQueueSubmit failed: ", VkResultToString(submitResult));
        m_freeFences.push_back(fence);
        m_freeCmdBuffers.push_back(cmd);
        return {};
    }

    m_inFlight.push_back({fence, cmd, ticket});
    return ticket;
}

void AsyncTransferContext::ReapCompletedLocked()
{
    for (auto it = m_inFlight.begin(); it != m_inFlight.end();) {
        VkResult result = vkGetFenceStatus(m_device, it->fence);
        if (result == VK_SUCCESS) {
            it = RetireLocked(it);
        } else if (result == VK_NOT_READY) {
            ++it;
        } else {
            throw std::runtime_error("Async upload fence polling failed");
        }
    }
}

std::vector<AsyncTransferContext::InFlightUpload>::iterator
AsyncTransferContext::RetireLocked(std::vector<InFlightUpload>::iterator completed)
{
    m_queueManager->MarkCompleted(completed->ticket);
    m_freeFences.push_back(completed->fence);
    m_freeCmdBuffers.push_back(completed->cmd);
    return m_inFlight.erase(completed);
}

bool AsyncTransferContext::IsComplete(rhi::SubmissionTicket submission)
{
    if (!submission.IsValid() || m_device == VK_NULL_HANDLE)
        return true;

    std::lock_guard<std::mutex> guard(m_mutex);
    const auto pending = std::find_if(m_inFlight.begin(), m_inFlight.end(),
                                      [&](const auto &upload) { return upload.ticket == submission; });
    if (pending == m_inFlight.end())
        return true;
    const auto result = vkGetFenceStatus(m_device, pending->fence);
    if (result == VK_NOT_READY)
        return false;
    if (result != VK_SUCCESS)
        throw std::runtime_error("Async upload fence polling failed");
    RetireLocked(pending);
    return true;
}

void AsyncTransferContext::Wait(rhi::SubmissionTicket submission)
{
    if (!submission.IsValid() || m_device == VK_NULL_HANDLE)
        return;

    std::lock_guard<std::mutex> guard(m_mutex);
    const auto pending = std::find_if(m_inFlight.begin(), m_inFlight.end(),
                                      [&](const auto &upload) { return upload.ticket == submission; });
    if (pending == m_inFlight.end())
        return;
    if (WaitFence(m_device, pending->fence) != VK_SUCCESS)
        throw std::runtime_error("Async upload wait failed; submission resources remain pending");
    RetireLocked(pending);
}

} // namespace vk
} // namespace infernux
