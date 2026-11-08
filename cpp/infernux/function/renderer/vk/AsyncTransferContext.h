#pragma once

#include "VulkanBackendApi.h"

#include "VkTypes.h"
#include <cstdint>
#include <function/renderer/rhi/RhiSubmission.h>
#include <mutex>
#include <vector>
#include <vulkan/vulkan.h>

namespace infernux
{
namespace vk
{

class VulkanQueueManager;

/// Owns upload command buffers and fences. Queue identities and timeline
/// signals belong to the device's VulkanQueueManager.
class AsyncTransferContext
{
  public:
    AsyncTransferContext() = default;
    INFERNUX_VULKAN_BACKEND_API ~AsyncTransferContext();

    AsyncTransferContext(const AsyncTransferContext &) = delete;
    AsyncTransferContext &operator=(const AsyncTransferContext &) = delete;
    AsyncTransferContext(AsyncTransferContext &&) = delete;
    AsyncTransferContext &operator=(AsyncTransferContext &&) = delete;

    /// Allocate command buffers in the selected native queue family.
    INFERNUX_VULKAN_BACKEND_API bool Initialize(VkDevice device, VulkanQueueManager &queueManager,
                                                rhi::QueueRole submissionRole, bool publishInFlight = true);

    /**
     * @brief Tear down all pooled fences / command buffers.
     *
     * Must be called BEFORE the owning device is destroyed. Idempotent.
     */
    INFERNUX_VULKAN_BACKEND_API void Destroy() noexcept;

    /// Begin recording; the uploader owns resource layout and visibility.
    [[nodiscard]] VkCommandBuffer Begin();

    /**
     * @brief Submit asynchronously. Returns a handle that can be polled
     *        via IsComplete() or blocked on via Wait(). The caller is
     *        responsible for keeping any source CPU staging memory alive
     *        until completion.
     */
    [[nodiscard]] rhi::SubmissionTicket EndAsync(VkCommandBuffer cmd);

    /**
     * @brief Non-blocking completion poll. Returns true once the GPU has
     *        signalled the upload's fence (also recycles the fence + cmd
     *        buffer back into the free list as a side effect).
     */
    [[nodiscard]] bool IsComplete(rhi::SubmissionTicket submission);

    /**
     * @brief Block until the upload completes. Returns immediately if the
     *        handle is invalid or already retired.
     */
    void Wait(rhi::SubmissionTicket submission);

    /// Native queue family of the upload command pool.
    [[nodiscard]] uint32_t GetQueueFamily() const noexcept
    {
        return m_queueFamily;
    }

    [[nodiscard]] VkSemaphore GetTimelineSemaphore() const noexcept
    {
        return m_publicationTimeline;
    }

  private:
    struct InFlightUpload
    {
        VkFence fence = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        rhi::SubmissionTicket ticket;
    };

    /// @brief Recycle (or lazily create) a fence ready for re-submission.
    [[nodiscard]] VkFence AcquireFenceLocked();

    /// @brief Recycle (or lazily create + record-prep) a command buffer.
    [[nodiscard]] VkCommandBuffer AcquireCommandBufferLocked();

    /// @brief Reap any in-flight uploads whose fence has signalled.
    void ReapCompletedLocked();

    std::vector<InFlightUpload>::iterator RetireLocked(std::vector<InFlightUpload>::iterator completed);

    VkDevice m_device = VK_NULL_HANDLE;
    VkCommandPool m_pool = VK_NULL_HANDLE;
    uint32_t m_queueFamily = 0;
    VulkanQueueManager *m_queueManager = nullptr;
    rhi::QueueRole m_submissionRole = rhi::QueueRole::Transfer;
    VkSemaphore m_publicationTimeline = VK_NULL_HANDLE;

    std::mutex m_mutex;
    std::vector<VkFence> m_freeFences;
    std::vector<VkFence> m_allFences;
    std::vector<VkCommandBuffer> m_freeCmdBuffers;
    std::vector<VkCommandBuffer> m_allCmdBuffers;
    std::vector<InFlightUpload> m_inFlight;
};

} // namespace vk
} // namespace infernux
