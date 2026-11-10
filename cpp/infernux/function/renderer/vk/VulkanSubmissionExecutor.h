#pragma once

#include "VulkanBackendApi.h"

#include "VulkanCommandUploads.h"
#include "VulkanSubmissionWaits.h"

#include <function/renderer/rhi/RenderSubmissionPlan.h>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace infernux::vk
{

class VkDeviceContext;
class VulkanQueueManager;

/// Records and submits one backend-neutral submission plan on a Vulkan device.
/// Includes standalone compute/transfer plans without graphics or presentation.
/// Command pools are frame-slot and queue-role owned; queue ordering and GPU
/// completion are delegated to VulkanQueueManager.
class VulkanSubmissionExecutor
{
  public:
    using BatchRecorder = std::function<bool(uint32_t batchIndex, VkCommandBuffer commandBuffer)>;

    struct ExternalSync
    {
        // Recording can add actual resource producers before queue reservation.
        std::vector<SubmissionWait> producers;
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkPipelineStageFlags imageAvailableStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSemaphore renderFinished = VK_NULL_HANDLE;
        VkFence completionFence = VK_NULL_HANDLE;
        VkSemaphore previousFrameTimeline = VK_NULL_HANDLE;
        uint64_t previousFrameTimelineValue = 0;
        VkPipelineStageFlags previousFrameStages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        // Generation-prime compute can reuse resident resources from the
        // previous graph. In that case the dependency must gate the first
        // submitted batch, rather than only the first Graphics batch.
        bool previousFrameWaitAtFirstBatch = false;
        rhi::SubmissionSerial completionEpoch = rhi::InvalidSubmissionSerial;
        std::array<double, 3> *cpuTimings = nullptr;
    };

    struct ExecuteResult
    {
        VkResult result = VK_ERROR_INITIALIZATION_FAILED;
        rhi::SubmissionTicket completionTicket{};
        VkSemaphore completionTimeline = VK_NULL_HANDLE;
        uint64_t completionTimelineValue = 0;
        // Prefix of the plan accepted by the queues, even if a later submit fails.
        size_t submittedBatchCount = 0;

        [[nodiscard]] bool Succeeded() const noexcept
        {
            return result == VK_SUCCESS && completionTicket.IsValid();
        }
    };

    VulkanSubmissionExecutor() = default;
    INFERNUX_VULKAN_BACKEND_API ~VulkanSubmissionExecutor();
    VulkanSubmissionExecutor(const VulkanSubmissionExecutor &) = delete;
    VulkanSubmissionExecutor &operator=(const VulkanSubmissionExecutor &) = delete;

    INFERNUX_VULKAN_BACKEND_API bool Initialize(VkDeviceContext &device, VulkanQueueManager &queues,
                                                uint32_t frameSlotCount);
    INFERNUX_VULKAN_BACKEND_API void Destroy() noexcept;

    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API ExecuteResult Execute(uint32_t frameSlot, const rhi::SubmissionPlan &plan,
                                                                    const BatchRecorder &recorder,
                                                                    const ExternalSync &sync);

    /// Publish completion for every queue ticket that contributed to a frame
    /// after the frame completion fence has signaled.
    INFERNUX_VULKAN_BACKEND_API void CompleteFrame(uint32_t frameSlot) noexcept;

    /// Actual retained page capacities across slots and roles. Observation
    /// neither collects completion nor resets or releases cached storage.
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API uint64_t GetRetainedUploadBytes() const noexcept;

    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API VkSemaphore GetDependencyTimeline(rhi::QueueRole role) const noexcept;

  private:
    friend class VulkanComputeQueue;
    [[nodiscard]] VulkanCommandUploads &CommandUploads(uint32_t frameSlot, rhi::QueueRole role) noexcept;

    struct RolePool
    {
        VkCommandPool pool = VK_NULL_HANDLE;
        std::vector<VkCommandBuffer> buffers;
        std::unique_ptr<VulkanCommandUploads> uploads;
        uint32_t cursor = 0;
    };

    struct FrameState
    {
        std::array<RolePool, static_cast<size_t>(rhi::QueueRole::Count)> roles{};
        std::vector<rhi::SubmissionTicket> submittedTickets;
    };

    [[nodiscard]] bool CreatePools(FrameState &frame);
    [[nodiscard]] VkCommandBuffer AcquireCommandBuffer(FrameState &frame, rhi::QueueRole role);

    VkDeviceContext *m_deviceContext = nullptr;
    VulkanQueueManager *m_queues = nullptr;
    VkDevice m_device = VK_NULL_HANDLE;
    // Borrowed device-owned timelines; executors own command pools, not queue identity.
    std::vector<VkSemaphore> m_laneTimelines;
    std::vector<FrameState> m_frames;
};

} // namespace infernux::vk
