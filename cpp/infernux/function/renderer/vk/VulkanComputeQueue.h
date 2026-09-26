#pragma once

#include "GpuTimestampQueries.h"
#include "VulkanSubmissionExecutor.h"
#include <function/renderer/rhi/RhiComputeQueue.h>
#include <memory>

namespace infernux::vk
{

/// A bounded command/fence pool within the existing engine device and queue
/// manager. No VkQueue, allocator, device, or independent scheduler is created.
class VulkanComputeQueue final : public rhi::ComputeQueue
{
  public:
    VulkanComputeQueue() = default;
    ~VulkanComputeQueue() override;
    bool Initialize(VkDeviceContext &context, VulkanQueueManager &queues, uint32_t slots = 3);
    void Destroy() noexcept;
    [[nodiscard]] bool IsInitialized() const noexcept
    {
        return m_context != nullptr;
    }
    rhi::SubmissionTicket Submit(const Recorder &record, std::shared_ptr<void> resources = {}) override;
    void Wait(rhi::SubmissionTicket ticket) override;
    bool IsComplete(rhi::SubmissionTicket ticket) override;
    void Collect() override;
    bool SetProfilingEnabled(bool enabled) override;
    rhi::GpuTimestampFrame GetProfile() const override
    {
        return m_timestamps ? m_timestamps->LatestFrame() : rhi::GpuTimestampFrame{};
    }
    [[nodiscard]] uint64_t GetPendingSubmissionCount() const noexcept override;
    /// Snapshot of the latest successful submission. Timeline handles are
    /// borrowed from this queue; consumers must complete before Destroy().
    [[nodiscard]] VulkanSubmissionExecutor::ExecuteResult LastSubmission() const noexcept
    {
        return m_lastSubmission;
    }
    /// Resolve one still-pending compute ticket to its exact Vulkan timeline
    /// dependency. Completed tickets require no queue wait and return empty.
    [[nodiscard]] VulkanSubmissionExecutor::ExecuteResult DependencyFor(rhi::SubmissionTicket ticket) const;
    /// Order background compute after the latest submitted graphics frame.
    /// Resident buffers are shared directly with vertex input, so a new
    /// in-place write must not race the previous frame's read.
    void SetPreviousGraphicsCompletion(VkSemaphore timeline, uint64_t value) noexcept;

  private:
    struct Slot
    {
        VkFence fence = VK_NULL_HANDLE;
        rhi::SubmissionTicket ticket;
        rhi::SubmissionSerial epoch = rhi::InvalidSubmissionSerial;
        VulkanSubmissionExecutor::ExecuteResult submission;
        std::shared_ptr<void> resources;
    };
    void Complete(uint32_t index);
    uint32_t Find(rhi::SubmissionTicket ticket) const;
    VkDeviceContext *m_context = nullptr;
    VulkanQueueManager *m_queues = nullptr;
    VulkanSubmissionExecutor m_executor;
    rhi::SubmissionPlan m_plan;
    std::vector<Slot> m_slots;
    uint32_t m_next = 0;
    rhi::SubmissionSerial m_completed = 0;
    VulkanSubmissionExecutor::ExecuteResult m_lastSubmission;
    VkSemaphore m_previousGraphicsTimeline = VK_NULL_HANDLE;
    uint64_t m_previousGraphicsTimelineValue = 0;
    std::unique_ptr<GpuTimestampQueries> m_timestamps;
};

} // namespace infernux::vk
