#pragma once

#include "VulkanBackendApi.h"

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
    INFERNUX_VULKAN_BACKEND_API VulkanComputeQueue();
    INFERNUX_VULKAN_BACKEND_API ~VulkanComputeQueue() override;
    INFERNUX_VULKAN_BACKEND_API bool Initialize(VkDeviceContext &context, VulkanQueueManager &queues,
                                                uint32_t slots = 3);
    INFERNUX_VULKAN_BACKEND_API void Destroy() noexcept;
    [[nodiscard]] bool IsInitialized() const noexcept
    {
        return m_context != nullptr;
    }
    INFERNUX_VULKAN_BACKEND_API rhi::SubmissionTicket Submit(const Recorder &record,
                                                             rhi::ComputeSubmissionResources resources = {}) override;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API std::optional<rhi::SubmissionTicket>
    TrySubmit(const Recorder &record, const rhi::ComputeSubmissionResources &resources = {},
              const Admission &admit = {}) override;
    INFERNUX_VULKAN_BACKEND_API void Wait(rhi::SubmissionTicket ticket) override;
    INFERNUX_VULKAN_BACKEND_API bool IsComplete(rhi::SubmissionTicket ticket) override;
    INFERNUX_VULKAN_BACKEND_API void Collect() override;
    INFERNUX_VULKAN_BACKEND_API bool SetProfilingEnabled(bool enabled) override;
    rhi::GpuTimestampFrame GetProfile() const override
    {
        return m_timestamps ? m_timestamps->LatestFrame() : rhi::GpuTimestampFrame{};
    }
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API uint64_t GetPendingSubmissionCount() const noexcept override;
    [[nodiscard]] uint64_t GetRetainedUploadBytes() const noexcept override
    {
        return m_executor.GetRetainedUploadBytes();
    }
    /// Snapshot of the latest successful submission. Timeline handles are
    /// borrowed from the device's queue manager.
    [[nodiscard]] VulkanSubmissionExecutor::ExecuteResult LastSubmission() const noexcept
    {
        return m_lastSubmission;
    }

  private:
    struct Slot
    {
        VkFence fence = VK_NULL_HANDLE;
        rhi::SubmissionTicket ticket;
        rhi::SubmissionSerial epoch = rhi::InvalidSubmissionSerial;
        rhi::ComputeSubmissionResources resources;
    };
    void Complete(uint32_t index);
    uint32_t Find(rhi::SubmissionTicket ticket) const;
    uint32_t FindFreeSlot() const noexcept;
    std::optional<rhi::SubmissionTicket> SubmitToSlot(uint32_t index, const Recorder &record,
                                                      rhi::ComputeSubmissionResources resources, bool managed,
                                                      const Admission &admit);
    VkDeviceContext *m_context = nullptr;
    VulkanQueueManager *m_queues = nullptr;
    VulkanSubmissionExecutor m_executor;
    rhi::SubmissionPlan m_plan;
    std::vector<Slot> m_slots;
    uint32_t m_next = 0;
    rhi::SubmissionSerial m_completed = 0;
    VulkanSubmissionExecutor::ExecuteResult m_lastSubmission;
    std::unique_ptr<GpuTimestampQueries> m_timestamps;
};

} // namespace infernux::vk
