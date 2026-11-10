#pragma once

#include "VulkanBackendApi.h"

#include <function/renderer/rhi/RhiSubmission.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>

namespace infernux::vk
{
/// Accepted producers consumed by one command batch. Serial numbers are
/// ordered within a role only; aliases are merged after resolving the semaphore.
struct SubmissionReads
{
    struct Read
    {
        rhi::SubmissionTicket producer;
        VkPipelineStageFlags stages = 0;
    };
    std::array<Read, static_cast<size_t>(rhi::QueueRole::Count)> reads{};
    bool empty = true;

    void Add(rhi::SubmissionTicket producer, VkPipelineStageFlags stages = 0) noexcept
    {
        if (!producer.IsValid())
            return;
        assert(producer.queue < rhi::QueueRole::Count);
        auto &read = reads[static_cast<size_t>(producer.queue)];
        assert(!read.producer.IsValid() || read.producer.device == producer.device);
        if (producer.serial > read.producer.serial)
            read.producer = producer;
        read.stages |= stages;
        empty = false;
    }

    void Add(const SubmissionReads &other, VkPipelineStageFlags stages) noexcept
    {
        if (other.empty)
            return;
        for (const auto &read : other.reads)
            Add(read.producer, read.stages | stages);
    }
};

class SubmissionReadScope
{
  public:
    SubmissionReadScope(VkCommandBuffer commands, SubmissionReads &reads) noexcept;
    ~SubmissionReadScope();
    SubmissionReadScope(const SubmissionReadScope &) = delete;
    SubmissionReadScope &operator=(const SubmissionReadScope &) = delete;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API static SubmissionReads *Current(VkCommandBuffer commands) noexcept;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API static SubmissionReads *Current() noexcept;

  private:
    VkCommandBuffer m_commands;
    SubmissionReads &m_reads;
    const SubmissionReadScope *m_previous;
    static thread_local const SubmissionReadScope *s_current;
};

struct SubmissionWait
{
    VkSemaphore timeline = VK_NULL_HANDLE;
    uint64_t value = 0;
    VkPipelineStageFlags stages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    rhi::QueueRole consumer = rhi::QueueRole::Count;
};

/// One wait per semaphore, shared by frame, compute and preview submissions.
/// Binary waits use zero values; timeline values are compared only after their
/// semaphore identity matches. Storage remains stable until queue submission.
struct SubmissionWaits
{
    std::vector<VkSemaphore> semaphores;
    std::vector<uint64_t> values;
    std::vector<VkPipelineStageFlags> stages;

    void Add(SubmissionWait wait)
    {
        const auto found = std::find(semaphores.begin(), semaphores.end(), wait.timeline);
        if (found == semaphores.end()) {
            semaphores.push_back(wait.timeline);
            values.push_back(wait.value);
            stages.push_back(wait.stages);
        } else {
            const auto index = static_cast<size_t>(found - semaphores.begin());
            values[index] = (std::max)(values[index], wait.value);
            stages[index] |= wait.stages;
        }
    }
};
} // namespace infernux::vk
