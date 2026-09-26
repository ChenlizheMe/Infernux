#include "VulkanComputeQueue.h"
#include "RhiVulkanTypes.h"
#include "VkDeviceContext.h"
#include "VulkanQueueManager.h"
#include "VulkanRhiDevice.h"
#include <algorithm>
#include <chrono>
#include <exception>
#include <stdexcept>

namespace infernux::vk
{
namespace
{
class RecordingContext final : public rhi::ComputeRecordingContext
{
  public:
    RecordingContext(VulkanRhiDevice &device, VkCommandBuffer commands) : m_commands(commands)
    {
        m_compute = device.MakeComputeCommandEncoder(m_computeContext, commands);
        m_transfer = device.MakeTransferCommandEncoder(m_transferContext, commands);
    }
    rhi::ComputeCommandEncoder Compute() override
    {
        return m_compute;
    }
    rhi::TransferCommandEncoder Transfer() override
    {
        return m_transfer;
    }
    void PipelineBarrier(rhi::PipelineStage source, rhi::Access sourceAccess, rhi::PipelineStage destination,
                         rhi::Access destinationAccess) override
    {
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = rhi::ToVkAccessFlags(sourceAccess);
        barrier.dstAccessMask = rhi::ToVkAccessFlags(destinationAccess);
        vkCmdPipelineBarrier(m_commands, rhi::ToVkPipelineStages(source, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT),
                             rhi::ToVkPipelineStages(destination, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT), 0, 1, &barrier,
                             0, nullptr, 0, nullptr);
    }

  private:
    VkCommandBuffer m_commands;
    VulkanComputeCommandContext m_computeContext;
    VulkanTransferCommandContext m_transferContext;
    rhi::ComputeCommandEncoder m_compute;
    rhi::TransferCommandEncoder m_transfer;
};
} // namespace

VulkanComputeQueue::~VulkanComputeQueue()
{
    Destroy();
}

bool VulkanComputeQueue::Initialize(VkDeviceContext &context, VulkanQueueManager &queues, uint32_t slots)
{
    if (IsInitialized() || !slots || !context.IsValid() || queues.GetDeviceId() != context.GetDeviceId())
        return false;
    m_context = &context;
    m_queues = &queues;
    m_slots.resize(slots);
    std::string error;
    if (!rhi::BuildSubmissionPlan({{0,
                                    context.GetDeviceId(),
                                    rhi::QueueRole::Compute,
                                    rhi::SubmissionDomain::Background,
                                    rhi::InvalidRenderViewId,
                                    rhi::PipelineStage::ComputeShader,
                                    {}}},
                                  m_plan, error)) {
        Destroy();
        return false;
    }
    if (!m_executor.Initialize(context, queues, slots)) {
        Destroy();
        return false;
    }
    VkFenceCreateInfo info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    for (auto &slot : m_slots) {
        if (vkCreateFence(context.GetDevice(), &info, nullptr, &slot.fence) != VK_SUCCESS) {
            Destroy();
            return false;
        }
    }
    return true;
}

void VulkanComputeQueue::Destroy() noexcept
{
    if (!m_context)
        return;
    for (uint32_t i = 0; i < m_slots.size(); ++i) {
        auto &slot = m_slots[i];
        if (slot.ticket.IsValid() &&
            vkWaitForFences(m_context->GetDevice(), 1, &slot.fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS)
            Complete(i);
        if (slot.fence != VK_NULL_HANDLE)
            vkDestroyFence(m_context->GetDevice(), slot.fence, nullptr);
    }
    m_timestamps.reset();
    m_executor.Destroy();
    m_slots.clear();
    m_plan.Clear();
    m_context = nullptr;
    m_queues = nullptr;
    m_next = 0;
    m_completed = 0;
    m_lastSubmission = {};
    m_previousGraphicsTimeline = VK_NULL_HANDLE;
    m_previousGraphicsTimelineValue = 0;
}

void VulkanComputeQueue::SetPreviousGraphicsCompletion(VkSemaphore timeline, uint64_t value) noexcept
{
    m_previousGraphicsTimeline = timeline;
    m_previousGraphicsTimelineValue = timeline == VK_NULL_HANDLE ? 0 : value;
}

void VulkanComputeQueue::Complete(uint32_t index)
{
    auto &slot = m_slots[index];
    if (m_timestamps)
        m_timestamps->CollectCompletedFrame(index);
    m_executor.CompleteFrame(index);
    m_queues->CompleteCompletionEpoch(slot.epoch);
    m_completed = (std::max)(m_completed, slot.ticket.serial);
    slot.ticket = {};
    slot.epoch = rhi::InvalidSubmissionSerial;
    slot.submission = {};
    // Publish completion before releasing owners whose older tickets may be queried.
    slot.resources.reset();
}

uint32_t VulkanComputeQueue::Find(rhi::SubmissionTicket ticket) const
{
    if (!m_context || !ticket.IsValid() || ticket.device != m_context->GetDeviceId() ||
        ticket.queue != rhi::QueueRole::Compute || ticket.serial > m_lastSubmission.completionTicket.serial)
        throw std::invalid_argument("Compute completion does not belong to this host queue");
    for (uint32_t i = 0; i < m_slots.size(); ++i)
        if (m_slots[i].ticket.serial == ticket.serial)
            return i;
    if (ticket.serial <= m_completed)
        return UINT32_MAX;
    throw std::invalid_argument("Unknown compute submission ticket");
}

VulkanSubmissionExecutor::ExecuteResult VulkanComputeQueue::DependencyFor(rhi::SubmissionTicket ticket) const
{
    const auto index = Find(ticket);
    return index == UINT32_MAX ? VulkanSubmissionExecutor::ExecuteResult{} : m_slots[index].submission;
}

void VulkanComputeQueue::Wait(rhi::SubmissionTicket ticket)
{
    const auto index = Find(ticket);
    if (index == UINT32_MAX)
        return;
    const auto begin = std::chrono::steady_clock::now();
    if (vkWaitForFences(m_context->GetDevice(), 1, &m_slots[index].fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        throw std::runtime_error("Compute submission wait failed");
    const auto end = std::chrono::steady_clock::now();
    RecordWait(std::chrono::duration<double, std::milli>(end - begin).count());
    Complete(index);
    Collect();
}

bool VulkanComputeQueue::IsComplete(rhi::SubmissionTicket ticket)
{
    const auto index = Find(ticket);
    if (index == UINT32_MAX)
        return true;
    const auto status = vkGetFenceStatus(m_context->GetDevice(), m_slots[index].fence);
    if (status == VK_NOT_READY)
        return false;
    if (status != VK_SUCCESS)
        throw std::runtime_error("Compute fence query failed");
    Complete(index);
    return true;
}

void VulkanComputeQueue::Collect()
{
    if (!m_context)
        return;
    for (uint32_t i = 0; i < m_slots.size(); ++i) {
        if (!m_slots[i].ticket.IsValid())
            continue;
        const auto status = vkGetFenceStatus(m_context->GetDevice(), m_slots[i].fence);
        if (status == VK_SUCCESS)
            Complete(i);
        else if (status != VK_NOT_READY)
            throw std::runtime_error("Compute fence query failed");
    }
    // The engine's central retirement pass consumes the completed epoch.
}

bool VulkanComputeQueue::SetProfilingEnabled(bool enabled)
{
    if (!m_context)
        throw std::logic_error("Compute profiling requires an initialized host");
    if (enabled == static_cast<bool>(m_timestamps))
        return true;
    for (auto &slot : m_slots)
        if (slot.ticket.IsValid())
            Wait(slot.ticket);
    if (!enabled) {
        m_timestamps.reset();
        return true;
    }
    auto queries = std::make_unique<GpuTimestampQueries>();
    if (!queries->Initialize(*m_context, static_cast<uint32_t>(m_slots.size()), 1,
                             m_queues->GetSnapshot(rhi::QueueRole::Compute).family))
        return false;
    m_timestamps = std::move(queries);
    return true;
}

uint64_t VulkanComputeQueue::GetPendingSubmissionCount() const noexcept
{
    return static_cast<uint64_t>(
        std::count_if(m_slots.begin(), m_slots.end(), [](const Slot &slot) { return slot.ticket.IsValid(); }));
}

rhi::SubmissionTicket VulkanComputeQueue::Submit(const Recorder &record, std::shared_ptr<void> resources)
{
    if (!m_context || !record)
        throw std::invalid_argument("Compute submission requires an initialized host and recorder");
    Collect();
    const auto index = m_next;
    auto &slot = m_slots[index];
    if (slot.ticket.IsValid())
        Wait(slot.ticket); // Bounded capacity, never reuse pending commands.
    if (vkResetFences(m_context->GetDevice(), 1, &slot.fence) != VK_SUCCESS)
        throw std::runtime_error("Compute fence reset failed");
    slot.epoch = m_queues->ReserveCompletionEpoch();
    VulkanSubmissionExecutor::ExternalSync sync;
    sync.completionFence = slot.fence;
    sync.completionEpoch = slot.epoch;
    sync.previousFrameTimeline = m_previousGraphicsTimeline;
    sync.previousFrameTimelineValue = m_previousGraphicsTimelineValue;
    sync.previousFrameStages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    sync.previousFrameWaitAtFirstBatch = true;
    std::exception_ptr recordingError;
    const auto submitBegin = std::chrono::steady_clock::now();
    const auto result = m_executor.Execute(
        index, m_plan,
        [&](uint32_t, VkCommandBuffer commands) {
            rhi::TimestampRegionHandle region;
            if (m_timestamps) {
                m_timestamps->BeginFrame(commands, index);
                region = m_timestamps->BeginRegion(commands, "compute_submission", VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
            }
            RecordingContext context(m_context->GetRhiDevice(), commands);
            bool recorded = false;
            try {
                recorded = record(context);
            } catch (...) {
                recordingError = std::current_exception();
            }
            if (m_timestamps) {
                m_timestamps->EndRegion(commands, region, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
                m_timestamps->FinishFrame(index);
            }
            return recorded;
        },
        sync);
    if (!result.Succeeded()) {
        // A one-batch plan has no earlier submitted batches to roll back.
        if (!result.submittedAny)
            m_queues->CompleteCompletionEpoch(slot.epoch);
        slot.epoch = rhi::InvalidSubmissionSerial;
        if (recordingError)
            std::rethrow_exception(recordingError);
        throw std::runtime_error("Compute submission failed");
    }
    slot.ticket = result.completionTicket;
    slot.submission = result;
    slot.resources = std::move(resources);
    const auto submitEnd = std::chrono::steady_clock::now();
    RecordSubmission(std::chrono::duration<double, std::milli>(submitEnd - submitBegin).count());
    if (m_timestamps)
        m_timestamps->MarkSubmitted(index);
    m_lastSubmission = result;
    m_next = (index + 1) % static_cast<uint32_t>(m_slots.size());
    return slot.ticket;
}

} // namespace infernux::vk
