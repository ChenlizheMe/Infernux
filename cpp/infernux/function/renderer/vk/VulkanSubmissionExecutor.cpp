#include "VulkanSubmissionExecutor.h"
#include "RhiVulkanTypes.h"
#include "VulkanCommandUploads.h"

#include "DescriptorBindTrace.h"
#include "VkDeviceContext.h"
#include "VulkanQueueManager.h"
#include "VulkanRhiDevice.h"

#include <core/log/InxLog.h>

#include <algorithm>
#include <chrono>
#include <limits>

namespace infernux::vk
{

namespace
{

constexpr size_t RoleIndex(rhi::QueueRole role) noexcept
{
    return static_cast<size_t>(role);
}

} // namespace

thread_local const SubmissionReadScope *SubmissionReadScope::s_current = nullptr;

SubmissionReadScope::SubmissionReadScope(VkCommandBuffer commands, SubmissionReads &reads) noexcept
    : m_commands(commands), m_reads(reads), m_previous(s_current)
{
    s_current = this;
}

SubmissionReadScope::~SubmissionReadScope()
{
    s_current = m_previous;
}

SubmissionReads *SubmissionReadScope::Current(VkCommandBuffer commands) noexcept
{
    return s_current && s_current->m_commands == commands ? &s_current->m_reads : nullptr;
}

SubmissionReads *SubmissionReadScope::Current() noexcept
{
    return s_current ? &s_current->m_reads : nullptr;
}

VulkanSubmissionExecutor::~VulkanSubmissionExecutor()
{
    Destroy();
}

bool VulkanSubmissionExecutor::Initialize(VkDeviceContext &device, VulkanQueueManager &queues, uint32_t frameSlotCount)
{
    Destroy();
    if (!device.IsValid() || queues.GetDeviceId() != device.GetDeviceId())
        return false;

    m_deviceContext = &device;
    m_queues = &queues;
    m_device = device.GetDevice();
    m_frames.resize((std::max)(frameSlotCount, 1u));
    for (FrameState &frame : m_frames) {
        if (!CreatePools(frame)) {
            Destroy();
            return false;
        }
    }

    if (device.IsTimelineSemaphoreEnabled()) {
        uint32_t maxLane = 0;
        for (const rhi::QueueRole role :
             {rhi::QueueRole::Graphics, rhi::QueueRole::Compute, rhi::QueueRole::Transfer, rhi::QueueRole::Present}) {
            const uint32_t lane = queues.GetSnapshot(role).nativeLane;
            if (lane != UINT32_MAX)
                maxLane = (std::max)(maxLane, lane);
        }
        m_laneTimelines.resize(static_cast<size_t>(maxLane) + 1);

        for (const auto role :
             {rhi::QueueRole::Graphics, rhi::QueueRole::Compute, rhi::QueueRole::Transfer, rhi::QueueRole::Present}) {
            const auto lane = queues.GetSnapshot(role).nativeLane;
            if (lane < m_laneTimelines.size())
                m_laneTimelines[lane] = queues.GetDependencyTimeline(role);
        }
    }
    return true;
}

void VulkanSubmissionExecutor::Destroy() noexcept
{
    if (m_device != VK_NULL_HANDLE) {
        for (FrameState &frame : m_frames) {
            for (RolePool &role : frame.roles) {
                if (role.pool != VK_NULL_HANDLE)
                    vkDestroyCommandPool(m_device, role.pool, nullptr);
            }
        }
    }
    m_frames.clear();
    m_laneTimelines.clear();
    m_device = VK_NULL_HANDLE;
    m_queues = nullptr;
    m_deviceContext = nullptr;
}

VkSemaphore VulkanSubmissionExecutor::GetDependencyTimeline(rhi::QueueRole role) const noexcept
{
    if (!m_queues || role == rhi::QueueRole::Count)
        return VK_NULL_HANDLE;
    const uint32_t lane = m_queues->GetSnapshot(role).nativeLane;
    return lane < m_laneTimelines.size() ? m_laneTimelines[lane] : VK_NULL_HANDLE;
}

bool VulkanSubmissionExecutor::CreatePools(FrameState &frame)
{
    for (const rhi::QueueRole role : {rhi::QueueRole::Graphics, rhi::QueueRole::Compute, rhi::QueueRole::Transfer}) {
        const auto snapshot = m_queues->GetSnapshot(role);
        if (snapshot.queue == VK_NULL_HANDLE)
            continue;

        VkCommandPoolCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        createInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        createInfo.queueFamilyIndex = snapshot.family;
        RolePool &target = frame.roles[RoleIndex(role)];
        target.uploads = std::make_unique<VulkanCommandUploads>(m_deviceContext->GetRhiDevice());
        if (vkCreateCommandPool(m_device, &createInfo, nullptr, &target.pool) != VK_SUCCESS)
            return false;
    }
    return true;
}

VkCommandBuffer VulkanSubmissionExecutor::AcquireCommandBuffer(FrameState &frame, rhi::QueueRole role)
{
    if (role == rhi::QueueRole::Present || role == rhi::QueueRole::Count)
        return VK_NULL_HANDLE;
    RolePool &owner = frame.roles[RoleIndex(role)];
    if (owner.pool == VK_NULL_HANDLE)
        return VK_NULL_HANDLE;
    if (owner.cursor < owner.buffers.size())
        return owner.buffers[owner.cursor++];

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = owner.pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(m_device, &allocInfo, &commandBuffer) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    owner.buffers.push_back(commandBuffer);
    ++owner.cursor;
    return commandBuffer;
}

VulkanSubmissionExecutor::ExecuteResult VulkanSubmissionExecutor::Execute(uint32_t frameSlot,
                                                                          const rhi::SubmissionPlan &plan,
                                                                          const BatchRecorder &recorder,
                                                                          const ExternalSync &sync)
{
    ExecuteResult output{};
    auto phaseStart = sync.cpuTimings ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (!m_deviceContext || !m_queues || frameSlot >= m_frames.size())
        return output;
    FrameState &frame = m_frames[frameSlot];
    if (!frame.submittedTickets.empty()) {
        INXLOG_ERROR("VulkanSubmissionExecutor frame slot must complete before command and upload reuse");
        return output;
    }
    struct UnsubmittedUploads
    {
        FrameState &frame;
        ~UnsubmittedUploads()
        {
            if (frame.submittedTickets.empty())
                for (auto &role : frame.roles)
                    if (role.uploads)
                        role.uploads->Discard();
        }
    } unsubmitted{frame};
    if (plan.batches.empty() || !recorder)
        return output;
    const auto firstGraphics = std::find_if(plan.batches.begin(), plan.batches.end(),
                                            [](const auto &batch) { return batch.queue == rhi::QueueRole::Graphics; });
    if (sync.completionFence == VK_NULL_HANDLE || sync.completionEpoch == rhi::InvalidSubmissionSerial) {
        INXLOG_ERROR("VulkanSubmissionExecutor requires a completion fence and a completion epoch");
        return output;
    }
    const bool hasGraphics = firstGraphics != plan.batches.end();
    if (!hasGraphics && (sync.imageAvailable != VK_NULL_HANDLE || sync.renderFinished != VK_NULL_HANDLE)) {
        INXLOG_ERROR("Presentation synchronization requires Graphics work");
        return output;
    }

    const auto joinRole = hasGraphics ? rhi::QueueRole::Graphics : plan.batches.back().queue;
    const uint32_t joinLane = m_queues->GetSnapshot(joinRole).nativeLane;
    std::vector<bool> coveredByTerminal(plan.batches.size(), false);
    const auto markCovered = [&](auto &&self, uint32_t batchIndex) -> void {
        if (batchIndex >= plan.batches.size() || coveredByTerminal[batchIndex])
            return;
        coveredByTerminal[batchIndex] = true;
        const auto &batch = plan.batches[batchIndex];
        if (batch.queuePredecessor != rhi::InvalidSubmissionBatchIndex)
            self(self, batch.queuePredecessor);
        for (const auto &dependency : batch.waitsFor)
            self(self, dependency.sourceBatch);

        // Submission order is also execution order for logical roles that
        // alias one native VkQueue.
        const uint32_t lane = m_queues->GetSnapshot(batch.queue).nativeLane;
        for (uint32_t previous = 0; previous < batchIndex; ++previous) {
            if (m_queues->GetSnapshot(plan.batches[previous].queue).nativeLane == lane)
                self(self, previous);
        }
    };
    markCovered(markCovered, static_cast<uint32_t>(plan.batches.size() - 1));
    const bool needsTerminalJoin =
        std::any_of(coveredByTerminal.begin(), coveredByTerminal.end(), [](bool covered) { return !covered; });
    const uint32_t terminalLane =
        needsTerminalJoin ? joinLane : m_queues->GetSnapshot(plan.batches.back().queue).nativeLane;
    bool hasCrossQueueDependency = false;
    for (size_t batchIndex = 0; batchIndex < plan.batches.size(); ++batchIndex) {
        const auto &batch = plan.batches[batchIndex];
        const uint32_t targetLane = m_queues->GetSnapshot(batch.queue).nativeLane;
        for (const auto &dependency : batch.waitsFor) {
            if (dependency.sourceBatch < plan.batches.size() &&
                m_queues->GetSnapshot(plan.batches[dependency.sourceBatch].queue).nativeLane != targetLane) {
                hasCrossQueueDependency = true;
                break;
            }
        }
        // A fallback terminal join needs timeline visibility for independent
        // native lanes that are not covered by the plan's own terminal batch.
        if (batchIndex + 1 < plan.batches.size() && targetLane != terminalLane)
            hasCrossQueueDependency = true;
    }
    if ((hasCrossQueueDependency || needsTerminalJoin) && m_laneTimelines.empty()) {
        INXLOG_ERROR("VulkanSubmissionExecutor cannot execute cross-queue dependencies without timeline semaphores");
        return output;
    }

    frame.submittedTickets.reserve(plan.batches.size() + (needsTerminalJoin ? 1u : 0u));
    for (RolePool &role : frame.roles) {
        role.cursor = 0;
        if (role.pool != VK_NULL_HANDLE && vkResetCommandPool(m_device, role.pool, 0) != VK_SUCCESS)
            return output;
    }

    std::vector<rhi::SubmissionTicket> tickets;
    // Keep each compiler batch's wait/signal boundary intact. Consecutive
    // same-role batches share one host vkQueueSubmit call, not one VkSubmitInfo.
    struct BatchSync
    {
        SubmissionWaits dependencies;
        std::vector<VkSemaphore> signals;
        std::vector<uint64_t> signalValues;
        VkTimelineSemaphoreSubmitInfo timeline{};
    };
    std::vector<BatchSync> batchSync(plan.batches.size());
    std::vector<VkCommandBuffer> commands;
    std::vector<uint64_t> signalValues(plan.batches.size(), 0);
    tickets.reserve(plan.batches.size() + (needsTerminalJoin ? 1u : 0u));
    commands.reserve(plan.batches.size());

    if (sync.cpuTimings) {
        const auto now = std::chrono::steady_clock::now();
        (*sync.cpuTimings)[0] = std::chrono::duration<double, std::milli>(now - phaseStart).count();
        phaseStart = now;
    }
    for (size_t batchIndex = 0; batchIndex < plan.batches.size(); ++batchIndex) {
        const rhi::SubmissionBatch &batch = plan.batches[batchIndex];
        VkCommandBuffer commandBuffer = AcquireCommandBuffer(frame, batch.queue);
        if (commandBuffer == VK_NULL_HANDLE) {
            return output;
        }
        commands.push_back(commandBuffer);
        VulkanCommandUploads::Scope uploadScope(*frame.roles[RoleIndex(batch.queue)].uploads, commandBuffer);
        SubmissionReads reads;
        SubmissionReadScope readScope(commandBuffer, reads);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            return output;
        }

        vkdebug::DescriptorRecordingScope descriptorRecording(&m_deviceContext->GetRhiDevice().GetDescriptorManager(),
                                                              sync.completionEpoch);
        if (!recorder(static_cast<uint32_t>(batchIndex), commandBuffer) ||
            vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            return output;
        }
        for (const auto &read : reads.reads) {
            if (!read.producer.IsValid())
                continue;
            const auto point = m_queues->DependencyFor(read.producer, batch.queue);
            if (point.semaphore != VK_NULL_HANDLE)
                batchSync[batchIndex].dependencies.Add({point.semaphore, point.value, read.stages});
        }
    }
    if (sync.cpuTimings) {
        const auto now = std::chrono::steady_clock::now();
        (*sync.cpuTimings)[1] = std::chrono::duration<double, std::milli>(now - phaseStart).count();
        phaseStart = now;
    }

    // A recorder can submit an upload. Reserve these consumers only after
    // recording so an upload on an aliased lane keeps its earlier queue order.
    struct Reservations
    {
        VulkanQueueManager &queues;
        const std::vector<rhi::SubmissionTicket> &tickets;
        size_t next = 0;

        void CancelRemaining() noexcept
        {
            while (next < tickets.size())
                (void)queues.CancelReservation(tickets[next++]);
        }

        ~Reservations()
        {
            CancelRemaining();
        }
    } reservations{*m_queues, tickets};
    for (size_t first = 0; first < plan.batches.size();) {
        const auto &batch = plan.batches[first];
        size_t end = first + 1;
        while (end < plan.batches.size() && plan.batches[end].queue == batch.queue)
            ++end;
        if (std::any_of(plan.batches.begin() + first, plan.batches.begin() + end,
                        [&](const auto &item) { return item.device != m_deviceContext->GetDeviceId(); })) {
            INXLOG_ERROR("VulkanSubmissionExecutor rejected a batch for another device");
            return output;
        }
        rhi::SubmissionTicket ticket = m_queues->Reserve(batch.queue, static_cast<uint32_t>(end - first));
        if (!ticket.IsValid())
            return output;
        for (size_t index = first; index < end; ++index, ++ticket.serial)
            tickets.push_back(ticket);
        first = end;
    }
    if (needsTerminalJoin) {
        const auto joinTicket = m_queues->Reserve(joinRole);
        if (!joinTicket.IsValid())
            return output;
        tickets.push_back(joinTicket);
    }
    for (size_t index = 0; index < plan.batches.size(); ++index) {
        if (!m_laneTimelines.empty())
            signalValues[index] = tickets[index].serial;
    }
    output.completionTicket = tickets.back();

    size_t firstGraphicsBatch = plan.batches.size();
    size_t lastGraphicsBatch = plan.batches.size();
    for (size_t index = 0; index < plan.batches.size(); ++index) {
        if (plan.batches[index].queue == rhi::QueueRole::Graphics) {
            if (firstGraphicsBatch == plan.batches.size())
                firstGraphicsBatch = index;
            lastGraphicsBatch = index;
        }
    }
    const size_t previousFrameWaitBatch = (sync.previousFrameWaitAtFirstBatch || !hasGraphics) ? 0 : firstGraphicsBatch;
    std::vector<bool> laneSubmitted((std::max)(m_laneTimelines.size(), static_cast<size_t>(4)), false);
    std::array<bool, static_cast<size_t>(rhi::QueueRole::Count)> roleSubmitted{};

    std::vector<VkSubmitInfo> submits(plan.batches.size());
    for (size_t batchIndex = 0; batchIndex < plan.batches.size(); ++batchIndex) {
        const rhi::SubmissionBatch &batch = plan.batches[batchIndex];
        const uint32_t targetLane = m_queues->GetSnapshot(batch.queue).nativeLane;
        auto &storage = batchSync[batchIndex];
        auto &waits = storage.dependencies.semaphores;
        auto &waitStages = storage.dependencies.stages;
        auto &waitValues = storage.dependencies.values;
        bool usesTimeline = !waits.empty();
        const auto addWait = [&](VkSemaphore semaphore, uint64_t value, VkPipelineStageFlags stages) {
            storage.dependencies.Add({semaphore, value, stages});
            usesTimeline |= value != 0;
        };
        if (batchIndex == firstGraphicsBatch && sync.imageAvailable != VK_NULL_HANDLE) {
            addWait(sync.imageAvailable, 0, sync.imageAvailableStages);
        }
        if (batchIndex == previousFrameWaitBatch && sync.previousFrameTimeline != VK_NULL_HANDLE &&
            sync.previousFrameTimelineValue != 0) {
            addWait(sync.previousFrameTimeline, sync.previousFrameTimelineValue, sync.previousFrameStages);
        }
        const bool firstOnLane = targetLane < laneSubmitted.size() && !laneSubmitted[targetLane];
        for (const auto &producer : sync.producers) {
            const bool firstConsumer = producer.consumer == rhi::QueueRole::Count
                                           ? firstOnLane
                                           : batch.queue == producer.consumer && !roleSubmitted[RoleIndex(batch.queue)];
            if (firstConsumer)
                addWait(producer.timeline, producer.value, producer.stages);
        }

        std::vector<uint64_t> laneWaitValues(m_laneTimelines.size(), 0);
        std::vector<VkPipelineStageFlags> laneWaitStages(m_laneTimelines.size(), 0);
        const auto addLaneWait = [&](uint32_t sourceBatch, VkPipelineStageFlags stages) {
            if (sourceBatch >= batchIndex || sourceBatch >= plan.batches.size())
                return false;
            const uint32_t sourceLane = m_queues->GetSnapshot(plan.batches[sourceBatch].queue).nativeLane;
            if (sourceLane == targetLane)
                return true;
            if (sourceLane >= m_laneTimelines.size() || signalValues[sourceBatch] == 0)
                return false;
            laneWaitValues[sourceLane] = (std::max)(laneWaitValues[sourceLane], signalValues[sourceBatch]);
            laneWaitStages[sourceLane] |= stages;
            return true;
        };

        for (const rhi::SubmissionBatchDependency &dependency : batch.waitsFor) {
            if (!addLaneWait(dependency.sourceBatch,
                             rhi::ToVkPipelineStages(dependency.waitStages, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)))
                return output;
        }

        const bool finalBatch = batchIndex + 1 == plan.batches.size();
        if (finalBatch && !needsTerminalJoin) {
            // Join every lane into the terminal fence, including work
            // that is topologically independent from the final render pass.
            for (uint32_t sourceBatch = 0; sourceBatch < batchIndex; ++sourceBatch) {
                if (!addLaneWait(sourceBatch, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT))
                    return output;
            }
        }

        for (size_t lane = 0; lane < laneWaitValues.size(); ++lane) {
            if (laneWaitValues[lane] == 0)
                continue;
            addWait(m_laneTimelines[lane], laneWaitValues[lane],
                    laneWaitStages[lane] != 0 ? laneWaitStages[lane] : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        }

        auto &signals = storage.signals;
        auto &signalSubmitValues = storage.signalValues;
        if (batchIndex == lastGraphicsBatch && sync.renderFinished != VK_NULL_HANDLE) {
            signals.push_back(sync.renderFinished);
            signalSubmitValues.push_back(0);
        }

        auto &timelineInfo = storage.timeline;
        if (usesTimeline) {
            timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            timelineInfo.waitSemaphoreValueCount = static_cast<uint32_t>(waitValues.size());
            timelineInfo.pWaitSemaphoreValues = waitValues.data();
            timelineInfo.signalSemaphoreValueCount = static_cast<uint32_t>(signalSubmitValues.size());
            timelineInfo.pSignalSemaphoreValues = signalSubmitValues.data();
        }

        auto &submitInfo = submits[batchIndex];
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.pNext = usesTimeline ? &timelineInfo : nullptr;
        submitInfo.waitSemaphoreCount = static_cast<uint32_t>(waits.size());
        submitInfo.pWaitSemaphores = waits.data();
        submitInfo.pWaitDstStageMask = waitStages.data();
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commands[batchIndex];
        submitInfo.signalSemaphoreCount = static_cast<uint32_t>(signals.size());
        submitInfo.pSignalSemaphores = signals.data();

        // The first consumer on each lane carries its external dependencies.
        if (targetLane < laneSubmitted.size())
            laneSubmitted[targetLane] = true;
        roleSubmitted[RoleIndex(batch.queue)] = true;
    }

    // Finish allocating wait metadata before the first queue accepts work.
    SubmissionWaits joinWaits;
    VkTimelineSemaphoreSubmitInfo joinTimeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    VkSubmitInfo joinSubmit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    if (needsTerminalJoin) {
        std::vector<uint64_t> joinedLaneValues(m_laneTimelines.size(), 0);
        for (size_t batchIndex = 0; batchIndex < plan.batches.size(); ++batchIndex) {
            const uint32_t lane = m_queues->GetSnapshot(plan.batches[batchIndex].queue).nativeLane;
            if (lane == joinLane || lane >= joinedLaneValues.size())
                continue;
            joinedLaneValues[lane] = (std::max)(joinedLaneValues[lane], signalValues[batchIndex]);
        }
        for (size_t lane = 0; lane < joinedLaneValues.size(); ++lane) {
            if (joinedLaneValues[lane] != 0)
                joinWaits.Add({m_laneTimelines[lane], joinedLaneValues[lane], VK_PIPELINE_STAGE_ALL_COMMANDS_BIT});
        }
        joinTimeline.waitSemaphoreValueCount = static_cast<uint32_t>(joinWaits.values.size());
        joinTimeline.pWaitSemaphoreValues = joinWaits.values.data();
        joinSubmit.pNext = &joinTimeline;
        joinSubmit.waitSemaphoreCount = joinTimeline.waitSemaphoreValueCount;
        joinSubmit.pWaitSemaphores = joinWaits.semaphores.data();
        joinSubmit.pWaitDstStageMask = joinWaits.stages.data();
    }

    const auto failSubmission = [&](VkResult result) {
        reservations.CancelRemaining();
        if (output.submittedBatchCount) {
            (void)m_queues->WaitIdleForAllQueues();
            for (const auto &accepted : frame.submittedTickets)
                m_queues->MarkCompleted(accepted);
        }
        output.result = result;
        return output;
    };
    for (size_t batchIndex = 0; batchIndex < plan.batches.size(); ++batchIndex) {
        const bool finalBatch = batchIndex + 1 == plan.batches.size();
        if (!finalBatch && plan.batches[batchIndex + 1].queue == plan.batches[batchIndex].queue)
            continue;
        const size_t groupFirst = reservations.next;
        const VkFence fence = finalBatch && !needsTerminalJoin ? sync.completionFence : VK_NULL_HANDLE;
        const VkResult result = m_queues->SubmitReserved(tickets[groupFirst], submits.data() + groupFirst,
                                                         static_cast<uint32_t>(batchIndex + 1 - groupFirst), fence);
        if (result != VK_SUCCESS)
            return failSubmission(result);
        output.submittedBatchCount = reservations.next = batchIndex + 1;
        frame.submittedTickets.insert(frame.submittedTickets.end(), tickets.begin() + groupFirst,
                                      tickets.begin() + reservations.next);
        if (finalBatch && !needsTerminalJoin && terminalLane < m_laneTimelines.size()) {
            output.completionTimeline = m_laneTimelines[terminalLane];
            output.completionTimelineValue = signalValues[batchIndex];
        }
    }
    if (needsTerminalJoin) {
        const auto joinTicket = tickets.back();
        const VkResult joinResult = m_queues->SubmitReserved(joinTicket, joinSubmit, sync.completionFence);
        if (joinResult != VK_SUCCESS)
            return failSubmission(joinResult);
        ++reservations.next;
        frame.submittedTickets.push_back(joinTicket);
        output.completionTimeline = m_laneTimelines[joinLane];
        output.completionTimelineValue = joinTicket.serial;
    }

    output.result = VK_SUCCESS;
    if (sync.cpuTimings)
        (*sync.cpuTimings)[2] =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - phaseStart).count();
    return output;
}

uint64_t VulkanSubmissionExecutor::GetRetainedUploadBytes() const noexcept
{
    uint64_t bytes = 0;
    for (const auto &frame : m_frames)
        for (const auto &role : frame.roles)
            if (role.uploads)
                bytes += role.uploads->RetainedBytes();
    return bytes;
}

VulkanCommandUploads &VulkanSubmissionExecutor::CommandUploads(uint32_t frameSlot, rhi::QueueRole role) noexcept
{
    return *m_frames[frameSlot].roles[RoleIndex(role)].uploads;
}

void VulkanSubmissionExecutor::CompleteFrame(uint32_t frameSlot) noexcept
{
    if (!m_queues || frameSlot >= m_frames.size())
        return;
    FrameState &frame = m_frames[frameSlot];
    for (const rhi::SubmissionTicket ticket : frame.submittedTickets)
        m_queues->MarkCompleted(ticket);
    if (!frame.submittedTickets.empty())
        for (auto &role : frame.roles)
            if (role.uploads)
                role.uploads->Recycle();
    frame.submittedTickets.clear();
}

} // namespace infernux::vk
