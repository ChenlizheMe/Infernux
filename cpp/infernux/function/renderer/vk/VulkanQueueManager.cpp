#include "VulkanQueueManager.h"

#include "VkDeviceContext.h"

#include <core/error/InxError.h>

#include <algorithm>
#include <new>
#include <stdexcept>

namespace infernux::vk
{

bool VulkanQueueManager::Initialize(const VkDeviceContext &context, uint32_t graphicsFrameSlots)
{
    Destroy();
    if (!context.IsValid() || context.GetDeviceId() == rhi::InvalidDeviceId)
        return false;

    std::lock_guard lock(m_mutex);
    m_device = context.GetDevice();
    m_deviceId = context.GetDeviceId();
    const auto &indices = context.GetQueueIndices();
    const uint32_t graphicsFamily = indices.graphicsFamily.value_or(0);
    const uint32_t computeFamily = indices.computeFamily.value_or(graphicsFamily);
    const uint32_t presentFamily = indices.presentFamily.value_or(graphicsFamily);
    const uint32_t transferFamily = indices.transferFamily.value_or(graphicsFamily);

    m_queues[RoleIndex(rhi::QueueRole::Graphics)].queue = context.GetGraphicsQueue();
    m_queues[RoleIndex(rhi::QueueRole::Graphics)].family = graphicsFamily;
    m_queues[RoleIndex(rhi::QueueRole::Compute)].queue = context.GetComputeQueue();
    m_queues[RoleIndex(rhi::QueueRole::Compute)].family = computeFamily;
    m_queues[RoleIndex(rhi::QueueRole::Transfer)].queue = context.GetTransferQueue();
    m_queues[RoleIndex(rhi::QueueRole::Transfer)].family = transferFamily;
    m_queues[RoleIndex(rhi::QueueRole::Present)].queue = context.GetPresentQueue();
    m_queues[RoleIndex(rhi::QueueRole::Present)].family = presentFamily;

    for (size_t role = 0; role < m_roleLanes.size(); ++role) {
        m_roleLanes[role] = role;
        const VkQueue queue = m_queues[role].queue;
        if (queue == VK_NULL_HANDLE)
            continue;
        for (size_t candidate = 0; candidate < role; ++candidate) {
            if (m_queues[candidate].queue == queue) {
                m_roleLanes[role] = m_roleLanes[candidate];
                break;
            }
        }
    }
    const auto failInitialization = [&] {
        for (auto &queue : m_queues)
            if (queue.timeline != VK_NULL_HANDLE)
                vkDestroySemaphore(m_device, queue.timeline, nullptr);
        for (const auto fence : m_graphicsFrameFences)
            if (fence != VK_NULL_HANDLE)
                vkDestroyFence(m_device, fence, nullptr);
        m_graphicsFrameFences.clear();
        m_graphicsFrameSlots.clear();
        m_queues = {};
        m_device = VK_NULL_HANDLE;
        m_deviceId = rhi::InvalidDeviceId;
        return false;
    };
    if (context.IsTimelineSemaphoreEnabled()) {
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        info.pNext = &type;
        for (size_t lane = 0; lane < m_queues.size(); ++lane)
            if (m_roleLanes[lane] == lane && m_queues[lane].queue != VK_NULL_HANDLE &&
                vkCreateSemaphore(m_device, &info, nullptr, &m_queues[lane].timeline) != VK_SUCCESS)
                return failInitialization();
    }
    const uint32_t frameSlotCount = graphicsFrameSlots;
    m_graphicsFrameSlots.resize(frameSlotCount);
    m_graphicsFrameFences.resize(frameSlotCount, VK_NULL_HANDLE);
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (uint32_t index = 0; index < frameSlotCount; ++index) {
        if (vkCreateFence(m_device, &fenceInfo, nullptr, &m_graphicsFrameFences[index]) != VK_SUCCESS)
            return failInitialization();
    }
    return true;
}

void VulkanQueueManager::Destroy() noexcept
{
    {
        std::lock_guard lock(m_mutex);
        if (m_device != VK_NULL_HANDLE) {
            for (const auto &queue : m_queues)
                if (queue.timeline != VK_NULL_HANDLE)
                    vkDestroySemaphore(m_device, queue.timeline, nullptr);
            for (VkFence fence : m_graphicsFrameFences) {
                if (fence != VK_NULL_HANDLE)
                    vkDestroyFence(m_device, fence, nullptr);
            }
        }
        for (auto &receipt : m_workCompletions) {
            receipt->complete = false;
            receipt->failed = true; // Destruction is not an observed successful completion.
        }
        m_workCompletions.clear();
        m_workObservation = uint64_t(-1);
        m_graphicsFrameFences.clear();
        m_deviceId = rhi::InvalidDeviceId;
        m_device = VK_NULL_HANDLE;
        m_queues = {};
        m_roleLanes = {};
        m_graphicsFrameSlots.clear();
        m_nextCompletionEpoch = 1;
        m_lastReservedCompletionEpoch = 0;
        m_completedCompletionEpoch = 0;
        m_completedOutOfOrderEpochs.clear();
    }
    m_submitCv.notify_all();
}

rhi::SubmissionTicket VulkanQueueManager::Reserve(rhi::QueueRole role, uint32_t count)
{
    std::lock_guard lock(m_mutex);
    if (m_deviceId == rhi::InvalidDeviceId || role == rhi::QueueRole::Count || count == 0)
        return {};
    auto &queue = m_queues[LaneIndex(role)];
    if (queue.queue == VK_NULL_HANDLE)
        return {};
    const auto serial = queue.nextSerial;
    queue.nextSerial += count;
    queue.lastReserved = queue.nextSerial - 1;
    return {m_deviceId, role, serial};
}

VkSemaphore VulkanQueueManager::GetDependencyTimeline(rhi::QueueRole role) const noexcept
{
    std::lock_guard lock(m_mutex);
    return role != rhi::QueueRole::Count ? m_queues[LaneIndex(role)].timeline : VK_NULL_HANDLE;
}

VulkanQueueManager::TimelinePoint VulkanQueueManager::DependencyFor(rhi::SubmissionTicket ticket,
                                                                    rhi::QueueRole consumer) const
{
    std::lock_guard lock(m_mutex);
    if (!ticket.IsValid() || ticket.device != m_deviceId || ticket.queue >= rhi::QueueRole::Count)
        throw std::invalid_argument("GPU producer belongs to another device or invalid queue");
    const auto &queue = m_queues[LaneIndex(ticket.queue)];
    const auto range = std::lower_bound(queue.accepted.begin(), queue.accepted.end(), ticket.serial,
                                        [](const auto &item, auto serial) { return item.last < serial; });
    if (range == queue.accepted.end() || ticket.serial < range->first)
        throw std::invalid_argument("GPU producer has not been accepted");
    const bool pending = ticket.serial > queue.completed;
    if (consumer < rhi::QueueRole::Count && LaneIndex(consumer) == LaneIndex(ticket.queue))
        return {};
    if (!pending && (consumer == rhi::QueueRole::Count || queue.timeline == VK_NULL_HANDLE))
        return {};
    if (queue.timeline == VK_NULL_HANDLE)
        throw std::logic_error("Pending GPU producer requires a timeline semaphore");
    return {queue.timeline, ticket.serial, pending};
}

VkResult VulkanQueueManager::SubmitReserved(rhi::SubmissionTicket ticket, const VkSubmitInfo &submitInfo,
                                            VkFence fence) noexcept
{
    return SubmitReserved(ticket, &submitInfo, 1, fence);
}

VkResult VulkanQueueManager::SubmitReserved(rhi::SubmissionTicket ticket, const VkSubmitInfo *submits, uint32_t count,
                                            VkFence fence) noexcept
{
    std::unique_lock lock(m_mutex);
    if (!ticket.IsValid() || ticket.device != m_deviceId || ticket.queue == rhi::QueueRole::Count || !submits ||
        count == 0)
        return VK_ERROR_INITIALIZATION_FAILED;
    auto &queue = m_queues[LaneIndex(ticket.queue)];
    if (queue.queue == VK_NULL_HANDLE || ticket.serial > queue.lastReserved ||
        ticket.serial < queue.nextSubmissionSerial || count - 1 > queue.lastReserved - ticket.serial)
        return VK_ERROR_INITIALIZATION_FAILED;

    m_submitCv.wait(lock, [&] { return m_device == VK_NULL_HANDLE || ticket.serial == queue.nextSubmissionSerial; });
    if (m_device == VK_NULL_HANDLE)
        return VK_ERROR_DEVICE_LOST;

    // Add the device-owned completion point to each actual submission. Keep
    // callers' binary signals, timeline values and downstream pNext nodes.
    struct SubmissionSync
    {
        VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        std::vector<VkSemaphore> signals;
        std::vector<uint64_t> values, binaryWaitValues;
    };
    std::vector<VkSubmitInfo> ownedSubmits;
    std::vector<SubmissionSync> ownedSync;
    VkResult result = VK_SUCCESS;
    try {
        if (queue.timeline != VK_NULL_HANDLE) {
            ownedSubmits.assign(submits, submits + count);
            ownedSync.resize(count);
            for (uint32_t index = 0; index < count; ++index) {
                auto &submit = ownedSubmits[index];
                auto &sync = ownedSync[index];
                if (submit.pNext) {
                    const auto *next = static_cast<const VkBaseInStructure *>(submit.pNext);
                    if (next->sType != VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO) {
                        result = VK_ERROR_FEATURE_NOT_PRESENT;
                        break;
                    }
                    sync.timeline = *static_cast<const VkTimelineSemaphoreSubmitInfo *>(submit.pNext);
                }
                if (submit.signalSemaphoreCount) {
                    sync.signals.assign(submit.pSignalSemaphores,
                                        submit.pSignalSemaphores + submit.signalSemaphoreCount);
                    sync.values.resize(submit.signalSemaphoreCount, 0);
                    if (sync.timeline.signalSemaphoreValueCount)
                        std::copy_n(sync.timeline.pSignalSemaphoreValues, submit.signalSemaphoreCount,
                                    sync.values.begin());
                }
                sync.signals.push_back(queue.timeline);
                sync.values.push_back(ticket.serial + index);
                sync.timeline.signalSemaphoreValueCount = static_cast<uint32_t>(sync.values.size());
                sync.timeline.pSignalSemaphoreValues = sync.values.data();
                // Binary waits still occupy a zero entry in a timeline submit.
                if (!sync.timeline.waitSemaphoreValueCount && submit.waitSemaphoreCount) {
                    sync.binaryWaitValues.resize(submit.waitSemaphoreCount, 0);
                    sync.timeline.waitSemaphoreValueCount = submit.waitSemaphoreCount;
                    sync.timeline.pWaitSemaphoreValues = sync.binaryWaitValues.data();
                }
                submit.pNext = &sync.timeline;
                submit.signalSemaphoreCount = static_cast<uint32_t>(sync.signals.size());
                submit.pSignalSemaphores = sync.signals.data();
            }
            submits = ownedSubmits.data();
        }
        // Publication cannot allocate after Vulkan has accepted the work.
        queue.accepted.reserve(queue.accepted.size() + 1);
    } catch (const std::bad_alloc &) {
        result = VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    if (result == VK_SUCCESS)
        result = vkQueueSubmit(queue.queue, count, submits, fence);
    queue.nextSubmissionSerial += count;
    if (result == VK_SUCCESS) {
        queue.lastSubmitted = queue.nextSubmissionSerial - 1;
        if (!queue.accepted.empty() && queue.accepted.back().last + 1 == ticket.serial)
            queue.accepted.back().last = queue.lastSubmitted;
        else
            queue.accepted.push_back({ticket.serial, queue.lastSubmitted});
    }
    lock.unlock();
    m_submitCv.notify_all();
    return result;
}

VkResult VulkanQueueManager::Present(const VkPresentInfoKHR &presentInfo) noexcept
{
    std::lock_guard lock(m_mutex);
    const auto &queue = m_queues[LaneIndex(rhi::QueueRole::Present)];
    if (m_device == VK_NULL_HANDLE || queue.queue == VK_NULL_HANDLE)
        return VK_ERROR_INITIALIZATION_FAILED;
    return vkQueuePresentKHR(queue.queue, &presentInfo);
}

VkResult VulkanQueueManager::WaitIdleForPresentation() noexcept
{
    return WaitIdleQueues(
        {rhi::QueueRole::Graphics, rhi::QueueRole::Present, rhi::QueueRole::Count, rhi::QueueRole::Count}, 2);
}

VkResult VulkanQueueManager::WaitIdleForGraphics() noexcept
{
    return WaitIdleQueues(
        {rhi::QueueRole::Graphics, rhi::QueueRole::Count, rhi::QueueRole::Count, rhi::QueueRole::Count}, 1);
}

VkResult VulkanQueueManager::WaitIdleForAllQueues() noexcept
{
    return WaitIdleQueues(
        {rhi::QueueRole::Graphics, rhi::QueueRole::Compute, rhi::QueueRole::Transfer, rhi::QueueRole::Present}, 4);
}

VkResult VulkanQueueManager::WaitIdleQueues(const std::array<rhi::QueueRole, 4> &roles, size_t roleCount) noexcept
{
    std::unique_lock lock(m_mutex);
    if (m_device == VK_NULL_HANDLE)
        return VK_ERROR_DEVICE_LOST;

    // A reservation may be recording outside the queue lock. Let it publish or
    // cancel before draining, otherwise it could submit work after WaitIdle and
    // still reference resources the caller is about to replace.
    m_submitCv.wait(lock, [&] {
        if (m_device == VK_NULL_HANDLE)
            return true;
        for (size_t index = 0; index < roleCount; ++index) {
            const auto role = roles[index];
            if (role == rhi::QueueRole::Count)
                return true;
            const auto &state = m_queues[LaneIndex(role)];
            if (state.nextSubmissionSerial <= state.lastReserved)
                return false;
        }
        return true;
    });
    if (m_device == VK_NULL_HANDLE)
        return VK_ERROR_DEVICE_LOST;

    std::array<VkQueue, 4> drained{};
    size_t drainedCount = 0;
    for (size_t index = 0; index < roleCount; ++index) {
        const auto role = roles[index];
        if (role == rhi::QueueRole::Count)
            return VK_ERROR_INITIALIZATION_FAILED;
        const VkQueue queue = m_queues[LaneIndex(role)].queue;
        if (queue == VK_NULL_HANDLE)
            return VK_ERROR_INITIALIZATION_FAILED;
        if (std::find(drained.begin(), drained.begin() + static_cast<std::ptrdiff_t>(drainedCount), queue) !=
            drained.begin() + static_cast<std::ptrdiff_t>(drainedCount)) {
            continue;
        }

        const VkResult result = vkQueueWaitIdle(queue);
        if (result != VK_SUCCESS)
            return result;
        drained[drainedCount++] = queue;
    }

    for (auto &state : m_queues) {
        if (std::find(drained.begin(), drained.begin() + static_cast<std::ptrdiff_t>(drainedCount), state.queue) !=
            drained.begin() + static_cast<std::ptrdiff_t>(drainedCount)) {
            state.completed = (std::max)(state.completed, state.lastSubmitted);
        }
    }
    return VK_SUCCESS;
}

bool VulkanQueueManager::CancelReservation(rhi::SubmissionTicket ticket) noexcept
{
    std::unique_lock lock(m_mutex);
    if (!ticket.IsValid() || ticket.device != m_deviceId || ticket.queue == rhi::QueueRole::Count)
        return false;
    auto &queue = m_queues[LaneIndex(ticket.queue)];
    if (ticket.serial > queue.lastReserved || ticket.serial < queue.nextSubmissionSerial)
        return false;
    m_submitCv.wait(lock, [&] { return m_device == VK_NULL_HANDLE || ticket.serial == queue.nextSubmissionSerial; });
    if (m_device == VK_NULL_HANDLE)
        return false;
    ++queue.nextSubmissionSerial;
    lock.unlock();
    m_submitCv.notify_all();
    return true;
}

bool VulkanQueueManager::AssociateFrameSlot(uint32_t frameSlot, rhi::SubmissionTicket ticket) noexcept
{
    std::lock_guard lock(m_mutex);
    if (!ticket.IsValid() || ticket.device != m_deviceId || ticket.queue == rhi::QueueRole::Count ||
        frameSlot >= m_graphicsFrameSlots.size() ||
        m_graphicsFrameSlots[frameSlot].completionEpoch == rhi::InvalidSubmissionSerial)
        return false;
    m_graphicsFrameSlots[frameSlot].terminalTicket = ticket;
    return true;
}

rhi::SubmissionTicket VulkanQueueManager::CompleteFrameSlot(uint32_t frameSlot) noexcept
{
    std::lock_guard lock(m_mutex);
    if (frameSlot >= m_graphicsFrameSlots.size())
        return {};
    auto &slot = m_graphicsFrameSlots[frameSlot];
    const auto ticket = slot.terminalTicket;
    if (ticket.IsValid()) {
        auto &queue = m_queues[LaneIndex(ticket.queue)];
        queue.completed = (std::max)(queue.completed, ticket.serial);
    }
    CompleteCompletionEpochLocked(slot.completionEpoch);
    slot = {};
    return ticket;
}

void VulkanQueueManager::MarkCompleted(rhi::SubmissionTicket ticket) noexcept
{
    std::lock_guard lock(m_mutex);
    if (!ticket.IsValid() || ticket.device != m_deviceId || ticket.queue == rhi::QueueRole::Count)
        return;
    auto &queue = m_queues[LaneIndex(ticket.queue)];
    queue.completed = (std::max)(queue.completed, ticket.serial);
}

bool VulkanQueueManager::ResetGraphicsFrameFence(uint32_t frameSlot) noexcept
{
    std::lock_guard lock(m_mutex);
    if (m_device == VK_NULL_HANDLE || frameSlot >= m_graphicsFrameFences.size() ||
        m_graphicsFrameFences[frameSlot] == VK_NULL_HANDLE ||
        m_graphicsFrameSlots[frameSlot].completionEpoch != rhi::InvalidSubmissionSerial)
        return false;
    if (vkResetFences(m_device, 1, &m_graphicsFrameFences[frameSlot]) != VK_SUCCESS)
        return false;
    m_graphicsFrameSlots[frameSlot].completionEpoch = ReserveCompletionEpochLocked();
    return true;
}

bool VulkanQueueManager::AbandonGraphicsFrameSlot(uint32_t frameSlot) noexcept
{
    std::lock_guard lock(m_mutex);
    if (m_device == VK_NULL_HANDLE || frameSlot >= m_graphicsFrameFences.size())
        return false;

    auto &slot = m_graphicsFrameSlots[frameSlot];
    CompleteCompletionEpochLocked(slot.completionEpoch);
    slot = {};

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VkFence replacement = VK_NULL_HANDLE;
    if (vkCreateFence(m_device, &fenceInfo, nullptr, &replacement) != VK_SUCCESS)
        return false;

    if (m_graphicsFrameFences[frameSlot] != VK_NULL_HANDLE)
        vkDestroyFence(m_device, m_graphicsFrameFences[frameSlot], nullptr);
    m_graphicsFrameFences[frameSlot] = replacement;
    return true;
}

VkFence VulkanQueueManager::GetGraphicsFrameFence(uint32_t frameSlot) const noexcept
{
    std::lock_guard lock(m_mutex);
    return frameSlot < m_graphicsFrameFences.size() ? m_graphicsFrameFences[frameSlot] : VK_NULL_HANDLE;
}

rhi::SubmissionSerial VulkanQueueManager::GetFrameCompletionEpoch(uint32_t frameSlot) const noexcept
{
    std::lock_guard lock(m_mutex);
    return frameSlot < m_graphicsFrameSlots.size() ? m_graphicsFrameSlots[frameSlot].completionEpoch
                                                   : rhi::InvalidSubmissionSerial;
}

rhi::SubmissionSerial VulkanQueueManager::GetLastReservedCompletionEpoch() const noexcept
{
    std::lock_guard lock(m_mutex);
    return m_lastReservedCompletionEpoch;
}

rhi::SubmissionSerial VulkanQueueManager::GetCompletedCompletionEpoch() const noexcept
{
    std::lock_guard lock(m_mutex);
    return m_completedCompletionEpoch;
}

bool VulkanQueueManager::IsCompletionEpochComplete(rhi::SubmissionSerial epoch) const noexcept
{
    std::lock_guard lock(m_mutex);
    if (epoch == rhi::InvalidSubmissionSerial)
        return false;
    return epoch <= m_completedCompletionEpoch ||
           m_completedOutOfOrderEpochs.find(epoch) != m_completedOutOfOrderEpochs.end();
}

rhi::SubmissionSerial VulkanQueueManager::ReserveCompletionEpoch() noexcept
{
    std::lock_guard lock(m_mutex);
    return m_deviceId != rhi::InvalidDeviceId ? ReserveCompletionEpochLocked() : rhi::InvalidSubmissionSerial;
}

void VulkanQueueManager::CompleteCompletionEpoch(rhi::SubmissionSerial epoch) noexcept
{
    std::lock_guard lock(m_mutex);
    CompleteCompletionEpochLocked(epoch);
}

rhi::SubmissionSerial VulkanQueueManager::ReserveCompletionEpochLocked() noexcept
{
    const auto epoch = m_nextCompletionEpoch++;
    m_lastReservedCompletionEpoch = epoch;
    return epoch;
}

void VulkanQueueManager::CompleteCompletionEpochLocked(rhi::SubmissionSerial epoch) noexcept
{
    if (epoch == rhi::InvalidSubmissionSerial || epoch <= m_completedCompletionEpoch)
        return;
    m_completedOutOfOrderEpochs.insert(epoch);
    while (!m_completedOutOfOrderEpochs.empty()) {
        const auto next = m_completedOutOfOrderEpochs.begin();
        if (*next != m_completedCompletionEpoch + 1)
            break;
        m_completedCompletionEpoch = *next;
        m_completedOutOfOrderEpochs.erase(next);
    }
}

void VulkanQueueManager::ReserveWorkCompletions(size_t additional)
{
    std::lock_guard lock(m_mutex);
    m_workCompletions.reserve(m_workCompletions.size() + additional);
}

void VulkanQueueManager::WatchWorkCompletion(rhi::SubmissionTicket ticket,
                                             std::shared_ptr<rhi::WorkCompletionReceipt> receipt)
{
    std::lock_guard lock(m_mutex);
    if (!receipt || !ticket.IsValid() || ticket.device != m_deviceId || ticket.queue >= rhi::QueueRole::Count)
        throw std::invalid_argument("Work receipt requires its actual device/queue ticket");
    const auto &queue = m_queues[LaneIndex(ticket.queue)];
    const auto accepted = std::lower_bound(queue.accepted.begin(), queue.accepted.end(), ticket.serial,
                                           [](const auto &range, auto serial) { return range.last < serial; });
    if (accepted == queue.accepted.end() || ticket.serial < accepted->first)
        throw std::invalid_argument("Unaccepted reservation cannot publish a work receipt");
    if (m_workCompletions.size() == m_workCompletions.capacity())
        throw std::logic_error("Work receipt metadata was not reserved before submission");
    receipt->submission = ticket;
    m_workCompletions.push_back(std::move(receipt));
}

void VulkanQueueManager::FailWorkCompletions() noexcept
{
    std::lock_guard lock(m_mutex);
    for (auto &receipt : m_workCompletions) {
        receipt->complete = false;
        receipt->failed = true;
    }
    m_workCompletions.clear();
}

void VulkanQueueManager::ObserveWorkCompletions(uint64_t opportunity)
{
    std::lock_guard lock(m_mutex);
    if (m_workObservation == opportunity)
        return;
    m_workObservation = opportunity;
    std::array<bool, static_cast<size_t>(rhi::QueueRole::Count)> queried{}, failed{};
    std::array<uint64_t, static_cast<size_t>(rhi::QueueRole::Count)> values{};
    for (auto it = m_workCompletions.begin(); it != m_workCompletions.end();) {
        auto &receipt = **it;
        const auto lane = LaneIndex(receipt.submission.queue);
        const auto &queue = m_queues[lane];
        if (!queried[lane]) {
            queried[lane] = true;
            failed[lane] = queue.timeline == VK_NULL_HANDLE ||
                           vkGetSemaphoreCounterValue(m_device, queue.timeline, &values[lane]) != VK_SUCCESS;
        }
        receipt.failed = failed[lane];
        receipt.complete = !receipt.failed && values[lane] >= receipt.submission.serial;
        if (receipt.complete || receipt.failed)
            it = m_workCompletions.erase(it);
        else
            ++it;
    }
}

VulkanQueueManager::QueueSnapshot VulkanQueueManager::GetSnapshot(rhi::QueueRole role) const noexcept
{
    std::lock_guard lock(m_mutex);
    if (role == rhi::QueueRole::Count)
        return {};
    const auto &queue = m_queues[LaneIndex(role)];
    return {queue.queue,        queue.family,        static_cast<uint32_t>(LaneIndex(role)),
            queue.lastReserved, queue.lastSubmitted, queue.completed};
}

bool VulkanQueueManager::PollFrameCompletion(uint32_t frameSlot)
{
    std::lock_guard lock(m_mutex);
    if (frameSlot >= m_graphicsFrameSlots.size())
        return false;
    const auto &slot = m_graphicsFrameSlots[frameSlot];
    if (!slot.terminalTicket.IsValid() || slot.completionEpoch <= m_completedCompletionEpoch ||
        m_completedOutOfOrderEpochs.count(slot.completionEpoch))
        return false;
    const auto status = vkGetFenceStatus(m_device, m_graphicsFrameFences[frameSlot]);
    if (status == VK_NOT_READY)
        return false;
    if (status != VK_SUCCESS)
        throw std::runtime_error("Graphics completion fence query failed");
    auto &queue = m_queues[LaneIndex(slot.terminalTicket.queue)];
    queue.completed = (std::max)(queue.completed, slot.terminalTicket.serial);
    CompleteCompletionEpochLocked(slot.completionEpoch);
    return true;
}

rhi::GpuQueuePressure VulkanQueueManager::GetFramePressure() const
{
    std::lock_guard lock(m_mutex);
    rhi::GpuQueuePressure pressure;
    pressure.frameSlots = static_cast<uint32_t>(m_graphicsFrameSlots.size());
    rhi::SubmissionSerial newest = rhi::InvalidSubmissionSerial;
    for (const auto &slot : m_graphicsFrameSlots)
        if (slot.terminalTicket.IsValid())
            newest = (std::max)(newest, slot.completionEpoch);
    for (size_t index = 0; index < m_graphicsFrameSlots.size(); ++index) {
        const auto &slot = m_graphicsFrameSlots[index];
        if (slot.completionEpoch == rhi::InvalidSubmissionSerial)
            continue;
        if (!slot.terminalTicket.IsValid()) {
            ++pressure.recordingFrames;
            continue;
        }
        const auto status = vkGetFenceStatus(m_device, m_graphicsFrameFences[index]);
        if (status == VK_SUCCESS)
            continue;
        if (status != VK_NOT_READY)
            throw std::runtime_error("Graphics frame pressure fence query failed");
        ++pressure.pendingFrames;
        pressure.olderPendingFrames += slot.completionEpoch != newest;
    }
    return pressure;
}

rhi::SubmissionSerial VulkanQueueManager::GetCompletedSerial(rhi::QueueRole role) const noexcept
{
    return GetSnapshot(role).completed;
}

} // namespace infernux::vk
