#include "CpuRangeSlice.h"

#include <atomic>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <thread>

namespace infernux
{

struct CpuRangeSlice::State
{
    struct Operation
    {
        CpuRangeEntry entry = nullptr;
        std::vector<void *> slots;
        std::vector<int32_t> lengths;
        int32_t count = 0, begin = 0, end = 0;
    };
    std::vector<Operation> operations;
    std::vector<uint64_t> operationNanoseconds;
    std::atomic<bool> cancelRequested{false};
    // A queued cancelled callback never executes. It must not look successful.
    std::atomic<int32_t> result{-1};
    std::thread::id caller = std::this_thread::get_id();
    std::thread::id worker;
    uint64_t executionNanoseconds = 0;
};

CpuRangeSlice::CpuRangeSlice(CpuRangeEntry entry, const std::vector<CpuRangeSlot> &slots, int32_t count, int32_t begin,
                             int32_t end)
    : m_state(std::make_shared<State>())
{
    Append(entry, slots, count, begin, end);
}

void CpuRangeSlice::Append(CpuRangeEntry entry, const std::vector<CpuRangeSlot> &slots, int32_t count, int32_t begin,
                           int32_t end)
{
    if (IsSubmitted() || m_cancelled)
        throw std::logic_error("cannot append to a submitted or cancelled CPU slice");
    if (!entry || slots.empty())
        throw std::invalid_argument("CPU range requires a compiled entry and POD slots");
    if (count < 0 || begin < 0 || begin > end || end > count)
        throw std::invalid_argument("CPU slice bounds are outside the logical range");
    State::Operation operation;
    operation.entry = entry;
    operation.count = count;
    operation.begin = begin;
    operation.end = end;
    operation.slots.reserve(slots.size());
    operation.lengths.reserve(slots.size());
    for (size_t i = 0; i < slots.size(); ++i) {
        const auto &slot = slots[i];
        const auto address = reinterpret_cast<uintptr_t>(slot.data);
        if (slot.elementBytes != 4 && slot.elementBytes != 8)
            throw std::invalid_argument("CPU range slot requires 32-bit or 64-bit POD elements");
        const size_t length = slot.byteSize / slot.elementBytes;
        if ((slot.byteSize && !slot.data) || address % slot.elementBytes || slot.byteSize % slot.elementBytes ||
            length > static_cast<size_t>(std::numeric_limits<int32_t>::max()) || (slot.scalar && length != 1) ||
            slot.byteSize > std::numeric_limits<uintptr_t>::max() - address || (slot.scalar && slot.writable))
            throw std::invalid_argument("CPU range slot requires aligned contiguous POD storage");
        if (slot.writable && slot.byteSize) {
            for (size_t j = 0; j < slots.size(); ++j) {
                const auto other = reinterpret_cast<uintptr_t>(slots[j].data);
                if (j != i && slots[j].byteSize &&
                    (other > std::numeric_limits<uintptr_t>::max() - slots[j].byteSize ||
                     (address < other + slots[j].byteSize && other < address + slot.byteSize)))
                    throw std::invalid_argument("CPU range output overlaps another slot");
            }
        }
        operation.slots.push_back(slot.data);
        operation.lengths.push_back(static_cast<int32_t>(length));
    }
    m_state->operations.push_back(std::move(operation));
}

void CpuRangeSlice::Submit()
{
    if (IsSubmitted() || m_cancelled)
        throw std::logic_error("CPU slice can only be submitted once");
    if (!JobSystem::IsAvailable() || JobSystem::Get().IsInline())
        throw std::runtime_error("CPU range execution requires the engine worker JobSystem");
    auto state = m_state;
    state->operationNanoseconds.reserve(state->operations.size());
    m_job = JobSystem::Get().Schedule(
        [state] {
            state->worker = std::this_thread::get_id();
            // The generated entry catches native numerical exceptions into status.
            // A failed range may have written a prefix; it is never replayed here.
            const auto started = std::chrono::steady_clock::now();
            int32_t result = 0;
            for (auto &operation : state->operations) {
                if (state->cancelRequested.load(std::memory_order_acquire)) {
                    result = -1;
                    break;
                }
                const auto operationStarted = std::chrono::steady_clock::now();
                result = operation.entry(operation.slots.data(), operation.lengths.data(), operation.count,
                                         operation.begin, operation.end);
                state->operationNanoseconds.push_back(
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                              std::chrono::steady_clock::now() - operationStarted)
                                              .count()));
                if (result != 0)
                    break;
            }
            state->executionNanoseconds = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started)
                    .count());
            state->result.store(result, std::memory_order_release);
        },
        JobDomain::Runtime, JobPriority::Normal, JobAffinity::WorkerOnly);
}

bool CpuRangeSlice::Cancel()
{
    if (m_cancelled || (IsSubmitted() && m_job.IsComplete()))
        return false;
    m_cancelled = true;
    m_state->cancelRequested.store(true, std::memory_order_release);
    if (IsSubmitted())
        m_job.Cancel();
    return true;
}

bool CpuRangeSlice::IsSubmitted() const noexcept
{
    return m_job.IsValid();
}
bool CpuRangeSlice::IsComplete() const noexcept
{
    return IsSubmitted() ? m_job.IsComplete() : m_cancelled;
}
bool CpuRangeSlice::IsCancelled() const noexcept
{
    return m_cancelled || m_job.IsCancellationRequested();
}

bool CpuRangeSlice::RanOnWorker() const
{
    if (!IsComplete())
        return false;
    return m_state->worker != std::thread::id{} && m_state->worker != m_state->caller;
}

std::optional<int32_t> CpuRangeSlice::GetResult() const
{
    if (!IsComplete())
        throw std::logic_error("CPU slice result is not ready");
    const int32_t result = m_state->result.load(std::memory_order_acquire);
    return result == -1 ? std::nullopt : std::optional<int32_t>(result);
}

std::optional<uint64_t> CpuRangeSlice::GetExecutionNanoseconds() const
{
    if (!IsComplete())
        throw std::logic_error("CPU slice execution time is not ready");
    if (m_state->worker == std::thread::id{})
        return std::nullopt;
    return m_state->executionNanoseconds;
}

const std::vector<uint64_t> &CpuRangeSlice::GetOperationNanoseconds() const
{
    if (!IsComplete())
        throw std::logic_error("CPU slice operation times are not ready");
    return m_state->operationNanoseconds;
}

void CpuRangeSlice::WaitForCompletion()
{
    if (IsSubmitted() && !m_job.IsComplete())
        JobSystem::Get().WaitPassiveCompletion(m_job);
}

} // namespace infernux
