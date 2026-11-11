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
    m_job = JobSystem::Get().Schedule([state] { Execute(*state); }, JobDomain::Runtime, JobPriority::Normal,
                                      JobAffinity::WorkerOnly);
}

void CpuRangeSlice::Execute(State &state)
{
    state.worker = std::this_thread::get_id();
    // A failed range may have written a prefix; it is never replayed here.
    const auto started = std::chrono::steady_clock::now();
    int32_t result = 0;
    for (auto &operation : state.operations) {
        if (state.cancelRequested.load(std::memory_order_acquire)) {
            result = -1;
            break;
        }
        const auto operationStarted = std::chrono::steady_clock::now();
        result = operation.entry(operation.slots.data(), operation.lengths.data(), operation.count, operation.begin,
                                 operation.end);
        state.operationNanoseconds.push_back(static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - operationStarted)
                .count()));
        if (result != 0)
            break;
    }
    state.executionNanoseconds = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count());
    state.result.store(result, std::memory_order_release);
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

CpuRangeRecipe::CpuRangeRecipe(std::vector<CpuRangeValueKind> kinds, std::vector<CpuRangeOperation> operations)
    : m_kinds(std::move(kinds)), m_operations(std::move(operations))
{
    if (m_operations.empty())
        throw std::invalid_argument("CPU recipe requires at least one operation");
    for (const auto kind : m_kinds)
        if (kind > CpuRangeValueKind::Int32)
            throw std::invalid_argument("unknown CPU operand kind");
    for (const auto &operation : m_operations) {
        if (!operation.entry || operation.domainParameter >= operation.operands.size())
            throw std::invalid_argument("CPU recipe requires a compiled entry and an array domain");
        for (const auto &operand : operation.operands)
            if (operand.binding >= m_kinds.size() ||
                (operand.writable && m_kinds[operand.binding] >= CpuRangeValueKind::Float32))
                throw std::invalid_argument("CPU recipe operand does not name a matching storage slot");
        if (m_kinds[operation.operands[operation.domainParameter].binding] >= CpuRangeValueKind::Float32)
            throw std::invalid_argument("CPU recipe domain must be an array");
    }
}

std::shared_ptr<CpuRangeSlice> CpuRangeRecipe::Bind(const std::vector<CpuRangeSlot> &bindings,
                                                    const std::vector<CpuRangeSelection> &ranges) const
{
    if (bindings.size() != m_kinds.size())
        throw std::invalid_argument("CPU recipe requires one value per declared binding");
    for (size_t i = 0; i < bindings.size(); ++i) {
        const auto kind = m_kinds[i];
        if (bindings[i].scalar != (kind >= CpuRangeValueKind::Float32) ||
            bindings[i].elementBytes != (kind == CpuRangeValueKind::Int64Array ? 8 : 4))
            throw std::invalid_argument("CPU recipe binding has a different POD layout");
    }
    std::shared_ptr<CpuRangeSlice> slice;
    const auto append = [&](uint32_t index, int32_t begin, std::optional<int32_t> end) {
        const auto &operation = m_operations.at(index);
        std::vector<CpuRangeSlot> slots;
        slots.reserve(operation.operands.size());
        for (const auto &operand : operation.operands) {
            auto slot = bindings[operand.binding];
            if (operand.writable && !slot.writable)
                throw std::invalid_argument("CPU recipe output is read-only");
            slot.writable = operand.writable;
            slots.push_back(slot);
        }
        const auto &domain = slots[operation.domainParameter];
        const auto count = domain.byteSize / domain.elementBytes;
        if (count > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
            throw std::invalid_argument("CPU recipe domain exceeds the i32 range");
        const auto stop = end.value_or(static_cast<int32_t>(count));
        if (!slice)
            slice = std::make_shared<CpuRangeSlice>(operation.entry, slots, static_cast<int32_t>(count), begin, stop);
        else
            slice->Append(operation.entry, slots, static_cast<int32_t>(count), begin, stop);
    };
    if (ranges.empty()) {
        for (uint32_t i = 0; i < m_operations.size(); ++i)
            append(i, 0, std::nullopt);
    } else {
        for (const auto &range : ranges)
            append(range.operation, range.begin, range.end);
    }
    // Compiler entries validate extents for an empty range without entering the
    // numerical body. Reject the whole binding before any operation can write.
    for (auto &operation : slice->m_state->operations)
        if (operation.entry(operation.slots.data(), operation.lengths.data(), operation.count, 0, 0) != 0)
            throw std::invalid_argument("CPU recipe bindings violate the compiled resource contract");
    return slice;
}

CpuRangeExecution CpuRangeRecipe::Execute(const std::vector<CpuRangeSlot> &bindings,
                                          const std::vector<CpuRangeSelection> &ranges) const
{
    auto slice = Bind(bindings, ranges);
    auto &state = *slice->m_state;
    state.operationNanoseconds.reserve(state.operations.size());
    CpuRangeSlice::Execute(state);
    return {state.result.load(std::memory_order_acquire), state.executionNanoseconds,
            std::move(state.operationNanoseconds)};
}

} // namespace infernux
