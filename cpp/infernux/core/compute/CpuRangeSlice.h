#pragma once

#include "core/FoundationApi.h"

#include <core/threading/JobSystem.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace infernux
{

// The compiled C wrapper is cdecl (the normal native convention on x64).
#if defined(_WIN32)
using CpuRangeEntry = int32_t(__cdecl *)(void **, const int32_t *, int32_t, int32_t, int32_t);
#else
using CpuRangeEntry = int32_t (*)(void **, const int32_t *, int32_t, int32_t, int32_t);
#endif

struct CpuRangeSlot
{
    void *data = nullptr;
    size_t byteSize = 0;
    bool scalar = false;
    bool writable = false;
    size_t elementBytes = sizeof(uint32_t);
};

// Borrowed POD only. The Engine owner pins code/storage before Submit and must
// retain them until IsComplete, including after cancellation. No Python owners
// or scene/GPU services may be captured by the worker's state.
class CpuRangeSlice
{
  public:
    INFERNUX_FOUNDATION_API CpuRangeSlice(CpuRangeEntry entry, const std::vector<CpuRangeSlot> &slots, int32_t count,
                                          int32_t begin, int32_t end);
    // Append before Submit; dependencies are ordered calls in this one job.
    INFERNUX_FOUNDATION_API void Append(CpuRangeEntry entry, const std::vector<CpuRangeSlot> &slots, int32_t count,
                                        int32_t begin, int32_t end);
    INFERNUX_FOUNDATION_API void Submit();
    INFERNUX_FOUNDATION_API bool Cancel();
    [[nodiscard]] INFERNUX_FOUNDATION_API bool IsSubmitted() const noexcept;
    [[nodiscard]] INFERNUX_FOUNDATION_API bool IsComplete() const noexcept;
    [[nodiscard]] INFERNUX_FOUNDATION_API bool IsCancelled() const noexcept;
    [[nodiscard]] INFERNUX_FOUNDATION_API bool RanOnWorker() const;
    [[nodiscard]] INFERNUX_FOUNDATION_API std::optional<int32_t> GetResult() const;
    // Actual entry execution only, excluding queueing and owner observation.
    // A cancelled job that never ran has no execution duration.
    [[nodiscard]] INFERNUX_FOUNDATION_API std::optional<uint64_t> GetExecutionNanoseconds() const;
    // Exact durations for the calls that actually ran, including a failed call.
    [[nodiscard]] INFERNUX_FOUNDATION_API const std::vector<uint64_t> &GetOperationNanoseconds() const;
    // Shutdown only. Never executes queued work on the caller; inline pools
    // are rejected before admission. Python binding releases the GIL here.
    INFERNUX_FOUNDATION_API void WaitForCompletion();

  private:
    struct State;
    std::shared_ptr<State> m_state;
    JobHandle m_job;
    bool m_cancelled = false;
};

} // namespace infernux
