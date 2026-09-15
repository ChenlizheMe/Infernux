#pragma once

#include "RhiCommand.h"
#include "RhiQuery.h"
#include "RhiSubmission.h"
#include <functional>

namespace infernux::rhi
{

struct ComputeQueueStatistics
{
    uint64_t submissionCount = 0;
    uint64_t dispatchCount = 0;
    uint64_t uploadRequestCount = 0;
    uint64_t uploadBytes = 0;
    uint64_t readbackRequestCount = 0;
    uint64_t readbackBytes = 0;
    uint64_t stagingAllocationCount = 0;
    uint64_t hostMapCount = 0;
    uint64_t nativeBoundaryCount = 0;
    uint64_t waitCount = 0;
    double cpuSubmitMilliseconds = 0.0;
    double waitMilliseconds = 0.0;
};

/// Borrowed encoders, valid only during the synchronous recording callback.
/// Resources remain caller-owned until the returned submission completes.
class ComputeRecordingContext
{
  public:
    virtual ~ComputeRecordingContext() = default;
    [[nodiscard]] virtual ComputeCommandEncoder Compute() = 0;
    [[nodiscard]] virtual TransferCommandEncoder Transfer() = 0;
    virtual void PipelineBarrier(PipelineStage sourceStage, Access sourceAccess, PipelineStage destinationStage,
                                 Access destinationAccess) = 0;
};

/// Host-owned background compute lane. Submission/collection are serialized
/// on the engine thread, not reentrant from a recording callback. Submit records
/// immediately, then returns without waiting unless the bounded pool is full.
/// Submission order does not replace the caller's explicit memory barriers.
class ComputeQueue
{
  public:
    using Recorder = std::function<bool(ComputeRecordingContext &)>;
    virtual ~ComputeQueue() = default;
    [[nodiscard]] virtual SubmissionTicket Submit(const Recorder &record) = 0;
    /// These accept this queue's tickets, including tickets from reused slots.
    virtual void Wait(SubmissionTicket ticket) = 0;
    [[nodiscard]] virtual bool IsComplete(SubmissionTicket ticket) = 0;
    virtual void Collect() = 0;
    /// Explicit diagnostic boundary: drains this queue before changing query
    /// resources. Disabled by default; false means timestamps are unavailable.
    virtual bool SetProfilingEnabled(bool enabled) = 0;
    /// Latest collected submission, not a device wait or a host wall-clock time.
    [[nodiscard]] virtual GpuTimestampFrame GetProfile() const = 0;
    [[nodiscard]] virtual uint64_t GetPendingSubmissionCount() const noexcept = 0;

    [[nodiscard]] ComputeQueueStatistics GetStatistics() const noexcept
    {
        return m_statistics;
    }
    void ResetStatistics() noexcept
    {
        m_statistics = {};
    }
    void RecordDispatches(uint64_t count) noexcept
    {
        m_statistics.dispatchCount += count;
    }
    void RecordUpload(uint64_t bytes) noexcept
    {
        ++m_statistics.uploadRequestCount;
        m_statistics.uploadBytes += bytes;
    }
    void RecordReadback(uint64_t bytes) noexcept
    {
        ++m_statistics.readbackRequestCount;
        m_statistics.readbackBytes += bytes;
    }
    void RecordStagingAllocation() noexcept
    {
        ++m_statistics.stagingAllocationCount;
    }
    void RecordHostMap() noexcept
    {
        ++m_statistics.hostMapCount;
    }
    void RecordNativeBoundary() noexcept
    {
        ++m_statistics.nativeBoundaryCount;
    }

  protected:
    void RecordSubmission(double milliseconds) noexcept
    {
        ++m_statistics.submissionCount;
        m_statistics.cpuSubmitMilliseconds += milliseconds;
    }
    void RecordWait(double milliseconds) noexcept
    {
        ++m_statistics.waitCount;
        m_statistics.waitMilliseconds += milliseconds;
    }

  private:
    ComputeQueueStatistics m_statistics;
};

} // namespace infernux::rhi
