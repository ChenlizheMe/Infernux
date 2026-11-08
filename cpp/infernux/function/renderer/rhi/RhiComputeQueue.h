#pragma once

#include "RhiCommand.h"
#include "RhiQuery.h"
#include "RhiSubmission.h"
#include <functional>
#include <memory>
#include <optional>

namespace infernux::rhi
{

struct ComputeQueueStatistics
{
    uint64_t submissionCount = 0;
    uint64_t dispatchCount = 0;
    uint64_t uploadRequestCount = 0;
    uint64_t uploadBytes = 0;
    uint64_t copyRequestCount = 0;
    uint64_t copyBytes = 0;
    uint64_t readbackRequestCount = 0;
    uint64_t readbackBytes = 0;
    uint64_t stagingAllocationCount = 0;
    uint64_t hostMapCount = 0;
    uint64_t nativeBoundaryCount = 0;
    uint64_t waitCount = 0;
    double cpuSubmitMilliseconds = 0.0;
    double waitMilliseconds = 0.0;
};

/// Owner-thread facts shared by a submission and its completion observers.
/// No device, resource, lease or callback is retained here.
struct ComputeExecutionReceipt
{
    bool complete = false;
    std::optional<double> deviceNanoseconds;
};

struct ComputeSubmissionResources
{
    std::shared_ptr<void> retained;
    std::shared_ptr<ComputeExecutionReceipt> receipt;
    uint64_t uploadStagingBytes = 0;
};

/// Borrowed encoders, valid only during the synchronous recording callback.
/// Resources must remain alive until completion, either through the caller or
/// the optional submission-owned resource bundle.
class ComputeRecordingContext
{
  public:
    virtual ~ComputeRecordingContext() = default;
    [[nodiscard]] virtual ComputeCommandEncoder Compute() = 0;
    [[nodiscard]] virtual TransferCommandEncoder Transfer() = 0;
    /// Declare the actual accepted value used by recorded commands. The backend
    /// orders this submission after its producer without waiting on the CPU.
    virtual void Require(SubmissionTicket producer) = 0;
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
    using Admission = std::function<bool()>;
    virtual ~ComputeQueue() = default;
    /// Optional submission-owned resources retire with the existing queue slot.
    /// Successful completion publishes its receipt before releasing retained owners.
    [[nodiscard]] virtual SubmissionTicket Submit(const Recorder &record,
                                                  ComputeSubmissionResources resources = {}) = 0;
    /// Check admission only after a slot is available. A busy slot or denied
    /// allocation leaves the task unrecorded and available for retry.
    /// Other failures throw. The caller retains rejected work for later retry.
    [[nodiscard]] virtual std::optional<SubmissionTicket> TrySubmit(const Recorder &record,
                                                                    const ComputeSubmissionResources &resources = {},
                                                                    const Admission &admit = {}) = 0;
    /// These accept this queue's tickets, including tickets from reused slots.
    virtual void Wait(SubmissionTicket ticket) = 0;
    [[nodiscard]] virtual bool IsComplete(SubmissionTicket ticket) = 0;
    virtual void Collect() = 0;
    /// Explicit diagnostic boundary: drains this queue before changing query
    /// resources. Cold initialization enables supported timestamps; false means unavailable.
    virtual bool SetProfilingEnabled(bool enabled) = 0;
    /// Latest collected submission, not a device wait or a host wall-clock time.
    [[nodiscard]] virtual GpuTimestampFrame GetProfile() const = 0;
    [[nodiscard]] virtual uint64_t GetPendingSubmissionCount() const noexcept = 0;
    /// Retained command upload pages. Completed pages return to the device pool.
    /// Pure observation; no collection, reset, allocation or driver calls.
    [[nodiscard]] virtual uint64_t GetRetainedUploadBytes() const noexcept = 0;

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
    void RecordCopy(uint64_t bytes) noexcept
    {
        ++m_statistics.copyRequestCount;
        m_statistics.copyBytes += bytes;
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
