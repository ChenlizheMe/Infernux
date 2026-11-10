#pragma once

#include "core/FoundationApi.h"

#include "InxPack.h"

#include <core/threading/JobSystem.h>
#include <memory>
#include <mutex>

namespace infernux
{
class FrameCpuStorageClaim;
class FrameWorkOwner;
} // namespace infernux

namespace infernux::inxpack
{
/// One archive cursor driven by the existing IO worker domain. Resident
/// inputs are immutable spans. Their owners stay alive until accepted work
/// has retired, including when the task is destroyed while it is active.
class WriteTask
{
  public:
    INFERNUX_FOUNDATION_API WriteTask(const std::filesystem::path &path, std::vector<WriteEntry> entries,
                                      const WriteOptions &options = {}, size_t blockBytes = 64u * 1024u,
                                      std::shared_ptr<void> inputOwner = {});
    INFERNUX_FOUNDATION_API ~WriteTask();
    INFERNUX_FOUNDATION_API void Append(const uint8_t *data, size_t bytes);
    INFERNUX_FOUNDATION_API void Advance(size_t maxBytes, uint64_t targetNs = 0, bool continuous = false);
    INFERNUX_FOUNDATION_API void Publish();
    INFERNUX_FOUNDATION_API bool Cancel();
    INFERNUX_FOUNDATION_API void Wait();
    INFERNUX_FOUNDATION_API bool IsPending() const;
    INFERNUX_FOUNDATION_API bool IsComplete() const;
    INFERNUX_FOUNDATION_API std::string GetStateName() const;
    INFERNUX_FOUNDATION_API std::string GetError() const;
    INFERNUX_FOUNDATION_API std::string GetInputPath() const;
    INFERNUX_FOUNDATION_API size_t GetInputBytes() const;
    INFERNUX_FOUNDATION_API WriteProgress GetProgress() const;
    INFERNUX_FOUNDATION_API IoWork DescribeWork(size_t maxBytes, bool publish = false) const;
    INFERNUX_FOUNDATION_API IoRecipe DescribeRemainingWork() const;
    INFERNUX_FOUNDATION_API IoWork GetSubmittedWork() const;
    INFERNUX_FOUNDATION_API Manifest GetManifest() const;
    INFERNUX_FOUNDATION_API size_t GetRetainedBytes() const;
    INFERNUX_FOUNDATION_API size_t GetReservationBytes() const;
    INFERNUX_FOUNDATION_API uint64_t GetExecutionNs() const;
    INFERNUX_FOUNDATION_API uint64_t GetSubmissionCount() const;
    INFERNUX_FOUNDATION_API bool RanOnWorker() const;

  private:
    struct State;
    struct StreamStep;
    std::shared_ptr<State> m_state;
    mutable std::mutex m_mutex;
    JobHandle m_job;
    std::optional<JobCompletionEvent> m_streamCompletion;
    IoWork m_submittedWork;
    // Kept on the caller's task. The worker never acquires or drops this lease.
    std::shared_ptr<void> m_inputOwner;
    bool m_borrowedInputs = false;
    bool PendingLocked() const;
    void RequireIdle() const;
    void Dispatch(bool publish, size_t maximum, uint64_t targetNs = 0, bool handoff = false);
};

/// One read cursor on the IO worker domain. An entry fills a private buffer,
/// produces bounded windows, or materializes a file for the native loader.
class ReadTask
{
  public:
    struct ReadBuffer
    {
        ReadBuffer(size_t bytes, std::shared_ptr<FrameCpuStorageClaim> claim);
        std::shared_ptr<FrameCpuStorageClaim> claim;
        std::unique_ptr<uint8_t[]> data;
        size_t bytes;
    };

    INFERNUX_FOUNDATION_API ReadTask(const std::filesystem::path &path, size_t metadataLimit,
                                     size_t blockBytes = 64u * 1024u);
    INFERNUX_FOUNDATION_API ~ReadTask();
    INFERNUX_FOUNDATION_API void Advance(size_t maxBytes, uint64_t targetNs = 0, size_t directoryExtraBytes = 0);
    INFERNUX_FOUNDATION_API void BeginChecksum();
    INFERNUX_FOUNDATION_API bool BeginReadBuffer(const std::string &entryPath, FrameWorkOwner &owner,
                                                 FrameCpuStorageClaim *reserved = nullptr);
    INFERNUX_FOUNDATION_API void BeginReadStream(const std::string &entryPath);
    INFERNUX_FOUNDATION_API bool ReserveReadWindow(FrameWorkOwner &owner, size_t maximum);
    INFERNUX_FOUNDATION_API uint64_t ReadBufferRemainingBytes() const;
    INFERNUX_FOUNDATION_API std::shared_ptr<ReadBuffer> TakeReadBuffer();
    INFERNUX_FOUNDATION_API void BeginMaterialize(const std::string &entryPath,
                                                  const std::filesystem::path &destination,
                                                  const std::string &expectedSha256);
    INFERNUX_FOUNDATION_API std::shared_ptr<VerifiedFile> TakeMaterializedFile();
    INFERNUX_FOUNDATION_API void Close();
    INFERNUX_FOUNDATION_API bool Cancel();
    INFERNUX_FOUNDATION_API void Wait();
    INFERNUX_FOUNDATION_API bool IsPending() const;
    INFERNUX_FOUNDATION_API bool IsComplete() const;
    INFERNUX_FOUNDATION_API std::string GetStateName() const;
    INFERNUX_FOUNDATION_API std::string GetError() const;
    INFERNUX_FOUNDATION_API bool IsChecksumVerified() const;
    INFERNUX_FOUNDATION_API ReadProgress GetProgress() const;
    INFERNUX_FOUNDATION_API IoWork DescribeWork(size_t maxBytes) const;
    INFERNUX_FOUNDATION_API IoRecipe DescribeRemainingWork() const;
    INFERNUX_FOUNDATION_API IoRecipe DescribeReadWork(const std::string &entryPath, bool windowed = false) const;
    INFERNUX_FOUNDATION_API IoWork GetSubmittedWork() const;
    INFERNUX_FOUNDATION_API Manifest GetManifest() const;
    INFERNUX_FOUNDATION_API size_t GetRetainedBytes() const;
    INFERNUX_FOUNDATION_API size_t GetReservationBytes() const;
    INFERNUX_FOUNDATION_API uint64_t GetExecutionNs() const;
    INFERNUX_FOUNDATION_API uint64_t GetSubmissionCount() const;
    INFERNUX_FOUNDATION_API bool RanOnWorker() const;

  private:
    struct State;
    struct EntryStep;
    std::shared_ptr<State> m_state;
    mutable std::mutex m_mutex;
    JobHandle m_job;
    std::optional<JobCompletionEvent> m_entryCompletion;
    IoWork m_submittedWork;
    State &Idle() const;
    bool PendingLocked() const;
    void Snapshot();
    void Dispatch(size_t maxBytes, uint64_t targetNs, bool handoff = false, size_t directoryExtraBytes = 0);
};
} // namespace infernux::inxpack
