#pragma once

#include "core/FoundationApi.h"

#include "AtomicFile.h"
#include "PathWriteScope.h"
#include <core/threading/JobSystem.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

namespace infernux
{

class DocumentStore;
class FrameCpuStorageClaim;
class FrameWorkOwner;
struct FrameWorkSource;

class DocumentWriteSuperseded final : public std::runtime_error
{
  public:
    using std::runtime_error::runtime_error;
};

class DocumentWriteCancelled final : public std::runtime_error
{
  public:
    using std::runtime_error::runtime_error;
};

using DocumentFileState = AtomicFileState;

struct DocumentWriteOptions
{
    FileWriteMode mode = FileWriteMode::Replace;
    std::optional<DocumentFileState> expectedFileState;
    // Explicit identity of one editor/document persistence chain.  A path is
    // only a lookup key; it must never make independent writers share CAS.
    std::string commitChainToken;
    PathWriteScope writeScope;
    // The producer already reserved this immutable output. The queue borrows
    // that claim rather than admitting or accounting the same bytes again.
    std::shared_ptr<FrameCpuStorageClaim> preclaimedStorage;
};

/// The producer's allocation and capacity claim move into the path queue.
struct DocumentWriteBuffer
{
    std::shared_ptr<FrameCpuStorageClaim> storage;
    std::unique_ptr<char[]> data;
    size_t size = 0;
};

struct DocumentPathMetrics
{
    uint64_t latestSubmittedGeneration = 0;
    uint64_t latestSucceededGeneration = 0;
    uint64_t latestFailedGeneration = 0;
    uint64_t pendingGeneration = 0;
    uint64_t activeGeneration = 0;
};

class DocumentWriteTicket final
{
  public:
    [[nodiscard]] const std::string &GetPath() const noexcept
    {
        return m_path;
    }
    [[nodiscard]] uint64_t GetGeneration() const noexcept
    {
        return m_generation;
    }

    [[nodiscard]] INFERNUX_FOUNDATION_API bool IsComplete() const;
    [[nodiscard]] INFERNUX_FOUNDATION_API std::string GetStatusName() const;
    [[nodiscard]] INFERNUX_FOUNDATION_API std::string GetError() const;

    INFERNUX_FOUNDATION_API void Wait() const;
    INFERNUX_FOUNDATION_API bool WaitFor(std::chrono::milliseconds timeout) const;
    [[nodiscard]] INFERNUX_FOUNDATION_API std::optional<DocumentFileState> GetCommittedFileState() const;
    [[nodiscard]] INFERNUX_FOUNDATION_API std::optional<FilePublication> GetPublication() const;

  private:
    friend class DocumentStore;

    enum class Status : uint8_t
    {
        Pending,
        Succeeded,
        Superseded,
        Cancelled,
        Failed,
    };

    DocumentWriteTicket(std::string path, uint64_t generation, DocumentStore *runtimeStore)
        : m_path(std::move(path)), m_generation(generation), m_runtimeStore(runtimeStore)
    {
    }

    void Expedite() const;
    void Complete(Status status, std::string error = {}, std::optional<AtomicWriteReceipt> receipt = std::nullopt,
                  std::exception_ptr failure = {});
    void ThrowIfFailed() const;

    std::string m_path;
    uint64_t m_generation = 0;
    DocumentStore *m_runtimeStore; // Only the process-owned store attaches to a frame owner.
    mutable std::mutex m_mutex;
    JobCompletionEvent m_completion{JobDomain::IO};
    Status m_status = Status::Pending;
    std::string m_error;
    std::exception_ptr m_failure;
    std::optional<AtomicWriteReceipt> m_receipt;
};

class DocumentStore final
{
  public:
    using Writer =
        std::function<AtomicWriteReceipt(const std::string &, std::string_view, const DocumentWriteOptions &)>;

    INFERNUX_FOUNDATION_API explicit DocumentStore(Writer writer = {});
    INFERNUX_FOUNDATION_API ~DocumentStore();

    DocumentStore(const DocumentStore &) = delete;
    DocumentStore &operator=(const DocumentStore &) = delete;

    INFERNUX_FOUNDATION_API static DocumentStore &Instance();
    INFERNUX_FOUNDATION_API static FrameWorkSource BeginFrameWork(FrameWorkOwner &owner);
    /// Drain accepted runtime writes before releasing the frame owner.
    INFERNUX_FOUNDATION_API static void EndFrameWork(FrameWorkOwner &owner);

    /// Shared workers execute runtime requests. Standalone/inline callers drive
    /// the same path queue synchronously, without owning another executor.
    INFERNUX_FOUNDATION_API std::shared_ptr<DocumentWriteTicket> Submit(const std::string &path, std::string content,
                                                                        DocumentWriteOptions options = {});
    INFERNUX_FOUNDATION_API std::shared_ptr<DocumentWriteTicket>
    SubmitBuffer(const std::string &path, DocumentWriteBuffer content, DocumentWriteOptions options = {});
    std::shared_ptr<DocumentWriteTicket> SubmitStaged(const std::string &path, std::string content,
                                                      const PathWriteScope &parent,
                                                      std::shared_ptr<FrameCpuStorageClaim> storage = {});
    /// Enqueue the journal's already-flushed file for the same path publication.
    /// The transaction persists the destination directory after the group.
    std::shared_ptr<DocumentWriteTicket> SubmitPrepared(const std::string &path, AtomicPreparedFile prepared,
                                                        DocumentWriteOptions options);
    /// The caller keeps the input immutable until the write returns.
    INFERNUX_FOUNDATION_API uint64_t WriteAndWait(const std::string &path, std::string_view content,
                                                  DocumentWriteOptions options = {});
    INFERNUX_FOUNDATION_API bool Cancel(const std::shared_ptr<DocumentWriteTicket> &ticket);
    [[nodiscard]] INFERNUX_FOUNDATION_API DocumentPathMetrics GetMetrics(const std::string &path) const;
    [[nodiscard]] INFERNUX_FOUNDATION_API DocumentFileState CaptureFileState(const std::string &path) const;
    /// True when no document write is queued or executing.  Editor-facing
    /// transactions use this to remain responsive while IO workers drain.
    [[nodiscard]] INFERNUX_FOUNDATION_API bool IsIdle() const;
    INFERNUX_FOUNDATION_API void Flush();
    INFERNUX_FOUNDATION_API void Flush(const std::string &path);
    INFERNUX_FOUNDATION_API void Shutdown();

  private:
    friend class DocumentWriteTicket;

    enum class State : uint8_t
    {
        Stopped,
        Running,
        Closing,
    };

    using Content = std::variant<std::string, std::string_view, DocumentWriteBuffer, AtomicPreparedFile>;

    struct Request
    {
        std::string key;
        std::string path;
        std::shared_ptr<FrameCpuStorageClaim> storage;
        Content content;
        DocumentWriteOptions options;
        std::shared_ptr<DocumentWriteTicket> ticket;
        PathWriteScope writeScope;
        uint64_t sequence = 0;
        bool cold = false;
        bool blocking = false;
        bool staged = false;
        std::unique_ptr<AtomicFileWriter> writer{};

        [[nodiscard]] std::string_view Bytes() const noexcept
        {
            if (const auto *owned = std::get_if<std::string>(&content))
                return *owned;
            if (const auto *borrowed = std::get_if<std::string_view>(&content))
                return *borrowed;
            if (const auto *buffer = std::get_if<DocumentWriteBuffer>(&content))
                return {buffer->data.get(), buffer->size};
            return {};
        }
    };
    static_assert(std::is_nothrow_move_constructible_v<Request> && std::is_nothrow_move_assignable_v<Request>);

    struct PathQueue
    {
        JobCompletionEvent idle{JobDomain::IO};
        std::optional<Request> active;
        std::optional<Request> pending;
        bool queued = false;
        bool flushing = false;
        bool keepHistory = false;
    };

    struct WriteStep
    {
        enum class Range
        {
            Output,
            ExpectedInput
        };

        size_t bytes = 0;
        uint64_t prediction = 0;
        bool publish = false;
        Range range = Range::Output;
        uint64_t actual = 0;
        uint64_t rangeNanoseconds = 0;
    };

    struct FrameWrite
    {
        JobHandle job;
        std::shared_ptr<WriteStep> step;
        uint64_t frame = 0;
        uint64_t grant = 0;
    };

    struct CommittedChainState
    {
        std::string commitChainToken;
        DocumentFileState fileState;
    };

    struct PathHistory
    {
        uint64_t submitted = 0;
        uint64_t succeeded = 0;
        uint64_t failed = 0;
        std::optional<CommittedChainState> committed;
    };

    static std::string ResolvePath(const std::string &path);
    void SubmitContent(const std::string &path, Content content, DocumentWriteOptions options,
                       std::shared_ptr<DocumentWriteTicket> &accepted, bool staged = false);
    void RetireBorrowedWrite(const std::shared_ptr<DocumentWriteTicket> &ticket);
    void Expedite(const DocumentWriteTicket &ticket);
    void Drain(FrameWorkOwner *frameOwner);
    [[nodiscard]] Request *NextRequestLocked(const std::string &key);
    [[nodiscard]] const Request *FrameCandidateLocked() const;
    [[nodiscard]] std::optional<DocumentFileState> ExpectedStateLocked(const Request &request) const;
    [[nodiscard]] WriteStep PlanFrameWriteLocked(const FrameWorkOwner &owner, const Request &request) const;
    void ObserveFrameWriteLocked(FrameWorkOwner &owner);
    bool ScheduleReadyPathLocked(const std::string &key);
    JobHandle SchedulePathLocked(const std::string &key, const std::shared_ptr<WriteStep> &step = {});
    void RunPath(const std::string &key, uint64_t scheduledGeneration = 0, const std::shared_ptr<WriteStep> &step = {});
    void CompleteWriteLocked(const std::string &key, std::optional<AtomicWriteReceipt> receipt,
                             std::exception_ptr failure = {});
    void FinishPathLocked(const std::string &key);
    [[nodiscard]] bool IsIdleLocked(const std::string *key = nullptr) const;

    Writer m_writer;
    mutable std::mutex m_mutex;
    std::condition_variable m_condition;
    State m_state = State::Stopped;
    std::vector<JobHandle> m_jobs;
    JobCompletionEvent m_idle{JobDomain::IO};
    bool m_flushing = false;
    FrameWorkOwner *m_frameOwner = nullptr;
    FrameWrite m_frameWrite;
    double m_writeBytesPerNanosecond = 0;
    double m_checkBytesPerNanosecond = 0;
    uint64_t m_publishPrediction = 0;
    uint64_t m_nextSequence = 1;
    size_t m_activeCount = 0;
    std::unordered_map<std::string, PathQueue> m_paths;
    std::unordered_map<std::string, PathHistory> m_history;
};

} // namespace infernux
