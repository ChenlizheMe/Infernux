#include "InxPackTask.h"
#include "InxPath.h"
#include "IoStep.h"
#include "OverlappedFile.h"

#include <atomic>
#include <chrono>
#include <core/threading/FrameWorkOwner.h>
#include <limits>

namespace infernux::inxpack
{
using detail::DispatchIo;
using detail::FinishIoStep;
using detail::IoClock;
using detail::IoElapsedNs;
using detail::IoStepState;

namespace
{
constexpr std::string_view kReadBufferOperation = "read.buffer";
constexpr std::string_view kReadDirectoryOperation = "read.directory";
constexpr std::string_view kWriteBatchOperation = "write.batch";
constexpr size_t kMaxNativeReadRange = 32u * 1024u * 1024u;

// A grant covers consecutive private cursor steps. Input, output, and
// publication remain handoff points. The first indivisible step can overrun.
template <typename Describe, typename Advance, typename Stop>
void AdvanceCursorBatch(size_t remaining, uint64_t targetNs, Describe describe, Advance advance, Stop stop)
{
    const auto started = IoClock::now();
    uint64_t previousNs = 0;
    while (remaining && !stop()) {
        const auto work = describe(std::min(remaining, detail::kMinimumFileWindow));
        if (work.operation.empty())
            break;
        const auto elapsed = IoElapsedNs(started);
        if (previousNs && (elapsed >= targetNs || previousNs > targetNs - elapsed || work.operation == "write.close"))
            break;
        const auto step = IoClock::now();
        advance((std::max)(work.bytes, size_t{1}));
        previousNs = (std::max)(IoElapsedNs(step), uint64_t{1});
        remaining -= work.bytes;
    }
}
} // namespace

struct WriteTask::State : IoStepState
{
    std::unique_ptr<Writer> writer;
    WriteProgress progress;

    void Snapshot()
    {
        progress = writer->GetProgress();
        retained.store(sizeof(State) + writer->GetRetainedBytes(), std::memory_order_relaxed);
        reservation.store(sizeof(State) + writer->GetReservationBytes(), std::memory_order_relaxed);
    }
    void Fail(const std::exception &failure)
    {
        error = failure.what();
        try {
            if (writer->GetState() != WriteState::Published)
                writer->Discard(); // Pointer remains stable for concurrent Cancel.
        } catch (const std::exception &cleanup) {
            error += std::string("; temporary cleanup failed: ") + cleanup.what();
        }
    }
    void RetireCancelled()
    {
        if (error.empty() && writer->GetState() != WriteState::Cancelled)
            writer->Advance(1);
    }

    bool Cancelled() const
    {
        return control.load(std::memory_order_acquire) & cancel;
    }

    void AdvanceBatch(size_t bytes, uint64_t targetNs)
    {
        AdvanceCursorBatch(
            bytes, targetNs, [&](size_t maximum) { return writer->DescribeWork(maximum); },
            [&](size_t maximum) { writer->Advance(maximum); }, [&] { return Cancelled(); });
    }
};

struct WriteTask::StreamStep : std::enable_shared_from_this<StreamStep>
{
    StreamStep(std::shared_ptr<State> state, size_t maximum, uint64_t targetNs)
        : dispatcher(JobSystem::Get().RetainExternalWork()), state(std::move(state)), maximum(maximum),
          targetNs(targetNs)
    {
    }

    void Queue()
    {
        dispatcher.Schedule([self = shared_from_this()] { self->Run(); }, JobDomain::IO, JobPriority::Low,
                            detail::IoJobAffinity);
    }

    void Run()
    {
        std::lock_guard lock(handoff);
        const auto started = IoClock::now();
        state->ranOnWorker.store(std::this_thread::get_id() != state->owner, std::memory_order_relaxed);
        try {
            state->AdvanceBatch(maximum, targetNs);
            if (!state->Cancelled() && state->writer->GetState() == WriteState::Working) {
                // Release the worker between slices. The private file and
                // cursor remain with the original accepted stream.
                Queue();
                state->executionNs.fetch_add(IoElapsedNs(started), std::memory_order_relaxed);
                return;
            }
        } catch (const std::exception &error) {
            state->Fail(error);
        }
        FinishIoStep(*state, started);
        completion.Complete();
    }

    JobSystem::ExternalWork dispatcher;
    std::shared_ptr<State> state;
    JobCompletionEvent completion{JobDomain::IO};
    std::mutex handoff;
    size_t maximum;
    uint64_t targetNs;
};

WriteTask::WriteTask(const std::filesystem::path &path, std::vector<WriteEntry> entries, const WriteOptions &options,
                     size_t blockBytes, std::shared_ptr<void> inputOwner)
    : m_state(std::make_shared<State>()), m_inputOwner(std::move(inputOwner))
{
    auto &jobs = JobSystem::Get();
    if (jobs.IsInline())
        throw std::runtime_error("asynchronous InxPack writing requires the existing worker pool");
    m_borrowedInputs = std::any_of(entries.begin(), entries.end(),
                                   [](const WriteEntry &entry) { return entry.residentData != nullptr; });
    m_state->writer = std::make_unique<Writer>(path, std::move(entries), options, blockBytes);
    m_state->retained = sizeof(State) + m_state->writer->GetRetainedBytes();
    m_state->reservation = sizeof(State) + m_state->writer->GetReservationBytes();
}

WriteTask::~WriteTask()
{
    if (m_borrowedInputs)
        Wait();
}

void WriteTask::RequireIdle() const
{
    if (PendingLocked())
        throw std::logic_error("InxPack IO step is still running");
    if (!m_state->error.empty())
        throw std::runtime_error(m_state->error);
}

bool WriteTask::PendingLocked() const
{
    return !m_job.IsComplete() || (m_streamCompletion && !m_streamCompletion->IsComplete());
}

void WriteTask::Dispatch(bool publish, size_t maximum, uint64_t targetNs, bool handoff)
{
    if (!handoff)
        RequireIdle();
    const auto work = handoff ? IoWork{"write.cleanup", 0, {}, 0} : m_state->writer->DescribeWork(maximum, publish);
    if (work.operation.empty())
        return;
    m_job = DispatchIo(m_state, [publish, maximum, targetNs](State &state) {
        if (publish)
            state.writer->Publish();
        else if (targetNs)
            state.AdvanceBatch(maximum, targetNs);
        else
            state.writer->Advance(maximum);
    });
    m_submittedWork = targetNs ? IoWork{kWriteBatchOperation, maximum, {}, 0} : work;
}

void WriteTask::Append(const uint8_t *data, size_t bytes)
{
    std::lock_guard lock(m_mutex);
    RequireIdle();
    m_state->writer->Append(data, bytes);
    m_state->progress = m_state->writer->GetProgress();
}
void WriteTask::Advance(size_t maxBytes, uint64_t targetNs, bool continuous)
{
    std::lock_guard lock(m_mutex);
    if (!continuous) {
        Dispatch(false, maxBytes, targetNs);
        return;
    }
    RequireIdle();
    if (m_state->writer->DescribeWork(maxBytes).operation.empty())
        return;
    auto stream = std::make_shared<StreamStep>(m_state, maxBytes, targetNs);
    m_state->control.fetch_or(State::active, std::memory_order_acq_rel);
    try {
        stream->Queue();
    } catch (...) {
        m_state->control.fetch_and(static_cast<uint8_t>(~State::active), std::memory_order_release);
        throw;
    }
    m_streamCompletion = stream->completion;
    m_state->submissions.fetch_add(1, std::memory_order_relaxed);
    m_submittedWork = {kWriteBatchOperation, maxBytes, {}, 0};
}
void WriteTask::Publish()
{
    std::lock_guard lock(m_mutex);
    Dispatch(true, 1);
}
bool WriteTask::Cancel()
{
    std::lock_guard lock(m_mutex);
    if (!PendingLocked() && !m_state->error.empty())
        return false;
    const bool accepted = m_state->writer->Cancel();
    if (accepted) {
        const auto previous = m_state->control.fetch_or(State::cancel, std::memory_order_acq_rel);
        // The preceding callback can have finished all writer access before
        // JobSystem publishes its handle. It is safe to hand off now, without
        // waiting or requiring callers to repeat cancellation to get cleanup.
        if (!(previous & State::active) && m_state->writer->GetState() != WriteState::Cancelled)
            Dispatch(false, 1, 0, true);
    }
    return accepted;
}
void WriteTask::Wait()
{
    while (true) {
        JobHandle job;
        std::optional<JobCompletionEvent> stream;
        {
            std::lock_guard lock(m_mutex);
            job = m_job;
            stream = m_streamCompletion;
        }
        if (stream && !stream->IsComplete())
            stream->WaitUntil(IoClock::time_point::max());
        if (!job.IsComplete())
            JobSystem::Get().WaitPassiveCompletion(job);
        std::lock_guard lock(m_mutex);
        if (!PendingLocked())
            return;
    }
}
bool WriteTask::IsPending() const
{
    std::lock_guard lock(m_mutex);
    return PendingLocked();
}
bool WriteTask::IsComplete() const
{
    std::lock_guard lock(m_mutex);
    if (PendingLocked())
        return false;
    return !m_state->error.empty() || m_state->writer->GetState() == WriteState::Published ||
           m_state->writer->GetState() == WriteState::Cancelled;
}
std::string WriteTask::GetStateName() const
{
    std::lock_guard lock(m_mutex);
    if (PendingLocked())
        return "running";
    if (!m_state->error.empty())
        return "failed";
    switch (m_state->writer->GetState()) {
    case WriteState::Working:
        return "working";
    case WriteState::NeedsInput:
        return "needs_input";
    case WriteState::ReadyToPublish:
        return "ready";
    case WriteState::Published:
        return "published";
    case WriteState::Cancelled:
        return "cancelled";
    }
    throw std::logic_error("unknown InxPack write state");
}
std::string WriteTask::GetError() const
{
    std::lock_guard lock(m_mutex);
    return PendingLocked() ? std::string{} : m_state->error;
}
std::string WriteTask::GetInputPath() const
{
    std::lock_guard lock(m_mutex);
    RequireIdle();
    return m_state->writer->GetInputPath();
}
size_t WriteTask::GetInputBytes() const
{
    std::lock_guard lock(m_mutex);
    RequireIdle();
    return m_state->writer->GetInputBytes();
}
WriteProgress WriteTask::GetProgress() const
{
    std::lock_guard lock(m_mutex);
    if (PendingLocked())
        throw std::logic_error("InxPack IO step is still running");
    return m_state->progress;
}
IoWork WriteTask::DescribeWork(size_t maxBytes, bool publish) const
{
    std::lock_guard lock(m_mutex);
    RequireIdle();
    return m_state->writer->DescribeWork(maxBytes, publish);
}
IoWork WriteTask::GetSubmittedWork() const
{
    std::lock_guard lock(m_mutex);
    return m_submittedWork;
}
IoRecipe WriteTask::DescribeRemainingWork() const
{
    std::lock_guard lock(m_mutex);
    RequireIdle();
    return m_state->writer->DescribeRemainingWork();
}
Manifest WriteTask::GetManifest() const
{
    std::lock_guard lock(m_mutex);
    RequireIdle();
    return m_state->writer->GetManifest();
}
size_t WriteTask::GetRetainedBytes() const
{
    return sizeof(*this) + m_state->retained.load(std::memory_order_relaxed);
}
size_t WriteTask::GetReservationBytes() const
{
    return sizeof(*this) + m_state->reservation.load(std::memory_order_relaxed);
}
uint64_t WriteTask::GetExecutionNs() const
{
    return m_state->executionNs.load(std::memory_order_relaxed);
}
uint64_t WriteTask::GetSubmissionCount() const
{
    return m_state->submissions.load(std::memory_order_relaxed);
}
bool WriteTask::RanOnWorker() const
{
    return m_state->ranOnWorker.load(std::memory_order_relaxed);
}

namespace
{
uint64_t BufferedProfile(bool compressed, bool windowed)
{
    return (windowed ? 2 : 0) + compressed;
}

void MarkBufferedRecipe(IoRecipe &recipe, bool windowed)
{
    for (auto &[spans, repeats] : recipe)
        for (auto &span : spans)
            if (span.kind == "codec") {
                span.parameter = BufferedProfile(span.operation == "read.zstd", windowed);
                span.operation = kReadBufferOperation;
            }
}
} // namespace

ReadTask::ReadBuffer::ReadBuffer(size_t bytes, std::shared_ptr<FrameCpuStorageClaim> claim)
    : claim(std::move(claim)), data(new uint8_t[bytes]), bytes(bytes)
{
}

struct ReadTask::State : IoStepState
{
    State(const std::filesystem::path &path, size_t metadataLimit, size_t blockBytes)
        : cursor(path, metadataLimit, blockBytes)
    {
    }
    ReadCursor cursor;
    std::shared_ptr<ReadBuffer> output;
    size_t outputOffset = 0;
    uint64_t rawLeft = 0;
    uint64_t storedLeft = 0;
    uint64_t bufferedBytes = 0;
    uint64_t bufferWorkBytes = 0;
    uint64_t overlappedReads = 0;
    uint64_t pendingOverlappedReads = 0;
    uint64_t fileCompletionNs = 0;
    uint64_t codec = 0;
    bool windowed = false;
#ifdef INX_PLATFORM_WINDOWS
    std::shared_ptr<OverlappedFile> rangeReader;
    std::shared_ptr<OverlappedFileRequest> pendingRange;
#endif

    void BeginEntry(const std::string &entryPath, const IoSpan &span, std::shared_ptr<ReadBuffer> first)
    {
        if (output)
            throw std::logic_error("take the completed InxPack read buffer before opening another entry");
        cursor.BeginRead(entryPath);
        windowed = !first;
        output = std::move(first);
        outputOffset = 0;
        rawLeft = span.outputBytes.value_or(0);
        storedLeft = span.inputBytes.value_or(0);
        codec = BufferedProfile(span.operation == "read.zstd", windowed);
    }

    bool WindowReady() const
    {
        return output && outputOffset == output->bytes &&
               (rawLeft > output->bytes || cursor.GetState() == ReadState::Ready);
    }

    bool NeedsWindow(const IoWork &work) const
    {
        return !output && rawLeft && cursor.GetState() == ReadState::Reading && work.operation != "read.zstd_header";
    }

    IoWork DescribeWork(size_t maxBytes) const
    {
        const auto work = cursor.DescribeWork(maxBytes);
        if (WindowReady() || NeedsWindow(work))
            return {};
        if (!output || work.operation.empty() || work.operation == "read.zstd_header" ||
            work.operation == "read.zstd_allocate_decode" || work.operation == "read.entry_end")
            return work;
        const auto remaining = std::max<uint64_t>(storedLeft, output->bytes - outputOffset);
        return IoByteWork(kReadBufferOperation, std::max<uint64_t>(1, remaining), maxBytes, codec);
    }

    void AdvanceCursor(size_t maxBytes)
    {
        cursor.Advance(maxBytes);
#ifdef INX_PLATFORM_WINDOWS
        // Pin the same file revision for async ranges before its pathname can
        // be replaced while the caller is using the parsed archive directory.
        if (!rangeReader && cursor.GetState() == ReadState::Opening)
            rangeReader = std::make_shared<OverlappedFile>(cursor.OpenedFile());
#endif
    }

    struct Destination
    {
        uint8_t *data;
        size_t capacity, maximum;
    };

    Destination GetDestination(size_t maxBytes) const
    {
        const auto destination = output ? output->data.get() + outputOffset : nullptr;
        const auto capacity = output ? output->bytes - outputOffset : 0;
        if (output && rawLeft > output->bytes)
            maxBytes = (std::min)(maxBytes, capacity);
        return {destination, capacity, maxBytes};
    }

    size_t Commit(Destination destination, size_t readBytes)
    {
        const auto produced =
            cursor.FinishReadInto(destination.data, destination.capacity, destination.maximum, readBytes);
        storedLeft -= readBytes;
        outputOffset += produced;
        bufferedBytes += produced;
        return std::max(readBytes, produced);
    }

    void Snapshot()
    {
        const auto storage = sizeof(State) + (output ? output->bytes : 0);
        size_t readerBytes = 0, readerReservation = 0;
#ifdef INX_PLATFORM_WINDOWS
        readerBytes = rangeReader ? sizeof(OverlappedFile) : 0;
        readerReservation = (rangeReader || cursor.GetState() == ReadState::Opening) ? sizeof(OverlappedFile) : 0;
#endif
        retained.store(storage + readerBytes + (output ? sizeof(ReadBuffer) : 0) + cursor.GetRetainedBytes(),
                       std::memory_order_relaxed);
        reservation.store(storage + readerReservation + sizeof(ReadBuffer) + cursor.GetReservationBytes(),
                          std::memory_order_relaxed);
    }
    void Fail(const std::exception &failure)
    {
        error = failure.what();
        RetireCancelled();
    }
    void RetireCancelled()
    {
#ifdef INX_PLATFORM_WINDOWS
        rangeReader.reset();
#endif
        cursor.Close();
        output.reset();
        rawLeft = 0;
    }
};

// One host grant can span several file ranges. Waiting on the OS owns the
// cursor and buffers, with no engine worker held until the next continuation.
struct ReadTask::EntryStep : std::enable_shared_from_this<EntryStep>
{
    EntryStep(std::shared_ptr<State> state, size_t bytes, uint64_t target, bool batch)
        : dispatcher(JobSystem::Get().RetainExternalWork()), state(std::move(state)), remaining(bytes),
          targetNs(target), batched(batch)
    {
    }

    JobSystem::ExternalWork dispatcher;
    std::shared_ptr<State> state;
    JobCompletionEvent completion{JobDomain::IO};
    std::mutex handoff;
    size_t remaining;
    uint64_t targetNs, spentNs = 0;
    bool batched;

    void Queue(std::optional<FileRangeCompletion> receipt = {})
    {
        dispatcher.Schedule([self = shared_from_this(), receipt] { self->Run(receipt); }, JobDomain::IO,
                            JobPriority::Low, detail::IoJobAffinity);
    }

    bool Cancelled() const
    {
        return state->control.load(std::memory_order_acquire) & State::cancel;
    }

    State::Destination Destination() const
    {
        size_t limit = kMaxNativeReadRange;
#ifndef INX_PLATFORM_WINDOWS
        // An explicit time grant measures blocking pread work in the IO worker.
        if (targetNs)
            limit = detail::kMinimumFileWindow;
#endif
        return state->GetDestination((std::min)(remaining, limit));
    }

    void RecordYield(IoClock::time_point started)
    {
        const auto elapsed = IoElapsedNs(started);
        spentNs += elapsed;
        state->executionNs.fetch_add(elapsed, std::memory_order_relaxed);
    }

    bool Commit(State::Destination destination, size_t readBytes, IoClock::time_point started,
                IoClock::time_point &previous)
    {
        const auto used = state->Commit(destination, readBytes);
        if (!batched)
            return false;
        state->bufferWorkBytes += used;
        remaining -= used;
        if (!used || !remaining || state->cursor.GetState() != ReadState::Reading || state->WindowReady())
            return false;
        if (!targetNs)
            return true;
        const auto elapsed = spentNs + IoElapsedNs(started);
        const auto lastStep = IoElapsedNs(previous);
        previous = IoClock::now();
        // The next indivisible codec step must fit the remaining worker time.
        return elapsed < targetNs && lastStep <= targetNs - elapsed;
    }

    void ObserveFileCompletion(FileRangeCompletion receipt)
    {
#ifdef INX_PLATFORM_WINDOWS
        std::atomic_store(&state->pendingRange, std::shared_ptr<OverlappedFileRequest>{});
#endif
        state->fileCompletionNs += receipt.completionNs;
    }

    void End(IoClock::time_point started)
    {
        FinishIoStep(*state, started);
        completion.Complete();
    }

#ifdef INX_PLATFORM_WINDOWS
    void Resume(FileRangeCompletion receipt) noexcept
    {
        try {
            Queue(receipt);
        } catch (const std::exception &error) {
            // Pool shutdown may close admission while the OS still owns this
            // range. Its completion can retire the private cursor directly.
            std::lock_guard lock(handoff);
            const auto started = IoClock::now();
            ObserveFileCompletion(receipt);
            if (!Cancelled())
                state->Fail(error);
            End(started);
        }
    }

    void Submit(EntryReadRange range)
    {
        const auto submission = state->rangeReader->Read(
            range.offset, range.destination, range.bytes,
            [self = shared_from_this()](FileRangeCompletion receipt) { self->Resume(receipt); });
        ++state->overlappedReads;
        state->pendingOverlappedReads += submission.pending;
        std::atomic_store(&state->pendingRange, submission.request);
        if (Cancelled())
            submission.request->Cancel();
    }
#endif

    void Run(std::optional<FileRangeCompletion> receipt)
    {
        std::lock_guard lock(handoff);
        const auto started = IoClock::now();
        auto previous = started;
        state->ranOnWorker.store(std::this_thread::get_id() != state->owner, std::memory_order_relaxed);
        try {
            bool more = !Cancelled();
            if (receipt) {
                ObserveFileCompletion(*receipt);
                if (more) {
                    if (receipt->endOfFile)
                        throw std::runtime_error("File input ended before the requested range");
                    if (receipt->error)
                        throw std::system_error(receipt->error, "Asynchronous file input failed");
                    more = Commit(Destination(), receipt->bytes, started, previous);
                }
            }
            while (more && !Cancelled()) {
                const auto destination = Destination();
                const auto range =
                    state->cursor.PrepareReadInto(destination.data, destination.capacity, destination.maximum);
                if (range.bytes) {
#ifdef INX_PLATFORM_WINDOWS
                    Submit(range);
                    RecordYield(started);
                    return;
#else
                    state->cursor.OpenedFile().ReadAt(range.offset, range.destination, range.bytes);
#endif
                }
                more = Commit(destination, range.bytes, started, previous);
#ifndef INX_PLATFORM_WINDOWS
                if (more && !targetNs && !Cancelled()) {
                    // Keep the same grant and output, but return the worker
                    // to the shared queue before the next file range.
                    Queue();
                    RecordYield(started);
                    return;
                }
#endif
            }
        } catch (const std::exception &error) {
            state->Fail(error);
        }
        End(started);
    }
};

ReadTask::ReadTask(const std::filesystem::path &path, size_t metadataLimit, size_t blockBytes)
    : m_state(std::make_shared<State>(path, metadataLimit, blockBytes))
{
    if (JobSystem::Get().IsInline())
        throw std::runtime_error("asynchronous InxPack reading requires the existing worker pool");
    Snapshot();
}
ReadTask::~ReadTask()
{
    // During normal engine life, dropping an idle sink hands its private-file
    // cleanup to the same IO workers. After pool shutdown this is cold RAII.
    try {
        if (JobSystem::IsAvailable() && !JobSystem::Get().IsInline())
            Cancel();
    } catch (...) {
    } // Noexcept destruction also covers shutdown or failed job allocation.
}
ReadTask::State &ReadTask::Idle() const
{
    if (PendingLocked())
        throw std::logic_error("InxPack IO step is still running");
    if (!m_state->error.empty())
        throw std::runtime_error(m_state->error);
    if (m_state->control.load(std::memory_order_acquire) & State::cancel)
        throw std::logic_error("InxPack read was cancelled");
    return *m_state;
}
bool ReadTask::PendingLocked() const
{
    return !m_job.IsComplete() || (m_entryCompletion && !m_entryCompletion->IsComplete());
}
void ReadTask::Snapshot()
{
    m_state->Snapshot();
}
void ReadTask::Dispatch(size_t maxBytes, uint64_t targetNs, bool handoff, size_t directoryExtraBytes)
{
    if (!handoff)
        Idle();
    const auto work = handoff ? IoWork{"read.cleanup", 0, {}, 0} : m_state->DescribeWork(maxBytes);
    if (work.operation.empty())
        return;
    const bool directory = targetNs && m_state->cursor.GetState() == ReadState::Opening;
    if (!handoff && m_state->cursor.GetState() == ReadState::Reading) {
        const bool batch = work.operation == kReadBufferOperation;
        auto step = std::make_shared<EntryStep>(m_state, batch ? work.bytes : maxBytes, targetNs, batch);
        m_state->control.fetch_or(State::active, std::memory_order_acq_rel);
        try {
            step->Queue();
        } catch (...) {
            m_state->control.fetch_and(static_cast<uint8_t>(~State::active), std::memory_order_release);
            throw;
        }
        m_entryCompletion = step->completion;
        m_state->submissions.fetch_add(1, std::memory_order_relaxed);
    } else if (directory) {
        const auto previous = m_state->reservation.load(std::memory_order_relaxed);
        if (directoryExtraBytes > (std::numeric_limits<size_t>::max)() - previous)
            throw std::length_error("InxPack directory reservation exceeds the address space");
        const auto limit = m_state->cursor.GetReservationBytes() + directoryExtraBytes;
        m_state->reservation.store(previous + directoryExtraBytes, std::memory_order_relaxed);
        try {
            m_job = DispatchIo(m_state, [maxBytes, targetNs, limit](State &state) {
                AdvanceCursorBatch(
                    maxBytes, targetNs, [&](size_t bytes) { return state.DescribeWork(bytes); },
                    [&](size_t bytes) { state.AdvanceCursor(bytes); },
                    [&] {
                        return (state.control.load(std::memory_order_acquire) & State::cancel) ||
                               state.cursor.GetReservationBytes() > limit;
                    });
            });
        } catch (...) {
            m_state->reservation.store(previous, std::memory_order_relaxed);
            throw;
        }
    } else {
        m_job = DispatchIo(m_state, [maxBytes](State &state) {
            if (!(state.control.load(std::memory_order_acquire) & State::cancel))
                state.AdvanceCursor(maxBytes);
        });
    }
    m_submittedWork = directory ? IoWork{kReadDirectoryOperation, maxBytes, {}, 0} : work;
}
void ReadTask::Advance(size_t maxBytes, uint64_t targetNs, size_t directoryExtraBytes)
{
    std::lock_guard lock(m_mutex);
    if (!maxBytes)
        throw std::invalid_argument("InxPack read step requires a positive byte budget");
    Dispatch(maxBytes, targetNs, false, directoryExtraBytes);
}
void ReadTask::BeginChecksum()
{
    std::lock_guard lock(m_mutex);
    Idle().cursor.BeginChecksum();
    Snapshot();
}
bool ReadTask::BeginReadBuffer(const std::string &entryPath, FrameWorkOwner &owner, FrameCpuStorageClaim *reserved)
{
    std::lock_guard lock(m_mutex);
    auto &s = Idle();
    if (s.output)
        throw std::logic_error("take the completed InxPack read buffer before opening another entry");
    const auto recipe = s.cursor.DescribeReadWork(entryPath);
    const auto &span = recipe.front().first.front();
    const auto bytes = span.outputBytes.value_or(0);
    if (bytes > std::numeric_limits<size_t>::max())
        throw std::length_error("InxPack read buffer exceeds the address space");
    if (!reserved && bytes > owner.CpuStorageLimit())
        throw std::length_error("InxPack read buffer exceeds the CPU storage limit");
    auto claim = reserved ? owner.SplitCpuStorage(*reserved, bytes) : owner.TryReserveCpuStorage(bytes);
    if (!claim)
        return false;
    s.BeginEntry(entryPath, span, std::make_shared<ReadBuffer>(static_cast<size_t>(bytes), std::move(claim)));
    Snapshot();
    return true;
}
void ReadTask::BeginReadStream(const std::string &entryPath)
{
    std::lock_guard lock(m_mutex);
    auto &s = Idle();
    const auto recipe = s.cursor.DescribeReadWork(entryPath);
    s.BeginEntry(entryPath, recipe.front().first.front(), nullptr);
    Snapshot();
}
bool ReadTask::ReserveReadWindow(FrameWorkOwner &owner, size_t maximum)
{
    std::lock_guard lock(m_mutex);
    auto &s = Idle();
    if (!maximum || !s.NeedsWindow(s.cursor.DescribeWork(1)))
        throw std::logic_error("InxPack is not waiting for a read window");
    const auto bytes = static_cast<size_t>((std::min)(s.rawLeft, uint64_t{maximum}));
    auto claim = owner.TryReserveCpuStorage(bytes);
    if (!claim)
        return false;
    s.output = std::make_shared<ReadBuffer>(bytes, std::move(claim));
    s.outputOffset = 0;
    Snapshot();
    return true;
}
uint64_t ReadTask::ReadBufferRemainingBytes() const
{
    std::lock_guard lock(m_mutex);
    return Idle().rawLeft;
}
std::shared_ptr<ReadTask::ReadBuffer> ReadTask::TakeReadBuffer()
{
    std::lock_guard lock(m_mutex);
    auto &s = Idle();
    if (!s.WindowReady())
        throw std::logic_error("InxPack read buffer has not finished");
    auto output = std::exchange(s.output, nullptr);
    s.rawLeft -= output->bytes;
    Snapshot();
    return output;
}
void ReadTask::BeginMaterialize(const std::string &entryPath, const std::filesystem::path &destination,
                                const std::string &expectedSha256)
{
    std::lock_guard lock(m_mutex);
    Idle().cursor.BeginMaterialize(entryPath, destination, expectedSha256);
    Snapshot();
}
std::shared_ptr<VerifiedFile> ReadTask::TakeMaterializedFile()
{
    std::lock_guard lock(m_mutex);
    auto file = Idle().cursor.TakeMaterializedFile();
    Snapshot();
    return file;
}
void ReadTask::Close()
{
    std::unique_lock lock(m_mutex);
    if (PendingLocked())
        throw std::logic_error("InxPack IO step is still running");
    if (m_state->cursor.HasMaterialization()) {
        // Explicit cold Close can wait; automatic retirement uses Cancel and
        // observes the exact cleanup job before calling Close on a closed cursor.
        lock.unlock();
        Cancel();
        Wait();
        return;
    }
    m_state->RetireCancelled();
    Snapshot();
}
bool ReadTask::Cancel()
{
    std::lock_guard lock(m_mutex);
    if (!PendingLocked() && m_state->cursor.GetState() == ReadState::Closed)
        return false;
    const auto previous = m_state->control.fetch_or(State::cancel, std::memory_order_acq_rel);
    if (previous & State::cancel)
        return false;
    if (previous & State::active) {
#ifdef INX_PLATFORM_WINDOWS
        if (const auto pending = std::atomic_load(&m_state->pendingRange))
            pending->Cancel();
#endif
    } else {
        if (m_state->cursor.HasFileResources())
            Dispatch(1, 0, true);
        else {
            m_state->RetireCancelled();
            Snapshot();
        }
    }
    return true;
}
void ReadTask::Wait()
{
    while (true) {
        JobHandle job;
        std::optional<JobCompletionEvent> entry;
        {
            std::lock_guard lock(m_mutex);
            if (!PendingLocked())
                return;
            job = m_job;
            entry = m_entryCompletion;
        }
        if (entry && !entry->IsComplete())
            entry->WaitUntil(IoClock::time_point::max());
        if (!job.IsComplete())
            JobSystem::Get().WaitPassiveCompletion(job);
    }
}
bool ReadTask::IsPending() const
{
    std::lock_guard lock(m_mutex);
    return PendingLocked();
}
bool ReadTask::IsComplete() const
{
    std::lock_guard lock(m_mutex);
    return !PendingLocked() && ((m_state->control.load(std::memory_order_acquire) & State::cancel) ||
                                !m_state->error.empty() || m_state->cursor.GetState() == ReadState::Closed);
}
std::string ReadTask::GetStateName() const
{
    std::lock_guard lock(m_mutex);
    if (PendingLocked())
        return "running";
    if (!m_state->error.empty())
        return "failed";
    if (m_state->control.load(std::memory_order_acquire) & State::cancel)
        return "cancelled";
    switch (m_state->cursor.GetState()) {
    case ReadState::Opening:
        return "opening";
    case ReadState::Ready:
        return "ready";
    case ReadState::Checking:
        return "checking";
    case ReadState::Reading:
        if (m_state->WindowReady())
            return "output";
        if (m_state->NeedsWindow(m_state->cursor.DescribeWork(1)))
            return "needs_buffer";
        return "reading";
    case ReadState::Materializing:
        return "materializing";
    case ReadState::Closed:
        return "closed";
    }
    throw std::logic_error("unknown InxPack read state");
}
std::string ReadTask::GetError() const
{
    std::lock_guard lock(m_mutex);
    return PendingLocked() ? std::string{} : m_state->error;
}
bool ReadTask::IsChecksumVerified() const
{
    std::lock_guard lock(m_mutex);
    return Idle().cursor.IsChecksumVerified();
}
ReadProgress ReadTask::GetProgress() const
{
    std::lock_guard lock(m_mutex);
    if (PendingLocked())
        throw std::logic_error("InxPack IO step is still running");
    auto progress = m_state->cursor.GetProgress();
    progress.bufferedBytes = m_state->bufferedBytes;
    progress.bufferWorkBytes = m_state->bufferWorkBytes;
    progress.overlappedReads = m_state->overlappedReads;
    progress.pendingOverlappedReads = m_state->pendingOverlappedReads;
    progress.fileCompletionNs = m_state->fileCompletionNs;
    return progress;
}
IoWork ReadTask::DescribeWork(size_t maxBytes) const
{
    std::lock_guard lock(m_mutex);
    return Idle().DescribeWork(maxBytes);
}
IoWork ReadTask::GetSubmittedWork() const
{
    std::lock_guard lock(m_mutex);
    return m_submittedWork;
}
IoRecipe ReadTask::DescribeRemainingWork() const
{
    std::lock_guard lock(m_mutex);
    auto &s = Idle();
    auto recipe = s.cursor.DescribeRemainingWork();
    if (s.cursor.GetState() == ReadState::Reading)
        MarkBufferedRecipe(recipe, s.windowed);
    return recipe;
}
IoRecipe ReadTask::DescribeReadWork(const std::string &entryPath, bool windowed) const
{
    std::lock_guard lock(m_mutex);
    auto recipe = Idle().cursor.DescribeReadWork(entryPath);
    MarkBufferedRecipe(recipe, windowed);
    return recipe;
}
Manifest ReadTask::GetManifest() const
{
    std::lock_guard lock(m_mutex);
    return Idle().cursor.GetManifest();
}
size_t ReadTask::GetRetainedBytes() const
{
    return sizeof(*this) + m_state->retained.load(std::memory_order_relaxed);
}
size_t ReadTask::GetReservationBytes() const
{
    return sizeof(*this) + m_state->reservation.load(std::memory_order_relaxed);
}
uint64_t ReadTask::GetExecutionNs() const
{
    return m_state->executionNs.load(std::memory_order_relaxed);
}
uint64_t ReadTask::GetSubmissionCount() const
{
    return m_state->submissions.load(std::memory_order_relaxed);
}
bool ReadTask::RanOnWorker() const
{
    return m_state->ranOnWorker.load(std::memory_order_relaxed);
}
} // namespace infernux::inxpack
