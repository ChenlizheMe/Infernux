#include "DocumentStore.h"

#include "AtomicFile.h"
#include "InxPath.h"
#include "IoStep.h"
#include <core/threading/FrameWorkOwner.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <limits>
#include <ostream>
#include <streambuf>
#include <utility>

namespace infernux
{
namespace
{
constexpr size_t kOpenWriteLimit = 4;
constexpr size_t kMaxDocumentRange = 64 * 1024 * 1024;

class AtomicDocumentStream final : public std::streambuf
{
  public:
    explicit AtomicDocumentStream(AtomicFileWriter &writer) : m_writer(writer)
    {
        setp(m_buffer.data(), m_buffer.data() + m_buffer.size());
    }

  private:
    int sync() override
    {
        const auto bytes = static_cast<size_t>(pptr() - pbase());
        if (bytes)
            m_writer.Append({pbase(), bytes});
        setp(m_buffer.data(), m_buffer.data() + m_buffer.size());
        return 0;
    }

    int_type overflow(int_type value) override
    {
        sync();
        if (!traits_type::eq_int_type(value, traits_type::eof())) {
            *pptr() = traits_type::to_char_type(value);
            pbump(1);
        }
        return traits_type::not_eof(value);
    }

    AtomicFileWriter &m_writer;
    std::array<char, 64 * 1024> m_buffer;
};

bool Await(const JobCompletionEvent &event,
           std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max())
{
    if (event.IsComplete())
        return true;
    return JobSystem::IsAvailable() ? JobSystem::Get().AwaitCompletion(event, deadline) : event.WaitUntil(deadline);
}

std::string FailureMessage(const std::exception_ptr &failure)
{
    try {
        std::rethrow_exception(failure);
    } catch (const std::exception &error) {
        return error.what();
    } catch (...) {
        return "document writer failed";
    }
}
} // namespace

void DocumentWriteTicket::Wait() const
{
    Expedite();
    Await(m_completion);
    ThrowIfFailed();
}

bool DocumentWriteTicket::WaitFor(std::chrono::milliseconds timeout) const
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    if (timeout.count() > 0)
        Expedite();
    if (!Await(m_completion, deadline))
        return false;
    ThrowIfFailed();
    return true;
}

void DocumentWriteTicket::Expedite() const
{
    if (m_runtimeStore && !m_completion.IsComplete())
        m_runtimeStore->Expedite(*this);
}

std::optional<DocumentFileState> DocumentWriteTicket::GetCommittedFileState() const
{
    std::lock_guard lock(m_mutex);
    if (m_status != Status::Succeeded)
        return std::nullopt;
    return m_receipt->fileState;
}

std::optional<FilePublication> DocumentWriteTicket::GetPublication() const
{
    std::lock_guard lock(m_mutex);
    return m_receipt ? std::optional(m_receipt->publication) : std::nullopt;
}

bool DocumentWriteTicket::IsComplete() const
{
    std::lock_guard lock(m_mutex);
    return m_status != Status::Pending;
}

std::string DocumentWriteTicket::GetStatusName() const
{
    std::lock_guard lock(m_mutex);
    switch (m_status) {
    case Status::Pending:
        return "pending";
    case Status::Succeeded:
        return "succeeded";
    case Status::Superseded:
        return "superseded";
    case Status::Cancelled:
        return "cancelled";
    case Status::Failed:
        return "failed";
    }
    return "invalid";
}

std::string DocumentWriteTicket::GetError() const
{
    std::lock_guard lock(m_mutex);
    return m_failure ? FailureMessage(m_failure) : m_error;
}

void DocumentWriteTicket::Complete(Status status, std::string error, std::optional<AtomicWriteReceipt> receipt,
                                   std::exception_ptr failure)
{
    {
        std::lock_guard lock(m_mutex);
        if (m_status != Status::Pending)
            throw std::logic_error("document write ticket completed more than once");
        m_status = status;
        m_error = std::move(error);
        m_receipt = std::move(receipt);
        m_failure = std::move(failure);
    }
    m_completion.Complete();
}

void DocumentWriteTicket::ThrowIfFailed() const
{
    std::lock_guard lock(m_mutex);
    if (m_status == Status::Superseded)
        throw DocumentWriteSuperseded(m_error);
    if (m_status == Status::Cancelled)
        throw DocumentWriteCancelled(m_error);
    if (m_status == Status::Failed)
        throw std::runtime_error(m_failure ? FailureMessage(m_failure) : m_error);
}

DocumentStore::DocumentStore(Writer writer) : m_writer(std::move(writer))
{
    m_idle.Complete();
}

DocumentStore::~DocumentStore()
{
    Shutdown();
}

DocumentStore &DocumentStore::Instance()
{
    static auto *instance = new DocumentStore();
    return *instance;
}

FrameWorkSource DocumentStore::BeginFrameWork(FrameWorkOwner &owner)
{
    auto &store = Instance();
    {
        std::lock_guard lock(store.m_mutex);
        store.m_frameOwner = &owner;
        store.m_writeBytesPerNanosecond = 0;
        store.m_checkBytesPerNanosecond = 0;
        store.m_publishPrediction = 0;
    }
    FrameWorkSource source;
    source.observe = [&store, &owner] {
        std::lock_guard lock(store.m_mutex);
        store.ObserveFrameWriteLocked(owner);
    };
    source.refresh = source.observe;
    source.describe = [&store, &owner] {
        std::lock_guard lock(store.m_mutex);
        FrameWorkFacts facts;
        const auto &job = store.m_frameWrite.job;
        facts.observe = job.IsValid();
        facts.workerBusy = facts.observe && !job.IsComplete();
        facts.pending = facts.observe || !store.m_paths.empty();
        if (!facts.observe) {
            if (const auto *request = store.FrameCandidateLocked()) {
                const auto step = store.PlanFrameWriteLocked(owner, *request);
                if (owner.CanAdmitResource(FrameWorkResource::Io, step.prediction, true)) {
                    facts.host = {request->sequence, 0, true, false};
                    facts.ioReady = true;
                }
            }
        }
        return facts;
    };
    source.advance = [&store, &owner](bool, uint64_t sequence) {
        std::lock_guard lock(store.m_mutex);
        const auto *request = store.FrameCandidateLocked();
        if (!request || request->sequence != sequence || store.m_frameWrite.job.IsValid())
            return false;
        const auto plan = store.PlanFrameWriteLocked(owner, *request);
        if (!owner.CanAdmitResource(FrameWorkResource::Io, plan.prediction, true))
            return false;
        const std::string key = request->key;
        auto step = std::make_shared<WriteStep>(plan);
        auto job = store.SchedulePathLocked(key, step);
        if (job.IsValid()) {
            const auto grant = owner.RecordAcceptedResource(FrameWorkResource::Io, plan.prediction);
            store.m_frameWrite = {std::move(job), std::move(step), owner.Opportunity(), grant};
        } else {
            store.ScheduleReadyPathLocked(key);
            store.FinishPathLocked(key);
        }
        return true;
    };
    return source;
}

void DocumentStore::EndFrameWork(FrameWorkOwner &owner)
{
    Instance().Drain(&owner);
}

DocumentStore::Request *DocumentStore::NextRequestLocked(const std::string &key)
{
    const auto found = m_paths.find(key);
    if (found == m_paths.end())
        return nullptr;
    auto &queue = found->second;
    if (queue.active)
        return &*queue.active;
    return queue.pending ? &*queue.pending : nullptr;
}

const DocumentStore::Request *DocumentStore::FrameCandidateLocked() const
{
    if (!m_frameOwner || m_state != State::Running)
        return nullptr;
    const Request *oldest = nullptr;
    const auto consider = [&](const Request &request) {
        if ((!oldest || request.sequence < oldest->sequence) && !request.cold && !request.blocking)
            oldest = &request;
    };
    // Limit handles held between host grants. A save that publishes within
    // its grant does not leave another handle open between frames.
    const bool canOpen = m_activeCount < kOpenWriteLimit;
    for (const auto &entry : m_paths) {
        const auto &queue = entry.second;
        if (queue.queued)
            continue;
        if (queue.active)
            consider(*queue.active);
        else if (queue.pending && (canOpen || PlanFrameWriteLocked(*m_frameOwner, *queue.pending).publish))
            consider(*queue.pending);
    }
    return oldest;
}

std::optional<DocumentFileState> DocumentStore::ExpectedStateLocked(const Request &request) const
{
    const auto &options = request.options;
    if (!request.writer && options.mode == FileWriteMode::Replace && options.expectedFileState &&
        !options.commitChainToken.empty()) {
        const auto &committed = m_history.at(request.key).committed;
        if (committed && committed->commitChainToken == options.commitChainToken)
            return committed->fileState;
    }
    return options.expectedFileState;
}

DocumentStore::WriteStep DocumentStore::PlanFrameWriteLocked(const FrameWorkOwner &owner, const Request &request) const
{
    const auto expected = ExpectedStateLocked(request);
    const auto expectedBytes =
        request.options.mode == FileWriteMode::Replace && expected && expected->exists ? expected->size : 0;
    const auto written = request.writer ? request.writer->WrittenBytes() : 0;
    const auto verified = request.writer ? request.writer->VerifiedBytes() : 0;
    const auto remainingOutput = request.Bytes().size() - written;
    const auto remainingInput = expectedBytes - verified;
    const bool verify = !remainingOutput && remainingInput;
    const auto remaining = verify ? remainingInput : remainingOutput;
    if (!remaining)
        return {0, m_publishPrediction, true};

    const auto allowance = owner.ResourceRemaining(FrameWorkResource::Io);
    const auto [bytes, prediction] = detail::PlanFileIoWindow(
        remaining, allowance, verify ? m_checkBytesPerNanosecond : m_writeBytesPerNanosecond, kMaxDocumentRange);
    const bool publish = (verify || !remainingInput) && bytes == remaining &&
                         (m_publishPrediction || bytes <= detail::kMinimumFileWindow) && prediction <= allowance &&
                         m_publishPrediction <= allowance - prediction;
    return {static_cast<size_t>(bytes), prediction + (publish ? m_publishPrediction : 0), publish,
            verify ? WriteStep::Range::ExpectedInput : WriteStep::Range::Output};
}

void DocumentStore::ObserveFrameWriteLocked(FrameWorkOwner &owner)
{
    if (!m_frameWrite.job.IsValid() || !m_frameWrite.job.IsComplete())
        return;
    const auto &step = *m_frameWrite.step;
    owner.ObserveResource(FrameWorkResource::Io, step.actual, m_frameWrite.frame, m_frameWrite.grant);
    auto &rate = step.range == WriteStep::Range::ExpectedInput ? m_checkBytesPerNanosecond : m_writeBytesPerNanosecond;
    detail::ObserveFileIoWindow(rate, step.bytes, step.rangeNanoseconds);
    if (step.publish && step.actual > step.rangeNanoseconds)
        m_publishPrediction = step.actual - step.rangeNanoseconds;
    m_frameWrite = {};
}

std::shared_ptr<DocumentWriteTicket> DocumentStore::Submit(const std::string &path, std::string content,
                                                           DocumentWriteOptions options)
{
    std::shared_ptr<DocumentWriteTicket> accepted;
    SubmitContent(path, std::move(content), std::move(options), accepted);
    return accepted;
}

std::shared_ptr<DocumentWriteTicket> DocumentStore::SubmitBuffer(const std::string &path, DocumentWriteBuffer content,
                                                                 DocumentWriteOptions options)
{
    std::shared_ptr<DocumentWriteTicket> accepted;
    SubmitContent(path, std::move(content), std::move(options), accepted);
    return accepted;
}

std::shared_ptr<DocumentWriteTicket> DocumentStore::SubmitPrepared(const std::string &path, AtomicPreparedFile prepared,
                                                                   DocumentWriteOptions options)
{
    std::shared_ptr<DocumentWriteTicket> accepted;
    SubmitContent(path, std::move(prepared), std::move(options), accepted);
    return accepted;
}

std::shared_ptr<DocumentWriteTicket> DocumentStore::SubmitStaged(const std::string &path, std::string content,
                                                                 const PathWriteScope &parent,
                                                                 std::shared_ptr<FrameCpuStorageClaim> storage)
{
    DocumentWriteOptions options;
    options.writeScope = parent;
    options.preclaimedStorage = std::move(storage);
    std::shared_ptr<DocumentWriteTicket> accepted;
    SubmitContent(path, std::move(content), std::move(options), accepted, true);
    return accepted;
}

void DocumentStore::SubmitContent(const std::string &path, Content content, DocumentWriteOptions options,
                                  std::shared_ptr<DocumentWriteTicket> &accepted, bool staged)
{
    const bool cold = !JobSystem::IsAvailable() || JobSystem::Get().IsInline();
    // Journal paths have already been resolved before their write set is claimed.
    const auto resolvedPath = staged || std::holds_alternative<AtomicPreparedFile>(content) ? path : ResolvePath(path);
    const std::string normalizedPath = LexicalFilesystemPathKey(resolvedPath);
    // Register before accepting/coalescing a generation. A rejected new save
    // must neither supersede a previous ticket nor outlive a path mutation.
    auto writeScope = PathWriteScope::BeginWrite({resolvedPath}, options.writeScope);
    std::shared_ptr<DocumentWriteTicket> superseded;
    bool drive = false;

    {
        std::lock_guard lock(m_mutex);
        if (m_state == State::Closing)
            throw std::runtime_error("DocumentStore is shutting down");
        if (cold) {
            m_jobs.erase(std::remove_if(m_jobs.begin(), m_jobs.end(), [](const auto &job) { return job.IsComplete(); }),
                         m_jobs.end());
            if (!m_jobs.empty())
                throw std::logic_error("shared document callbacks must retire before a cold write");
        }
        const bool managed = m_frameOwner && !cold;
        const auto *owned = std::get_if<std::string>(&content);
        const auto *borrowed = std::get_if<std::string_view>(&content);
        const auto bytes = owned ? owned->capacity() : borrowed ? borrowed->size() : 0;
        auto storage = options.preclaimedStorage;
        if (storage && (!owned || bytes > storage->Bytes()))
            throw std::logic_error("document output exceeds its producer's CPU reservation");
        if (!storage && managed && bytes)
            storage = m_frameOwner->TrackOwnedCpuStorage(bytes);
        auto &history = m_history[normalizedPath];
        const uint64_t generation = history.submitted + 1;
        auto ticket = std::shared_ptr<DocumentWriteTicket>(
            new DocumentWriteTicket(normalizedPath, generation, managed ? this : nullptr));
        std::string supersededError;
        if (const auto existing = m_paths.find(normalizedPath); existing != m_paths.end() && existing->second.pending) {
            superseded = existing->second.pending->ticket;
            supersededError = "document generation " + std::to_string(superseded->GetGeneration()) + " for '" +
                              normalizedPath + "' was superseded by generation " + std::to_string(generation);
        }

        const bool blocking = !owned && !std::holds_alternative<DocumentWriteBuffer>(content);
        Request request{normalizedPath,
                        resolvedPath,
                        std::move(storage),
                        std::move(content),
                        std::move(options),
                        ticket,
                        std::move(writeScope),
                        m_nextSequence++,
                        cold,
                        blocking,
                        staged};
        std::optional<JobCompletionEvent> idle;
        if (IsIdleLocked())
            idle.emplace(JobDomain::IO);
        auto &queue = m_paths.try_emplace(normalizedPath).first->second;
        queue.keepHistory |= !staged;
        queue.pending = std::move(request);
        accepted = std::move(ticket);
        history.submitted = generation;
        if (idle)
            m_idle = std::move(*idle);
        m_state = State::Running;
        if (superseded)
            superseded->Complete(DocumentWriteTicket::Status::Superseded, std::move(supersededError));
        drive = ScheduleReadyPathLocked(normalizedPath);
        FinishPathLocked(normalizedPath);
    }

    if (drive)
        RunPath(normalizedPath);
}

void DocumentStore::Expedite(const DocumentWriteTicket &ticket)
{
    const auto &key = ticket.GetPath();
    std::lock_guard lock(m_mutex);
    const auto found = m_paths.find(key);
    if (found == m_paths.end())
        return;
    auto &queue = found->second;
    auto &request = queue.active && queue.active->ticket.get() == &ticket ? queue.active : queue.pending;
    if (!request || request->ticket.get() != &ticket)
        return;
    request->blocking = true;
    ScheduleReadyPathLocked(key);
    FinishPathLocked(key);
}

uint64_t DocumentStore::WriteAndWait(const std::string &path, std::string_view content, DocumentWriteOptions options)
{
    return WriteBlocking(path, content, std::move(options));
}

uint64_t DocumentStore::WriteStreamAndWait(const std::string &path, StreamWriter content, DocumentWriteOptions options)
{
    return WriteBlocking(path, std::move(content), std::move(options));
}

uint64_t DocumentStore::WriteBlocking(const std::string &path, Content content, DocumentWriteOptions options)
{
    std::shared_ptr<DocumentWriteTicket> ticket;
    try {
        SubmitContent(path, std::move(content), std::move(options), ticket);
        Await(ticket->m_completion);
    } catch (...) {
        if (ticket && !ticket->IsComplete())
            RetireBorrowedWrite(ticket);
        throw;
    }
    ticket->ThrowIfFailed();
    return ticket->GetGeneration();
}

void DocumentStore::RetireBorrowedWrite(const std::shared_ptr<DocumentWriteTicket> &ticket)
{
    const auto &key = ticket->GetPath();
    {
        std::lock_guard lock(m_mutex);
        const auto found = m_paths.find(key);
        if (found == m_paths.end())
            return;
        auto &queue = found->second;
        if (queue.pending && queue.pending->ticket == ticket) {
            queue.pending.reset();
            ticket->Complete(DocumentWriteTicket::Status::Cancelled);
            FinishPathLocked(key);
            return;
        }
        if (!queue.active || queue.active->ticket != ticket)
            return;
    }
    // An active synchronous write has already acquired the caller's bytes.
    Await(ticket->m_completion);
}

bool DocumentStore::Cancel(const std::shared_ptr<DocumentWriteTicket> &ticket)
{
    if (!ticket)
        return false;

    const auto &key = ticket->GetPath();
    std::lock_guard lock(m_mutex);
    const auto found = m_paths.find(key);
    if (found == m_paths.end() || !found->second.pending || found->second.pending->ticket != ticket)
        return false;
    auto error = "document generation " + std::to_string(ticket->GetGeneration()) + " for '" + key +
                 "' was cancelled before IO began";
    found->second.pending.reset();
    // The queued activation still runs to retire its path and owner access.
    ticket->Complete(DocumentWriteTicket::Status::Cancelled, std::move(error));
    FinishPathLocked(key);
    return true;
}

DocumentPathMetrics DocumentStore::GetMetrics(const std::string &path) const
{
    const std::string key = FilesystemPathKey(path);
    std::lock_guard lock(m_mutex);
    DocumentPathMetrics metrics;
    if (const auto history = m_history.find(key); history != m_history.end()) {
        metrics.latestSubmittedGeneration = history->second.submitted;
        metrics.latestSucceededGeneration = history->second.succeeded;
        metrics.latestFailedGeneration = history->second.failed;
    }
    if (const auto found = m_paths.find(key); found != m_paths.end()) {
        const auto &queue = found->second;
        if (queue.pending)
            metrics.pendingGeneration = queue.pending->ticket->GetGeneration();
        if (queue.active)
            metrics.activeGeneration = queue.active->ticket->GetGeneration();
    }
    return metrics;
}

DocumentFileState DocumentStore::CaptureFileState(const std::string &path) const
{
    return CaptureAtomicFileState(ResolvePath(path));
}

bool DocumentStore::IsIdle() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return IsIdleLocked(nullptr);
}

void DocumentStore::Flush()
{
    const auto event = [this] {
        std::lock_guard lock(m_mutex);
        m_flushing = !IsIdleLocked();
        for (auto item = m_paths.begin(); item != m_paths.end();) {
            const auto &key = (item++)->first;
            ScheduleReadyPathLocked(key);
            FinishPathLocked(key);
        }
        return m_idle;
    }();
    Await(event);
}

void DocumentStore::Flush(const std::string &path)
{
    const std::string key = FilesystemPathKey(path);
    const auto event = [this, &key]() -> std::optional<JobCompletionEvent> {
        std::lock_guard lock(m_mutex);
        const auto found = m_paths.find(key);
        if (found == m_paths.end())
            return std::nullopt;
        found->second.flushing = true;
        const auto completion = found->second.idle;
        ScheduleReadyPathLocked(key);
        FinishPathLocked(key);
        return completion;
    }();
    if (event)
        Await(*event);
}

void DocumentStore::Shutdown()
{
    Drain(nullptr);
}

void DocumentStore::Drain(FrameWorkOwner *frameOwner)
{
    const auto releaseFrameOwner = [&] {
        if (frameOwner && m_frameOwner == frameOwner) {
            ObserveFrameWriteLocked(*frameOwner);
            m_frameOwner = nullptr;
        }
    };
    std::vector<JobHandle> jobs;
    {
        std::unique_lock lock(m_mutex);
        m_condition.wait(lock, [this] { return m_state != State::Closing; });
        if (m_state == State::Stopped) {
            releaseFrameOwner();
            return;
        }
        m_state = State::Closing;
    }
    Flush();
    {
        std::lock_guard lock(m_mutex);
        jobs = std::move(m_jobs);
    }
    // The durable result and logical idle can precede the callback's epilogue.
    // Keep the store alive until every accepted JobHandle really completes.
    for (const auto &job : jobs)
        if (!job.IsComplete())
            JobSystem::Get().WaitPassiveCompletion(job);
    {
        std::lock_guard lock(m_mutex);
        releaseFrameOwner();
        m_state = State::Stopped;
    }
    m_condition.notify_all();
}

std::string DocumentStore::ResolvePath(const std::string &path)
{
    if (path.empty())
        throw std::invalid_argument("document path must not be empty");
    return ResolveFilesystemPath(path);
}

bool DocumentStore::ScheduleReadyPathLocked(const std::string &key)
{
    auto &queue = m_paths.at(key);
    if (queue.queued)
        return false;
    while (const auto *request = NextRequestLocked(key)) {
        if (request->cold) {
            queue.queued = true;
            return true;
        }
        const bool waitingSuccessor = queue.pending && queue.pending->blocking;
        if (m_frameOwner && !request->blocking && !waitingSuccessor && !m_flushing && !queue.flushing)
            return false;
        if (SchedulePathLocked(key).IsValid())
            return false;
    }
    return false;
}

JobHandle DocumentStore::SchedulePathLocked(const std::string &key, const std::shared_ptr<WriteStep> &step)
{
    auto &queue = m_paths.at(key);
    try {
        const auto generation = NextRequestLocked(key)->ticket->GetGeneration();
        queue.queued = true;
        m_jobs.erase(std::remove_if(m_jobs.begin(), m_jobs.end(), [](const auto &job) { return job.IsComplete(); }),
                     m_jobs.end());
        m_jobs.reserve(m_jobs.size() + 1); // Allocate before Schedule can accept the callback.
        m_jobs.push_back(JobSystem::Get().Schedule([this, key, generation, step] { RunPath(key, generation, step); },
                                                   JobDomain::IO, JobPriority::Low, detail::IoJobAffinity));
        return m_jobs.back();
    } catch (...) {
        queue.queued = false;
        CompleteWriteLocked(key, std::nullopt, std::current_exception());
        return {};
    }
}

void DocumentStore::RunPath(const std::string &key, uint64_t scheduledGeneration,
                            const std::shared_ptr<WriteStep> &step)
{
    // A worker advances one write. Standalone callers also drive successors inline.
    while (true) {
        Request *request;
        const FrameWorkOwner *storageOwner;
        {
            std::lock_guard lock(m_mutex);
            auto &queue = m_paths.at(key);
            request = NextRequestLocked(key);
            if (!request || (scheduledGeneration && request->ticket->GetGeneration() != scheduledGeneration)) {
                queue.queued = false;
                ScheduleReadyPathLocked(key);
                FinishPathLocked(key);
                return;
            }
            if (!queue.active) {
                queue.active.emplace(std::move(*queue.pending));
                queue.pending.reset();
                ++m_activeCount;
                request = &*queue.active;
                request->options.expectedFileState = ExpectedStateLocked(*request);
            }
            storageOwner = m_frameOwner;
        }

        std::exception_ptr failure;
        std::optional<AtomicWriteReceipt> receipt;
        bool terminal = true;
        const auto started = step ? detail::IoClock::now() : detail::IoClock::time_point{};
        try {
            const auto *produce = std::get_if<StreamWriter>(&request->content);
            if (const auto *prepared = std::get_if<AtomicPreparedFile>(&request->content)) {
                AtomicFileWriter writer(request->path, {request->options.mode, request->options.expectedFileState},
                                        *prepared);
                writer.VerifyExpected((std::numeric_limits<uint64_t>::max)());
                receipt = writer.Publish();
            } else if (m_writer && !request->staged && !produce) {
                receipt = m_writer(request->path, request->Bytes(), request->options);
            } else {
                if (!request->writer) {
                    const auto parent = ToFsPath(request->path).parent_path();
                    if (!request->staged && !parent.empty())
                        std::filesystem::create_directories(NativeFilePath(parent));
                    request->writer = std::make_unique<AtomicFileWriter>(
                        request->path, AtomicWriteOptions{request->options.mode, request->options.expectedFileState},
                        step || request->options.preclaimedStorage ? 0 : request->Bytes().size(), request->staged);
                }
                auto &writer = *request->writer;
                if (produce) {
                    AtomicDocumentStream buffer(writer);
                    std::ostream stream(&buffer);
                    stream.exceptions(std::ios::badbit | std::ios::failbit);
                    (*produce)(stream);
                    stream.flush();
                } else if (step && step->range == WriteStep::Range::ExpectedInput) {
                    writer.VerifyExpected(step->bytes);
                } else {
                    const auto remaining = request->Bytes().substr(writer.WrittenBytes());
                    writer.Append(step ? remaining.substr(0, step->bytes) : remaining, storageOwner);
                }
                if (step && step->bytes)
                    step->rangeNanoseconds = detail::IoElapsedNs(started);
                if (!step || produce)
                    writer.VerifyExpected((std::numeric_limits<uint64_t>::max)());
                if (!step || produce || step->publish)
                    receipt = writer.Publish();
                else
                    terminal = false;
            }
        } catch (...) {
            failure = std::current_exception();
            request->writer.reset();
        }
        if (step)
            step->actual = detail::IoElapsedNs(started);

        bool drive = false;
        {
            std::lock_guard lock(m_mutex);
            m_paths.at(key).queued = false;
            if (terminal)
                CompleteWriteLocked(key, std::move(receipt), std::move(failure));
            else
                request->sequence = m_nextSequence++;
            drive = ScheduleReadyPathLocked(key);
            FinishPathLocked(key);
        }
        if (!drive)
            return;
    }
}

void DocumentStore::CompleteWriteLocked(const std::string &key, std::optional<AtomicWriteReceipt> receipt,
                                        std::exception_ptr failure)
{
    auto &queue = m_paths.at(key);
    auto &history = m_history.at(key);
    auto *request = NextRequestLocked(key);
    const auto ticket = request->ticket;
    if (receipt && receipt->publication == FilePublication::Written) {
        auto &chain = request->options.commitChainToken;
        if (!chain.empty() && receipt->fileState)
            history.committed.emplace(CommittedChainState{std::move(chain), *receipt->fileState});
        else
            history.committed.reset();
    }
    request->writeScope.Close();
    request->options.writeScope.Close();
    if (queue.active) {
        queue.active.reset();
        --m_activeCount;
    } else {
        queue.pending.reset();
    }

    auto &generation = receipt ? history.succeeded : history.failed;
    generation = ticket->GetGeneration();
    if (receipt)
        ticket->Complete(DocumentWriteTicket::Status::Succeeded, {}, std::move(receipt));
    else
        ticket->Complete(DocumentWriteTicket::Status::Failed, {}, std::nullopt, std::move(failure));
}

void DocumentStore::FinishPathLocked(const std::string &key)
{
    const auto found = m_paths.find(key);
    auto &queue = found->second;
    if (queue.pending || queue.active || queue.queued)
        return;
    queue.idle.Complete();
    if (!queue.keepHistory)
        m_history.erase(key);
    m_paths.erase(found);
    if (m_paths.empty()) {
        m_flushing = false;
        m_idle.Complete();
    }
}

bool DocumentStore::IsIdleLocked(const std::string *key) const
{
    if (!key)
        return m_paths.empty();
    return m_paths.find(*key) == m_paths.end();
}

} // namespace infernux
