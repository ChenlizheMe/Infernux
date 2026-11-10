/**
 * @file JobSystem.h
 * @brief Engine-wide worker pool and its scheduling primitives.
 *
 * The pool deliberately remains the only native general-purpose worker owner.
 * Domains, priorities and groups describe work submitted to that pool; they
 * never create a second executor. Existing Schedule/ScheduleBatch/ParallelFor
 * calls use the Default domain and Normal priority.
 */

#pragma once

#include "core/FoundationApi.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

namespace infernux
{

class JobCancelled final : public std::runtime_error
{
  public:
    using std::runtime_error::runtime_error;
};

/** Logical scheduling domain. A value outside the named values is valid for
 * project-specific domains and is still owned by the same JobSystem. */
enum class JobDomain : uint16_t
{
    Default = 0,
    Runtime = 1,
    Render = 2,
    Asset = 3,
    Physics = 4,
    IO = 5,
    User = 1024,
};

/** Lower values are preferred only after the aging/fairness policy is applied. */
enum class JobPriority : uint8_t
{
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
};

/** WorkerOnly jobs cannot consume an owner-thread Wait/RunPendingJobs budget. */
enum class JobAffinity : uint8_t
{
    AnyThread,
    WorkerOnly,
};

struct JobProfilerCounters
{
    uint64_t submitted = 0;
    uint64_t started = 0;
    uint64_t completed = 0;
    uint64_t failed = 0;
    uint64_t cancelled = 0;
    uint64_t blocked = 0;
    uint64_t helped = 0;
    uint64_t running = 0;
    uint64_t queued = 0;
};

/**
 * @brief Opaque completion handle for one job or a batch.
 *
 * Handles are copyable views of shared completion state. A fence returned by
 * TaskGroup::Fence additionally waits for the group to be closed.
 */
class JobHandle
{
  public:
    JobHandle() = default;
    JobHandle(const JobHandle &) = default;
    JobHandle &operator=(const JobHandle &) = default;
    JobHandle(JobHandle &&) noexcept = default;
    JobHandle &operator=(JobHandle &&) noexcept = default;

    [[nodiscard]] bool IsValid() const noexcept
    {
        return static_cast<bool>(m_state);
    }

    [[nodiscard]] INFERNUX_FOUNDATION_API bool IsComplete() const noexcept;

    /** Request cooperative cancellation. Queued callbacks are skipped; a
     * callback that has already begun is not
     * preempted. */
    INFERNUX_FOUNDATION_API bool Cancel() noexcept;

    [[nodiscard]] bool IsCancellationRequested() const noexcept
    {
        return m_state && m_state->cancelRequested.load(std::memory_order_acquire);
    }

  public:
    friend class JobSystem;
    friend class TaskGroup;

    struct State
    {
        explicit State(uint32_t count = 0, bool groupState = false) : remaining(count), closed(!groupState)
        {
        }

        std::atomic<uint32_t> remaining{0};
        std::atomic<bool> cancelRequested{false};
        std::atomic<bool> closed{true};
        mutable std::mutex completionMutex;
        std::condition_variable completionCv;
        std::exception_ptr failure;
    };

  private:
    explicit JobHandle(std::shared_ptr<State> state, bool waitsForClose = false)
        : m_state(std::move(state)), m_waitsForClose(waitsForClose)
    {
    }

    std::shared_ptr<State> m_state;
    bool m_waitsForClose = false;
};

using JobFence = JobHandle;

/** One externally completed dependency, independent of an executing callback.
 * Copies observe the same result. The producer completes it exactly once. */
class JobCompletionEvent
{
  public:
    INFERNUX_FOUNDATION_API explicit JobCompletionEvent(JobDomain domain);
    [[nodiscard]] bool IsComplete() const noexcept;
    INFERNUX_FOUNDATION_API void Complete();
    INFERNUX_FOUNDATION_API bool WaitUntil(std::chrono::steady_clock::time_point deadline) const;

  private:
    friend class JobSystem;
    JobDomain m_domain;
    std::shared_ptr<JobHandle::State> m_state;
};

/** A dynamic set of jobs sharing cancellation, error and domain semantics. */
class TaskGroup
{
  public:
    explicit TaskGroup(JobDomain domain = JobDomain::Default, JobPriority priority = JobPriority::Normal);
    INFERNUX_FOUNDATION_API ~TaskGroup();

    TaskGroup(const TaskGroup &) = delete;
    TaskGroup &operator=(const TaskGroup &) = delete;
    TaskGroup(TaskGroup &&other) noexcept;
    TaskGroup &operator=(TaskGroup &&other) noexcept;

    /** Prevent further submissions. Existing work is still drained. */
    INFERNUX_FOUNDATION_API void Close() noexcept;
    [[nodiscard]] INFERNUX_FOUNDATION_API bool IsClosed() const noexcept;
    [[nodiscard]] INFERNUX_FOUNDATION_API bool IsComplete() const noexcept;
    [[nodiscard]] INFERNUX_FOUNDATION_API bool IsCancellationRequested() const noexcept;
    /** Request cooperative cancellation for the group. Queued callbacks are
     * skipped; callbacks that have already
     * begun are not preempted. */
    INFERNUX_FOUNDATION_API bool Cancel() noexcept;

    /** Returns a fence that completes only after Close() and all jobs. */
    [[nodiscard]] INFERNUX_FOUNDATION_API JobFence Fence() const;
    [[nodiscard]] JobDomain GetDomain() const noexcept
    {
        return m_domain;
    }
    [[nodiscard]] JobPriority GetPriority() const noexcept
    {
        return m_priority;
    }

  private:
    friend class JobSystem;

    std::shared_ptr<JobHandle::State> m_state;
    mutable std::mutex m_mutex;
    JobDomain m_domain = JobDomain::Default;
    JobPriority m_priority = JobPriority::Normal;
};

/** Engine-wide worker pool. Shutdown joins all owned workers and rejects executing-job reentry.
 * References returned by Get remain valid only within that engine lifetime. */
class JobSystem
{
  public:
    using JobFn = std::function<void()>;

    enum class State : uint8_t
    {
        Running,
        Draining,
        Stopped,
    };

    INFERNUX_FOUNDATION_API static void Initialize(uint32_t workerCount = 0);
    /** Initialize an owner-thread executor for platforms without worker threads. */
    INFERNUX_FOUNDATION_API static void InitializeInline();
    INFERNUX_FOUNDATION_API static void Shutdown();
    [[nodiscard]] INFERNUX_FOUNDATION_API static JobSystem &Get();
    [[nodiscard]] INFERNUX_FOUNDATION_API static bool IsAvailable() noexcept;

    /// Pin this worker pool while a native asynchronous operation owns a
    /// continuation. Shutdown joins the workers and then drains these owners.
    class ExternalWork
    {
      public:
        ExternalWork(ExternalWork &&other) noexcept;
        ~ExternalWork();
        JobHandle Schedule(JobFn job, JobDomain domain, JobPriority priority, JobAffinity affinity) const;

      private:
        friend class JobSystem;
        explicit ExternalWork(JobSystem &system) noexcept : m_system(&system)
        {
        }
        JobSystem *m_system;
    };
    ExternalWork RetainExternalWork();

    JobSystem(const JobSystem &) = delete;
    JobSystem &operator=(const JobSystem &) = delete;
    JobSystem(JobSystem &&) = delete;
    JobSystem &operator=(JobSystem &&) = delete;

    /** Existing callers retain Default/Normal behavior. */
    INFERNUX_FOUNDATION_API JobHandle Schedule(JobFn job, JobDomain domain = JobDomain::Default,
                                               JobPriority priority = JobPriority::Normal,
                                               JobAffinity affinity = JobAffinity::AnyThread);

    INFERNUX_FOUNDATION_API JobHandle Schedule(TaskGroup &group, JobFn job);
    INFERNUX_FOUNDATION_API JobHandle Schedule(TaskGroup &group, JobFn job, JobPriority priority);

    INFERNUX_FOUNDATION_API JobHandle ScheduleBatch(uint32_t count, std::function<JobFn(uint32_t index)> factory,
                                                    JobDomain domain = JobDomain::Default,
                                                    JobPriority priority = JobPriority::Normal);
    INFERNUX_FOUNDATION_API JobHandle ScheduleBatch(TaskGroup &group, uint32_t count,
                                                    std::function<JobFn(uint32_t index)> factory);
    INFERNUX_FOUNDATION_API JobHandle ScheduleBatch(TaskGroup &group, uint32_t count,
                                                    std::function<JobFn(uint32_t index)> factory, JobPriority priority);

    INFERNUX_FOUNDATION_API void ParallelFor(uint32_t count, std::function<void(uint32_t index)> body,
                                             JobDomain domain = JobDomain::Default,
                                             JobPriority priority = JobPriority::Normal);
    /** Execute contiguous ranges instead of allocating one task per element. */
    INFERNUX_FOUNDATION_API void ParallelForChunks(uint32_t count, uint32_t chunkSize,
                                                   std::function<void(uint32_t begin, uint32_t end)> body,
                                                   JobDomain domain = JobDomain::Default,
                                                   JobPriority priority = JobPriority::Normal);
    INFERNUX_FOUNDATION_API void ParallelFor(TaskGroup &group, uint32_t count,
                                             std::function<void(uint32_t index)> body);

    INFERNUX_FOUNDATION_API TaskGroup CreateTaskGroup(JobDomain domain = JobDomain::Default,
                                                      JobPriority priority = JobPriority::Normal) const;

    /** Wait helps work from the target group/domain while preserving permits. */
    INFERNUX_FOUNDATION_API void Wait(const JobHandle &handle);
    INFERNUX_FOUNDATION_API void Wait(TaskGroup &group);

    /** Wait and help only this dependency. Use while holding a resource that unrelated work may need. */
    INFERNUX_FOUNDATION_API void WaitDependency(const JobHandle &handle);

    /** Wait without executing work; a worker temporarily releases its permit. */
    INFERNUX_FOUNDATION_API void WaitPassive(const JobHandle &handle);
    INFERNUX_FOUNDATION_API void WaitPassive(TaskGroup &group);
    /** Wait for completion; the owner reads failure/cancellation from its result ticket. */
    INFERNUX_FOUNDATION_API void WaitPassiveCompletion(const JobHandle &handle);

    /** Owners wait passively; real workers may help the dependency's domain.
     * A deadline stops new help/waits, not a running atom or permit restoration. */
    INFERNUX_FOUNDATION_API bool
    AwaitCompletion(const JobCompletionEvent &event,
                    std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max());

    /** 0 means unlimited. Changes affect tasks not yet dequeued. */
    INFERNUX_FOUNDATION_API void SetDomainConcurrency(JobDomain domain, uint32_t maxConcurrency);
    [[nodiscard]] INFERNUX_FOUNDATION_API uint32_t GetDomainConcurrency(JobDomain domain) const;
    [[nodiscard]] INFERNUX_FOUNDATION_API uint32_t GetDomainActiveCount(JobDomain domain) const;

    [[nodiscard]] INFERNUX_FOUNDATION_API JobProfilerCounters GetProfilerCounters() const;
    [[nodiscard]] INFERNUX_FOUNDATION_API JobProfilerCounters GetProfilerCounters(JobDomain domain) const;
    /** Clear accumulated observations while preserving queued/running gauges.
     * Jobs crossing this boundary count when they start/finish; totals are not
     * a cohort of jobs submitted since reset. Safe while work is in flight. */
    INFERNUX_FOUNDATION_API void ResetProfilerCounters();

    /** Execute queued work on the calling thread. Zero drains the queue. */
    INFERNUX_FOUNDATION_API uint32_t RunPendingJobs(uint32_t maxJobs = 0);

    [[nodiscard]] bool IsInline() const noexcept
    {
        return m_inline;
    }

    [[nodiscard]] uint32_t GetWorkerCount() const noexcept
    {
        return static_cast<uint32_t>(m_workers.size());
    }
    [[nodiscard]] State GetState() const noexcept
    {
        return m_state.load(std::memory_order_acquire);
    }
    [[nodiscard]] INFERNUX_FOUNDATION_API size_t GetQueuedTaskCount() const;
    [[nodiscard]] uint32_t GetActiveTaskCount() const noexcept
    {
        return m_activeTasks.load(std::memory_order_acquire);
    }

  private:
    JobSystem() = default;
    enum class WaitPolicy : uint8_t
    {
        Passive,
        HelpEligible,
        HelpDependency
    };
    void WaitForCompletion(const JobHandle &handle, WaitPolicy policy);
    ~JobSystem();

    struct ProfilerState
    {
        std::atomic<uint64_t> submitted{0};
        std::atomic<uint64_t> started{0};
        std::atomic<uint64_t> completed{0};
        std::atomic<uint64_t> failed{0};
        std::atomic<uint64_t> cancelled{0};
        std::atomic<uint64_t> blocked{0};
        std::atomic<uint64_t> helped{0};
        std::atomic<uint64_t> running{0};
        std::atomic<uint64_t> queued{0};
    };

    struct Task
    {
        JobFn fn;
        std::shared_ptr<JobHandle::State> state;
        std::shared_ptr<JobHandle::State> groupState;
        std::shared_ptr<ProfilerState> profiler;
        JobDomain domain = JobDomain::Default;
        JobPriority priority = JobPriority::Normal;
        uint64_t sequence = 0;
        JobAffinity affinity = JobAffinity::AnyThread;
    };

    struct ExecutionContext
    {
        JobSystem *system = nullptr;
        JobDomain domain = JobDomain::Default;
        std::shared_ptr<JobHandle::State> state;
        std::shared_ptr<JobHandle::State> groupState;
        bool ownsPermit = false;
    };

    static thread_local ExecutionContext *s_currentContext;
    // Unlike ExecutionContext, only WorkerLoop establishes this identity.
    static thread_local JobSystem *s_workerSystem;

    void WorkerLoop();
    bool TryRunOne();
    bool TryRunOneForWait(const JobHandle &handle, WaitPolicy policy);
    void Execute(Task task, bool helped) noexcept;
    void StopAndJoin() noexcept;

    JobHandle ScheduleInternal(JobFn job, JobDomain domain, JobPriority priority,
                               std::shared_ptr<JobHandle::State> groupState,
                               JobAffinity affinity = JobAffinity::AnyThread);
    JobHandle ScheduleBatchInternal(uint32_t count, std::function<JobFn(uint32_t index)> factory, JobDomain domain,
                                    JobPriority priority, std::shared_ptr<JobHandle::State> groupState,
                                    std::mutex *groupMutex);

    bool DequeueTask(Task &task, const std::function<bool(const Task &)> &allowed);
    bool CanRunDomainLocked(JobDomain domain) const;
    void AcquirePermitLocked(JobDomain domain);
    void ReleasePermit(JobDomain domain) noexcept;
    bool SuspendCurrentPermit();
    void ResumeCurrentPermit();
    [[nodiscard]] bool WaitSatisfied(const JobHandle &handle) const;
    void PropagateFailureAndCancellation(const JobHandle &handle);

    std::shared_ptr<ProfilerState> ProfilerForDomain(JobDomain domain);
    static JobProfilerCounters Snapshot(const std::shared_ptr<ProfilerState> &state);
    static void RecordFailure(const std::shared_ptr<JobHandle::State> &state, std::exception_ptr failure) noexcept;
    static void CompleteState(const std::shared_ptr<JobHandle::State> &state) noexcept;

    std::vector<std::thread> m_workers;
    std::array<std::deque<Task>, 4> m_queues;
    mutable std::mutex m_queueMutex;
    std::condition_variable m_queueCv;
    std::atomic<State> m_state{State::Stopped};
    std::atomic<uint32_t> m_activeTasks{0};
    uint32_t m_externalWork = 0;
    bool m_accepting = true;
    bool m_stopRequested = false;
    bool m_inline = false;
    uint64_t m_nextSequence = 0;
    std::unordered_map<uint16_t, uint32_t> m_domainLimits;
    std::unordered_map<uint16_t, uint32_t> m_domainActive;

    std::shared_ptr<ProfilerState> m_globalProfiler = std::make_shared<ProfilerState>();
    mutable std::mutex m_profilerMutex;
    std::unordered_map<uint16_t, std::shared_ptr<ProfilerState>> m_domainProfilers;
    std::condition_variable m_permitCv;
};

} // namespace infernux
