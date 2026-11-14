#pragma once

#include "core/FoundationApi.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace infernux
{

enum class FrameWorkResource : uint8_t
{
    Cpu,
    Gpu,
    Io,
    Transfer,
    Count
};
enum class FrameGpuAvailability : uint8_t
{
    Ready,
    ForegroundDeferred,
    ForegroundRecording,
    ForegroundBacklog,
    BackgroundPending
};

enum class FrameWorkWaitResult : uint8_t
{
    Progress,
    Waiting,
    Blocked
};

struct FrameWorkAllowance
{
    uint64_t limit = 1'000'000;
    uint64_t allowance = limit;
    uint64_t admitted = 0;
    uint64_t observed = 0;
    uint64_t calibrations = 0;
    uint64_t frameCalibrations = 0;
    uint64_t indivisibleOverruns = 0;
};

struct FrameWorkCandidate
{
    uint64_t token = 0;
    uint64_t hostNanoseconds = 0;
    bool ready = false;
    bool reserved = false;
};

struct FrameWorkFacts
{
    FrameWorkCandidate host;
    FrameWorkCandidate gpu;
    bool pending = false;
    bool gpuBusy = false;
    bool reservation = false;
    bool observe = false;
    bool workerBusy = false;
    // The selected host candidate advances IO.
    bool ioReady = false;
};

struct FrameWorkPeers
{
    bool otherReady = false;
    bool ioReady = false;
    bool workerBusy = false;
    bool gpuBusy = false;
    uint64_t ioServiceSerial = 0;
};

struct FrameWorkSource
{
    std::function<void()> observe;
    /// Receipt collection between service rounds; no lifecycle tick or full plan.
    std::function<void()> refresh;
    std::function<FrameWorkFacts()> describe;
    /// True only for retained state changes, accepted work or a new terminal result.
    std::function<bool(bool, uint64_t)> advance;
    // On-demand assets and editor images are not background throughput work.
    // They retain ordinary storage, submission and completion accounting.
    bool interactive = false;
};

struct FrameWorkCost
{
    uint64_t chargedNanoseconds = 0;
    uint64_t overrunNanoseconds = 0;
};

struct FrameCpuResidencySource
{
    std::function<uint64_t()> bytes;
    // Query may collect expired identities. Only admission may evict live entries.
    std::function<bool(uint64_t, bool)> fit;
};

/// A claim follows its storage, including accepted worker work that outlives
/// the requesting object. Only the owner admits new claims.
class FrameCpuStorageClaim final
{
  public:
    INFERNUX_FOUNDATION_API ~FrameCpuStorageClaim();
    FrameCpuStorageClaim(const FrameCpuStorageClaim &) = delete;
    FrameCpuStorageClaim &operator=(const FrameCpuStorageClaim &) = delete;
    [[nodiscard]] uint64_t Bytes() const noexcept
    {
        return m_bytes;
    }
    INFERNUX_FOUNDATION_API void ReduceTo(uint64_t bytes) noexcept;

  private:
    friend class FrameWorkOwner;
    FrameCpuStorageClaim(std::shared_ptr<std::atomic<uint64_t>> total, uint64_t bytes) noexcept;
    std::shared_ptr<std::atomic<uint64_t>> m_total;
    uint64_t m_bytes;
};

/// Frame admission only. Sources retain their existing queues, cursors and
/// execution tickets; this owner never executes a second worker/GPU queue.
class FrameWorkOwner
{
  public:
    using SourceId = uint64_t;
    static constexpr uint64_t DefaultTransferBytesPerHost = 8ULL * 1024 * 1024;
    FrameWorkOwner() = default;
    explicit FrameWorkOwner(std::function<uint64_t()> clock) : m_clock(std::move(clock))
    {
    }
    FrameWorkOwner(const FrameWorkOwner &) = delete;
    FrameWorkOwner &operator=(const FrameWorkOwner &) = delete;
    FrameWorkOwner(FrameWorkOwner &&) = delete;
    FrameWorkOwner &operator=(FrameWorkOwner &&) = delete;

    class Scope
    {
      public:
        explicit Scope(FrameWorkOwner &owner) : m_owner(owner)
        {
            m_owner.BeginScope();
        }
        ~Scope()
        {
            m_owner.EndScope();
        }
        Scope(const Scope &) = delete;
        Scope &operator=(const Scope &) = delete;

      private:
        FrameWorkOwner &m_owner;
    };

    INFERNUX_FOUNDATION_API SourceId AddSource(FrameWorkSource source);
    INFERNUX_FOUNDATION_API void RemoveSource(SourceId id);
    INFERNUX_FOUNDATION_API void Wake(SourceId id);
    /// Owner-thread notification from a nested resource operation. Admission
    /// and accounting remain separate; only the current advance observes it.
    void ReportProgress() noexcept
    {
        ++m_progressSerial;
    }
    [[nodiscard]] uint64_t ProgressSerial() const noexcept
    {
        return m_progressSerial;
    }
    [[nodiscard]] INFERNUX_FOUNDATION_API FrameWorkPeers PeerWork(SourceId id);

    INFERNUX_FOUNDATION_API void Configure(uint64_t ownerNanoseconds, uint64_t foregroundReserveNanoseconds);
    INFERNUX_FOUNDATION_API void ConfigureResources(uint64_t cpuNanoseconds, uint64_t gpuNanoseconds,
                                                    uint64_t ioNanoseconds, uint64_t transferBytes);
    [[nodiscard]] INFERNUX_FOUNDATION_API uint64_t ResourceRemaining(FrameWorkResource resource) const noexcept;
    [[nodiscard]] INFERNUX_FOUNDATION_API bool CanAdmitResource(FrameWorkResource resource, uint64_t prediction,
                                                                bool indivisible = false) const noexcept;
    INFERNUX_FOUNDATION_API uint64_t AdmitResource(FrameWorkResource resource, uint64_t prediction,
                                                   bool indivisible = false);
    /// Called after a successful queue/worker submission that passed CanAdmit.
    /// Stop blocks new admission, never accounting for work already accepted.
    INFERNUX_FOUNDATION_API uint64_t RecordAcceptedResource(FrameWorkResource resource, uint64_t prediction) noexcept;
    INFERNUX_FOUNDATION_API void ObserveResource(FrameWorkResource resource, uint64_t actual,
                                                 uint64_t submittedFrame = 0, uint64_t grant = 0) noexcept;
    [[nodiscard]] INFERNUX_FOUNDATION_API const FrameWorkAllowance &
    ResourceState(FrameWorkResource resource) const noexcept;
    void ConfigureCpuStorage(uint64_t bytes) noexcept
    {
        m_cpuStorageLimit = bytes;
    }
    void SetCpuResidencySource(const std::shared_ptr<FrameCpuResidencySource> &source) noexcept
    {
        m_cpuResidency = source;
    }
    [[nodiscard]] uint64_t CpuStorageLimit() const noexcept
    {
        return m_cpuStorageLimit;
    }
    [[nodiscard]] INFERNUX_FOUNDATION_API uint64_t CpuStorageBytes() const;
    [[nodiscard]] uint64_t CpuStorageReservedBytes() const noexcept
    {
        return m_cpuStorage->load(std::memory_order_relaxed);
    }
    [[nodiscard]] INFERNUX_FOUNDATION_API bool CanReserveCpuStorage(uint64_t bytes) const;
    [[nodiscard]] INFERNUX_FOUNDATION_API std::shared_ptr<FrameCpuStorageClaim> TryReserveCpuStorage(uint64_t bytes);
    /// Finish an already accepted operation during drain, within the current
    /// storage limit and without evicting unrelated residency.
    [[nodiscard]] INFERNUX_FOUNDATION_API std::shared_ptr<FrameCpuStorageClaim>
    TryReserveCompletionCpuStorage(uint64_t bytes);
    [[nodiscard]] INFERNUX_FOUNDATION_API bool TryGrowCpuStorage(FrameCpuStorageClaim &claim, uint64_t bytes);
    /// Account a payload already owned by a producer. Only the counter changes;
    /// an IO job that releases this storage can still run under memory pressure.
    [[nodiscard]] INFERNUX_FOUNDATION_API std::shared_ptr<FrameCpuStorageClaim>
    TrackOwnedCpuStorage(uint64_t bytes) const;
    [[nodiscard]] std::shared_ptr<FrameCpuStorageClaim> SplitCpuStorage(FrameCpuStorageClaim &reserved, uint64_t bytes);
    INFERNUX_FOUNDATION_API uint64_t BeginOpportunity();
    /// Observe receipts and service interactive or explicitly reserved host
    /// prerequisites before runtime/foreground work. Ordinary background
    /// throughput waits for AdvanceLate; explicit dependencies use AdvanceRequired.
    INFERNUX_FOUNDATION_API bool AdvanceEarly();
    /// Advance only the source required by a pre-render consumer. Repeat calls
    /// share this opportunity's allowances; true means work remains pending.
    INFERNUX_FOUNDATION_API bool AdvanceRequired(SourceId id,
                                                 const std::function<FrameGpuAvailability()> &gpuAvailability);
    /// Advance an existing asynchronous producer dependency within this same
    /// opportunity. Late service shares ordinary background CPU/GPU credit and
    /// intersects foreground GPU pressure; it never drains or refills a budget.
    INFERNUX_FOUNDATION_API bool AdvanceDependency(SourceId id,
                                                   const std::function<FrameGpuAvailability()> &gpuAvailability);
    /// Finish only the required source, without opening/refilling a host frame.
    /// Soft frame allowances may overrun; storage and queue admission still
    /// apply. Normal return means the source has no pending work. Callers must
    /// still inspect its logical result. Stop, suspension and blocked progress
    /// throw diagnostics instead of returning a partial synchronous result.
    /// Waiting must mean a real bounded wait for this source's accepted ticket.
    /// An explicitly known dependency may be drained from the wait or source
    /// callback; requests for an already active source and cycles are rejected.
    INFERNUX_FOUNDATION_API void
    DrainRequired(SourceId id, const std::function<FrameGpuAvailability()> &gpuAvailability,
                  const std::function<FrameWorkWaitResult(SourceId, uint64_t)> &waitForProgress = {});
    INFERNUX_FOUNDATION_API bool AdvanceLate(const std::function<FrameGpuAvailability()> &gpuAvailability);
    INFERNUX_FOUNDATION_API void EndOpportunity() noexcept;
    void Stop() noexcept
    {
        m_admission.store(Admission::Closed, std::memory_order_release);
    }
    void SuspendDriver() noexcept
    {
        auto open = Admission::Open;
        m_admission.compare_exchange_strong(open, Admission::DriverSuspended, std::memory_order_acq_rel);
    }
    void Reopen() noexcept
    {
        m_admission.store(Admission::Open, std::memory_order_release);
    }
    INFERNUX_FOUNDATION_API void ClearSources();

    INFERNUX_FOUNDATION_API void BeginScope();
    INFERNUX_FOUNDATION_API FrameWorkCost EndScope();
    [[nodiscard]] INFERNUX_FOUNDATION_API int64_t Remaining(bool background = false) const;
    /// Time available to the current source callback, including its service's
    /// foreground reservation policy. This does not start or refill a budget.
    [[nodiscard]] INFERNUX_FOUNDATION_API int64_t ServiceRemaining() const;
    [[nodiscard]] uint64_t Opportunity() const noexcept
    {
        return m_opportunity;
    }
    [[nodiscard]] uint64_t TotalNanoseconds() const noexcept
    {
        return m_totalNanoseconds;
    }
    [[nodiscard]] uint64_t OverrunNanoseconds() const noexcept
    {
        return m_overrunNanoseconds;
    }
    [[nodiscard]] uint64_t OwnerNanoseconds() const noexcept
    {
        return m_ownerNanoseconds;
    }
    [[nodiscard]] uint64_t ForegroundReserveNanoseconds() const noexcept
    {
        return m_foregroundReserveNanoseconds;
    }
    [[nodiscard]] bool IsActive() const noexcept
    {
        return m_active;
    }
    [[nodiscard]] bool IsAccepting() const noexcept
    {
        return m_admission.load(std::memory_order_acquire) == Admission::Open;
    }
    [[nodiscard]] bool WasServiced() const noexcept
    {
        return m_serviced;
    }
    [[nodiscard]] bool IsLate() const noexcept
    {
        return m_late;
    }
    [[nodiscard]] bool IsEarlyAdvanced() const noexcept
    {
        return m_earlyAdvanced;
    }
    [[nodiscard]] bool IsGpuForegroundBlocked() const noexcept
    {
        return m_gpuAvailability != FrameGpuAvailability::Ready &&
               m_gpuAvailability != FrameGpuAvailability::BackgroundPending;
    }
    [[nodiscard]] INFERNUX_FOUNDATION_API const char *GpuUnavailableReason() const noexcept;

  private:
    enum class ServiceKind : uint8_t
    {
        Background,
        Interactive,
        Required,
        SynchronousRequired
    };
    class ServiceScope
    {
      public:
        ServiceScope(FrameWorkOwner &owner, bool interactive)
            : ServiceScope(owner, owner.m_serviceKind == ServiceKind::Required ||
                                          owner.m_serviceKind == ServiceKind::SynchronousRequired
                                      ? owner.m_serviceKind
                                  : interactive ? ServiceKind::Interactive
                                                : ServiceKind::Background)
        {
        }
        ServiceScope(FrameWorkOwner &owner, ServiceKind kind)
            : m_owner(owner), m_previous(std::exchange(owner.m_serviceKind, kind))
        {
        }
        ~ServiceScope()
        {
            m_owner.m_serviceKind = m_previous;
        }

      private:
        FrameWorkOwner &m_owner;
        ServiceKind m_previous;
    };
    class QuantumScope
    {
      public:
        explicit QuantumScope(FrameWorkOwner &owner);
        ~QuantumScope();

      private:
        FrameWorkOwner &m_owner;
        uint64_t m_previousStart;
        uint32_t m_previousAdmissions;
    };
    class CallbackScope
    {
      public:
        CallbackScope(FrameWorkOwner &owner, SourceId id);
        ~CallbackScope();

      private:
        FrameWorkOwner &m_owner;
        QuantumScope m_quantum;
    };
    class RequiredScope
    {
      public:
        RequiredScope(FrameWorkOwner &owner, SourceId id, bool synchronous);
        ~RequiredScope();

      private:
        FrameWorkOwner &m_owner;
        FrameGpuAvailability m_previousGpu;
        bool m_restoreGpu;
    };
    enum class Admission : uint8_t
    {
        Open,
        DriverSuspended,
        Closed
    };
    struct Source
    {
        SourceId id;
        FrameWorkSource callbacks;
        uint64_t observedOpportunity = UINT64_MAX;
        bool awake = true;
        uint64_t lastIoService = 0;
    };
    struct Choice
    {
        std::shared_ptr<Source> source;
        FrameWorkFacts facts;
    };

    void ObserveSources();
    void ObserveSource(const std::shared_ptr<Source> &source);
    void RefreshSources();
    bool Drive(const std::function<FrameGpuAvailability()> &gpuAvailability = {});
    bool DriveRequired(SourceId id, ServiceKind kind, const std::function<FrameGpuAvailability()> &gpuAvailability,
                       const std::function<FrameWorkWaitResult(SourceId, uint64_t)> &waitForProgress);
    std::vector<Choice> DescribeSources();
    std::optional<FrameWorkFacts> DescribeSource(const std::shared_ptr<Source> &source, bool interactive);
    bool AdvanceChoice(Choice &choice, bool gpu, bool interactive);
    bool Select(std::vector<Choice> &choices, bool gpu, bool interactive = false);
    static bool HasPending(const FrameWorkFacts &facts);
    bool HasPending(const std::vector<Choice> &choices) const;
    bool FitCpuStorage(uint64_t bytes, bool reclaim) const;
    uint64_t RunningNanoseconds() const;
    uint64_t Now() const;

    std::vector<std::shared_ptr<Source>> m_sources;
    SourceId m_nextSource = 1;
    SourceId m_nextHostSource = 0;
    SourceId m_nextGpuSource = 0;
    uint64_t m_progressSerial = 0;
    uint64_t m_opportunity = 0;
    uint64_t m_ownerNanoseconds = 1'000'000;
    uint64_t m_allowanceNanoseconds = m_ownerNanoseconds;
    uint64_t m_foregroundReserveNanoseconds = 0;
    uint64_t m_spentNanoseconds = 0;
    uint64_t m_totalNanoseconds = 0;
    uint64_t m_overrunNanoseconds = 0;
    uint32_t m_scopeDepth = 0;
    std::function<uint64_t()> m_clock;
    uint64_t m_scopeStarted = 0;
    std::array<FrameWorkAllowance, static_cast<size_t>(FrameWorkResource::Count)> m_resources = {
        FrameWorkAllowance{}, FrameWorkAllowance{}, FrameWorkAllowance{},
        FrameWorkAllowance{DefaultTransferBytesPerHost}};
    std::shared_ptr<std::atomic<uint64_t>> m_cpuStorage = std::make_shared<std::atomic<uint64_t>>(0);
    uint64_t m_cpuStorageLimit = 512ULL * 1024 * 1024;
    std::weak_ptr<FrameCpuResidencySource> m_cpuResidency;
    std::atomic<Admission> m_admission{Admission::Open};
    bool m_active = false;
    bool m_late = false;
    bool m_earlyAdvanced = false;
    bool m_serviced = false;
    bool m_preferGpu = false;
    ServiceKind m_serviceKind = ServiceKind::Background;
    FrameGpuAvailability m_gpuAvailability = FrameGpuAvailability::Ready;
    std::vector<SourceId> m_callbackSources;
    std::vector<SourceId> m_requiredSources;
    uint64_t m_quantumStarted = 0;
    uint32_t m_quantumAdmissions = 0;
};

} // namespace infernux
