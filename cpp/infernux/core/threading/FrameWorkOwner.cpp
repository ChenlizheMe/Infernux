#include "FrameWorkOwner.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace infernux
{

FrameWorkOwner::QuantumScope::QuantumScope(FrameWorkOwner &owner)
    : m_owner(owner), m_previousStart(owner.m_quantumStarted), m_previousAdmissions(owner.m_quantumAdmissions)
{
    if (owner.m_serviceKind == ServiceKind::SynchronousRequired) {
        owner.m_quantumStarted = owner.Now();
        owner.m_quantumAdmissions = 0;
    }
}

FrameWorkOwner::QuantumScope::~QuantumScope()
{
    m_owner.m_quantumStarted = m_previousStart;
    m_owner.m_quantumAdmissions = m_previousAdmissions;
}

FrameWorkOwner::CallbackScope::CallbackScope(FrameWorkOwner &owner, SourceId id) : m_owner(owner), m_quantum(owner)
{
    if (std::find(owner.m_callbackSources.begin(), owner.m_callbackSources.end(), id) != owner.m_callbackSources.end())
        throw std::logic_error("Frame work source callback dependency cycle at source " + std::to_string(id));
    owner.m_callbackSources.push_back(id);
}

FrameWorkOwner::CallbackScope::~CallbackScope()
{
    m_owner.m_callbackSources.pop_back();
}

FrameWorkOwner::RequiredScope::RequiredScope(FrameWorkOwner &owner, SourceId id, bool synchronous)
    : m_owner(owner), m_previousGpu(owner.m_gpuAvailability), m_restoreGpu(synchronous)
{
    if (std::find(owner.m_callbackSources.begin(), owner.m_callbackSources.end(), id) != owner.m_callbackSources.end())
        throw std::logic_error("Required frame work callback dependency cycle at source " + std::to_string(id));
    if (std::find(owner.m_requiredSources.begin(), owner.m_requiredSources.end(), id) != owner.m_requiredSources.end())
        throw std::logic_error("Required frame work dependency cycle at source " + std::to_string(id));
    if (!owner.m_requiredSources.empty() && !synchronous)
        throw std::logic_error("Nested required frame work must explicitly drain its dependency");
    owner.m_requiredSources.push_back(id);
}

FrameWorkOwner::RequiredScope::~RequiredScope()
{
    m_owner.m_requiredSources.pop_back();
    if (m_restoreGpu)
        m_owner.m_gpuAvailability = m_previousGpu;
}

FrameCpuStorageClaim::FrameCpuStorageClaim(std::shared_ptr<std::atomic<uint64_t>> total, uint64_t bytes) noexcept
    : m_total(std::move(total)), m_bytes(bytes)
{
    m_total->fetch_add(bytes, std::memory_order_relaxed);
}

FrameCpuStorageClaim::~FrameCpuStorageClaim()
{
    ReduceTo(0);
}

void FrameCpuStorageClaim::ReduceTo(uint64_t bytes) noexcept
{
    if (bytes >= m_bytes)
        return;
    m_total->fetch_sub(m_bytes - bytes, std::memory_order_relaxed);
    m_bytes = bytes;
}

uint64_t FrameWorkOwner::CpuStorageBytes() const
{
    const auto residency = m_cpuResidency.lock();
    return CpuStorageReservedBytes() + (residency ? residency->bytes() : 0);
}

bool FrameWorkOwner::FitCpuStorage(uint64_t bytes, bool reclaim) const
{
    const auto admission = m_admission.load(std::memory_order_acquire);
    // Explicit asset work can run between frame drivers while the engine owns its resources.
    if (admission == Admission::Closed || (admission == Admission::DriverSuspended && m_active))
        return false;
    const auto used = CpuStorageReservedBytes();
    const auto residency = m_cpuResidency.lock();
    return used <= m_cpuStorageLimit && bytes <= m_cpuStorageLimit - used &&
           (!residency || residency->fit(m_cpuStorageLimit - used - bytes, reclaim));
}

bool FrameWorkOwner::CanReserveCpuStorage(uint64_t bytes) const
{
    return FitCpuStorage(bytes, false);
}

std::shared_ptr<FrameCpuStorageClaim> FrameWorkOwner::TryReserveCpuStorage(uint64_t bytes)
{
    return FitCpuStorage(bytes, true) ? TrackOwnedCpuStorage(bytes) : nullptr;
}

std::shared_ptr<FrameCpuStorageClaim> FrameWorkOwner::TryReserveCompletionCpuStorage(uint64_t bytes)
{
    const auto used = CpuStorageBytes();
    return used <= m_cpuStorageLimit && bytes <= m_cpuStorageLimit - used ? TrackOwnedCpuStorage(bytes) : nullptr;
}

bool FrameWorkOwner::TryGrowCpuStorage(FrameCpuStorageClaim &claim, uint64_t bytes)
{
    if (claim.m_total != m_cpuStorage)
        throw std::logic_error("CPU storage can only grow on its original owner");
    if (bytes <= claim.m_bytes)
        return true;
    const auto extra = bytes - claim.m_bytes;
    if (!FitCpuStorage(extra, true))
        return false;
    m_cpuStorage->fetch_add(extra, std::memory_order_relaxed);
    claim.m_bytes = bytes;
    return true;
}

std::shared_ptr<FrameCpuStorageClaim> FrameWorkOwner::TrackOwnedCpuStorage(uint64_t bytes) const
{
    return std::shared_ptr<FrameCpuStorageClaim>(new FrameCpuStorageClaim(m_cpuStorage, bytes));
}

std::shared_ptr<FrameCpuStorageClaim> FrameWorkOwner::SplitCpuStorage(FrameCpuStorageClaim &reserved, uint64_t bytes)
{
    if (reserved.m_total != m_cpuStorage || bytes > reserved.m_bytes)
        throw std::logic_error("CPU storage split must fit a reservation from this owner");
    auto part = std::shared_ptr<FrameCpuStorageClaim>(new FrameCpuStorageClaim(m_cpuStorage, 0));
    part->m_bytes = bytes;
    reserved.m_bytes -= bytes;
    return part;
}

FrameWorkOwner::SourceId FrameWorkOwner::AddSource(FrameWorkSource source)
{
    if (!source.describe || !source.advance)
        throw std::invalid_argument("A frame work source requires describe and advance callbacks");
    const auto id = m_nextSource++;
    m_sources.push_back(std::make_shared<Source>(Source{id, std::move(source)}));
    return id;
}

void FrameWorkOwner::RemoveSource(SourceId id)
{
    m_sources.erase(std::remove_if(m_sources.begin(), m_sources.end(),
                                   [id](const auto &source) {
                                       if (source->id != id)
                                           return false;
                                       source->awake = false;
                                       return true;
                                   }),
                    m_sources.end());
}

void FrameWorkOwner::Wake(SourceId id)
{
    for (const auto &source : m_sources)
        if (source->id == id)
            source->awake = true;
}

void FrameWorkOwner::Configure(uint64_t ownerNanoseconds, uint64_t foregroundReserveNanoseconds)
{
    if (!ownerNanoseconds || ownerNanoseconds > static_cast<uint64_t>((std::numeric_limits<int64_t>::max)()) ||
        foregroundReserveNanoseconds >= ownerNanoseconds)
        throw std::invalid_argument("Frame work requires a positive signed host allowance and a smaller reserve");
    const auto overrun = [](uint64_t spent, uint64_t cap) { return spent > cap ? spent - cap : 0; };
    // A larger setting takes effect at the next frame; spent credit is never refilled.
    const auto allowance =
        !m_spentNanoseconds && !m_scopeDepth ? ownerNanoseconds : (std::min)(m_allowanceNanoseconds, ownerNanoseconds);
    m_overrunNanoseconds +=
        overrun(m_spentNanoseconds, allowance) - overrun(m_spentNanoseconds, m_allowanceNanoseconds);
    m_allowanceNanoseconds = allowance;
    m_ownerNanoseconds = ownerNanoseconds;
    m_foregroundReserveNanoseconds = foregroundReserveNanoseconds;
}

uint64_t FrameWorkOwner::BeginOpportunity()
{
    if (m_active || m_scopeDepth)
        throw std::logic_error("A frame work opportunity cannot be opened recursively");
    ++m_opportunity;
    m_active = true;
    m_late = m_serviced = m_earlyAdvanced = false;
    m_spentNanoseconds = 0;
    m_allowanceNanoseconds = m_ownerNanoseconds;
    for (auto &resource : m_resources) {
        resource.allowance = resource.limit;
        resource.admitted = 0;
        resource.frameCalibrations = 0;
    }
    m_gpuAvailability = FrameGpuAvailability::Ready;
    return m_opportunity;
}

void FrameWorkOwner::ConfigureResources(uint64_t cpuNanoseconds, uint64_t gpuNanoseconds, uint64_t ioNanoseconds,
                                        uint64_t transferBytes)
{
    const std::array limits{cpuNanoseconds, gpuNanoseconds, ioNanoseconds, transferBytes};
    if (std::any_of(limits.begin(), limits.end(), [](auto value) { return value == 0; }))
        throw std::invalid_argument("Frame resource allowances must be positive");
    for (size_t index = 0; index < limits.size(); ++index) {
        auto &resource = m_resources[index];
        resource.limit = limits[index];
        resource.allowance = m_active ? (std::min)(resource.allowance, limits[index]) : limits[index];
    }
}

const FrameWorkAllowance &FrameWorkOwner::ResourceState(FrameWorkResource resource) const noexcept
{
    return m_resources[static_cast<size_t>(resource)];
}

uint64_t FrameWorkOwner::ResourceRemaining(FrameWorkResource resource) const noexcept
{
    const auto &state = ResourceState(resource);
    if (m_serviceKind == ServiceKind::SynchronousRequired)
        return m_quantumAdmissions & (1u << static_cast<uint8_t>(resource)) ? 0 : state.limit;
    const auto remaining = state.admitted < state.allowance ? state.allowance - state.admitted : 0;
    // Interactive work uses the configured window per submission, rather
    // than waiting for background credit to refill on another host frame.
    return m_serviceKind == ServiceKind::Interactive ? (std::max)(remaining, state.limit) : remaining;
}

bool FrameWorkOwner::CanAdmitResource(FrameWorkResource resource, uint64_t prediction, bool indivisible) const noexcept
{
    const auto &state = ResourceState(resource);
    const auto remaining = ResourceRemaining(resource);
    if (m_serviceKind == ServiceKind::SynchronousRequired)
        return IsAccepting() && remaining && (!prediction || prediction <= remaining || indivisible);
    // Background work gets one unknown-duration calibration per host frame.
    // Interactive admission is limited by actual storage and queue readiness.
    return m_active && IsAccepting() && remaining &&
           (m_serviceKind == ServiceKind::Interactive ||
            (prediction ? prediction <= remaining || (indivisible && state.admitted == 0)
                        : state.frameCalibrations == 0));
}

uint64_t FrameWorkOwner::AdmitResource(FrameWorkResource resource, uint64_t prediction, bool indivisible)
{
    if (!CanAdmitResource(resource, prediction, indivisible))
        throw std::logic_error("Accepted work exceeds this host frame's resource allowance");
    return RecordAcceptedResource(resource, prediction);
}

uint64_t FrameWorkOwner::RecordAcceptedResource(FrameWorkResource resource, uint64_t prediction) noexcept
{
    ReportProgress();
    auto &state = m_resources[static_cast<size_t>(resource)];
    const auto grant = prediction ? prediction : ResourceRemaining(resource);
    const auto remaining = state.admitted < state.allowance ? state.allowance - state.admitted : 0;
    state.indivisibleOverruns += grant > remaining;
    state.admitted += grant;
    state.calibrations += prediction == 0;
    state.frameCalibrations += prediction == 0;
    if (m_serviceKind == ServiceKind::SynchronousRequired)
        m_quantumAdmissions |= 1u << static_cast<uint8_t>(resource);
    return grant;
}

void FrameWorkOwner::ObserveResource(FrameWorkResource resource, uint64_t actual, uint64_t submittedFrame,
                                     uint64_t grant) noexcept
{
    if (actual || grant)
        ReportProgress();
    auto &state = m_resources[static_cast<size_t>(resource)];
    state.observed += actual;
    // A same-frame receipt replaces its own estimate with actual work. A late
    // receipt never changes the new frame's grants. Cancellation alone returns
    // no allowance; only a real execution receipt can reconcile a grant.
    if ((m_active || m_serviceKind == ServiceKind::SynchronousRequired) && submittedFrame == m_opportunity)
        state.admitted = state.admitted - grant + actual;
}

void FrameWorkOwner::BeginScope()
{
    if (m_scopeDepth++ == 0)
        m_scopeStarted = Now();
}

uint64_t FrameWorkOwner::Now() const
{
    return m_clock ? m_clock()
                   : static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                               std::chrono::steady_clock::now().time_since_epoch())
                                               .count());
}

uint64_t FrameWorkOwner::RunningNanoseconds() const
{
    return m_scopeDepth ? Now() - m_scopeStarted : 0;
}

FrameWorkCost FrameWorkOwner::EndScope()
{
    if (!m_scopeDepth)
        throw std::logic_error("Frame work accounting scope is not open");
    const auto elapsed = RunningNanoseconds();
    if (--m_scopeDepth)
        return {};
    const auto remaining =
        m_spentNanoseconds < m_allowanceNanoseconds ? m_allowanceNanoseconds - m_spentNanoseconds : 0;
    const auto overrun = elapsed > remaining ? elapsed - remaining : 0;
    m_spentNanoseconds += elapsed;
    m_totalNanoseconds += elapsed;
    m_overrunNanoseconds += overrun;
    return {elapsed, overrun};
}

int64_t FrameWorkOwner::Remaining(bool background) const
{
    if (m_serviceKind == ServiceKind::Interactive)
        return (std::numeric_limits<int64_t>::max)();
    return static_cast<int64_t>(m_allowanceNanoseconds) - static_cast<int64_t>(m_spentNanoseconds) -
           static_cast<int64_t>(RunningNanoseconds()) -
           (background ? static_cast<int64_t>(m_foregroundReserveNanoseconds) : 0);
}

int64_t FrameWorkOwner::ServiceRemaining() const
{
    if (m_serviceKind == ServiceKind::SynchronousRequired)
        return static_cast<int64_t>(m_ownerNanoseconds) - static_cast<int64_t>(Now() - m_quantumStarted);
    return Remaining(m_serviceKind == ServiceKind::Background);
}

void FrameWorkOwner::ObserveSources()
{
    const auto sources = m_sources;
    for (const auto &source : sources)
        ObserveSource(source);
}

void FrameWorkOwner::ObserveSource(const std::shared_ptr<Source> &source)
{
    if (!source->awake || source->observedOpportunity == m_opportunity)
        return;
    source->observedOpportunity = m_opportunity;
    if (source->callbacks.observe) {
        Scope scope(*this);
        CallbackScope callback(*this, source->id);
        source->callbacks.observe();
    }
}

std::vector<FrameWorkOwner::Choice> FrameWorkOwner::DescribeSources()
{
    std::vector<Choice> choices;
    const auto sources = m_sources;
    for (const auto &source : sources) {
        auto description = DescribeSource(source, source->callbacks.interactive);
        if (!description)
            continue;
        const auto &facts = *description;
        source->awake = facts.observe || HasPending(facts);
        choices.push_back({source, facts});
    }
    return choices;
}

std::optional<FrameWorkFacts> FrameWorkOwner::DescribeSource(const std::shared_ptr<Source> &source, bool interactive)
{
    if (m_serviceKind == ServiceKind::SynchronousRequired &&
        (m_requiredSources.empty() || source->id != m_requiredSources.back()))
        return std::nullopt;
    if (!source->awake || source->observedOpportunity != m_opportunity)
        return std::nullopt;
    Scope scope(*this);
    ServiceScope service(*this, interactive);
    CallbackScope callback(*this, source->id);
    auto facts = source->callbacks.describe();
    if (!source->awake)
        return std::nullopt; // Removed during the callback.
    return facts;
}

void FrameWorkOwner::RefreshSources()
{
    const auto sources = m_sources;
    for (const auto &source : sources) {
        if (!source->awake || source->observedOpportunity != m_opportunity || !source->callbacks.refresh)
            continue;
        Scope scope(*this);
        CallbackScope callback(*this, source->id);
        source->callbacks.refresh();
    }
}

FrameWorkPeers FrameWorkOwner::PeerWork(SourceId id)
{
    Scope scope(*this);
    FrameWorkPeers peers;
    const auto sources = m_sources;
    for (const auto &source : sources) {
        if (source->id == id)
            continue;
        peers.ioServiceSerial = std::max(peers.ioServiceSerial, source->lastIoService);
        const auto facts = DescribeSource(source, source->callbacks.interactive);
        if (!facts)
            continue;
        peers.otherReady |= facts->gpu.ready || (facts->host.ready && !facts->ioReady);
        peers.ioReady |= facts->host.ready && facts->ioReady;
        peers.workerBusy |= facts->workerBusy;
        peers.gpuBusy |= facts->gpuBusy;
    }
    return peers;
}

bool FrameWorkOwner::HasPending(const FrameWorkFacts &facts)
{
    return facts.pending || facts.host.ready || facts.gpu.ready || facts.gpuBusy || facts.workerBusy;
}

bool FrameWorkOwner::HasPending(const std::vector<Choice> &choices) const
{
    return std::any_of(choices.begin(), choices.end(), [](const auto &choice) { return HasPending(choice.facts); });
}

bool FrameWorkOwner::AdvanceChoice(Choice &choice, bool gpu, bool interactive)
{
    if (!choice.source->awake || !IsAccepting())
        return false;
    Scope scope(*this);
    ServiceScope service(*this, interactive);
    CallbackScope callback(*this, choice.source->id);
    const auto progressSerial = m_progressSerial;
    const auto &candidate = gpu ? choice.facts.gpu : choice.facts.host;
    const bool ownProgress = choice.source->callbacks.advance(gpu, candidate.token);
    const bool progressed = ownProgress || progressSerial != m_progressSerial;
    m_serviced |= progressed;
    m_preferGpu = !gpu;
    if (progressed && !gpu && choice.facts.ioReady)
        choice.source->lastIoService = ++m_progressSerial;
    return progressed;
}

bool FrameWorkOwner::Select(std::vector<Choice> &choices, bool gpu, bool interactive)
{
    if (!IsAccepting() || (!interactive && Remaining(true) <= 0))
        return false;
    if (gpu && interactive && (!m_late || m_gpuAvailability == FrameGpuAvailability::ForegroundRecording))
        return false;
    const bool gpuReserved =
        std::any_of(choices.begin(), choices.end(), [](const auto &choice) { return choice.facts.reservation; });
    const bool gpuBusy =
        std::any_of(choices.begin(), choices.end(), [](const auto &choice) { return choice.facts.gpuBusy; });
    const bool gpuAvailable = m_gpuAvailability == FrameGpuAvailability::Ready && !gpuBusy;
    const auto fits = [this, interactive](const auto &candidate) {
        return candidate.ready && (interactive || candidate.hostNanoseconds <=
                                                      static_cast<uint64_t>((std::max)(int64_t{0}, Remaining(true))));
    };
    const auto gpuClaimsHostCredit = [&] {
        return m_late && gpuAvailable && std::any_of(choices.begin(), choices.end(), [&](const auto &choice) {
                   return !choice.source->callbacks.interactive && choice.facts.gpu.reserved && fits(choice.facts.gpu);
               });
    };
    if (m_late && !interactive && !gpu && m_preferGpu && gpuAvailable &&
        std::any_of(choices.begin(), choices.end(), [&](const auto &choice) {
            const auto &candidate = choice.facts.gpu;
            return !choice.source->callbacks.interactive && (!gpuReserved || candidate.reserved) && fits(candidate);
        }))
        return false;
    auto &next = gpu ? m_nextGpuSource : m_nextHostSource;
    const auto order = [gpu, next](const auto &choice) {
        const auto &candidate = gpu ? choice.facts.gpu : choice.facts.host;
        return std::pair(!candidate.reserved, std::pair(choice.source->id < next, choice.source->id));
    };
    std::stable_sort(choices.begin(), choices.end(),
                     [&](const auto &left, const auto &right) { return order(left) < order(right); });
    for (auto &choice : choices) {
        if (!IsAccepting() || (!interactive && Remaining(true) <= 0))
            break;
        if (!choice.source->awake || choice.source->callbacks.interactive != interactive)
            continue;
        auto &candidate = gpu ? choice.facts.gpu : choice.facts.host;
        // Early service precedes runtime/foreground continuation. Only an
        // explicit host reservation may spend that credit speculatively;
        // ordinary throughput uses the remaining allowance in late service.
        // Required dependencies enter through AdvanceRequired, not this lane.
        if (!m_late && !interactive && !candidate.reserved)
            continue;
        if (!fits(candidate) || (!interactive && !candidate.reserved && (gpu ? gpuReserved : gpuClaimsHostCredit())))
            continue;
        next = choice.source->id + 1;
        // An unsuccessful GPU attempt can still spend the host allowance.
        // Yield the next turn to its CPU/IO prerequisites rather than deferring
        // them again merely because this attempt retained no new state.
        if (AdvanceChoice(choice, gpu, interactive))
            return true;
        candidate.ready = false;
    }
    if (gpu)
        m_preferGpu = false;
    return false;
}

const char *FrameWorkOwner::GpuUnavailableReason() const noexcept
{
    switch (m_gpuAvailability) {
    case FrameGpuAvailability::ForegroundDeferred:
        return "foreground did not make GPU service available this opportunity";
    case FrameGpuAvailability::ForegroundRecording:
        return "foreground frame recording still owns GPU admission";
    case FrameGpuAvailability::ForegroundBacklog:
        return "older foreground frames still occupy GPU service";
    case FrameGpuAvailability::BackgroundPending:
        return "accepted background GPU work has not retired";
    case FrameGpuAvailability::Ready:
        return "";
    }
    return "";
}

bool FrameWorkOwner::Drive(const std::function<FrameGpuAvailability()> &gpuAvailability)
{
    bool first = true;
    for (;;) {
        const auto roundProgress = m_progressSerial;
        ObserveSources(); // Newly registered sources get their first host observation.
        if (m_late) {
            Scope scope(*this);
            m_gpuAvailability = gpuAvailability();
        }
        if (!first || m_late)
            RefreshSources();
        first = false;
        auto choices = DescribeSources();
        const bool pending = HasPending(choices);
        if (!IsAccepting())
            return pending;
        // Interactive consumers must make progress even after background
        // service spent this frame's allowance. Never wait for a worker or
        // fence here: an unavailable step simply stays pending.
        if ((m_late && Select(choices, true, true)) || Select(choices, false, true))
            continue;
        if (Remaining(true) <= 0) {
            if (m_late && !m_serviced)
                m_preferGpu = false;
            return pending;
        }
        if (!m_late) {
            if (!Select(choices, false))
                return pending;
            continue;
        }
        if (m_gpuAvailability == FrameGpuAvailability::Ready &&
            std::any_of(choices.begin(), choices.end(), [](const auto &choice) { return choice.facts.gpuBusy; }))
            m_gpuAvailability = FrameGpuAvailability::BackgroundPending;
        // Failed attempts are visited once per lane in this round. Repeat only
        // after retained progress, then refresh receipts before choosing again.
        if (m_gpuAvailability == FrameGpuAvailability::Ready) {
            const bool gpu = m_preferGpu;
            if (Select(choices, gpu) || Select(choices, !gpu) || (!gpu && Select(choices, false)))
                continue;
        } else {
            m_preferGpu = false;
            if (Select(choices, false))
                continue;
        }
        if (roundProgress != m_progressSerial)
            continue;
        return pending;
    }
}

bool FrameWorkOwner::AdvanceEarly()
{
    if (!m_active || m_late || m_earlyAdvanced)
        throw std::logic_error("Early work requires the current early frame opportunity");
    m_earlyAdvanced = true;
    return Drive();
}

bool FrameWorkOwner::AdvanceRequired(SourceId id, const std::function<FrameGpuAvailability()> &gpuAvailability)
{
    if (!m_active || m_late)
        throw std::logic_error("Required work must run in the current pre-render opportunity");
    return DriveRequired(id, ServiceKind::Required, gpuAvailability, {});
}

bool FrameWorkOwner::AdvanceDependency(SourceId id, const std::function<FrameGpuAvailability()> &gpuAvailability)
{
    if (!m_active)
        throw std::logic_error("An asynchronous dependency needs an active frame opportunity");
    if (!m_late)
        return DriveRequired(id, ServiceKind::Required, gpuAvailability, {});
    const auto outerGpu = m_gpuAvailability;
    struct RestoreGpu
    {
        FrameGpuAvailability &current;
        FrameGpuAvailability previous;
        ~RestoreGpu()
        {
            current = previous;
        }
    } restore{m_gpuAvailability, outerGpu};
    const auto available = [outerGpu, &gpuAvailability] {
        const auto foregroundBlocked = [](FrameGpuAvailability value) {
            return value == FrameGpuAvailability::ForegroundDeferred ||
                   value == FrameGpuAvailability::ForegroundRecording ||
                   value == FrameGpuAvailability::ForegroundBacklog;
        };
        if (foregroundBlocked(outerGpu))
            return outerGpu;
        const auto dependencyGpu = gpuAvailability();
        if (foregroundBlocked(dependencyGpu))
            return dependencyGpu;
        // Accepted compute predecessors do not prohibit this exact source's
        // ready successor. Existing queue slots/resource credits still admit it.
        return FrameGpuAvailability::Ready;
    };
    return DriveRequired(id, ServiceKind::Background, available, {});
}

void FrameWorkOwner::DrainRequired(SourceId id, const std::function<FrameGpuAvailability()> &gpuAvailability,
                                   const std::function<FrameWorkWaitResult(SourceId, uint64_t)> &waitForProgress)
{
    (void)DriveRequired(id, ServiceKind::SynchronousRequired, gpuAvailability, waitForProgress);
}

bool FrameWorkOwner::DriveRequired(SourceId id, ServiceKind kind,
                                   const std::function<FrameGpuAvailability()> &gpuAvailability,
                                   const std::function<FrameWorkWaitResult(SourceId, uint64_t)> &waitForProgress)
{
    const bool synchronous = kind == ServiceKind::SynchronousRequired;
    RequiredScope required(*this, id, synchronous);
    if (!gpuAvailability)
        throw std::invalid_argument("Required frame work needs a GPU availability callback");
    const auto found =
        std::find_if(m_sources.begin(), m_sources.end(), [id](const auto &source) { return source->id == id; });
    if (found == m_sources.end())
        throw std::invalid_argument("Required work source is not registered");
    const auto source = *found;
    const auto blocked = [id](const std::string &reason) {
        throw std::runtime_error("Synchronous frame work source " + std::to_string(id) + ": " + reason);
    };
    const auto blockedProgress = [&](const char *reason) {
        const auto gpuReason = GpuUnavailableReason();
        blocked(std::string(reason) + (*gpuReason ? std::string("; ") + gpuReason : std::string{}));
    };
    const auto requireAdmission = [&] {
        if (!IsAccepting())
            blocked(m_admission.load(std::memory_order_acquire) == Admission::DriverSuspended
                        ? "the frame driver is suspended"
                        : "work admission is stopped");
    };
    if (synchronous)
        requireAdmission();
    Scope scope(*this);
    ServiceScope service(*this, kind);
    QuantumScope quantum(*this);
    source->awake = true;
    ObserveSource(source);
    for (;;) {
        if (synchronous)
            requireAdmission();
        const auto roundProgress = m_progressSerial;
        m_gpuAvailability = gpuAvailability();
        if (source->awake && source->callbacks.refresh) {
            CallbackScope callback(*this, source->id);
            source->callbacks.refresh();
        }
        const auto description = DescribeSource(source, false);
        if (!description) {
            if (synchronous)
                blocked("the required source was removed before completion");
            return false;
        }
        Choice choice{source, *description};
        const bool pending = HasPending(choice.facts);
        source->awake = pending || choice.facts.observe;
        if (synchronous)
            requireAdmission();
        if (!pending)
            return false;
        if (!IsAccepting() || (!synchronous && ServiceRemaining() <= 0))
            return pending;
        const auto fits = [this, synchronous](const FrameWorkCandidate &candidate) {
            return candidate.ready &&
                   (synchronous ||
                    candidate.hostNanoseconds <= static_cast<uint64_t>((std::max)(int64_t{0}, ServiceRemaining())));
        };
        if (m_gpuAvailability == FrameGpuAvailability::Ready && choice.facts.gpuBusy && !choice.facts.gpu.ready)
            m_gpuAvailability = FrameGpuAvailability::BackgroundPending;
        if (m_gpuAvailability == FrameGpuAvailability::Ready && fits(choice.facts.gpu) &&
            AdvanceChoice(choice, true, false))
            continue;
        if (fits(choice.facts.host) && AdvanceChoice(choice, false, false))
            continue;
        if (!source->awake) {
            if (synchronous)
                blocked("the required source was removed before completion");
            return false;
        }
        if (roundProgress != m_progressSerial)
            continue;
        if (!synchronous)
            return pending;
        requireAdmission();
        if (choice.facts.gpu.ready && !choice.facts.gpuBusy &&
            m_gpuAvailability == FrameGpuAvailability::ForegroundRecording)
            blocked(GpuUnavailableReason());
        if (!waitForProgress)
            blockedProgress(
                "pending work has no bounded wait handler; storage or an unaccepted dependency may block progress");
        FrameWorkWaitResult result;
        {
            QuantumScope waitQuantum(*this);
            result = waitForProgress(id, m_ownerNanoseconds);
        }
        if (result == FrameWorkWaitResult::Blocked)
            blockedProgress("no accepted target ticket or explicit dependency can make progress");
        // Progress reports a retained change. Waiting reports a real bounded
        // ticket timeout. Both return through the same refresh/describe loop;
        // no peer source or global queue is driven implicitly.
    }
}

bool FrameWorkOwner::AdvanceLate(const std::function<FrameGpuAvailability()> &gpuAvailability)
{
    if (!m_active || m_late)
        throw std::logic_error("Late work can run only once in the current frame opportunity");
    m_late = true;
    return Drive(gpuAvailability);
}

void FrameWorkOwner::EndOpportunity() noexcept
{
    m_active = false;
}

void FrameWorkOwner::ClearSources()
{
    for (const auto &source : m_sources)
        source->awake = false;
    m_sources.clear();
}

} // namespace infernux
