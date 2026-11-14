#include <core/threading/FrameWorkOwner.h>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace infernux;

static void SameOpportunityTicks()
{
    uint64_t now = 0;
    FrameWorkOwner owner([&] { return now; });
    owner.Configure(100, 20);
    owner.ConfigureResources(100, 10, 100, 100);
    unsigned observations = 0, completed = 0, requested = 5, unrelated = 0;
    uint64_t grant = 0;
    bool busy = false;
    FrameWorkSource target;
    target.observe = [&] { ++observations; };
    target.refresh = [&] {
        if (!busy)
            return;
        busy = false;
        ++completed;
        owner.ObserveResource(FrameWorkResource::Gpu, 2, owner.Opportunity(), grant);
    };
    target.describe = [&] {
        FrameWorkFacts facts;
        facts.pending = completed < requested;
        facts.gpuBusy = busy;
        facts.gpu = {completed, 3, facts.pending && !busy && owner.CanAdmitResource(FrameWorkResource::Gpu, 2)};
        return facts;
    };
    target.advance = [&](bool gpu, uint64_t token) {
        assert(gpu && token == completed && !busy);
        grant = owner.AdmitResource(FrameWorkResource::Gpu, 2);
        now += 3;
        busy = true;
        return true;
    };
    // Required service must respect credit even for a source that otherwise
    // participates in the interactive editor lane.
    target.interactive = true;
    const auto id = owner.AddSource(std::move(target));
    FrameWorkSource other;
    other.observe = [&] { ++unrelated; };
    other.describe = [&] {
        ++unrelated;
        return FrameWorkFacts{};
    };
    other.advance = [&](bool, uint64_t) {
        ++unrelated;
        return false;
    };
    owner.AddSource(std::move(other));
    owner.BeginOpportunity();
    const auto opportunity = owner.Opportunity();
    const auto ready = [] { return FrameGpuAvailability::Ready; };
    assert(!owner.AdvanceRequired(id, ready));
    assert(completed == 5 && observations == 1 && unrelated == 0);
    assert(owner.ResourceState(FrameWorkResource::Gpu).admitted == 10);
    assert(owner.Remaining() == 85 && !owner.IsLate());
    requested = 6;
    assert(owner.AdvanceRequired(id, ready));
    assert(completed == 5 && observations == 1 && owner.Opportunity() == opportunity);
    assert(owner.ResourceRemaining(FrameWorkResource::Gpu) == 0);
    owner.EndOpportunity();
    owner.BeginOpportunity();
    assert(!owner.AdvanceRequired(id, ready));
    assert(completed == 6 && observations == 2 && unrelated == 0);
    owner.EndOpportunity();
}

static void ForegroundReserveAndWait()
{
    uint64_t now = 0;
    FrameWorkOwner owner([&] { return now; });
    owner.Configure(100, 60);
    unsigned observations = 0, executions = 0;
    FrameWorkSource target;
    target.observe = [&] { ++observations; };
    target.describe = [&] {
        FrameWorkFacts facts;
        facts.host = {0, 50, executions == 0, true};
        facts.pending = facts.host.ready;
        return facts;
    };
    target.advance = [&](bool gpu, uint64_t) {
        assert(!gpu);
        ++executions;
        now += 50;
        return true;
    };
    const auto id = owner.AddSource(std::move(target));
    owner.BeginOpportunity();
    assert(owner.AdvanceEarly()); // 40 background ns cannot admit the 50 ns step.
    const auto ready = [] { return FrameGpuAvailability::Ready; };
    assert(!owner.AdvanceRequired(id, ready));
    assert(executions == 1 && observations == 1 && owner.Remaining() == 50);
    executions = 0;
    {
        FrameWorkOwner::Scope wait(owner);
        now += 20;
    }
    assert(owner.AdvanceRequired(id, ready));
    assert(executions == 0 && observations == 1 && owner.Remaining() == 30);
    assert(owner.AdvanceLate(ready));
    assert(executions == 0 && owner.Remaining() == 30);
    owner.EndOpportunity();
}

static void BlockedAndNoProgress()
{
    FrameWorkOwner owner([] { return uint64_t{0}; });
    unsigned observations = 0, hostSteps = 0, gpuAttempts = 0, polls = 0;
    FrameWorkSource target;
    target.observe = [&] { ++observations; };
    target.describe = [&] {
        FrameWorkFacts facts;
        facts.host = {0, 1, hostSteps == 0};
        facts.gpu = {1, 1, hostSteps != 0};
        facts.pending = true;
        return facts;
    };
    target.advance = [&](bool gpu, uint64_t) {
        if (gpu) {
            ++gpuAttempts;
            return false; // Busy submission retains its existing cursor.
        }
        ++hostSteps;
        return true;
    };
    const auto id = owner.AddSource(std::move(target));
    owner.BeginOpportunity();
    assert(owner.AdvanceRequired(id, [&] {
        ++polls;
        return FrameGpuAvailability::ForegroundRecording;
    }));
    assert(hostSteps == 1 && gpuAttempts == 0 && observations == 1 && polls == 2);
    polls = 0;
    assert(owner.AdvanceRequired(id, [&] {
        ++polls;
        return FrameGpuAvailability::Ready;
    }));
    assert(gpuAttempts == 1 && polls == 1 && observations == 1);
    assert(!owner.IsLate());

    FrameWorkOwner::SourceId removed = 0;
    FrameWorkSource retiring;
    retiring.describe = [] {
        FrameWorkFacts facts;
        facts.host.ready = facts.gpu.ready = true;
        return facts;
    };
    retiring.advance = [&](bool gpu, uint64_t) {
        assert(gpu); // Removal during the GPU attempt must suppress the host attempt.
        owner.RemoveSource(removed);
        return false;
    };
    removed = owner.AddSource(std::move(retiring));
    assert(!owner.AdvanceRequired(removed, [] { return FrameGpuAvailability::Ready; }));

    FrameWorkSource stopping;
    stopping.describe = [] {
        FrameWorkFacts facts;
        facts.host.ready = facts.gpu.ready = true;
        return facts;
    };
    stopping.advance = [&](bool gpu, uint64_t) {
        assert(gpu); // Stop during a failed submission must not advance the host lane.
        owner.Stop();
        return false;
    };
    const auto stopped = owner.AddSource(std::move(stopping));
    assert(owner.AdvanceRequired(stopped, [] { return FrameGpuAvailability::Ready; }));
    owner.EndOpportunity();
}

static void CallbackServiceBudgets()
{
    uint64_t now = 0;
    FrameWorkOwner owner([&] { return now; });
    owner.Configure(100, 60);
    std::vector<int64_t> callbackBudgets;
    bool ready = true;
    const auto describe = [&] {
        FrameWorkFacts facts;
        // This test compares early/reserved, required and late service scopes.
        facts.host = {0, 1, ready, true};
        return facts;
    };
    FrameWorkSource target;
    target.describe = describe;
    target.advance = [&](bool gpu, uint64_t) {
        assert(!gpu);
        callbackBudgets.push_back(owner.ServiceRemaining());
        now += 10;
        ready = false;
        return true;
    };
    const auto id = owner.AddSource(std::move(target));
    owner.BeginOpportunity();
    assert(owner.ServiceRemaining() == 40 && owner.Remaining() == 100);
    assert(!owner.AdvanceEarly());
    ready = true;
    assert(!owner.AdvanceRequired(id, [] { return FrameGpuAvailability::Ready; }));
    assert(owner.ServiceRemaining() == 20 && owner.Remaining() == 80);
    ready = true;
    owner.Wake(id);
    assert(!owner.AdvanceLate([] { return FrameGpuAvailability::Ready; }));
    assert((callbackBudgets == std::vector<int64_t>{40, 90, 20}));
    assert(owner.ServiceRemaining() == 10 && owner.Remaining() == 70);
    owner.EndOpportunity();

    FrameWorkOwner interactive([] { return uint64_t{0}; });
    interactive.Configure(100, 60);
    ready = true;
    FrameWorkSource editor;
    editor.interactive = true;
    editor.describe = describe;
    editor.advance = [&](bool, uint64_t) {
        assert(interactive.ServiceRemaining() == (std::numeric_limits<int64_t>::max)());
        ready = false;
        return true;
    };
    interactive.AddSource(std::move(editor));
    interactive.BeginOpportunity();
    assert(!interactive.AdvanceEarly());
    assert(interactive.ServiceRemaining() == 40 && interactive.Remaining() == 100);
    interactive.EndOpportunity();
}

static void SynchronousOutsideAndInsideHost()
{
    uint64_t now = 0;
    FrameWorkOwner owner([&] { return now; });
    owner.Configure(10, 0);
    owner.ConfigureResources(10, 2, 10, 3);
    owner.ConfigureCpuStorage(4);
    auto retained = owner.TryReserveCpuStorage(4);
    assert(retained);
    unsigned completed = 0, requested = 4, observations = 0, unrelated = 0;
    bool busy = false;
    uint64_t gpuGrant = 0, transferGrant = 0;
    FrameWorkSource target;
    target.observe = [&] { ++observations; };
    target.refresh = [&] {
        if (!busy)
            return;
        busy = false;
        ++completed;
        owner.ObserveResource(FrameWorkResource::Gpu, 3, owner.Opportunity(), gpuGrant);
        owner.ObserveResource(FrameWorkResource::Transfer, 3, owner.Opportunity(), transferGrant);
    };
    target.describe = [&] {
        FrameWorkFacts facts;
        facts.pending = completed < requested;
        facts.gpuBusy = busy;
        facts.gpu.ready = facts.pending && !busy && owner.CanAdmitResource(FrameWorkResource::Gpu, 3, true) &&
                          owner.CanAdmitResource(FrameWorkResource::Transfer, 3);
        return facts;
    };
    target.advance = [&](bool gpu, uint64_t) {
        assert(gpu && !busy && owner.ServiceRemaining() == 10);
        if (owner.IsActive())
            assert(owner.Remaining() < 0);
        assert(owner.ResourceRemaining(FrameWorkResource::Gpu) == 2);
        assert(!owner.CanAdmitResource(FrameWorkResource::Gpu, 3));
        assert(!owner.TryReserveCpuStorage(1)); // Synchronous service never bypasses hard residency.
        gpuGrant = owner.AdmitResource(FrameWorkResource::Gpu, 3, true);
        assert(!owner.CanAdmitResource(FrameWorkResource::Gpu, 1, true));
        transferGrant = owner.AdmitResource(FrameWorkResource::Transfer, 3);
        assert(owner.ResourceRemaining(FrameWorkResource::Transfer) == 0);
        now += 6;
        assert(owner.ServiceRemaining() == 4);
        busy = true;
        return true;
    };
    // The nested bool service scope must retain synchronous service instead of
    // switching this source to the interactive lane's unbounded host budget.
    target.interactive = true;
    const auto id = owner.AddSource(std::move(target));
    FrameWorkSource other;
    other.observe = [&] { ++unrelated; };
    other.describe = [&] {
        ++unrelated;
        return FrameWorkFacts{};
    };
    other.advance = [&](bool, uint64_t) {
        ++unrelated;
        return false;
    };
    owner.AddSource(std::move(other));
    const auto ready = [] { return FrameGpuAvailability::Ready; };
    owner.DrainRequired(id, ready);
    assert(!owner.IsActive() && owner.Opportunity() == 0 && completed == 4 && observations == 1 && unrelated == 0);
    assert(owner.TotalNanoseconds() == 24 && owner.OverrunNanoseconds() == 14);
    assert(owner.ResourceState(FrameWorkResource::Gpu).allowance == 2);
    assert(owner.ResourceState(FrameWorkResource::Gpu).admitted == 12);
    assert(owner.ResourceState(FrameWorkResource::Gpu).observed == 12);
    assert(owner.ResourceState(FrameWorkResource::Transfer).admitted == 12);
    requested = 5;
    owner.DrainRequired(id, ready);
    assert(completed == 5 && observations == 1 && owner.Opportunity() == 0 && unrelated == 0);

    owner.BeginOpportunity();
    const auto opportunity = owner.Opportunity();
    owner.AdmitResource(FrameWorkResource::Gpu, 2);
    owner.AdmitResource(FrameWorkResource::Transfer, 3);
    {
        FrameWorkOwner::Scope spent(owner);
        now += 12;
    }
    assert(owner.Remaining() < 0 && owner.ResourceRemaining(FrameWorkResource::Gpu) == 0);
    requested = 7;
    owner.DrainRequired(id, ready);
    assert(owner.IsActive() && !owner.IsLate() && owner.Opportunity() == opportunity);
    assert(completed == 7 && observations == 2 && unrelated == 0);
    assert(owner.ResourceState(FrameWorkResource::Gpu).allowance == 2);
    assert(owner.ResourceState(FrameWorkResource::Gpu).admitted == 8);
    assert(owner.ResourceState(FrameWorkResource::Transfer).admitted == 9);
    requested = 8;
    owner.DrainRequired(id, ready);
    assert(completed == 8 && observations == 2 && owner.Opportunity() == opportunity && unrelated == 0);
    owner.EndOpportunity();
}

static void SynchronousWaitAndDiagnostics()
{
    uint64_t now = 0;
    FrameWorkOwner owner([&] { return now; });
    owner.Configure(10, 0);
    bool signaled = false, complete = false;
    FrameWorkSource target;
    target.refresh = [&] { complete |= signaled; };
    target.describe = [&] {
        FrameWorkFacts facts;
        facts.pending = facts.gpuBusy = !complete;
        return facts;
    };
    target.advance = [](bool, uint64_t) {
        assert(false);
        return false;
    };
    const auto id = owner.AddSource(std::move(target));
    unsigned waits = 0;
    owner.DrainRequired(
        id, [] { return FrameGpuAvailability::BackgroundPending; },
        [&](FrameWorkOwner::SourceId source, uint64_t maximum) {
            assert(source == id && maximum == 10 && !owner.IsActive());
            now += maximum;
            if (++waits == 1)
                return FrameWorkWaitResult::Waiting;
            signaled = true;
            return FrameWorkWaitResult::Progress;
        });
    assert(complete && waits == 2 && owner.TotalNanoseconds() == 20);

    const auto expectBlocked = [&](const auto &operation, const char *reason) {
        bool rejected = false;
        try {
            operation();
        } catch (const std::runtime_error &error) {
            rejected = std::string(error.what()).find(reason) != std::string::npos;
        }
        assert(rejected);
    };
    complete = signaled = false;
    expectBlocked([&] { owner.DrainRequired(id, [] { return FrameGpuAvailability::Ready; }); }, "wait handler");
    expectBlocked(
        [&] {
            owner.DrainRequired(
                id, [] { return FrameGpuAvailability::Ready; },
                [](auto, auto) { return FrameWorkWaitResult::Blocked; });
        },
        "accepted target");
    owner.SuspendDriver();
    expectBlocked([&] { owner.DrainRequired(id, [] { return FrameGpuAvailability::Ready; }); }, "suspended");
    owner.Reopen();
    owner.Stop();
    expectBlocked([&] { owner.DrainRequired(id, [] { return FrameGpuAvailability::Ready; }); }, "stopped");
    owner.Reopen();

    FrameWorkSource foreground;
    foreground.describe = [] {
        FrameWorkFacts facts;
        facts.pending = facts.gpu.ready = true;
        return facts;
    };
    foreground.advance = [](bool, uint64_t) {
        assert(false);
        return false;
    };
    const auto foregroundId = owner.AddSource(std::move(foreground));
    expectBlocked(
        [&] {
            owner.DrainRequired(
                foregroundId, [] { return FrameGpuAvailability::ForegroundRecording; },
                [](auto, auto) {
                    assert(false);
                    return FrameWorkWaitResult::Blocked;
                });
        },
        "foreground frame recording");

    bool admitted = false;
    auto pressure = FrameGpuAvailability::ForegroundBacklog;
    FrameWorkSource waitingForFrame;
    waitingForFrame.describe = [&] {
        FrameWorkFacts facts;
        facts.pending = facts.gpu.ready = !admitted;
        return facts;
    };
    waitingForFrame.advance = [&](bool gpu, uint64_t) {
        assert(gpu);
        admitted = true;
        return true;
    };
    const auto waitingId = owner.AddSource(std::move(waitingForFrame));
    const auto releaseExplicitDependency = [&](auto source, auto maximum) {
        assert(source == waitingId);
        now += maximum;
        pressure = FrameGpuAvailability::Ready;
        return FrameWorkWaitResult::Progress;
    };
    owner.DrainRequired(waitingId, [&] { return pressure; }, releaseExplicitDependency);
    assert(admitted);
    admitted = false;
    pressure = FrameGpuAvailability::ForegroundDeferred;
    owner.DrainRequired(waitingId, [&] { return pressure; }, releaseExplicitDependency);
    assert(admitted);
}

static void SynchronousDependenciesAndReentry()
{
    FrameWorkOwner owner([] { return uint64_t{0}; });
    const auto ready = [] { return FrameGpuAvailability::Ready; };
    bool dependencyComplete = false, complete = false;
    FrameWorkSource target;
    target.refresh = [&] { complete |= dependencyComplete; };
    target.describe = [&] {
        FrameWorkFacts facts;
        facts.pending = !complete;
        return facts;
    };
    target.advance = [](bool, uint64_t) {
        assert(false);
        return false;
    };
    const auto id = owner.AddSource(std::move(target));
    FrameWorkSource dependency;
    dependency.describe = [&] {
        FrameWorkFacts facts;
        facts.host.ready = facts.pending = !dependencyComplete;
        return facts;
    };
    dependency.advance = [&](bool gpu, uint64_t) {
        assert(!gpu);
        dependencyComplete = true;
        return true;
    };
    const auto dependencyId = owner.AddSource(std::move(dependency));
    owner.DrainRequired(id, ready, [&](auto source, auto) {
        assert(source == id);
        owner.DrainRequired(dependencyId, ready);
        return FrameWorkWaitResult::Progress;
    });
    assert(complete && dependencyComplete);

    bool cycleRejected = false;
    complete = dependencyComplete = false;
    FrameWorkSource blocked;
    blocked.describe = [] {
        FrameWorkFacts facts;
        facts.pending = true;
        return facts;
    };
    blocked.advance = [](bool, uint64_t) { return false; };
    const auto blockedId = owner.AddSource(std::move(blocked));
    try {
        owner.DrainRequired(id, ready, [&](auto, auto) {
            owner.DrainRequired(blockedId, ready, [&](auto, auto) {
                owner.DrainRequired(id, ready);
                return FrameWorkWaitResult::Progress;
            });
            return FrameWorkWaitResult::Progress;
        });
    } catch (const std::logic_error &error) {
        cycleRejected = std::string(error.what()).find("cycle") != std::string::npos;
    }
    assert(cycleRejected);

    bool callbackComplete = false;
    FrameWorkSource reentrant;
    reentrant.describe = [&] {
        FrameWorkFacts facts;
        facts.host.ready = !callbackComplete;
        return facts;
    };
    reentrant.advance = [&](bool, uint64_t) {
        owner.DrainRequired(dependencyId, ready);
        callbackComplete = true;
        return true;
    };
    const auto reentrantId = owner.AddSource(std::move(reentrant));
    owner.DrainRequired(reentrantId, ready);
    assert(callbackComplete && dependencyComplete);

    FrameWorkOwner::SourceId callbackA = 0, callbackB = 0;
    const auto cycleFacts = [] {
        FrameWorkFacts facts;
        facts.host.ready = true;
        return facts;
    };
    FrameWorkSource first;
    first.describe = cycleFacts;
    first.advance = [&](bool, uint64_t) {
        owner.DrainRequired(callbackB, ready);
        return true;
    };
    callbackA = owner.AddSource(std::move(first));
    FrameWorkSource second;
    second.describe = cycleFacts;
    second.advance = [&](bool, uint64_t) {
        owner.DrainRequired(callbackA, ready);
        return true;
    };
    callbackB = owner.AddSource(std::move(second));
    bool reentryRejected = false;
    try {
        owner.DrainRequired(callbackA, ready);
    } catch (const std::logic_error &error) {
        reentryRejected = std::string(error.what()).find("cycle") != std::string::npos;
    }
    assert(reentryRejected);

    FrameWorkOwner::SourceId removed = 0;
    FrameWorkSource retiring;
    retiring.describe = [] {
        FrameWorkFacts facts;
        facts.gpu.ready = facts.pending = true;
        return facts;
    };
    retiring.advance = [&](bool, uint64_t) {
        owner.RemoveSource(removed);
        return false;
    };
    removed = owner.AddSource(std::move(retiring));
    bool removalRejected = false;
    try {
        owner.DrainRequired(removed, ready);
    } catch (const std::runtime_error &error) {
        removalRejected = std::string(error.what()).find("removed") != std::string::npos;
    }
    assert(removalRejected);

    // All nested guards unwind after failure; an unrelated explicit request
    // can still finish through the same owner without opening a host frame.
    owner.DrainRequired(dependencyId, ready);
    assert(dependencyComplete && !owner.IsActive() && owner.Opportunity() == 0);
}

static void ForegroundBeforeBackgroundWithNoReserve()
{
    uint64_t now = 0;
    FrameWorkOwner owner([&] { return now; });
    owner.Configure(100, 0);
    unsigned completed[2]{}, foreground = 0;
    std::vector<unsigned> order;
    for (unsigned index = 0; index < 2; ++index) {
        FrameWorkSource source;
        source.describe = [&, index] {
            FrameWorkFacts facts;
            facts.host = {index, 0, completed[index] < 3, false};
            facts.pending = facts.host.ready;
            return facts;
        };
        source.advance = [&, index](bool gpu, uint64_t token) {
            assert(!gpu && token == index);
            order.push_back(index);
            ++completed[index];
            now += 20; // One indivisible, actually charged background quantum.
            return true;
        };
        owner.AddSource(std::move(source));
    }
    const auto ready = [] { return FrameGpuAvailability::Ready; };
    const auto runtimeContinuation = [&] {
        FrameWorkOwner::Scope runtime(owner);
        assert(owner.Remaining() == 100);
        ++foreground;
        now += 10;
    };
    owner.BeginOpportunity();
    assert(owner.AdvanceEarly());
    assert(order.empty() && owner.Remaining() == 100);
    runtimeContinuation();
    assert(owner.AdvanceLate(ready));
    // Late repeats while credit remains, with source round-robin, and stops
    // after one indivisible overrun instead of admitting another quantum.
    assert((order == std::vector<unsigned>{0, 1, 0, 1, 0}));
    assert(owner.Remaining() == -10 && owner.OverrunNanoseconds() == 10);
    owner.EndOpportunity();
    owner.BeginOpportunity();
    assert(owner.AdvanceEarly());
    assert(order.size() == 5);
    runtimeContinuation();
    assert(!owner.AdvanceLate(ready));
    assert(foreground == 2 && completed[0] == 3 && completed[1] == 3);
    assert(order.back() == 1 && owner.Remaining() == 70);
    owner.EndOpportunity();
}

static void EarlyPrerequisitesAndGpuPreference()
{
    uint64_t now = 0;
    FrameWorkOwner owner([&] { return now; });
    owner.Configure(100, 0);
    unsigned reserved = 0, background = 0, gpuSteps = 0;
    FrameWorkSource gpu;
    gpu.describe = [&] {
        FrameWorkFacts facts;
        facts.gpu = {0, 0, gpuSteps == 0, false};
        facts.pending = facts.gpu.ready;
        return facts;
    };
    gpu.advance = [&](bool isGpu, uint64_t) {
        assert(isGpu);
        ++gpuSteps;
        ++now;
        return true;
    };
    const auto gpuId = owner.AddSource(std::move(gpu));
    FrameWorkSource prerequisite;
    prerequisite.describe = [&] {
        FrameWorkFacts facts;
        facts.host = {reserved, 0, reserved < 2, true};
        facts.pending = facts.host.ready;
        return facts;
    };
    prerequisite.advance = [&](bool isGpu, uint64_t token) {
        assert(!isGpu && token == reserved);
        ++reserved;
        now += 2;
        return true;
    };
    owner.AddSource(std::move(prerequisite));
    FrameWorkSource throughput;
    throughput.describe = [&] {
        FrameWorkFacts facts;
        facts.host = {0, 0, background == 0, false};
        facts.pending = facts.host.ready;
        // A source-wide GPU reservation does not make an arbitrary selected
        // host candidate a known foreground prerequisite.
        facts.reservation = true;
        return facts;
    };
    throughput.advance = [&](bool isGpu, uint64_t) {
        assert(!isGpu);
        ++background;
        ++now;
        return true;
    };
    owner.AddSource(std::move(throughput));
    owner.BeginOpportunity();
    assert(owner.AdvanceEarly());
    assert(reserved == 2 && background == 0 && gpuSteps == 0);
    // A real consumer can still explicitly service its GPU dependency before
    // the foreground phase; this does not open a new opportunity or drain peers.
    const auto opportunity = owner.Opportunity();
    const auto ready = [] { return FrameGpuAvailability::Ready; };
    assert(!owner.AdvanceRequired(gpuId, ready));
    assert(gpuSteps == 1 && background == 0 && !owner.IsLate());
    assert(!owner.AdvanceLate(ready));
    assert(background == 1 && owner.Opportunity() == opportunity);
    owner.EndOpportunity();
}

static void EarlyObservationAndSameHostLateIo()
{
    uint64_t now = 0;
    FrameWorkOwner owner([&] { return now; });
    owner.Configure(100, 0);
    unsigned observed = 0, refreshed = 0;
    bool started = false, complete = false;
    FrameWorkSource io;
    io.observe = [&] { ++observed; };
    io.refresh = [&] {
        ++refreshed;
        if (started)
            complete = true;
    };
    io.describe = [&] {
        FrameWorkFacts facts;
        facts.pending = !complete;
        facts.host = {0, 0, !started, false};
        facts.ioReady = facts.host.ready;
        return facts;
    };
    io.advance = [&](bool gpu, uint64_t) {
        assert(!gpu && !started);
        started = true;
        now += 2;
        return true;
    };
    owner.AddSource(std::move(io));
    owner.BeginOpportunity();
    const auto opportunity = owner.Opportunity();
    assert(owner.AdvanceEarly());
    assert(observed == 1 && !started);
    assert(!owner.AdvanceLate([] { return FrameGpuAvailability::Ready; }));
    assert(started && complete && refreshed >= 2 && owner.Opportunity() == opportunity);
    owner.EndOpportunity();
}

static void LateDependencySharesBackgroundCredit()
{
    uint64_t now = 0;
    FrameWorkOwner owner([&] { return now; });
    owner.Configure(100, 60);
    owner.ConfigureResources(100, 10, 100, 100);
    bool inDependency = false, parentComplete = false;
    unsigned childSteps = 0;
    FrameWorkSource child;
    child.describe = [&] {
        FrameWorkFacts facts;
        facts.pending = childSteps < 2;
        facts.host = {childSteps, 20,
                      inDependency && facts.pending && owner.CanAdmitResource(FrameWorkResource::Cpu, 3) &&
                          owner.CanAdmitResource(FrameWorkResource::Gpu, 2)};
        return facts;
    };
    child.advance = [&](bool gpu, uint64_t token) {
        assert(!gpu && token == childSteps);
        assert(owner.ServiceRemaining() == 30);
        owner.AdmitResource(FrameWorkResource::Cpu, 3);
        owner.AdmitResource(FrameWorkResource::Gpu, 2);
        ++childSteps;
        now += 20;
        return true;
    };
    const auto childId = owner.AddSource(std::move(child));
    const auto ready = [] { return FrameGpuAvailability::Ready; };
    FrameWorkSource parent;
    parent.describe = [&] {
        FrameWorkFacts facts;
        facts.host = {0, 0, !parentComplete};
        facts.pending = facts.host.ready;
        return facts;
    };
    parent.advance = [&](bool gpu, uint64_t) {
        assert(!gpu && owner.IsLate() && owner.ServiceRemaining() == 40);
        now += 10;
        owner.AdmitResource(FrameWorkResource::Cpu, 2);
        owner.AdmitResource(FrameWorkResource::Gpu, 6);
        inDependency = true;
        // Parent + one child spend 30 of the same 40 background ns. The next
        // 20 ns child cannot borrow the 60 ns foreground reserve.
        assert(owner.AdvanceDependency(childId, ready));
        inDependency = false;
        assert(childSteps == 1 && owner.ServiceRemaining() == 10 && owner.Remaining() == 70);
        parentComplete = true;
        return true;
    };
    owner.AddSource(std::move(parent));
    owner.BeginOpportunity();
    const auto opportunity = owner.Opportunity();
    assert(owner.AdvanceEarly());
    assert(owner.AdvanceLate(ready));
    assert(childSteps == 1 && parentComplete && owner.Remaining(true) == 10);
    assert(owner.ResourceState(FrameWorkResource::Cpu).admitted == 5);
    assert(owner.ResourceState(FrameWorkResource::Gpu).admitted == 8);
    assert(owner.ResourceRemaining(FrameWorkResource::Cpu) == 95);
    assert(owner.ResourceRemaining(FrameWorkResource::Gpu) == 2);
    const auto serial = owner.ProgressSerial();
    inDependency = true;
    assert(owner.AdvanceDependency(childId, ready));
    inDependency = false;
    assert(childSteps == 1 && now == 30 && owner.Remaining(true) == 10);
    assert(owner.ProgressSerial() == serial && owner.Opportunity() == opportunity);
    assert(owner.ResourceState(FrameWorkResource::Cpu).admitted == 5);
    assert(owner.ResourceState(FrameWorkResource::Gpu).admitted == 8);
    owner.EndOpportunity();
}

static void LateDependencyPreservesForegroundPressure()
{
    for (const auto pressure : {FrameGpuAvailability::ForegroundDeferred, FrameGpuAvailability::ForegroundRecording,
                                FrameGpuAvailability::ForegroundBacklog}) {
        for (const bool outerBlocked : {false, true}) {
            FrameWorkOwner owner([] { return uint64_t{0}; });
            owner.Configure(100, 0);
            bool inDependency = false, parentComplete = false;
            unsigned gpuAttempts = 0;
            FrameWorkSource child;
            child.describe = [&] {
                FrameWorkFacts facts;
                facts.pending = true;
                facts.gpu = {0, 0, inDependency};
                facts.gpuBusy = inDependency;
                return facts;
            };
            child.advance = [&](bool gpu, uint64_t) {
                assert(gpu);
                ++gpuAttempts;
                return false;
            };
            const auto childId = owner.AddSource(std::move(child));
            const auto outer = [&] { return outerBlocked ? pressure : FrameGpuAvailability::Ready; };
            const auto dependency = [&] { return outerBlocked ? FrameGpuAvailability::Ready : pressure; };
            FrameWorkSource parent;
            parent.describe = [&] {
                FrameWorkFacts facts;
                facts.host = {0, 0, !parentComplete};
                facts.pending = facts.host.ready;
                return facts;
            };
            parent.advance = [&](bool gpu, uint64_t) {
                assert(!gpu && owner.IsLate());
                const std::string outerReason = owner.GpuUnavailableReason();
                const auto serial = owner.ProgressSerial();
                const auto opportunity = owner.Opportunity();
                inDependency = true;
                assert(owner.AdvanceDependency(childId, dependency)); // Pending is not progress.
                inDependency = false;
                assert(gpuAttempts == 0 && owner.ProgressSerial() == serial);
                assert(owner.Remaining() == 100 && owner.Opportunity() == opportunity);
                // Check inside the parent, before another owner round could
                // refresh and accidentally hide a leaked child GPU decision.
                assert(std::string(owner.GpuUnavailableReason()) == outerReason);
                assert(owner.IsGpuForegroundBlocked() == outerBlocked);
                parentComplete = true;
                return true;
            };
            owner.AddSource(std::move(parent));
            owner.BeginOpportunity();
            assert(owner.AdvanceEarly());
            assert(owner.AdvanceLate(outer));
            assert(parentComplete && gpuAttempts == 0);
            owner.EndOpportunity();
        }
    }
}

static void LateDependencySuccessorAndBlockedProgress()
{
    uint64_t now = 0;
    FrameWorkOwner owner([&] { return now; });
    owner.Configure(100, 0);
    owner.ConfigureResources(100, 12, 100, 100);
    bool inDependency = false, parentComplete = false;
    unsigned attempts = 0, accepted = 0;
    FrameWorkSource child;
    child.describe = [&] {
        FrameWorkFacts facts;
        facts.pending = accepted < 2;
        // One accepted predecessor can coexist with its ready exact successor.
        facts.gpuBusy = inDependency && facts.pending;
        facts.gpu = {accepted, 1, inDependency && facts.pending && owner.CanAdmitResource(FrameWorkResource::Gpu, 2)};
        return facts;
    };
    child.advance = [&](bool gpu, uint64_t token) {
        assert(gpu && token == accepted);
        ++attempts;
        ++now; // Even a rejected submission costs actual shared host time.
        if (accepted)
            return false;
        owner.AdmitResource(FrameWorkResource::Gpu, 2);
        ++accepted;
        now += 2;
        return true;
    };
    const auto childId = owner.AddSource(std::move(child));
    const auto ready = [] { return FrameGpuAvailability::Ready; };
    FrameWorkSource parent;
    parent.describe = [&] {
        FrameWorkFacts facts;
        facts.host = {0, 0, !parentComplete};
        facts.pending = facts.host.ready;
        return facts;
    };
    parent.advance = [&](bool gpu, uint64_t) {
        assert(!gpu && owner.IsLate());
        const std::string outerReason = owner.GpuUnavailableReason();
        assert(!outerReason.empty() && !owner.IsGpuForegroundBlocked());
        now += 5;
        owner.AdmitResource(FrameWorkResource::Gpu, 8);
        const auto opportunity = owner.Opportunity();
        inDependency = true;
        assert(owner.AdvanceDependency(childId, ready));
        assert(accepted == 1 && attempts == 2 && now == 9);
        assert(owner.ResourceState(FrameWorkResource::Gpu).admitted == 10);
        assert(owner.ResourceRemaining(FrameWorkResource::Gpu) == 2);
        assert(std::string(owner.GpuUnavailableReason()) == outerReason);
        const auto serial = owner.ProgressSerial();
        const auto remaining = owner.ServiceRemaining();
        assert(owner.AdvanceDependency(childId, ready));
        inDependency = false;
        // The still-blocked successor is attempted once, honestly returns no
        // progress, keeps its GPU credit and cannot spin/refill this Host.
        assert(accepted == 1 && attempts == 3 && now == 10);
        assert(owner.ProgressSerial() == serial && owner.Opportunity() == opportunity);
        assert(owner.ServiceRemaining() == remaining - 1);
        assert(owner.ResourceState(FrameWorkResource::Gpu).admitted == 10);
        assert(std::string(owner.GpuUnavailableReason()) == outerReason);
        parentComplete = true;
        return true;
    };
    owner.AddSource(std::move(parent));
    owner.BeginOpportunity();
    assert(owner.AdvanceEarly());
    assert(owner.AdvanceLate([] { return FrameGpuAvailability::BackgroundPending; }));
    assert(parentComplete && accepted == 1 && attempts == 3 && owner.Remaining() == 90);
    owner.EndOpportunity();
}

int main()
{
    SameOpportunityTicks();
    ForegroundReserveAndWait();
    BlockedAndNoProgress();
    CallbackServiceBudgets();
    SynchronousOutsideAndInsideHost();
    SynchronousWaitAndDiagnostics();
    SynchronousDependenciesAndReentry();
    ForegroundBeforeBackgroundWithNoReserve();
    EarlyPrerequisitesAndGpuPreference();
    EarlyObservationAndSameHostLateIo();
    LateDependencySharesBackgroundCredit();
    LateDependencyPreservesForegroundPressure();
    LateDependencySuccessorAndBlockedProgress();
    std::cout << "FRAME_WORK_REQUIRED_OK\n";
}
