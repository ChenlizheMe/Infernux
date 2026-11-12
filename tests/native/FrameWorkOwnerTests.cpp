#include <core/threading/FrameWorkOwner.h>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>
#include <limits>

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
        facts.host = {0, 50, executions == 0};
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
        facts.host = {0, 1, ready};
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

int main()
{
    SameOpportunityTicks();
    ForegroundReserveAndWait();
    BlockedAndNoProgress();
    CallbackServiceBudgets();
    std::cout << "FRAME_WORK_REQUIRED_OK\n";
}
