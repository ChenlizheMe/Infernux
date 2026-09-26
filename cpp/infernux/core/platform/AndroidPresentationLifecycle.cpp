#include "AndroidPresentationLifecycle.h"

#include <condition_variable>
#include <mutex>

namespace infernux
{
namespace
{
struct LifecycleState
{
    std::mutex mutex;
    std::condition_variable changed;
    bool runtimeActive = false;
    bool presentationSuspended = true;
};

LifecycleState &State()
{
    static LifecycleState state;
    return state;
}
} // namespace

void AndroidPresentationLifecycle::ActivateRuntime() noexcept
{
    LifecycleState &state = State();
    std::lock_guard lock(state.mutex);
    state.runtimeActive = true;
    state.presentationSuspended = false;
}

void AndroidPresentationLifecycle::DeactivateRuntime() noexcept
{
    LifecycleState &state = State();
    {
        std::lock_guard lock(state.mutex);
        state.runtimeActive = false;
        state.presentationSuspended = true;
    }
    state.changed.notify_all();
}

void AndroidPresentationLifecycle::MarkPresentationSuspended() noexcept
{
    LifecycleState &state = State();
    {
        std::lock_guard lock(state.mutex);
        state.presentationSuspended = true;
    }
    state.changed.notify_all();
}

void AndroidPresentationLifecycle::MarkPresentationResumed() noexcept
{
    LifecycleState &state = State();
    std::lock_guard lock(state.mutex);
    if (state.runtimeActive)
        state.presentationSuspended = false;
}

void AndroidPresentationLifecycle::WaitForPresentationSuspended()
{
    LifecycleState &state = State();
    std::unique_lock lock(state.mutex);
    state.changed.wait(lock, [&state]() { return !state.runtimeActive || state.presentationSuspended; });
}

bool AndroidPresentationLifecycle::IsRuntimeActive() noexcept
{
    LifecycleState &state = State();
    std::lock_guard lock(state.mutex);
    return state.runtimeActive;
}

bool AndroidPresentationLifecycle::IsPresentationSuspended() noexcept
{
    LifecycleState &state = State();
    std::lock_guard lock(state.mutex);
    return state.presentationSuspended;
}

} // namespace infernux
