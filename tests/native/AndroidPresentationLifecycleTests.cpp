#include <core/platform/AndroidPresentationLifecycle.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <thread>

namespace
{
void WaitUntil(const std::atomic_bool &value)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!value.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    assert(value.load(std::memory_order_acquire));
}
} // namespace

int main()
{
    using infernux::AndroidPresentationLifecycle;

    AndroidPresentationLifecycle::DeactivateRuntime();
    assert(!AndroidPresentationLifecycle::IsRuntimeActive());
    assert(AndroidPresentationLifecycle::IsPresentationSuspended());

    AndroidPresentationLifecycle::ActivateRuntime();
    assert(AndroidPresentationLifecycle::IsRuntimeActive());
    assert(!AndroidPresentationLifecycle::IsPresentationSuspended());

    // Showing the native window is not a lifetime boundary.  A second
    // activation is intentionally idempotent and must not publish suspension.
    AndroidPresentationLifecycle::ActivateRuntime();
    assert(AndroidPresentationLifecycle::IsRuntimeActive());
    assert(!AndroidPresentationLifecycle::IsPresentationSuspended());

    std::atomic_bool waiterEntered{false};
    std::atomic_bool waiterCompleted{false};
    std::thread surfaceDestroyThread([&]() {
        waiterEntered.store(true, std::memory_order_release);
        AndroidPresentationLifecycle::WaitForPresentationSuspended();
        waiterCompleted.store(true, std::memory_order_release);
    });
    WaitUntil(waiterEntered);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    assert(!waiterCompleted.load(std::memory_order_acquire));

    // This is the renderer's post-drain publication boundary. Only now may
    // Android's SurfaceView callback continue and release its ANativeWindow.
    AndroidPresentationLifecycle::MarkPresentationSuspended();
    surfaceDestroyThread.join();
    assert(waiterCompleted.load(std::memory_order_acquire));
    assert(AndroidPresentationLifecycle::IsPresentationSuspended());

    AndroidPresentationLifecycle::MarkPresentationResumed();
    assert(!AndroidPresentationLifecycle::IsPresentationSuspended());

    waiterEntered.store(false, std::memory_order_release);
    waiterCompleted.store(false, std::memory_order_release);
    std::thread shutdownThread([&]() {
        waiterEntered.store(true, std::memory_order_release);
        AndroidPresentationLifecycle::WaitForPresentationSuspended();
        waiterCompleted.store(true, std::memory_order_release);
    });
    WaitUntil(waiterEntered);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    assert(!waiterCompleted.load(std::memory_order_acquire));
    AndroidPresentationLifecycle::DeactivateRuntime();
    shutdownThread.join();
    assert(waiterCompleted.load(std::memory_order_acquire));
    assert(!AndroidPresentationLifecycle::IsRuntimeActive());
    assert(AndroidPresentationLifecycle::IsPresentationSuspended());
    return 0;
}
