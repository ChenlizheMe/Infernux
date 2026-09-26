#pragma once

namespace infernux
{

/// Process-wide rendezvous between Android's SurfaceView callback (UI thread)
/// and the renderer-owned Vulkan presentation lifetime (SDL thread).
///
/// Android may invoke SurfaceHolder.Callback::surfaceDestroyed before SDL's
/// queued pause event has reached the renderer. The Java callback therefore
/// waits here until the renderer has stopped presentation, drained the device,
/// and released the old swapchain/surface generation.
class AndroidPresentationLifecycle final
{
  public:
    static void ActivateRuntime() noexcept;
    static void DeactivateRuntime() noexcept;
    static void MarkPresentationSuspended() noexcept;
    static void MarkPresentationResumed() noexcept;
    static void WaitForPresentationSuspended();

    [[nodiscard]] static bool IsRuntimeActive() noexcept;
    [[nodiscard]] static bool IsPresentationSuspended() noexcept;
};

} // namespace infernux
