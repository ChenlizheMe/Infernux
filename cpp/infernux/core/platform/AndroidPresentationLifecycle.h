#pragma once

#include "core/FoundationApi.h"

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
    INFERNUX_FOUNDATION_API static void ActivateRuntime() noexcept;
    INFERNUX_FOUNDATION_API static void DeactivateRuntime() noexcept;
    INFERNUX_FOUNDATION_API static void MarkPresentationSuspended() noexcept;
    INFERNUX_FOUNDATION_API static void MarkPresentationResumed() noexcept;
    INFERNUX_FOUNDATION_API static void WaitForPresentationSuspended();

    [[nodiscard]] INFERNUX_FOUNDATION_API static bool IsRuntimeActive() noexcept;
    [[nodiscard]] INFERNUX_FOUNDATION_API static bool IsPresentationSuspended() noexcept;
};

} // namespace infernux
