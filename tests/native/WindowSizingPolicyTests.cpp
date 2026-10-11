#include <platform/window/WindowPresentationPolicy.h>
#include <platform/window/WindowSizingPolicy.h>

#include <cassert>
#include <stdexcept>
#include <string_view>
#include <vector>

static_assert(!infernux::ShouldRecreateAndroidSurfaceForPixelExtent(false, 0, 0, 3200, 1440));
static_assert(!infernux::ShouldRecreateAndroidSurfaceForPixelExtent(true, 3200, 1440, 3200, 1440));
static_assert(infernux::ShouldRecreateAndroidSurfaceForPixelExtent(true, 3200, 1440, 1440, 3200));
static_assert(!infernux::ShouldRecreateAndroidSurfaceForPixelExtent(true, 3200, 1440, 0, 0));

int main()
{
    using infernux::ResolveEditorInitialWindowSize;

    const auto reference = ResolveEditorInitialWindowSize(1600, 900, 1920, 1080);
    assert(reference.width == 1600);
    assert(reference.height == 900);

    const auto constrained = ResolveEditorInitialWindowSize(1600, 900, 1024, 768);
    assert(constrained.width == 921);
    assert(constrained.height == 691);

    const auto compact = ResolveEditorInitialWindowSize(800, 600, 2560, 1440);
    assert(compact.width == 800);
    assert(compact.height == 600);

    bool rejected = false;
    try {
        (void)ResolveEditorInitialWindowSize(1600, 900, 0, 768);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);

    const auto ordinaryWayland = infernux::ResolveWindowPresentationPolicy(false, false, "wayland");
    assert(ordinaryWayland.focusable);
    assert(ordinaryWayland.activateWhenShown);
    assert(!ordinaryWayland.showBeforeSurface);
    assert(ordinaryWayland.syncInitialMaximize);
    assert(!ordinaryWayland.createMaximized);

    const auto ordinaryWindows = infernux::ResolveWindowPresentationPolicy(false, false, "windows");
    assert(ordinaryWindows.focusable);
    assert(ordinaryWindows.activateWhenShown);
    assert(!ordinaryWindows.revealAfterFirstPresentation);
    assert(!ordinaryWindows.activateAfterFirstPresentation);
    assert(!ordinaryWindows.showBeforeSurface);
    assert(ordinaryWindows.syncInitialMaximize);
    assert(ordinaryWindows.createMaximized);

    const auto ordinaryX11 = infernux::ResolveWindowPresentationPolicy(false, false, "x11");
    assert(ordinaryX11.focusable);
    assert(!ordinaryX11.activateWhenShown);
    assert(ordinaryX11.showBeforeSurface);
    assert(!ordinaryX11.syncInitialMaximize);
    assert(!ordinaryX11.createMaximized);

    // A Player owns its presentation through BuildManifest.json. Platform
    // editor policy must never maximize it or replace its requested window
    // extent, regardless of the desktop window system.
    const auto playerWindows = infernux::ResolveWindowPresentationPolicy(true, false, "windows");
    assert(playerWindows.focusable);
    assert(!playerWindows.activateWhenShown);
    assert(playerWindows.revealAfterFirstPresentation);
    assert(playerWindows.activateAfterFirstPresentation);
    assert(!playerWindows.createMaximized);
    assert(!playerWindows.showBeforeSurface);

    const auto playerX11 = infernux::ResolveWindowPresentationPolicy(true, false, "x11");
    assert(playerX11.focusable);
    assert(!playerX11.createMaximized);
    assert(playerX11.showBeforeSurface);

    const auto playerWayland = infernux::ResolveWindowPresentationPolicy(true, false, "wayland");
    assert(playerWayland.focusable);
    assert(!playerWayland.createMaximized);
    assert(!playerWayland.showBeforeSurface);

    const auto controlledWayland = infernux::ResolveWindowPresentationPolicy(true, true, "wayland");
    assert(!controlledWayland.focusable);
    assert(!controlledWayland.activateWhenShown);
    assert(controlledWayland.showBeforeSurface);

    const auto controlledX11 = infernux::ResolveWindowPresentationPolicy(true, true, "x11");
    assert(controlledX11.focusable);
    assert(!controlledX11.activateWhenShown);
    assert(controlledX11.showBeforeSurface);

    const auto controlledWindows = infernux::ResolveWindowPresentationPolicy(true, true, "windows");
    assert(controlledWindows.focusable);
    assert(!controlledWindows.activateWhenShown);
    assert(controlledWindows.revealAfterFirstPresentation);
    assert(!controlledWindows.activateAfterFirstPresentation);
    assert(!controlledWindows.showBeforeSurface);
    assert(!controlledWindows.createMaximized);

    const std::vector<std::string_view> waylandExtensions = {"VK_KHR_surface", "VK_KHR_wayland_surface"};
    const std::vector<std::string_view> xlibExtensions = {"VK_KHR_surface", "VK_KHR_xlib_surface"};
    const std::vector<std::string_view> xcbExtensions = {"VK_KHR_surface", "VK_KHR_xcb_surface"};
    const std::vector<std::string_view> surfaceOnlyExtensions = {"VK_KHR_surface"};
    assert(infernux::ValidateVulkanWindowExtensions("wayland", waylandExtensions));
    assert(infernux::ValidateVulkanWindowExtensions("x11", xlibExtensions));
    assert(infernux::ValidateVulkanWindowExtensions("x11", xcbExtensions));
    assert(!infernux::ValidateVulkanWindowExtensions("wayland", xlibExtensions));
    assert(!infernux::ValidateVulkanWindowExtensions("x11", surfaceOnlyExtensions));
    assert(infernux::ValidateVulkanWindowExtensions("custom", surfaceOnlyExtensions));

    const std::vector<std::string_view> allX11Extensions = {"VK_KHR_surface", "VK_KHR_xlib_surface",
                                                            "VK_KHR_xcb_surface"};
    const auto resolvedX11 = infernux::ResolveVulkanWindowExtensions("x11", xcbExtensions, allX11Extensions);
    assert(resolvedX11.size() == 3);
    assert(resolvedX11[0] == "VK_KHR_surface");
    assert(resolvedX11[1] == "VK_KHR_xcb_surface");
    assert(resolvedX11[2] == "VK_KHR_xlib_surface");

    using infernux::ShouldSuspendWindowRendering;
    using infernux::WindowVisibility;
    assert(!ShouldSuspendWindowRendering(WindowVisibility::Visible, false, false));
    assert(!ShouldSuspendWindowRendering(WindowVisibility::Occluded, false, false));
    assert(ShouldSuspendWindowRendering(WindowVisibility::Minimized, false, false));
    assert(ShouldSuspendWindowRendering(WindowVisibility::Visible, true, false));
    assert(ShouldSuspendWindowRendering(WindowVisibility::Visible, false, true));

    using infernux::ShouldRecreateAndroidSurfaceForPixelExtent;
    assert(!ShouldRecreateAndroidSurfaceForPixelExtent(false, 0, 0, 3200, 1440));
    assert(!ShouldRecreateAndroidSurfaceForPixelExtent(true, 3200, 1440, 3200, 1440));
    assert(ShouldRecreateAndroidSurfaceForPixelExtent(true, 3200, 1440, 1440, 3200));
    assert(!ShouldRecreateAndroidSurfaceForPixelExtent(true, 3200, 1440, 0, 0));
    return 0;
}
