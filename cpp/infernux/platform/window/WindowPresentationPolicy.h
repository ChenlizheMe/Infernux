#pragma once

#include <string_view>
#include <vector>

namespace infernux
{

struct WindowPresentationPolicy
{
    bool focusable = true;
    bool activateWhenShown = true;
    bool showBeforeSurface = false;
};

enum class WindowVisibility
{
    Visible,
    Occluded,
    Minimized,
};

/// Occlusion is only a desktop-compositor visibility hint. The swapchain is
/// still valid and off-screen render targets (including MCP captures) must
/// keep advancing. Only an actual minimize, mobile background transition, or
/// pending native-surface replacement suspends presentation.
inline constexpr bool ShouldSuspendWindowRendering(WindowVisibility visibility, bool applicationInBackground,
                                                   bool surfaceRecreationPending) noexcept
{
    return visibility == WindowVisibility::Minimized || applicationInBackground || surfaceRecreationPending;
}

/// Validate the instance extensions returned by SDL against the selected
/// native video backend before Vulkan instance creation.  SDL is the source
/// of truth for backend selection; this helper deliberately leaves unknown
/// or custom backends to Vulkan so future SDL drivers keep their own errors.
inline bool ValidateVulkanWindowExtensions(std::string_view videoDriver,
                                           const std::vector<std::string_view> &extensions) noexcept
{
    const auto has = [&extensions](std::string_view name) {
        for (const auto extension : extensions) {
            if (extension == name)
                return true;
        }
        return false;
    };

    if (videoDriver.empty())
        return true;

    const bool knownBackend = videoDriver == "wayland" || videoDriver == "x11" || videoDriver == "windows" ||
                              videoDriver == "android" || videoDriver == "cocoa" || videoDriver == "kmsdrm";
    if (!knownBackend)
        return true;

    if (!has("VK_KHR_surface"))
        return false;

    if (videoDriver == "wayland")
        return has("VK_KHR_wayland_surface");
    if (videoDriver == "x11")
        return has("VK_KHR_xlib_surface") || has("VK_KHR_xcb_surface");
    if (videoDriver == "windows")
        return has("VK_KHR_win32_surface");
    if (videoDriver == "android")
        return has("VK_KHR_android_surface");
    if (videoDriver == "cocoa")
        return has("VK_EXT_metal_surface");
    if (videoDriver == "kmsdrm")
        return has("VK_KHR_display");

    return false;
}

inline WindowPresentationPolicy ResolveWindowPresentationPolicy(bool hasPlayerControlChannel,
                                                                std::string_view videoDriver)
{
    if (!hasPlayerControlChannel)
        return {};

    WindowPresentationPolicy policy;
    policy.focusable = false;
    policy.activateWhenShown = false;
    // Wayland does not assign an xdg-surface size until the toplevel is mapped
    // and configured. Vulkan surface preparation therefore cannot precede the
    // show handshake for an automated Player, even though X11 permits it.
    policy.showBeforeSurface = videoDriver == "wayland";
    return policy;
}

} // namespace infernux
