#pragma once

#include <cmath>
#include <imgui.h>

namespace infernux
{

inline bool GuiDrawDataMatchesFramebuffer(const ImDrawData &draw, int width, int height)
{
    // Allow floating-point roundoff at fractional framebuffer densities, not
    // a one-pixel resize. Content/UI scale is deliberately NOT part of this.
    return std::abs(draw.DisplaySize.x * draw.FramebufferScale.x - width) < 0.5f &&
           std::abs(draw.DisplaySize.y * draw.FramebufferScale.y - height) < 0.5f;
}

inline bool GuiDrawDataMatchesWindow(const ImDrawData *draw, int width, int height, int pixelWidth, int pixelHeight)
{
    return draw != nullptr && draw->Valid && draw->DisplaySize.x == static_cast<float>(width) &&
           draw->DisplaySize.y == static_cast<float>(height) &&
           GuiDrawDataMatchesFramebuffer(*draw, pixelWidth, pixelHeight);
}

// A surface can change after SDL NewFrame(), while acquiring a Vulkan image.
// Map this one already-built frame to the actual attachment instead of letting
// the backend install an oversized viewport/scissor and crop the UI. Rebuild
// layout/input on the following frame; never persist this transient density in
// ImGui IO or the cached draw data (nor multiply it by Windows content scale).
class ScopedGuiPresentationScale
{
  public:
    ScopedGuiPresentationScale(ImDrawData &draw, int width, int height)
        : m_draw(draw), m_originalScale(draw.FramebufferScale)
    {
        if (width > 0 && height > 0 && std::isfinite(draw.DisplaySize.x) && std::isfinite(draw.DisplaySize.y) &&
            draw.DisplaySize.x > 0 && draw.DisplaySize.y > 0)
            draw.FramebufferScale = ImVec2(width / draw.DisplaySize.x, height / draw.DisplaySize.y);
    }
    ~ScopedGuiPresentationScale()
    {
        m_draw.FramebufferScale = m_originalScale;
    }
    ScopedGuiPresentationScale(const ScopedGuiPresentationScale &) = delete;
    ScopedGuiPresentationScale &operator=(const ScopedGuiPresentationScale &) = delete;

  private:
    ImDrawData &m_draw;
    ImVec2 m_originalScale;
};

} // namespace infernux
