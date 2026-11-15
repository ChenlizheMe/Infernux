#pragma once

#include "EditorThemeRegistry.h"
#include <algorithm>
#include <array>
#include <imgui_internal.h>

namespace infernux::theme_drawing
{

struct FrameStyle
{
    ImVec4 top, bottom, frame, inner, highlight, glow;
    float width, inset, cut, radius;

    bool HasFill() const { return top.w > 0.0f || bottom.w > 0.0f; }
    bool HasBorder() const { return width > 0.0f && frame.w > 0.0f; }
};

struct Styles
{
    FrameStyle panel, button;
    ImVec4 buttonHoverTop, buttonHoverBottom, buttonActiveTop, buttonActiveBottom;
};

inline FrameStyle ReadFrameStyle(const std::string &prefix)
{
    return {EditorThemeRegistry::Color(prefix + "_GRADIENT_TOP"),
            EditorThemeRegistry::Color(prefix + "_GRADIENT_BOTTOM"),
            EditorThemeRegistry::Color(prefix + "_FRAME_COLOR"),
            EditorThemeRegistry::Color(prefix + "_FRAME_INNER_COLOR"),
            EditorThemeRegistry::Color(prefix + "_FRAME_HIGHLIGHT"),
            EditorThemeRegistry::Color(prefix + "_GLOW_COLOR"),
            EditorThemeRegistry::Float(prefix + "_FRAME_WIDTH"),
            EditorThemeRegistry::Float(prefix + "_FRAME_INSET"),
            EditorThemeRegistry::Float(prefix + "_CORNER_CUT"),
            EditorThemeRegistry::Float(prefix + "_GLOW_RADIUS")};
}

inline const Styles &ActiveStyles()
{
    static unsigned long generation = 0;
    static Styles styles;
    if (generation != EditorThemeRegistry::Generation()) {
        styles = {ReadFrameStyle("PANEL"), ReadFrameStyle("BUTTON"),
                  EditorThemeRegistry::Color("BUTTON_HOVER_GRADIENT_TOP"),
                  EditorThemeRegistry::Color("BUTTON_HOVER_GRADIENT_BOTTOM"),
                  EditorThemeRegistry::Color("BUTTON_ACTIVE_GRADIENT_TOP"),
                  EditorThemeRegistry::Color("BUTTON_ACTIVE_GRADIENT_BOTTOM")};
        generation = EditorThemeRegistry::Generation();
    }
    return styles;
}

inline std::array<ImVec2, 8> Corners(const ImRect &rect, float cut)
{
    cut = std::min(cut, std::min(rect.GetWidth(), rect.GetHeight()) * 0.24f);
    return {ImVec2(rect.Min.x + cut, rect.Min.y), ImVec2(rect.Max.x - cut, rect.Min.y),
            ImVec2(rect.Max.x, rect.Min.y + cut), ImVec2(rect.Max.x, rect.Max.y - cut),
            ImVec2(rect.Max.x - cut, rect.Max.y), ImVec2(rect.Min.x + cut, rect.Max.y),
            ImVec2(rect.Min.x, rect.Max.y - cut), ImVec2(rect.Min.x, rect.Min.y + cut)};
}

inline ImU32 Color(ImVec4 color)
{
    color.w *= ImGui::GetStyle().Alpha;
    return ImGui::ColorConvertFloat4ToU32(color);
}

inline void Fill(ImDrawList *draw, const ImRect &rect, const FrameStyle &style, float scale, float rounding = 0.0f)
{
    if (!style.HasFill())
        return;
    const int first = draw->VtxBuffer.Size;
    if (style.cut > 0.0f) {
        const auto points = Corners(rect, style.cut * scale);
        draw->AddConvexPolyFilled(points.data(), static_cast<int>(points.size()), IM_COL32_WHITE);
    } else {
        draw->AddRectFilled(rect.Min, rect.Max, IM_COL32_WHITE, rounding);
    }
    // Interpolate RGBA while preserving the polygon edge's AA coverage.
    for (int index = first; index < draw->VtxBuffer.Size; ++index) {
        auto &vertex = draw->VtxBuffer[index];
        const float t = std::clamp((vertex.pos.y - rect.Min.y) / rect.GetHeight(), 0.0f, 1.0f);
        auto color = ImLerp(style.top, style.bottom, t);
        color.w *= static_cast<float>((vertex.col >> IM_COL32_A_SHIFT) & 255U) / 255.0f;
        vertex.col = Color(color);
    }
}

inline void Border(ImDrawList *draw, ImRect rect, const FrameStyle &style, float scale)
{
    if (!style.HasBorder())
        return;
    // Put the outline and glow inside the widget/window's allocated rectangle.
    rect.Expand(-(style.radius + style.width * 0.5f) * scale);
    if (rect.GetWidth() <= 0.0f || rect.GetHeight() <= 0.0f)
        return;
    const auto points = Corners(rect, style.cut * scale);
    if (style.radius > 0.0f && style.glow.w > 0.0f) {
        for (int layer = 3; layer >= 1; --layer) {
            auto glow = style.glow;
            glow.w *= 0.09f * (4 - layer);
            draw->AddPolyline(points.data(), static_cast<int>(points.size()), Color(glow), ImDrawFlags_Closed,
                              (style.width + style.radius * layer * 0.66f) * scale);
        }
    }
    draw->AddPolyline(points.data(), static_cast<int>(points.size()), Color(style.frame), ImDrawFlags_Closed,
                      style.width * scale);
    rect.Expand(-style.inset * scale);
    if (style.inner.w > 0.0f && rect.GetWidth() > 0.0f && rect.GetHeight() > 0.0f) {
        const auto inner = Corners(rect, std::max(0.0f, style.cut - style.inset) * scale);
        draw->AddPolyline(inner.data(), static_cast<int>(inner.size()), Color(style.inner), ImDrawFlags_Closed, scale);
    }
    draw->AddLine(points[0], points[1], Color(style.highlight), scale);
}

} // namespace infernux::theme_drawing
