#pragma once

#include "EditorThemeRegistry.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <imgui_internal.h>
#include <vector>

namespace infernux::theme_drawing
{

struct FrameStyle
{
    ImVec4 top, bottom, frame, inner, highlight, glow;
    float width, inset, cut, radius, roughness, bevel, fasteners;

    bool HasFill() const { return top.w > 0.0f || bottom.w > 0.0f; }
    bool HasBorder() const { return width > 0.0f && frame.w > 0.0f; }
};

struct Styles
{
    FrameStyle panel, button;
    ImVec4 buttonHoverTop, buttonHoverBottom, buttonActiveTop, buttonActiveBottom;
    bool monospace;
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
            EditorThemeRegistry::Float(prefix + "_GLOW_RADIUS"),
            EditorThemeRegistry::Float(prefix + "_FRAME_ROUGHNESS"),
            EditorThemeRegistry::Float(prefix + "_BEVEL_DEPTH"),
            EditorThemeRegistry::Float(prefix + "_FASTENER_SIZE")};
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
                  EditorThemeRegistry::Color("BUTTON_ACTIVE_GRADIENT_BOTTOM"),
                  EditorThemeRegistry::Float("EDITOR_MONO_UI") > 0.5f};
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

inline void Outline(ImDrawList *draw, const std::array<ImVec2, 8> &points, ImU32 color,
                    float width, float roughness, float scale)
{
    if (roughness <= 0.0f) {
        draw->AddPolyline(points.data(), static_cast<int>(points.size()), color, ImDrawFlags_Closed, width);
        return;
    }
    std::vector<ImVec2> contour;
    for (size_t edge = 0; edge < points.size(); ++edge) {
        const auto a = points[edge], b = points[(edge + 1) % points.size()];
        const float dx = b.x - a.x, dy = b.y - a.y;
        const float length = std::sqrt(dx * dx + dy * dy);
        const int segments = std::max(1, static_cast<int>(length / (18.0f * scale)));
        for (int index = 0; index < segments; ++index) {
            const float t = static_cast<float>(index) / segments;
            const float wobble = std::sin(t * 19.0f + edge * 7.0f) * roughness * scale;
            const float inverse = length > 0 ? 1.0f / length : 0.0f;
            contour.emplace_back(a.x + dx * t - dy * inverse * wobble,
                                 a.y + dy * t + dx * inverse * wobble);
        }
    }
    draw->AddPolyline(contour.data(), static_cast<int>(contour.size()), color, ImDrawFlags_Closed, width);
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
    Outline(draw, points, Color(style.frame), style.width * scale, style.roughness, scale);
    if (style.bevel > 0.0f) {
        for (int layer = 0; layer < static_cast<int>(style.bevel); ++layer) {
            ImRect inset = rect;
            inset.Expand(-(layer + 1) * scale);
            auto highlight = style.highlight;
            highlight.w *= 1.0f - static_cast<float>(layer) / style.bevel;
            draw->AddLine(inset.Min, ImVec2(inset.Max.x, inset.Min.y), Color(highlight), scale);
            draw->AddLine(inset.Min, ImVec2(inset.Min.x, inset.Max.y), Color(highlight), scale);
            auto shadow = style.inner;
            shadow.w *= 0.8f;
            draw->AddLine(ImVec2(inset.Min.x, inset.Max.y), inset.Max, Color(shadow), scale);
            draw->AddLine(ImVec2(inset.Max.x, inset.Min.y), inset.Max, Color(shadow), scale);
        }
    }
    if (style.fasteners > 0.0f && rect.GetWidth() > 100.0f * scale && rect.GetHeight() > 70.0f * scale) {
        const float radius = style.fasteners * scale;
        for (float x : {rect.Min.x + radius * 2.5f, rect.Max.x - radius * 2.5f}) {
            for (float y : {rect.Min.y + radius * 2.5f, rect.Max.y - radius * 2.5f}) {
                draw->AddCircleFilled(ImVec2(x, y), radius, Color(style.inner), 12);
                draw->AddCircle(ImVec2(x, y), radius, Color(style.highlight), 12, scale);
                draw->AddLine(ImVec2(x - radius * 0.55f, y + radius * 0.35f),
                              ImVec2(x + radius * 0.55f, y - radius * 0.35f), Color(style.frame), scale);
            }
        }
    }
    rect.Expand(-style.inset * scale);
    if (style.inner.w > 0.0f && rect.GetWidth() > 0.0f && rect.GetHeight() > 0.0f) {
        const auto inner = Corners(rect, std::max(0.0f, style.cut - style.inset) * scale);
        Outline(draw, inner, Color(style.inner), scale, style.roughness * 0.5f, scale);
    }
    draw->AddLine(points[0], points[1], Color(style.highlight), scale);
}

} // namespace infernux::theme_drawing
