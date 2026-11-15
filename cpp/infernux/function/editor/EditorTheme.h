#pragma once

#include <imgui.h>
#include "EditorThemeRegistry.h"

#include <algorithm>

namespace infernux
{

/// Lightweight theme constants for C++ editor panels.
/// Matches the Python Theme class values from theme.py.
/// Colors are in linear sRGB, RGBA order (matching ImGui's convention).
namespace EditorTheme
{

// (reuses TOOLBAR_FRAME_PAD / TOOLBAR_ITEM_SPC / TOOLBAR_WIN_PAD)

  // ItemSpacing
 // FramePadding
         // IndentSpacing per level

constexpr const char *PREFAB_ICON = "\xe2\x97\x86"; // ◆ U+25C6

constexpr const char *ICON_WARNING = "\xe2\x96\xb2"; // ▲ U+25B2
constexpr const char *ICON_ERROR = "\xe2\x97\x8f";   // ● U+25CF

/// Scales only the checkbox square (label text stays at ambient font size).
/// Fixed side length (px) of every checkbox square — one size everywhere.
/// Fixed width of the numeric input box beside a ranged slider row.
/// Negative values move the track line upward within the slider row.

inline void PushFlatButtonStyle(const ImVec4 &base)
{
    ImGui::PushStyleColor(ImGuiCol_Button, base);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                          ImVec4{(std::min)(base.x + 0.06f, 1.0f), (std::min)(base.y + 0.06f, 1.0f),
                                 (std::min)(base.z + 0.06f, 1.0f), base.w});
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                          ImVec4{(std::min)(base.x + 0.12f, 1.0f), (std::min)(base.y + 0.12f, 1.0f),
                                 (std::min)(base.z + 0.12f, 1.0f), base.w});
}

inline void PushGhostButtonStyle()
{
    ImGui::PushStyleColor(ImGuiCol_Button, EditorThemeRegistry::Color("BTN_GHOST"));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, EditorThemeRegistry::Color("BTN_GHOST_HOVERED"));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, EditorThemeRegistry::Color("BTN_GHOST_ACTIVE"));
}

} // namespace EditorTheme

} // namespace infernux
