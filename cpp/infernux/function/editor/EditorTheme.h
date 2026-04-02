#pragma once

#include <imgui.h>

#include <algorithm>

namespace infernux
{

/// Lightweight theme constants for C++ editor panels.
/// Matches the Python Theme class values from theme.py.
/// Colors are in linear sRGB, RGBA order (matching ImGui's convention).
namespace EditorTheme
{

// ── Console log level colors ─────────────────────────────────────────
constexpr ImVec4 LOG_INFO{0.82f, 0.82f, 0.85f, 1.0f};
constexpr ImVec4 LOG_WARNING{0.890f, 0.710f, 0.300f, 1.0f};
constexpr ImVec4 LOG_ERROR{0.922f, 0.341f, 0.341f, 1.0f};
constexpr ImVec4 LOG_TRACE{0.50f, 0.50f, 0.50f, 1.0f};
constexpr ImVec4 LOG_BADGE{0.55f, 0.55f, 0.55f, 1.0f};
constexpr ImVec4 LOG_DIM{0.133f, 0.133f, 0.133f, 0.6f};

// ── Selection / Row colors ───────────────────────────────────────────
constexpr ImVec4 SELECTION_BG{0.173f, 0.365f, 0.529f, 1.0f};
constexpr ImVec4 ROW_ALT{0.0f, 0.0f, 0.0f, 0.06f};
constexpr ImVec4 ROW_NONE{0.0f, 0.0f, 0.0f, 0.0f};

// ── Transparent / ghost ──────────────────────────────────────────────
constexpr ImVec4 BTN_GHOST{0.0f, 0.0f, 0.0f, 0.0f};
constexpr ImVec4 BTN_GHOST_HOVERED{1.0f, 1.0f, 1.0f, 0.06f};
constexpr ImVec4 BTN_GHOST_ACTIVE{1.0f, 1.0f, 1.0f, 0.10f};
constexpr ImVec4 BORDER_TRANSPARENT{0.0f, 0.0f, 0.0f, 0.0f};

// ── Status bar button ────────────────────────────────────────────────
constexpr ImVec4 BTN_SB_HOVERED{0.20f, 0.20f, 0.20f, 0.6f};
constexpr ImVec4 BTN_SB_ACTIVE{0.25f, 0.25f, 0.25f, 0.8f};

// ── Toolbar play-mode buttons ────────────────────────────────────────
constexpr ImVec4 PLAY_ACTIVE{0.20f, 0.45f, 0.30f, 1.0f};
constexpr ImVec4 PAUSE_ACTIVE{0.50f, 0.40f, 0.15f, 1.0f};
constexpr ImVec4 BTN_IDLE{0.18f, 0.18f, 0.18f, 1.0f};
constexpr ImVec4 BTN_DISABLED{0.15f, 0.15f, 0.15f, 0.4f};

// ── Headers / Selectables ────────────────────────────────────────────
constexpr ImVec4 HEADER_HOVERED{0.28f, 0.24f, 0.24f, 1.0f};
constexpr ImVec4 HEADER_ACTIVE{0.32f, 0.25f, 0.25f, 1.0f};

// ── Panel backgrounds ────────────────────────────────────────────────
constexpr ImVec4 STATUS_BAR_BG{0.13f, 0.13f, 0.13f, 1.0f};
constexpr ImVec4 MENU_BAR_BG{0.16f, 0.16f, 0.16f, 1.0f};
constexpr ImVec4 POPUP_BG{0.24f, 0.24f, 0.24f, 0.96f};

// ── Status bar progress ──────────────────────────────────────────────
constexpr ImVec4 STATUS_PROGRESS_CLR{235.0f / 255.0f, 87.0f / 255.0f, 87.0f / 255.0f, 1.0f};
constexpr ImVec4 STATUS_PROGRESS_BG{0.10f, 0.10f, 0.10f, 1.0f};
constexpr ImVec4 STATUS_PROGRESS_LABEL_CLR{0.65f, 0.65f, 0.65f, 1.0f};
constexpr float STATUS_PROGRESS_FRACTION = 0.25f;

// ── Splitter ─────────────────────────────────────────────────────────
constexpr ImVec4 SPLITTER_HOVER{0.35f, 0.25f, 0.25f, 0.6f};
constexpr ImVec4 SPLITTER_ACTIVE{0.40f, 0.28f, 0.28f, 0.8f};

// ── Console toolbar compact spacing ──────────────────────────────────
constexpr float CONSOLE_FRAME_PAD_X = 4.0f;
constexpr float CONSOLE_FRAME_PAD_Y = 3.0f;
constexpr float CONSOLE_ITEM_SPC_X = 6.0f;
constexpr float CONSOLE_ITEM_SPC_Y = 4.0f;
constexpr float TOOLBAR_FRAME_BRD = 0.0f;

// ── Toolbar spacing ──────────────────────────────────────────────────
constexpr ImVec2 TOOLBAR_WIN_PAD{4.0f, 4.0f};
constexpr ImVec2 TOOLBAR_FRAME_PAD{6.0f, 4.0f};
constexpr ImVec2 TOOLBAR_ITEM_SPC{6.0f, 4.0f};
constexpr float TOOLBAR_FRAME_RND = 3.0f;

// ── Menu bar spacing ─────────────────────────────────────────────────
// (reuses TOOLBAR_FRAME_PAD / TOOLBAR_ITEM_SPC / TOOLBAR_WIN_PAD)

// ── Popup spacing ────────────────────────────────────────────────────
constexpr ImVec2 POPUP_WIN_PAD{8.0f, 8.0f};
constexpr ImVec2 POPUP_ITEM_SPC{8.0f, 4.0f};
constexpr ImVec2 POPUP_FRAME_PAD{4.0f, 3.0f};

// ── Status bar spacing ───────────────────────────────────────────────
constexpr ImVec2 STATUS_BAR_WIN_PAD{6.0f, 4.0f};
constexpr ImVec2 STATUS_BAR_ITEM_SPC{8.0f, 0.0f};
constexpr ImVec2 STATUS_BAR_FRAME_PAD{0.0f, 0.0f};
constexpr float STATUS_BAR_BASE_HEIGHT = 24.0f;

// ── Hierarchy / Tree ─────────────────────────────────────────────────
constexpr ImVec2 TREE_ITEM_SPC{0.0f, 0.0f};   // ItemSpacing (Unity: 0)
constexpr ImVec2 TREE_FRAME_PAD{2.0f, 2.0f};  // FramePadding (Unity-compact)
constexpr float TREE_INDENT = 14.0f;           // IndentSpacing per level
constexpr ImVec4 PREFAB_TEXT{235.0f / 255.0f, 87.0f / 255.0f, 87.0f / 255.0f, 1.0f};
constexpr ImVec4 TEXT_DISABLED{0.40f, 0.40f, 0.40f, 1.0f};
constexpr ImVec4 DND_REORDER_LINE{1.0f, 1.0f, 1.0f, 0.90f};
constexpr float DND_REORDER_LINE_THICKNESS = 2.0f;
constexpr float DND_REORDER_SEPARATOR_H = 3.0f;
constexpr const char *PREFAB_ICON = "\xe2\x97\x86"; // ◆ U+25C6

// ── Icons (Unicode glyphs) ───────────────────────────────────────────
constexpr const char *ICON_WARNING = "\xe2\x96\xb2"; // ▲ U+25B2
constexpr const char *ICON_ERROR = "\xe2\x97\x8f";   // ● U+25CF

// ── Helper: push flat-color button style (3 colors) ─────────────────
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
    ImGui::PushStyleColor(ImGuiCol_Button, BTN_GHOST);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, BTN_GHOST_HOVERED);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, BTN_GHOST_ACTIVE);
}

} // namespace EditorTheme

} // namespace infernux
