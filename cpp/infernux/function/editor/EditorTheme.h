#pragma once

namespace infernux
{

/// Lightweight theme constants for C++ editor panels.
/// Matches the Python Theme class values from theme.py.
/// Colors are in linear sRGB, RGBA order (matching ImGui's convention).
namespace EditorTheme
{

// Console log level colors
constexpr ImVec4 LOG_INFO{0.82f, 0.82f, 0.85f, 1.0f};
constexpr ImVec4 LOG_WARNING{0.890f, 0.710f, 0.300f, 1.0f};
constexpr ImVec4 LOG_ERROR{0.922f, 0.341f, 0.341f, 1.0f};
constexpr ImVec4 LOG_TRACE{0.50f, 0.50f, 0.50f, 1.0f};
constexpr ImVec4 LOG_BADGE{0.55f, 0.55f, 0.55f, 1.0f};

// Selection / Row colors
constexpr ImVec4 SELECTION_BG{0.173f, 0.365f, 0.529f, 1.0f};
constexpr ImVec4 ROW_ALT{0.0f, 0.0f, 0.0f, 0.06f};
constexpr ImVec4 ROW_NONE{0.0f, 0.0f, 0.0f, 0.0f};

// Transparent / ghost
constexpr ImVec4 BTN_GHOST{0.0f, 0.0f, 0.0f, 0.0f};
constexpr ImVec4 BORDER_TRANSPARENT{0.0f, 0.0f, 0.0f, 0.0f};

// Splitter
constexpr ImVec4 SPLITTER_HOVER{0.35f, 0.25f, 0.25f, 0.6f};
constexpr ImVec4 SPLITTER_ACTIVE{0.40f, 0.28f, 0.28f, 0.8f};

// Console toolbar compact spacing
constexpr float CONSOLE_FRAME_PAD_X = 4.0f;
constexpr float CONSOLE_FRAME_PAD_Y = 3.0f;
constexpr float CONSOLE_ITEM_SPC_X = 6.0f;
constexpr float CONSOLE_ITEM_SPC_Y = 4.0f;
constexpr float TOOLBAR_FRAME_BRD = 0.0f;

} // namespace EditorTheme

} // namespace infernux
