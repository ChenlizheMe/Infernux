#pragma once

#include <imgui_internal.h>

namespace infernux
{

// Keep Dear ImGui's menu layout and input handling; only the vertical menu
// checkmark uses the editor's smaller size. Text and hit targets stay intact.
inline bool EditorMenuItem(const char *label, const char *shortcut, bool selected, bool enabled)
{
    ImGuiWindow *window = ImGui::GetCurrentWindow();
    if (window->SkipItems || window->DC.LayoutType == ImGuiLayoutType_Horizontal)
        return ImGui::MenuItem(label, shortcut, selected, enabled);

    const ImVec2 position = window->DC.CursorPos;
    const float baseline = window->DC.CurrLineTextBaseOffset;
    const float availableWidth = ImGui::GetContentRegionAvail().x;
    if (!enabled)
        ImGui::BeginDisabled();
    const bool clicked = ImGui::MenuItem(label, shortcut, false, enabled);
    if (selected && ImGui::IsItemVisible()) {
        const ImGuiMenuColumns &columns = window->DC.MenuColumns;
        const float minWidth = static_cast<float>(ImMax(columns.TotalWidth, columns.NextTotalWidth));
        const float stretch = ImMax(0.0f, availableWidth - minWidth);
        const float fontSize = ImGui::GetFontSize();
        const float markSize = fontSize * 0.66f;
        const ImVec2 markPosition(position.x + columns.OffsetMark + stretch + fontSize * 0.503f,
                                  position.y + baseline + (fontSize - markSize) * 0.5f);
        ImGui::RenderCheckMark(window->DrawList, markPosition, ImGui::GetColorU32(ImGuiCol_Text), markSize);
    }
    if (!enabled)
        ImGui::EndDisabled();
    return clicked;
}

} // namespace infernux
