#pragma once

#include "InxGUIContext.h"
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
    const bool hasShortcut = shortcut && shortcut[0];
    if (hasShortcut) {
        ImGui::PushFont(InxGUIContext::s_monoFont, ImGui::GetFontSize());
        const float shortcutWidth = ImGui::CalcTextSize(shortcut).x;
        ImGui::PopFont();
        window->DC.MenuColumns.DeclColumns(0, 0, shortcutWidth, 0);
    }
    if (!enabled)
        ImGui::BeginDisabled();
    const bool clicked = ImGui::MenuItem(label, nullptr, false, enabled);
    if (ImGui::IsItemVisible()) {
        const ImGuiMenuColumns &columns = window->DC.MenuColumns;
        const float minWidth = static_cast<float>(ImMax(columns.TotalWidth, columns.NextTotalWidth));
        const float stretch = ImMax(0.0f, availableWidth - minWidth);
        const float fontSize = ImGui::GetFontSize();
        if (hasShortcut) {
            ImGui::PushFont(InxGUIContext::s_monoFont, fontSize);
            window->DrawList->AddText(ImVec2(position.x + columns.OffsetShortcut + stretch, position.y + baseline),
                                      ImGui::GetColorU32(ImGuiCol_TextDisabled), shortcut);
            ImGui::PopFont();
        }
        if (selected) {
            const float markSize = fontSize * 0.66f;
            const ImVec2 markPosition(position.x + columns.OffsetMark + stretch + fontSize * 0.503f,
                                      position.y + baseline + (fontSize - markSize) * 0.5f);
            ImGui::RenderCheckMark(window->DrawList, markPosition, ImGui::GetColorU32(ImGuiCol_Text), markSize);
        }
    }
    if (!enabled)
        ImGui::EndDisabled();
    return clicked;
}

} // namespace infernux
