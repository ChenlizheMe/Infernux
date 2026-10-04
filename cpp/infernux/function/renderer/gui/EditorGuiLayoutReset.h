#pragma once

#include <imgui_internal.h>

namespace infernux
{

inline void ResetEditorGuiLayout()
{
    ImGui::ClearIniSettings();
    // ClearIniSettings removes saved settings but leaves existing windows'
    // FirstUseEver conditions consumed. Re-arm them at the explicit reset
    // boundary so authored defaults apply once when each view reopens.
    for (ImGuiWindow *window : ImGui::GetCurrentContext()->Windows) {
        window->SetWindowPosAllowFlags |= ImGuiCond_FirstUseEver;
        window->SetWindowSizeAllowFlags |= ImGuiCond_FirstUseEver;
        window->SetWindowCollapsedAllowFlags |= ImGuiCond_FirstUseEver;
        window->SetWindowDockAllowFlags |= ImGuiCond_FirstUseEver;
    }
}

} // namespace infernux
