#pragma once

#include <imgui_internal.h>
#include <algorithm>

namespace infernux
{

inline void BringDockTreeToDisplayFront(ImGuiWindow *window)
{
    if (window == nullptr)
        return;

    ImGuiWindow *root = window->RootWindowDockTree != nullptr ? window->RootWindowDockTree : window;
    ImGuiWindow *focusRoot = window->RootWindow != nullptr ? window->RootWindow : window;
    // Tab selection changes navigation focus, not the layer of a background
    // dockspace. Match FocusWindow's policy, including child/root flags.
    if ((window->Flags | focusRoot->Flags | root->Flags) & ImGuiWindowFlags_NoBringToFrontOnFocus)
        return;

    ImGuiContext &imgui = *ImGui::GetCurrentContext();
    // A floating dock tree has several entries. Move its complete presentation
    // group while retaining the relative order of hosts and children.
    std::stable_partition(imgui.Windows.begin(), imgui.Windows.end(), [root](ImGuiWindow *candidate) {
        return candidate == nullptr || candidate->RootWindowDockTree != root;
    });
}

} // namespace infernux
