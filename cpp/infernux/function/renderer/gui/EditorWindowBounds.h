#pragma once

#include <algorithm>
#include <cmath>
#include <imgui_internal.h>

namespace infernux
{

inline void RescaleDockspaceForViewport(ImGuiID dockspaceId, const ImVec2 &size)
{
    ImGuiDockNode *root = ImGui::DockBuilderGetNode(dockspaceId);
    if (!root || !root->IsDockSpace() || root->Size.x <= 0 || root->Size.y <= 0 || size.x <= 0 || size.y <= 0)
        return;
    if (std::abs(root->Size.x - size.x) < 0.5f && std::abs(root->Size.y - size.y) < 0.5f)
        return;

    // ImGui gives the central node whatever remains after absolute sidebar
    // SizeRef values. A layout saved on a large monitor can therefore reduce
    // Scene to a few pixels on a smaller window. Preserve the user's split
    // proportions and topology; never reset their layout or CentralNode flags.
    const ImVec2 scale(size.x / root->Size.x, size.y / root->Size.y);
    const auto rescale = [&](auto &&self, ImGuiDockNode *node) -> void {
        if (!node)
            return;
        node->SizeRef.x *= scale.x;
        node->SizeRef.y *= scale.y;
        node->Size.x *= scale.x;
        node->Size.y *= scale.y;
        self(self, node->ChildNodes[0]);
        self(self, node->ChildNodes[1]);
    };
    rescale(rescale, root->ChildNodes[0]);
    rescale(rescale, root->ChildNodes[1]);
    // Leave root->Size unchanged: DockSpace must still observe the host resize
    // and propagate the new bounds. Fixed-height toolbar sizing runs after us.
    ImGui::MarkIniSettingsDirty();
}

inline void ConstrainNextFloatingWindowToMainViewport(const char *name, int flags)
{
    ImGuiViewport *viewport = ImGui::GetMainViewport();
    if (!viewport || (flags & ImGuiWindowFlags_ChildWindow) != 0)
        return;
    ImGuiWindow *window = ImGui::FindWindowByName(name);
    if (!window)
        return;

    ImGuiDockNode *floatingRoot = nullptr;
    if (window->DockNode != nullptr) {
        floatingRoot = ImGui::DockNodeGetRootNode(window->DockNode);
        // AlwaysTabBar also creates dock nodes for floating windows. Only a
        // dockspace belongs to the main layout; floating groups keep their
        // topology while their owning root is constrained as one window.
        if (!floatingRoot->IsFloatingNode())
            return;
    }

    const ImVec2 workMin = viewport->WorkPos;
    const ImVec2 workSize = viewport->WorkSize;
    if (workSize.x <= 0.0f || workSize.y <= 0.0f)
        return;
    const ImVec2 size = floatingRoot ? floatingRoot->Size : window->SizeFull;
    const ImVec2 pos = floatingRoot ? floatingRoot->Pos : window->Pos;
    const ImVec2 constrainedSize(std::min(size.x, workSize.x), std::min(size.y, workSize.y));
    const ImVec2 maxPos(workMin.x + workSize.x - constrainedSize.x, workMin.y + workSize.y - constrainedSize.y);
    const ImVec2 constrainedPos(std::clamp(pos.x, workMin.x, maxPos.x), std::clamp(pos.y, workMin.y, maxPos.y));
    const bool sizeChanged = std::abs(constrainedSize.x - size.x) > 0.5f || std::abs(constrainedSize.y - size.y) > 0.5f;
    const bool positionChanged = std::abs(constrainedPos.x - pos.x) > 0.5f || std::abs(constrainedPos.y - pos.y) > 0.5f;
    if (floatingRoot) {
        if (sizeChanged)
            ImGui::DockBuilderSetNodeSize(floatingRoot->ID, constrainedSize);
        if (positionChanged)
            ImGui::DockBuilderSetNodePos(floatingRoot->ID, constrainedPos);
    } else {
        if (sizeChanged)
            ImGui::SetNextWindowSize(constrainedSize, ImGuiCond_Always);
        if (positionChanged)
            ImGui::SetNextWindowPos(constrainedPos, ImGuiCond_Always);
    }
    if (positionChanged || sizeChanged)
        ImGui::SetNextWindowFocus();
}

} // namespace infernux
