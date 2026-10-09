#include <function/renderer/gui/EditorGuiLayoutReset.h>
#include <function/renderer/gui/EditorWindowBounds.h>
#include <function/renderer/gui/EditorWindowPresentation.h>

#include <cassert>
#include <iostream>
#include <vector>

static void TestDockspaceViewportResize()
{
    ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DeltaTime = 1.0f / 60.0f;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    unsigned char *pixels;
    int width, height;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    ImGuiID left = 0, center = 0, right = 0;
    for (const ImVec2 size : {ImVec2(2560, 1440), ImVec2(1024, 701), ImVec2(2560, 1440), ImVec2(1280, 800)}) {
        io.DisplaySize = size;
        for (int frame = 0; frame < 3; ++frame) {
            ImGui::NewFrame();
            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(size);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
            ImGui::Begin("Resize workspace", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                                         ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoMove);
            const auto id = ImGui::GetID("main");
            if (left == 0) {
                ImGui::DockBuilderAddNode(id, ImGuiDockNodeFlags_DockSpace);
                ImGui::DockBuilderSetNodeSize(id, size);
                ImGuiID main;
                ImGui::DockBuilderSplitNode(id, ImGuiDir_Right, 0.25f, &right, &main);
                ImGui::DockBuilderSplitNode(main, ImGuiDir_Left, 0.2f, &left, &center);
                ImGui::DockBuilderDockWindow("Hierarchy", left);
                ImGui::DockBuilderDockWindow("Scene", center);
                ImGui::DockBuilderDockWindow("Inspector", right);
                ImGui::DockBuilderFinish(id);
            }
            infernux::RescaleDockspaceForViewport(id, size);
            ImGui::DockSpace(id, size);
            ImGui::End();
            ImGui::PopStyleVar();
            for (const char *name : {"Hierarchy", "Scene", "Inspector"}) {
                ImGui::Begin(name);
                ImGui::TextUnformatted(name);
                ImGui::End();
            }
            ImGui::Render();
            if (frame == 2) {
                assert(ImGui::FindWindowByName("Scene")->DockId == center);
                assert(ImGui::FindWindowByName("Hierarchy")->DockId == left);
                assert(ImGui::FindWindowByName("Inspector")->DockId == right);
                const auto *scene = ImGui::DockBuilderGetNode(center);
                const auto *inspector = ImGui::DockBuilderGetNode(right);
                assert(scene->Size.x > size.x * 0.5f);
                assert(std::abs(inspector->Size.x / size.x - 0.25f) < 0.02f);
                assert(inspector->Pos.x + inspector->Size.x <= size.x + 1);
            }
        }
    }
    ImGui::DestroyContext();
}

static void TestWindowPresentation()
{
    ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = ImVec2(1000, 700);
    io.DeltaTime = 1.0f / 60.0f;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    unsigned char *pixels;
    int width, height;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

    for (int frame = 0; frame < 8; ++frame) {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("Workspace", nullptr, ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoDocking);
        const ImGuiID main = ImGui::GetID("Main");
        if (frame == 0) {
            ImGui::DockBuilderAddNode(main, ImGuiDockNodeFlags_DockSpace);
            ImGui::DockBuilderSetNodeSize(main, io.DisplaySize);
            ImGui::DockBuilderDockWindow("Game", main);
            ImGui::DockBuilderDockWindow("Scene", main);
            ImGui::DockBuilderFinish(main);
        }
        ImGui::DockSpace(main);
        ImGui::End();
        for (const char *name : {"Game", "Scene"}) {
            ImGui::Begin(name);
            ImGui::TextUnformatted(name);
            ImGui::End();
        }
        ImGui::SetNextWindowPos(ImVec2(120, 100));
        ImGui::SetNextWindowSize(ImVec2(320, 200));
        ImGui::Begin("Hello", nullptr, ImGuiWindowFlags_NoDocking);
        ImGui::TextUnformatted("Floating plugin panel");
        ImGui::BeginChild("Content", ImVec2(250, 100));
        ImGui::TextUnformatted("Plugin controls");
        ImGui::EndChild();
        ImGui::End();

        ImGuiWindow *game = ImGui::FindWindowByName(frame % 2 ? "Game" : "Scene");
        assert(game && game->DockNode);
        ImGui::FocusWindow(game);
        infernux::BringDockTreeToDisplayFront(game);
        assert(ImGui::GetCurrentContext()->NavWindow == game);
        ImGui::Render();

        // Play/Stop switches docked focus repeatedly. The floating plugin
        // must remain the actual pointer hit target above the workspace.
        if (frame > 1) {
            ImGuiWindow *hovered = nullptr;
            ImGui::FindHoveredWindowEx(ImVec2(270, 180), false, &hovered, nullptr);
            assert(hovered && hovered->RootWindow == ImGui::FindWindowByName("Hello"));
        }
    }

    // Floating dock groups can still be raised, with their host/child order
    // preserved. This must not turn every editor panel into an always-on-top.
    for (int frame = 0; frame < 3; ++frame) {
        ImGui::NewFrame();
        if (frame == 0) {
            const ImGuiID group = ImGui::GetID("Floating tools");
            ImGui::DockBuilderAddNode(group);
            ImGui::DockBuilderSetNodePos(group, ImVec2(150, 150));
            ImGui::DockBuilderSetNodeSize(group, ImVec2(500, 300));
            ImGuiID left, right;
            ImGui::DockBuilderSplitNode(group, ImGuiDir_Left, 0.5f, &left, &right);
            ImGui::DockBuilderDockWindow("ToolLeft", left);
            ImGui::DockBuilderDockWindow("ToolRight", right);
            ImGui::DockBuilderFinish(group);
        }
        for (const char *name : {"ToolLeft", "ToolRight"}) {
            ImGui::Begin(name);
            ImGui::TextUnformatted(name);
            ImGui::End();
        }
        ImGui::Render();
    }
    ImGuiWindow *tool = ImGui::FindWindowByName("ToolLeft");
    assert(tool->DockNode && ImGui::DockNodeGetRootNode(tool->DockNode)->IsFloatingNode());
    ImGuiWindow *toolRoot = tool->RootWindowDockTree;
    const ImVector<ImGuiWindow *> before = ImGui::GetCurrentContext()->Windows;
    infernux::BringDockTreeToDisplayFront(tool);
    std::vector<ImGuiWindow *> oldGroup, newGroup;
    for (ImGuiWindow *window : before)
        if (window->RootWindowDockTree == toolRoot)
            oldGroup.push_back(window);
    const auto &after = ImGui::GetCurrentContext()->Windows;
    for (ImGuiWindow *window : after)
        if (window->RootWindowDockTree == toolRoot)
            newGroup.push_back(window);
    assert(oldGroup == newGroup && oldGroup.size() >= 3);
    assert(after.back()->RootWindowDockTree == toolRoot);
    ImGui::DestroyContext();
}

static void RenderWindow(const char *name)
{
    infernux::ConstrainNextFloatingWindowToMainViewport(name, 0);
    ImGui::Begin(name);
    ImGui::TextUnformatted("Build controls");
    ImGui::End();
}

int main()
{
    ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = ImVec2(958, 699);
    io.DeltaTime = 1.0f / 60.0f;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigDockingAlwaysTabBar = true;
    unsigned char *pixels;
    int width, height;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

    for (int frame = 0; frame < 6; ++frame) {
        ImGui::NewFrame();
        if (frame == 0) {
            ImGui::SetNextWindowPos(ImVec2(60, 60));
            ImGui::SetNextWindowSize(ImVec2(980, 720));
        }
        RenderWindow("Build###build_settings");
        ImGui::Render();
    }
    auto *window = ImGui::FindWindowByName("Build###build_settings");
    assert(window && window->DockNode);
    auto *root = ImGui::DockNodeGetRootNode(window->DockNode);
    assert(root->IsFloatingNode());
    assert(root->Pos.x >= 0 && root->Pos.y >= 0);
    assert(root->Pos.x + root->Size.x <= io.DisplaySize.x);
    assert(root->Pos.y + root->Size.y <= io.DisplaySize.y);
    assert(window->Pos.y + window->Size.y <= io.DisplaySize.y);

    // Resizing the application also constrains an existing floating tab group.
    const ImGuiID floatingId = root->ID;
    io.DisplaySize = ImVec2(720, 500);
    for (int frame = 0; frame < 4; ++frame) {
        ImGui::NewFrame();
        RenderWindow("Build###build_settings");
        ImGui::Render();
    }
    assert(ImGui::DockNodeGetRootNode(window->DockNode)->ID == floatingId);
    assert(root->Pos.x + root->Size.x <= 720);
    assert(root->Pos.y + root->Size.y <= 500);

    // Multiple floating tabs/splits must keep their shared root and topology.
    ImGui::NewFrame();
    const ImGuiID group = ImGui::GetID("floating-split");
    ImGui::DockBuilderAddNode(group);
    ImGui::DockBuilderSetNodePos(group, ImVec2(100, 100));
    ImGui::DockBuilderSetNodeSize(group, ImVec2(1100, 800));
    ImGuiID left, right;
    ImGui::DockBuilderSplitNode(group, ImGuiDir_Left, 0.5f, &left, &right);
    ImGui::DockBuilderDockWindow("Left", left);
    ImGui::DockBuilderDockWindow("Right", right);
    ImGui::DockBuilderFinish(group);
    RenderWindow("Left");
    RenderWindow("Right");
    ImGui::Render();
    for (int frame = 0; frame < 4; ++frame) {
        ImGui::NewFrame();
        RenderWindow("Left");
        RenderWindow("Right");
        ImGui::Render();
    }
    auto *split = ImGui::DockBuilderGetNode(group);
    assert(split && split->IsFloatingNode());
    assert(split->ChildNodes[0]->ID == left && split->ChildNodes[1]->ID == right);
    assert(split->Pos.x + split->Size.x <= 720);
    assert(split->Pos.y + split->Size.y <= 500);
    for (const char *name : {"Left", "Right"}) {
        auto *child = ImGui::FindWindowByName(name);
        assert(child->Pos.x + child->Size.x <= 720);
        assert(child->Pos.y + child->Size.y <= 500);
    }

    // A dockspace is owned by its host layout, never by an individual panel.
    ImGui::NewFrame();
    const ImGuiID dockspace = ImGui::GetID("owned-dockspace");
    ImGui::DockBuilderAddNode(dockspace, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodePos(dockspace, ImVec2(30, 40));
    ImGui::DockBuilderSetNodeSize(dockspace, ImVec2(1200, 1000));
    ImGui::DockBuilderDockWindow("Docked", dockspace);
    ImGui::DockBuilderFinish(dockspace);
    RenderWindow("Docked");
    infernux::ConstrainNextFloatingWindowToMainViewport("Docked", 0);
    auto *owned = ImGui::DockBuilderGetNode(dockspace);
    assert(owned->Pos.x == 30 && owned->Pos.y == 40);
    assert(owned->Size.x == 1200 && owned->Size.y == 1000);
    ImGui::Render();
    // Reset re-applies authored defaults to an existing plugin window, once.
    auto renderPlugin = [](float userWidth = 0.0f) {
        ImGui::NewFrame();
        ImGui::SetNextWindowSize(ImVec2(420, 240), ImGuiCond_FirstUseEver);
        if (userWidth > 0)
            ImGui::SetNextWindowSize(ImVec2(userWidth, 240), ImGuiCond_Always);
        ImGui::Begin("Plugin###layout-reset-plugin", nullptr, ImGuiWindowFlags_NoDocking);
        const ImVec2 size = ImGui::GetWindowSize();
        ImGui::TextUnformatted("Plugin content");
        ImGui::End();
        ImGui::Render();
        return size;
    };
    assert(renderPlugin().x == 420);
    assert(renderPlugin(180).x == 180);
    assert(renderPlugin().x == 180);
    infernux::ResetEditorGuiLayout();
    if (renderPlugin().x != 420) {
        std::cerr << "Layout reset did not restore the plugin's authored width\n";
        ImGui::DestroyContext();
        return 1;
    }
    assert(renderPlugin(500).x == 500);
    assert(renderPlugin().x == 500);
    ImGui::DestroyContext();
    TestWindowPresentation();
    TestDockspaceViewportResize();
}
