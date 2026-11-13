#include <function/editor/ConsolePanel.h>
#include <function/editor/EditorObjectIcon.h>
#include <function/editor/InspectorPanel.h>
#include <function/renderer/gui/InxGUIContext.h>
#include <function/renderer/gui/InxGUISemantics.h>
#include <imgui.h>
#include <imgui_internal.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

// A field can synchronously publish selection while its component is being
// drawn. That must not destroy the rows still borrowed by the native frame.
static void CheckSelectionDuringFieldDraw(bool multi, bool selectAsset)
{
    ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = ImVec2(1280, 900);
    io.DeltaTime = 1.0f / 60.0f;
    unsigned char *pixels;
    int width, height;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    infernux::InxGUIContext context;
    infernux::InxGUIContext::s_monoFont = io.Fonts->Fonts[0];
    context.resolveEditorIcon = [](const std::string &) { return uint64_t{0}; };
    infernux::InspectorPanel panel;
    panel.translate = [](const std::string &key) { return key; };
    panel.getRevisionSnapshot = [] { return infernux::InspectorPanel::RevisionSnapshot{1, 1, 1, 1}; };
    panel.getObjectInfo = [](uint64_t) {
        infernux::InspectorPanel::ObjectInfo info;
        info.name = "Owner";
        info.hideTransform = true;
        return info;
    };
    panel.isMultiSelection = [multi] { return multi; };
    panel.getSelectedIds = [] { return std::vector<uint64_t>{10, 20}; };
    panel.getAllTags = [] { return std::vector<std::string>{"Untagged"}; };
    panel.getAllLayers = [] { return std::vector<std::string>{"Default"}; };
    int enumerations = 0;
    panel.getComponentList = [&](uint64_t objectId) {
        ++enumerations;
        std::vector<infernux::ComponentInfo> result;
        for (const char *name : {"MeshRenderer", "Rigidbody"}) {
            infernux::ComponentInfo info;
            info.typeName = name;
            info.componentId = objectId * 10 + result.size() + 1;
            info.enabled = true;
            result.push_back(std::move(info));
        }
        return result;
    };
    std::vector<std::string> iconTypes;
    panel.getComponentIconId = [&](const std::string &type, bool) {
        iconTypes.push_back(type);
        return uint64_t{0};
    };
    bool changed = false;
    auto select = [&] {
        if (!changed) {
            changed = true;
            if (selectAsset)
                panel.SetSelectedFile("Assets/Selected.mat", "material");
            else
                panel.ClearSelectedFile();
        }
    };
    panel.renderComponentBody = [&](auto *, uint64_t, const std::string &, uint64_t, bool) { select(); };
    panel.renderMultiComponentBody = [&](auto *, const auto &, const std::string &, const auto &, bool) { select(); };
    panel.SetSelectedObjectId(10);
    for (int frame = 0; frame < 4; ++frame) {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(20, 20));
        ImGui::SetNextWindowSize(ImVec2(640, 800));
        panel.OnRender(&context);
        ImGui::Render();
    }
    assert(changed);
    assert(std::find(iconTypes.begin(), iconTypes.end(), "") == iconTypes.end());
    assert(std::find(iconTypes.begin(), iconTypes.end(), "Rigidbody") != iconTypes.end());
    if (selectAsset)
        assert(panel.GetSelectedFile() == "Assets/Selected.mat");
    else
        assert(enumerations >= (multi ? 4 : 2)); // Invalidated rows rebuild on the next frame.
    infernux::InxGUIContext::s_monoFont = nullptr;
    ImGui::DestroyContext();
}

static void CheckDiagnosticInspector()
{
    ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = ImVec2(1280, 900);
    io.DeltaTime = 1.0f / 60.0f;
    unsigned char *pixels;
    int width, height;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    infernux::InxGUIContext context;
    infernux::InxGUIContext::s_monoFont = io.Fonts->Fonts[0];
    context.resolveEditorIcon = [](const std::string &) { return uint64_t{0}; };
    {
        infernux::ConsolePanel console;
        console.translate = [](const std::string &key) {
            if (key == "console.copy_details")
                return std::string("Copy all");
            if (key == "console.stack_trace")
                return std::string("CALL STACK");
            if (key == "console.level_error")
                return std::string("Error");
            return std::string("Log Details");
        };
        const std::string path = "Assets/" + std::string(180, 'p') + "/Probe.py";
        const std::string message = "Message with details " + std::string(240, 'm');
        const std::string stack = "Full stack trace\n  File \"" + path + "\", line 12, in update\n    move_player()";
        console.LogFromPython(infernux::LOG_ERROR, message, stack, path, 12);
        const auto entry = console.GetVisibleLogSnapshot(1).front();
        console.SetSelectionSnapshot(entry.uid);
        infernux::InspectorPanel inspector;
        int draws = 0;
        inspector.renderDiagnosticInspector = [&](auto *ctx) {
            ++draws;
            console.RenderSelectionInspector(ctx);
        };
        inspector.ShowDiagnostic();
        infernux::InxGUISemantics::SetCaptureEnabled(true);
        for (float panelWidth : {240.0f, 400.0f, 640.0f}) {
            for (int frame = 0; frame < 3; ++frame) {
                ImGui::NewFrame();
                infernux::InxGUISemantics::BeginFrame(frame);
                ImGui::SetNextWindowSize(ImVec2(panelWidth, 800));
                inspector.OnRender(&context);
                infernux::InxGUISemantics::EndFrame();
                const auto snapshot = infernux::InxGUISemantics::GetSnapshot();
                bool wrappedMessage = false, wrappedSource = false, wrappedStack = false;
                for (const auto &target : snapshot.targets) {
                    if (target.kind == "diagnostic_message" || target.kind == "source_link")
                        assert(target.width <= panelWidth - 16 + 0.1f);
                    if (target.kind == "diagnostic_message")
                        wrappedMessage |= target.height > 32;
                    if (target.semanticId == "console.detail.source")
                        wrappedSource |= target.height > 28;
                    if (target.semanticId.find(".stack.") != std::string::npos && target.kind == "source_link")
                        wrappedStack |= target.height > 28;
                }
                assert(wrappedMessage && wrappedSource && wrappedStack);
                auto *window = ImGui::FindWindowByName("Inspector###inspector");
                assert(window && window->ScrollMax.x == 0.0f);
                ImGui::Render();
            }
        }
        infernux::InxGUISemantics::SetCaptureEnabled(false);
        assert(draws > 0);
        assert(console.CopySelectedEntry());
        assert(std::string(ImGui::GetClipboardText()).find("Full stack trace") != std::string::npos);
        assert(std::string(ImGui::GetClipboardText()).find(path) != std::string::npos);
        // Search controls the list, not the selected diagnostic in Inspector.
        console.SetSearchQuery("does-not-match");
        assert(console.GetVisibleLogSnapshot(1).empty());
        assert(console.CopySelectedEntry());
        console.Clear();
        assert(!console.CopySelectedEntry());
    }
    infernux::InxGUIContext::s_monoFont = nullptr;
    ImGui::DestroyContext();
}

static void CheckObjectHeaderIcons()
{
    ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = ImVec2(1280, 900);
    io.DeltaTime = 1.0f / 60.0f;
    unsigned char *pixels;
    int width, height;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    infernux::InxGUIContext context;
    infernux::InxGUIContext::s_monoFont = io.Fonts->Fonts[0];
    std::vector<std::string> icons;
    context.resolveEditorIcon = [&](const std::string &resource) {
        icons.push_back(resource);
        return uint64_t{0};
    };
    {
        infernux::InspectorPanel panel;
        panel.getRevisionSnapshot = [] { return infernux::InspectorPanel::RevisionSnapshot{}; };
        panel.getObjectInfo = [](uint64_t) {
            infernux::InspectorPanel::ObjectInfo info;
            info.name = "Object";
            return info;
        };
        const std::vector<std::pair<std::string, std::string>> cases{
            {"Camera", "components/component_camera"},
            {"Light", "components/component_light"},
            {"MeshRenderer", "components/component_meshrenderer"},
            {"ParticleSystem", "components/component_particlesystem"},
            {"UICanvas", "components/component_uicanvas"},
            {"", "components/component_transform"},
        };
        for (size_t index = 0; index < cases.size(); ++index) {
            panel.getComponentList = [&](uint64_t) {
                std::vector<infernux::ComponentInfo> rows;
                if (!cases[index].first.empty()) {
                    infernux::ComponentInfo row;
                    row.typeName = cases[index].first;
                    row.componentId = 100;
                    rows.push_back(std::move(row));
                }
                return rows;
            };
            panel.SetSelectedObjectId(index + 1);
            icons.clear();
            ImGui::NewFrame();
            ImGui::SetNextWindowSize(ImVec2(640, 800));
            panel.OnRender(&context);
            assert(std::find(icons.begin(), icons.end(), cases[index].second) != icons.end());
            ImGui::Render();
        }
        infernux::EditorObjectIcon priority;
        priority.Consider("MeshRenderer");
        priority.Consider("Light");
        priority.Consider("Camera");
        assert(std::string(priority.Resource()) == "components/component_camera");
        assert(std::string(priority.Resource(true)) == "prefab");
    }
    infernux::InxGUIContext::s_monoFont = nullptr;
    ImGui::DestroyContext();
}

int main()
{
    for (bool multi : {false, true})
        for (bool selectAsset : {false, true})
            CheckSelectionDuringFieldDraw(multi, selectAsset);
    CheckDiagnosticInspector();
    CheckObjectHeaderIcons();
    std::cout << "Inspector selection lifetime passed\n";
}
