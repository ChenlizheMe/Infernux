#include <function/editor/EditorShortcutInput.h>
#include <platform/input/InputManager.h>

#include <cassert>
#include <string>
#include <vector>

int main()
{
    ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = ImVec2(640, 480);
    io.DeltaTime = 1.0f / 60.0f;
    io.ConfigInputTrickleEventQueue = false;
    unsigned char *pixels;
    int width, height;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    infernux::InxGUIContext context;
    infernux::EditorShortcutInput input;
    std::vector<std::string> chords;
    input.routeShortcut = [&](const std::string &chord, bool, bool) {
        chords.push_back(chord);
        return true;
    };
    auto frame = [&] {
        ImGui::NewFrame();
        input.OnRender(&context);
        input.OnRender(&context); // One route per frame, even with duplicate presenters.
        ImGui::EndFrame();
    };
    frame();
    io.AddKeyEvent(ImGuiKey_F9, true);
    frame();
    assert((chords == std::vector<std::string>{"F9"}));
    frame(); // Holding the key never replays a command.
    assert(chords.size() == 1);
    io.AddKeyEvent(ImGuiKey_F9, false);
    frame();
    io.AddKeyEvent(ImGuiKey_LeftCtrl, true);
    io.AddKeyEvent(ImGuiMod_Ctrl, true);
    io.AddKeyEvent(ImGuiKey_LeftAlt, true);
    io.AddKeyEvent(ImGuiMod_Alt, true);
    io.AddKeyEvent(ImGuiKey_K, true);
    frame();
    assert(chords.back() == "Ctrl+Alt+K");
    assert(chords.size() == 2);
    io.AddKeyEvent(ImGuiKey_K, false);
    io.AddKeyEvent(ImGuiKey_LeftCtrl, false);
    io.AddKeyEvent(ImGuiMod_Ctrl, false);
    io.AddKeyEvent(ImGuiKey_LeftAlt, false);
    io.AddKeyEvent(ImGuiMod_Alt, false);
    frame();
    io.AddKeyEvent(ImGuiKey_LeftArrow, true);
    io.AddKeyEvent(ImGuiKey_Enter, true);
    io.AddKeyEvent(ImGuiKey_KeypadEnter, true);
    frame();
    assert((chords == std::vector<std::string>{"F9", "Ctrl+Alt+K", "Left", "Enter"}));
    io.AddKeyEvent(ImGuiKey_LeftArrow, false);
    io.AddKeyEvent(ImGuiKey_Enter, false);
    io.AddKeyEvent(ImGuiKey_KeypadEnter, false);
    frame();
    const auto press = [&](std::initializer_list<ImGuiKey> keys, const char *expected) {
        const size_t before = chords.size();
        for (auto key : keys)
            io.AddKeyEvent(key, true);
        frame();
        assert(chords.size() == before + 1);
        assert(chords.back() == expected);
        for (auto key : keys)
            io.AddKeyEvent(key, false);
        frame();
    };
    press({ImGuiKey_LeftCtrl, ImGuiMod_Ctrl, ImGuiKey_LeftShift, ImGuiMod_Shift, ImGuiKey_P}, "Ctrl+Shift+P");
    press({ImGuiKey_F}, "F");
    press({ImGuiKey_LeftCtrl, ImGuiMod_Ctrl, ImGuiKey_F}, "Ctrl+F");
    press({ImGuiKey_LeftCtrl, ImGuiMod_Ctrl, ImGuiKey_LeftShift, ImGuiMod_Shift, ImGuiKey_F}, "Ctrl+Shift+F");
    press({ImGuiKey_LeftAlt, ImGuiMod_Alt, ImGuiKey_LeftArrow}, "Alt+Left");
    press({ImGuiKey_LeftAlt, ImGuiMod_Alt, ImGuiKey_RightArrow}, "Alt+Right");
    press({ImGuiKey_LeftCtrl, ImGuiMod_Ctrl, ImGuiKey_Z}, "Ctrl+Z");
    press({ImGuiKey_LeftCtrl, ImGuiMod_Ctrl, ImGuiKey_LeftShift, ImGuiMod_Shift, ImGuiKey_Z}, "Ctrl+Shift+Z");
    press({ImGuiKey_LeftCtrl, ImGuiMod_Ctrl, ImGuiKey_Y}, "Ctrl+Y");

    auto &gameInput = infernux::InputManager::Instance();
    const int escape = infernux::InputManager::NameToScancode("escape");
    gameInput.BeginFrame();
    gameInput.ProcessKeyEvent(escape, true);
    gameInput.ConsumeKeyDown(escape);
    const size_t beforeEscape = chords.size();
    io.AddKeyEvent(ImGuiKey_Escape, true);
    frame();
    assert(chords.size() == beforeEscape);
    io.AddKeyEvent(ImGuiKey_Escape, false);
    frame();
    gameInput.BeginFrame();
    gameInput.ProcessKeyEvent(escape, false);
    gameInput.ProcessKeyEvent(escape, true);
    press({ImGuiKey_Escape}, "Escape");

    const auto mouseFrame = [&](bool captured) {
        infernux::EditorShortcutInput::PrepareFrame(captured);
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(400, 300));
        ImGui::Begin("Hierarchy");
        const bool selected = ImGui::Selectable("Object");
        const auto itemOrigin = ImGui::GetCursorScreenPos();
        ImGui::PushClipRect(itemOrigin, ImVec2(itemOrigin.x + 25, itemOrigin.y + 25), true);
        ImGui::InvisibleButton("Clipped viewport", ImVec2(100, 100));
        const auto visible = context.GetItemVisibleRect();
        assert(visible[2] == 25 && visible[3] == 25);
        ImGui::PopClipRect();
        ImGui::End();
        ImGui::EndFrame();
        return selected;
    };
    mouseFrame(false);
    io.AddMousePosEvent(30, 35);
    io.AddMouseButtonEvent(0, true);
    io.AddMouseWheelEvent(0, 1);
    io.AddKeyEvent(ImGuiKey_Escape, true);
    assert(!mouseFrame(true));
    assert(!io.MouseClicked[0] && !io.MouseDown[0] && io.MouseWheel == 0);
    assert(ImGui::IsKeyPressed(ImGuiKey_Escape, false));
    io.AddKeyEvent(ImGuiKey_Escape, false);
    io.AddMousePosEvent(30, 35);
    io.AddMouseButtonEvent(0, true);
    mouseFrame(false);
    assert(io.MouseClicked[0]);
    io.AddMouseButtonEvent(0, false);
    assert(mouseFrame(false));

    ImVec2 viewportPoint;
    const auto gameClickFrame = [&](int button) {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(400, 300));
        ImGui::Begin("Game input");
        ImGui::BeginChild("Viewport", ImVec2(220, 200));
        context.InvisibleButton("Game surface", 200, 150);
        const auto origin = ImGui::GetItemRectMin();
        viewportPoint = ImVec2(origin.x + 20, origin.y + 20);
        const bool clicked = context.IsItemClicked(button);
        ImGui::EndChild();
        ImGui::End();
        ImGui::EndFrame();
        return clicked;
    };
    gameClickFrame(0);
    io.AddMousePosEvent(viewportPoint.x, viewportPoint.y);
    gameClickFrame(0);
    for (int button = 0; button < 5; ++button) {
        io.AddMouseButtonEvent(button, true);
        assert(gameClickFrame(button));
        io.AddMouseButtonEvent(button, false);
        gameClickFrame(button);
    }
    ImGui::DestroyContext();
}
