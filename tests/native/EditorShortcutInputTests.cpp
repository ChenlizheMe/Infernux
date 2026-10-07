#include <function/editor/EditorShortcutInput.h>

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
    ImGui::DestroyContext();
}
