#include "EditorShortcutInput.h"

#include <function/renderer/gui/InxGUIContext.h>

#include <imgui.h>
#include <imgui_internal.h>

namespace infernux
{

void EditorShortcutInput::OnRender(InxGUIContext *ctx)
{
    const int frame = ImGui::GetFrameCount();
    if (m_lastFrame == frame)
        return;
    m_lastFrame = frame;

    const auto isDown = [ctx](ImGuiKey key) { return ctx->IsKeyDown(static_cast<int>(key)); };
    const bool ctrl = isDown(ImGuiKey_LeftCtrl) || isDown(ImGuiKey_RightCtrl);
    const bool shift = isDown(ImGuiKey_LeftShift) || isDown(ImGuiKey_RightShift);
    const bool alt = isDown(ImGuiKey_LeftAlt) || isDown(ImGuiKey_RightAlt);
    const bool super = isDown(ImGuiKey_LeftSuper) || isDown(ImGuiKey_RightSuper);
    const auto pressedOnce = [](ImGuiKey key) { return ImGui::IsKeyPressed(key, false); };
    const bool textInputActive = ImGui::GetIO().WantTextInput;
    const bool popupActiveNow = ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId);
    const bool modalActiveNow = ImGui::GetTopMostPopupModal() != nullptr;
    // ImGui may consume Escape and retire a popup before this late-frame input
    // adapter runs. Preserve the previous rendered frame as a capture barrier
    // so the same key edge cannot also execute a panel command.
    const bool popupActive = popupActiveNow || m_popupActivePreviousFrame;
    const bool modalActive = modalActiveNow || m_modalActivePreviousFrame;
    m_popupActivePreviousFrame = popupActiveNow;
    m_modalActivePreviousFrame = modalActiveNow;
    const auto dispatch = [&](const char *chord, bool allowTransientPopup = false) {
        if (routeShortcut)
            routeShortcut(chord, textInputActive, allowTransientPopup ? modalActive : popupActive);
    };

    // The registry, not this input adapter, defines available shortcuts.
    // Only keyboard edges cross into Python; held keys do not repeat commands.
    for (int value = ImGuiKey_Tab; value < ImGuiKey_GamepadStart; ++value) {
        const auto key = static_cast<ImGuiKey>(value);
        if ((key >= ImGuiKey_LeftCtrl && key <= ImGuiKey_RightSuper) || !pressedOnce(key))
            continue;
        if (key == ImGuiKey_KeypadEnter && pressedOnce(ImGuiKey_Enter))
            continue;

        const char *name = ImGui::GetKeyName(key);
        switch (key) {
        case ImGuiKey_LeftArrow:
            name = "Left";
            break;
        case ImGuiKey_RightArrow:
            name = "Right";
            break;
        case ImGuiKey_UpArrow:
            name = "Up";
            break;
        case ImGuiKey_DownArrow:
            name = "Down";
            break;
        case ImGuiKey_KeypadEnter:
            name = "Enter";
            break;
        default:
            break;
        }
        std::string chord;
        if (ctrl)
            chord += "Ctrl+";
        if (shift)
            chord += "Shift+";
        if (alt)
            chord += "Alt+";
        if (super)
            chord += "Super+";
        chord += name;
        const bool historyChord = ctrl && !alt && !super && (key == ImGuiKey_Z || (!shift && key == ImGuiKey_Y));
        dispatch(chord.c_str(), historyChord);
    }
}

} // namespace infernux
