#include "WebKeyboard.h"
#include <platform/input/InputManager.h>

#include <cassert>
#include <iostream>
#include <string>

int main()
{
    using infernux::web::BrowserCodeToScancode;
    auto &input = infernux::InputManager::Instance();
    const auto check = [&](const std::string &code, int expected) {
        const int actual = BrowserCodeToScancode(code);
        assert(actual == expected);
        input.BeginFrame();
        input.ProcessKeyEvent(actual, true);
        assert(input.GetKey(expected) && input.GetKeyDown(expected));
        input.BeginFrame();
        input.ProcessKeyEvent(actual, true); // Browser auto-repeat must not create a second down edge.
        assert(input.GetKey(expected) && !input.GetKeyDown(expected));
        input.ProcessKeyEvent(actual, false);
        assert(!input.GetKey(expected) && input.GetKeyUp(expected));
        input.BeginFrame();
        assert(!input.GetKeyUp(expected));
    };
    for (char letter = 'A'; letter <= 'Z'; ++letter)
        check(std::string("Key") + letter, SDL_SCANCODE_A + letter - 'A');
    for (int digit = 1; digit <= 9; ++digit) {
        check("Digit" + std::to_string(digit), SDL_SCANCODE_1 + digit - 1);
        check("Numpad" + std::to_string(digit), SDL_SCANCODE_KP_1 + digit - 1);
    }
    for (int number = 1; number <= 24; ++number)
        check("F" + std::to_string(number),
              number <= 12 ? SDL_SCANCODE_F1 + number - 1 : SDL_SCANCODE_F13 + number - 13);
    check("Digit0", SDL_SCANCODE_0);
    check("Numpad0", SDL_SCANCODE_KP_0);
    check("Enter", SDL_SCANCODE_RETURN);
    check("NumpadEnter", SDL_SCANCODE_KP_ENTER);
    check("Delete", SDL_SCANCODE_DELETE);
    check("NumpadDecimal", SDL_SCANCODE_KP_PERIOD);
    check("ControlLeft", SDL_SCANCODE_LCTRL);
    check("ControlRight", SDL_SCANCODE_RCTRL);
    check("ShiftLeft", SDL_SCANCODE_LSHIFT);
    check("ShiftRight", SDL_SCANCODE_RSHIFT);
    check("AltLeft", SDL_SCANCODE_LALT);
    check("AltRight", SDL_SCANCODE_RALT);
    check("MetaLeft", SDL_SCANCODE_LGUI);
    check("MetaRight", SDL_SCANCODE_RGUI);
    check("Semicolon", SDL_SCANCODE_SEMICOLON);
    check("IntlBackslash", SDL_SCANCODE_NONUSBACKSLASH);
    check("IntlYen", SDL_SCANCODE_INTERNATIONAL3);
    check("IntlRo", SDL_SCANCODE_INTERNATIONAL1);
    check("BrowserBack", SDL_SCANCODE_AC_BACK);
    // Left/right modifiers remain separate. Losing focus releases both.
    input.ProcessKeyEvent(BrowserCodeToScancode("ShiftLeft"), true);
    input.ProcessKeyEvent(BrowserCodeToScancode("ShiftRight"), true);
    input.ProcessKeyEvent(BrowserCodeToScancode("ShiftLeft"), false);
    assert(!input.GetKey(SDL_SCANCODE_LSHIFT) && input.GetKey(SDL_SCANCODE_RSHIFT));
    input.ProcessFocusEvent(false);
    assert(!input.GetKey(SDL_SCANCODE_RSHIFT) && !input.GetKeyDown(SDL_SCANCODE_RSHIFT));
    for (const auto *code : {"", "Unidentified", "Keya", "KeyAA", "Digit10", "Numpad10", "F0", "F01", "F25", "F1x"}) {
        assert(BrowserCodeToScancode(code) == -1);
        input.ProcessKeyEvent(BrowserCodeToScancode(code), true);
    }
    for (int scan = 0; scan < infernux::INPUT_MAX_KEYS; ++scan)
        assert(!input.GetKey(scan));
    std::cout << "Web keyboard mapping, edges, repeat, modifiers and focus release passed\n";
}
