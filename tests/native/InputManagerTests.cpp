#include <platform/input/InputManager.h>

#include <SDL3/SDL.h>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

using infernux::InputManager;
using infernux::MotionSensorType;
using infernux::TouchPhase;

namespace
{
SDL_Event KeyEvent(SDL_EventType type, SDL_Scancode scancode)
{
    SDL_Event event{};
    event.type = type;
    event.key.scancode = scancode;
    return event;
}

SDL_Event MouseButtonEvent(SDL_EventType type, Uint8 button)
{
    SDL_Event event{};
    event.type = type;
    event.button.button = button;
    return event;
}

SDL_Event TouchEvent(SDL_EventType type, Uint64 touchId, Uint64 fingerId, float x, float y, float dx = 0.0f,
                     float dy = 0.0f, float pressure = 1.0f, Uint64 timestamp = 123456789)
{
    SDL_Event event{};
    event.type = type;
    event.tfinger.touchID = touchId;
    event.tfinger.fingerID = fingerId;
    event.tfinger.timestamp = timestamp;
    event.tfinger.windowID = 7;
    event.tfinger.x = x;
    event.tfinger.y = y;
    event.tfinger.dx = dx;
    event.tfinger.dy = dy;
    event.tfinger.pressure = pressure;
    return event;
}

bool CheckInputFrameContracts()
{
    auto &input = InputManager::Instance();
    int passed = 0, failed = 0;
    auto check = [&](bool ok, const std::string &name) {
        std::cout << "INPUT_FRAME " << name << (ok ? " PASS\n" : " FAIL\n");
        ok ? ++passed : ++failed;
    };
    constexpr SDL_Scancode keypad[] = {SDL_SCANCODE_KP_0, SDL_SCANCODE_KP_1, SDL_SCANCODE_KP_2,
        SDL_SCANCODE_KP_3, SDL_SCANCODE_KP_4, SDL_SCANCODE_KP_5, SDL_SCANCODE_KP_6,
        SDL_SCANCODE_KP_7, SDL_SCANCODE_KP_8, SDL_SCANCODE_KP_9};
    for (int digit = 0; digit < 10; ++digit) {
        input.ResetAll();
        const int code = InputManager::NameToScancode("Keypad " + std::to_string(digit));
        input.BeginFrame();
        input.ProcessSDLEvent(KeyEvent(SDL_EVENT_KEY_DOWN, keypad[digit]));
        bool ok = code == keypad[digit] && input.GetKey(code) && input.GetKeyDown(code);
        input.BeginFrame();
        input.ProcessSDLEvent(KeyEvent(SDL_EVENT_KEY_UP, keypad[digit]));
        ok = ok && input.GetKeyUp(code) && !input.GetKey(code);
        check(ok, "keypad_" + std::to_string(digit));
    }
    const auto near = [](float a, float b) { return std::abs(a - b) < 1.0e-6f; };
    for (const bool sdl : {false, true}) {
        for (const auto phase : {TouchPhase::Moved, TouchPhase::Ended, TouchPhase::Canceled}) {
            input.ResetAll();
            auto send = [&](TouchPhase next, float x, float y, float dx, float dy, uint64_t time) {
                if (sdl) {
                    const auto type = next == TouchPhase::Began ? SDL_EVENT_FINGER_DOWN :
                        next == TouchPhase::Moved ? SDL_EVENT_FINGER_MOTION :
                        next == TouchPhase::Ended ? SDL_EVENT_FINGER_UP : SDL_EVENT_FINGER_CANCELED;
                    input.ProcessSDLEvent(TouchEvent(type, 1, 2, x, y, dx, dy, 1, time));
                } else {
                    input.ProcessTouchEvent(1, 2, time, 7, x, y, dx, dy, 1, next);
                }
            };
            send(TouchPhase::Began, .1f, .2f, 0, 0, 1'000'000'000);
            input.BeginFrame();
            send(TouchPhase::Moved, .15f, .23f, .05f, .03f, 1'010'000'000);
            send(TouchPhase::Moved, .22f, .22f, .07f, -.01f, 1'020'000'000);
            if (phase != TouchPhase::Moved)
                send(phase, .22f, .22f, 0, 0, 1'030'000'000);
            const auto contact = input.GetTouch(0);
            bool ok = near(contact.deltaX, .12f) && near(contact.deltaY, .02f) &&
                near(contact.x, .22f) && near(contact.y, .22f) && contact.phase == phase;
            input.BeginFrame();
            ok = ok && (phase == TouchPhase::Moved ? input.GetTouchCount() == 1 &&
                input.GetTouch(0).deltaX == 0 && input.GetTouch(0).deltaY == 0 : input.GetTouchCount() == 0);
            check(ok, std::string(sdl ? "sdl_" : "semantic_") + std::to_string(static_cast<int>(phase)));
        }
    }
    input.ResetAll();
    input.ProcessTouchEvent(1, 2, 1, 7, .1f, .2f, 0, 0, 1, TouchPhase::Began);
    input.ProcessTouchEvent(1, 2, 2, 7, .18f, .24f, .08f, .04f, 1, TouchPhase::Moved);
    input.ProcessTouchEvent(1, 2, 3, 7, .18f, .24f, 0, 0, 0, TouchPhase::Ended);
    check(near(input.GetTouch(0).deltaX, .08f) && near(input.GetTouch(0).deltaY, .04f) &&
          input.GetTouch(0).beganThisFrame && near(input.GetTouch(0).beginX, .1f), "same_frame_tap");
    input.ProcessTouchEvent(1, 2, 4, 7, .4f, .5f, 0, 0, 1, TouchPhase::Began);
    check(input.GetTouch(0).deltaX == 0 && input.GetTouch(0).deltaY == 0 &&
          near(input.GetTouch(0).beginX, .4f), "new_contact_resets_displacement");
    input.ProcessTouchEvent(2, 2, 5, 7, .4f, .5f, 0, 0, 1, TouchPhase::Began);
    input.BeginFrame();
    input.ProcessTouchEvent(1, 2, 6, 7, .3f, .6f, -.1f, .1f, 1, TouchPhase::Moved);
    input.ProcessTouchEvent(2, 2, 7, 7, .6f, .3f, .2f, -.2f, 1, TouchPhase::Moved);
    input.ProcessTouchEvent(1, 2, 8, 7, .2f, .7f, -.1f, .1f, 1, TouchPhase::Moved);
    check(input.GetTouchCount() == 2 && near(input.GetTouch(0).deltaX, -.2f) &&
          near(input.GetTouch(0).deltaY, .2f) && near(input.GetTouch(1).deltaX, .2f) &&
          near(input.GetTouch(1).deltaY, -.2f), "devices_accumulate_independently");
    input.ResetAll();
    std::cout << "INPUT_FRAME_SUMMARY passed=" << passed << " failed=" << failed << '\n';
    return failed == 0;
}
} // namespace

int main()
{
    if (!CheckInputFrameContracts())
        return 1;
    auto &input = InputManager::Instance();
    input.ResetAll();

    // Native cursor polling must not steal hover between remote move and click.
    float syntheticX = 0.0f, syntheticY = 0.0f;
    assert(!input.GetSyntheticMousePosition(syntheticX, syntheticY));
    input.SetSyntheticMousePosition(123.0f, 456.0f);
    input.BeginFrame();
    input.BeginFrame();
    assert(input.GetSyntheticMousePosition(syntheticX, syntheticY));
    assert(syntheticX == 123.0f && syntheticY == 456.0f);
    input.ReleaseSyntheticMousePosition();
    assert(!input.GetSyntheticMousePosition(syntheticX, syntheticY));
    input.SetSyntheticMousePosition(3.0f, 4.0f);
    input.ResetAll();
    assert(!input.GetSyntheticMousePosition(syntheticX, syntheticY));

    input.SetCursorVisible(false);
    assert(!input.IsCursorVisible());
    input.SetCursorConfined(true);
    assert(input.IsCursorConfined());
    input.SetCursorLocked(true);
    assert(input.IsCursorLocked());
    input.ProcessKeyEvent(SDL_SCANCODE_ESCAPE, true);
    assert(input.IsCursorLocked() && input.GetKeyDown(SDL_SCANCODE_ESCAPE));
    input.ProcessFocusEvent(true);
    assert(input.IsCursorLocked()); // Player activation retains the game's request.
    input.ResetAll();
    input.SetCursorLocked(false);
    input.SetCursorConfined(false);
    input.SetCursorVisible(true);
    assert(!input.IsCursorLocked());
    assert(!input.IsCursorConfined());
    assert(input.IsCursorVisible());
    assert(!input.WarpCursor(10.0f, 20.0f));

    input.SetEditorGameViewport({10, 10, 100, 100}, {10, 10, 80, 80});
    input.ProcessPointerMotionEvent(20, 20, 0, 0);
    input.SetEditorCursorCaptureAllowed(false);
    input.SetCursorLocked(true);
    assert(!input.IsCursorLocked());
    input.ProcessPointerButtonEvent(0, true);
    input.SetEditorCursorCaptureAllowed(true);
    assert(input.IsCursorLocked());
    assert(std::get<0>(input.GetGameMouseFrameState(0)) == 60);
    assert(std::get<1>(input.GetGameMouseFrameState(0)) == 60);
    assert(input.GetMousePositionX() == 20 && input.GetMousePositionY() == 20);
    assert(input.GetMouseButton(0) && input.GetMouseButtonDown(0));
    input.ProcessPointerButtonEvent(0, false);
    assert(input.GetMouseButtonUp(0));
    input.ProcessPointerButtonEvent(0, true);
    assert(input.GetMouseButtonDown(0));
    input.ProcessPointerButtonEvent(0, false);
    input.ProcessKeyEvent(SDL_SCANCODE_ESCAPE, true);
    assert(input.IsCursorLocked() && input.IsEditorCursorCaptureAllowed());
    assert(input.GetKeyDown(SDL_SCANCODE_ESCAPE));
    input.ConsumeKeyDown(SDL_SCANCODE_ESCAPE);
    assert(input.IsKeyDownConsumed(SDL_SCANCODE_ESCAPE));
    assert(input.GetKeyDown(SDL_SCANCODE_ESCAPE));
    input.BeginFrame();
    assert(!input.IsKeyDownConsumed(SDL_SCANCODE_ESCAPE));
    input.SetEditorCursorCaptureAllowed(false);
    assert(!input.IsCursorLocked());
    input.SetEditorCursorCaptureAllowed(true);
    assert(input.IsCursorLocked());
    input.SetEditorMouseCapture(true);
    SDL_Event activation{};
    activation.type = SDL_EVENT_WINDOW_FOCUS_GAINED;
    input.ProcessSDLEvent(activation);
    assert(!input.IsCursorLocked() && !input.IsEditorMouseCaptureActive());
    input.ProcessPointerMotionEvent(20, 20, 0, 0);
    input.ProcessPointerButtonEvent(0, true);
    assert(input.GetGameMouseButtonDown(0));
    input.SetEditorCursorCaptureAllowed(true);
    assert(input.IsCursorLocked()); // The first viewport click works without Escape.
    input.ProcessPointerButtonEvent(0, false);
    input.ProcessFocusEvent(false);
    input.ProcessFocusEvent(true);
    assert(!input.IsCursorLocked());
    input.SetEditorCursorCaptureAllowed(true);
    assert(input.IsCursorLocked());
    input.SetCursorLocked(false);
    input.SetEditorCursorCaptureAllowed(false);
    input.SetEditorCursorCaptureAllowed(true);
    assert(!input.IsCursorLocked());
    input.BeginFrame();
    input.ProcessPointerMotionEvent(5, 5, 0, 0);
    input.ProcessPointerButtonEvent(0, true);
    assert(input.GetMouseButtonDown(0) && !input.GetGameMouseButtonDown(0));
    input.ProcessPointerMotionEvent(20, 20, 0, 0);
    assert(!input.GetGameMouseButton(0));
    input.ProcessPointerButtonEvent(0, false);
    assert(input.GetMouseButtonUp(0));
    input.BeginFrame();
    input.ProcessPointerButtonEvent(0, true);
    assert(input.GetGameMouseButtonDown(0));
    input.ProcessPointerMotionEvent(150, 150, 0, 0);
    assert(input.GetMouseButton(0) && input.GetGameMouseButton(0));
    input.ProcessPointerButtonEvent(0, false);
    assert(input.GetMouseButtonUp(0));
    input.SetWindow(nullptr);
    input.ResetAll();

    const uint64_t previousFrame = input.GetFrameIndex();
    input.BeginFrame();
    assert(input.GetFrameIndex() == previousFrame + 1);
    auto keyDown = KeyEvent(SDL_EVENT_KEY_DOWN, SDL_SCANCODE_W);
    auto keyUp = KeyEvent(SDL_EVENT_KEY_UP, SDL_SCANCODE_W);
    input.TrackSyntheticEvent(keyDown);
    input.ProcessSDLEvent(keyDown);
    assert(input.HasSyntheticGameplayInput());
    assert(input.GetKeyDown(SDL_SCANCODE_W));
    assert(input.GetKey(SDL_SCANCODE_W));
    assert(input.AnyKeyDown());
    input.BeginFrame();
    assert(input.HasSyntheticGameplayInput());
    input.TrackSyntheticEvent(keyUp);
    input.ProcessSDLEvent(keyUp);
    assert(!input.GetKeyDown(SDL_SCANCODE_W));
    assert(input.GetKeyUp(SDL_SCANCODE_W));
    assert(!input.GetKey(SDL_SCANCODE_W));
    assert(!input.AnyKeyDown());
    assert(input.IsSyntheticInputFrame());

    input.BeginFrame();
    assert(!input.HasSyntheticGameplayInput());
    assert(!input.GetKeyDown(SDL_SCANCODE_W));
    assert(!input.GetKeyUp(SDL_SCANCODE_W));

    // Background MCP validation owns its synthetic held state independently
    // from the physical window focus lifecycle.
    input.TrackSyntheticEvent(keyDown);
    input.ProcessSDLEvent(keyDown);
    auto focusLost = SDL_Event{};
    focusLost.type = SDL_EVENT_WINDOW_FOCUS_LOST;
    input.ProcessSDLEvent(focusLost);
    assert(input.HasSyntheticGameplayInput());
    assert(input.GetKey(SDL_SCANCODE_W));
    input.TrackSyntheticEvent(keyUp);
    input.ProcessSDLEvent(keyUp);
    assert(input.HasSyntheticGameplayInput());
    assert(input.IsSyntheticInputFrame());
    assert(!input.GetKey(SDL_SCANCODE_W));
    input.BeginFrame();
    assert(!input.HasSyntheticGameplayInput());

    // A physical key still releases normally on focus loss.
    input.ProcessSDLEvent(keyDown);
    assert(input.GetKey(SDL_SCANCODE_W));
    input.ProcessSDLEvent(focusLost);
    assert(!input.GetKey(SDL_SCANCODE_W));

    input.ProcessSDLEvent(keyDown);
    input.ProcessSDLEvent(keyDown);
    assert(input.GetKeyDown(SDL_SCANCODE_W));
    assert(input.GetKey(SDL_SCANCODE_W));
    input.BeginFrame();
    input.ProcessSDLEvent(keyDown);
    assert(!input.GetKeyDown(SDL_SCANCODE_W));
    input.ProcessSDLEvent(keyUp);
    assert(input.GetKeyUp(SDL_SCANCODE_W));

    input.BeginFrame();
    auto mouseDown = MouseButtonEvent(SDL_EVENT_MOUSE_BUTTON_DOWN, SDL_BUTTON_LEFT);
    auto mouseUp = MouseButtonEvent(SDL_EVENT_MOUSE_BUTTON_UP, SDL_BUTTON_LEFT);
    input.ProcessSDLEvent(mouseDown);
    input.ProcessSDLEvent(mouseUp);
    assert(input.GetMouseButtonDown(0));
    assert(input.GetMouseButtonUp(0));
    assert(!input.GetMouseButton(0));

    // Touch contacts persist across frames, retain stable first-contact order,
    // and publish terminal phases for exactly one frame.
    input.ResetAll();
    input.BeginFrame();
    input.ProcessSDLEvent(TouchEvent(SDL_EVENT_FINGER_DOWN, 3, 101, 0.25f, 0.75f, 0.0f, 0.0f, 0.6f));
    input.ProcessSDLEvent(TouchEvent(SDL_EVENT_FINGER_DOWN, 3, 202, 0.80f, 0.20f));
    assert(input.GetTouchCount() == 2);
    assert(input.GetTouch(0).fingerId == 101);
    assert(input.GetTouch(0).phase == TouchPhase::Began);
    assert(input.GetTouch(0).beganThisFrame);
    assert(input.GetTouch(0).beginX == 0.25f);
    assert(input.GetTouch(0).beginY == 0.75f);
    assert(input.GetTouch(0).pressure == 0.6f);
    assert(input.GetTouch(0).isPrimary);
    assert(input.GetTouch(1).fingerId == 202);
    assert(!input.GetTouch(1).isPrimary);

    input.BeginFrame();
    assert(input.GetTouchCount() == 2);
    assert(input.GetTouch(0).phase == TouchPhase::Stationary);
    assert(!input.GetTouch(0).beganThisFrame);
    input.ProcessSDLEvent(TouchEvent(SDL_EVENT_FINGER_MOTION, 3, 101, 0.30f, 0.70f, 0.05f, -0.05f, 0.7f, 139456789));
    assert(input.GetTouch(0).phase == TouchPhase::Moved);
    assert(input.GetTouch(0).deltaX == 0.05f);
    assert(input.GetTouch(0).deltaTime > 0.015f);
    assert(input.GetTouch(0).deltaTime < 0.017f);
    input.ProcessSDLEvent(TouchEvent(SDL_EVENT_FINGER_UP, 3, 202, 0.80f, 0.20f));
    assert(input.GetTouch(1).phase == TouchPhase::Ended);

    input.BeginFrame();
    assert(input.GetTouchCount() == 1);
    assert(input.GetTouch(0).fingerId == 101);
    input.ProcessSDLEvent(focusLost);
    assert(input.GetTouchCount() == 1);
    assert(input.GetTouch(0).phase == TouchPhase::Canceled);
    input.BeginFrame();
    assert(input.GetTouchCount() == 0);

    bool invalidTouchRejected = false;
    try {
        static_cast<void>(input.GetTouch(0));
    } catch (const std::out_of_range &) {
        invalidTouchRejected = true;
    }
    assert(invalidTouchRejected);

    // A press and release in one frame retains its initial location so UI
    // dispatch can replay both transitions without treating a drag as a tap.
    input.BeginFrame();
    input.ProcessTouchEvent(3, 303, 1000, 0, 0.25f, 0.75f, 0.0f, 0.0f, 0.6f, TouchPhase::Began);
    input.ProcessTouchEvent(3, 303, 2000, 0, 0.80f, 0.20f, 0.55f, -0.55f, 0.0f, TouchPhase::Ended);
    assert(input.GetTouchCount() == 1);
    assert(input.GetTouch(0).phase == TouchPhase::Ended);
    assert(input.GetTouch(0).beganThisFrame);
    assert(input.GetTouch(0).beginX == 0.25f);
    assert(input.GetTouch(0).beginY == 0.75f);
    input.BeginFrame();
    assert(input.GetTouchCount() == 0);

    // SDL compatibility mouse events generated from touch must not create a
    // second gameplay action beside the first-class touch stream.
    input.BeginFrame();
    auto compatibilityMouse = MouseButtonEvent(SDL_EVENT_MOUSE_BUTTON_DOWN, SDL_BUTTON_LEFT);
    compatibilityMouse.button.which = SDL_TOUCH_MOUSEID;
    input.ProcessSDLEvent(compatibilityMouse);
    assert(!input.GetMouseButtonDown(0));
    assert(!input.GetMouseButton(0));

    // Non-SDL hosts feed the same semantic state machine directly. Browser
    // key, pointer, text, wheel, and touch events must therefore preserve the
    // desktop edge and frame-lifecycle contract.
    input.ResetAll();
    input.BeginFrame();
    input.ProcessKeyEvent(SDL_SCANCODE_A, true);
    input.ProcessPointerButtonEvent(1, true);
    input.ProcessPointerMotionEvent(120.0f, 80.0f, 4.0f, -2.0f);
    input.ProcessScrollEvent(0.5f, -1.5f);
    input.ProcessTextInputEvent("web");
    input.ProcessTouchEvent(8, 42, 999, 0, 0.2f, 0.3f, 0.0f, 0.0f, 0.75f, TouchPhase::Began);
    assert(input.GetKeyDown(SDL_SCANCODE_A));
    assert(input.GetKey(SDL_SCANCODE_A));
    assert(input.GetMouseButtonDown(1));
    assert(input.GetMouseButton(1));
    assert(input.GetMousePositionX() == 120.0f);
    assert(input.GetMousePositionY() == 80.0f);
    assert(input.GetMouseDeltaX() == 4.0f);
    assert(input.GetMouseDeltaY() == -2.0f);
    assert(input.GetMouseScrollDeltaX() == 0.5f);
    assert(input.GetMouseScrollDeltaY() == -1.5f);
    assert(input.GetInputString() == "web");
    assert(input.GetTouchCount() == 1);
    assert(input.GetTouch(0).phase == TouchPhase::Began);

    input.ProcessTouchEvent(8, 42, 999, 0, 0.2f, 0.3f, 0.0f, 0.0f, 0.75f, TouchPhase::Moved, 0.04f, 0.06f, true);
    assert(input.GetTouch(0).contactWidth == 0.04f);
    assert(input.GetTouch(0).contactHeight == 0.06f);
    assert(input.GetTouch(0).isPrimary);

    input.ProcessScreenMetrics(1080, 2400, 1080, 2400, 1.0f, 0, 80, 1080, 2240, true, 700);
    const auto &screen = input.GetScreenState();
    assert(screen.logicalWidth == 1080);
    assert(screen.logicalHeight == 2400);
    assert(screen.safeAreaY == 80);
    assert(screen.safeAreaHeight == 2240);
    assert(screen.keyboardInsetKnown);
    assert(screen.keyboardInset == 700);
    assert(input.GetTouch(0).phase == TouchPhase::Canceled);
    assert(input.GetTouch(0).cancelReason == "viewport_changed");

    // Crossing monitors at 150/200/250% changes framebuffer density without
    // changing the logical viewport. The screen revision must advance while
    // an active touch remains valid in the same logical coordinate system.
    input.BeginFrame();
    input.ProcessTouchEvent(8, 42, 1000, 0, 0.2f, 0.3f, 0.0f, 0.0f, 0.75f, TouchPhase::Began);
    struct DpiCase
    {
        float scale;
        int framebufferWidth;
        int framebufferHeight;
    };
    const std::array<DpiCase, 3> dpiCases{{
        {1.5f, 1620, 3600},
        {2.0f, 2160, 4800},
        {2.5f, 2700, 6000},
    }};
    uint64_t previousScreenRevision = input.GetScreenState().revision;
    for (const DpiCase &dpiCase : dpiCases) {
        input.ProcessScreenMetrics(1080, 2400, dpiCase.framebufferWidth, dpiCase.framebufferHeight, dpiCase.scale, 0,
                                   80, 1080, 2240, true, 700);
        const auto &dpiScreen = input.GetScreenState();
        assert(dpiScreen.pixelRatio == dpiCase.scale);
        assert(dpiScreen.framebufferWidth == dpiCase.framebufferWidth);
        assert(dpiScreen.framebufferHeight == dpiCase.framebufferHeight);
        assert(dpiScreen.revision == previousScreenRevision + 1);
        assert(input.GetTouch(0).phase != TouchPhase::Canceled);
        previousScreenRevision = dpiScreen.revision;
    }

    bool invalidPixelRatioRejected = false;
    try {
        input.ProcessScreenMetrics(1080, 2400, 1080, 2400, 0.0f, 0, 80, 1080, 2240, true, 700);
    } catch (const std::invalid_argument &) {
        invalidPixelRatioRejected = true;
    }
    assert(invalidPixelRatioRejected);

    input.BeginFrame();
    input.ProcessKeyEvent(SDL_SCANCODE_A, false);
    input.ProcessPointerButtonEvent(1, false);
    input.ProcessTouchEvent(8, 42, 1000, 0, 0.2f, 0.3f, 0.0f, 0.0f, 0.0f, TouchPhase::Ended);
    assert(input.GetKeyUp(SDL_SCANCODE_A));
    assert(input.GetMouseButtonUp(1));
    assert(input.GetTouch(0).phase == TouchPhase::Ended);

    // Mobile motion sensors share the frame snapshot contract. SDL reports
    // acceleration in m/s²; the public engine API exposes Unity-compatible g.
    input.BeginFrame();
    input.ProcessMotionSensorEvent(MotionSensorType::Accelerometer, 1'000'000'000, SDL_STANDARD_GRAVITY, 0.0f,
                                   -SDL_STANDARD_GRAVITY);
    input.ProcessMotionSensorEvent(MotionSensorType::Accelerometer, 1'020'000'000, 0.0f, SDL_STANDARD_GRAVITY, 0.0f);
    assert(input.HasAccelerometer());
    assert(input.GetAccelerationEvents().size() == 2);
    assert(input.GetAccelerationEvents()[0].acceleration[0] == 1.0f);
    assert(input.GetAccelerationEvents()[0].acceleration[2] == -1.0f);
    assert(input.GetAccelerationEvents()[1].deltaTime > 0.019f);
    assert(input.GetAccelerationEvents()[1].deltaTime < 0.021f);
    assert(input.GetAcceleration()[1] == 1.0f);

    input.ProcessMotionSensorEvent(MotionSensorType::Gyroscope, 1'020'000'000, 0.25f, -0.5f, 1.0f);
    assert(input.HasGyroscope());
    assert(input.GetGyroscopeRotationRate()[0] == 0.25f);
    assert(input.GetGyroscopeRotationRate()[1] == -0.5f);
    assert(input.GetGyroscopeRotationRate()[2] == 1.0f);

    input.BeginFrame();
    assert(input.GetAccelerationEvents().empty());
    assert(input.GetAcceleration()[1] == 1.0f);

    // Focus acquisition and Editor input still require GUI; owned gameplay does not.
    input.ResetAll();
    input.SetEditorGameViewport({10, 10, 100, 100}, {10, 10, 80, 80});
    auto gamePress = MouseButtonEvent(SDL_EVENT_MOUSE_BUTTON_DOWN, SDL_BUTTON_LEFT);
    gamePress.button.x = gamePress.button.y = 20;
    input.SetEditorCursorCaptureAllowed(false);
    input.ProcessSDLEvent(gamePress);
    assert(!input.IsEditorGameInputEvent(gamePress));
    input.SetEditorCursorCaptureAllowed(true);
    for (Uint8 button = SDL_BUTTON_LEFT; button <= SDL_BUTTON_X2; ++button) {
        input.ResetAll();
        gamePress.button.button = button;
        input.ProcessSDLEvent(gamePress);
        assert(input.IsEditorGameInputEvent(gamePress));
        auto release = gamePress;
        release.type = SDL_EVENT_MOUSE_BUTTON_UP;
        input.ProcessSDLEvent(release);
        assert(input.IsEditorGameInputEvent(release));
    }
    input.ResetAll();
    gamePress.button.button = SDL_BUTTON_LEFT;
    gamePress.button.x = 95; // Image pixels clipped out of the interactive viewport.
    input.ProcessSDLEvent(gamePress);
    assert(!input.IsEditorGameInputEvent(gamePress));
    auto editorRelease = gamePress;
    editorRelease.type = SDL_EVENT_MOUSE_BUTTON_UP;
    editorRelease.button.x = 20;
    input.ProcessSDLEvent(editorRelease);
    assert(!input.IsEditorGameInputEvent(editorRelease)); // Moving in cannot transfer a press.
    SDL_Event wheel{};
    wheel.type = SDL_EVENT_MOUSE_WHEEL;
    assert(input.IsEditorGameInputEvent(wheel));
    input.ProcessPointerMotionEvent(95, 20, 0, 0);
    assert(!input.IsEditorGameInputEvent(wheel));
    auto gameKey = KeyEvent(SDL_EVENT_KEY_DOWN, SDL_SCANCODE_W);
    assert(input.IsEditorGameInputEvent(gameKey));
    gameKey.type = SDL_EVENT_KEY_UP;
    assert(input.IsEditorGameInputEvent(gameKey));
    for (SDL_Keymod modifier : {SDL_KMOD_LCTRL, SDL_KMOD_LALT, SDL_KMOD_LGUI}) {
        gameKey.key.mod = modifier;
        assert(!input.IsEditorGameInputEvent(gameKey));
    }
    gameKey.key.mod = SDL_KMOD_NONE;
    for (SDL_Scancode code : {SDL_SCANCODE_ESCAPE, SDL_SCANCODE_DELETE, SDL_SCANCODE_F2, SDL_SCANCODE_F13,
                              SDL_SCANCODE_LCTRL, SDL_SCANCODE_RGUI}) {
        gameKey.key.scancode = code;
        assert(!input.IsEditorGameInputEvent(gameKey));
    }
    gameKey.key.scancode = SDL_SCANCODE_LSHIFT;
    assert(input.IsEditorGameInputEvent(gameKey));
    input.SetEditorMouseCapture(true);
    assert(!input.IsEditorGameInputEvent(gameKey));
    input.SetEditorMouseCapture(false);

    // Editor-owned input retains raw diagnostics without entering gameplay.
    input.ResetAll();
    gamePress.button.x = gamePress.button.y = 20;
    input.ProcessSDLEvent(gamePress, false);
    assert(input.GetMouseButtonDown(0) && !input.GetGameMouseButtonDown(0));
    auto release = gamePress;
    release.type = SDL_EVENT_MOUSE_BUTTON_UP;
    input.ProcessSDLEvent(release, false);
    assert(input.GetMouseButtonUp(0) && !input.GetGameMouseButtonUp(0));
    assert(!std::get<6>(input.GetGameMouseFrameState(0)));
    input.BeginFrame();
    input.ProcessSDLEvent(gamePress, true);
    assert(input.GetGameMouseButtonDown(0));
    input.BeginFrame();
    release.button.x = release.button.y = 200;
    input.ProcessSDLEvent(release, false);
    assert(input.GetGameMouseButtonUp(0) && std::get<6>(input.GetGameMouseFrameState(0)));
    input.BeginFrame();
    assert(!input.GetGameMouseButtonUp(0));

    input.ProcessKeyEvent(SDL_SCANCODE_A, true, false);
    assert(input.GetKey(SDL_SCANCODE_A) && input.GetKeyDown(SDL_SCANCODE_A));
    assert(!input.GetGameKey(SDL_SCANCODE_A) && !input.GetGameKeyDown(SDL_SCANCODE_A));
    assert(input.AnyKey() && !input.AnyGameKey(true));
    input.ProcessKeyEvent(SDL_SCANCODE_A, true, true);
    assert(!input.GetGameKey(SDL_SCANCODE_A)); // Focus cannot transfer an existing press.
    input.ProcessKeyEvent(SDL_SCANCODE_W, true, true);
    assert(input.GetGameKey(SDL_SCANCODE_W) && input.AnyGameKey(true));
    input.BeginFrame();
    input.ProcessKeyEvent(SDL_SCANCODE_W, false, false);
    input.ProcessKeyEvent(SDL_SCANCODE_A, false, false);
    assert(input.GetGameKeyUp(SDL_SCANCODE_W) && !input.GetGameKeyUp(SDL_SCANCODE_A));
    input.ProcessScrollEvent(.5f, -2.f, false);
    input.ProcessTextInputEvent("Editor", false);
    assert(input.GetMouseScrollDeltaY() == -2 && input.GetGameScrollDeltaY() == 0);
    assert(input.GetInputString() == "Editor" && input.GetGameInputString().empty());
    input.ProcessScrollEvent(1, 1, true);
    input.ProcessTextInputEvent("Game", true);
    assert(input.GetMouseScrollDeltaX() == 1.5f && input.GetGameScrollDeltaX() == 1);
    assert(input.GetInputString() == "EditorGame" && input.GetGameInputString() == "Game");
    assert(std::get<3>(input.GetGameMouseFrameState(0)) == 1);
    input.ProcessKeyEvent(SDL_SCANCODE_A, true, false);
    const auto syntheticPress = KeyEvent(SDL_EVENT_KEY_DOWN, SDL_SCANCODE_A);
    input.TrackSyntheticEvent(syntheticPress);
    input.ProcessSDLEvent(syntheticPress);
    assert(input.GetGameKeyDown(SDL_SCANCODE_A));
    input.ProcessFocusEvent(false);
    assert(input.GetGameKey(SDL_SCANCODE_A));
    const auto syntheticRelease = KeyEvent(SDL_EVENT_KEY_UP, SDL_SCANCODE_A);
    input.TrackSyntheticEvent(syntheticRelease);
    input.ProcessSDLEvent(syntheticRelease);
    assert(input.GetGameKeyUp(SDL_SCANCODE_A));
    input.BeginFrame();
    assert(input.GetGameScrollDeltaY() == 0 && input.GetGameInputString().empty());
    input.SetWindow(nullptr);
    input.ResetAll();
    input.ProcessKeyEvent(SDL_SCANCODE_W, true);
    assert(input.GetGameKey(SDL_SCANCODE_W) && input.GetGameKeyDown(SDL_SCANCODE_W));
    input.ResetAll();
    return 0;
}
