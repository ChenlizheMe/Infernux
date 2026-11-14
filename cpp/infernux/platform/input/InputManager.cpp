/**
 * @file InputManager.cpp
 * @brief Implementation of the unified input state manager.
 *
 * Uses SDL3 event types:
 *   SDL_EVENT_KEY_DOWN / SDL_EVENT_KEY_UP
 *   SDL_EVENT_MOUSE_BUTTON_DOWN / SDL_EVENT_MOUSE_BUTTON_UP
 *   SDL_EVENT_MOUSE_MOTION
 *   SDL_EVENT_MOUSE_WHEEL
 *   SDL_EVENT_TEXT_INPUT
 *   SDL_EVENT_FINGER_DOWN / SDL_EVENT_FINGER_MOTION /
 *   SDL_EVENT_FINGER_UP / SDL_EVENT_FINGER_CANCELED
 *
 * SDL_EVENT_WINDOW_FOCUS_LOST
 */

#include "InputManager.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <core/log/InxLog.h>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace infernux
{

namespace
{
thread_local InputFrameLease s_readInput;
thread_local uint64_t s_readWorldId = 0;
thread_local std::shared_ptr<InputFrameReadState> s_readState;
} // namespace

InputReadScope::InputReadScope(InputFrameLease lease, uint64_t worldId) noexcept
    : m_previous(std::move(s_readInput)), m_previousWorld(s_readWorldId), m_previousReadState(std::move(s_readState))
{
    s_readInput = std::move(lease);
    s_readWorldId = worldId;
    s_readState = s_readInput && worldId ? InputManager::Instance().FindWorldReadState(worldId) : nullptr;
}
InputReadScope::~InputReadScope()
{
    s_readInput = std::move(m_previous);
    s_readWorldId = m_previousWorld;
    s_readState = std::move(m_previousReadState);
}
const InputFrameSample *InputReadScope::Current() noexcept
{
    if (!s_readState && s_readWorldId)
        s_readState = InputManager::Instance().FindWorldReadState(s_readWorldId);
    if (s_readInput && s_readState && s_readState->overrideValue &&
        s_readState->overrideValue->sampleIdentity == s_readInput->sampleIdentity)
        return s_readState->overrideValue.get();
    return s_readInput.get();
}
uint64_t InputReadScope::WorldId() noexcept
{
    return s_readWorldId;
}

std::shared_ptr<InputFrameReadState> InputManager::AcquireWorldReadState(uint64_t worldId)
{
    auto &state = m_worldReadStates[worldId];
    if (!state)
        state = std::make_shared<InputFrameReadState>();
    return state;
}
std::shared_ptr<InputFrameReadState> InputManager::FindWorldReadState(uint64_t worldId) noexcept
{
    const auto found = m_worldReadStates.find(worldId);
    return found == m_worldReadStates.end() ? nullptr : found->second;
}
void InputManager::RetireWorldInput(uint64_t worldId) noexcept
{
    if (const auto found = m_worldReadStates.find(worldId); found != m_worldReadStates.end()) {
        found->second->retired = true;
        m_worldReadStates.erase(found);
    }
}
std::shared_ptr<InputFrameReadState> InputManager::RetainWorldInputReadState()
{
    if (!s_readInput || !s_readWorldId)
        throw std::logic_error("World input read owner requires a logical input scope");
    return AcquireWorldReadState(s_readWorldId);
}
InputFrameSample *InputManager::OverrideReadFrame()
{
    const auto *sample = InputReadScope::Current();
    if (!sample)
        return nullptr;
    if (!s_readState && s_readWorldId)
        s_readState = AcquireWorldReadState(s_readWorldId);
    if (!s_readState)
        throw std::logic_error("Mutable sampled input requires a World owner");
    if (!s_readState->overrideValue || s_readState->overrideValue->sampleIdentity != sample->sampleIdentity)
        s_readState->overrideValue = std::make_shared<InputFrameSample>(*sample);
    ++s_readState->revision;
    return s_readState->overrideValue.get();
}
uint64_t InputManager::GetInputReadRevision() const noexcept
{
    if (InputReadScope::Current())
        return s_readState ? s_readState->revision : 0;
    return m_hostReadRevision;
}
void InputManager::SetGameplayFocused(bool value)
{
    if (auto *sample = OverrideReadFrame())
        sample->gameplayFocused = value;
    else {
        m_gameplayFocused = value;
        ++m_hostReadRevision;
        InvalidateCapture();
    }
}
void InputManager::SetGameplayViewportOrigin(float x, float y)
{
    if (auto *sample = OverrideReadFrame())
        sample->gameplayViewportOrigin = {x, y};
    else {
        m_gameplayViewportOrigin = {x, y};
        ++m_hostReadRevision;
        InvalidateCapture();
    }
}
void InputManager::SetGameplayViewportSize(float x, float y)
{
    if (auto *sample = OverrideReadFrame())
        sample->gameplayViewportSize = {x, y};
    else {
        m_gameplayViewportSize = {x, y};
        ++m_hostReadRevision;
        InvalidateCapture();
    }
}

void InputManager::SetGameplayViewOutput(float width, float height, std::array<float, 4> touchViewport)
{
    if (auto *sample = OverrideReadFrame()) {
        sample->gameplayRenderSize = {width, height};
        sample->gameplayTouchViewport = touchViewport;
    } else {
        m_gameplayRenderSize = {width, height};
        m_gameplayTouchViewport = touchViewport;
        InvalidateCapture();
    }
}
std::array<float, 2> InputManager::GetGameplayRenderSize() const noexcept
{
    const auto *sample = InputReadScope::Current();
    return sample ? sample->gameplayRenderSize : m_gameplayRenderSize;
}
std::array<float, 4> InputManager::GetGameplayTouchViewport() const noexcept
{
    const auto *sample = InputReadScope::Current();
    return sample ? sample->gameplayTouchViewport : m_gameplayTouchViewport;
}

void InputManager::AppendSampleEvent(SampleEvent event)
{
    m_sampleEvents.push_back(std::move(event));
    InvalidateCapture();
}
void InputManager::ClearSampleTransients(InputFrameSample &sample)
{
    sample.keyDown.fill(0);
    sample.keyUp.fill(0);
    sample.keyDownConsumed.fill(0);
    sample.mouseDown.fill(0);
    sample.mouseUp.fill(0);
    sample.mouseDX = sample.mouseDY = sample.scrollX = sample.scrollY = 0;
    sample.gameScroll = {};
    sample.inputString.clear();
    sample.gameInputString.clear();
    sample.accelerationEvents.clear();
    sample.droppedFiles.clear();
    sample.anyKeyDown = false;
    sample.syntheticInputFrame = false;
    sample.syntheticGameplayInput = sample.syntheticHeld;
    sample.touches.erase(std::remove_if(sample.touches.begin(), sample.touches.end(),
                                        [](const auto &touch) {
                                            return touch.phase == TouchPhase::Ended ||
                                                   touch.phase == TouchPhase::Canceled;
                                        }),
                         sample.touches.end());
    for (auto &touch : sample.touches) {
        touch.phase = TouchPhase::Stationary;
        touch.deltaX = touch.deltaY = touch.deltaTime = 0;
        touch.beganThisFrame = false;
    }
    for (auto &mouse : sample.gameMouseFrames) {
        std::get<2>(mouse) = std::get<3>(mouse) = 0;
        std::get<5>(mouse) = std::get<6>(mouse) = false;
    }
}
void InputManager::SetWorldGameplayRouting(uint64_t selectedWorld, bool broadcast,
                                           std::vector<std::tuple<uint64_t, uint32_t, uint32_t>> outputs,
                                           std::vector<std::array<float, 4>> imageRects)
{
    if (InputReadScope::Current())
        throw std::logic_error("World tile routing is recorded by the Host, not a gameplay callback");
    std::sort(outputs.begin(), outputs.end());
    if (!selectedWorld && outputs.empty()) {
        if (m_worldGameplayRouting) {
            m_worldGameplayRouting.reset();
            InvalidateCapture();
        }
        return;
    }
    if (m_worldGameplayRouting && m_worldGameplayRouting->selectedWorld == selectedWorld &&
        m_worldGameplayRouting->broadcast == broadcast && m_worldGameplayRouting->outputs == outputs &&
        m_worldGameplayRouting->imageRects == imageRects)
        return;
    auto routing = std::make_shared<WorldGameplayRouting>();
    routing->selectedWorld = selectedWorld;
    routing->broadcast = broadcast;
    routing->outputs = std::move(outputs);
    routing->imageRects = std::move(imageRects);
    m_worldGameplayRouting = std::move(routing);
    InvalidateCapture();
}

InputFrameLease InputManager::BeginWorldFrameInput(uint64_t worldId, InputFrameLease external,
                                                   uint64_t logicalFrameIndex)
{
    if (!worldId || !external || external->logical)
        throw std::invalid_argument("World input admission requires its owner and an external Host sample");
    auto state = AcquireWorldReadState(worldId);
    if (state->logicalValue && state->lastLogicalFrameIndex == logicalFrameIndex) {
        if (state->lastExternalSample != external->sampleIdentity)
            throw std::logic_error("A World input admission cannot replace its already accepted external sample");
        return state->logicalValue;
    }
    auto value = std::make_shared<InputFrameSample>(*external);
    if (state->lastExternalSample == external->sampleIdentity)
        ClearSampleTransients(*value);
    if (const auto &routing = value->worldRouting) {
        const auto output = std::lower_bound(routing->outputs.begin(), routing->outputs.end(), worldId,
                                             [](const auto &entry, uint64_t id) { return std::get<0>(entry) < id; });
        value->gameplayRenderSize = output != routing->outputs.end() && std::get<0>(*output) == worldId
                                        ? std::array<float, 2>{float(std::get<1>(*output)), float(std::get<2>(*output))}
                                        : std::array<float, 2>{0, 0};
        if (!routing->broadcast && routing->selectedWorld != worldId) {
            value->gameplayFocused = false;
            value->keys.fill(0);
            value->keyDown.fill(0);
            value->keyUp.fill(0);
            value->gameKeyOwned.fill(0);
            value->mouseButtons.fill(0);
            value->mouseDown.fill(0);
            value->mouseUp.fill(0);
            value->gameMouseOwned.fill(0);
            for (auto &mouse : value->gameMouseFrames)
                mouse = {};
            value->mouseX = value->mouseY = value->mouseDX = value->mouseDY = 0;
            value->scrollX = value->scrollY = 0;
            value->gameScroll = {};
            value->inputString.clear();
            value->gameInputString.clear();
            value->touches.clear();
            value->droppedFiles.clear();
            value->anyKeyDown = value->syntheticGameplayInput = value->syntheticHeld = false;
        }
    }
    value->logical = true;
    value->logicalFrameIndex = logicalFrameIndex;
    value->sourceSampleIdentity = external->sampleIdentity;
    value->sampleIdentity = ++m_sampleIdentity;
    state->overrideValue.reset();
    state->lastLogicalFrameIndex = logicalFrameIndex;
    state->lastExternalSample = external->sampleIdentity;
    state->logicalValue = std::move(value);
    return state->logicalValue;
}

InputFrameLease InputManager::CaptureFrame()
{
    // Never recapture an open World's old scope. One shared immutable value
    // is sealed after event ingestion for this Host; callers retain its lease.
    if (m_capturedFrame)
        return m_capturedFrame;
    InputReadScope live(nullptr);
    auto sample = std::make_shared<InputFrameSample>();
    sample->hostFrameIndex = m_frameIndex;
    sample->sampleIdentity = ++m_sampleIdentity;
    sample->sourceSampleIdentity = sample->sampleIdentity;
    sample->keys = m_keys;
    for (size_t i = 0; i < sample->keyDown.size(); ++i) {
        sample->gameKeyOwned[i] = IsGameKeyOwned(static_cast<int>(i));
    }
    sample->mouseButtons = m_mouseButtons;
    for (size_t i = 0; i < sample->gameMouseOwned.size(); ++i) {
        sample->gameMouseOwned[i] = IsGameMouseButtonOwned(static_cast<int>(i));
        sample->gameMouseFrames[i] = GetGameMouseFrameState(static_cast<int>(i));
    }
    sample->mouseX = m_mouseX;
    sample->mouseY = m_mouseY;
    sample->mouseDX = m_mouseDX;
    sample->mouseDY = m_mouseDY;
    sample->scrollX = m_scrollX;
    sample->scrollY = m_scrollY;
    sample->gameScroll = m_gameScroll;
    sample->inputString = m_inputString;
    sample->gameInputString = m_gameInputString;
    sample->touches = m_touches;
    sample->accelerationEvents = m_accelerationEvents;
    sample->droppedFiles = m_droppedFiles;
    sample->acceleration = m_acceleration;
    sample->gyroscopeRotationRate = m_gyroscopeRotationRate;
    sample->screen = m_screenState;
    sample->accelerometerAvailable = m_accelerometerAvailable;
    sample->gyroscopeAvailable = m_gyroscopeAvailable;
    sample->syntheticInputFrame = m_syntheticInputThisFrame;
    sample->syntheticGameplayInput = m_syntheticInputThisFrame || m_syntheticHeldCount != 0;
    sample->syntheticHeld = m_syntheticHeldCount != 0;
    sample->worldRouting = m_worldGameplayRouting;
    sample->gameplayFocused = m_gameplayFocused;
    sample->gameplayViewportOrigin = m_gameplayViewportOrigin;
    sample->gameplayViewportSize = m_gameplayViewportSize;
    sample->gameplayRenderSize = m_gameplayRenderSize;
    sample->gameplayTouchViewport = m_gameplayTouchViewport;
    // Raw Host queries retain their own edges/text; World delivery consumes
    // only this sealed occurrence prefix, shared by all Host admissions.
    auto originalTouches = std::move(sample->touches);
    ClearSampleTransients(*sample);
    for (auto &touch : originalTouches) {
        const bool terminal = touch.phase == TouchPhase::Ended || touch.phase == TouchPhase::Canceled;
        const bool changed = std::any_of(m_sampleEvents.begin(), m_sampleEvents.end(), [&](const auto &event) {
            return event.kind == SampleEventKind::Touch && event.touch.touchId == touch.touchId &&
                   event.touch.fingerId == touch.fingerId;
        });
        if (terminal && !changed)
            continue;
        touch.phase = TouchPhase::Stationary;
        touch.deltaX = touch.deltaY = touch.deltaTime = 0;
        touch.beganThisFrame = false;
        sample->touches.push_back(std::move(touch));
    }
    for (const auto &event : m_sampleEvents) {
        switch (event.kind) {
        case SampleEventKind::KeyDown:
            sample->keyDown[event.index] = 1;
            sample->anyKeyDown = true;
            break;
        case SampleEventKind::SyntheticKeyDown:
            sample->keyDown[event.index] = 1;
            break;
        case SampleEventKind::KeyUp:
            sample->keyUp[event.index] = 1;
            break;
        case SampleEventKind::MouseDown:
            sample->mouseDown[event.index] = 1;
            break;
        case SampleEventKind::MouseUp:
            sample->mouseUp[event.index] = 1;
            break;
        case SampleEventKind::Motion:
            sample->mouseDX += event.x;
            sample->mouseDY += event.y;
            break;
        case SampleEventKind::Scroll:
            sample->scrollX += event.x;
            sample->scrollY += event.y;
            if (event.game) {
                sample->gameScroll.first += event.x;
                sample->gameScroll.second += event.y;
            }
            break;
        case SampleEventKind::Text:
            sample->inputString += event.text;
            if (event.game)
                sample->gameInputString += event.text;
            break;
        case SampleEventKind::Acceleration:
            sample->accelerationEvents.push_back(event.acceleration);
            break;
        case SampleEventKind::Synthetic:
            sample->syntheticInputFrame = true;
            break;
        case SampleEventKind::DroppedFile:
            sample->droppedFiles.push_back(event.text);
            break;
        case SampleEventKind::Touch: {
            auto found = std::find_if(sample->touches.begin(), sample->touches.end(), [&](const auto &touch) {
                return touch.touchId == event.touch.touchId && touch.fingerId == event.touch.fingerId;
            });
            if (found == sample->touches.end()) {
                sample->touches.push_back(event.touch);
                break;
            }
            const bool began = event.touch.phase == TouchPhase::Began;
            const float dx = began ? event.touch.deltaX : found->deltaX + event.touch.deltaX;
            const float dy = began ? event.touch.deltaY : found->deltaY + event.touch.deltaY;
            const bool beganInSample = began || found->beganThisFrame;
            const float beginX = began ? event.touch.x : found->beginX;
            const float beginY = began ? event.touch.y : found->beginY;
            *found = event.touch;
            found->deltaX = dx;
            found->deltaY = dy;
            found->beganThisFrame = beganInSample;
            found->beginX = beginX;
            found->beginY = beginY;
            break;
        }
        }
    }
    sample->syntheticGameplayInput = sample->syntheticInputFrame || m_syntheticHeldCount != 0;
    for (size_t index = 0; index < sample->gameMouseFrames.size(); ++index) {
        auto &mouse = sample->gameMouseFrames[index];
        std::get<2>(mouse) = sample->gameScroll.first;
        std::get<3>(mouse) = sample->gameScroll.second;
        std::get<5>(mouse) = sample->gameMouseOwned[index] && sample->mouseDown[index];
        std::get<6>(mouse) = sample->gameMouseOwned[index] && sample->mouseUp[index];
    }
    m_sampleEvents.clear();
    m_capturedFrame = std::move(sample);
    return m_capturedFrame;
}

bool InputManager::GetGameplayFocused() const noexcept
{
    const auto *sample = InputReadScope::Current();
    return sample ? sample->gameplayFocused : m_gameplayFocused;
}
std::array<float, 2> InputManager::GetGameplayViewportOrigin() const noexcept
{
    const auto *sample = InputReadScope::Current();
    return sample ? sample->gameplayViewportOrigin : m_gameplayViewportOrigin;
}
std::array<float, 2> InputManager::GetGameplayViewportSize() const noexcept
{
    const auto *sample = InputReadScope::Current();
    return sample ? sample->gameplayViewportSize : m_gameplayViewportSize;
}

// ============================================================================
// Static name table
// ============================================================================

std::unordered_map<std::string, int> InputManager::s_nameToScancode;
bool InputManager::s_nameTableBuilt = false;

void InputManager::BuildNameTable()
{
    if (s_nameTableBuilt)
        return;
    s_nameTableBuilt = true;

    // Letters a-z
    for (int i = 0; i < 26; ++i) {
        std::string lower(1, static_cast<char>('a' + i));
        s_nameToScancode[lower] = SDL_SCANCODE_A + i;
    }

    // Digits 0-9
    s_nameToScancode["0"] = SDL_SCANCODE_0;
    for (int i = 1; i <= 9; ++i) {
        s_nameToScancode[std::to_string(i)] = SDL_SCANCODE_1 + (i - 1);
    }

    // Alpha aliases (Unity KeyCode style, lowercase)
    for (int i = 0; i <= 9; ++i) {
        s_nameToScancode["alpha" + std::to_string(i)] = (i == 0) ? SDL_SCANCODE_0 : SDL_SCANCODE_1 + (i - 1);
    }

    // Function keys F1-F12
    for (int i = 1; i <= 12; ++i) {
        s_nameToScancode["f" + std::to_string(i)] = SDL_SCANCODE_F1 + (i - 1);
    }

    // Arrow keys
    s_nameToScancode["up"] = SDL_SCANCODE_UP;
    s_nameToScancode["down"] = SDL_SCANCODE_DOWN;
    s_nameToScancode["left"] = SDL_SCANCODE_LEFT;
    s_nameToScancode["right"] = SDL_SCANCODE_RIGHT;
    s_nameToScancode["up_arrow"] = SDL_SCANCODE_UP;
    s_nameToScancode["down_arrow"] = SDL_SCANCODE_DOWN;
    s_nameToScancode["left_arrow"] = SDL_SCANCODE_LEFT;
    s_nameToScancode["right_arrow"] = SDL_SCANCODE_RIGHT;

    // Modifiers
    s_nameToScancode["left_shift"] = SDL_SCANCODE_LSHIFT;
    s_nameToScancode["right_shift"] = SDL_SCANCODE_RSHIFT;
    s_nameToScancode["left_control"] = SDL_SCANCODE_LCTRL;
    s_nameToScancode["right_control"] = SDL_SCANCODE_RCTRL;
    s_nameToScancode["left_ctrl"] = SDL_SCANCODE_LCTRL;
    s_nameToScancode["right_ctrl"] = SDL_SCANCODE_RCTRL;
    s_nameToScancode["left_alt"] = SDL_SCANCODE_LALT;
    s_nameToScancode["right_alt"] = SDL_SCANCODE_RALT;
    s_nameToScancode["left_command"] = SDL_SCANCODE_LGUI;
    s_nameToScancode["right_command"] = SDL_SCANCODE_RGUI;
    s_nameToScancode["left_super"] = SDL_SCANCODE_LGUI;
    s_nameToScancode["right_super"] = SDL_SCANCODE_RGUI;

    // Common keys
    s_nameToScancode["space"] = SDL_SCANCODE_SPACE;
    s_nameToScancode["return"] = SDL_SCANCODE_RETURN;
    s_nameToScancode["enter"] = SDL_SCANCODE_RETURN;
    s_nameToScancode["escape"] = SDL_SCANCODE_ESCAPE;
    s_nameToScancode["backspace"] = SDL_SCANCODE_BACKSPACE;
    s_nameToScancode["tab"] = SDL_SCANCODE_TAB;
    s_nameToScancode["delete"] = SDL_SCANCODE_DELETE;
    s_nameToScancode["insert"] = SDL_SCANCODE_INSERT;
    s_nameToScancode["home"] = SDL_SCANCODE_HOME;
    s_nameToScancode["end"] = SDL_SCANCODE_END;
    s_nameToScancode["page_up"] = SDL_SCANCODE_PAGEUP;
    s_nameToScancode["page_down"] = SDL_SCANCODE_PAGEDOWN;
    s_nameToScancode["caps_lock"] = SDL_SCANCODE_CAPSLOCK;
    s_nameToScancode["num_lock"] = SDL_SCANCODE_NUMLOCKCLEAR;
    s_nameToScancode["scroll_lock"] = SDL_SCANCODE_SCROLLLOCK;
    s_nameToScancode["print_screen"] = SDL_SCANCODE_PRINTSCREEN;
    s_nameToScancode["pause"] = SDL_SCANCODE_PAUSE;

    // Punctuation / symbols
    s_nameToScancode["minus"] = SDL_SCANCODE_MINUS;
    s_nameToScancode["equals"] = SDL_SCANCODE_EQUALS;
    s_nameToScancode["left_bracket"] = SDL_SCANCODE_LEFTBRACKET;
    s_nameToScancode["right_bracket"] = SDL_SCANCODE_RIGHTBRACKET;
    s_nameToScancode["backslash"] = SDL_SCANCODE_BACKSLASH;
    s_nameToScancode["semicolon"] = SDL_SCANCODE_SEMICOLON;
    s_nameToScancode["quote"] = SDL_SCANCODE_APOSTROPHE;
    s_nameToScancode["backquote"] = SDL_SCANCODE_GRAVE;
    s_nameToScancode["comma"] = SDL_SCANCODE_COMMA;
    s_nameToScancode["period"] = SDL_SCANCODE_PERIOD;
    s_nameToScancode["slash"] = SDL_SCANCODE_SLASH;

    // Numpad
    constexpr SDL_Scancode keypad[] = {SDL_SCANCODE_KP_0, SDL_SCANCODE_KP_1, SDL_SCANCODE_KP_2, SDL_SCANCODE_KP_3,
                                       SDL_SCANCODE_KP_4, SDL_SCANCODE_KP_5, SDL_SCANCODE_KP_6, SDL_SCANCODE_KP_7,
                                       SDL_SCANCODE_KP_8, SDL_SCANCODE_KP_9};
    for (int i = 0; i <= 9; ++i) {
        s_nameToScancode["keypad_" + std::to_string(i)] = keypad[i];
    }
    s_nameToScancode["keypad_plus"] = SDL_SCANCODE_KP_PLUS;
    s_nameToScancode["keypad_minus"] = SDL_SCANCODE_KP_MINUS;
    s_nameToScancode["keypad_multiply"] = SDL_SCANCODE_KP_MULTIPLY;
    s_nameToScancode["keypad_divide"] = SDL_SCANCODE_KP_DIVIDE;
    s_nameToScancode["keypad_enter"] = SDL_SCANCODE_KP_ENTER;
    s_nameToScancode["keypad_period"] = SDL_SCANCODE_KP_PERIOD;
}

// ============================================================================
// Singleton
// ============================================================================

InputManager &InputManager::Instance()
{
    static InputManager instance;
    return instance;
}

InputManager::InputManager()
{
    m_keys.fill(0);
    m_keyDown.fill(0);
    m_keyUp.fill(0);
    m_mouseButtons.fill(0);
    m_mouseButtonDown.fill(0);
    m_mouseButtonUp.fill(0);
    BuildNameTable();
}

// ============================================================================
// Frame lifecycle
// ============================================================================

void InputManager::BeginFrame()
{
    m_capturedFrame.reset();
    m_sampleEvents.clear();
    ++m_frameIndex;
    m_keyDown.fill(0);
    m_keyUp.fill(0);
    m_mouseButtonDown.fill(0);
    m_mouseButtonUp.fill(0);

    // Clear per-frame deltas
    m_mouseDX = 0.f;
    m_mouseDY = 0.f;
    m_scrollX = 0.f;
    m_scrollY = 0.f;
    m_gameScroll = {};
    m_inputString.clear();
    m_gameInputString.clear();
    m_accelerationEvents.clear();
    m_touches.erase(std::remove_if(m_touches.begin(), m_touches.end(),
                                   [](const TouchState &touch) {
                                       return touch.phase == TouchPhase::Ended || touch.phase == TouchPhase::Canceled;
                                   }),
                    m_touches.end());
    for (TouchState &touch : m_touches) {
        touch.phase = TouchPhase::Stationary;
        touch.deltaX = 0.0f;
        touch.deltaY = 0.0f;
        touch.beganThisFrame = false;
    }
    m_droppedFiles.clear();
    m_syntheticInputThisFrame = false;
    m_syntheticKeyDown.fill(0);
    m_keyDownConsumed.fill(0);

    // Android publishes the keyboard inset from the activity independently of
    // SDL window events. Desktop metrics follow the window event stream.
#if defined(__ANDROID__) && !defined(INFERNUX_INPUT_SEMANTIC_HOST)
    RefreshScreenState();
#endif
}

void InputManager::ProcessKeyEvent(int scancode, bool pressed, bool gameInput)
{
    InvalidateCapture();
    if (scancode < 0 || scancode >= INPUT_MAX_KEYS)
        return;
    const size_t index = static_cast<size_t>(scancode);
    if (pressed) {
        if (m_keys[index] == 0 || !gameInput || m_syntheticKeyDown[index] != 0)
            m_gameKeyOwnership[index] = gameInput;
        if (m_keys[index] == 0) {
            m_keyDown[index] = 1;
            AppendSampleEvent({SampleEventKind::KeyDown, scancode});
        }
        m_keys[index] = 1;
    } else {
        if (m_keys[index] != 0) {
            m_keyUp[index] = 1;
            AppendSampleEvent({SampleEventKind::KeyUp, scancode});
        }
        m_keys[index] = 0;
    }
}

void InputManager::ProcessPointerButtonEvent(int button, bool pressed, bool gameInput)
{
    InvalidateCapture();
    if (button < 0 || button >= INPUT_MAX_MOUSE_BUTTONS)
        return;
    const bool wasHidden = IsCursorHidden();
    const size_t index = static_cast<size_t>(button);
    if (pressed) {
        if (m_mouseButtons[index] == 0) {
            const bool inGameView = IsInEditorGameInputRect(m_mouseX, m_mouseY);
            if (gameInput && (!m_editorCursorCaptureManaged || IsCursorLocked() || inGameView ||
                              m_syntheticMouseButtons[index] != 0))
                m_gameMouseButtonOwnership |= 1u << button;
            else
                m_gameMouseButtonOwnership &= ~(1u << button);
            m_mouseButtonDown[index] = 1;
            AppendSampleEvent({SampleEventKind::MouseDown, button});
        }
        m_mouseButtons[index] = 1;
    } else {
        if (m_mouseButtons[index] != 0) {
            m_mouseButtonUp[index] = 1;
            AppendSampleEvent({SampleEventKind::MouseUp, button});
        }
        m_mouseButtons[index] = 0;
    }
    if (wasHidden != IsCursorHidden())
        ApplyCursorState();
}

void InputManager::ProcessPointerMotionEvent(float x, float y, float deltaX, float deltaY)
{
    InvalidateCapture();
    if (m_warpMotionPending) {
        m_warpMotionPending = false;
        if (std::abs(x - m_warpTargetX) <= 1.0f && std::abs(y - m_warpTargetY) <= 1.0f) {
            m_mouseX = x;
            m_mouseY = y;
            return;
        }
    }
    m_mouseX = x;
    m_mouseY = y;
    m_mouseDX += deltaX;
    m_mouseDY += deltaY;
    AppendSampleEvent({SampleEventKind::Motion, 0, deltaX, deltaY});
    if (m_editorMouseCaptured) {
        m_editorMouseDX += deltaX;
        m_editorMouseDY += deltaY;
    }
}

void InputManager::ProcessScrollEvent(float deltaX, float deltaY, bool gameInput)
{
    m_scrollX += deltaX;
    m_scrollY += deltaY;
    AppendSampleEvent({SampleEventKind::Scroll, 0, deltaX, deltaY, gameInput});
    if (gameInput) {
        m_gameScroll.first += deltaX;
        m_gameScroll.second += deltaY;
    }
}

void InputManager::ProcessTextInputEvent(const std::string &text, bool gameInput)
{
    m_inputString += text;
    AppendSampleEvent({SampleEventKind::Text, 0, 0, 0, gameInput, text});
    if (gameInput)
        m_gameInputString += text;
}

void InputManager::ProcessMotionSensorEvent(MotionSensorType type, uint64_t timestampNs, float x, float y, float z)
{
    if (type == MotionSensorType::Accelerometer) {
        const float inverseGravity = 1.0f / SDL_STANDARD_GRAVITY;
        m_acceleration = {x * inverseGravity, y * inverseGravity, z * inverseGravity};
        const float deltaTime = m_lastAccelerationTimestampNs != 0 && timestampNs >= m_lastAccelerationTimestampNs
                                    ? static_cast<float>(timestampNs - m_lastAccelerationTimestampNs) * 1.0e-9f
                                    : 0.0f;
        m_lastAccelerationTimestampNs = timestampNs;
        m_accelerationEvents.push_back({m_acceleration, deltaTime});
        SampleEvent sample{SampleEventKind::Acceleration};
        sample.acceleration = m_accelerationEvents.back();
        AppendSampleEvent(std::move(sample));
        m_accelerometerAvailable = true;
        return;
    }

    InvalidateCapture();
    m_gyroscopeRotationRate = {x, y, z};
    m_gyroscopeAvailable = true;
}

void InputManager::InitializeMotionSensors()
{
#if !defined(INFERNUX_INPUT_SEMANTIC_HOST)
    ShutdownMotionSensors();
    int sensorCount = 0;
    SDL_SensorID *sensorIds = SDL_GetSensors(&sensorCount);
    if (sensorIds == nullptr)
        return;

    for (int index = 0; index < sensorCount; ++index) {
        const SDL_SensorID sensorId = sensorIds[index];
        const SDL_SensorType sensorType = SDL_GetSensorTypeForID(sensorId);
        if (sensorType == SDL_SENSOR_ACCEL && m_accelerometer == nullptr) {
            m_accelerometer = SDL_OpenSensor(sensorId);
            if (m_accelerometer != nullptr) {
                m_accelerometerId = sensorId;
                m_accelerometerAvailable = true;
            }
        } else if (sensorType == SDL_SENSOR_GYRO && m_gyroscope == nullptr) {
            m_gyroscope = SDL_OpenSensor(sensorId);
            if (m_gyroscope != nullptr) {
                m_gyroscopeId = sensorId;
                m_gyroscopeAvailable = true;
            }
        }
    }
    SDL_free(sensorIds);
#endif
}

void InputManager::ShutdownMotionSensors()
{
#if !defined(INFERNUX_INPUT_SEMANTIC_HOST)
    if (m_accelerometer != nullptr)
        SDL_CloseSensor(m_accelerometer);
    if (m_gyroscope != nullptr)
        SDL_CloseSensor(m_gyroscope);
#endif
    m_accelerometer = nullptr;
    m_gyroscope = nullptr;
    m_accelerometerId = 0;
    m_gyroscopeId = 0;
    m_lastAccelerationTimestampNs = 0;
    m_accelerometerAvailable = false;
    m_gyroscopeAvailable = false;
    m_acceleration = {};
    m_gyroscopeRotationRate = {};
    m_accelerationEvents.clear();
}

void InputManager::ProcessTouchEvent(uint64_t touchId, uint64_t fingerId, uint64_t timestampNs, uint32_t windowId,
                                     float x, float y, float deltaX, float deltaY, float pressure, TouchPhase phase,
                                     float contactWidth, float contactHeight, bool isPrimary,
                                     const std::string &cancelReason)
{
    const auto existing = std::find_if(m_touches.begin(), m_touches.end(), [&](const TouchState &touch) {
        return touch.touchId == touchId && touch.fingerId == fingerId;
    });
    TouchState *touch = nullptr;
    if (existing == m_touches.end()) {
        const bool hasPrimaryContact = std::any_of(m_touches.begin(), m_touches.end(), [](const TouchState &candidate) {
            return candidate.isPrimary && candidate.phase != TouchPhase::Ended &&
                   candidate.phase != TouchPhase::Canceled;
        });
        m_touches.emplace_back();
        touch = &m_touches.back();
        touch->touchId = touchId;
        touch->fingerId = fingerId;
        touch->isPrimary = isPrimary || !hasPrimaryContact;
    } else {
        touch = &*existing;
    }
    const uint64_t previousTimestampNs = touch->timestampNs;
    touch->timestampNs = timestampNs;
    touch->windowId = windowId;
    touch->x = x;
    touch->y = y;
    // The snapshot exposes movement over this input frame, not just the
    // final event. A newly begun contact starts its own displacement even
    // when the platform reuses a finger ID before the next BeginFrame.
    if (phase == TouchPhase::Began) {
        touch->deltaX = deltaX;
        touch->deltaY = deltaY;
    } else {
        touch->deltaX += deltaX;
        touch->deltaY += deltaY;
    }
    touch->deltaTime = previousTimestampNs != 0 && timestampNs >= previousTimestampNs
                           ? static_cast<float>(timestampNs - previousTimestampNs) * 1.0e-9f
                           : 0.0f;
    touch->pressure = pressure;
    touch->contactWidth = contactWidth;
    touch->contactHeight = contactHeight;
    touch->isPrimary = touch->isPrimary || isPrimary;
    if (phase == TouchPhase::Began) {
        touch->beganThisFrame = true;
        touch->beginX = x;
        touch->beginY = y;
    }
    touch->cancelReason = phase == TouchPhase::Canceled ? cancelReason : std::string{};
    touch->phase = phase;
    SampleEvent sample{SampleEventKind::Touch};
    sample.touch = *touch;
    sample.touch.deltaX = deltaX;
    sample.touch.deltaY = deltaY;
    AppendSampleEvent(std::move(sample));
}

void InputManager::ProcessFocusEvent(bool focused)
{
    InvalidateCapture();
    // OS activation starts a new editor interaction; only a viewport click
    // may acquire capture again, even if no loss event preceded activation.
    m_editorMouseCaptured = false;
    if (m_editorCursorCaptureManaged)
        SetEditorCursorCaptureAllowed(false);
    if (focused) {
        if (!m_screenState.focused) {
            m_screenState.focused = true;
            ++m_screenState.revision;
        }
#if !defined(INFERNUX_INPUT_SEMANTIC_HOST)
        ApplyCursorState();
#endif
        return;
    }
    StopTextInput();
    ResetPhysicalInputForFocusLoss();
    if (m_screenState.focused) {
        m_screenState.focused = false;
        ++m_screenState.revision;
    }
#if !defined(INFERNUX_INPUT_SEMANTIC_HOST)
    ApplyCursorState();
#endif
}

void InputManager::ProcessScreenMetrics(int logicalWidth, int logicalHeight, int framebufferWidth,
                                        int framebufferHeight, float pixelRatio, int safeAreaX, int safeAreaY,
                                        int safeAreaWidth, int safeAreaHeight, bool keyboardInsetKnown,
                                        int keyboardInset)
{
    InvalidateCapture();
    if (!std::isfinite(pixelRatio) || pixelRatio <= 0.0f)
        throw std::invalid_argument("screen pixel ratio must be finite and positive");
    ScreenState next = m_screenState;
    next.logicalWidth = std::max(1, logicalWidth);
    next.logicalHeight = std::max(1, logicalHeight);
    next.framebufferWidth = std::max(1, framebufferWidth);
    next.framebufferHeight = std::max(1, framebufferHeight);
    next.pixelRatio = pixelRatio;
    next.safeAreaX = std::clamp(safeAreaX, 0, next.logicalWidth);
    next.safeAreaY = std::clamp(safeAreaY, 0, next.logicalHeight);
    next.safeAreaWidth = std::clamp(safeAreaWidth, 0, next.logicalWidth - next.safeAreaX);
    next.safeAreaHeight = std::clamp(safeAreaHeight, 0, next.logicalHeight - next.safeAreaY);
    next.keyboardInsetKnown = keyboardInsetKnown;
    next.keyboardInset = keyboardInsetKnown ? std::clamp(keyboardInset, 0, next.logicalHeight) : 0;
    const bool changed =
        next.logicalWidth != m_screenState.logicalWidth || next.logicalHeight != m_screenState.logicalHeight ||
        next.framebufferWidth != m_screenState.framebufferWidth ||
        next.framebufferHeight != m_screenState.framebufferHeight || next.pixelRatio != m_screenState.pixelRatio ||
        next.safeAreaX != m_screenState.safeAreaX || next.safeAreaY != m_screenState.safeAreaY ||
        next.safeAreaWidth != m_screenState.safeAreaWidth || next.safeAreaHeight != m_screenState.safeAreaHeight ||
        next.keyboardInsetKnown != m_screenState.keyboardInsetKnown ||
        next.keyboardInset != m_screenState.keyboardInset;
    if (!changed)
        return;
    next.revision = m_screenState.revision + 1;
    const bool viewportChanged =
        next.logicalWidth != m_screenState.logicalWidth || next.logicalHeight != m_screenState.logicalHeight ||
        next.safeAreaX != m_screenState.safeAreaX || next.safeAreaY != m_screenState.safeAreaY ||
        next.safeAreaWidth != m_screenState.safeAreaWidth || next.safeAreaHeight != m_screenState.safeAreaHeight;
    m_screenState = next;
    if (viewportChanged) {
        CancelActiveTouches("viewport_changed");
        if (m_editorMouseCaptured || IsCursorLocked())
            ApplyCursorState();
    }
}

bool InputManager::StartTextInput()
{
    if (m_textInputActive)
        return true;
#if defined(INFERNUX_INPUT_SEMANTIC_HOST)
    m_textInputActive = true;
    return true;
#else
    if (m_window == nullptr)
        return false;
#if defined(__ANDROID__)
    // An explicit text-input request needs the software editor even when an
    // emulator or attached physical keyboard makes SDL_HasKeyboard() true.
    SDL_SetHint(SDL_HINT_ENABLE_SCREEN_KEYBOARD, "1");
    const char *screenKeyboardHint = SDL_GetHint(SDL_HINT_ENABLE_SCREEN_KEYBOARD);
    SDL_Log("INFERNUX_ANDROID_TEXT_INPUT_SDL_REQUEST keyboard=%d hint=%s", SDL_HasKeyboard(),
            screenKeyboardHint != nullptr ? screenKeyboardHint : "unset");
#endif
    const bool started = SDL_StartTextInput(m_window);
#if defined(__ANDROID__)
    SDL_Log("INFERNUX_ANDROID_TEXT_INPUT_SDL_RESULT started=%d active=%d error=%s", started,
            SDL_TextInputActive(m_window), started ? "none" : SDL_GetError());
#endif
    if (!started)
        return false;
    m_textInputActive = true;
    return true;
#endif
}

void InputManager::StopTextInput()
{
    if (!m_textInputActive)
        return;
#if !defined(INFERNUX_INPUT_SEMANTIC_HOST)
    if (m_window != nullptr)
        SDL_StopTextInput(m_window);
#endif
    m_textInputActive = false;
}

void InputManager::ProcessSDLEvent(const SDL_Event &event, bool gameInput)
{
    // Helper: Remap SDL button index to Unity convention
    // SDL3: 1=left, 2=middle, 3=right, 4=X1, 5=X2
    // Unity: 0=left, 1=right, 2=middle, 3=X1, 4=X2
    auto remapButton = [](int sdlButton) -> int {
        switch (sdlButton) {
        case SDL_BUTTON_LEFT:
            return 0;
        case SDL_BUTTON_RIGHT:
            return 1;
        case SDL_BUTTON_MIDDLE:
            return 2;
        case SDL_BUTTON_X1:
            return 3;
        case SDL_BUTTON_X2:
            return 4;
        default:
            return sdlButton - 1;
        }
    };
    switch (event.type) {

    // ---- Keyboard ----
    case SDL_EVENT_KEY_DOWN:
        ProcessKeyEvent(static_cast<int>(event.key.scancode), true, gameInput);
        break;
    case SDL_EVENT_KEY_UP:
        ProcessKeyEvent(static_cast<int>(event.key.scancode), false, gameInput);
        break;

    // ---- Mouse buttons ----
    // SDL3 mouse button indices: 1=left, 2=middle, 3=right, 4/5=side
    // We remap to Unity convention: 0=left, 1=right, 2=middle, 3/4=side
    case SDL_EVENT_MOUSE_BUTTON_DOWN: {
        if (event.button.which == SDL_TOUCH_MOUSEID)
            break;
        ProcessPointerMotionEvent(event.button.x, event.button.y, 0.0f, 0.0f);
        ProcessPointerButtonEvent(remapButton(event.button.button), true, gameInput);
        break;
    }
    case SDL_EVENT_MOUSE_BUTTON_UP: {
        if (event.button.which == SDL_TOUCH_MOUSEID)
            break;
        ProcessPointerMotionEvent(event.button.x, event.button.y, 0.0f, 0.0f);
        ProcessPointerButtonEvent(remapButton(event.button.button), false, gameInput);
        break;
    }

    // ---- Mouse motion ----
    case SDL_EVENT_MOUSE_MOTION:
        if (event.motion.which == SDL_TOUCH_MOUSEID)
            break;
        ProcessPointerMotionEvent(event.motion.x, event.motion.y, event.motion.xrel, event.motion.yrel);
        break;

    // ---- Mouse wheel ----
    case SDL_EVENT_MOUSE_WHEEL:
        ProcessScrollEvent(event.wheel.x, event.wheel.y, gameInput);
        break;

    // ---- Text input ----
    case SDL_EVENT_TEXT_INPUT:
        ProcessTextInputEvent(event.text.text, gameInput);
        break;

    // ---- Motion sensors ----
    case SDL_EVENT_SENSOR_UPDATE:
        if (event.sensor.which == m_accelerometerId) {
            ProcessMotionSensorEvent(MotionSensorType::Accelerometer, event.sensor.sensor_timestamp,
                                     event.sensor.data[0], event.sensor.data[1], event.sensor.data[2]);
        } else if (event.sensor.which == m_gyroscopeId) {
            ProcessMotionSensorEvent(MotionSensorType::Gyroscope, event.sensor.sensor_timestamp, event.sensor.data[0],
                                     event.sensor.data[1], event.sensor.data[2]);
        }
        break;

    // ---- Touch ----
    case SDL_EVENT_FINGER_DOWN:
    case SDL_EVENT_FINGER_MOTION:
    case SDL_EVENT_FINGER_UP:
    case SDL_EVENT_FINGER_CANCELED: {
        const SDL_TouchFingerEvent &finger = event.tfinger;
        TouchPhase phase = TouchPhase::Canceled;
        switch (event.type) {
        case SDL_EVENT_FINGER_DOWN:
            phase = TouchPhase::Began;
            break;
        case SDL_EVENT_FINGER_MOTION:
            phase = (finger.dx == 0.0f && finger.dy == 0.0f) ? TouchPhase::Stationary : TouchPhase::Moved;
            break;
        case SDL_EVENT_FINGER_UP:
            phase = TouchPhase::Ended;
            break;
        default:
            break;
        }
        ProcessTouchEvent(finger.touchID, finger.fingerID, finger.timestamp, finger.windowID, finger.x, finger.y,
                          finger.dx, finger.dy, finger.pressure, phase, 0.0f, 0.0f, false,
                          phase == TouchPhase::Canceled ? "platform_cancel" : "");
        break;
    }

    // ---- File drop from OS ----
    // SDL drop events carry the last drop position even when the window did
    // not receive ordinary mouse-motion events while another application
    // owned the drag. Keep the shared pointer state aligned with that target.
    case SDL_EVENT_DROP_POSITION:
        m_mouseX = event.drop.x;
        m_mouseY = event.drop.y;
        break;
    case SDL_EVENT_DROP_FILE: {
        m_mouseX = event.drop.x;
        m_mouseY = event.drop.y;
        if (event.drop.data) {
            m_droppedFiles.emplace_back(event.drop.data);
            AppendSampleEvent({SampleEventKind::DroppedFile, 0, 0, 0, false, event.drop.data});
        }
        break;
    }

    // ---- Window focus lost → release editor drag capture and clear inputs ----
    case SDL_EVENT_WINDOW_FOCUS_LOST:
        ProcessFocusEvent(false);
        break;
    case SDL_EVENT_WINDOW_FOCUS_GAINED:
        ProcessFocusEvent(true);
        break;
    case SDL_EVENT_WINDOW_RESIZED:
    case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
    case SDL_EVENT_WINDOW_DISPLAY_CHANGED:
    case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED:
    case SDL_EVENT_WINDOW_SAFE_AREA_CHANGED:
    case SDL_EVENT_WINDOW_SHOWN:
        RefreshScreenState();
        break;
    case SDL_EVENT_WINDOW_OCCLUDED:
        if (!m_screenState.occluded) {
            m_screenState.occluded = true;
            ++m_screenState.revision;
        }
        break;
    case SDL_EVENT_WINDOW_EXPOSED:
    case SDL_EVENT_WINDOW_RESTORED:
        if (m_screenState.occluded) {
            m_screenState.occluded = false;
            ++m_screenState.revision;
        }
        RefreshScreenState();
        break;

    default:
        break;
    }
}

void InputManager::SetSyntheticMousePosition(float x, float y)
{
    InvalidateCapture();
    m_mouseX = x;
    m_mouseY = y;
    m_syntheticMouseX = x;
    m_syntheticMouseY = y;
    m_hasSyntheticMousePosition = true;
}

bool InputManager::GetSyntheticMousePosition(float &x, float &y) const
{
    if (!m_hasSyntheticMousePosition)
        return false;
    x = m_syntheticMouseX;
    y = m_syntheticMouseY;
    return true;
}

void InputManager::ReleaseSyntheticMousePosition()
{
    InvalidateCapture();
    m_hasSyntheticMousePosition = false;
}

void InputManager::MarkSyntheticInputForFrame()
{
    AppendSampleEvent({SampleEventKind::Synthetic});
    m_syntheticInputThisFrame = true;
}

void InputManager::TrackSyntheticEvent(const SDL_Event &event)
{
    AppendSampleEvent({SampleEventKind::Synthetic});
    m_syntheticInputThisFrame = true;
    const auto setHeld = [this](auto &states, int index, bool held) {
        if (index < 0 || index >= static_cast<int>(states.size()))
            return;
        const uint8_t value = held ? 1u : 0u;
        if (states[static_cast<size_t>(index)] == value)
            return;
        states[static_cast<size_t>(index)] = value;
        if (held)
            ++m_syntheticHeldCount;
        else if (m_syntheticHeldCount > 0)
            --m_syntheticHeldCount;
    };

    if (event.type == SDL_EVENT_KEY_DOWN || event.type == SDL_EVENT_KEY_UP) {
        const int scancode = static_cast<int>(event.key.scancode);
        const bool pressed = event.type == SDL_EVENT_KEY_DOWN;
        const bool wasHeld = scancode >= 0 && scancode < static_cast<int>(m_syntheticKeys.size())
                                 ? m_syntheticKeys[static_cast<size_t>(scancode)] != 0
                                 : false;
        setHeld(m_syntheticKeys, scancode, pressed);
        if (pressed && !wasHeld && scancode >= 0 && scancode < static_cast<int>(m_syntheticKeyDown.size())) {
            m_syntheticKeyDown[static_cast<size_t>(scancode)] = 1;
            AppendSampleEvent({SampleEventKind::SyntheticKeyDown, scancode});
        }
        return;
    }
    if (event.type != SDL_EVENT_MOUSE_BUTTON_DOWN && event.type != SDL_EVENT_MOUSE_BUTTON_UP)
        return;

    int button = -1;
    switch (event.button.button) {
    case SDL_BUTTON_LEFT:
        button = 0;
        break;
    case SDL_BUTTON_RIGHT:
        button = 1;
        break;
    case SDL_BUTTON_MIDDLE:
        button = 2;
        break;
    case SDL_BUTTON_X1:
        button = 3;
        break;
    case SDL_BUTTON_X2:
        button = 4;
        break;
    default:
        break;
    }
    setHeld(m_syntheticMouseButtons, button, event.type == SDL_EVENT_MOUSE_BUTTON_DOWN);
}

// ============================================================================
// Keyboard queries
// ============================================================================

bool InputManager::GetKey(int scancode) const
{
    if (const auto *sample = InputReadScope::Current())
        return scancode >= 0 && scancode < INPUT_MAX_KEYS && sample->keys[static_cast<size_t>(scancode)] != 0;
    if (scancode < 0 || scancode >= INPUT_MAX_KEYS)
        return false;
    return m_keys[scancode] != 0;
}

bool InputManager::GetKeyDown(int scancode) const
{
    if (const auto *sample = InputReadScope::Current())
        return scancode >= 0 && scancode < INPUT_MAX_KEYS && sample->keyDown[static_cast<size_t>(scancode)] != 0;
    if (scancode < 0 || scancode >= INPUT_MAX_KEYS)
        return false;
    return m_keyDown[scancode] != 0 || m_syntheticKeyDown[scancode] != 0;
}

bool InputManager::GetKeyUp(int scancode) const
{
    if (const auto *sample = InputReadScope::Current())
        return scancode >= 0 && scancode < INPUT_MAX_KEYS && sample->keyUp[static_cast<size_t>(scancode)] != 0;
    if (scancode < 0 || scancode >= INPUT_MAX_KEYS)
        return false;
    return m_keyUp[scancode] != 0;
}

bool InputManager::AnyKey() const
{
    if (const auto *sample = InputReadScope::Current())
        return std::any_of(sample->keys.begin(), sample->keys.end(), [](uint8_t value) { return value != 0; });
    for (int i = 0; i < INPUT_MAX_KEYS; ++i) {
        if (m_keys[i])
            return true;
    }
    return false;
}

bool InputManager::AnyKeyDown() const
{
    if (const auto *sample = InputReadScope::Current())
        return sample->anyKeyDown;
    for (int i = 0; i < INPUT_MAX_KEYS; ++i) {
        if (m_keyDown[i])
            return true;
    }
    return false;
}

bool InputManager::IsGameKeyOwned(int scancode) const
{
    if (const auto *sample = InputReadScope::Current())
        return scancode >= 0 && scancode < INPUT_MAX_KEYS && sample->gameKeyOwned[static_cast<size_t>(scancode)] != 0;
    return scancode >= 0 && scancode < INPUT_MAX_KEYS &&
           (!m_editorCursorCaptureManaged || m_gameKeyOwnership[scancode] != 0 || m_syntheticKeys[scancode] != 0 ||
            m_syntheticKeyDown[scancode] != 0);
}

bool InputManager::AnyGameKey(bool down) const
{
    for (int i = 0; i < INPUT_MAX_KEYS; ++i)
        if (IsGameKeyOwned(i) && (down ? GetKeyDown(i) : GetKey(i)))
            return true;
    return false;
}

void InputManager::ConsumeKeyDown(int scancode)
{
    if (!GetKeyDown(scancode))
        return;
    if (InputReadScope::Current()) {
        OverrideReadFrame()->keyDownConsumed[static_cast<size_t>(scancode)] = 1;
        return;
    }
    m_keyDownConsumed[static_cast<size_t>(scancode)] = 1;
}

bool InputManager::IsKeyDownConsumed(int scancode) const
{
    if (scancode < 0 || scancode >= INPUT_MAX_KEYS)
        return false;
    if (const auto *sample = InputReadScope::Current())
        return sample->keyDownConsumed[static_cast<size_t>(scancode)] != 0;
    return m_keyDownConsumed[static_cast<size_t>(scancode)] != 0;
}

// ============================================================================
// Mouse button queries
// ============================================================================

bool InputManager::GetMouseButton(int button) const
{
    if (const auto *sample = InputReadScope::Current())
        return button >= 0 && button < INPUT_MAX_MOUSE_BUTTONS &&
               sample->mouseButtons[static_cast<size_t>(button)] != 0;
    if (button < 0 || button >= INPUT_MAX_MOUSE_BUTTONS)
        return false;
    return m_mouseButtons[button] != 0;
}

bool InputManager::GetMouseButtonDown(int button) const
{
    if (const auto *sample = InputReadScope::Current())
        return button >= 0 && button < INPUT_MAX_MOUSE_BUTTONS && sample->mouseDown[static_cast<size_t>(button)] != 0;
    if (button < 0 || button >= INPUT_MAX_MOUSE_BUTTONS)
        return false;
    return m_mouseButtonDown[button] != 0;
}

bool InputManager::GetMouseButtonUp(int button) const
{
    if (const auto *sample = InputReadScope::Current())
        return button >= 0 && button < INPUT_MAX_MOUSE_BUTTONS && sample->mouseUp[static_cast<size_t>(button)] != 0;
    if (button < 0 || button >= INPUT_MAX_MOUSE_BUTTONS)
        return false;
    return m_mouseButtonUp[button] != 0;
}

bool InputManager::GetGameMouseButton(int button) const
{
    return IsGameMouseButtonOwned(button) && GetMouseButton(button);
}

bool InputManager::GetGameMouseButtonDown(int button) const
{
    return IsGameMouseButtonOwned(button) && GetMouseButtonDown(button);
}

std::tuple<float, float, float, float, bool, bool, bool> InputManager::GetGameMouseFrameState(int button) const
{
    if (const auto *sample = InputReadScope::Current()) {
        if (button >= 0 && button < INPUT_MAX_MOUSE_BUTTONS)
            return sample->gameMouseFrames[static_cast<size_t>(button)];
        return {sample->mouseX, sample->mouseY, sample->gameScroll.first, sample->gameScroll.second, false,
                false,          false};
    }
    auto state = GetMouseFrameState(button);
    if (IsCursorLocked()) {
        const auto [x, y] = CursorLockPosition(true);
        std::get<0>(state) = x;
        std::get<1>(state) = y;
    }
    if (!IsGameMouseButtonOwned(button))
        std::get<4>(state) = std::get<5>(state) = std::get<6>(state) = false;
    std::get<2>(state) = m_gameScroll.first;
    std::get<3>(state) = m_gameScroll.second;
    return state;
}

bool InputManager::IsGameMouseButtonOwned(int button) const
{
    if (const auto *sample = InputReadScope::Current())
        return button >= 0 && button < INPUT_MAX_MOUSE_BUTTONS &&
               sample->gameMouseOwned[static_cast<size_t>(button)] != 0;
    return button >= 0 && button < INPUT_MAX_MOUSE_BUTTONS &&
           (!m_editorCursorCaptureManaged || (m_gameMouseButtonOwnership & (1u << button)) != 0);
}

bool InputManager::IsInEditorGameInputRect(float x, float y) const
{
    if (m_worldGameplayRouting)
        return std::any_of(m_worldGameplayRouting->imageRects.begin(), m_worldGameplayRouting->imageRects.end(),
                           [x, y](const auto &rect) {
                               const auto &[left, top, width, height] = rect;
                               return x >= left && y >= top && x < left + width && y < top + height;
                           });
    const auto &[left, top, width, height] = m_editorGameInputRect;
    return x >= left && y >= top && x < left + width && y < top + height;
}

bool InputManager::IsEditorGameInputEvent(const SDL_Event &event) const
{
    if (!m_editorCursorCaptureAllowed || m_editorMouseCaptured)
        return false;
    switch (event.type) {
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP: {
        const int button = event.button.button == SDL_BUTTON_RIGHT    ? 1
                           : event.button.button == SDL_BUTTON_MIDDLE ? 2
                                                                      : event.button.button - 1;
        return event.button.which != SDL_TOUCH_MOUSEID && IsGameMouseButtonOwned(button) &&
               (IsCursorLocked() || IsInEditorGameInputRect(event.button.x, event.button.y));
    }
    case SDL_EVENT_MOUSE_WHEEL:
        return IsCursorLocked() || IsInEditorGameInputRect(m_mouseX, m_mouseY);
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP: {
        const auto code = event.key.scancode;
        const bool shortcutModifier = code >= SDL_SCANCODE_LCTRL && code <= SDL_SCANCODE_RGUI &&
                                      code != SDL_SCANCODE_LSHIFT && code != SDL_SCANCODE_RSHIFT;
        return (event.key.mod & (SDL_KMOD_CTRL | SDL_KMOD_ALT | SDL_KMOD_GUI)) == 0 && !shortcutModifier &&
               code != SDL_SCANCODE_ESCAPE && code != SDL_SCANCODE_DELETE &&
               !(code >= SDL_SCANCODE_F1 && code <= SDL_SCANCODE_F12) &&
               !(code >= SDL_SCANCODE_F13 && code <= SDL_SCANCODE_F24);
    }
    default:
        return false;
    }
}

void InputManager::SetEditorGameViewport(const std::array<float, 4> &imageRect, const std::array<float, 4> &inputRect,
                                         uint32_t windowId)
{
    m_editorGameWindowId = windowId;
    if (imageRect == m_editorGameViewport && inputRect == m_editorGameInputRect)
        return;
    m_editorGameViewport = imageRect;
    m_editorGameInputRect = inputRect;
    if (inputRect[2] <= 0 || inputRect[3] <= 0)
        SetEditorCursorCaptureAllowed(false);
    else if (IsCursorLocked())
        ApplyCursorState();
}

std::tuple<float, float, float, float, bool, bool, bool> InputManager::GetMouseFrameState(int button) const
{
    if (const auto *sample = InputReadScope::Current())
        return {sample->mouseX,          sample->mouseY,         sample->scrollX,
                sample->scrollY,         GetMouseButton(button), GetMouseButtonDown(button),
                GetMouseButtonUp(button)};
    if (button < 0 || button >= INPUT_MAX_MOUSE_BUTTONS) {
        return {m_mouseX, m_mouseY, m_scrollX, m_scrollY, false, false, false};
    }

    const bool held = (m_mouseButtons[button] != 0);
    const bool down = (m_mouseButtonDown[button] != 0);
    const bool up = (m_mouseButtonUp[button] != 0);
    return {m_mouseX, m_mouseY, m_scrollX, m_scrollY, held, down, up};
}

// ============================================================================
// Reset
// ============================================================================

void InputManager::ResetAll()
{
    if (auto *sample = OverrideReadFrame()) {
        sample->keys.fill(0);
        sample->keyDown.fill(0);
        sample->keyUp.fill(0);
        sample->keyDownConsumed.fill(0);
        sample->gameKeyOwned.fill(0);
        sample->mouseButtons.fill(0);
        sample->mouseDown.fill(0);
        sample->mouseUp.fill(0);
        sample->gameMouseOwned.fill(0);
        sample->mouseX = sample->mouseY = sample->mouseDX = sample->mouseDY = 0;
        sample->scrollX = sample->scrollY = 0;
        sample->gameScroll = {};
        for (auto &mouse : sample->gameMouseFrames)
            mouse = {0, 0, 0, 0, false, false, false};
        sample->inputString.clear();
        sample->gameInputString.clear();
        sample->touches.clear();
        sample->accelerationEvents.clear();
        sample->droppedFiles.clear();
        sample->syntheticInputFrame = sample->syntheticGameplayInput = sample->anyKeyDown = false;
        return;
    }
    InvalidateCapture();
    m_sampleEvents.clear();
    ++m_hostReadRevision;
    const bool wasHidden = IsCursorHidden();
    m_gameMouseButtonOwnership = 0;
    m_keys.fill(0);
    m_keyDown.fill(0);
    m_keyUp.fill(0);
    m_keyDownConsumed.fill(0);
    m_gameKeyOwnership.fill(0);
    m_mouseButtons.fill(0);
    m_mouseButtonDown.fill(0);
    m_mouseButtonUp.fill(0);
    m_mouseX = m_mouseY = 0.f;
    m_mouseDX = m_mouseDY = 0.f;
    m_editorMouseDX = m_editorMouseDY = 0.f;
    m_scrollX = m_scrollY = 0.f;
    m_gameScroll = {};
    m_syntheticMouseX = m_syntheticMouseY = 0.f;
    m_hasSyntheticMousePosition = false;
    m_syntheticInputThisFrame = false;
    m_syntheticKeys.fill(0);
    m_syntheticKeyDown.fill(0);
    m_syntheticMouseButtons.fill(0);
    m_syntheticHeldCount = 0;
    m_inputString.clear();
    m_gameInputString.clear();
    m_touches.clear();
    m_accelerationEvents.clear();
    m_droppedFiles.clear();
    m_warpMotionPending = false;
    if (wasHidden != IsCursorHidden())
        ApplyCursorState();
}

void InputManager::ResetPhysicalInputForFocusLoss()
{
    // Losing OS focus must release real device state, but trusted synthetic
    // input intentionally drives gameplay while the Editor remains in the
    // background. Preserve that independently tracked held state and rebuild
    // the shared query arrays from it after clearing physical input.
    const auto syntheticKeys = m_syntheticKeys;
    const auto syntheticMouseButtons = m_syntheticMouseButtons;
    const uint32_t gameMouseButtonOwnership = m_gameMouseButtonOwnership;
    const uint32_t syntheticHeldCount = m_syntheticHeldCount;
    auto interruptedTouches = std::move(m_touches);
    ResetAll();
    for (TouchState &touch : interruptedTouches) {
        if (touch.phase != TouchPhase::Ended && touch.phase != TouchPhase::Canceled) {
            touch.phase = TouchPhase::Canceled;
            touch.cancelReason = "focus_lost";
            touch.deltaX = 0.0f;
            touch.deltaY = 0.0f;
            m_touches.emplace_back(std::move(touch));
            SampleEvent sample{SampleEventKind::Touch};
            sample.touch = m_touches.back();
            AppendSampleEvent(std::move(sample));
        }
    }
    m_syntheticKeys = syntheticKeys;
    m_syntheticMouseButtons = syntheticMouseButtons;
    m_syntheticHeldCount = syntheticHeldCount;
    m_syntheticInputThisFrame = syntheticHeldCount != 0;
    for (size_t index = 0; index < m_syntheticKeys.size(); ++index) {
        m_keys[index] = m_syntheticKeys[index];
        m_gameKeyOwnership[index] = m_syntheticKeys[index];
    }
    for (size_t index = 0; index < m_syntheticMouseButtons.size(); ++index) {
        m_mouseButtons[index] = m_syntheticMouseButtons[index];
        if (m_syntheticMouseButtons[index] != 0)
            m_gameMouseButtonOwnership |= gameMouseButtonOwnership & (1u << index);
    }
}

void InputManager::CancelActiveTouches(const std::string &reason)
{
    for (TouchState &touch : m_touches) {
        if (touch.phase == TouchPhase::Ended || touch.phase == TouchPhase::Canceled)
            continue;
        touch.phase = TouchPhase::Canceled;
        touch.cancelReason = reason;
        touch.deltaX = 0.0f;
        touch.deltaY = 0.0f;
        SampleEvent sample{SampleEventKind::Touch};
        sample.touch = touch;
        AppendSampleEvent(std::move(sample));
    }
}

const TouchState &InputManager::GetTouch(int index) const
{
    const auto &touches = GetTouches();
    if (index < 0 || static_cast<size_t>(index) >= touches.size())
        throw std::out_of_range("Touch index is outside the current frame snapshot");
    return touches[static_cast<size_t>(index)];
}

// ============================================================================
// Name ↔ scancode mapping
// ============================================================================

int InputManager::NameToScancode(const std::string &name)
{
    BuildNameTable();

    // Lowercase + replace spaces with underscores
    std::string key = name;
    std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return std::tolower(c); });
    std::replace(key.begin(), key.end(), ' ', '_');

    auto it = s_nameToScancode.find(key);
    if (it != s_nameToScancode.end()) {
        return it->second;
    }

#if !defined(INFERNUX_INPUT_SEMANTIC_HOST)
    // Desktop adapters retain SDL's extended name vocabulary. Semantic hosts
    // deliberately expose only the engine's stable cross-platform table.
    SDL_Scancode sc = SDL_GetScancodeFromName(name.c_str());
    if (sc != SDL_SCANCODE_UNKNOWN) {
        return static_cast<int>(sc);
    }
#endif

    return -1;
}

const char *InputManager::ScancodeToName(int scancode)
{
    if (scancode < 0 || scancode >= INPUT_MAX_KEYS)
        return "unknown";
#if defined(INFERNUX_INPUT_SEMANTIC_HOST)
    BuildNameTable();
    const auto it = std::find_if(s_nameToScancode.begin(), s_nameToScancode.end(),
                                 [scancode](const auto &entry) { return entry.second == scancode; });
    return it != s_nameToScancode.end() ? it->first.c_str() : "unknown";
#else
    return SDL_GetScancodeName(static_cast<SDL_Scancode>(scancode));
#endif
}

// ============================================================================
// Cursor lock (FPS-style capture)
// ============================================================================

void InputManager::SetWindow(SDL_Window *window)
{
    m_window = window;
    if (window == nullptr) {
        m_editorCursorCaptureManaged = false;
        m_editorCursorCaptureAllowed = false;
        m_editorGameViewport = {};
        m_editorGameInputRect = {};
        m_editorGameWindowId = 0;
        m_gameMouseButtonOwnership = 0;
    }
    m_warpMotionPending = false;
#if !defined(INFERNUX_INPUT_SEMANTIC_HOST)
    if (window != nullptr)
        SDL_SetHint(SDL_HINT_MOUSE_RELATIVE_MODE_CENTER, "0");
#endif
    RefreshScreenState();
    ApplyCursorState();
}

void InputManager::RefreshScreenState()
{
#if defined(INFERNUX_INPUT_SEMANTIC_HOST)
    return;
#else
    if (m_window == nullptr)
        return;
    int logicalWidth = 1;
    int logicalHeight = 1;
    int framebufferWidth = 1;
    int framebufferHeight = 1;
    SDL_GetWindowSize(m_window, &logicalWidth, &logicalHeight);
    SDL_GetWindowSizeInPixels(m_window, &framebufferWidth, &framebufferHeight);
    SDL_Rect safeArea{0, 0, logicalWidth, logicalHeight};
    SDL_GetWindowSafeArea(m_window, &safeArea);
    const float displayScale = SDL_GetWindowDisplayScale(m_window);
    bool keyboardInsetKnown = false;
    int keyboardInset = 0;
#if defined(__ANDROID__)
    const char *keyboardInsetKnownValue = std::getenv("INFERNUX_ANDROID_KEYBOARD_INSET_KNOWN");
    const char *keyboardInsetValue = std::getenv("INFERNUX_ANDROID_KEYBOARD_INSET");
    keyboardInsetKnown = keyboardInsetKnownValue != nullptr && std::strcmp(keyboardInsetKnownValue, "1") == 0 &&
                         keyboardInsetValue != nullptr;
    if (keyboardInsetKnown) {
        char *end = nullptr;
        const long parsedInset = std::strtol(keyboardInsetValue, &end, 10);
        keyboardInsetKnown = end != keyboardInsetValue && end != nullptr && *end == '\0';
        if (keyboardInsetKnown) {
            keyboardInset = static_cast<int>(std::clamp(parsedInset, 0L, static_cast<long>(logicalHeight)));
        }
    }
#endif
    if (!std::isfinite(displayScale) || displayScale <= 0.0f)
        throw std::runtime_error("SDL reported an invalid screen display scale");
    ProcessScreenMetrics(logicalWidth, logicalHeight, framebufferWidth, framebufferHeight, displayScale, safeArea.x,
                         safeArea.y, safeArea.w, safeArea.h, keyboardInsetKnown, keyboardInset);
#endif
}

void InputManager::SetCursorLocked(bool locked)
{
    if (locked == m_cursorLocked)
        return;

    m_cursorLocked = locked;
    if (locked)
        m_warpMotionPending = false;
    ApplyCursorState();
}

void InputManager::SetCursorVisible(bool visible)
{
    if (visible == m_cursorVisible)
        return;

    m_cursorVisible = visible;
    ApplyCursorState();
}

void InputManager::SetEditorCursorCaptureAllowed(bool allowed)
{
    if (m_editorCursorCaptureManaged && allowed == m_editorCursorCaptureAllowed)
        return;
    m_editorCursorCaptureManaged = true;
    m_editorCursorCaptureAllowed = allowed;
    m_mouseDX = m_mouseDY = m_scrollX = m_scrollY = 0.0f;
    m_gameScroll = {};
    ApplyCursorState();
}

void InputManager::SetCursorConfined(bool confined)
{
    if (confined == m_cursorConfined)
        return;

    m_cursorConfined = confined;
    ApplyCursorState();
}

bool InputManager::WarpCursor(float x, float y)
{
#if defined(INFERNUX_INPUT_SEMANTIC_HOST)
    (void)x;
    (void)y;
    return false;
#else
    // A confined cursor is owned by the window manager. Do not issue a
    // competing absolute warp while confinement is active; callers must
    // release confinement first, matching the visible/non-relative warp
    // contract exposed by the Python API.
    if (!m_window || m_cursorLocked || m_cursorConfined || m_editorMouseCaptured)
        return false;

    const float targetX = std::clamp(x, 0.0f, static_cast<float>(std::max(0, m_screenState.logicalWidth - 1)));
    const float targetY = std::clamp(y, 0.0f, static_cast<float>(std::max(0, m_screenState.logicalHeight - 1)));
    m_mouseX = targetX;
    m_mouseY = targetY;
    m_warpTargetX = targetX;
    m_warpTargetY = targetY;
    m_warpMotionPending = true;
    SDL_WarpMouseInWindow(m_window, targetX, targetY);
    return true;
#endif
}

void InputManager::SetEditorMouseCapture(bool captured)
{
    if (captured == m_editorMouseCaptured)
        return;

    m_editorMouseDX = 0.f;
    m_editorMouseDY = 0.f;
    m_editorMouseCaptured = captured;
    ApplyCursorState();
}

std::pair<float, float> InputManager::ConsumeEditorMouseDelta()
{
    const std::pair<float, float> delta{m_editorMouseDX, m_editorMouseDY};
    m_editorMouseDX = 0.f;
    m_editorMouseDY = 0.f;
    return delta;
}

std::pair<float, float> InputManager::CursorLockPosition(bool gameLocked) const
{
    if (gameLocked && m_editorCursorCaptureManaged) {
        const auto &[x, y, width, height] = m_editorGameViewport;
        return {x + width * 0.5f, y + height * 0.5f};
    }
    return {m_screenState.logicalWidth * 0.5f, m_screenState.logicalHeight * 0.5f};
}

void InputManager::ApplyCursorState()
{
#if defined(INFERNUX_INPUT_SEMANTIC_HOST)
    // Browser and other semantic hosts own pointer-lock, visibility, and
    // confinement policy in their window adapter. Preserve the requested
    // logical state for that adapter.
    return;
#else
    const bool relativeMouseEnabled = IsCursorLocked() || m_editorMouseCaptured;
    const bool editorReleased = m_editorCursorCaptureManaged && !m_editorCursorCaptureAllowed;

    if (!m_window) {
        return;
    }

    const Uint64 windowFlags = SDL_GetWindowFlags(m_window);
    const bool focused = (windowFlags & SDL_WINDOW_INPUT_FOCUS) != 0;
    if (!focused) {
        SDL_SetWindowRelativeMouseMode(m_window, false);
        SDL_SetWindowMouseRect(m_window, nullptr);
        SDL_SetWindowMouseGrab(m_window, false);
        SDL_ShowCursor();
        return;
    }

    const bool wasRelative = SDL_GetWindowRelativeMouseMode(m_window);
    const SDL_Rect *currentRect = SDL_GetWindowMouseRect(m_window);
    if (relativeMouseEnabled) {
        const auto [x, y] = CursorLockPosition(IsCursorLocked());
        const SDL_Rect lockPoint{static_cast<int>(std::round(x)), static_cast<int>(std::round(y)), 1, 1};
        const bool moved = !currentRect || !SDL_RectsEqual(currentRect, &lockPoint);
        if (moved)
            SDL_SetWindowMouseRect(m_window, &lockPoint);
        if (moved || !wasRelative)
            SDL_WarpMouseInWindow(m_window, x, y);
        if (!wasRelative)
            SDL_SetWindowRelativeMouseMode(m_window, true);
    } else {
        if (wasRelative) {
            if (currentRect) {
                m_mouseX = static_cast<float>(currentRect->x);
                m_mouseY = static_cast<float>(currentRect->y);
                SDL_WarpMouseInWindow(m_window, m_mouseX, m_mouseY);
            }
            SDL_SetWindowRelativeMouseMode(m_window, false);
        }
        if (currentRect)
            SDL_SetWindowMouseRect(m_window, nullptr);
    }
    SDL_SetWindowMouseGrab(m_window, relativeMouseEnabled || (!editorReleased && m_cursorConfined));
    if (IsCursorHidden())
        SDL_HideCursor();
    else
        SDL_ShowCursor();
#endif
}

} // namespace infernux
