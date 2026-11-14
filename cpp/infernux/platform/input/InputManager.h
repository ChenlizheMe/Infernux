/**
 * @file InputManager.h
 * @brief Unified input state manager — SDL event-driven, Unity-style query API.
 *
 * Design:
 *   - Persistent held state plus explicit per-frame down/up edge arrays.
 *   - O(1) lookup by SDL_Scancode index (no
 * hash maps on hot path).
 *   - BeginFrame() clears edge arrays and per-frame deltas.
 *   - ProcessSDLEvent() fills current-frame state from raw
 * SDL events.
 *
 * Integration:
 *   InxView::ProcessEvent() calls BeginFrame() once per frame, then
 *   feeds every SDL_Event through ProcessSDLEvent().
 *
 * Exposed to Python via BindingInput.cpp; the Python `Input` static class
 * wraps these methods with Unity-style naming (get_key_down, etc.).
 */

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

union SDL_Event;
struct SDL_Window;
struct SDL_Sensor;

namespace infernux
{

/// Maximum number of keyboard scancodes tracked (SDL_SCANCODE_COUNT ≈ 512).
static constexpr int INPUT_MAX_KEYS = 512;

/// Maximum number of mouse buttons tracked (SDL supports up to 5, reserve 8).
static constexpr int INPUT_MAX_MOUSE_BUTTONS = 8;

enum class TouchPhase : uint8_t
{
    Began,
    Moved,
    Stationary,
    Ended,
    Canceled,
};

enum class MotionSensorType : uint8_t
{
    Accelerometer,
    Gyroscope,
};

struct AccelerationEvent
{
    std::array<float, 3> acceleration{};
    float deltaTime = 0.0f;
};

struct TouchState
{
    uint64_t touchId = 0;
    uint64_t fingerId = 0;
    uint64_t timestampNs = 0;
    uint32_t windowId = 0;
    float x = 0.0f;
    float y = 0.0f;
    float deltaX = 0.0f;
    float deltaY = 0.0f;
    float deltaTime = 0.0f;
    float pressure = 0.0f;
    float contactWidth = 0.0f;
    float contactHeight = 0.0f;
    bool isPrimary = false;
    // Preserve a press that begins and ends between two game frames.
    bool beganThisFrame = false;
    float beginX = 0.0f;
    float beginY = 0.0f;
    std::string cancelReason;
    TouchPhase phase = TouchPhase::Stationary;
};

struct ScreenState
{
    uint64_t revision = 0;
    int logicalWidth = 1;
    int logicalHeight = 1;
    int framebufferWidth = 1;
    int framebufferHeight = 1;
    float pixelRatio = 1.0f;
    int safeAreaX = 0;
    int safeAreaY = 0;
    int safeAreaWidth = 1;
    int safeAreaHeight = 1;
    int keyboardInset = 0;
    bool keyboardInsetKnown = false;
    bool focused = true;
    bool occluded = false;
};

/// A sealed sample of the existing host device, shared by World admissions.
/// No script/action state, callbacks, platform handles or mutable routing owner.
struct WorldGameplayRouting final
{
    uint64_t selectedWorld = 0;
    bool broadcast = true;
    std::vector<std::tuple<uint64_t, uint32_t, uint32_t>> outputs;
    std::vector<std::array<float, 4>> imageRects;
};

struct InputFrameSample final
{
    uint64_t hostFrameIndex = 0;
    uint64_t sampleIdentity = 0, sourceSampleIdentity = 0, logicalFrameIndex = 0;
    bool logical = false;
    std::shared_ptr<const WorldGameplayRouting> worldRouting;
    std::array<uint8_t, INPUT_MAX_KEYS> keyDownConsumed{};
    std::array<uint8_t, INPUT_MAX_KEYS> keys{}, keyDown{}, keyUp{}, gameKeyOwned{};
    std::array<uint8_t, INPUT_MAX_MOUSE_BUTTONS> mouseButtons{}, mouseDown{}, mouseUp{}, gameMouseOwned{};
    std::array<std::tuple<float, float, float, float, bool, bool, bool>, INPUT_MAX_MOUSE_BUTTONS> gameMouseFrames{};
    float mouseX = 0, mouseY = 0, mouseDX = 0, mouseDY = 0, scrollX = 0, scrollY = 0;
    std::pair<float, float> gameScroll{};
    std::string inputString, gameInputString;
    std::vector<TouchState> touches;
    std::vector<AccelerationEvent> accelerationEvents;
    std::vector<std::string> droppedFiles;
    std::array<float, 3> acceleration{}, gyroscopeRotationRate{};
    ScreenState screen;
    bool accelerometerAvailable = false, gyroscopeAvailable = false;
    bool syntheticInputFrame = false, syntheticGameplayInput = false, syntheticHeld = false, anyKeyDown = false;
    bool gameplayFocused = true;
    std::array<float, 2> gameplayViewportOrigin{}, gameplayViewportSize{}, gameplayRenderSize{};
    std::array<float, 4> gameplayTouchViewport{0, 0, 1, 1};
};
using InputFrameLease = std::shared_ptr<const InputFrameSample>;

/// Mutable owner reads, never stored in the shared external Host sample.
struct InputFrameReadState final
{
    std::shared_ptr<InputFrameSample> overrideValue;
    bool retired = false;
    uint64_t revision = 0;
    uint64_t lastExternalSample = 0, lastLogicalFrameIndex = 0;
    InputFrameLease logicalValue;
};

/// A short owner-thread read scope. Null explicitly restores live Host reads.
/// WorldFrame owns the lease across Host waits; this scope never spans a wait.
class InputReadScope final
{
  public:
    explicit InputReadScope(InputFrameLease lease, uint64_t worldId = 0) noexcept;
    ~InputReadScope();
    InputReadScope(const InputReadScope &) = delete;
    InputReadScope &operator=(const InputReadScope &) = delete;
    [[nodiscard]] static const InputFrameSample *Current() noexcept;
    [[nodiscard]] static uint64_t WorldId() noexcept;

  private:
    InputFrameLease m_previous;
    uint64_t m_previousWorld;
    std::shared_ptr<InputFrameReadState> m_previousReadState;
};

/**
 * @class InputManager
 * @brief Singleton that accumulates SDL input events and exposes Unity-style queries.
 *
 * Frame lifecycle:
 *   1. BeginFrame()        — swap buffers, clear deltas
 *   2. ProcessSDLEvent()×N — accumulate events
 *   3. User code queries   — GetKey/GetKeyDown/GetKeyUp/GetMouseButton/etc.
 */
class InputManager
{
  public:
    static InputManager &Instance();

    // ---- Per-frame lifecycle (called by InxView) ----

    /// @brief Begin a new input frame. Clears transition edges and deltas.
    void BeginFrame();

    /// Seal after the Host event pump/routing, before gameplay callbacks.
    /// Call every admitted Host, including while another WorldFrame is open.
    /// Reads raw live members even under an older World InputReadScope.
    [[nodiscard]] InputFrameLease CaptureFrame();
    void SetWorldGameplayRouting(uint64_t selectedWorld, bool broadcast,
                                 std::vector<std::tuple<uint64_t, uint32_t, uint32_t>> outputs,
                                 std::vector<std::array<float, 4>> imageRects);
    /// Exactly once per new WorldFrame admission; scope entry never consumes.
    [[nodiscard]] InputFrameLease BeginWorldFrameInput(uint64_t worldId, InputFrameLease external,
                                                       uint64_t logicalFrameIndex);
    void RetireWorldInput(uint64_t worldId) noexcept;
    [[nodiscard]] std::shared_ptr<InputFrameReadState> RetainWorldInputReadState();
    [[nodiscard]] uint64_t GetInputReadRevision() const noexcept;

    // Host panel routing metadata travels with the external input sample.
    void SetGameplayFocused(bool value);
    void SetGameplayViewportOrigin(float x, float y);
    void SetGameplayViewportSize(float x, float y);
    void SetGameplayViewOutput(float width, float height, std::array<float, 4> touchViewport);
    [[nodiscard]] std::array<float, 2> GetGameplayRenderSize() const noexcept;
    [[nodiscard]] std::array<float, 4> GetGameplayTouchViewport() const noexcept;
    [[nodiscard]] bool GetGameplayFocused() const noexcept;
    [[nodiscard]] std::array<float, 2> GetGameplayViewportOrigin() const noexcept;
    [[nodiscard]] std::array<float, 2> GetGameplayViewportSize() const noexcept;

    /// @brief Monotonic input-frame identity used by higher-level action maps.
    [[nodiscard]] uint64_t GetFrameIndex() const noexcept
    {
        if (const auto *sample = InputReadScope::Current())
            return sample->sampleIdentity;
        return m_frameIndex;
    }

    /// Raw state is retained; gameInput assigns the event's gameplay ownership.
    void ProcessSDLEvent(const SDL_Event &event, bool gameInput = true);

    // ---- Platform-neutral event ingestion ----
    // Window-system adapters translate their native events once and feed this
    // semantic layer. Web, Android, and desktop therefore share the exact same
    // held/edge/touch lifecycle instead of maintaining parallel input models.
    void ProcessKeyEvent(int scancode, bool pressed, bool gameInput = true);
    void ProcessPointerButtonEvent(int button, bool pressed, bool gameInput = true);
    void ProcessPointerMotionEvent(float x, float y, float deltaX, float deltaY);
    void ProcessScrollEvent(float deltaX, float deltaY, bool gameInput = true);
    void ProcessTextInputEvent(const std::string &text, bool gameInput = true);
    /// Feed one motion-sensor sample. Accelerometer values use SI m/s²;
    /// gyroscope values use radians per second.
    void ProcessMotionSensorEvent(MotionSensorType type, uint64_t timestampNs, float x, float y, float z);
    void ProcessTouchEvent(uint64_t touchId, uint64_t fingerId, uint64_t timestampNs, uint32_t windowId, float x,
                           float y, float deltaX, float deltaY, float pressure, TouchPhase phase,
                           float contactWidth = 0.0f, float contactHeight = 0.0f, bool isPrimary = false,
                           const std::string &cancelReason = {});
    void ProcessFocusEvent(bool focused);
    void ProcessScreenMetrics(int logicalWidth, int logicalHeight, int framebufferWidth, int framebufferHeight,
                              float pixelRatio, int safeAreaX, int safeAreaY, int safeAreaWidth, int safeAreaHeight,
                              bool keyboardInsetKnown = false, int keyboardInset = 0);

    /// Open the first platform accelerometer and gyroscope exposed by SDL.
    /// Missing hardware is a supported capability state, not an initialization error.
    void InitializeMotionSensors();
    void ShutdownMotionSensors();

    /// @brief Give the trusted automation pointer ownership until native pointer input resumes.
    ///
    /// SDL's ImGui backend can otherwise fall back to the physical OS cursor
    /// between automation events. Preserve hover across move/press/release;
    /// native pointer input explicitly returns ownership to the OS cursor.
    void SetSyntheticMousePosition(float x, float y);

    [[nodiscard]] bool GetSyntheticMousePosition(float &x, float &y) const;
    void ReleaseSyntheticMousePosition();

    /// @brief Mark that the current frame is processing trusted synthetic input.
    ///
    /// Editor workflows use this to keep automation-only controls out of
    /// normal desktop interaction, such as native file dialogs.
    void MarkSyntheticInputForFrame();

    /// @brief Track trusted automation-held state without claiming OS or editor focus.
    ///
    /// Synthetic key/button presses can span many rendered frames. Keeping their
    /// ownership separate from the ordinary SDL held-state lets gameplay accept
    /// MCP input while the Editor remains in the background.
    void TrackSyntheticEvent(const SDL_Event &event);

    /// @brief True when at least one trusted synthetic event was handled this frame.
    [[nodiscard]] bool IsSyntheticInputFrame() const
    {
        if (const auto *sample = InputReadScope::Current())
            return sample->syntheticInputFrame;
        return m_syntheticInputThisFrame;
    }

    /// @brief True while trusted automation supplied this frame's input or holds a control.
    [[nodiscard]] bool HasSyntheticGameplayInput() const
    {
        if (const auto *sample = InputReadScope::Current())
            return sample->syntheticGameplayInput;
        return m_syntheticInputThisFrame || m_syntheticHeldCount != 0;
    }

    // ---- Keyboard queries (Unity: Input.GetKey / GetKeyDown / GetKeyUp) ----

    /// @brief Returns true while the key identified by scancode is held down.
    [[nodiscard]] bool GetKey(int scancode) const;

    /// @brief Returns true during the frame the key was pressed down.
    [[nodiscard]] bool GetKeyDown(int scancode) const;

    /// @brief Returns true during the frame the key was released.
    [[nodiscard]] bool GetKeyUp(int scancode) const;

    /// @brief Returns true if any key is currently held down.
    [[nodiscard]] bool AnyKey() const;

    /// @brief Returns true during the frame any key was pressed down.
    [[nodiscard]] bool AnyKeyDown() const;

    [[nodiscard]] bool GetGameKey(int scancode) const
    {
        return IsGameKeyOwned(scancode) && GetKey(scancode);
    }
    [[nodiscard]] bool GetGameKeyDown(int scancode) const
    {
        return IsGameKeyOwned(scancode) && GetKeyDown(scancode);
    }
    [[nodiscard]] bool GetGameKeyUp(int scancode) const
    {
        return IsGameKeyOwned(scancode) && GetKeyUp(scancode);
    }
    [[nodiscard]] bool AnyGameKey(bool down = false) const;

    /// @brief Consume one key-down edge before Editor shortcut routing.
    ///
    /// Gameplay UI uses this for modal keys such as Escape. The key remains
    /// visible to gameplay queries, but the Editor does not execute the same
    /// edge as a global shortcut.
    void ConsumeKeyDown(int scancode);
    [[nodiscard]] bool IsKeyDownConsumed(int scancode) const;

    // ---- Mouse button queries (Unity: Input.GetMouseButton / Down / Up) ----
    // button: 0 = left, 1 = right, 2 = middle, 3/4 = side buttons

    /// @brief Returns true while the given mouse button is held down.
    [[nodiscard]] bool GetMouseButton(int button) const;

    /// @brief Returns true during the frame the mouse button was pressed.
    [[nodiscard]] bool GetMouseButtonDown(int button) const;

    /// @brief Returns true during the frame the mouse button was released.
    [[nodiscard]] bool GetMouseButtonUp(int button) const;

    [[nodiscard]] bool GetGameMouseButton(int button) const;
    [[nodiscard]] bool GetGameMouseButtonDown(int button) const;
    [[nodiscard]] bool GetGameMouseButtonUp(int button) const
    {
        return IsGameMouseButtonOwned(button) && GetMouseButtonUp(button);
    }
    [[nodiscard]] std::tuple<float, float, float, float, bool, bool, bool> GetGameMouseFrameState(int button) const;
    void SetEditorGameViewport(const std::array<float, 4> &imageRect, const std::array<float, 4> &inputRect,
                               uint32_t windowId = 0);
    [[nodiscard]] uint32_t GetEditorGameWindowId() const
    {
        return m_editorGameWindowId;
    }
    [[nodiscard]] bool IsEditorInputManaged() const
    {
        return m_editorCursorCaptureManaged;
    }
    [[nodiscard]] bool IsInEditorGameInputRect(float x, float y) const;
    /// Native routing only: already-owned gameplay input need not rebuild Editor panels.
    [[nodiscard]] bool IsEditorGameInputEvent(const SDL_Event &event) const;

    /// @brief Returns a batched snapshot for a mouse button this frame.
    /// Tuple layout: (mouseX, mouseY, scrollX, scrollY, held, down, up).
    [[nodiscard]] std::tuple<float, float, float, float, bool, bool, bool> GetMouseFrameState(int button) const;

    // ---- Mouse position & delta (Unity: Input.mousePosition, mouseDelta) ----
    // Coordinates are in window-space pixels, origin top-left.

    /// @brief Current mouse X position (window-space pixels).
    [[nodiscard]] float GetMousePositionX() const
    {
        if (const auto *sample = InputReadScope::Current())
            return sample->mouseX;
        return m_mouseX;
    }

    /// @brief Current mouse Y position (window-space pixels).
    [[nodiscard]] float GetMousePositionY() const
    {
        if (const auto *sample = InputReadScope::Current())
            return sample->mouseY;
        return m_mouseY;
    }

    /// @brief Mouse X movement this frame.
    [[nodiscard]] float GetMouseDeltaX() const
    {
        if (const auto *sample = InputReadScope::Current())
            return sample->mouseDX;
        return m_mouseDX;
    }

    /// @brief Mouse Y movement this frame.
    [[nodiscard]] float GetMouseDeltaY() const
    {
        if (const auto *sample = InputReadScope::Current())
            return sample->mouseDY;
        return m_mouseDY;
    }

    // ---- Scroll wheel (Unity: Input.mouseScrollDelta) ----

    /// @brief Vertical scroll delta this frame (positive = scroll up).
    [[nodiscard]] float GetMouseScrollDeltaY() const
    {
        if (const auto *sample = InputReadScope::Current())
            return sample->scrollY;
        return m_scrollY;
    }

    /// @brief Horizontal scroll delta this frame.
    [[nodiscard]] float GetMouseScrollDeltaX() const
    {
        if (const auto *sample = InputReadScope::Current())
            return sample->scrollX;
        return m_scrollX;
    }
    [[nodiscard]] float GetGameScrollDeltaX() const
    {
        if (const auto *sample = InputReadScope::Current())
            return sample->gameScroll.first;
        return m_gameScroll.first;
    }
    [[nodiscard]] float GetGameScrollDeltaY() const
    {
        if (const auto *sample = InputReadScope::Current())
            return sample->gameScroll.second;
        return m_gameScroll.second;
    }

    // ---- Text input (Unity: Input.inputString) ----

    /// @brief Characters typed this frame (UTF-8).
    [[nodiscard]] const std::string &GetInputString() const
    {
        if (const auto *sample = InputReadScope::Current())
            return sample->inputString;
        return m_inputString;
    }
    [[nodiscard]] const std::string &GetGameInputString() const
    {
        if (const auto *sample = InputReadScope::Current())
            return sample->gameInputString;
        return m_gameInputString;
    }

    /// @brief Ask the current platform window to begin committed text input.
    ///
    /// Desktop and Android use SDL's platform text-input bridge. Browser hosts
    /// install a DOM-backed runtime service and keep this logical state in sync.
    bool StartTextInput();

    /// @brief End platform text input and dismiss its software keyboard.
    void StopTextInput();

    /// @brief True while gameplay requested platform text input.
    [[nodiscard]] bool IsTextInputActive() const
    {
        return m_textInputActive;
    }

    // ---- Touch ----

    /// @brief Number of touch contacts reported by the current frame snapshot.
    [[nodiscard]] int GetTouchCount() const
    {
        if (const auto *sample = InputReadScope::Current())
            return static_cast<int>(sample->touches.size());
        return static_cast<int>(m_touches.size());
    }

    /// @brief Return one stable-indexed touch from the current frame snapshot.
    [[nodiscard]] const TouchState &GetTouch(int index) const;

    /// @brief Return every touch in first-contact order for this frame.
    [[nodiscard]] const std::vector<TouchState> &GetTouches() const
    {
        if (const auto *sample = InputReadScope::Current())
            return sample->touches;
        return m_touches;
    }

    [[nodiscard]] bool HasAccelerometer() const noexcept
    {
        if (const auto *sample = InputReadScope::Current())
            return sample->accelerometerAvailable;
        return m_accelerometerAvailable;
    }

    [[nodiscard]] bool HasGyroscope() const noexcept
    {
        if (const auto *sample = InputReadScope::Current())
            return sample->gyroscopeAvailable;
        return m_gyroscopeAvailable;
    }

    [[nodiscard]] const std::array<float, 3> &GetAcceleration() const noexcept
    {
        if (const auto *sample = InputReadScope::Current())
            return sample->acceleration;
        return m_acceleration;
    }

    [[nodiscard]] const std::array<float, 3> &GetGyroscopeRotationRate() const noexcept
    {
        if (const auto *sample = InputReadScope::Current())
            return sample->gyroscopeRotationRate;
        return m_gyroscopeRotationRate;
    }

    [[nodiscard]] const std::vector<AccelerationEvent> &GetAccelerationEvents() const noexcept
    {
        if (const auto *sample = InputReadScope::Current())
            return sample->accelerationEvents;
        return m_accelerationEvents;
    }

    /// @brief Current logical/framebuffer/safe-area snapshot for the game window.
    [[nodiscard]] const ScreenState &GetScreenState() const
    {
        if (const auto *sample = InputReadScope::Current())
            return sample->screen;
        return m_screenState;
    }

    // ---- File drop (OS drag-drop) ----

    /// @brief Returns true if one or more files were dropped onto the window this frame.
    [[nodiscard]] bool HasDroppedFiles() const
    {
        if (const auto *sample = InputReadScope::Current())
            return !sample->droppedFiles.empty();
        return !m_droppedFiles.empty();
    }

    /// @brief Returns the list of file paths dropped onto the window this frame.
    [[nodiscard]] const std::vector<std::string> &GetDroppedFiles() const
    {
        if (const auto *sample = InputReadScope::Current())
            return sample->droppedFiles;
        return m_droppedFiles;
    }

    // ---- Cursor lock (FPS-style mouse capture) ----

    /// @brief Store the SDL window handle so cursor lock can target it.
    void SetWindow(SDL_Window *window);

    /// @brief Enable/disable cursor lock (SDL relative mouse mode).
    ///        When locked the cursor is hidden and mouse deltas are captured.
    void SetCursorLocked(bool locked);

    /// Editor capture is allowed only after a Game View click. Releasing
    /// capture preserves the game's requested lock for the next click.
    void SetEditorCursorCaptureAllowed(bool allowed);
    [[nodiscard]] bool IsEditorCursorCaptureAllowed() const
    {
        return m_editorCursorCaptureAllowed;
    }

    /// @brief Control cursor visibility independently from relative mouse mode.
    void SetCursorVisible(bool visible);

    [[nodiscard]] bool IsCursorVisible() const
    {
        return m_cursorVisible;
    }

    /// Effective visibility while the native window has input focus.
    [[nodiscard]] bool IsCursorHidden() const
    {
        return IsCursorLocked() || IsEditorMouseCaptureActive() ||
               (!m_cursorVisible &&
                (!m_editorCursorCaptureManaged || m_editorCursorCaptureAllowed || GetGameMouseButton(0)));
    }

    /// @brief Confine the visible cursor to the game window without enabling relative motion.
    void SetCursorConfined(bool confined);

    [[nodiscard]] bool IsCursorConfined() const
    {
        return m_cursorConfined;
    }

    /// @brief Warp the visible cursor in logical window coordinates.
    /// @return false when the current host cannot perform a hardware cursor warp.
    bool WarpCursor(float x, float y);
    [[nodiscard]] bool HasPendingCursorWarp() const
    {
        return m_warpMotionPending;
    }

    /// @brief Enable/disable editor-only mouse capture for Scene view camera drag.
    ///        Uses the same SDL relative mouse mode backend but does not report
    ///        as gameplay cursor lock to Python game scripts or ImGui suppression.
    void SetEditorMouseCapture(bool captured);

    /// @brief Consume all relative mouse movement received during editor capture.
    ///
    /// Editor UI construction can run slower than the renderer. Keeping this
    /// accumulator independent from the per-frame gameplay delta prevents
    /// Scene view navigation from dropping motion between UI updates.
    [[nodiscard]] std::pair<float, float> ConsumeEditorMouseDelta();

    /// @brief Returns true when cursor lock is active.
    [[nodiscard]] bool IsCursorLocked() const
    {
        return m_cursorLocked && (!m_editorCursorCaptureManaged || m_editorCursorCaptureAllowed);
    }

    /// @brief Returns true when editor Scene view mouse capture is active.
    [[nodiscard]] bool IsEditorMouseCaptureActive() const
    {
        return m_editorMouseCaptured;
    }

    // ---- Utility ----

    /// @brief Reset all input state (e.g. on window focus loss or scene change).
    void ResetAll();

    /// @brief Map a human-readable key name to SDL_Scancode. Case-insensitive.
    ///        Returns -1 if the name is unknown.
    static int NameToScancode(const std::string &name);

    /// @brief Get the human-readable name for a scancode.
    static const char *ScancodeToName(int scancode);

  private:
    InputManager();
    ~InputManager() = default;
    InputManager(const InputManager &) = delete;
    InputManager &operator=(const InputManager &) = delete;

    // ---- State buffers ----
    std::array<uint8_t, INPUT_MAX_KEYS> m_keys{};
    std::array<uint8_t, INPUT_MAX_KEYS> m_keyDown{};
    std::array<uint8_t, INPUT_MAX_KEYS> m_keyUp{};
    std::array<uint8_t, INPUT_MAX_KEYS> m_keyDownConsumed{};
    std::array<uint8_t, INPUT_MAX_KEYS> m_gameKeyOwnership{};
    std::array<uint8_t, INPUT_MAX_MOUSE_BUTTONS> m_mouseButtons{};
    std::array<uint8_t, INPUT_MAX_MOUSE_BUTTONS> m_mouseButtonDown{};
    std::array<uint8_t, INPUT_MAX_MOUSE_BUTTONS> m_mouseButtonUp{};

    float m_mouseX = 0.f;
    float m_mouseY = 0.f;
    float m_mouseDX = 0.f;
    float m_mouseDY = 0.f;
    float m_editorMouseDX = 0.f;
    float m_editorMouseDY = 0.f;
    float m_scrollX = 0.f;
    float m_scrollY = 0.f;
    std::pair<float, float> m_gameScroll{};

    float m_syntheticMouseX = 0.f;
    float m_syntheticMouseY = 0.f;
    bool m_hasSyntheticMousePosition = false;
    bool m_syntheticInputThisFrame = false;
    std::array<uint8_t, INPUT_MAX_KEYS> m_syntheticKeys{};
    // Synthetic key edges are latched independently of SDL's focus routing.
    // MCP automation may run while the editor window is not foregrounded;
    // gameplay must still observe one GetKeyDown edge for the queued press.
    std::array<uint8_t, INPUT_MAX_KEYS> m_syntheticKeyDown{};
    std::array<uint8_t, INPUT_MAX_MOUSE_BUTTONS> m_syntheticMouseButtons{};
    uint32_t m_syntheticHeldCount = 0;

    std::string m_inputString;
    std::string m_gameInputString;
    std::vector<TouchState> m_touches;
    std::vector<AccelerationEvent> m_accelerationEvents;
    std::vector<std::string> m_droppedFiles;

    std::array<float, 3> m_acceleration{};
    std::array<float, 3> m_gyroscopeRotationRate{};
    SDL_Sensor *m_accelerometer = nullptr;
    SDL_Sensor *m_gyroscope = nullptr;
    uint32_t m_accelerometerId = 0;
    uint32_t m_gyroscopeId = 0;
    uint64_t m_lastAccelerationTimestampNs = 0;
    bool m_accelerometerAvailable = false;
    bool m_gyroscopeAvailable = false;

    SDL_Window *m_window = nullptr;
    bool m_cursorLocked = false;
    bool m_editorCursorCaptureManaged = false;
    bool m_editorCursorCaptureAllowed = false;
    std::array<float, 4> m_editorGameViewport{};
    std::array<float, 4> m_editorGameInputRect{};
    uint32_t m_editorGameWindowId = 0;
    uint32_t m_gameMouseButtonOwnership = 0;
    bool m_cursorVisible = true;
    bool m_cursorConfined = false;
    bool m_warpMotionPending = false;
    float m_warpTargetX = 0.0f;
    float m_warpTargetY = 0.0f;
    bool m_editorMouseCaptured = false;
    bool m_textInputActive = false;
    ScreenState m_screenState;
    uint64_t m_frameIndex = 0;
    friend class InputReadScope;
    [[nodiscard]] std::shared_ptr<InputFrameReadState> AcquireWorldReadState(uint64_t worldId);
    [[nodiscard]] std::shared_ptr<InputFrameReadState> FindWorldReadState(uint64_t worldId) noexcept;
    [[nodiscard]] InputFrameSample *OverrideReadFrame();
    std::unordered_map<uint64_t, std::shared_ptr<InputFrameReadState>> m_worldReadStates;
    InputFrameLease m_capturedFrame;
    std::shared_ptr<const WorldGameplayRouting> m_worldGameplayRouting;
    uint64_t m_hostReadRevision = 0;
    uint64_t m_sampleIdentity = 0;
    enum class SampleEventKind
    {
        KeyDown,
        KeyUp,
        SyntheticKeyDown,
        MouseDown,
        MouseUp,
        Motion,
        Scroll,
        Text,
        Touch,
        Acceleration,
        Synthetic,
        DroppedFile
    };
    struct SampleEvent
    {
        SampleEventKind kind;
        int index = 0;
        float x = 0, y = 0;
        bool game = false;
        std::string text;
        TouchState touch;
        AccelerationEvent acceleration;
    };
    std::vector<SampleEvent> m_sampleEvents;
    void InvalidateCapture() noexcept
    {
        m_capturedFrame.reset();
    }
    void AppendSampleEvent(SampleEvent event);
    static void ClearSampleTransients(InputFrameSample &sample);
    bool m_gameplayFocused = true;
    std::array<float, 2> m_gameplayViewportOrigin{}, m_gameplayViewportSize{}, m_gameplayRenderSize{};
    std::array<float, 4> m_gameplayTouchViewport{0, 0, 1, 1};

    // ---- Name → scancode lookup ----
    static std::unordered_map<std::string, int> s_nameToScancode;
    static bool s_nameTableBuilt;
    static void BuildNameTable();

    void ResetPhysicalInputForFocusLoss();
    void CancelActiveTouches(const std::string &reason);
    void RefreshScreenState();
    void ApplyCursorState();
    [[nodiscard]] bool IsGameMouseButtonOwned(int button) const;
    [[nodiscard]] bool IsGameKeyOwned(int scancode) const;
    [[nodiscard]] std::pair<float, float> CursorLockPosition(bool gameLocked) const;
};

} // namespace infernux
