#pragma once

#include <chrono>

#include <core/log/InxLog.h>
#include <core/types/InxApplication.h>

#include <vulkan/vulkan.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

namespace infernux
{

/// Power-save / idle configuration for the editor main loop.
/// When no user input is detected for a short period, the loop sleeps
/// via ``SDL_WaitEventTimeout`` to reduce CPU/GPU usage.
/// An editor FPS cap also limits the maximum frame rate in edit mode.
struct FpsIdling
{
    float fpsIdle = 10.0f;          ///< Target FPS when idling (0 = disable idle)
    float editorFpsCap = 60.0f;     ///< Max FPS in editor mode (0 = uncapped)
    bool enableIdling = true;       ///< Master switch for idle detection
    bool isIdling = false;          ///< Output — true when the last frame went idle
};

class InxView
{
  public:
    friend class InxRenderer;

    InxView();

    InxView(const InxView &) = delete;
    InxView(InxView &&) = delete;
    InxView &operator=(const InxView &) = delete;
    InxView &operator=(InxView &&) = delete;

    const char *const *GetVkExtensions(uint32_t *count);

    void Init(int width, int height);
    void ProcessEvent();
    void Quit();

    int GetUserEvent();
    void Show();
    void Hide();
    void SetWindowIcon(const std::string &iconPath);
    void SetWindowFullscreen(bool fullscreen);
    void SetWindowTitle(const std::string &title);
    void SetWindowMaximized(bool maximized);
    void SetWindowResizable(bool resizable);

    /// Close-request interception: SDL_EVENT_QUIT sets this flag instead of
    /// immediately terminating.  Python checks the flag each frame and may
    /// show a "save before exit?" dialog before calling ConfirmClose().
    bool IsCloseRequested() const
    {
        return m_closeRequested;
    }
    void ConfirmClose()
    {
        m_keepRunning = false;
    }
    void CancelClose()
    {
        m_closeRequested = false;
    }

    bool IsMinimized() const
    {
        return m_isMinimized;
    }

    // ---- Power-save / idle accessors ----
    FpsIdling &GetIdling()
    {
        return m_idling;
    }
    const FpsIdling &GetIdling() const
    {
        return m_idling;
    }

    /// Tell InxView whether the engine is in play mode.
    /// When true, the frame-rate cap and idle sleep are both disabled.
    void SetPlayMode(bool play)
    {
        m_isPlayMode = play;
    }
    bool IsPlayMode() const
    {
        return m_isPlayMode;
    }

    /// Signal that the current frame required full-speed rendering
    /// (e.g. animation playing, programmatic scene change).
    /// Resets the idle cooldown so the next few frames run at editor cap.
    void RequestFullSpeedFrame()
    {
        m_activeFramesRemaining = ACTIVE_COOLDOWN_FRAMES;
    }

    void CreateSurface(VkInstance *vkInstance, VkSurfaceKHR *vkSurface);
    void SetAppMetadata(InxAppMetadata appMetaData);

  private:
    int m_windowWidth = 0;
    int m_windowHeight = 0;

    SDL_Window *m_window = nullptr;

    bool m_keepRunning;
    bool m_closeRequested = false;
    bool m_isMinimized = false;
    bool m_isPlayMode = false;
    InxAppMetadata m_appMetadata;

    // ---- Power-save idle state ----
    FpsIdling m_idling;
    /// Number of full-speed frames remaining after the last user interaction.
    /// When this reaches 0 and idling is enabled the loop will sleep more.
    static constexpr int ACTIVE_COOLDOWN_FRAMES = 3;
    int m_activeFramesRemaining = ACTIVE_COOLDOWN_FRAMES;

    /// Timestamp of the last frame start — used to compute the remaining
    /// frame budget so the sleep duration adapts to actual render time.
    std::chrono::steady_clock::time_point m_lastFrameStart = std::chrono::steady_clock::now();

    void SDLInit();
};
} // namespace infernux
