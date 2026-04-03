#include "InxView.h"

#include <chrono>
#include <iostream>

#include <imgui_impl_sdl3.h>
#include <platform/filesystem/InxPath.h>
#include <platform/input/InputManager.h>
#include <stb_image.h>

namespace infernux
{
InxView::InxView()
{
}

const char *const *InxView::GetVkExtensions(uint32_t *count)
{
    INXLOG_DEBUG("Get Vulkan Extensions.");
    unsigned int extensionCount = 0;
    const char *const *extensions = SDL_Vulkan_GetInstanceExtensions(&extensionCount);
    if (!extensions) {
        INXLOG_ERROR("SDL_Vulkan_GetInstanceExtensions failed: ", SDL_GetError());
        return nullptr;
    }
    if (count) {
        *count = extensionCount;
    }
    return extensions;
}

void InxView::Init(int width, int height)
{
    m_keepRunning = true;
    m_windowWidth = width;
    m_windowHeight = height;

    INXLOG_DEBUG("Initialize InxView Window with size: ", m_windowWidth, "x", m_windowHeight);
    SDLInit();
}

void InxView::ProcessEvent()
{
    // Begin a new input frame: swap current → previous, clear deltas
    InputManager::Instance().SetWindow(m_window);
    InputManager::Instance().BeginFrame();

    // ====================================================================
    // Frame-rate limiter
    //
    // Three tiers:
    //   play mode      → no sleep, full speed (bypass entirely)
    //   editor active  → hard cap to editorFpsCap via SDL_Delay
    //   editor idle    → sleep via SDL_WaitEventTimeout, wake on input
    //
    // We measure elapsed time since the last frame start and sleep only
    // for the *remaining* budget.  Active mode uses SDL_Delay (hard cap);
    // idle mode uses SDL_WaitEventTimeout with a real event struct so the
    // thread wakes immediately on user input and no events are lost.
    // ====================================================================
    m_idling.isIdling = false;

    SDL_Event firstEvent{};
    bool gotFirstEvent = false;

    if (!m_isPlayMode) {
        bool isIdle = m_idling.enableIdling && m_idling.fpsIdle > 0.0f && m_activeFramesRemaining <= 0;
        float targetFps = isIdle ? m_idling.fpsIdle : m_idling.editorFpsCap;

        if (targetFps > 0.0f) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - m_lastFrameStart).count();
            double budget = 1.0 / static_cast<double>(targetFps);
            int sleepMs = static_cast<int>((budget - elapsed) * 1000.0);

            if (sleepMs > 0) {
                if (isIdle) {
                    // Idle: block until an event arrives OR the timeout expires.
                    // A real event struct is used so the event data is preserved.
                    gotFirstEvent = SDL_WaitEventTimeout(&firstEvent, sleepMs);

                    auto afterWait = std::chrono::steady_clock::now();
                    double waitSec = std::chrono::duration<double>(afterWait - now).count();
                    m_idling.isIdling = (waitSec > budget * 0.9);
                } else {
                    // Active editor: hard sleep for the remaining frame budget.
                    SDL_Delay(sleepMs);
                }
            }
        }
    }

    // Always keep m_lastFrameStart current (even in play mode) so the
    // first editor frame after exiting play mode doesn't see a huge elapsed.
    m_lastFrameStart = std::chrono::steady_clock::now();

    // ---- Poll & process all pending events ----
    bool hadInputEvent = false;

    auto processOneEvent = [&](SDL_Event &e) {
        bool forwardToImGui = true;
        if (InputManager::Instance().IsEditorMouseCaptureActive()) {
            if (e.type == SDL_EVENT_MOUSE_MOTION)
                forwardToImGui = false;
        }

        if (forwardToImGui) {
            ImGui_ImplSDL3_ProcessEvent(&e);
        }

        InputManager::Instance().ProcessSDLEvent(e);

        switch (e.type) {
        case SDL_EVENT_MOUSE_MOTION:
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
        case SDL_EVENT_MOUSE_BUTTON_UP:
        case SDL_EVENT_MOUSE_WHEEL:
        case SDL_EVENT_KEY_DOWN:
        case SDL_EVENT_KEY_UP:
        case SDL_EVENT_TEXT_INPUT:
        case SDL_EVENT_DROP_FILE:
        case SDL_EVENT_DROP_TEXT:
            hadInputEvent = true;
            break;
        default:
            break;
        }

        if (e.type == SDL_EVENT_QUIT) {
            m_closeRequested = true;
        }

        if (e.type == SDL_EVENT_WINDOW_MINIMIZED) {
            m_isMinimized = true;
        }
        if (e.type == SDL_EVENT_WINDOW_RESTORED || e.type == SDL_EVENT_WINDOW_EXPOSED ||
            e.type == SDL_EVENT_WINDOW_FOCUS_GAINED) {
            m_isMinimized = false;
            hadInputEvent = true;
        }
        if (e.type == SDL_EVENT_WINDOW_OCCLUDED) {
            m_isMinimized = true;
        }
    };

    // Process the event captured by SDL_WaitEventTimeout (if any)
    if (gotFirstEvent) {
        processOneEvent(firstEvent);
    }

    // Drain remaining queued events
    SDL_Event event{};
    while (SDL_PollEvent(&event)) {
        processOneEvent(event);
        if (m_closeRequested)
            break;
    }

    // Reset idle cooldown when user interacted
    if (hadInputEvent) {
        m_activeFramesRemaining = ACTIVE_COOLDOWN_FRAMES;
    } else if (m_activeFramesRemaining > 0) {
        --m_activeFramesRemaining;
    }

    SDL_GetWindowSize(m_window, &m_windowWidth, &m_windowHeight);
}

void InxView::Quit()
{
    if (m_window) {
        SDL_DestroyWindow(m_window);
        m_window = nullptr;
    }
    // Note: We intentionally don't call SDL_Quit() here to avoid
    // affecting other parts of the application (like a launcher).
    // SDL_Quit() would terminate all SDL subsystems which could
    // cause issues if the application continues running.
    INXLOG_DEBUG("Quit the InxView Window.");
}

int InxView::GetUserEvent()
{
    return m_keepRunning ? 1 : 0;
}

void InxView::Show()
{
    if (m_window) {
        SDL_ShowWindow(m_window);
    } else {
        INXLOG_ERROR("InxView Window is not initialized.");
    }
}

void InxView::Hide()
{
    if (m_window) {
        SDL_HideWindow(m_window);
    } else {
        INXLOG_ERROR("InxView Window is not initialized.");
    }
}

void InxView::SetWindowIcon(const std::string &iconPath)
{
    if (!m_window) {
        INXLOG_ERROR("Cannot set window icon: window not initialized.");
        return;
    }

    int w = 0, h = 0, channels = 0;
    // Read via ReadFileBytes to support Unicode paths on Windows
    std::vector<unsigned char> fileBytes;
    if (!ReadFileBytes(iconPath, fileBytes) || fileBytes.empty()) {
        INXLOG_ERROR("Failed to read icon file: ", iconPath);
        return;
    }
    unsigned char *pixels =
        stbi_load_from_memory(fileBytes.data(), static_cast<int>(fileBytes.size()), &w, &h, &channels, 4);
    if (!pixels) {
        INXLOG_ERROR("Failed to load icon: ", iconPath);
        return;
    }

    SDL_Surface *surface = SDL_CreateSurfaceFrom(w, h, SDL_PIXELFORMAT_RGBA32, pixels, w * 4);
    if (surface) {
        SDL_SetWindowIcon(m_window, surface);
        SDL_DestroySurface(surface);
        INXLOG_DEBUG("Window icon set from: ", iconPath);
    } else {
        INXLOG_ERROR("Failed to create SDL surface for icon: ", SDL_GetError());
    }

    stbi_image_free(pixels);
}

void InxView::SetWindowFullscreen(bool fullscreen)
{
    if (!m_window) {
        INXLOG_ERROR("Cannot set fullscreen: window not initialized.");
        return;
    }
    if (!SDL_SetWindowFullscreen(m_window, fullscreen)) {
        INXLOG_ERROR("SDL_SetWindowFullscreen failed: ", SDL_GetError());
    }
}

void InxView::SetWindowTitle(const std::string &title)
{
    if (!m_window) {
        INXLOG_ERROR("Cannot set window title: window not initialized.");
        return;
    }
    if (!SDL_SetWindowTitle(m_window, title.c_str())) {
        INXLOG_ERROR("SDL_SetWindowTitle failed: ", SDL_GetError());
    }
}

void InxView::SetWindowMaximized(bool maximized)
{
    if (!m_window) {
        INXLOG_ERROR("Cannot set maximized: window not initialized.");
        return;
    }
    if (maximized) {
        SDL_MaximizeWindow(m_window);
    } else {
        SDL_RestoreWindow(m_window);
    }
}

void InxView::SetWindowResizable(bool resizable)
{
    if (!m_window) {
        INXLOG_ERROR("Cannot set resizable: window not initialized.");
        return;
    }
    SDL_SetWindowResizable(m_window, resizable);
}

void InxView::SDLInit()
{
    SDL_SetLogPriorities(SDL_LOG_PRIORITY_VERBOSE);
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO)) {
        INXLOG_ERROR("SDL_Init failed: ", SDL_GetError());
    } else {
        INXLOG_DEBUG("SDL_Init succeeded.");
    }

    INXLOG_DEBUG("Window engine: SDL Vulkan");
    m_window =
        SDL_CreateWindow(m_appMetadata.appName, m_windowWidth, m_windowHeight,
                         SDL_WINDOW_RESIZABLE | SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (!m_window) {
        INXLOG_ERROR("Could not create a window: ", SDL_GetError());
    } else {
        INXLOG_DEBUG("Window created successfully.");
    }

    SDL_MaximizeWindow(m_window);
}

void InxView::CreateSurface(VkInstance *vkInstance, VkSurfaceKHR *vkSurface)
{
    if (!SDL_Vulkan_CreateSurface(m_window, *vkInstance, nullptr, vkSurface)) {
        INXLOG_FATAL("Could not create Vulkan surface: ", SDL_GetError());
    } else {
        INXLOG_DEBUG("Vulkan surface created successfully.");
    }
}

void InxView::SetAppMetadata(InxAppMetadata appMetaData)
{
    m_appMetadata = appMetaData;
    INXLOG_DEBUG("Set InxView application metadata: ", m_appMetadata.appName);
}
} // namespace infernux
