#include "WindowsDpiPolicy.h"

#include <SDL3/SDL.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace infernux
{

void ConfigureRequiredWindowsDpiPolicy()
{
#if defined(_WIN32)
    if (!SDL_SetHint("SDL_WINDOWS_DPI_AWARENESS", "permonitorv2"))
        throw std::runtime_error("SDL rejected the required Windows Per-Monitor V2 DPI policy");
#endif
}

void VerifyRequiredWindowsDpiPolicy(SDL_Window *window)
{
#if defined(_WIN32)
    HMODULE user32 = GetModuleHandleW(L"user32.dll");
    if (user32 == nullptr)
        throw std::runtime_error("Windows user32.dll is unavailable for DPI policy verification");

    using GetThreadDpiAwarenessContextFn = HANDLE(WINAPI *)();
    using AreDpiAwarenessContextsEqualFn = BOOL(WINAPI *)(HANDLE, HANDLE);
    const auto getThreadContext =
        reinterpret_cast<GetThreadDpiAwarenessContextFn>(GetProcAddress(user32, "GetThreadDpiAwarenessContext"));
    const auto contextsEqual =
        reinterpret_cast<AreDpiAwarenessContextsEqualFn>(GetProcAddress(user32, "AreDpiAwarenessContextsEqual"));
    if (getThreadContext == nullptr || contextsEqual == nullptr)
        throw std::runtime_error("Windows Per-Monitor V2 DPI APIs are unavailable");

    const HANDLE perMonitorV2 = reinterpret_cast<HANDLE>(static_cast<intptr_t>(-4));
    if (!contextsEqual(getThreadContext(), perMonitorV2))
        throw std::runtime_error("Windows rejected the required Per-Monitor V2 DPI policy");

    if (window != nullptr) {
        using GetWindowDpiAwarenessContextFn = HANDLE(WINAPI *)(HWND);
        const auto getWindowContext =
            reinterpret_cast<GetWindowDpiAwarenessContextFn>(GetProcAddress(user32, "GetWindowDpiAwarenessContext"));
        const auto hwnd = static_cast<HWND>(
            SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr));
        if (!hwnd || !getWindowContext || !contextsEqual(getWindowContext(hwnd), perMonitorV2))
            throw std::runtime_error("The native window is not Per-Monitor V2 DPI aware; check Windows high-DPI "
                                     "compatibility overrides on the Python/Player executable");
    }
#else
    (void)window;
#endif
}

std::string DescribeWindowsDpiPolicy(SDL_Window *window)
{
#if defined(_WIN32)
    const auto hwnd = static_cast<HWND>(
        SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr));
    using GetDpiForWindowFn = UINT(WINAPI *)(HWND);
    const auto getDpi =
        reinterpret_cast<GetDpiForWindowFn>(GetProcAddress(GetModuleHandleW(L"user32.dll"), "GetDpiForWindow"));
    RECT client{};
    if (!hwnd || !getDpi || !GetClientRect(hwnd, &client))
        return "native_dpi=unavailable";
    std::ostringstream details;
    details << "native_dpi=" << getDpi(hwnd) << " native_client=" << client.right - client.left << 'x'
            << client.bottom - client.top;
    return details.str();
#else
    (void)window;
    return {};
#endif
}

} // namespace infernux
