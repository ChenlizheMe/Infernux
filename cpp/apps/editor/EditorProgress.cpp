#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>

// A separate event loop keeps progress repainting while the Editor thread or
// Python GIL is occupied. EOF also closes the window if the Editor exits.
int main(int, char **)
{
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::fprintf(stderr, "%s\n", SDL_GetError());
        return 1;
    }
    SDL_Window *window = SDL_CreateWindow("Infernux - Preparing Play", 520, 150,
                                          SDL_WINDOW_HIDDEN | SDL_WINDOW_BORDERLESS | SDL_WINDOW_ALWAYS_ON_TOP |
                                              SDL_WINDOW_NOT_FOCUSABLE);
    SDL_Renderer *renderer = window ? SDL_CreateRenderer(window, "software") : nullptr;
    if (!renderer) {
        std::fprintf(stderr, "%s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
    std::atomic<bool> done{false};
    std::mutex mutex;
    float progress = 0.0f;
    std::string message = "Preparing Play mode...";
    std::thread input([&] {
        char line[512];
        while (std::fgets(line, sizeof(line), stdin)) {
            float value;
            int offset = 0;
            if (std::sscanf(line, "%f\t%n", &value, &offset) != 1 || offset == 0)
                continue;
            std::lock_guard lock(mutex);
            progress = std::clamp(value, 0.0f, 1.0f);
            message = line + offset;
            message.erase(message.find_last_not_of("\r\n") + 1);
        }
        done.store(true);
    });
    const Uint64 started = SDL_GetTicks();
    bool shown = false;
    while (!done.load()) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
        }
        const Uint64 elapsed = SDL_GetTicks() - started;
        if (!shown && elapsed >= 400) {
            SDL_ShowWindow(window);
            shown = true;
        }
        if (shown) {
            float value;
            std::string label;
            {
                std::lock_guard lock(mutex);
                value = progress;
                label = message;
            }
            SDL_SetRenderDrawColor(renderer, 27, 27, 30, 255);
            SDL_RenderClear(renderer);
            SDL_SetRenderScale(renderer, 2.0f, 2.0f);
            SDL_SetRenderDrawColor(renderer, 239, 235, 225, 255);
            SDL_RenderDebugText(renderer, 12, 10, "INFERNUX / PREPARING PLAY");
            SDL_SetRenderScale(renderer, 1.5f, 1.5f);
            SDL_RenderDebugText(renderer, 16, 40, label.c_str());
            SDL_SetRenderScale(renderer, 1.0f, 1.0f);
            SDL_FRect bar{24, 108, 472, 6};
            SDL_SetRenderDrawColor(renderer, 64, 64, 68, 255);
            SDL_RenderFillRect(renderer, &bar);
            bar.w *= value;
            SDL_SetRenderDrawColor(renderer, 201, 54, 64, 255);
            SDL_RenderFillRect(renderer, &bar);
            SDL_FRect pulse{24 + static_cast<float>(elapsed % 1600) / 1600.0f * 460, 122, 12, 2};
            SDL_RenderFillRect(renderer, &pulse);
            SDL_RenderPresent(renderer);
        }
        SDL_Delay(30);
    }
    input.join();
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
