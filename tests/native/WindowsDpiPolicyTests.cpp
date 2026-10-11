#include <platform/window/WindowsDpiPolicy.h>

#include <SDL3/SDL.h>

#include <exception>
#include <iostream>

int main()
{
    try {
        infernux::ConfigureRequiredWindowsDpiPolicy();
        if (!SDL_Init(SDL_INIT_VIDEO)) {
            std::cerr << "SDL video initialization failed: " << SDL_GetError() << '\n';
            return 1;
        }
        infernux::VerifyRequiredWindowsDpiPolicy();
        SDL_Window *window = SDL_CreateWindow("DPI policy regression", 800, 600,
                                              SDL_WINDOW_HIDDEN | SDL_WINDOW_HIGH_PIXEL_DENSITY | SDL_WINDOW_RESIZABLE);
        if (!window)
            throw std::runtime_error(SDL_GetError());
        infernux::VerifyRequiredWindowsDpiPolicy(window);
        const std::string description = infernux::DescribeWindowsDpiPolicy(window);
        if (description.find("native_client=800x600") == std::string::npos ||
            description.find("native_dpi=unavailable") != std::string::npos)
            throw std::runtime_error(description);
        std::cout << description << '\n';
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 0;
    } catch (const std::exception &error) {
        SDL_Quit();
        std::cerr << error.what() << '\n';
        return 2;
    }
}
