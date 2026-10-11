#pragma once

#include <string>

struct SDL_Window;

namespace infernux
{

/// Configure the Windows process for Per-Monitor V2 before SDL video starts.
void ConfigureRequiredWindowsDpiPolicy();

/// Verify that SDL established the required Windows Per-Monitor V2 context.
void VerifyRequiredWindowsDpiPolicy(SDL_Window *window = nullptr);

/// Startup diagnostics use the native window, not the monitor's advertised scale.
std::string DescribeWindowsDpiPolicy(SDL_Window *window);

} // namespace infernux
