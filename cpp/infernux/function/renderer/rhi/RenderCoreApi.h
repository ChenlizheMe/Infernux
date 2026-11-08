#pragma once

#if defined(_WIN32)
#if defined(INFERNUX_RENDER_CORE_EXPORTS)
#define INFERNUX_RENDER_CORE_API __declspec(dllexport)
#else
#define INFERNUX_RENDER_CORE_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define INFERNUX_RENDER_CORE_API __attribute__((visibility("default")))
#else
#define INFERNUX_RENDER_CORE_API
#endif
