#pragma once

#if defined(_WIN32)
#if defined(INFERNUX_ASSET_RUNTIME_EXPORTS)
#define INFERNUX_ASSET_RUNTIME_API __declspec(dllexport)
#else
#define INFERNUX_ASSET_RUNTIME_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define INFERNUX_ASSET_RUNTIME_API __attribute__((visibility("default")))
#else
#define INFERNUX_ASSET_RUNTIME_API
#endif
