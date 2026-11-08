#pragma once

#if defined(_WIN32)
#if defined(INFERNUX_FOUNDATION_EXPORTS)
#define INFERNUX_FOUNDATION_API __declspec(dllexport)
#elif defined(INFERNUX_FOUNDATION_IMPORTS)
#define INFERNUX_FOUNDATION_API __declspec(dllimport)
#else
#define INFERNUX_FOUNDATION_API
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define INFERNUX_FOUNDATION_API __attribute__((visibility("default")))
#else
#define INFERNUX_FOUNDATION_API
#endif
