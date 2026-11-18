#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace infernux::theme_drawing
{

inline constexpr int SurfacePatternSize = 256;

inline float PatternNoise(uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return static_cast<float>((value ^ (value >> 16)) & 65535U) / 65535.0f;
}

// White RGBA masks are tinted by theme data. The GUI uploads one tile per
// requested kind and keeps it resident; no rasterization runs during replay.
inline std::vector<unsigned char> BuildSurfacePattern(int kind)
{
    constexpr int size = SurfacePatternSize;
    std::vector<unsigned char> pixels(size * size * 4, 255);
    const auto ink = [](float distance, float width = 0.8f) {
        return std::clamp(width - std::abs(distance), 0.0f, 1.0f);
    };
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float density = 0.0f;
            if (kind == 1) {
                const float dx = x - 128.0f, dy = y - 128.0f;
                const float radius = std::sqrt(dx * dx + dy * dy);
                const float angle = std::atan2(dy, dx);
                const float wobble = std::sin(angle * 13.0f) * 0.30f;
                density = std::max(ink(radius - 83.0f + wobble), ink(radius - 48.0f + wobble) * 0.65f);
                if (radius < 111.0f)
                    density = std::max(density, std::max(ink(dx, 0.55f), ink(dy, 0.55f)) * 0.6f);
                if (radius > 75.0f && radius < 94.0f)
                    density = std::max(density, ink(std::sin(angle * 24.0f) * radius, 0.7f) * 0.7f);
                const float diagonal = dy - dx * 0.65f;
                if (radius < 107.0f)
                    density = std::max(density, ink(diagonal, 0.7f) * 0.55f);
                // Fine construction marks and broken cross-hatching.
                if ((x + y) % 13 == 0 && x > 145 && y > 150 && radius < 78)
                    density = std::max(density, 0.25f);
                density *= 0.65f + PatternNoise(x / 3U + y * 31U) * 0.35f;
            } else if (kind == 2) {
                const float band = PatternNoise(y * 719U);
                const float strand = PatternNoise(x / 36U + y * 991U);
                density = 0.10f + band * 0.52f + strand * 0.12f;
                if (band > 0.95f && (x + y * 7) % 171 < 93)
                    density += 0.22f;
            } else if (kind == 3) {
                density = y % 4 == 0 ? 0.45f : 0.0f;
                if (x % 4 == 0 && y % 4 == 2)
                    density = 0.24f;
                if (y % 64 == 0)
                    density = 0.7f;
            }
            pixels[(y * size + x) * 4 + 3] =
                static_cast<unsigned char>(std::clamp(density, 0.0f, 1.0f) * 255.0f);
        }
    }
    return pixels;
}

} // namespace infernux::theme_drawing
