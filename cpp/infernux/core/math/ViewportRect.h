#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <glm/glm.hpp>

namespace infernux
{
/// Preserve authored Rect separately. Pixel metrics and raster work use the
/// visible, nonnegative intersection with the normalized output rectangle.
inline glm::vec4 ClipViewportRect(const glm::vec4 &rect)
{
    const auto begin = glm::clamp(glm::vec2(rect), glm::vec2(0), glm::vec2(1));
    const auto end = glm::clamp(glm::vec2(rect) + glm::vec2(rect.z, rect.w), begin, glm::vec2(1));
    return glm::vec4(begin, end - begin);
}
/// Top-left integer rectangle used by both camera pixel metrics and raster work.
inline glm::uvec4 RasterViewportRect(const glm::vec4 &rect, uint32_t width, uint32_t height)
{
    const auto visible = ClipViewportRect(rect);
    const auto pixels = [](float normalized, uint32_t extent) {
        // Keep the existing float multiplication/rounding before the conversion.
        // Clamp before conversion as UINT32_MAX is not exactly representable in float.
        return static_cast<uint32_t>(std::min(static_cast<double>(extent),
                                              static_cast<double>(std::round(std::max(0.0f, normalized) * extent))));
    };
    const auto x = pixels(visible.x, width);
    const auto y = pixels(1.0f - visible.y - visible.w, height);
    return {x, y, std::min(pixels(visible.z, width), width - x), std::min(pixels(visible.w, height), height - y)};
}
} // namespace infernux
