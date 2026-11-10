#pragma once
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
} // namespace infernux
