#pragma once

#include <algorithm>
#include <cmath>
#include <glm/gtc/quaternion.hpp>

namespace infernux::rotation
{
inline glm::quat Slerp(const glm::quat &start, const glm::quat &end, float progress)
{
    const float t = std::clamp(progress, 0.f, 1.f);
    const float dot = glm::dot(start, end);
    const glm::quat target = dot < 0.f ? -end : end;
    const float cosine = std::abs(dot);
    // Near-parallel rotations use normalized linear interpolation. This is a
    // numerical branch, not input repair: the spherical branch retains the
    // supplied magnitudes, including zero-valued quaternion storage.
    if (cosine >= .95f)
        return glm::normalize(start * (1.f - t) + target * t);
    const float angle = std::acos(cosine);
    return (start * std::sin((1.f - t) * angle) + target * std::sin(t * angle)) / std::sin(angle);
}
} // namespace infernux::rotation
