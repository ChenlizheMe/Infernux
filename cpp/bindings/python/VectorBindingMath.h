#pragma once

#include <glm/glm.hpp>

namespace infernux::vector_binding
{
template <typename Vector> Vector Normalize(const Vector &value)
{
    const float magnitude = glm::length(value);
    if (magnitude < 1e-6f)
        return Vector(0.f);
    return value / magnitude;
}

template <typename Vector> Vector ClampMagnitude(const Vector &value, float maxLength)
{
    const float magnitude = glm::length(value);
    if (magnitude > maxLength && magnitude > 1e-6f)
        return value * (maxLength / magnitude);
    return value;
}
} // namespace infernux::vector_binding
