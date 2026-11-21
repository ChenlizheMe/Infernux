#pragma once

#include <glm/vec3.hpp>

namespace infernux
{
/// Numerical inputs captured together when a World admits a fixed step.
struct PhysicsFixedInputs
{
    float duration;
    glm::vec3 gravity;

    [[nodiscard]] bool operator==(const PhysicsFixedInputs &other) const noexcept
    {
        return duration == other.duration && gravity == other.gravity;
    }
};
} // namespace infernux
