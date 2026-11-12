#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace infernux
{

/// Authored configuration, kept outside the per-frame physics data pool.
struct RigidbodyMassProperties
{
    glm::vec3 center{0.0f};
    glm::vec3 moments{1.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
};

} // namespace infernux
