#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <variant>

namespace infernux
{

struct PhysicsQuerySphere
{
    glm::vec3 center;
    float radius;
};

struct PhysicsQueryBox
{
    glm::vec3 center;
    glm::vec3 halfExtents;
    glm::quat rotation;
};

struct PhysicsQueryCapsule
{
    glm::vec3 point0;
    glm::vec3 point1;
    float radius;
};

using PhysicsQueryShape = std::variant<PhysicsQuerySphere, PhysicsQueryBox, PhysicsQueryCapsule>;

} // namespace infernux
