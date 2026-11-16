#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace infernux
{
struct PhysicsBodyPoseUpdate
{
    uint32_t bodyId = 0xFFFFFFFF;
    glm::vec3 position{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
};

/// Fields requested from one accepted body state. Public identity and shape
/// revision describe the same accepted model as the returned numerical state.
enum class PhysicsStateFields : uint32_t
{
    None = 0,
    Position = 1u << 0,
    CenterOfMass = 1u << 1,
    LinearVelocity = 1u << 2,
    AngularVelocity = 1u << 3,
    InverseMass = 1u << 4,
    Rotation = 1u << 5,
    InverseInertia = 1u << 6,
    BodyId = 1u << 7,
    ShapeRevision = 1u << 8,
    All = (1u << 7) - 1,
};

constexpr bool HasPhysicsStateField(PhysicsStateFields fields, PhysicsStateFields field)
{
    return (static_cast<uint32_t>(fields) & static_cast<uint32_t>(field)) != 0;
}

/// One coherent solver sample. Position is the body origin, while velocity
/// is measured at its center of mass. Response tensors include world-axis locks;
/// inertia is the unprojected physical tensor of an authored dynamic body.
struct PhysicsBodyMotionState
{
    uint32_t bodyId = 0xFFFFFFFF;
    uint64_t shapeRevision = 0;
    glm::vec3 position{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 centerOfMass{0.0f};
    glm::vec3 linearVelocity{0.0f};
    glm::vec3 angularVelocity{0.0f};
    glm::vec3 inverseMass{0.0f};
    glm::mat3 inverseInertia{0.0f};
    glm::mat3 inertia{0.0f};
};

struct PhysicsPenetrationResult
{
    glm::vec3 direction{0.0f}; ///< Unit direction that separates A from B.
    float distance = 0.0f;
    glm::vec3 pointA{0.0f};
    glm::vec3 pointB{0.0f};
};

} // namespace infernux
