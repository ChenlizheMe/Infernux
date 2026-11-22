#pragma once

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <optional>
#include <vector>

namespace JPH
{
class Shape;
}

namespace infernux
{
class Collider;

enum class PhysicsShapeType : uint32_t
{
    Box,
    Sphere,
    Capsule,
    Mesh
};

/// Immutable baked leaf geometry. The published pose rotates about its center of mass.
struct PhysicsShapeGeometry
{
    uint64_t identity = 0;
    PhysicsShapeType type = PhysicsShapeType::Mesh;
    glm::vec4 dimensions{0.0f};
    glm::vec3 boundsMin{0.0f};
    glm::vec3 boundsMax{0.0f};
    bool convex = false;

    [[nodiscard]] static std::shared_ptr<const PhysicsShapeGeometry> DescribeLeaf(const JPH::Shape &shape);
    [[nodiscard]] std::vector<glm::vec3> GetTriangles() const;

  private:
    friend class JoltCpuQueries;
    std::shared_ptr<const JPH::Shape> shape;
    std::optional<std::array<glm::vec3, 3>> triangle;
};

struct PhysicsShapeState
{
    Collider *collider = nullptr;
    uint32_t memberIndex = 0;
    uint32_t layerBit = 0;
    bool trigger = false;
    bool moving = false;
    std::shared_ptr<const PhysicsShapeGeometry> geometry;
    glm::vec3 center{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 bodyCenterOfMass{0.0f};
    glm::vec3 linearVelocity{0.0f};
    glm::vec3 angularVelocity{0.0f};
    float friction = 0.0f;
    float bounciness = 0.0f;
    uint32_t frictionCombine = 0;
    uint32_t bounceCombine = 0;
};
} // namespace infernux
