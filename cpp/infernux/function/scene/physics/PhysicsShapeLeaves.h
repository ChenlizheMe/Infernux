#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Collision/TransformedShape.h>
#include <vector>

namespace infernux
{

struct PhysicsShapeLeaf
{
    JPH::TransformedShape shape; // In root center-of-mass space.
    uint32_t member;
};

/// Flatten already-scaled authoring geometry in Collider member and subshape order.
[[nodiscard]] std::vector<PhysicsShapeLeaf> CollectPhysicsShapeLeaves(const JPH::Shape &shape, uint32_t memberCount);

} // namespace infernux
