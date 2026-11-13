#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/GPU/PhysicsTypes.h>
#include <Jolt/Math/Vec3.h>
#include <Jolt/Physics/Collision/Shape/SubShapeID.h>

#include <cstdint>
#include <vector>

namespace JPH
{
class Shape;
}

namespace infernux
{

/// One primitive in the root COM frame, retaining its actual compound path.
struct PreparedGpuLeaf
{
    JPH::GPU::ShapeModel model;
    JPH::SubShapeID subShapeID;
    uint32_t subShapeBits = 0;
    uint32_t memberIndex = 0;
};

/// Complete authored leaf stream. An empty body has no collidable dummy leaf.
struct PreparedGpuShape
{
    std::vector<PreparedGpuLeaf> leaves;
    /// Root COM in the authored rigid frame; component scale is already baked.
    /// Initial world COM = actor position + actor rotation * centerOfMass.
    JPH::Vec3 centerOfMass;
};

/// Resolve decorators and compound child user data through the authored root.
/// Sphere, Box and Capsule leaves use baked unit scale; unsupported geometry
/// throws before any model is staged.
/// memberCount is the actual contributing Collider count, not a leaf count.
[[nodiscard]] PreparedGpuShape PrepareJoltGpuShape(const JPH::Shape &shape, uint32_t memberCount);

} // namespace infernux
