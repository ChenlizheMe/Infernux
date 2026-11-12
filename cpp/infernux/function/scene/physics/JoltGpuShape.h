#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/GPU/PhysicsTypes.h>
#include <Jolt/Math/Vec3.h>
#include <Jolt/Physics/Collision/Shape/SubShapeID.h>

#include <cstdint>

namespace JPH
{
class Shape;
}

namespace infernux
{

/// One GPU primitive prepared from immutable authored Jolt geometry.
struct PreparedGpuShape
{
    JPH::GPU::ShapeModel model;
    uint32_t type;
    /// Root COM in the authored rigid frame; component scale is already baked.
    /// Initial world COM = actor position + actor rotation * centerOfMass.
    JPH::Vec3 centerOfMass;
    /// Path to the single leaf in the supplied root shape.
    JPH::SubShapeID subShapeID;
};

/// Resolve decorators through Jolt's leaf collector. The GPU bootstrap accepts
/// exactly one Sphere or Box with unit leaf scale; unsupported geometry throws
/// std::invalid_argument. Pending/null geometry remains the owner's concern.
[[nodiscard]] PreparedGpuShape PrepareJoltGpuShape(const JPH::Shape &shape);

} // namespace infernux
