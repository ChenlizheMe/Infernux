#pragma once

#include <cstdint>
#include <memory>

namespace infernux
{
namespace rhi
{
class ComputeBuffer;
}

enum class PhysicsDeviceQueryField : uint32_t
{
    MotionPosition,
    MotionRotation,
    MotionCenterOfMass,
    MotionLinearVelocity,
    MotionAngularVelocity,
    MotionInverseMass,
    MotionInverseInertia,
    ShapeBodyIndex,
    ShapeCenter,
    ShapeHalfExtents,
    ShapeRotation,
    ShapeFriction,
    ShapeBounciness,
};

struct PhysicsDeviceQueryOutput
{
    PhysicsDeviceQueryField field;
    std::shared_ptr<rhi::ComputeBuffer> buffer;
    uint64_t byteOffset = 0;
    uint32_t recordFirst = 0;
    uint32_t recordCount = UINT32_MAX;
    uint32_t bodyIndex = UINT32_MAX; // Aggregate input row for shape body-index projection.
    uint64_t byteEnd = UINT64_MAX;   // Exclusive end of the caller's writable view.
};
} // namespace infernux
