#pragma once

#include <array>
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

inline constexpr std::array<uint32_t, 13> PhysicsDeviceQueryWidths{3, 4, 3, 3, 3, 3, 9, 1, 3, 3, 4, 1, 1};

struct PhysicsDeviceQueryOutput
{
    PhysicsDeviceQueryField field;
    std::shared_ptr<rhi::ComputeBuffer> buffer;
    uint64_t byteOffset = 0;
    uint32_t recordFirst = 0;
    uint32_t recordCount = UINT32_MAX;
    uint64_t byteEnd = UINT64_MAX; // Exclusive end of the caller's writable view.
};
} // namespace infernux
