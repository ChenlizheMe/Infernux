#pragma once

#include <Jolt/Physics/Collision/Shape/Shape.h>
#include "JoltGpuShape.h"

#include <optional>

namespace infernux
{

class Collider;
class GameObject;
struct PhysicsActorData;

/// Owner-thread creation data. Only initialState is an initial condition;
/// subsequent authored model changes must retain the resident GPU state.
struct PreparedGpuActor
{
    JPH::GPU::BodyModel body;
    PreparedGpuShape geometry;
    JPH::GPU::BodyState initialState;
    JPH::RefConst<JPH::Shape> authoredShape;
    /// Source of the single collision leaf's material and layer, null for Empty.
    /// The owner retains its normal component identity/lifetime separately.
    Collider *surfaceCollider = nullptr;
};

/// Capture ordinary component authoring without creating or reading a CPU Body.
/// Null means the complete shape candidate is not ready. Unsupported authored
/// capabilities throw at this boundary. Pending velocities are observed without
/// consumption; the owner consumes them once creation has been accepted.
[[nodiscard]] std::optional<PreparedGpuActor> PrepareJoltGpuActor(const PhysicsActorData &actor, uint32_t shapeSlot,
                                                                  uint32_t worldIndex);

/// Refresh only material/filter fields, preserving mass, motion and GPU state.
/// Collider must be the actual leaf source retained at creation; null is Empty.
void RefreshGpuActorSurface(JPH::GPU::BodyModel &body, const Collider *collider, const GameObject &owner);

} // namespace infernux
