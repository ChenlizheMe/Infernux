#pragma once

#include <Jolt/GPU/SleepTypes.h>
#include <Jolt/Physics/Collision/Shape/Shape.h>
#include "JoltGpuShape.h"
#include "PhysicsShapeAuthoring.h"
#include "PhysicsTargetReference.h"

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
    JPH::GPU::BodySleepAuthor sleep;
    PreparedGpuShape geometry;
    JPH::GPU::BodyState initialState;
    JPH::GPU::Float4 rootScale{1, 1, 1, 0};
    PhysicsActorPublication publication;
    JPH::RefConst<JPH::Shape> authoredShape;
    struct Member
    {
        PhysicsTargetReference target;
        uint64_t colliderId, generation, gameObjectId;
    };
    // Ordinals come from the authored outer compound.
    std::vector<Member> members;

    void SetOriginPose(const glm::vec3 &position, const glm::quat &rotation);
};

/// Capture ordinary component authoring without creating or reading a CPU Body.
/// Null means the complete shape candidate is not ready. Unsupported authored
/// capabilities throw at this boundary. Pending velocities are observed without
/// consumption; the owner consumes them when that Rigidbody binding's model
/// packet is accepted, retaining any following velocity commands in FIFO order.
[[nodiscard]] std::optional<PreparedGpuActor> PrepareJoltGpuActor(const PhysicsActorData &actor, uint32_t bodySlot,
                                                                  uint32_t worldIndex);

/// Refresh only material/filter fields, preserving mass, motion and GPU state.
/// Collider is the actual contributing leaf source. The Scene captures its
/// full component identity separately; no primary surface stands for a body.
void RefreshGpuActorSurface(JPH::GPU::ShapeModel &leaf, const Collider &collider);

} // namespace infernux
