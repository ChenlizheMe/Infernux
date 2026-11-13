#pragma once

#include <Jolt/Physics/Collision/Shape/Shape.h>
#include "JoltGpuShape.h"
#include "PhysicsECSStore.h"

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
    JPH::GPU::Float4 rootScale{1, 1, 1, 0};
    // Creation inputs belong to this exact component generation, even when a
    // different Rigidbody is attached to the actor before GPU acceptance.
    PhysicsECSStore::RigidbodyHandle initialRigidbody;
    /// Full component identity captured with initialRigidbody; zero without
    /// an enabled Rigidbody contributing to this authored model.
    uint64_t rigidbodyComponentId = 0;
    bool hasInitialLinearVelocity = false;
    bool hasInitialAngularVelocity = false;
    struct ColliderCenter
    {
        PhysicsECSStore::ColliderHandle collider;
        glm::vec3 center;
    };
    std::vector<ColliderCenter> colliderCenters;
    JPH::RefConst<JPH::Shape> authoredShape;
    /// Exact contributing members, captured immediately by Scene into safe
    /// per-leaf identities. Ordinals come from the authored outer compound.
    std::vector<Collider *> members;
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
