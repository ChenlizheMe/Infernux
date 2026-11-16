#pragma once

#include "PhysicsECSStore.h"
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace JPH
{
class Shape;
template <class T> class RefConst;
} // namespace JPH

namespace infernux
{
class Collider;
class Rigidbody;
struct CookedGeometry;
struct PhysicsActorData;

/// Author metadata retained until a complete shape and its model are accepted.
struct PhysicsActorPublication
{
    PhysicsECSStore::RigidbodyHandle rigidbody;
    uint64_t rigidbodyComponentId = 0;
    std::optional<glm::vec3> linearVelocity, angularVelocity;

    /// Capture inside the author's pose scope, after all contributing shapes are ready.
    [[nodiscard]] static PhysicsActorPublication Capture(const PhysicsActorData &,
                                                         const std::vector<Collider *> &members, const Rigidbody *);
    void AcceptGeometry(PhysicsActorHandle actor, uint32_t bodyId, uint64_t shapeRevision) const noexcept;
    /// The backend calls this only after it accepts the captured initial velocities.
    void ConsumeInitialVelocities(PhysicsActorHandle actor, uint32_t bodyId) const noexcept;

  private:
    struct ColliderCenter
    {
        PhysicsECSStore::ColliderHandle collider;
        glm::vec3 center;
    };
    struct MeshPublication
    {
        PhysicsECSStore::ColliderHandle collider;
        std::shared_ptr<const CookedGeometry> geometry;
    };
    std::vector<ColliderCenter> colliderCenters;
    std::vector<MeshPublication> meshPublications;
};

/// Owner-thread authoring: capture all active collider geometry as one shape.
/// Null means the whole candidate is not ready; pending children still begin
/// cooking. Returned children already contain world scale, so relative actor
/// transforms contribute only rigid translation and rotation.
JPH::RefConst<JPH::Shape> BuildActorShape(const PhysicsActorData &actor, const Collider *exclude = nullptr,
                                          size_t *outShapeCount = nullptr,
                                          std::vector<Collider *> *outMembers = nullptr);
} // namespace infernux
