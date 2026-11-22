#pragma once

#include <Jolt/Physics/Collision/Shape/Shape.h>
#include "PhysicsECSStore.h"
#include "PhysicsMass.h"
#include "PhysicsTargetReference.h"
#include <function/resources/PhysicMaterial/PhysicMaterial.h>
#include <memory>
#include <optional>
#include <vector>

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
    // Static replicas of the same authored shape publish one collider revision.
    std::optional<uint64_t> sharedGeometryRevision;
    std::optional<glm::vec3> linearVelocity, angularVelocity;

    void AcceptGeometry(PhysicsActorHandle actor, uint32_t bodyId, uint64_t shapeRevision) const noexcept;
    /// The backend calls this only after it accepts the captured initial velocities.
    void ConsumeInitialVelocities(PhysicsActorHandle actor, uint32_t bodyId) const noexcept;

  private:
    friend struct PhysicsActorAuthoring;

    [[nodiscard]] static PhysicsActorPublication Capture(const PhysicsActorData &,
                                                         const std::vector<Collider *> &members, const Rigidbody *);
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

struct PhysicsSurfaceAuthoring
{
    PhysicsTargetReference target;
    uint64_t colliderId, generation;
    uint32_t layer, collisionMask;
    float friction, restitution;
    PhysicsMaterialCombine frictionCombine, bounceCombine;
    bool sensor;
    uint64_t gameObjectId = 0;
};

struct PhysicsRigidbodyAuthoring
{
    PhysicsTargetReference target;
    std::optional<PreparedBodyMass> mass;
    int constraints, collisionDetectionMode;
    float drag, angularDrag, maxLinearVelocity, maxAngularVelocity;
    bool kinematic;
    float gravityFactor;
};

/// Backend-independent component and shape input belonging to one author pose.
/// The owner publishes the captured geometry and consumes staged velocities
/// only after the selected backend accepts them.
struct PhysicsActorAuthoring
{
    JPH::RefConst<JPH::Shape> shape;
    glm::vec3 position, scale;
    glm::quat rotation;
    uint64_t objectId;
    int objectLayer;
    std::vector<PhysicsSurfaceAuthoring> surfaces;
    std::optional<PhysicsRigidbodyAuthoring> body;
    PhysicsActorPublication publication;

    /// A pending child cook leaves the entire previous actor revision in use.
    [[nodiscard]] static std::optional<PhysicsActorAuthoring> Capture(const PhysicsActorData &actor);
};

/// Owner-thread authoring: capture all active collider geometry as one shape.
/// Null means the whole candidate is not ready; pending children still begin
/// cooking. Returned children already contain world scale, so relative actor
/// transforms contribute only rigid translation and rotation.
JPH::RefConst<JPH::Shape> BuildActorShape(const PhysicsActorData &actor);
} // namespace infernux
