#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include "PhysicsMass.h"
#include "PhysicsShapeAuthoring.h"
#include <function/resources/PhysicMaterial/PhysicMaterial.h>

#include <optional>
#include <vector>

namespace infernux
{

/// Owner-thread capture of ordinary components for the native CPU solver.
/// Geometry, COM and inertia use the same authoring functions as GPU actors.
/// Initial velocities are consumed by the owner only after creation succeeds.
struct PreparedCpuActor
{
    struct Surface
    {
        PhysicsTargetReference target;
        uint64_t colliderId, generation;
        uint32_t layer, collisionMask;
        float friction, restitution;
        PhysicsMaterialCombine frictionCombine, bounceCombine;
        bool sensor;
    };
    JPH::BodyCreationSettings settings;
    std::optional<PreparedBodyMass> mass;
    std::vector<Surface> surfaces;
    int collisionDetectionMode = 0;
    bool frozen = false;
    bool authoredDynamic = false;
    PhysicsTargetReference bodyTarget;
    PhysicsActorPublication publication;
};

/// Null means a complete collider cook is still pending. No body or backend
/// state is created, and no authored input is consumed at this boundary.
[[nodiscard]] std::optional<PreparedCpuActor> PrepareJoltCpuActor(const PhysicsActorData &actor);

} // namespace infernux
