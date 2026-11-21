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

/// Prepared authoring for the native CPU solver.
/// Geometry, COM and inertia use the same authoring functions as GPU actors.
/// Initial velocities are consumed by the owner only after creation succeeds.
struct PreparedCpuActor
{
    using Surface = PhysicsSurfaceAuthoring;
    JPH::BodyCreationSettings settings;
    JPH::Vec3 rootScale = JPH::Vec3::sOne();
    std::optional<PreparedBodyMass> mass;
    std::vector<Surface> surfaces;
    int collisionDetectionMode = 0;
    bool frozen = false;
    bool authoredDynamic = false;
    PhysicsTargetReference bodyTarget;
    PhysicsActorPublication publication;
};

[[nodiscard]] PreparedCpuActor PrepareJoltCpuActor(PhysicsActorAuthoring actor);

} // namespace infernux
