#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include "PhysicsECSStore.h"
#include "PhysicsMass.h"

#include <optional>
#include <vector>

namespace infernux
{

/// Owner-thread capture of ordinary components for the native CPU solver.
/// Geometry, COM and inertia use the same authoring functions as GPU actors.
/// Initial velocities are consumed by the owner only after creation succeeds.
struct PreparedCpuActor
{
    JPH::BodyCreationSettings settings;
    std::optional<PreparedBodyMass> mass;
    std::vector<Collider *> members;
    PhysicsECSStore::RigidbodyHandle initialRigidbody;
    bool hasInitialLinearVelocity = false;
    bool hasInitialAngularVelocity = false;
};

/// Null means a complete collider cook is still pending. No body or backend
/// state is created, and no authored input is consumed at this boundary.
[[nodiscard]] std::optional<PreparedCpuActor> PrepareJoltCpuActor(const PhysicsActorData &actor);

} // namespace infernux
