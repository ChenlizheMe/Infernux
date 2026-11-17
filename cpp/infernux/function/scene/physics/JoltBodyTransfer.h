#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/Math/Quat.h>
#include <Jolt/Math/Real.h>
#include "PhysicsTargetReference.h"

#include <cstdint>
#include <memory>
#include <optional>

namespace infernux
{

// One body at an accepted control boundary. Position is the body origin;
// linear velocity and accumulated torque are relative to its center of mass.
struct JoltBodyTransfer
{
    struct Drive
    {
        JPH::RVec3 position = JPH::RVec3::sZero();
        JPH::Quat rotation = JPH::Quat::sIdentity();
        float duration = 0.0f;
        float maxSpeed = 0.0f;
        PhysicsTargetSource positionSource, rotationSource;
        bool pending = false;
        bool ownsVelocity = false;
        bool temporaryStatic = false;
    };

    JPH::RVec3 position;
    JPH::Quat rotation;
    JPH::Vec3 linearVelocity, angularVelocity;
    JPH::Vec3 force, torque;
    std::optional<Drive> drive;
    uint64_t shapeRevision = 0;
    bool resident = false;
    bool awake = false;
};

struct JoltBodyImport
{
    uint32_t id;
    std::shared_ptr<std::optional<JoltBodyTransfer>> state;
};

} // namespace infernux
