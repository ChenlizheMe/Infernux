#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/Math/Quat.h>

#include <cstdint>
#include <memory>
#include <optional>

namespace infernux
{

/// Accepted constraint coordinates, independent of solver rows and warm starts.
/// The anchors use the body origin and its authored scale. For a world
/// endpoint the anchor is a world position and the frame a world rotation.
struct JoltJointTransfer
{
    struct Endpoint
    {
        JPH::Vec3 unscaled = JPH::Vec3::sZero();
        JPH::Vec3 collapsed = JPH::Vec3::sZero();
        JPH::Quat frame = JPH::Quat::sIdentity();
    };
    Endpoint connected, owner;
    std::optional<JPH::Quat> poseDrive;
    float automaticDistance = 0.0f;
    double rotorSpeed = 0.0;
};

struct JoltJointImport
{
    uint64_t id;
    // A joint moved before its first solve still needs its initial binding.
    std::shared_ptr<std::optional<JoltJointTransfer>> state;
};

} // namespace infernux
