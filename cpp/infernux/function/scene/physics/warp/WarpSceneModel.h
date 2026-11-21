// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../PhysicsMass.h"
#include "../PhysicsShapeAuthoring.h"
#include "WarpCollisionModel.h"

#include <Jolt/Physics/Collision/Shape/Shape.h>

namespace infernux::physics::warp
{

struct WarpSceneCollider
{
    uint64_t componentId;
    uint64_t generation;
    uint32_t layer;
    uint32_t collisionMask;
    float friction;
    bool trigger;
};

struct WarpSceneActor
{
    uint64_t objectId = 0;
    JPH::RefConst<JPH::Shape> shape;
    glm::vec3 position{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 linearVelocity{0.0f};       // World velocity at the center of mass.
    glm::vec3 angularVelocity{0.0f};      // World axes.
    std::optional<PreparedBodyMass> mass; // Authored or geometry-derived dynamic tensor.
    bool kinematic = false;
    bool useGravity = true;
    std::vector<WarpSceneCollider> colliders;
};

struct WarpSceneCapture
{
    WarpSceneActor actor;
    PhysicsActorPublication publication;
};

struct WarpSceneBodyBinding
{
    uint64_t objectId;
    uint32_t bodyRow;
    uint32_t position;
    uint32_t velocity;
};

struct WarpSceneGeometryBinding
{
    WarpSceneCollider collider;
    uint32_t subShape;
};

struct WarpSceneModel
{
    WarpModel model;
    WarpCollisionModel collision;
    std::vector<WarpSceneBodyBinding> bodies;         // Author actor order.
    std::vector<WarpSceneGeometryBinding> geometries; // Device geometry order.
};

/// Capture on the owner thread. A pending shape cook leaves publication untouched.
[[nodiscard]] std::optional<WarpSceneCapture> CaptureWarpSceneActor(const PhysicsActorData &actor);

/// Translate one authored topology. The result owns no scene objects or Jolt shapes.
[[nodiscard]] WarpSceneModel BuildWarpSceneModel(const std::vector<WarpSceneActor> &actors);

} // namespace infernux::physics::warp
