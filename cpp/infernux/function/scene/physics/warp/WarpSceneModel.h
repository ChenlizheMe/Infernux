// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../PhysicsMass.h"
#include "../PhysicsShapeAuthoring.h"
#include "WarpCollisionModel.h"

#include <Jolt/Physics/Collision/Shape/Shape.h>
#include <function/resources/PhysicMaterial/PhysicMaterial.h>

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
    float bounciness = 0.0f;
    PhysicsMaterialCombine frictionCombine = PhysicsMaterialCombine::Average;
    PhysicsMaterialCombine bounceCombine = PhysicsMaterialCombine::Average;
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

struct WarpSceneSurfaceIdentity
{
    PhysicsTargetReference target;
    uint64_t gameObjectId;
    uint64_t colliderId;
};

struct WarpSceneCapture
{
    WarpSceneActor actor;
    PhysicsActorPublication publication;
    std::vector<WarpSceneSurfaceIdentity> surfaces;
};

struct WarpSceneBodyBinding
{
    uint64_t objectId;
    uint32_t bodyRow;
    uint32_t position;
    uint32_t velocity;
    uint32_t surfaceCount;
};

struct WarpSceneGeometryBinding
{
    WarpSceneCollider collider;
    uint32_t subShape;
    uint32_t surface;
};

struct WarpSceneHinge
{
    glm::vec3 axis{0.0f, 1.0f, 0.0f};
    bool useLimits = false;
    float minimum = 0.0f, maximum = 0.0f;
    WarpHingeDrive drive;
};

struct WarpSceneJoint
{
    uint32_t ownerActor;
    uint32_t connectedActor = NoModelIndex;
    glm::vec3 ownerAnchor{0.0f}; // Baked offset from the owner transform origin, in body axes.
    bool enableCollision = false;
    std::optional<WarpSceneHinge> hinge;
};

struct WarpSceneModel
{
    WarpModel model;
    WarpCollisionModel collision;
    std::vector<WarpSceneBodyBinding> bodies;         // Author actor order.
    std::vector<WarpSceneGeometryBinding> geometries; // Device geometry order.
    std::vector<uint32_t> hingeRows; // Author joint order, NoModelIndex for nonmoving hinges or other joints.
};

[[nodiscard]] WarpSceneCapture PrepareWarpSceneActor(PhysicsActorAuthoring author);

/// Translate one authored topology. The result owns no scene objects or Jolt shapes.
[[nodiscard]] WarpSceneModel BuildWarpSceneModel(const std::vector<WarpSceneActor> &actors,
                                                 const std::vector<WarpSceneJoint> &joints = {});

} // namespace infernux::physics::warp
