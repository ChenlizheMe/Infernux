// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../JointSettings.h"
#include "../PhysicsContactTypes.h"
#include "../PhysicsMass.h"
#include "../PhysicsShapeAuthoring.h"
#include "InxPhyGpuCollisionModel.h"

#include <Jolt/Physics/Collision/Shape/Shape.h>
#include <function/resources/PhysicMaterial/PhysicMaterial.h>
#include <variant>

namespace infernux::physics::inxphygpu
{

struct InxPhyGpuSceneCollider
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

struct InxPhyGpuSceneActor
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
    std::vector<InxPhyGpuSceneCollider> colliders;
};

struct InxPhyGpuSceneSurfaceIdentity
{
    PhysicsTargetReference target;
    uint64_t gameObjectId;
    uint64_t colliderId;
    uint64_t generation = 0;
};

struct InxPhyGpuSceneCapture
{
    InxPhyGpuSceneActor actor;
    PhysicsActorPublication publication;
    std::vector<InxPhyGpuSceneSurfaceIdentity> surfaces;
};

struct InxPhyGpuSceneBodyBinding
{
    uint64_t objectId;
    uint32_t bodyRow;
    uint32_t position;
    uint32_t velocity;
    uint32_t surfaceCount;
};

struct InxPhyGpuSceneGeometryBinding
{
    InxPhyGpuSceneCollider collider;
    uint32_t subShape;
    uint32_t surface;
};

struct InxPhyGpuSceneHinge
{
    glm::vec3 axis{0.0f, 1.0f, 0.0f};
    bool useLimits = false;
    float minimum = 0.0f, maximum = 0.0f;
    InxPhyGpuHingeDrive drive;
};

struct InxPhyGpuSceneSlider
{
    glm::vec3 axis{0.0f, 1.0f, 0.0f};
    bool useLimits = false;
    float minimum = 0.0f, maximum = 0.0f;
};

struct InxPhyGpuSceneDistance
{
    glm::vec3 connectedAnchor{0.0f}; // Baked local offset, or world position without a connected body.
    bool autoDistance = true;
    float minimum = 0.0f, maximum = 0.0f;
    float stiffness = 0.0f, damping = 0.0f;
};

struct InxPhyGpuSwingTwistTargets
{
    glm::vec4 orientation{0, 0, 0, 1}; // Authored XYZW, in constraint space.
    glm::vec3 angularVelocity{0.0f};   // Degrees/s, in owner constraint space.

    [[nodiscard]] ModelJointVectors Pack() const;
};

struct InxPhyGpuSwingTwistParameters
{
    ModelSwingTwistControls controls;
    InxPhyGpuSwingTwistTargets targets;

    explicit InxPhyGpuSwingTwistParameters(const SwingTwistJointSettings &settings);
};

struct InxPhyGpuSceneJoint
{
    uint32_t ownerActor;
    uint32_t connectedActor = NoModelIndex;
    glm::vec3 ownerAnchor{0.0f}; // Baked offset from the owner transform origin, in body axes.
    bool enableCollision = false;
    // This order matches the scene joint kinds in spatial_constraints.slang.
    std::variant<std::monostate, InxPhyGpuSceneHinge, InxPhyGpuSceneSlider, InxPhyGpuSceneDistance,
                 SwingTwistJointSettings>
        settings;
};

enum class InxPhyGpuSceneJointKind : uint32_t
{
    Fixed,
    Hinge,
    Slider,
    Distance,
    SwingTwist,
};

struct InxPhyGpuSceneJointBinding
{
    uint32_t row = NoModelIndex;
    InxPhyGpuSceneJointKind kind = InxPhyGpuSceneJointKind::Fixed;
    bool autoDistance = false;
    uint32_t parameterOffset = 0; // In ModelWords records within one world.
};

struct InxPhyGpuSceneModel
{
    InxPhyGpuModel model;
    InxPhyGpuCollisionModel collision;
    std::vector<InxPhyGpuSceneBodyBinding> bodies;         // Author actor order.
    std::vector<InxPhyGpuSceneGeometryBinding> geometries; // Device geometry order.
    std::vector<InxPhyGpuSceneJointBinding> jointBindings; // Author joint order.
};

[[nodiscard]] InxPhyGpuSceneCapture PrepareInxPhyGpuSceneActor(PhysicsActorAuthoring author);

/// Translate one authored topology. The result owns no scene objects or Jolt shapes.
[[nodiscard]] InxPhyGpuSceneModel BuildInxPhyGpuSceneModel(const std::vector<InxPhyGpuSceneActor> &actors,
                                                           const std::vector<InxPhyGpuSceneJoint> &joints = {},
                                                           const PhysicsColliderPairs &ignoredColliders = {});

} // namespace infernux::physics::inxphygpu
