// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

namespace infernux::physics::inxphygpu
{

inline constexpr uint32_t NoModelIndex = UINT32_MAX;

using ModelWords = std::array<uint32_t, 4>;
using ModelVector = std::array<float, 4>;

using ModelJointVectors = std::array<ModelVector, 2>;
using ModelSwingTwistControls = std::array<ModelWords, 4>;
using ModelSixDOFControls = std::array<ModelWords, 17>;

struct InxPhyGpuHingeDrive
{
    bool enabled = false;
    float target = 0.0f, stiffness = 0.0f, damping = 0.0f, maxTorque = 0.0f;
    float jointDamping = 0.0f, frictionTorque = 0.0f, armature = 0.0f;

    [[nodiscard]] ModelJointVectors Pack() const noexcept
    {
        return {
            {{target, stiffness, damping, enabled ? maxTorque : 0.0f}, {jointDamping, frictionTorque, armature, 0.0f}}};
    }
};

struct InxPhyGpuJointLimits
{
    bool enabled = false;
    float minimum = 0.0f, maximum = 0.0f;

    [[nodiscard]] ModelWords Pack() const noexcept;
};

struct InxPhyGpuDistanceSpring
{
    float stiffness = 0.0f, damping = 0.0f;
};

struct ModelPose
{
    ModelVector position{};
    ModelVector rotation{1.0f, 0.0f, 0.0f, 0.0f}; // wxyz
};

struct ModelRigidMotion
{
    static constexpr float Unlimited = std::numeric_limits<float>::max();
    float linearDamping = 0.0f, angularDamping = 0.0f;
    float maxLinearVelocity = Unlimited, maxAngularVelocity = Unlimited;

    [[nodiscard]] bool HasSettings() const noexcept
    {
        return linearDamping != 0.0f || angularDamping != 0.0f || maxLinearVelocity != Unlimited ||
               maxAngularVelocity != Unlimited;
    }
};

static_assert(sizeof(ModelRigidMotion) == sizeof(ModelVector));

enum class ModelJointType : uint32_t
{
    Free,
    Ball,
    Slide,
    Hinge
};

struct ModelJointDefinition
{
    ModelJointType type;
    std::array<float, 3> anchor{};
    std::array<float, 3> axis{0.0f, 0.0f, 1.0f};
    float reference = 0.0f;
    std::array<float, 6> armature{};
    bool gravityCompensatedByActuator = false;
};

struct ModelBodyDefinition
{
    uint32_t parent = NoModelIndex; // Source index, or the world.
    ModelPose pose;
    ModelPose inertiaPose;
    std::array<float, 3> principalInertia{};
    float mass = 0.0f;
    float gravityCompensation = 0.0f;
    bool kinematic = false;
    std::vector<ModelJointDefinition> joints;
    bool editableStatic = false;
};

struct ModelGeometryDefinition
{
    uint32_t body = NoModelIndex;
    uint32_t type = 0;
    uint32_t collisionType = 0;
    uint32_t collisionAffinity = 0;
    ModelPose pose;
};

// These records are the storage-buffer ABI in model.slang.
struct ModelDimensions
{
    ModelWords state{};
    ModelWords layout{};
};

struct ModelBody
{
    ModelWords hierarchy{};
    ModelWords subtree{};
    ModelPose pose;
    ModelPose inertiaPose;
    ModelVector inertiaMass{};
    ModelVector subtreeMass{};
    ModelWords dofs{};
};

struct ModelJoint
{
    ModelWords indices{};
    ModelVector anchor{};
    ModelVector axis{};
};

struct ModelGeometry
{
    ModelWords indices{};
    ModelPose pose;
};

struct ModelDof
{
    ModelWords indices{};
    float armature = 0.0f;
    uint32_t flags = 0;
    std::array<float, 2> reserved{};
};

struct ModelMassFactors
{
    ModelWords layout{};
    std::vector<ModelWords> smallBlocks;
    std::vector<ModelWords> tiledBlocks;
    std::vector<uint32_t> tileElements;
    uint32_t sparseBlockCount = 0;
    std::vector<uint32_t> sparseTopology;
};

struct InxPhyGpuModel
{
    ModelDimensions dimensions;
    std::vector<ModelBody> bodies;
    std::vector<ModelJoint> joints;
    std::vector<ModelGeometry> geometries;
    std::vector<ModelDof> dofs;
    std::vector<uint32_t> bodyRows; // Source index to device body row.
    std::vector<uint32_t> branches;
    std::vector<uint32_t> massAdjacency;
    ModelMassFactors massFactors;
    std::vector<float> initialPositions;
    std::vector<float> initialVelocities; // Empty initializes all velocity coordinates to zero.
    std::vector<ModelPose> initialKinematicPoses;
    // Rows mirror PrescribedMotion: linear/mode, angular, and COM/accepted step.
    std::vector<std::array<ModelVector, 3>> initialKinematicMotion;
    std::vector<ModelPose> initialStaticPoses;
    std::vector<ModelRigidMotion> sceneBodyMotion; // Shared by independent scene bodies in device row order.
    std::vector<ModelWords> sceneJoints;
    std::vector<ModelWords> initialSceneJointParameters; // Packed by joint kind, shared as the seed for each world.
};

/// Compile shared topology and initial coordinates. The source can name parents in any order.
/// A free joint occupies a root alone. A kinematic body is a jointless root.
[[nodiscard]] InxPhyGpuModel BuildInxPhyGpuModel(const std::vector<ModelBodyDefinition> &bodies,
                                                 const std::vector<ModelGeometryDefinition> &geometries);

/// Reference-pose factorization in the packed ancestor layout, including small islands.
[[nodiscard]] ModelMassFactors BuildInxPhyGpuReferenceMassFactors(const InxPhyGpuModel &model);

} // namespace infernux::physics::inxphygpu
