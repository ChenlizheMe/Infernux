// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "ConstraintTypes.h"

namespace JPH
{
namespace GPU
{

// A stage recipe only: the integrating engine owns compilation, resources,
// barriers, admission, submission, completion and publication.
enum class Resource : std::uint32_t
{
    StepParams = 1u << 0,
    BodyModels = 1u << 1,
    ShapeModels = 1u << 2,
    Worlds = 1u << 3,
    BodyForces = 1u << 4,
    BodyStates = 1u << 5,
    Bounds = 1u << 6,
    Contacts = 1u << 7,
    KinematicTargets = 1u << 8,
    ContactPatches = 1u << 9,
    Joints = 1u << 10,
    SleepStates = 1u << 11,
    StageWork = 1u << 12,
    ConstraintSchedule = 1u << 13,
    ConvexGeometry = 1u << 14,
    ArticulationControl = 1u << 15,
    ContactHistory = 1u << 16,
    SweepBodies = 1u << 17,
};

constexpr std::uint32_t Mask(Resource inResource)
{
    return static_cast<std::uint32_t>(inResource);
}

enum class Stage : std::uint32_t
{
    ApplyForces,
    BuildBounds,
    CollidePairs,
    SetupContacts,
    SolveVelocity,
    Integrate,
    SolvePosition,
    PrepareKinematics,
    SetupJoints,
    FinishJoints,
    WarmJoints,
    SolveJointVelocity,
    SolveJointPosition,
    WarmContacts,
    StoreContactHistory,
    CaptureSweepMotion,
    BuildSweptBounds,
    IntegrateSwept,
    SolveVelocityRounds,
    SolvePositionRounds,
};

enum class Domain : std::uint32_t
{
    Bodies,
    Shapes,
    Patches,
    Contacts,
    IslandPatches,
    IslandJoints,
    Joints
};

constexpr bool IsContactRoundStage(Stage inStage)
{
    return inStage == Stage::SolveVelocityRounds || inStage == Stage::SolvePositionRounds;
}

// Total rectangular items per dispatch, independent of island/world capacity.
// A unit must satisfy laneCount * localCount <= this quantum, including padding.
// One lane alone may consume the entire quantum. Domain lanes are not capped
// by a world-domain lane limit. Host validation uses checked arithmetic.
constexpr std::uint32_t WorkItemLimit(Domain inDomain)
{
    switch (inDomain) {
    case Domain::IslandJoints:
        return IslandJointQuantum;
    case Domain::IslandPatches:
        return IslandPatchQuantum;
    default:
        return 2048;
    }
}

struct StageDefinition
{
    Stage stage;
    const char *entry; // Host metadata; never part of a device buffer.
    Domain domain;
    std::uint32_t groupSizeX, groupSizeY, groupSizeZ;
    std::uint32_t reads, writes;
    const char *directEntry = nullptr; // Same island stage and resources, one independent range per lane.
};

inline constexpr const char *StepModuleName = "jph_gpu_rigid_step";
inline constexpr const char *StepSourcePath = "gpu/Shaders/rigid_step.slang";

inline constexpr auto ContactVelocityReads =
    Mask(Resource::StageWork) | Mask(Resource::StepParams) | Mask(Resource::ConstraintSchedule) |
    Mask(Resource::BodyModels) | Mask(Resource::BodyStates) | Mask(Resource::Contacts) |
    Mask(Resource::KinematicTargets) | Mask(Resource::ContactPatches) | Mask(Resource::SleepStates);
inline constexpr auto ContactVelocityWrites = Mask(Resource::BodyStates) | Mask(Resource::Contacts);
inline constexpr auto ContactPositionReads =
    Mask(Resource::StageWork) | Mask(Resource::StepParams) | Mask(Resource::ConstraintSchedule) |
    Mask(Resource::BodyModels) | Mask(Resource::Contacts) | Mask(Resource::BodyStates) | Mask(Resource::SleepStates);

inline constexpr StageDefinition Stages[] = {
    {Stage::ApplyForces, "apply_forces", Domain::Bodies, 64, 1, 1,
     Mask(Resource::StageWork) | Mask(Resource::StepParams) | Mask(Resource::BodyModels) | Mask(Resource::BodyForces) |
         Mask(Resource::BodyStates) | Mask(Resource::SleepStates),
     Mask(Resource::BodyStates)},
    {Stage::BuildBounds, "build_bounds", Domain::Shapes, 64, 1, 1,
     Mask(Resource::StageWork) | Mask(Resource::StepParams) | Mask(Resource::ShapeModels) | Mask(Resource::BodyStates) |
         Mask(Resource::ConvexGeometry),
     Mask(Resource::Bounds)},
    {Stage::CollidePairs, "narrowphase_candidates", Domain::Patches, 64, 1, 1,
     Mask(Resource::StageWork) | Mask(Resource::StepParams) | Mask(Resource::ShapeModels) | Mask(Resource::BodyStates) |
         Mask(Resource::ContactPatches),
     Mask(Resource::Contacts) | Mask(Resource::ContactPatches)},
    {Stage::SetupContacts, "setup_contacts", Domain::Contacts, 64, 1, 1,
     Mask(Resource::StageWork) | Mask(Resource::StepParams) | Mask(Resource::BodyModels) | Mask(Resource::BodyForces) |
         Mask(Resource::BodyStates) | Mask(Resource::Contacts) | Mask(Resource::KinematicTargets) |
         Mask(Resource::ShapeModels) | Mask(Resource::ContactPatches) | Mask(Resource::SleepStates),
     Mask(Resource::Contacts)},
    {Stage::SolveVelocity, "solve_velocity", Domain::IslandPatches, ContactSolverLanes, 1, 1, ContactVelocityReads,
     ContactVelocityWrites, "solve_velocity_direct"},
    {Stage::Integrate, "integrate", Domain::Bodies, 64, 1, 1,
     Mask(Resource::ArticulationControl) | Mask(Resource::StageWork) | Mask(Resource::StepParams) |
         Mask(Resource::BodyModels) | Mask(Resource::BodyStates) | Mask(Resource::KinematicTargets) |
         Mask(Resource::SleepStates),
     Mask(Resource::BodyStates)},
    {Stage::SolvePosition, "solve_position", Domain::IslandPatches, ContactSolverLanes, 1, 1, ContactPositionReads,
     Mask(Resource::BodyStates), "solve_position_direct"},
    {Stage::PrepareKinematics, "prepare_kinematics", Domain::Bodies, 64, 1, 1,
     Mask(Resource::StageWork) | Mask(Resource::StepParams) | Mask(Resource::BodyModels) | Mask(Resource::BodyStates) |
         Mask(Resource::KinematicTargets) | Mask(Resource::SleepStates),
     Mask(Resource::BodyStates) | Mask(Resource::KinematicTargets) | Mask(Resource::SleepStates)},
    {Stage::SetupJoints, "setup_joints", Domain::Joints, 64, 1, 1,
     Mask(Resource::StageWork) | Mask(Resource::StepParams) | Mask(Resource::BodyModels) | Mask(Resource::BodyStates) |
         Mask(Resource::KinematicTargets) | Mask(Resource::Joints) | Mask(Resource::SleepStates),
     Mask(Resource::Joints)},
    {Stage::FinishJoints, "finish_joints", Domain::Joints, 64, 1, 1,
     Mask(Resource::StageWork) | Mask(Resource::StepParams) | Mask(Resource::Joints) | Mask(Resource::SleepStates) |
         Mask(Resource::BodyModels),
     Mask(Resource::Joints)},
    {Stage::WarmJoints, "warm_joints", Domain::IslandJoints, 64, 1, 1,
     Mask(Resource::ArticulationControl) | Mask(Resource::StageWork) | Mask(Resource::StepParams) |
         Mask(Resource::ConstraintSchedule) | Mask(Resource::BodyModels) | Mask(Resource::BodyStates) |
         Mask(Resource::KinematicTargets) | Mask(Resource::Joints) | Mask(Resource::SleepStates),
     Mask(Resource::BodyStates) | Mask(Resource::Joints)},
    {Stage::SolveJointVelocity, "solve_joint_velocity", Domain::IslandJoints, 64, 1, 1,
     Mask(Resource::ArticulationControl) | Mask(Resource::StageWork) | Mask(Resource::StepParams) |
         Mask(Resource::ConstraintSchedule) | Mask(Resource::BodyModels) | Mask(Resource::BodyStates) |
         Mask(Resource::KinematicTargets) | Mask(Resource::Joints) | Mask(Resource::SleepStates),
     Mask(Resource::BodyStates) | Mask(Resource::Joints)},
    {Stage::SolveJointPosition, "solve_joint_position", Domain::IslandJoints, 64, 1, 1,
     Mask(Resource::ArticulationControl) | Mask(Resource::StageWork) | Mask(Resource::StepParams) |
         Mask(Resource::ConstraintSchedule) | Mask(Resource::BodyModels) | Mask(Resource::BodyStates) |
         Mask(Resource::KinematicTargets) | Mask(Resource::Joints) | Mask(Resource::SleepStates),
     Mask(Resource::BodyStates) | Mask(Resource::Joints)},
    {Stage::WarmContacts, "warm_contacts", Domain::IslandPatches, ContactSolverLanes, 1, 1,
     Mask(Resource::StageWork) | Mask(Resource::StepParams) | Mask(Resource::ConstraintSchedule) |
         Mask(Resource::BodyModels) | Mask(Resource::BodyStates) | Mask(Resource::Contacts) |
         Mask(Resource::KinematicTargets) | Mask(Resource::ContactPatches) | Mask(Resource::SleepStates) |
         Mask(Resource::ContactHistory),
     Mask(Resource::BodyStates) | Mask(Resource::Contacts), "warm_contacts_direct"},
    {Stage::StoreContactHistory, "store_contact_history", Domain::Patches, 64, 1, 1,
     Mask(Resource::StageWork) | Mask(Resource::StepParams) | Mask(Resource::Contacts) |
         Mask(Resource::ContactPatches) | Mask(Resource::SleepStates) | Mask(Resource::BodyModels) |
         Mask(Resource::ConstraintSchedule),
     Mask(Resource::ContactHistory)},
    {Stage::CaptureSweepMotion, "capture_sweep_motion", Domain::Bodies, 64, 1, 1,
     Mask(Resource::StageWork) | Mask(Resource::StepParams) | Mask(Resource::BodyModels) | Mask(Resource::BodyStates) |
         Mask(Resource::KinematicTargets) | Mask(Resource::SleepStates),
     Mask(Resource::SweepBodies)},
    {Stage::BuildSweptBounds, "build_swept_bounds", Domain::Shapes, 64, 1, 1,
     Mask(Resource::StageWork) | Mask(Resource::StepParams) | Mask(Resource::ShapeModels) |
         Mask(Resource::SweepBodies) | Mask(Resource::ConvexGeometry),
     Mask(Resource::Bounds)},
    {Stage::IntegrateSwept, "integrate_swept", Domain::Bodies, 64, 1, 1,
     Mask(Resource::ArticulationControl) | Mask(Resource::StageWork) | Mask(Resource::StepParams) |
         Mask(Resource::BodyModels) | Mask(Resource::BodyStates) | Mask(Resource::KinematicTargets) |
         Mask(Resource::SleepStates) | Mask(Resource::SweepBodies),
     Mask(Resource::BodyStates) | Mask(Resource::SleepStates)},
    {Stage::SolveVelocityRounds, "solve_velocity_rounds", Domain::IslandPatches, ContactSolverLanes, 1, 1,
     ContactVelocityReads, ContactVelocityWrites},
    {Stage::SolvePositionRounds, "solve_position_rounds", Domain::IslandPatches, ContactSolverLanes, 1, 1,
     ContactPositionReads, Mask(Resource::BodyStates)},
};

// PrepareKinematics runs once before all substeps. One substep consists of
// ApplyForces, BuildBounds, the sparse collector, CollidePairs candidate
// narrowphase, SetupContacts, SetupJoints, WarmJoints, WarmContacts
// (only with nonzero velocity iterations), then SolveJointVelocity followed by
// SolveVelocity for each counts.w iteration, Integrate, SolveJointPosition then
// SolvePosition for each solve.x iteration, FinishJoints, StoreContactHistory. StageWork bounds each
// dispatch's items/island lanes; all units of an iteration precede its successor.
// Contact rounds can share the same admitted islands and row ordering when
// there are no joint or participant updates between their iterations.
// Joint-only worlds use joint stages without contact stages. The logical-tick
// owner retires final active-set forces before sleep Commit. Kinematic preparation
// stops the existing drive when the next tick has no new target. Control-only jobs
// leave the drive untouched. Sleeping composes engine-owned bounded
// control producers around this numerical recipe; contact publication and pair
// policy retain their existing producers. Bodies address zero or more cooked primitive/Hull
// leaves in their COM frame, including explicit
// COM offsets. Each collected canonical leaf pair owns four Contact records;
// sensors record real overlap but do not participate in response. The collector
// supplies actual candidate counts and runtime world ranges before narrowphase.
// ContactPatches is a word arena: word zero holds the final candidate-bank base,
// and patch loads add that base to candidate ordinals. Contact ordinals remain
// four times the candidate ordinal. Every numerical island lane exclusively
// owns its Dynamic body write domain and preserves each kind's source order.
// The host must validate finite inputs, normalized body/shape/inertia rotations,
// zero shape local-position padding, positive dimensions, layer indices < 32,
// material combine modes in [MaterialCombineAverage, MaterialCombineMaximum],
// nonnegative material/damping/speed bounds, convex radius <= min half extent,
// disjoint complete author body/leaf world ranges with bodies.zw reserved zero,
// each leaf's owning body and range, matching body.meta.w, actual runtime world
// candidate begin/count ranges, StepParams.solve.zw equal to 4C/C, candidate and
// Contact counts within device indexing and buffer/device address limits.
// Cross-island numerical work remains bounded. A single connected island is
// serial in this first slice; this is not a published performance result.
// JointStream contains one header, a three-word directory per joint, then
// compact author and runtime payloads sized by the shared kind schema. Word
// offsets are local to this resource generation. World joint ordinal ranges
// are disjoint and complete; endpoints
// belong to that world or use InvalidQueryBodySlot for the world endpoint.
// Fixed/Hinge author parameters must be finite, axes unit length and
// stiffness/damping/torque/friction/armature nonnegative. SetupJoints writes
// runtime independently; WarmJoints applies warmstart once per substep, only
// with nonzero velocity iterations. Joints
// and contacts use the same combined island partition. Runtime stores never
// modify the header or author prefix. No entry uses over eight storage ports.

// Existing substep and iteration order is unchanged. Build and independently
// validate the combined Dynamic-writer constraint plan after all SetupContacts
// and all selected SetupJoints and before WarmJoints. Setup/Finish retain
// complete selected-world joint coverage. No failed/uninitialized accepted joint may be compacted into
// an apparently successful empty schedule; the original failure gate remains.
// All units of a phase/iteration precede its successor. Each island phase uses
// StageWork = (localFirst, localCount, islandTableFirst, islandLaneCount).
// Accepted units cover every island's ascending local sequence exactly once;
// contact position's local ordinal is patchLocal * 4 + point. Bound each unit's
// rectangular items by WorkItemLimit and dispatch ceil(islandLaneCount / 64).
// Dynamic body writes are exclusive across island lanes. Static/kinematic/world
// endpoints remain read-only. Graph rows refer to original accepted JointStream
// and canonical candidate ordinals; ContactPatch word addresses still add the
// ContactPatches[0] bank base. No impulses, Contact records or Joint runtime move.
// ConstraintSchedule replaces the numerical consumer's Worlds storage binding.
// With the shared sleep activity input ContactVelocity uses eight storage
// ports, Joint numerical stages six and ContactPosition five. No numerical consumer binds labels, ranges or ID lists
// through separate storage buffers. Graph stages have independent recipes.
// The complete plan workset owns frozen input identities and source allocations
// through all accepted graph/numerical receipts. An invalid plan is never
// installed or dispatched; consumer guard returns do not establish validity.
// There is one production numerical path. Independent original-trajectory GPU
// diagnostics are required before claiming exactness or measuring throughput.

} // namespace GPU
} // namespace JPH
