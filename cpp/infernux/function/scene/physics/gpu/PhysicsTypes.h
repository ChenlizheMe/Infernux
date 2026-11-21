// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

#ifndef JPH_GPU_PHYSICS_TYPES_H
#define JPH_GPU_PHYSICS_TYPES_H

// Shared, pointer-free C++ / Slang storage contract. This header intentionally
// does not include Jolt.h, Body, PhysicsSystem, a graphics API or an allocator.
// Body positions are centers of mass. Each body addresses a contiguous leaf
// range; every leaf's local pose is relative to that body's center of mass.
#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace JPH
{
namespace GPU
{

struct alignas(16) Float4
{
    float x, y, z, w;
};
struct alignas(16) UInt4
{
    std::uint32_t x, y, z, w;
};
#define JPH_GPU_ALIGN alignas(16)
#define JPH_GPU_FLOAT4 Float4
#define JPH_GPU_UINT4 UInt4
#define JPH_GPU_UINT std::uint32_t
#define JPH_GPU_CONSTANT(name, value) inline constexpr std::uint32_t name = value
#else
#define JPH_GPU_ALIGN
#define JPH_GPU_FLOAT4 float4
#define JPH_GPU_UINT4 uint4
#define JPH_GPU_UINT uint
#define JPH_GPU_CONSTANT(name, value) static const uint name = value
#endif

JPH_GPU_CONSTANT(ShapeSphere, 0);
JPH_GPU_CONSTANT(ShapeBox, 1);
JPH_GPU_CONSTANT(ShapeCapsule, 2);
JPH_GPU_CONSTANT(ShapeCylinder, 3);
JPH_GPU_CONSTANT(ShapeConvexHull, 4);
JPH_GPU_CONSTANT(ShapeSensor, 1);
// Exact cooked mesh triangles share the planar hull support/face representation.
// The original winding and edge policy remain separate from convex geometry.
JPH_GPU_CONSTANT(ShapeOneSidedTriangle, 2);
JPH_GPU_CONSTANT(ShapeTriangleActiveEdgeShift, 2);
JPH_GPU_CONSTANT(PatchActive, 1);
JPH_GPU_CONSTANT(PatchSensor, 2);
JPH_GPU_CONSTANT(PatchOverlapping, 4);
JPH_GPU_CONSTANT(MotionStatic, 0);
JPH_GPU_CONSTANT(MotionKinematic, 1);
JPH_GPU_CONSTANT(MotionDynamic, 2);
// Residency is separate from motion type. A suspended actor retains its
// dynamic properties and state, but does not integrate or produce contacts.
JPH_GPU_CONSTANT(MotionNotResident, 4);
// Infernux PhysicsMaterialCombine values and priority (higher mode wins).
JPH_GPU_CONSTANT(MaterialCombineAverage, 0);
JPH_GPU_CONSTANT(MaterialCombineMinimum, 1);
JPH_GPU_CONSTANT(MaterialCombineMultiply, 2);
JPH_GPU_CONSTANT(MaterialCombineMaximum, 3);
// Initial allocation, grown by the owner without changing stable body slots.
JPH_GPU_CONSTANT(InitialBodyCapacity, 32);
// Fixed point records per independent ordered leaf pair.
JPH_GPU_CONSTANT(MaxContactPointsPerPair, 4);
// One sorted pair identity, a normal/count, friction and two words per point.
JPH_GPU_CONSTANT(ContactHistoryWords, 3 + 2 * MaxContactPointsPerPair);
// Sentinel for a render hierarchy node without a physical ancestor.
// FIFO body commands. These values match Infernux ForceMode.
JPH_GPU_CONSTANT(CommandSetPosition, 0);
JPH_GPU_CONSTANT(CommandSetRotation, 1);
JPH_GPU_CONSTANT(CommandSetPose, 2);
JPH_GPU_CONSTANT(CommandSetLinearVelocity, 3);
JPH_GPU_CONSTANT(CommandSetAngularVelocity, 4);
JPH_GPU_CONSTANT(CommandAddForce, 5);
JPH_GPU_CONSTANT(CommandAddTorque, 6);
JPH_GPU_CONSTANT(CommandAddForceAtPosition, 7);
// Internal motion-type boundary: preserve pose and clear velocities/forces,
// including after accepting a Static model that ignores dynamic API commands.
JPH_GPU_CONSTANT(CommandResetMotion, 8);
JPH_GPU_CONSTANT(CommandMovePosition, 9);
JPH_GPU_CONSTANT(CommandMoveRotation, 10);
JPH_GPU_CONSTANT(CommandMovePose, 11);
JPH_GPU_CONSTANT(CommandMotionBoundary, 12);
JPH_GPU_CONSTANT(CommandSleep, 13);
JPH_GPU_CONSTANT(CommandWake, 14);
// meta.z is decoded by command kind, independently of ForceMode.
JPH_GPU_CONSTANT(CommandPreserveKinematicMove, 1);
JPH_GPU_CONSTANT(CommandTemporaryStatic, 1);
JPH_GPU_CONSTANT(MotionBoundaryActionMask, 3);
JPH_GPU_CONSTANT(MotionBoundaryTypeChange, 0);
JPH_GPU_CONSTANT(MotionBoundarySuspendPreserve, 1);
JPH_GPU_CONSTANT(MotionBoundarySuspendReset, 2);
JPH_GPU_CONSTANT(MotionBoundaryProjectMobility, 3);
JPH_GPU_CONSTANT(MotionBoundaryCarryAuthoredPosition, 4);
JPH_GPU_CONSTANT(MotionBoundaryCarryAuthoredRotation, 8);
JPH_GPU_CONSTANT(TargetPending, 1);
JPH_GPU_CONSTANT(TargetDriven, 4);
JPH_GPU_CONSTANT(TargetTemporaryStatic, 8);
JPH_GPU_CONSTANT(CommandForce, 0);
JPH_GPU_CONSTANT(CommandAcceleration, 1);
JPH_GPU_CONSTANT(CommandImpulse, 2);
JPH_GPU_CONSTANT(CommandVelocityChange, 3);
JPH_GPU_CONSTANT(MaxCommandsPerBodyDispatch, 64);
JPH_GPU_CONSTANT(MaxCommandBodiesPerDispatch, 64);

struct JPH_GPU_ALIGN BodyModel
{
    // First leaf, leaf count, motion type | MotionNotResident, world index.
    // Zero leaves represent an empty collision body without a dummy primitive.
    JPH_GPU_UINT4 meta;
    // Raw inverse mass, exact world translation freeze bits (0..7), exact
    // world-space rotation freeze bits (0..7), gravity factor.
    JPH_GPU_FLOAT4 mass;
    // Principal inverse inertia xyz. w is zero for Discrete, +inner radius
    // for Continuous, or -inner radius for ContinuousDynamic.
    JPH_GPU_FLOAT4 invInertiaDiagonal;
    // Principal inertia frame relative to the body frame, quaternion xyzw.
    JPH_GPU_FLOAT4 inertiaRotation;
    // Linear damping, angular damping, maximum linear/angular speed.
    JPH_GPU_FLOAT4 damping;
};

struct JPH_GPU_ALIGN ShapeModel
{
    // Shape type, owning body slot, root SubShapeID prefix, sensor/triangle flags.
    JPH_GPU_UINT4 meta;
    // Sphere: xyz padding, w radius. Box: xyz half extents, w convex radius.
    // Capsule: (0, half cylinder height, 0, radius), with the core along local Y.
    // Cylinder: (radius, half height, convex radius,0), exact local-Y cylinder.
    // Hull: (convex radius,0,0,0); cooked triangle hulls use yzw for their CCW unit normal.
    // Hull: (convex radius,0,0,0); full/core/topology in geometryReference.
    // Box core = half extents - convex radius. The radius is not discarded.
    JPH_GPU_FLOAT4 geometry;
    // Primitive center in the body COM frame, xyz. A collider center already
    // included in the body's COM must not be added a second time.
    // For mesh triangles, w carries the common mesh ray origin X in body COM
    // space. Its YZ are in material.zw. Other shapes have zero padding here.
    JPH_GPU_FLOAT4 localPositionCOM;
    // Primitive frame relative to the body frame, normalized quaternion xyzw.
    // This does not rotate the body's principal inertia frame.
    JPH_GPU_FLOAT4 localRotation;
    // Friction and restitution in xy. Mesh triangle ray origin YZ in zw, zero otherwise.
    JPH_GPU_FLOAT4 material;
    // Layer index [0, 31], collision mask, friction combine, bounce combine.
    // Both masks must admit the opposite leaf's layer for a pair to collide.
    JPH_GPU_UINT4 surface;
    // Collider ID low/high words, lifetime generation low/high words.
    // SubShapeID/prefix-bit metadata remains in the accepted host leaf directory.
    JPH_GPU_UINT4 source;
    // Immutable cold record absolute word offset/length; zw zero. Inline shapes use zero.
    JPH_GPU_UINT4 geometryReference;
};

struct JPH_GPU_ALIGN ColliderPairKey
{
    // Each endpoint is (ID low/high, generation low/high). Canonical order is
    // unsigned ID64, then generation64; no device uint64 arithmetic is needed.
    JPH_GPU_UINT4 a;
    JPH_GPU_UINT4 b;
};

struct JPH_GPU_ALIGN BodyState
{
    JPH_GPU_FLOAT4 positionCOM;     // xyz position; w padding.
    JPH_GPU_FLOAT4 rotation;        // Normalized quaternion xyzw.
    JPH_GPU_FLOAT4 linearVelocity;  // xyz; w padding.
    JPH_GPU_FLOAT4 angularVelocity; // xyz; w padding.
};

struct JPH_GPU_ALIGN BodyForces
{
    // Accumulator for one logical tick, shared by all of its substeps. The
    // tick owner clears it after the last substep. Continuous controllers
    // submit their contribution for each logical tick.
    JPH_GPU_FLOAT4 force;  // World space xyz; w padding.
    JPH_GPU_FLOAT4 torque; // World space xyz; w padding.
};

// Device-owned target/arrival-stop state. No CPU dynamic pose mirror is used.
// The inactive payload is not read; initialization and retirement clear it.
struct JPH_GPU_ALIGN KinematicTarget
{
    JPH_GPU_FLOAT4 positionSpeed; // Target body origin xyz; w maximum drive speed (0 = uncapped).
    JPH_GPU_FLOAT4 rotation;      // Target normalized quaternion xyzw.
    JPH_GPU_FLOAT4 rootCOMDt;     // Accepted local root COM xyz; w requested duration.
    JPH_GPU_UINT4 state;          // Target flags in x; yzw zero.
};

struct JPH_GPU_ALIGN WorldRange
{
    // Body begin/count. Author models reserve zw as zero; a substep runtime
    // view writes its actual candidate begin/count there after the GPU scan.
    JPH_GPU_UINT4 bodies;
    // Leaf begin/count, joint begin/count in the accepted JointStream.
    JPH_GPU_UINT4 leaves;
};

JPH_GPU_CONSTANT(JointFixed, 0);
JPH_GPU_CONSTANT(JointHinge, 1);
JPH_GPU_CONSTANT(JointSlider, 2);
JPH_GPU_CONSTANT(JointDistance, 3);
JPH_GPU_CONSTANT(JointSwingTwist, 4);
JPH_GPU_CONSTANT(JointSixDOF, 5);
JPH_GPU_CONSTANT(JointEnabled, 1);
JPH_GPU_CONSTANT(JointCollisionEnabled, 2);
JPH_GPU_CONSTANT(JointLimitsEnabled, 4);
JPH_GPU_CONSTANT(JointMotorEnabled, 8);
JPH_GPU_CONSTANT(JointAutoDistance, 16);
JPH_GPU_CONSTANT(JointResetWarmStart, 1);
JPH_GPU_CONSTANT(JointResetRotor, 2);
JPH_GPU_CONSTANT(JointResetSwingLimits, 4);
JPH_GPU_CONSTANT(JointResetSwingMotor, 8);
JPH_GPU_CONSTANT(JointResetTwistMotor, 16);
JPH_GPU_CONSTANT(JointResetMobility, 32);
JPH_GPU_CONSTANT(JointResetSixClassification, 64);
JPH_GPU_CONSTANT(JointResetPoseDrive, 128);
JPH_GPU_CONSTANT(JointPoseDriveActive, 1);
JPH_GPU_CONSTANT(JointPoseDriveClear, 0);
JPH_GPU_CONSTANT(JointPoseDriveSet, 1);
// Per-axis reset bits: translation limits, translation motors, angular motors.
JPH_GPU_CONSTANT(JointResetSixLimitShift, 8);
JPH_GPU_CONSTANT(JointResetSixLinearMotorShift, 11);
JPH_GPU_CONSTANT(JointResetSixAngularMotorShift, 14);
JPH_GPU_CONSTANT(JointStatusSuccess, 0);
JPH_GPU_CONSTANT(JointStatusInvalid, 1);
JPH_GPU_CONSTANT(JointStatusNumericalFailure, 2);
#include "JointPayloadFields.inl"
#define JPH_PAYLOAD_F(name) JPH_GPU_FLOAT4 name;
#define JPH_PAYLOAD_U(name) JPH_GPU_UINT4 name;
// Header: (count, authorEndWord, totalWordCount, 0).
// Directory layout: (authorWordOffset, runtimeWordOffset, 0, 0).
// Runtime flags: (initialized, status, precedingDt float bits, kind flags).
// Hinge rotor.w is the cached accepted armature; other kinds have no rotor.
struct JPH_GPU_ALIGN JointDirectory
{
    JPH_GPU_UINT4 bodies;
    JPH_GPU_UINT4 identity;
    JPH_GPU_UINT4 layout;
};
JPH_GPU_CONSTANT(JointDirectoryWords, 3);
// One captured runtime intent: current row, stable ID low/high, operation.
struct JPH_GPU_ALIGN JointPoseDrive
{
    JPH_GPU_UINT4 meta;
    JPH_GPU_FLOAT4 relative;
};
struct JPH_GPU_ALIGN JointFixedAuthor
{
#ifndef __cplusplus
    uint4 bodies, identity;
#endif
    JPH_FIXED_AUTHOR_FIELDS(JPH_PAYLOAD_F, JPH_PAYLOAD_U)
};
JPH_GPU_CONSTANT(JointFixedAuthorWords, 6);
struct JPH_GPU_ALIGN JointFixedState{JPH_FIXED_STATE_FIELDS(JPH_PAYLOAD_F, JPH_PAYLOAD_U)};
JPH_GPU_CONSTANT(JointFixedStateWords, 9);
struct JPH_GPU_ALIGN JointHingeAuthor
{
#ifndef __cplusplus
    uint4 bodies, identity;
#endif
    JPH_HINGE_AUTHOR_FIELDS(JPH_PAYLOAD_F, JPH_PAYLOAD_U)
};
JPH_GPU_CONSTANT(JointHingeAuthorWords, 10);
struct JPH_GPU_ALIGN JointHingeState{JPH_HINGE_STATE_FIELDS(JPH_PAYLOAD_F, JPH_PAYLOAD_U)};
JPH_GPU_CONSTANT(JointHingeStateWords, 13);
struct JPH_GPU_ALIGN JointSliderAuthor
{
#ifndef __cplusplus
    uint4 bodies, identity;
#endif
    JPH_SLIDER_AUTHOR_FIELDS(JPH_PAYLOAD_F, JPH_PAYLOAD_U)
};
JPH_GPU_CONSTANT(JointSliderAuthorWords, 8);
struct JPH_GPU_ALIGN JointSliderState{JPH_SLIDER_STATE_FIELDS(JPH_PAYLOAD_F, JPH_PAYLOAD_U)};
JPH_GPU_CONSTANT(JointSliderStateWords, 11);
struct JPH_GPU_ALIGN JointDistanceAuthor
{
#ifndef __cplusplus
    uint4 bodies, identity;
#endif
    JPH_DISTANCE_AUTHOR_FIELDS(JPH_PAYLOAD_F, JPH_PAYLOAD_U)
};
JPH_GPU_CONSTANT(JointDistanceAuthorWords, 10);
struct JPH_GPU_ALIGN JointDistanceState{JPH_DISTANCE_STATE_FIELDS(JPH_PAYLOAD_F, JPH_PAYLOAD_U)};
JPH_GPU_CONSTANT(JointDistanceStateWords, 8);
struct JPH_GPU_ALIGN JointSwingAuthor
{
#ifndef __cplusplus
    uint4 bodies, identity;
#endif
    JPH_SWING_AUTHOR_FIELDS(JPH_PAYLOAD_F, JPH_PAYLOAD_U)
};
JPH_GPU_CONSTANT(JointSwingAuthorWords, 13);
struct JPH_GPU_ALIGN JointSwingState{JPH_SWING_STATE_FIELDS(JPH_PAYLOAD_F, JPH_PAYLOAD_U)};
JPH_GPU_CONSTANT(JointSwingStateWords, 17);
struct JPH_GPU_ALIGN JointSixAuthor
{
#ifndef __cplusplus
    uint4 bodies, identity;
#endif
    JPH_SIX_AUTHOR_FIELDS(JPH_PAYLOAD_F, JPH_PAYLOAD_U)
};
JPH_GPU_CONSTANT(JointSixAuthorWords, 26);
struct JPH_GPU_ALIGN JointSixState{JPH_SIX_STATE_FIELDS(JPH_PAYLOAD_F, JPH_PAYLOAD_U)};
JPH_GPU_CONSTANT(JointSixStateWords, 29);
#undef JPH_PAYLOAD_F
#undef JPH_PAYLOAD_U
#ifdef __cplusplus
// Accessor availability is derived from the same field schema as the typed
// records; no second list of author field declarations is maintained here.
#define JPH_AUTHOR_TRAIT(name)                                                                                         \
    template <class T, class = void> struct JointHas_##name : std::false_type                                          \
    {                                                                                                                  \
    };                                                                                                                 \
    template <class T> struct JointHas_##name<T, std::void_t<decltype(std::declval<T>().name)>> : std::true_type       \
    {                                                                                                                  \
    };
JPH_ALL_AUTHOR_FIELDS(JPH_AUTHOR_TRAIT, JPH_AUTHOR_TRAIT)
#undef JPH_AUTHOR_TRAIT
struct JointAuthor
{
    UInt4 bodies{}, identity{};
    using Payload = std::variant<JointFixedAuthor, JointHingeAuthor, JointSliderAuthor, JointDistanceAuthor,
                                 JointSwingAuthor, JointSixAuthor>;
    Payload payload = JointFixedAuthor{};
    explicit JointAuthor(std::uint32_t kind = JointFixed)
    {
        SetKind(kind);
    }
    void SetKind(std::uint32_t kind)
    {
        bodies.z = kind;
        switch (kind) {
        case JointFixed:
            payload.emplace<JointFixedAuthor>();
            break;
        case JointHinge:
            payload.emplace<JointHingeAuthor>();
            break;
        case JointSlider:
            payload.emplace<JointSliderAuthor>();
            break;
        case JointDistance:
            payload.emplace<JointDistanceAuthor>();
            break;
        case JointSwingTwist:
            payload.emplace<JointSwingAuthor>();
            break;
        case JointSixDOF:
            payload.emplace<JointSixAuthor>();
            break;
        default:
            throw std::invalid_argument("Unknown joint payload kind");
        }
    }
    bool HasMatchingKind() const noexcept
    {
        switch (bodies.z) {
        case JointFixed:
            return std::holds_alternative<JointFixedAuthor>(payload);
        case JointHinge:
            return std::holds_alternative<JointHingeAuthor>(payload);
        case JointSlider:
            return std::holds_alternative<JointSliderAuthor>(payload);
        case JointDistance:
            return std::holds_alternative<JointDistanceAuthor>(payload);
        case JointSwingTwist:
            return std::holds_alternative<JointSwingAuthor>(payload);
        case JointSixDOF:
            return std::holds_alternative<JointSixAuthor>(payload);
        default:
            return false;
        }
    }
#define JPH_AUTHOR_ACCESS(type, name)                                                                                  \
    type &name()                                                                                                       \
    {                                                                                                                  \
        return std::visit(                                                                                             \
            [](auto &p) -> type & {                                                                                    \
                if constexpr (JointHas_##name<std::decay_t<decltype(p)>>::value)                                       \
                    return p.name;                                                                                     \
                else                                                                                                   \
                    throw std::logic_error("Field is absent from this joint kind: " #name);                            \
            },                                                                                                         \
            payload);                                                                                                  \
    }                                                                                                                  \
    const type &name() const                                                                                           \
    {                                                                                                                  \
        static constexpr type zero{};                                                                                  \
        return std::visit(                                                                                             \
            [](const auto &p) -> const type & {                                                                        \
                if constexpr (JointHas_##name<std::decay_t<decltype(p)>>::value)                                       \
                    return p.name;                                                                                     \
                else                                                                                                   \
                    return zero;                                                                                       \
            },                                                                                                         \
            payload);                                                                                                  \
    }
#define JPH_AUTHOR_ACCESS_F(name) JPH_AUTHOR_ACCESS(Float4, name)
#define JPH_AUTHOR_ACCESS_U(name) JPH_AUTHOR_ACCESS(UInt4, name)
    JPH_ALL_AUTHOR_FIELDS(JPH_AUTHOR_ACCESS_F, JPH_AUTHOR_ACCESS_U)
#undef JPH_AUTHOR_ACCESS_F
#undef JPH_AUTHOR_ACCESS_U
#undef JPH_AUTHOR_ACCESS
};
#endif

// Demand-only typed motion/pose results and grouped status scratch.
// This is not the persistent per-kind runtime state or a full-stream CPU read.
struct JPH_GPU_ALIGN JointReadback
{
    JPH_GPU_FLOAT4 values;      // Scalar coordinate/velocity or Six position.xyz/0.
    JPH_GPU_FLOAT4 orientation; // Swing/Six XYZW; zero for scalar/status rows.
    JPH_GPU_UINT4 status;
};

struct JPH_GPU_ALIGN JointParams
{
    // Initialize: new count, old count, 0, 0. Motion: count, requested kind, rows, body base.
    // Update: count, 0, requests, 0. Status: input first/count, output first, level.
    JPH_GPU_UINT4 counts;
};

struct JPH_GPU_ALIGN Aabb
{
    JPH_GPU_FLOAT4 minimum; // xyz; during CCD, w is positive for initiators, -1 for other dynamic, 0 for prescribed.
    JPH_GPU_FLOAT4 maximum; // xyz; w padding.
};

// Selected-world motion after velocity solving and before translation.
struct JPH_GPU_ALIGN SweepBody
{
    JPH_GPU_FLOAT4 positionCOM; // Segment origin xyz; w records an impact in an earlier segment.
    JPH_GPU_FLOAT4 rotation;
    JPH_GPU_FLOAT4 linearVelocity;
    JPH_GPU_FLOAT4 angularVelocity;
    JPH_GPU_FLOAT4 displacement;    // full substep translation xyz and signed sweep role w (initiator inner radius).
    JPH_GPU_FLOAT4 linearResponse;  // velocity change xyz and travel fraction w.
    JPH_GPU_FLOAT4 angularResponse; // angular velocity change xyz, w marks a solved impact.
};

JPH_GPU_CONSTANT(SweepInitial, 1);
JPH_GPU_CONSTANT(SweepContinuation, 2);
JPH_GPU_CONSTANT(SweepEventContacts, 3);
JPH_GPU_CONSTANT(SweepNoImpact, 0);
JPH_GPU_CONSTANT(SweepLastImpact, 1);
JPH_GPU_CONSTANT(SweepMoreMotion, 2);
JPH_GPU_CONSTANT(SweepMissingContact, 67);
JPH_GPU_CONSTANT(SweepUnsettled, 68);
JPH_GPU_CONSTANT(SweepMotionChunk, 2048);
JPH_GPU_CONSTANT(SweepMotionDispatch, 0);
JPH_GPU_CONSTANT(SweepReportDispatch, 1);
JPH_GPU_CONSTANT(SweepIntegrateDispatch, 2);
JPH_GPU_CONSTANT(SweepArgumentWords, 3);
// The same directory supplies the link/solve phase before it supplies motion.
JPH_GPU_CONSTANT(SweepLinkDispatch, 0);
JPH_GPU_CONSTANT(SweepFlattenDispatch, 1);
JPH_GPU_CONSTANT(SweepSolveDispatch, 2);
JPH_GPU_CONSTANT(SweepSolverReceipt, 1);
// Internal status of the sweep progress receipt. Numeric work cannot consume
// the candidate prefix until the broadphase has removed this condition.
JPH_GPU_CONSTANT(SweepCandidatesIncomplete, 4);

struct JPH_GPU_ALIGN ContactPatch
{
    // Global leaf A/B indices and patch flags. After narrowphase w is the
    // actual manifold point count; broadphase candidates use w for world row.
    // Each patch owns MaxContactPointsPerPair Contact records. Sensor patches
    // retain real overlap/source but never activate response points.
    JPH_GPU_UINT4 leaves;
};

struct JPH_GPU_ALIGN Contact
{
    // Global body A/B indices, active (0/1), primitive contact feature.
    JPH_GPU_UINT4 meta;
    // Normal points from A towards B; positive depth means penetration.
    JPH_GPU_FLOAT4 normalDepth;
    // Body COM frame xyz. In the first record of a pair, w stores the
    // accumulated manifold angular-friction impulse; other records keep w=0.
    JPH_GPU_FLOAT4 localAnchorA;
    JPH_GPU_FLOAT4 localAnchorB; // Body COM frame xyz; combined friction in the first record's w.
    // x is this point's normal impulse; w is its fixed normal velocity bias.
    // yz in the first pair record are the manifold's two linear-friction
    // impulses; other records keep yz=0. Setup resets this substep's scratch
    // before the island optionally applies matching contact history.
    JPH_GPU_FLOAT4 impulse;
};

struct JPH_GPU_ALIGN StepParams
{
    // Body count, world count, total leaf count, velocity iterations.
    JPH_GPU_UINT4 counts;
    // Position iterations, substep count, total contact count, total patch count.
    JPH_GPU_UINT4 solve;
    // Gravity xyz and the delta time of this substep.
    JPH_GPU_FLOAT4 gravityDt;
    // Speculative distance, penetration slop, max position correction, Baumgarte.
    JPH_GPU_FLOAT4 contact;
    // Minimum restitution speed, linear sweep threshold and max penetration (relative to inner radius), w zero.
    JPH_GPU_FLOAT4 restitution;
    // One accepted layout address, never a stable RuntimeWorld identity.
    // selection=(world row, body base, leaf base, joint base).
    // selectedCounts=(body capacity, leaf count, joint count, zero).
    // Counts above remain TOTAL array extents for global address validation.
    JPH_GPU_UINT4 selection;
    JPH_GPU_UINT4 selectedCounts;
};

// One body-domain dispatch over immutable selections in the same accepted layout.
// Every selection is an unmodified StepParams value, including its own dt/gravity.
struct JPH_GPU_ALIGN BodyBatchHeader
{
    // Selection count, tile count, total BodyModel extent, total WorldRange extent.
    JPH_GPU_UINT4 counts;
};

// Structured storage stride is 12 bytes; no vector alignment or spare fields.
// One group of 64 lanes addresses only this tile's selected body interval.
struct BodyBatchTile
{
    JPH_GPU_UINT selectionIndex;
    JPH_GPU_UINT bodyFirst;
    JPH_GPU_UINT bodyCount;
};

// One immutable dispatch unit. Independent domains use global item ordinals;
// island domains use local ordinals and an explicit island-table range.
struct JPH_GPU_ALIGN StageWork
{
    JPH_GPU_UINT4 range; // Global item first/count, or local first/count and island table first/count.
};

// Sparse body mutations at an owner-controlled boundary between logical ticks.
// Each list addresses unique slots. Neither operation advances simulation time.
struct JPH_GPU_ALIGN BodyMutationParams
{
    // Initialization count, COM update count, first initialization record, padding.
    JPH_GPU_UINT4 counts;
};

struct JPH_GPU_ALIGN BodyInitialization
{
    // x is the body slot; yzw are zero padding. The slot's actor identity is
    // owned by the engine registry, not encoded in the device array index.
    JPH_GPU_UINT4 bodyIndex;
    BodyState state;
    BodyForces forces;
};

struct JPH_GPU_ALIGN CenterOfMassUpdate
{
    JPH_GPU_UINT4 bodyIndex; // x is the body slot; yzw are zero padding.
    // New minus old root COM in the authored body frame, including baked scale.
    JPH_GPU_FLOAT4 deltaLocalCOM;
};

// Cold command payload, separate from the hot 96-byte BodyModel. Pose commands
// carry the accepted root COM in the authored body frame, including baked scale.
// Positions supplied by the public API are body origins, not COM positions.
struct JPH_GPU_ALIGN BodyCommand
{
    JPH_GPU_UINT4 meta; // Body slot, kind, kind-specific mode/flags, zero padding.
    // Position/velocity/force/torque xyz. Move kinds use w for requested dt;
    // all other consumed vector w values are zero.
    JPH_GPU_FLOAT4 value;
    // SetRotation/SetPose: normalized quaternion xyzw. Force-at-position:
    // world point xyz with zero w. MoveRotation/MovePose also consume quaternion.
    // Other commands do not consume this field.
    JPH_GPU_FLOAT4 auxiliary;
    // Pose/Move local root COM xyz. Move kinds use w for maximum drive speed;
    // ordinary pose commands use zero w. Boundary carries use device Target.
    JPH_GPU_FLOAT4 rootCOM;
};

struct JPH_GPU_ALIGN BodyCommandSpan
{
    // Body slot, first command, command count (1..64), zero padding.
    JPH_GPU_UINT4 range;
};

struct JPH_GPU_ALIGN BodyCommandParams
{
    // First span, span count (1..64); zw are zero padding.
    JPH_GPU_UINT4 spans;
};

JPH_GPU_CONSTANT(InvalidQueryBodySlot, 0xFFFFFFFFu);
JPH_GPU_CONSTANT(QueryOverlap, 1);
JPH_GPU_CONSTANT(QueryClosestPoint, 2);
JPH_GPU_CONSTANT(QueryBodyBounds, 3);
JPH_GPU_CONSTANT(QueryPenetration, 4);
JPH_GPU_CONSTANT(QueryShapeCast, 7);
JPH_GPU_CONSTANT(QueryClosest, 0);
JPH_GPU_CONSTANT(QueryAll, 1);
JPH_GPU_CONSTANT(QueryUniqueCollider, 2);
JPH_GPU_CONSTANT(QueryUniqueBody, 3);
JPH_GPU_CONSTANT(QueryAny, 4);
JPH_GPU_CONSTANT(QueryTargetCollider, 1);
JPH_GPU_CONSTANT(QueryExcludeBody, 2);
JPH_GPU_CONSTANT(QuerySweepPolicy, 4);
// Internal: the native Sweep source lifetime matches this retained policy.
JPH_GPU_CONSTANT(QuerySweepPairPolicy, 8);
JPH_GPU_CONSTANT(QuerySuccess, 0);
JPH_GPU_CONSTANT(QueryUnsupported, 1);
JPH_GPU_CONSTANT(QueryNumericalFailure, 2);
// Internal continuation only; never a completed public query status.
JPH_GPU_CONSTANT(QueryPending, 3);
// QueryParams.selectedCounts.w combines the optional motion projection mask
// with this fixed input-row mapping proof for device-only exports.

struct JPH_GPU_ALIGN QueryParams
{
    // Request count, selected global body base/count, selected leaf count.
    JPH_GPU_UINT4 counts;
    // Explicit primitive count, scratch record stride, record capacity, padding.
    JPH_GPU_UINT4 sizes;
    // Same selected layout address as StepParams; model/shape indices remain global.
    // BodyState indices are globalSlot-selection.y in the compact completed output.
    JPH_GPU_UINT4 selection;
    JPH_GPU_UINT4 selectedCounts;
};

struct JPH_GPU_ALIGN QueryInput
{
    // Operation, collector, query primitive type, filter flags.
    JPH_GPU_UINT4 meta;
    // Layer mask, include sensors, exact Collider ID low/high when requested.
    JPH_GPU_UINT4 filter;
    // Excluded body slot; second primitive / gather target slot. zw carry motion fields,
    // the sweeping GameObject.
    JPH_GPU_UINT4 source;
    // Ray origin / point / query shape center / bounds center; w maximum travel.
    JPH_GPU_FLOAT4 positionMaximum;
    // Ray direction or bounds half extent; w zero padding.
    JPH_GPU_FLOAT4 direction;
    JPH_GPU_FLOAT4 rotation; // Query primitive quaternion xyzw.
    JPH_GPU_FLOAT4 geometry; // Same cooked geometry law as ShapeModel.
    // Source Sweep Collider generation low/high; zw zero. Other queries use zero.
    JPH_GPU_UINT4 lifetime;
    // Exact explicit first operand; never inferred from current global cook.
    JPH_GPU_UINT4 geometryReference;
};

struct JPH_GPU_ALIGN QueryCastParams
{
    // Work quantum, body identity prefix count, pair suffix base in 16-byte words, pair count.
    JPH_GPU_UINT4 controls;
};

struct JPH_GPU_ALIGN QueryCastState
{
    // Current global leaf, phase (0 visit / 1 advance / 2 done), result count, status.
    JPH_GPU_UINT4 cursor;
    // Current travel, previous separation, previous advance, best travel (world units).
    JPH_GPU_FLOAT4 progress;
    // Previous unit A-to-B normal xyz and support gap in world units.
    JPH_GPU_FLOAT4 normal;
    // Total request work units, current pair steps, restart used, no-progress count.
    JPH_GPU_UINT4 iteration;
};

struct JPH_GPU_ALIGN QueryPrimitive
{
    JPH_GPU_UINT4 meta; // Shape type, then zero padding.
    JPH_GPU_FLOAT4 geometry;
    JPH_GPU_FLOAT4 position;
    JPH_GPU_FLOAT4 rotation;
    JPH_GPU_UINT4 geometryReference;
};

struct JPH_GPU_ALIGN QueryBodyIdentity
{
    // Body prefix: public body ID, has Rigidbody, Rigidbody component ID low/high.
    // Leaf suffix: owning GameObject ID low/high, then zero. Cast controls.y is its base.
    JPH_GPU_UINT4 ids;
};

struct JPH_GPU_ALIGN QueryHit
{
    JPH_GPU_UINT4 meta;              // Body slot, valid flag, global leaf ordinal, feature.
    JPH_GPU_FLOAT4 positionDistance; // Target surface point xyz and travel/distance.
    JPH_GPU_FLOAT4 normalFraction;   // Target outward normal xyz and travel fraction.
    JPH_GPU_FLOAT4 pointADepth;      // Query surface witness xyz and positive penetration.
};

struct JPH_GPU_ALIGN QueryRecordWord
{
    JPH_GPU_UINT4 value;
};

struct JPH_GPU_ALIGN QuerySpan
{
    JPH_GPU_UINT4 range; // Compact offset, record count, status, padding.
};

struct JPH_GPU_ALIGN QuerySummary
{
    JPH_GPU_UINT4 counts; // Total records, status, request count, zero padding.
};

#ifdef __cplusplus
static_assert(sizeof(Float4) == 16 && alignof(Float4) == 16);
static_assert(sizeof(UInt4) == 16 && alignof(UInt4) == 16);
#define JPH_GPU_CHECK_TYPE(type, bytes)                                                                                \
    static_assert(sizeof(type) == bytes && alignof(type) == 16);                                                       \
    static_assert(std::is_standard_layout_v<type> && std::is_trivially_copyable_v<type>)
JPH_GPU_CHECK_TYPE(BodyModel, 80);
JPH_GPU_CHECK_TYPE(ShapeModel, 128);
JPH_GPU_CHECK_TYPE(BodyState, 64);
JPH_GPU_CHECK_TYPE(BodyForces, 32);
JPH_GPU_CHECK_TYPE(KinematicTarget, 64);
JPH_GPU_CHECK_TYPE(WorldRange, 32);
JPH_GPU_CHECK_TYPE(JointDirectory, 48);
JPH_GPU_CHECK_TYPE(JointPoseDrive, 32);
static_assert(offsetof(JointPoseDrive, meta) == 0);
static_assert(offsetof(JointPoseDrive, relative) == 16);
static_assert(offsetof(JointHingeState, poseDriveRelative) == 192);
static_assert(offsetof(JointSwingState, poseDriveRelative) == 256);
static_assert(offsetof(JointSixState, poseDriveRelative) == 448);
JPH_GPU_CHECK_TYPE(JointFixedAuthor, 96);
JPH_GPU_CHECK_TYPE(JointFixedState, 144);
JPH_GPU_CHECK_TYPE(JointHingeAuthor, 160);
JPH_GPU_CHECK_TYPE(JointHingeState, 208);
JPH_GPU_CHECK_TYPE(JointSliderAuthor, 128);
JPH_GPU_CHECK_TYPE(JointSliderState, 176);
JPH_GPU_CHECK_TYPE(JointDistanceAuthor, 160);
JPH_GPU_CHECK_TYPE(JointDistanceState, 128);
JPH_GPU_CHECK_TYPE(JointSwingAuthor, 208);
JPH_GPU_CHECK_TYPE(JointSwingState, 272);
JPH_GPU_CHECK_TYPE(JointSixAuthor, 416);
JPH_GPU_CHECK_TYPE(JointSixState, 464);
JPH_GPU_CHECK_TYPE(JointReadback, 48);
JPH_GPU_CHECK_TYPE(JointParams, 16);
static_assert(offsetof(JointFixedAuthor, ownerAnchor) == 0);
static_assert(offsetof(JointFixedAuthor, ownerLocalAnchor) == 16);
static_assert(offsetof(JointFixedAuthor, rootCOM1) == 32);
static_assert(offsetof(JointFixedAuthor, rootCOM2) == 48);
static_assert(offsetof(JointFixedAuthor, scale1) == 64);
static_assert(offsetof(JointFixedAuthor, scale2) == 80);
static_assert(offsetof(JointHingeAuthor, ownerAnchor) == 0);
static_assert(offsetof(JointHingeAuthor, ownerLocalAnchor) == 16);
static_assert(offsetof(JointHingeAuthor, rootCOM1) == 32);
static_assert(offsetof(JointHingeAuthor, rootCOM2) == 48);
static_assert(offsetof(JointHingeAuthor, scale1) == 64);
static_assert(offsetof(JointHingeAuthor, scale2) == 80);
static_assert(offsetof(JointHingeAuthor, ownerAxis) == 96);
static_assert(offsetof(JointHingeAuthor, limits) == 112);
static_assert(offsetof(JointHingeAuthor, motor) == 128);
static_assert(offsetof(JointHingeAuthor, passive) == 144);
static_assert(offsetof(JointSliderAuthor, ownerAnchor) == 0);
static_assert(offsetof(JointSliderAuthor, ownerLocalAnchor) == 16);
static_assert(offsetof(JointSliderAuthor, rootCOM1) == 32);
static_assert(offsetof(JointSliderAuthor, rootCOM2) == 48);
static_assert(offsetof(JointSliderAuthor, scale1) == 64);
static_assert(offsetof(JointSliderAuthor, scale2) == 80);
static_assert(offsetof(JointSliderAuthor, ownerAxis) == 96);
static_assert(offsetof(JointSliderAuthor, limits) == 112);
static_assert(offsetof(JointDistanceAuthor, ownerAnchor) == 0);
static_assert(offsetof(JointDistanceAuthor, ownerLocalAnchor) == 16);
static_assert(offsetof(JointDistanceAuthor, rootCOM1) == 32);
static_assert(offsetof(JointDistanceAuthor, rootCOM2) == 48);
static_assert(offsetof(JointDistanceAuthor, scale1) == 64);
static_assert(offsetof(JointDistanceAuthor, scale2) == 80);
static_assert(offsetof(JointDistanceAuthor, connectedAnchor) == 96);
static_assert(offsetof(JointDistanceAuthor, connectedLocalAnchor) == 112);
static_assert(offsetof(JointDistanceAuthor, limits) == 128);
static_assert(offsetof(JointDistanceAuthor, linearSpring) == 144);
static_assert(offsetof(JointSwingAuthor, ownerAnchor) == 0);
static_assert(offsetof(JointSwingAuthor, ownerLocalAnchor) == 16);
static_assert(offsetof(JointSwingAuthor, rootCOM1) == 32);
static_assert(offsetof(JointSwingAuthor, rootCOM2) == 48);
static_assert(offsetof(JointSwingAuthor, scale1) == 64);
static_assert(offsetof(JointSwingAuthor, scale2) == 80);
static_assert(offsetof(JointSwingAuthor, angularFrame) == 96);
static_assert(offsetof(JointSwingAuthor, angularLimits) == 112);
static_assert(offsetof(JointSwingAuthor, angularModes) == 128);
static_assert(offsetof(JointSwingAuthor, angularSpring) == 144);
static_assert(offsetof(JointSwingAuthor, angularTorque) == 160);
static_assert(offsetof(JointSwingAuthor, angularTarget) == 176);
static_assert(offsetof(JointSwingAuthor, angularVelocity) == 192);
static_assert(offsetof(JointFixedState, flags) == 0);
static_assert(offsetof(JointFixedState, anchor1) == 16);
static_assert(offsetof(JointFixedState, anchor2) == 32);
static_assert(offsetof(JointFixedState, collapsed1) == 48);
static_assert(offsetof(JointFixedState, collapsed2) == 64);
static_assert(offsetof(JointFixedState, frame1) == 80);
static_assert(offsetof(JointFixedState, frame2) == 96);
static_assert(offsetof(JointFixedState, pointImpulse) == 112);
static_assert(offsetof(JointFixedState, angularImpulse) == 128);
static_assert(offsetof(JointHingeState, flags) == 0);
static_assert(offsetof(JointHingeState, anchor1) == 16);
static_assert(offsetof(JointHingeState, anchor2) == 32);
static_assert(offsetof(JointHingeState, collapsed1) == 48);
static_assert(offsetof(JointHingeState, collapsed2) == 64);
static_assert(offsetof(JointHingeState, frame1) == 80);
static_assert(offsetof(JointHingeState, frame2) == 96);
static_assert(offsetof(JointHingeState, pointImpulse) == 112);
static_assert(offsetof(JointHingeState, angularImpulse) == 128);
static_assert(offsetof(JointHingeState, axisImpulse) == 144);
static_assert(offsetof(JointHingeState, rotor) == 160);
static_assert(offsetof(JointHingeState, setup) == 176);
static_assert(offsetof(JointSliderState, flags) == 0);
static_assert(offsetof(JointSliderState, anchor1) == 16);
static_assert(offsetof(JointSliderState, anchor2) == 32);
static_assert(offsetof(JointSliderState, collapsed1) == 48);
static_assert(offsetof(JointSliderState, collapsed2) == 64);
static_assert(offsetof(JointSliderState, frame1) == 80);
static_assert(offsetof(JointSliderState, frame2) == 96);
static_assert(offsetof(JointSliderState, pointImpulse) == 112);
static_assert(offsetof(JointSliderState, angularImpulse) == 128);
static_assert(offsetof(JointSliderState, axisImpulse) == 144);
static_assert(offsetof(JointSliderState, setup) == 160);
static_assert(offsetof(JointDistanceState, flags) == 0);
static_assert(offsetof(JointDistanceState, anchor1) == 16);
static_assert(offsetof(JointDistanceState, anchor2) == 32);
static_assert(offsetof(JointDistanceState, collapsed1) == 48);
static_assert(offsetof(JointDistanceState, collapsed2) == 64);
static_assert(offsetof(JointDistanceState, axisImpulse) == 80);
static_assert(offsetof(JointDistanceState, setup) == 96);
static_assert(offsetof(JointDistanceState, distanceReference) == 112);
static_assert(offsetof(JointSwingState, flags) == 0);
static_assert(offsetof(JointSwingState, anchor1) == 16);
static_assert(offsetof(JointSwingState, anchor2) == 32);
static_assert(offsetof(JointSwingState, collapsed1) == 48);
static_assert(offsetof(JointSwingState, collapsed2) == 64);
static_assert(offsetof(JointSwingState, frame1) == 80);
static_assert(offsetof(JointSwingState, frame2) == 96);
static_assert(offsetof(JointSwingState, pointImpulse) == 112);
static_assert(offsetof(JointSwingState, angularLimitX) == 128);
static_assert(offsetof(JointSwingState, angularLimitY) == 144);
static_assert(offsetof(JointSwingState, angularLimitZ) == 160);
static_assert(offsetof(JointSwingState, angularLimitImpulse) == 176);
static_assert(offsetof(JointSwingState, angularMotorImpulse) == 192);
static_assert(offsetof(JointSwingState, angularMotorSetupX) == 208);
static_assert(offsetof(JointSwingState, angularMotorSetupY) == 224);
static_assert(offsetof(JointSwingState, angularMotorSetupZ) == 240);
static_assert(offsetof(JointReadback, orientation) == 16 && offsetof(JointReadback, status) == 32);
JPH_GPU_CHECK_TYPE(Aabb, 32);
JPH_GPU_CHECK_TYPE(SweepBody, 112);
JPH_GPU_CHECK_TYPE(ContactPatch, 16);
JPH_GPU_CHECK_TYPE(Contact, 80);
JPH_GPU_CHECK_TYPE(StepParams, 112);
JPH_GPU_CHECK_TYPE(StageWork, 16);
JPH_GPU_CHECK_TYPE(BodyMutationParams, 16);
JPH_GPU_CHECK_TYPE(BodyInitialization, 112);
JPH_GPU_CHECK_TYPE(CenterOfMassUpdate, 32);
JPH_GPU_CHECK_TYPE(BodyCommand, 64);
JPH_GPU_CHECK_TYPE(BodyCommandSpan, 16);
JPH_GPU_CHECK_TYPE(BodyCommandParams, 16);
JPH_GPU_CHECK_TYPE(QueryParams, 64);
static_assert(offsetof(QueryParams, selection) == 32 && offsetof(QueryParams, selectedCounts) == 48);
JPH_GPU_CHECK_TYPE(QueryInput, 144);
JPH_GPU_CHECK_TYPE(ColliderPairKey, 32);
JPH_GPU_CHECK_TYPE(QueryCastParams, 16);
JPH_GPU_CHECK_TYPE(QueryCastState, 64);
static_assert(offsetof(QueryCastState, progress) == 16 && offsetof(QueryCastState, normal) == 32 &&
              offsetof(QueryCastState, iteration) == 48);
JPH_GPU_CHECK_TYPE(QueryPrimitive, 80);
JPH_GPU_CHECK_TYPE(QueryBodyIdentity, 16);
JPH_GPU_CHECK_TYPE(QueryHit, 64);
JPH_GPU_CHECK_TYPE(QueryRecordWord, 16);
JPH_GPU_CHECK_TYPE(QuerySpan, 16);
JPH_GPU_CHECK_TYPE(QuerySummary, 16);
static_assert(offsetof(QueryPrimitive, geometryReference) == 64);
static_assert(offsetof(QueryInput, geometry) == 96 && offsetof(QueryInput, filter) == 16 &&
              offsetof(QueryInput, lifetime) == 112 && offsetof(QueryInput, geometryReference) == 128);
static_assert(offsetof(ColliderPairKey, a) == 0 && offsetof(ColliderPairKey, b) == 16);
static_assert(offsetof(QueryHit, positionDistance) == 16 && offsetof(QueryHit, pointADepth) == 48);
static_assert(offsetof(BodyCommand, value) == 16 && offsetof(BodyCommand, auxiliary) == 32 &&
              offsetof(BodyCommand, rootCOM) == 48);
static_assert(offsetof(KinematicTarget, positionSpeed) == 0 && offsetof(KinematicTarget, rotation) == 16 &&
              offsetof(KinematicTarget, rootCOMDt) == 32 && offsetof(KinematicTarget, state) == 48);
static_assert(offsetof(BodyInitialization, state) == 16 && offsetof(BodyInitialization, forces) == 80);
static_assert(offsetof(CenterOfMassUpdate, deltaLocalCOM) == 16);
static_assert(offsetof(BodyModel, meta) == 0 && offsetof(BodyModel, mass) == 16 &&
              offsetof(BodyModel, invInertiaDiagonal) == 32 && offsetof(BodyModel, inertiaRotation) == 48 &&
              offsetof(BodyModel, damping) == 64);
static_assert(offsetof(ShapeModel, meta) == 0 && offsetof(ShapeModel, geometry) == 16 &&
              offsetof(ShapeModel, localPositionCOM) == 32 && offsetof(ShapeModel, localRotation) == 48 &&
              offsetof(ShapeModel, material) == 64 && offsetof(ShapeModel, surface) == 80 &&
              offsetof(ShapeModel, source) == 96 && offsetof(ShapeModel, geometryReference) == 112);
static_assert(offsetof(WorldRange, bodies) == 0 && offsetof(WorldRange, leaves) == 16);
static_assert(offsetof(ContactPatch, leaves) == 0);
static_assert(offsetof(BodyState, rotation) == 16 && offsetof(BodyState, angularVelocity) == 48);
static_assert(offsetof(Contact, normalDepth) == 16 && offsetof(Contact, impulse) == 64);
static_assert(offsetof(StepParams, gravityDt) == 32 && offsetof(StepParams, restitution) == 64);
static_assert(offsetof(StepParams, selection) == 80 && offsetof(StepParams, selectedCounts) == 96);
static_assert(sizeof(BodyBatchHeader) == 16 && alignof(BodyBatchHeader) == 16);
static_assert(std::is_standard_layout<BodyBatchHeader>::value && std::is_trivially_copyable<BodyBatchHeader>::value);
static_assert(sizeof(BodyBatchTile) == 12 && alignof(BodyBatchTile) == 4);
static_assert(std::is_standard_layout<BodyBatchTile>::value && std::is_trivially_copyable<BodyBatchTile>::value);
static_assert(offsetof(BodyBatchTile, selectionIndex) == 0 && offsetof(BodyBatchTile, bodyFirst) == 4 &&
              offsetof(BodyBatchTile, bodyCount) == 8);
static_assert(offsetof(StageWork, range) == 0);
#undef JPH_GPU_CHECK_TYPE
}
} // namespace JPH::GPU
#endif

#undef JPH_GPU_ALIGN
#undef JPH_GPU_FLOAT4
#undef JPH_GPU_UINT4
#undef JPH_GPU_UINT
#undef JPH_GPU_CONSTANT
#ifndef __cplusplus
// Identity keys use unsigned ID64/generation64 ordering without device uint64 math.
bool jph_collider_id_less(uint2 a, uint2 b)
{
    return a.y < b.y || (a.y == b.y && a.x < b.x);
}
bool jph_collider_lifetime_less(uint4 a, uint4 b)
{
    return jph_collider_id_less(a.xy, b.xy) || (all(a.xy == b.xy) && jph_collider_id_less(a.zw, b.zw));
}
ColliderPairKey jph_collider_pair_key(uint4 a, uint4 b)
{
    ColliderPairKey key;
    bool reverse = jph_collider_lifetime_less(b, a);
    key.a = reverse ? b : a;
    key.b = reverse ? a : b;
    return key;
}
bool jph_collider_pair_less(ColliderPairKey a, ColliderPairKey b)
{
    return jph_collider_lifetime_less(a.a, b.a) || (all(a.a == b.a) && jph_collider_lifetime_less(a.b, b.b));
}
bool jph_collider_pair_equal(ColliderPairKey a, ColliderPairKey b)
{
    return all(a.a == b.a) && all(a.b == b.b);
}
#endif
#endif // JPH_GPU_PHYSICS_TYPES_H
