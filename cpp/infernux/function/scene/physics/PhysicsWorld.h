#pragma once

/**
 * @file PhysicsWorld.h
 * @brief GPU-only Jolt scene facade using the shared engine compute runtime.
 *
 * Manages Jolt initialisation/shutdown, stepping, body management,
 * and raycast queries. Stepped by SceneManager's fixed-time accumulator.
 */

#include "JoltGpuScene.h"
#include "PhysicsConstraintOwner.h"
#include "PhysicsTargetReference.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

// Forward declaration
namespace infernux
{
class FrameWorkOwner;
class Infernux;
class JoltGpuScene;
class JoltGpuQueries;
struct PreparedGpuActor;
struct ContactEvent;
namespace rhi
{
class ComputeHost;
class PreparedComputeBatch;
} // namespace rhi
} // namespace infernux

// Immutable author geometry is retained; no native CPU Body/System is owned.
namespace JPH
{
class Shape;
}

namespace infernux
{

class GameObject;
class Transform;
class Component;
class HingeJoint;
class Collider;
class BoxCollider;
class Rigidbody;
class Joint;
struct RigidbodyMassProperties;
struct SwingTwistJointSettings;
struct SixDOFJointSettings;
enum class JointSwingType;

struct PhysicsBodyPoseUpdate
{
    uint32_t bodyId = 0xFFFFFFFF;
    glm::vec3 position{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
};

/// Requested solver fields. Body identity is read under the same Jolt lock;
/// shape revision is actor metadata supplied by the owner-thread Rigidbody.
enum class PhysicsStateFields : uint32_t
{
    None = 0,
    Position = 1u << 0,
    CenterOfMass = 1u << 1,
    LinearVelocity = 1u << 2,
    AngularVelocity = 1u << 3,
    InverseMass = 1u << 4,
    Rotation = 1u << 5,
    InverseInertia = 1u << 6,
    BodyId = 1u << 7,
    ShapeRevision = 1u << 8,
    All = (1u << 7) - 1,
};

constexpr bool HasPhysicsStateField(PhysicsStateFields fields, PhysicsStateFields field)
{
    return (static_cast<uint32_t>(fields) & static_cast<uint32_t>(field)) != 0;
}

/// Solver state read under one body lock. Translation locks are world-axis masks.
struct PhysicsBodyMotionState
{
    uint32_t bodyId = 0xFFFFFFFF;
    uint64_t shapeRevision = 0;
    glm::vec3 position{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 centerOfMass{0.0f};
    glm::vec3 linearVelocity{0.0f};
    glm::vec3 angularVelocity{0.0f};
    glm::vec3 inverseMass{0.0f};
    glm::mat3 inverseInertia{0.0f};
};

struct PhysicsPenetrationResult
{
    glm::vec3 direction{0.0f}; ///< Unit direction that separates A from B.
    float distance = 0.0f;
    glm::vec3 pointA{0.0f};
    glm::vec3 pointB{0.0f};
};

/// One Box leaf from the currently installed Jolt body shape, in solver space.
struct PhysicsBoxShapeState
{
    uint32_t memberIndex = 0;
    glm::vec3 center{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 halfExtents{0.0f};
    float friction = 0.0f;
    float bounciness = 0.0f;
};

enum class PhysicsShapeType : uint32_t
{
    Box,
    Sphere,
    Capsule,
    Mesh
};

/// Immutable geometry from an installed Jolt leaf. Dimensions and triangles
/// already contain leaf scale; poses below are rigid transforms about its COM.
struct PhysicsShapeGeometry
{
    uint64_t identity = 0;
    PhysicsShapeType type = PhysicsShapeType::Mesh;
    glm::vec4 dimensions{0.0f};
    glm::vec3 boundsMin{0.0f};
    glm::vec3 boundsMax{0.0f};
    bool convex = false;
    [[nodiscard]] std::vector<glm::vec3> GetTriangles() const;

  private:
    friend class PhysicsWorld;
    std::shared_ptr<const JPH::Shape> shape;
    glm::vec3 scale{1.0f};
};

struct PhysicsShapeState
{
    Collider *collider = nullptr;
    uint32_t memberIndex = 0;
    uint32_t layerBit = 0;
    bool trigger = false;
    bool moving = false;
    std::shared_ptr<const PhysicsShapeGeometry> geometry;
    glm::vec3 center{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 bodyCenterOfMass{0.0f};
    glm::vec3 linearVelocity{0.0f};
    glm::vec3 angularVelocity{0.0f};
};

/// Actual velocity-solver impulse from the most recent fixed step. The
/// impulse is applied to body B (its negative to A). Each solver midpoint
/// retains its normal impulse and an equal share of manifold tangent friction.
/// Independent torsional-friction angular impulses are not included. Opt-in.
struct ContactImpulse
{
    uint32_t bodyIdA = 0xFFFFFFFF;
    uint32_t bodyIdB = 0xFFFFFFFF;
    uint32_t subShapeIdA = 0;
    uint32_t subShapeIdB = 0;
    glm::vec3 contactPoint{0.0f};
    glm::vec3 contactNormal{0.0f};
    glm::vec3 impulse{0.0f};
};

/**
 * @brief Result of a physics raycast (Unity: RaycastHit).
 */
struct RaycastHit
{
    glm::vec3 point{0.0f};               ///< World-space hit point
    glm::vec3 normal{0.0f};              ///< Surface normal at hit
    float distance = 0.0f;               ///< Distance from ray origin
    uint32_t bodyId = 0xFFFFFFFF;        ///< Hit Jolt body id (index+sequence)
    uint32_t subShapeId = 0;             ///< Jolt sub-shape identity inside the body
    uint32_t triangleIndex = 0xFFFFFFFF; ///< Cooked triangle index for non-convex mesh hits
    GameObject *gameObject = nullptr;    ///< Hit GameObject
    Collider *collider = nullptr;        ///< Hit Collider component
    PhysicsTargetReference target;       ///< Safe identity for retained query values
    // Stable numeric identities captured while the published query epoch
    // is held.  Batch bindings must use these rather than dereferencing the
    // raw convenience pointers after the native query boundary has ended.
    uint64_t gameObjectId = 0;
    uint64_t colliderId = 0;
};

/// Fixed-step wall time. Jolt includes its collision/solver jobs and their wait.
struct PhysicsStepProfile
{
    uint64_t prepareNs = 0;
    uint64_t ccdNs = 0;
    uint64_t joltNs = 0;
    uint64_t finishNs = 0;
};

/// Optional diagnostic counters for one RaycastBatch call. Timings inside
/// parallel workers are summed CPU time; dispatchWallNs is caller wall time.
/// A null profile pointer keeps the hot path free of clock reads.
struct RaycastBatchProfile
{
    uint64_t snapshotSyncNs = 0;
    uint64_t snapshotLockWaitNs = 0;
    uint64_t dispatchWallNs = 0;
    uint64_t joltQueryCpuNs = 0;
    uint64_t narrowPhaseCpuNs = 0;
    uint64_t sortFilterCpuNs = 0;
    uint64_t hitPublishCpuNs = 0;
    uint64_t broadphaseCandidates = 0;
    uint64_t narrowphaseHits = 0;
    uint64_t publishedHits = 0;
};

/**
 * @brief Singleton physics world wrapping Jolt Physics.
 */
class PhysicsWorld
{
  public:
    static PhysicsWorld &Instance();

    /// Initialise Jolt (call once after engine starts)
    void Initialize();

    /// Shut down Jolt (call on engine cleanup).
    ///
    /// Tear-down order is fixed:
    ///   1. ContactListener::ClearAll() — drop pair tracking before bodies die.
    ///   2. Body ownership and query publications are cleared.
    ///   3. ContactListener / PhysicsSystem / JobSystem / TempAllocator / Layers
    ///      destroyed in dependency order.
    ///   4. Jolt globals (Factory, registered types) torn down.
    /// Idempotent: subsequent calls are no-ops.
    void Shutdown();
    [[nodiscard]] size_t GetBodyCount() const;

    /// @return true if initialised
    [[nodiscard]] bool IsInitialized() const
    {
        return m_initialized;
    }

    /// Global gravity; authored configuration is available before initialization.
    [[nodiscard]] glm::vec3 GetGravity() const;
    void SetGravity(const glm::vec3 &gravity);

    /// Advance the simulation by one fixed step.
    /// True only when this admitted fixed step has completed and published.
    /// False retains the same epoch/step and its shared-source continuation.
    bool Step(float deltaTime, PhysicsStepProfile *profile = nullptr);
    void CompletePendingStep();
    /// Budgeted author-only GPU pose publication after the Transform frame cache.
    /// Retains the previous complete pose while this same-source control waits.
    void RefreshRenderPoses();
    /// Continue an existing managed control within this host's shared budget.
    /// Does not capture author input, admit a solver or wait synchronously.
    void ServiceGpuWork();
    /// Promote a complete immutable export after the fixed publication gate.
    /// Control-only callers promote immediately when no fixed gate is pending.
    [[nodiscard]] bool PublishRenderPoses() const;
    [[nodiscard]] bool HasPendingStep() const noexcept;
    [[nodiscard]] bool HasPendingControl() const noexcept;
    /// Whole-runtime Stop only; never cancel other loaded actors for a Scene swap.
    void CancelPendingStep() noexcept;
    /// Implemented at the CPU Transform access boundary; never returns a Future.
    void EnsureCpuPhysicsPose(Transform &transform);
    /// Transfer a surviving object's final pose before detaching its Rigidbody.
    /// Destruction of the whole object does not require this CPU consumer.
    void MaterializeCpuPoseForRelease(Transform &transform);

    /// Bodies that were active immediately before or after the latest step.
    /// The union includes bodies that entered sleep during the step so their
    /// final solver pose is still read back exactly once.
    [[nodiscard]] const std::vector<uint32_t> &GetPoseReadbackBodyIds() const;
    [[nodiscard]] size_t GetLastDynamicCCDSplitCount() const
    {
        return 0;
    }

    /// Enable the engine contact stream used by custom solvers and tooling.
    /// When enabled, resolved collision/trigger events remain available after
    /// the fixed step through GetContactEvents(). Callback dispatch is still
    /// independent and follows the normal component interest mask.
    void SetContactEventStreamEnabled(bool enabled, bool includeTriggers = false);

    [[nodiscard]] bool IsContactEventStreamEnabled() const noexcept
    {
        return m_contactEventStreamEnabled;
    }

    /// Resolved events from the most recent completed fixed step. The buffer
    /// is replaced at the next step; callers must consume it before then.
    [[nodiscard]] const std::vector<ContactEvent> &GetContactEvents() const;

    void SetContactImpulseStreamEnabled(bool enabled);
    [[nodiscard]] bool IsContactImpulseStreamEnabled() const noexcept
    {
        return m_contactImpulseStreamEnabled;
    }
    [[nodiscard]] const std::vector<ContactImpulse> &GetContactImpulses() const;

    // ========================================================================
    // Body management (published by PhysicsECSStore actors)
    // ========================================================================

    /// Create one actor body from its currently enabled collision shapes.
    uint32_t CreateBody(struct PhysicsActorData &actor);

    /// Retire an actor after its queued residency commands have been canceled
    /// and its resident body removed from Jolt's broadphase.
    void DestroyBody(uint32_t bodyId);
    /// Retire joint endpoints when their Rigidbody stops participating, even
    /// when its Collider keeps the same Jolt body as a static actor.
    void DestroyConstraintsForBody(uint32_t bodyId);

    /// Inform the physics world that a body has moved (kinematic / editor move).
    void SetBodyPosition(uint32_t bodyId, const glm::vec3 &pos, const glm::quat &rot, bool cancelKinematicMove = false);

    void SetBodyPositionOnly(uint32_t bodyId, const glm::vec3 &position);
    void SetBodyRotationOnly(uint32_t bodyId, const glm::quat &rotation);

    /// Update many static body poses with one broadphase notification.
    void SetBodyPositionsBatch(const std::vector<PhysicsBodyPoseUpdate> &updates);

    /// Episode reset is legal only on the owner between complete fixed steps,
    /// outside contact dispatch. Does not advance the world's clocks/epoch.
    void RequireResetPhase() const;
    /// Internal Rigidbody batch boundary; poses/velocities are prevalidated.
    void ResetBodyMotionStates(const std::vector<PhysicsBodyMotionState> &states);

    /// Notify that a body's shape or properties changed.
    void UpdateBodyShape(Collider *collider, const Collider *exclude = nullptr);
    void UpdateBodyShape(struct PhysicsActorData &actor, const Collider *exclude = nullptr);

    /// Update a body's sensor (trigger) flag at runtime without recreating the body.
    void RefreshBodyTriggerState(uint32_t bodyId);

    /// Clear cached contact-pair tracking for a body so the next physics step
    /// produces fresh Enter events.  Call after changing sensor flag at runtime.
    void InvalidateContactPairsForBody(uint32_t bodyId);

    /// Batch-add bodies to the broadphase using Jolt's AddBodiesPrepare/Finalize.
    /// Publish final motion state and queued forces once for each resident body.
    void AddBodiesBatch(const std::vector<uint32_t> &bodies);

    /// True only after publication to Jolt, not merely after an add was queued.
    [[nodiscard]] bool IsBodyInBroadphase(uint32_t bodyId) const;

    /// Remove a body from the broadphase (body stays alive for re-adding later).
    void RemoveBodyFromBroadphase(uint32_t bodyId, bool preserveVelocity = false);

    // ========================================================================
    // Body dynamics — used by Rigidbody component
    // ========================================================================

    /// Switch a body between Static, Dynamic, Kinematic.
    /// motionType: 0 = Static, 1 = Kinematic, 2 = Dynamic.
    void SetBodyMotionType(uint32_t bodyId, int motionType);

    /// Update a body's user layer while preserving whether it is moving/static.
    void RefreshBodyLayers(uint32_t bodyId);

    /// Set linear / angular damping (drag).
    void SetBodyDamping(uint32_t bodyId, float linearDamping, float angularDamping);

    /// Override per-body gravity factor (0 = no gravity, 1 = normal).
    void SetBodyGravityFactor(uint32_t bodyId, float factor);

    /// Set per-body friction coefficient [0..1].
    void SetBodyFriction(uint32_t bodyId, float friction);

    /// Set per-body restitution (bounciness) [0..1].
    void SetBodyRestitution(uint32_t bodyId, float restitution);

    // ---- Velocity ----

    [[nodiscard]] glm::vec3 GetBodyLinearVelocity(uint32_t bodyId) const;
    void SetBodyLinearVelocity(uint32_t bodyId, const glm::vec3 &vel);

    [[nodiscard]] glm::vec3 GetBodyAngularVelocity(uint32_t bodyId) const;
    void SetBodyAngularVelocity(uint32_t bodyId, const glm::vec3 &vel);

    // ---- Forces ----

    void AddBodyForce(uint32_t bodyId, const glm::vec3 &force);
    void AddBodyImpulse(uint32_t bodyId, const glm::vec3 &impulse);
    void AddBodyTorque(uint32_t bodyId, const glm::vec3 &torque);
    void AddBodyAngularImpulse(uint32_t bodyId, const glm::vec3 &impulse);

    /// Raw component command: kind 0=Force, 1=Torque, 2=ForceAtPosition;
    /// mode is the existing ForceMode 0..3, interpreted on the device.
    void QueueBodyForce(uint32_t bodyId, int kind, int mode, const glm::vec3 &value, const glm::vec3 &point = {});

    // ---- Forces at position ----

    void AddBodyForceAtPosition(uint32_t bodyId, const glm::vec3 &force, const glm::vec3 &point);
    void AddBodyImpulseAtPosition(uint32_t bodyId, const glm::vec3 &impulse, const glm::vec3 &point);

    // ---- Constraints / Motion quality ----

    /// Set allowed degrees-of-freedom (Jolt EAllowedDOFs bitmask). Recalculates mass for the new DOFs.
    void SetBodyAllowedDOFs(uint32_t bodyId, int allowedDOFs, float mass);
    /// Pure candidate validation before a component publishes mass properties.
    void ValidateBodyMassProperties(const Rigidbody &body, float mass, const RigidbodyMassProperties *properties) const;

    /// Set collision mode: 0 = Discrete, 1 = Continuous static sweep,
    /// 2 = ContinuousDynamic with dynamic-pair TOI subdivision.
    void SetBodyMotionQuality(uint32_t bodyId, int quality);

    /// Set max angular velocity (rad/s).
    void SetBodyMaxAngularVelocity(uint32_t bodyId, float maxVel);

    /// Set max linear velocity (m/s).
    void SetBodyMaxLinearVelocity(uint32_t bodyId, float maxVel);

    /// Create a world-space hinge between body A and body B, or body A and
    /// the fixed world when body B is invalid. Limits are radians.
    uint64_t CreateHingeConstraint(PhysicsConstraintOwner &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                   const glm::vec3 &worldAnchor, const glm::vec3 &worldAxis,
                                   const glm::vec3 &localAnchor, bool useLimits, float minimumAngle, float maximumAngle,
                                   bool enableCollision);
    /// Create a prismatic constraint that permits only translation along one
    /// world-space axis. Limits are metres relative to the creation pose.
    uint64_t CreateSliderConstraint(PhysicsConstraintOwner &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                    const glm::vec3 &worldAnchor, const glm::vec3 &worldAxis,
                                    const glm::vec3 &localAnchor, bool useLimits, float minimumDistance,
                                    float maximumDistance, bool enableCollision);
    void DestroyConstraint(uint64_t constraintId);
    uint64_t CreateFixedConstraint(PhysicsConstraintOwner &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                   const glm::vec3 &worldAnchor, const glm::vec3 &localAnchor, bool enableCollision);
    void SetConstraintCollisionEnabled(uint64_t constraintId, bool enabled);
    uint64_t CreateDistanceConstraint(PhysicsConstraintOwner &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                      const glm::vec3 &localAnchor, const glm::vec3 &connectedLocalAnchor,
                                      bool autoDistance, float minimum, float maximum, float stiffness, float damping,
                                      bool enableCollision);
    void UpdateDistanceConstraintSettings(uint64_t constraintId, bool updateLimits, float minimum, float maximum,
                                          float stiffness, float damping, bool enableCollision);
    float GetDistanceConstraintLength(uint64_t constraintId) const;
    uint64_t CreateSwingTwistConstraint(PhysicsConstraintOwner &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                        const glm::vec3 &localAnchor, const SwingTwistJointSettings &settings,
                                        bool enableCollision);
    void UpdateSwingTwistConstraint(uint64_t constraintId, const SwingTwistJointSettings &settings,
                                    bool enableCollision);
    void SetSwingTwistConstraintType(uint64_t constraintId, JointSwingType type);
    glm::quat GetAngularConstraintOrientation(uint64_t constraintId) const;
    uint64_t CreateSixDOFConstraint(PhysicsConstraintOwner &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                    const glm::vec3 &localAnchor, const SixDOFJointSettings &settings,
                                    bool enableCollision);
    void UpdateSixDOFConstraint(uint64_t constraintId, const SixDOFJointSettings &settings, bool enableCollision);
    glm::vec3 GetSixDOFConstraintPosition(uint64_t constraintId) const;
    void UpdateHingeConstraintSettings(uint64_t constraintId, bool useLimits, float minimum, float maximum,
                                       bool enableCollision);
    /// Position motor: radians, stiffness N m/rad, damping N m s/rad, torque N m.
    void UpdateHingeConstraintMotor(uint64_t constraintId, bool enabled, float targetAngle, float stiffness,
                                    float damping, float maxTorque, float jointDamping, float frictionTorque,
                                    float armature);
    void UpdateHingeConstraintTarget(uint64_t constraintId, float targetAngle);
    void UpdateSliderConstraintSettings(uint64_t constraintId, bool useLimits, float minimum, float maximum,
                                        bool enableCollision);
    [[nodiscard]] float GetHingeConstraintAngle(uint64_t constraintId) const;
    [[nodiscard]] float GetHingeConstraintAngularVelocity(uint64_t constraintId) const;
    [[nodiscard]] glm::vec2 GetHingeConstraintMotionState(uint64_t constraintId) const;
    /// Scene/constraint identities may only be traversed on the physics owner.
    void RequireOwnerThread() const;
    [[nodiscard]] float GetSliderConstraintPosition(uint64_t constraintId) const;

    // ---- Kinematic move ----

    /// Speed cap for transform-driven body moves (gizmo drags, kinematic
    /// transform writes). Matches PhysX's default maxDepenetrationVelocity.
    static constexpr float kMaxTransformDriveSpeed = 10.0f;

    /// Move a kinematic body towards target position/rotation over deltaTime.
    ///
    /// Jolt implements this by giving the body the velocity needed to reach
    /// the target during the next step — and that velocity persists after
    /// arrival. Bodies moved through here are therefore tracked, and any body
    /// that does not receive a new target before the next Step() has its
    /// velocity zeroed so it stops exactly at the target instead of gliding
    /// away from its Transform forever.
    ///
    /// With maxSpeed > 0 the velocity the move can impart is capped: excess
    /// displacement is applied as a teleport, and only the final stretch is
    /// driven with velocity. Transform writes use this so long-range scripted
    /// teleports stay teleports while gizmo drags carry momentum. Script APIs
    /// (Rigidbody::MovePosition) pass 0 — uncapped, like Unity.
    void MoveBodyKinematic(uint32_t bodyId, const glm::vec3 &targetPos, const glm::quat &targetRot, float deltaTime,
                           float maxSpeed = 0.0f);

    /// Compose partial script targets against the pose already queued for this step.
    void MoveBodyKinematicPosition(uint32_t bodyId, const glm::vec3 &targetPos, float deltaTime);
    void MoveBodyKinematicRotation(uint32_t bodyId, const glm::quat &targetRot, float deltaTime);

    /// Move a collider-only (static) body to a new pose with real velocity so
    /// overlapping dynamic bodies receive momentum (Unity-like drag push).
    ///
    /// Teleporting a static body only produces positional depenetration in
    /// Jolt: dynamic bodies are squeezed out with zero exit velocity and stop
    /// dead. This call temporarily switches the body to Kinematic and drives
    /// it with MoveKinematic; once move commands stop arriving the body is
    /// stopped and restored to Static (see SettleKinematicMoves).
    void MoveStaticBodyWithVelocity(uint32_t bodyId, const glm::vec3 &targetPos, const glm::quat &targetRot,
                                    float deltaTime);

    // ---- Sleep ----

    [[nodiscard]] bool IsBodySleeping(uint32_t bodyId) const;
    [[nodiscard]] bool IsBodySensor(uint32_t bodyId) const;
    void ActivateBody(uint32_t bodyId);
    void DeactivateBody(uint32_t bodyId);

    /// Wake all dynamic bodies whose AABBs overlap the given world-space box.
    /// Used after moving a static collider to wake sleeping bodies that were resting on it.
    void ActivateBodiesInAABB(const glm::vec3 &min, const glm::vec3 &max);

    /// Wake dynamic bodies overlapping a specific body's world-space AABB.
    /// Convenience wrapper for use after moving a static collider.
    void WakeBodiesTouchingStatic(uint32_t bodyId);

    // ---- Read-back (physics → transform) ----

    [[nodiscard]] glm::vec3 GetBodyPosition(uint32_t bodyId) const;
    [[nodiscard]] glm::quat GetBodyRotation(uint32_t bodyId) const;
    [[nodiscard]] glm::vec3 GetBodyCenterOfMassPosition(uint32_t bodyId) const;
    [[nodiscard]] PhysicsBodyMotionState GetBodyMotionState(uint32_t bodyId,
                                                            PhysicsStateFields fields = PhysicsStateFields::All) const;
    [[nodiscard]] std::optional<PhysicsPenetrationResult>
    ComputePenetration(const Collider &a, const glm::vec3 &positionA, const glm::quat &rotationA, const Collider &b,
                       const glm::vec3 &positionB, const glm::quat &rotationB) const;

    /// Get the world-space inertia tensor on the body's allowed angular subspace.
    /// Frozen axes and invalid/static bodies produce zero rows and columns.
    [[nodiscard]] glm::mat3 GetBodyWorldSpaceInertiaTensor(uint32_t bodyId) const;

    // ========================================================================
    // Raycast API (Unity: Physics.Raycast)
    // ========================================================================

    /// Resolve omitted query masks at the call boundary, after project configuration.
    [[nodiscard]] static uint32_t GetDefaultQueryLayerMask();

    /// Cast a ray and return the closest hit.  Returns true if hit.
    bool Raycast(const glm::vec3 &origin, const glm::vec3 &direction, float maxDistance, RaycastHit &outHit,
                 uint32_t layerMask = GetDefaultQueryLayerMask(), bool queryTriggers = true) const;

    /// Cast a contiguous batch of XYZ float rays against one stable world epoch.
    /// The engine owner publishes pending authoring state, dispatches GPU ray
    /// lanes and decodes one immutable publication. No CPU world query runs.
    /// Every input row produces one mask entry and one initialized result row;
    /// caller-owned storage must contain @p count elements.
    void RaycastBatch(const float *originsXYZ, const float *directionsXYZ, size_t count, float maxDistance,
                      RaycastHit *outHits, uint8_t *outHitMask, uint32_t layerMask = GetDefaultQueryLayerMask(),
                      bool queryTriggers = true, uint64_t *outQueryGeneration = nullptr,
                      RaycastBatchProfile *outProfile = nullptr) const;

    /// Publish pending authored collider state on the engine owner thread.
    /// @return true when the synchronization boundary has completed.
    bool PrepareRaycastBatchQuery() const;

    /// Dispatch against a complete publication on the engine owner thread.
    /// Results retain the publication identity used by their GPU computation.
    void RaycastBatchPublished(const float *originsXYZ, const float *directionsXYZ, size_t count, float maxDistance,
                               RaycastHit *outHits, uint8_t *outHitMask,
                               uint32_t layerMask = GetDefaultQueryLayerMask(), bool queryTriggers = true,
                               uint64_t *outQueryGeneration = nullptr, RaycastBatchProfile *outProfile = nullptr,
                               bool directionsNormalized = false) const;

    /// Monotonic identity of the currently published query world. The value
    /// changes whenever body membership, pose, layer/trigger state, or a
    /// collider shape changes. Query consumers can retain this token with
    /// their results instead of inferring validity from object counts.
    [[nodiscard]] uint64_t GetQueryGeneration() const noexcept
    {
        return m_queryGeneration.load(std::memory_order_acquire);
    }

    /// Cast a ray against one authored Collider sub-shape. Unlike a world
    /// query, this intentionally ignores layer, trigger and pair filters.
    bool RaycastCollider(const Collider &collider, const glm::vec3 &origin, const glm::vec3 &direction,
                         float maxDistance, RaycastHit &outHit) const;

    /// Return the closest world-space point on one primitive Collider. A point
    /// already inside the collider is returned unchanged.
    [[nodiscard]] glm::vec3 ClosestPointOnCollider(const Collider &collider, const glm::vec3 &point) const;

    /// Cast a ray and return all hits.
    std::vector<RaycastHit> RaycastAll(const glm::vec3 &origin, const glm::vec3 &direction, float maxDistance,
                                       uint32_t layerMask = GetDefaultQueryLayerMask(),
                                       bool queryTriggers = true) const;

    // ========================================================================
    // Overlap queries (Unity: Physics.OverlapSphere / OverlapBox)
    // ========================================================================

    /// Find all Colliders within a sphere. Returns list of Collider*.
    std::vector<Collider *> OverlapSphere(const glm::vec3 &center, float radius,
                                          uint32_t layerMask = GetDefaultQueryLayerMask(),
                                          bool queryTriggers = true) const;

    /// Find all Colliders within an oriented box.
    std::vector<Collider *> OverlapBox(const glm::vec3 &center, const glm::vec3 &halfExtents,
                                       const glm::quat &orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
                                       uint32_t layerMask = GetDefaultQueryLayerMask(),
                                       bool queryTriggers = true) const;

    /// Find all Colliders within a capsule defined by world-space segment endpoints.
    std::vector<Collider *> OverlapCapsule(const glm::vec3 &point0, const glm::vec3 &point1, float radius,
                                           uint32_t layerMask = GetDefaultQueryLayerMask(),
                                           bool queryTriggers = true) const;

    /// Return Rigidbody components whose resident Jolt body bounds intersect
    /// the supplied world-space AABB. This is a broad-phase candidate query:
    /// callers perform their own exact contact test against the body shape.
    std::vector<Rigidbody *> QueryRigidbodiesInBounds(const glm::vec3 &minimum, const glm::vec3 &maximum,
                                                      uint32_t layerMask = GetDefaultQueryLayerMask(),
                                                      bool queryTriggers = false) const;

    // ========================================================================
    // Shape cast queries (Unity: Physics.SphereCast / BoxCast)
    // ========================================================================

    /// Cast a sphere along a direction. Returns closest RaycastHit or empty.
    bool SphereCast(const glm::vec3 &origin, float radius, const glm::vec3 &direction, float maxDistance,
                    RaycastHit &outHit, uint32_t layerMask = GetDefaultQueryLayerMask(),
                    bool queryTriggers = true) const;

    /// Cast a box along a direction. Returns closest RaycastHit or empty.
    bool BoxCast(const glm::vec3 &center, const glm::vec3 &halfExtents, const glm::vec3 &direction,
                 const glm::quat &orientation, float maxDistance, RaycastHit &outHit,
                 uint32_t layerMask = GetDefaultQueryLayerMask(), bool queryTriggers = true) const;

    /// Sweep a BoxCollider at a proposed world pose, reporting the first blocking contact.
    bool SweepBox(const BoxCollider &box, const glm::vec3 &position, const glm::quat &rotation,
                  const glm::vec3 &displacement, RaycastHit &outHit) const;

    /// Cast a capsule defined by world-space segment endpoints.
    bool CapsuleCast(const glm::vec3 &point0, const glm::vec3 &point1, float radius, const glm::vec3 &direction,
                     float maxDistance, RaycastHit &outHit, uint32_t layerMask = GetDefaultQueryLayerMask(),
                     bool queryTriggers = true) const;

    // ========================================================================
    // Lookup
    // ========================================================================

    /// Find the first published Collider member, or nullptr for an empty body.
    Collider *FindColliderByBodyId(uint32_t bodyId) const;
    Rigidbody *FindRigidbodyByBodyId(uint32_t bodyId) const;

    /// A Jolt Ragdoll view over the existing component-owned bodies and joints.
    /// Returns zero while cooking/body/constraint publication is incomplete.
    uint64_t CreateRagdollAssembly(const std::vector<Rigidbody *> &bodies, const std::vector<Joint *> &parentJoints,
                                   const std::vector<int> &parents);
    void DestroyRagdollAssembly(uint64_t id);
    [[nodiscard]] bool HasRagdollAssembly(uint64_t id) const;
    void DriveRagdollPose(uint64_t id, const std::vector<glm::quat> &worldRotations, bool enableMotors = true);
    [[nodiscard]] std::vector<PhysicsBoxShapeState> GetBodyBoxShapeStates(uint32_t bodyId, bool queryTriggers) const;
    [[nodiscard]] std::vector<PhysicsShapeState> GetBodyShapeStates(uint32_t bodyId, bool queryTriggers = true) const;
    /// Resolve through the accepted immutable slot identity, never CPU Body data.
    Collider *ResolveColliderForSubShape(uint32_t bodyId, uint32_t subShapeIdValue) const;

    /// Runtime-only collision policy for one exact Collider pair. This is
    /// separate from joint-owned whole-body suppression.
    void SetColliderPairIgnored(Collider *colliderA, Collider *colliderB, bool ignored);
    [[nodiscard]] bool GetColliderPairIgnored(const Collider *colliderA, const Collider *colliderB) const;
    void RemoveIgnoredPairsForCollider(const Collider *collider);

    /// Ensure all Colliders in the given scene have registered bodies
    /// and their transforms are up to date. Call before editor-mode raycasts.
    void EnsureSceneBodiesRegistered(class Scene *scene);

    /// Rebuild the broad-phase tree so raycasts find newly added static bodies.
    void OptimizeBroadPhase();

    /// @brief Dispatch buffered contact events to Component callbacks.
    ///        Call once per fixed step, immediately after Step().
    [[nodiscard]] size_t DispatchContactEvents();

  private:
    friend class Infernux;
    /// Engine-owned services, borrowed until Shutdown; no external host lease.
    void BindGpuServices(std::unique_ptr<rhi::ComputeHost> host, FrameWorkOwner *work);
    friend class HingeJointBatch;
    void ReadHingeMotion(const std::vector<HingeJoint *> &joints, float *output) const;
    void SetHingeMotorTargets(const std::vector<HingeJoint *> &joints, const float *targets);

    PhysicsWorld() = default;
    ~PhysicsWorld();
    PhysicsWorld(const PhysicsWorld &) = delete;
    PhysicsWorld &operator=(const PhysicsWorld &) = delete;

    struct CompiledGpuPrograms;
    struct GpuRenderState;
    struct CachedMotionState
    {
        uint64_t generation = 0;
        PhysicsBodyMotionState state;
        bool hasInverseMass = false;
    };
    bool EnsureGpuReady();
    void RefreshGpuSurfaces();
    void RequireHealthyStep() const;
    void FailStep() noexcept;
    bool AdvanceFixedStep(bool synchronous);
    bool AdvanceGpuJob(bool synchronous) const;
    bool BeginGpuControl(bool numericalChange, const JoltGpuScene::Consumer &consumer = {}) const;
    bool AdvanceGpuControl(bool synchronous) const;
    void RetryCapacityBodyCreations();
    void RequireGpu() const;
    void FlushAuthoring() const;
    void DrainGpuSource() const;
    void ObserveGpuPublication(bool numericalChange) const;
    [[nodiscard]] std::function<std::shared_ptr<rhi::PreparedComputeBatch>(const JoltGpuScene::Publication &)>
    PrepareRenderConsumer(bool numericalChange) const;
    void RunControl(const std::function<std::shared_ptr<rhi::PreparedComputeBatch>(const JoltGpuScene::Publication &)>
                        &consumer) const;
    void RefreshGpuActor(uint32_t bodyId, const std::function<void(PreparedGpuActor &)> &edit = {});
    [[nodiscard]] PhysicsBodyMotionState ReadMotionState(uint32_t bodyId, PhysicsStateFields fields) const;
    void CheckContactSupport() const;

    bool m_initialized = false;
    bool m_ownsJoltGlobals = false;
    std::thread::id m_ownerThreadId{};
    std::unique_ptr<rhi::ComputeHost> m_gpuHost;
    FrameWorkOwner *m_gpuWork = nullptr;
    std::unique_ptr<CompiledGpuPrograms> m_gpuPrograms;
    std::unique_ptr<JoltGpuScene> m_gpuScene;
    std::unique_ptr<JoltGpuQueries> m_gpuQueries;
    std::unique_ptr<GpuRenderState> m_gpuRenderState;
    bool m_gpuCancelled = false;
    std::vector<PhysicsActorHandle> m_recreateActors;
    uint64_t m_surfaceRevision = 0;
    bool m_stepPending = false;
    bool m_stepCompleted = false;
    enum class FixedGpuPhase
    {
        Idle,
        RetireControl,
        RetryCreation,
        Solver,
        Failed
    };
    FixedGpuPhase m_fixedGpuPhase = FixedGpuPhase::Idle;
    bool m_waitingForSlot = false;
    mutable bool m_controlPending = false;
    mutable bool m_controlNumericalChange = false;
    float m_admittedDeltaTime = 0;
    std::optional<JoltGpuStepParameters> m_admittedParameters;
    uint64_t m_stepEpoch = 0;
    uint64_t m_stepIndex = 0;
    mutable bool m_flushingAuthoring = false;
    mutable bool m_synchronousRead = false;
    mutable std::atomic<uint64_t> m_queryGeneration{1};
    mutable std::shared_ptr<const void> m_lastPublication;
    mutable std::unordered_map<uint32_t, CachedMotionState> m_requestedMotion;
    bool m_contactEventStreamEnabled = false;
    bool m_contactImpulseStreamEnabled = false;
};

} // namespace infernux
