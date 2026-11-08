#pragma once

/**
 * @file PhysicsWorld.h
 * @brief Singleton physics world backed by Jolt Physics.
 *
 * Manages Jolt initialisation/shutdown, stepping, body management,
 * and raycast queries. Stepped by SceneManager's fixed-time accumulator.
 */

#include "PhysicsConstraintOwner.h"
#include "PhysicsTargetReference.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Forward declaration
namespace infernux
{
class InxContactListener;
struct ContactEvent;
} // namespace infernux

// Forward-declare Jolt types to avoid Jolt.h leak into every TU
namespace JPH
{
class PhysicsSystem;
class TempAllocatorImpl;
class Body;
class BodyID;
class Shape;
class Constraint;
class TwoBodyConstraintSettings;
} // namespace JPH

namespace infernux
{

class GameObject;
class Component;
class Collider;
class BoxCollider;
class Rigidbody;
class Joint;
struct RigidbodyMassProperties;
struct SwingTwistJointSettings;
struct SixDOFJointSettings;
enum class JointSwingType;
class InfernuxJoltJobSystemAdapter;

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
    [[nodiscard]] size_t GetBodyCount() const
    {
        std::shared_lock lock(m_querySnapshotMutex);
        return m_bodies.size();
    }

    /// @return true if initialised
    [[nodiscard]] bool IsInitialized() const
    {
        return m_initialized;
    }

    /// Advance the simulation by one fixed step.
    void Step(float deltaTime);

    /// Bodies that were active immediately before or after the latest step.
    /// The union includes bodies that entered sleep during the step so their
    /// final solver pose is still read back exactly once.
    [[nodiscard]] const std::vector<uint32_t> &GetPoseReadbackBodyIds() const
    {
        return m_poseReadbackBodyIds;
    }

    [[nodiscard]] size_t GetLastDynamicCCDSplitCount() const
    {
        return m_lastDynamicCCDSplitCount;
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
    [[nodiscard]] const std::vector<ContactImpulse> &GetContactImpulses() const
    {
        return m_contactImpulses;
    }

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
    /// The owner thread first publishes pending authoring state; worker callers
    /// consume the last complete published epoch. Large batches fan out through
    /// the engine JobSystem. Fixed-step and collider mutation publication wait
    /// until the batch releases its epoch.
    /// Every input row produces one mask entry and one initialized result row;
    /// caller-owned storage must contain @p count elements.
    void RaycastBatch(const float *originsXYZ, const float *directionsXYZ, size_t count, float maxDistance,
                      RaycastHit *outHits, uint8_t *outHitMask, uint32_t layerMask = GetDefaultQueryLayerMask(),
                      bool queryTriggers = true, uint64_t *outQueryGeneration = nullptr,
                      RaycastBatchProfile *outProfile = nullptr) const;

    /// Publish pending authored collider state only when called by the physics
    /// owner thread. Worker threads deliberately consume the last complete
    /// query epoch and never touch SceneManager-owned authoring state.
    /// @return true when owner-thread synchronization was performed.
    bool PrepareRaycastBatchQuery() const;

    /// Consume the last complete published query epoch without touching
    /// SceneManager. Call PrepareRaycastBatchQuery first on the owner thread;
    /// background workers call this method directly.
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
    [[nodiscard]] bool HasRagdollAssembly(uint64_t id) const
    {
        return m_ragdolls.count(id) != 0;
    }
    void DriveRagdollPose(uint64_t id, const std::vector<glm::quat> &worldRotations, bool enableMotors = true);
    [[nodiscard]] std::vector<PhysicsBoxShapeState> GetBodyBoxShapeStates(uint32_t bodyId, bool queryTriggers) const;
    [[nodiscard]] std::vector<PhysicsShapeState> GetBodyShapeStates(uint32_t bodyId, bool queryTriggers = true) const;
    [[nodiscard]] bool HasRetiredShapeMembers() const
    {
        return m_retiredShapeBodyCount != 0;
    }
    [[nodiscard]] bool AcceptsSubShapeQuery(const JPH::Body &body, uint32_t subShapeId, uint32_t layerMask) const;
    [[nodiscard]] bool AcceptsCCDCandidate(const JPH::Body &source, const JPH::Body &target, uint32_t sourceSubShape,
                                           uint32_t targetSubShape) const;

    /// Resolve a specific subshape hit/contact back to the owning Collider.
    Collider *ResolveColliderForSubShape(uint32_t bodyId, uint32_t subShapeIdValue) const;
    Collider *ResolveColliderForSubShape(const JPH::Body &body, uint32_t bodyId, uint32_t subShapeIdValue) const;

    /// Rebind a body lookup entry to another collider on the same body.

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

    /// Get the Jolt PhysicsSystem (for advanced usage). May be nullptr.
    JPH::PhysicsSystem *GetJoltSystem() const
    {
        return m_physicsSystem.get();
    }

  private:
    // Callers must hold m_querySnapshotMutex exclusively. Keeping the lock at
    // the public-operation boundary lets compound mutations publish as one
    // query epoch without recursively locking std::shared_mutex.
    void SetBodyPositionUnlocked(uint32_t bodyId, const glm::vec3 &pos, const glm::quat &rot,
                                 bool cancelKinematicMove = false);
    void MoveBodyKinematicUnlocked(uint32_t bodyId, const glm::vec3 &targetPos, const glm::quat &targetRot,
                                   float deltaTime, float maxSpeed);

    struct QueryTargetIdentity
    {
        Collider *collider = nullptr;
        GameObject *gameObject = nullptr;
        uint64_t colliderId = 0;
        uint64_t gameObjectId = 0;
        uint32_t memberIndex = UINT32_MAX;
        PhysicsTargetReference target;
        bool isTrigger = false;
        uint32_t layerBit = 0;
    };

    /// Rebuild immutable application identity/trigger data while holding the
    /// query epoch write lock. Query workers never dereference mutable
    /// Collider or GameObject state.
    void PublishBodyQueryIdentitiesUnlocked(uint32_t bodyId);
    [[nodiscard]] const QueryTargetIdentity *ResolvePublishedQueryIdentity(const JPH::Body &body, uint32_t bodyId,
                                                                           uint32_t subShapeIdValue) const;

    bool RaycastCurrent(const glm::vec3 &origin, const glm::vec3 &direction, float maxDistance, RaycastHit &outHit,
                        uint32_t layerMask, bool queryTriggers, bool queryEpochHeld = false,
                        RaycastBatchProfile *profile = nullptr, bool directionNormalized = false,
                        uint64_t queryGeneration = 0) const;

    PhysicsWorld() = default;
    ~PhysicsWorld();
    PhysicsWorld(const PhysicsWorld &) = delete;
    PhysicsWorld &operator=(const PhysicsWorld &) = delete;

    /// @brief Shared overlap implementation for OverlapSphere/OverlapBox.
    std::vector<Collider *> OverlapShapeImpl(const JPH::Shape &shape, const glm::vec3 &center,
                                             const glm::quat &orientation, uint32_t layerMask,
                                             bool queryTriggers) const;

    /// @brief Shared shape cast implementation for SphereCast/BoxCast.
    bool ShapeCastImpl(const JPH::Shape &shape, const glm::vec3 &origin, const glm::quat &orientation,
                       const glm::vec3 &direction, float maxDistance, RaycastHit &outHit, uint32_t layerMask,
                       bool queryTriggers, const Collider *movingCollider = nullptr) const;

    [[nodiscard]] float FindEarliestDynamicCCDFraction(float deltaTime) const;
    [[nodiscard]] float FindEarliestStaticCCDFraction(float deltaTime) const;
    void SetConstraintPairSuppressed(uint32_t bodyIdA, uint32_t bodyIdB, bool suppressed);

    /// Stop tracked kinematic-move bodies that did not receive a new target
    /// since the previous Step(), and restore temporarily-kinematic statics.
    /// Called at the start of every Step().
    void SettleKinematicMoves();

    bool m_initialized = false;

    std::unique_ptr<JPH::TempAllocatorImpl> m_tempAllocator;
    std::unique_ptr<InfernuxJoltJobSystemAdapter> m_jobSystem;
    std::unique_ptr<JPH::PhysicsSystem> m_physicsSystem;

    // Layer interfaces (must outlive PhysicsSystem)
    struct LayerInterfaces;
    std::unique_ptr<LayerInterfaces> m_layers;

    // One owner and immutable shape-identity publication per native body.
    // All Collider members share the Rigidbody actor and its query record.
    struct BodyRecord
    {
        GameObject *gameObject = nullptr;
        Rigidbody *rigidbody = nullptr;
        bool hasRetiredShapeMembers = false;
        std::vector<Collider *> members;
        std::vector<QueryTargetIdentity> targets;
        struct ShapeLeaf
        {
            uint32_t subShapeId = 0;
            glm::vec3 center{0.0f};
            glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
            std::shared_ptr<const PhysicsShapeGeometry> geometry;
        };
        std::vector<ShapeLeaf> shapeLeaves;
    };
    void PublishBodyShapeGeometryUnlocked(uint32_t bodyId);
    uint64_t m_nextShapeGeometryIdentity = 0;
    void SetBodyActorOwner(uint32_t bodyId, const PhysicsActorData &actor);
    std::unordered_map<uint32_t, BodyRecord> m_bodies;
    struct RagdollAssembly;
    std::unordered_map<uint64_t, std::shared_ptr<RagdollAssembly>> m_ragdolls;
    uint64_t m_nextRagdollId = 1;
    void InvalidateRagdollsForBody(uint32_t bodyId);
    void InvalidateRagdollsForConstraint(const JPH::Constraint *constraint);
    void UpdateRagdollCollision(const JPH::Constraint *constraint, bool enabled);
    size_t m_retiredShapeBodyCount = 0;
    // Queries acquire the read side after SceneManager has published pending
    // transforms.  Fixed steps and every operation that changes query-visible
    // body membership, pose, layer, sensor or shape acquire the write side.
    // This turns query_generation into an epoch boundary rather than a
    // best-effort before/after race detector.
    mutable std::shared_mutex m_querySnapshotMutex;
    std::atomic<uint64_t> m_queryGeneration{1};
    std::thread::id m_ownerThreadId{};

    struct ConstraintRecord
    {
        PhysicsConstraintOwner &owner;
        JPH::Constraint *constraint = nullptr;
        uint32_t bodyIdA = 0xFFFFFFFF;
        uint32_t bodyIdB = 0xFFFFFFFF;
        bool ignoresCollision = false;
        glm::vec3 anchorA{0.0f};
        glm::vec3 anchorB{0.0f};
        glm::vec3 collapsedOffsetA{0.0f};
        glm::vec3 collapsedOffsetB{0.0f};
        glm::vec3 scaleA{1.0f};
        glm::vec3 scaleB{1.0f};
    };
    void CaptureConstraintAnchors(ConstraintRecord &record, const glm::vec3 &localAnchor,
                                  const glm::vec3 &connectedLocalAnchor);
    uint64_t CreateTwoBodyConstraint(PhysicsConstraintOwner &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                     const JPH::TwoBodyConstraintSettings &settings, const glm::vec3 &localAnchor,
                                     bool enableCollision, const glm::vec3 &connectedLocalAnchor = glm::vec3(0.0f));
    void UpdateConstraintScale(ConstraintRecord &record, uint32_t bodyId, const glm::vec3 &scale);
    void ActivateConstraintBodies(const ConstraintRecord &record);
    void ReplaceConstraint(ConstraintRecord &record, JPH::Constraint *replacement);
    void UpdateConstraintCollision(ConstraintRecord &record, bool enableCollision);
    std::unordered_map<uint64_t, ConstraintRecord> m_constraints;
    uint64_t m_nextConstraintId = 1;

    // Dense active-body union produced by the latest completed Step().
    std::vector<uint32_t> m_poseReadbackBodyIds;

    // Full Jolt IDs explicitly configured for each CCD contract. Plain
    // Continuous uses an engine-owned static-only sweep; ContinuousDynamic
    // keeps Jolt LinearCast and adds relative dynamic-pair subdivision.
    std::unordered_set<uint32_t> m_staticContinuousBodyIds;
    std::unordered_set<uint32_t> m_continuousBodyIds;
    size_t m_lastDynamicCCDSplitCount = 0;
    float m_previousSolverStepDuration = 0.0f;

    /// Bodies whose velocity was authored by a MoveKinematic-style target
    /// move. See MoveBodyKinematic / MoveStaticBodyWithVelocity.
    struct KinematicMoveState
    {
        glm::vec3 targetPosition{0.0f};
        glm::quat targetRotation{1.0f, 0.0f, 0.0f, 0.0f};
        bool movedThisStep = true;  ///< Received a target since the last Step().
        bool restoreStatic = false; ///< Body is a collider-only static, temporarily kinematic.
        int idleSteps = 0;          ///< Steps elapsed without a new target.
    };
    std::unordered_map<uint32_t, KinematicMoveState> m_kinematicMoveStates;

    // Contact listener for collision/trigger callbacks
    std::unique_ptr<InxContactListener> m_contactListener;
    bool m_contactEventStreamEnabled = false;
    bool m_dispatchingContacts = false;
    bool m_contactEventStreamIncludeTriggers = false;
    bool m_contactImpulseStreamEnabled = false;
    std::vector<ContactImpulse> m_contactImpulses;
};

} // namespace infernux
