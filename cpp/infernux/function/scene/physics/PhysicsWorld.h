#pragma once

/**
 * @file PhysicsWorld.h
 * @brief Shared physics world, body ownership and execution facade.
 *
 * Manages Jolt initialisation/shutdown, stepping, body management,
 * and raycast queries. Stepped by SceneManager's fixed-time accumulator.
 */

#include "JoltCpuQueries.h"
#include "JoltGpuQueries.h"
#include "JoltGpuScene.h"
#include "PhysicsBodyState.h"
#include "PhysicsConstraintOwner.h"
#include "PhysicsContactPublication.h"
#include "PhysicsContactTypes.h"
#include "PhysicsParticipant.h"
#include "PhysicsTargetReference.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <map>
#include <memory>
#include <optional>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Forward declaration
namespace infernux
{
class FrameWorkOwner;
class RuntimeWorld;
struct PhysicsDomain;
class JoltGpuRenderPoses;
class JoltGpuContactReports;
class JoltGpuMixedContacts;
class JoltMixedConstraintSolver;
class Infernux;
class JoltGpuScene;
class JoltGpuQueries;
class JoltCpuQueries;
class JoltGpuJoints;
struct JoltGpuQueryResult;
struct PreparedCpuActor;
struct PhysicsStaticBoundaryModel;
struct PreparedGpuActor;
namespace rhi
{
class ComputeHost;
class PreparedComputeBatch;
} // namespace rhi
} // namespace infernux

// Author geometry is shared; native bodies remain private to their backend.
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
class Ragdoll;
struct RigidbodyMassProperties;
struct SwingTwistJointSettings;
struct SixDOFJointSettings;
enum class JointSwingType;

class PhysicsPresentationPosePending final : public std::exception
{
  public:
    const char *what() const noexcept override
    {
        return "Presentation CPU pose dependency is not ready";
    }
};

class PhysicsPresentationPoseReadScope final
{
  public:
    PhysicsPresentationPoseReadScope();
    ~PhysicsPresentationPoseReadScope();
    PhysicsPresentationPoseReadScope(const PhysicsPresentationPoseReadScope &) = delete;
    PhysicsPresentationPoseReadScope &operator=(const PhysicsPresentationPoseReadScope &) = delete;

  private:
    PhysicsDomain *m_domain = nullptr;
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

struct PhysicsRigidbodyBoxState
{
    uint32_t bodyIndex = 0;
    PhysicsBoxShapeState shape;
};

/// One publication and input order for a whole state request. With GPU export
/// targets, states contains identity metadata only and boxes stays on-device.
struct PhysicsRigidbodyBatchSource
{
    uint64_t worldId = 0, simulationEpoch = 0, fixedStepIndex = 0, numericalGeneration = 0;
};
struct PhysicsRigidbodyStateBatch
{
    std::vector<Rigidbody *> bodies;
    std::vector<PhysicsBodyMotionState> states;
    std::vector<PhysicsRigidbodyBoxState> boxes;
    size_t boxCount = 0;
    std::vector<PhysicsRigidbodyBatchSource> sources;
    std::vector<uint32_t> sourceForBody;
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
    friend class JoltCpuQueries;
    std::shared_ptr<const JPH::Shape> shape;
    glm::vec3 scale{1.0f};
    std::optional<std::array<glm::vec3, 3>> triangle;
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
    float friction = 0.0f;
    float bounciness = 0.0f;
    uint32_t frictionCombine = 0;
    uint32_t bounceCombine = 0;
};

/**
 * @brief Result of a physics raycast (Unity: RaycastHit).
 */
struct RaycastHit
{
    glm::vec3 point{0.0f};               ///< World-space hit point
    glm::vec3 normal{0.0f};              ///< Surface normal at hit
    float distance = 0.0f;               ///< Distance from ray origin
    uint32_t bodyId = 0xFFFFFFFF;        ///< Stable engine body identity
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
    friend class PhysicsPresentationPoseReadScope;

  public:
    static PhysicsWorld &Instance();
    void BeginRetireDomain(RuntimeWorld &world) noexcept;
    [[nodiscard]] bool CanRetireDomain(const RuntimeWorld &world) const noexcept;
    void RetireDomain(PhysicsDomain &domain) noexcept;

    /// Initialise Jolt (call once after engine starts)
    /// Native hosts supply device services before initialization. These are
    /// host-owned facilities, not part of the Python Rigidbody interface.
    void BindGpuServices(std::unique_ptr<rhi::ComputeHost> host, FrameWorkOwner *work);
    void Initialize();

    /// Shut down Jolt (call on engine cleanup).
    ///
    /// Tear-down order is fixed:
    ///   1. ContactListener::ClearAll() — drop pair tracking before bodies die.
    ///   2. Body ownership and query publications are cleared.
    ///   3. GPU scene resources and their host services are released.
    /// Geometry authoring types outlive numerical execution and remain usable
    /// before Play and between engine instances.
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
    void SetRigidbodyBackend(const std::vector<Rigidbody *> &bodies, bool gpu);
    std::shared_ptr<PhysicsParticipant> RegisterParticipant(const Component &component,
                                                            PhysicsParticipant::Recipes recipes,
                                                            PhysicsParticipant::Controls controls, uint32_t count,
                                                            const std::vector<Rigidbody *> &rigidbodies,
                                                            const PhysicsParticipant::ExchangeStates &exchangeStates);
    /// Existing Host service / Tick source only; never waits for GPU completion.
    bool TryPrepareStartupPrograms();
    bool ServiceParticipantPreparation();
    bool ServiceParticipantOutputs();
    [[nodiscard]] bool HasActiveParticipants() const;
    void CompletePendingStep();
    /// Budgeted author-only GPU pose publication after the Transform frame cache.
    /// Retains the previous complete pose while this same-source control waits.
    void RefreshRenderPoses();
    /// Settle the admitted physical prefix and zero-dt authors without consuming Fixed.
    [[nodiscard]] bool ServicePausedAuthorPresentation();
    [[nodiscard]] bool ServicePresentationPoseReads(const std::vector<Transform *> &world,
                                                    const std::vector<Transform *> &local);
    [[nodiscard]] bool ServiceMixedRenderPoses();
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

    [[nodiscard]] bool IsContactEventStreamEnabled() const noexcept;

    /// Resolved events from the most recent completed fixed step. The buffer
    /// is replaced at the next step; callers must consume it before then.
    [[nodiscard]] const std::vector<ContactEvent> &GetContactEvents() const;

    void SetContactImpulseStreamEnabled(bool enabled);
    [[nodiscard]] bool IsContactImpulseStreamEnabled() const noexcept;
    [[nodiscard]] const std::vector<ContactImpulse> &GetContactImpulses() const;
    [[nodiscard]] ContactStreamIdentity GetContactStreamIdentity() const;

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
    void SetBodyPosition(uint32_t bodyId, const glm::vec3 &pos, const glm::quat &rot, bool cancelKinematicMove = true);

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

    /// Set max angular velocity (rad/s).
    void SetBodyMaxAngularVelocity(uint32_t bodyId, float maxVel);

    /// Set max linear velocity (m/s).
    void SetBodyMaxLinearVelocity(uint32_t bodyId, float maxVel);

    /// Create a world-space hinge between body A and body B, or body A and
    /// the fixed world when body B is invalid. Limits are radians.
    uint64_t CreateHingeConstraint(HingeJoint &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                   const glm::vec3 &localBakedAnchor, const glm::vec3 &localAnchor,
                                   const JoltJointImport *imported);
    /// Create a prismatic constraint that permits only translation along one
    /// captured owner-local axis. Limits are metres relative to the creation pose.
    uint64_t CreateSliderConstraint(PhysicsConstraintOwner &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                    const glm::vec3 &localBakedAnchor, const glm::vec3 &ownerAxis,
                                    const glm::vec3 &localAnchor, bool useLimits, float minimumDistance,
                                    float maximumDistance, bool enableCollision, const JoltJointImport *imported);
    void DestroyConstraint(uint64_t constraintId);
    uint64_t CreateFixedConstraint(PhysicsConstraintOwner &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                   const glm::vec3 &localBakedAnchor, const glm::vec3 &localAnchor,
                                   bool enableCollision, const JoltJointImport *imported);
    void SetConstraintCollisionEnabled(uint64_t constraintId, bool enabled);
    uint64_t CreateDistanceConstraint(PhysicsConstraintOwner &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                      const glm::vec3 &localBakedAnchor, const glm::vec3 &localAnchor,
                                      const glm::vec3 &connectedBakedAnchor, const glm::vec3 &connectedLocalAnchor,
                                      bool autoDistance, float minimum, float maximum, float stiffness, float damping,
                                      bool enableCollision, const JoltJointImport *imported);
    void UpdateDistanceConstraintSettings(uint64_t constraintId, bool updateLimits, float minimum, float maximum,
                                          float stiffness, float damping, bool enableCollision);
    float GetDistanceConstraintLength(uint64_t constraintId) const;
    uint64_t CreateSwingTwistConstraint(PhysicsConstraintOwner &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                        const glm::vec3 &localBakedAnchor, const glm::vec3 &localAnchor,
                                        const SwingTwistJointSettings &settings, bool enableCollision,
                                        const JoltJointImport *imported);
    void UpdateSwingTwistConstraint(uint64_t constraintId, const SwingTwistJointSettings &settings,
                                    bool enableCollision, bool claimTarget = false);
    glm::quat GetAngularConstraintOrientation(uint64_t constraintId) const;
    uint64_t CreateSixDOFConstraint(PhysicsConstraintOwner &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                    const glm::vec3 &localBakedAnchor, const glm::vec3 &localAnchor,
                                    const SixDOFJointSettings &settings, bool enableCollision,
                                    const JoltJointImport *imported);
    void UpdateSixDOFConstraint(uint64_t constraintId, const SixDOFJointSettings &settings, bool enableCollision,
                                bool claimTarget = false);
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

    /// Internal Transform authoring path. The returned source protects only
    /// the authored channels until their numerical publication is complete.
    PhysicsTargetSource MoveBodyKinematicAuthored(uint32_t bodyId, const glm::vec3 &targetPos,
                                                  const glm::quat &targetRot, uint32_t channels, float deltaTime,
                                                  float maxSpeed, const std::array<uint64_t, 2> &authoredRevision);

    /// Move a collider-only (static) body to a new pose with real velocity so
    /// overlapping dynamic bodies receive momentum (Unity-like drag push).
    ///
    /// The device gives this static author model temporary kinematic velocity
    /// for one logical tick. Its authored motion type stays Static; without a
    /// new target the following tick clears only the movement-owned velocity.
    void MoveStaticBodyWithVelocity(uint32_t bodyId, const glm::vec3 &targetPos, const glm::quat &targetRot,
                                    float deltaTime);

    // ---- Sleep ----

    [[nodiscard]] bool IsBodySleeping(uint32_t bodyId) const;
    void ActivateBody(uint32_t bodyId);
    void DeactivateBody(uint32_t bodyId);

    // ---- Read-back (physics → transform) ----

    [[nodiscard]] glm::vec3 GetBodyPosition(uint32_t bodyId) const;
    [[nodiscard]] glm::quat GetBodyRotation(uint32_t bodyId) const;
    [[nodiscard]] glm::vec3 GetBodyCenterOfMassPosition(uint32_t bodyId) const;
    [[nodiscard]] PhysicsBodyMotionState GetBodyMotionState(uint32_t bodyId,
                                                            PhysicsStateFields fields = PhysicsStateFields::All) const;
    [[nodiscard]] PhysicsRigidbodyStateBatch
    GatherRigidbodyStates(const std::vector<Rigidbody *> &bodies, PhysicsStateFields fields, bool includeBoxes,
                          bool queryTriggers, const std::vector<JoltGpuQueryExportBinding> &exports = {}) const;
    [[nodiscard]] PhysicsRigidbodyStateBatch
    GatherRigidbodyStatesInBounds(const glm::vec3 &minimum, const glm::vec3 &maximum, uint32_t layers,
                                  bool queryTriggers, PhysicsStateFields fields, bool includeBoxes,
                                  const std::vector<JoltGpuQueryExportBinding> &exports = {}) const;
    [[nodiscard]] std::optional<PhysicsPenetrationResult>
    ComputePenetration(const Collider &a, const glm::vec3 &positionA, const glm::quat &rotationA, const Collider &b,
                       const glm::vec3 &positionB, const glm::quat &rotationB) const;

    /// Get the world-space inertia tensor on the body's allowed angular subspace.
    /// Frozen axes and invalid/static bodies produce zero rows and columns.
    [[nodiscard]] glm::mat3 GetBodyWorldSpaceInertiaTensor(uint32_t bodyId) const;
    [[nodiscard]] glm::vec3 GetBodyPrincipalInertia(uint32_t bodyId) const;
    [[nodiscard]] glm::quat GetBodyPrincipalInertiaRotation(uint32_t bodyId) const;

    // ========================================================================
    // Raycast API (Unity: Physics.Raycast)
    // ========================================================================

    /// Resolve omitted query masks at the call boundary, after project configuration.
    [[nodiscard]] static uint32_t GetDefaultQueryLayerMask();

    /// Cast a ray and return the closest hit.  Returns true if hit.
    bool Raycast(const glm::vec3 &origin, const glm::vec3 &direction, float maxDistance, RaycastHit &outHit,
                 uint32_t layerMask = GetDefaultQueryLayerMask(), bool queryTriggers = true) const;

    /// Cast a contiguous batch of XYZ float rays against one stable world epoch.
    /// The engine owner publishes pending authoring state, queries each active
    /// backend and merges hits from the same completed world boundary.
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
    /// Results retain the common publication identity used by both backends.
    void RaycastBatchPublished(const float *originsXYZ, const float *directionsXYZ, size_t count, float maxDistance,
                               RaycastHit *outHits, uint8_t *outHitMask,
                               uint32_t layerMask = GetDefaultQueryLayerMask(), bool queryTriggers = true,
                               uint64_t *outQueryGeneration = nullptr, RaycastBatchProfile *outProfile = nullptr,
                               bool directionsNormalized = false) const;

    /// Monotonic identity of the currently published query world. The value
    /// changes whenever body membership, pose, layer/trigger state, or a
    /// collider shape changes. Query consumers can retain this token with
    /// their results instead of inferring validity from object counts.
    [[nodiscard]] uint64_t GetQueryGeneration() const noexcept;

    /// Cast a ray against one authored Collider sub-shape. Unlike a world
    /// query, this intentionally ignores layer, trigger and pair filters.
    bool RaycastCollider(const Collider &collider, const glm::vec3 &origin, const glm::vec3 &direction,
                         float maxDistance, RaycastHit &outHit) const;

    /// Return the closest world-space point on an authored Collider. Convex
    /// colliders preserve interior points. Non-convex meshes use triangle surfaces.
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

    Rigidbody *FindRigidbodyByBodyId(uint32_t bodyId) const;

    /// Identity view of ordinary component-owned GPU bodies and parent joints.
    void DestroyRagdollAssembly(uint64_t id);
    [[nodiscard]] bool HasRagdollAssembly(uint64_t id) const;
    void DriveRagdollPose(uint64_t id, const std::vector<glm::quat> &worldRotations, bool enableMotors = true);
    [[nodiscard]] std::vector<PhysicsBoxShapeState> GetBodyBoxShapeStates(uint32_t bodyId, bool queryTriggers) const;
    [[nodiscard]] std::vector<PhysicsShapeState> GetBodyShapeStates(uint32_t bodyId, bool queryTriggers = true) const;
    [[nodiscard]] std::shared_ptr<particle::GpuParticleCollisionDevicePayload>
    PrepareParticleCollisionScene(class Scene &scene,
                                  const std::shared_ptr<const particle::GpuParticleCollisionDevicePayload> &previous,
                                  uint64_t revision) const;

    /// Runtime-only collision policy for one exact Collider pair. This is
    /// separate from joint-owned whole-body suppression.
    void SetColliderPairIgnored(Collider *colliderA, Collider *colliderB, bool ignored);
    [[nodiscard]] bool GetColliderPairIgnored(const Collider *colliderA, const Collider *colliderB) const;
    void RemoveIgnoredPairsForCollider(const Collider *collider);

    /// Ensure all Colliders in the given scene have registered bodies
    /// and their transforms are up to date. Call before editor-mode raycasts.
    void EnsureSceneBodiesRegistered(class Scene *scene);

    /// @brief Dispatch buffered contact events to Component callbacks.
    ///        Call once per fixed step, immediately after Step().
    [[nodiscard]] size_t DispatchContactEvents();

  private:
    friend class Infernux;
    friend class SceneManager;
    friend class Ragdoll;
    friend class Joint;
    friend class PhysicsECSStore;
    friend class Rigidbody;
    friend class Transform;
    friend class JoltGpuRenderBindings;
    friend class HingeJoint;
    friend class SwingTwistJoint;
    friend class SixDOFJoint;
    using RagdollToken = std::array<uint64_t, 2>;
    struct RagdollMember
    {
        PhysicsTargetReference body, joint;
        RagdollToken bodyIdentity{};
        uint32_t bodyId = UINT32_MAX;
        uint64_t jointId = 0;
        uint32_t kind = JPH::GPU::JointFixed;
        int parent = -1;
        bool animationTarget = false;
        bool gpuJoint = false;
    };
    struct RagdollAssembly
    {
        PhysicsDomain *domain = nullptr;
        PhysicsTargetReference owner;
        RagdollToken ownerIdentity{};
        uint64_t epoch = 0;
        std::vector<RagdollMember> members;
        std::vector<PhysicsColliderPair> pairs;
        bool ready = false;
    };
    void QueueRagdollRebuild(Ragdoll &owner) noexcept;
    void CancelRagdollRebuild(Ragdoll &owner) noexcept;
    void ReconcileRagdolls(PhysicsDomain &domain);
    void SuspendRagdolls(PhysicsDomain &domain) noexcept;
    void RetireRagdolls(PhysicsDomain &domain) noexcept;
    void ClearRagdollDrive(uint64_t id);
    struct RagdollPoseDrive
    {
        uint64_t jointId;
        uint32_t kind;
        bool gpu;
        std::optional<glm::quat> relative;
    };
    void QueueRagdollPoseDrives(PhysicsDomain &domain, const std::vector<RagdollPoseDrive> &drives);
    void DriveRagdollPoseImpl(uint64_t id, const std::vector<glm::quat> &worldRotations, bool enableMotors,
                              bool animationController, bool deferred);
    void ClaimRagdollTarget(const Joint &joint) noexcept;
    void PrepareRagdollControl(const Joint &joint);
    uint32_t RagdollTargetReset(uint32_t bodyId) const;
    uint64_t m_nextRagdollId = 1;
    std::unordered_map<uint64_t, RagdollAssembly> m_ragdolls;
    std::map<RagdollToken, uint64_t> m_ragdollBodyOwners;
    [[nodiscard]] std::shared_ptr<const PhysicsColliderPairs> CaptureColliderPairs(PhysicsDomain &domain,
                                                                                   bool userOnly = false) const;
    void QueueColliderPairs(PhysicsDomain &domain) const;
    void InvalidateColliderPairSnapshot(PhysicsDomain &domain) const;
    void RestoreColliderPairs(RuntimeWorld &world, const PhysicsColliderPairs &source,
                              const std::unordered_map<uint64_t, uint64_t> &remap);
    void AcquireColliderPairOwner(Collider *a, Collider *b, const Component &owner);
    void ReleaseColliderPairOwner(PhysicsDomain &domain, const std::array<uint64_t, 2> &ownerIdentity);
    friend class SkeletonPhysicsPoseRead;
    void PrepareCpuPoseReads(const std::vector<Transform *> &transforms, bool worldSpace);
    [[nodiscard]] const PhysicsBodyMotionState *FindCachedMotionState(PhysicsDomain &domain, uint32_t bodyId,
                                                                      PhysicsStateFields fields) const;
    friend class HingeJointBatch;
    void ReadHingeMotion(const std::vector<HingeJoint *> &joints, float *output) const;
    void SetHingeMotorTargets(const std::vector<HingeJoint *> &joints, const float *targets);

    PhysicsWorld();
    ~PhysicsWorld();
    PhysicsWorld(const PhysicsWorld &) = delete;
    PhysicsWorld &operator=(const PhysicsWorld &) = delete;

    struct CpuServices;
    enum class Backend : uint8_t
    {
        Cpu,
        Gpu,
        StaticBoundary
    };
    struct BodyRegistration
    {
        PhysicsActorHandle actor;
        Backend backend;
        PhysicsECSStore::RigidbodyHandle rigidbody{};
        // Allocations persist under this actor ID. An allocation outside the
        // current author role is parked outside broadphase.
        bool cpuStaged = false, gpuStaged = false, staticResident = false;
        uint64_t stagedShapeRevision = 0;
        std::shared_ptr<PhysicsStaticBoundaryModel> staticModel;
        mutable size_t cpuPoseIndex = SIZE_MAX;
        // The source owns the published actor until the transfer's receipt.
        // Later author writes belong to the already staged target FIFO.
        std::optional<Backend> transferTarget;

        [[nodiscard]] bool HasCpuBody() const noexcept
        {
            return cpuStaged && backend != Backend::Gpu;
        }
        [[nodiscard]] bool HasGpuBody() const noexcept
        {
            return gpuStaged && backend != Backend::Cpu;
        }
        [[nodiscard]] Backend AuthorBackend() const noexcept
        {
            return transferTarget.value_or(backend);
        }
        [[nodiscard]] bool HasCpuAuthor() const noexcept
        {
            return cpuStaged && AuthorBackend() != Backend::Gpu;
        }
        [[nodiscard]] bool HasGpuAuthor() const noexcept
        {
            return gpuStaged && AuthorBackend() != Backend::Cpu;
        }
    };

    enum class JointBackend : uint8_t
    {
        Cpu,
        Gpu,
        Mixed
    };
    [[nodiscard]] static JointBackend SelectJointBackend(Backend owner, Backend connected) noexcept;
    void ValidateMixedJointPermission(uint32_t owner, uint32_t connected) const;
    [[nodiscard]] JointBackend PrepareJointEndpoints(PhysicsDomain &, uint32_t ownerBody, uint32_t connectedBody,
                                                     bool supportsMixed = false);
    [[nodiscard]] PhysicsDomain *FindCpuJointDomain(uint64_t jointId) const noexcept;
    void RetireCpuJointRoutes(const std::vector<uint64_t> &jointIds) noexcept;
    void RetireCpuJointRoutes(PhysicsDomain &) noexcept;
    PhysicsConstraintOwner &DetachMixedJointRoute(PhysicsDomain &, uint64_t jointId) noexcept;
    void RetireMixedJoint(PhysicsDomain &, uint64_t jointId) noexcept;
    void RetireMixedJointRoutes(PhysicsDomain &) noexcept;
    [[nodiscard]] bool HasMixedJointForBody(const PhysicsDomain &, uint32_t bodyId) const noexcept;
    PhysicsDomain &CurrentDomain() const;
    void RegisterBodyIdentity(PhysicsDomain &domain, uint32_t bodyId, PhysicsActorHandle actor, Backend backend);
    void UnregisterBodyIdentity(PhysicsDomain &domain, uint32_t bodyId) noexcept;
    void ChangeBodyBackend(PhysicsDomain &domain, BodyRegistration &entry, Backend backend) noexcept;
    [[nodiscard]] const BodyRegistration *FindBodyRegistration(uint32_t bodyId) const noexcept;
    [[nodiscard]] PhysicsActorHandle FindBodyActor(uint32_t bodyId) const noexcept;
    [[nodiscard]] Backend ReadBodyBackend(PhysicsDomain &domain, uint32_t bodyId) const;
    PhysicsActorData &ActorForBody(uint32_t bodyId) const;
    PhysicsDomain &DomainForBody(uint32_t bodyId) const;
    PhysicsDomain &DomainForActor(const PhysicsActorData &actor) const;
    PhysicsDomain &DomainForComponent(const Component &component) const;
    [[nodiscard]] bool HasCpuControl(const PhysicsDomain &domain) const noexcept;
    bool HasDomainControl(PhysicsDomain &domain) const;
    void CompleteDomainStep(PhysicsDomain &domain);
    void RunPreparedWork(PhysicsDomain &domain, JoltGpuControlWork work) const;
    void RegisterDomain(PhysicsDomain &domain);
    void EnsureCpuReady(PhysicsDomain &domain);
    JoltGpuScene &EnsureSharedGpuScene();
    JoltMixedConstraintSolver &EnsureMixedConstraintSolver();
    bool EnsureGpuReady(PhysicsDomain &domain);
    void RetireIdleGpuPartition(PhysicsDomain &domain);
    void RetireGpuPartition(PhysicsDomain &domain);
    void ServiceGpuRetirements();
    void RefreshBodySurfaces(PhysicsDomain &domain);
    void RequireHealthyStep(PhysicsDomain &domain) const;
    void FailStep(PhysicsDomain &domain) noexcept;
    bool AdvanceFixedStep(PhysicsDomain &domain, bool synchronous);
    void ValidateBodyTransfer(const Rigidbody &, bool gpu) const;
    void RequestBodyTransfer(const Rigidbody &);
    struct JointTransferBinding
    {
        Joint *joint;
        JointBackend destination;
    };
    [[nodiscard]] std::vector<JointTransferBinding>
    CollectTransferJoints(const PhysicsDomain &, const std::vector<uint32_t> &, Backend target) const;
    void StageBodyTransfer(PhysicsDomain &domain);
    bool TryStageBodyTransfer(PhysicsDomain &domain, const std::vector<uint32_t> &bodies, Backend target);
    bool AdvanceBodyTransfer(PhysicsDomain &domain, bool synchronous);
    JoltGpuParticipantStages CaptureParticipants(PhysicsDomain &domain);
    bool TryBeginFixed(PhysicsDomain &domain);
    void FinishFixed(PhysicsDomain &domain);
    [[nodiscard]] PhysicsContactPublication::Policy CaptureContactPolicy(const PhysicsDomain &domain) const;
    void RunCpuControl(PhysicsDomain &domain) const;
    void ApplyCpuControl(PhysicsDomain &domain) const;
    void CaptureCpuPoses(PhysicsDomain &domain, bool fixed) const;
    [[nodiscard]] Rigidbody *FindCpuPoseBody(const PhysicsDomain &domain, uint32_t id) const;
    void RebaseCpuPoseDescendants(const GameObject &parent) const;
    [[nodiscard]] bool PublishCpuPoses(PhysicsDomain &domain, float alpha,
                                       const std::vector<Rigidbody *> &gpuAncestors = {}) const;
    void CheckMixedRenderHierarchy(PhysicsDomain &domain) const;
    [[nodiscard]] bool
    ServicePresentationPoseReads(PhysicsDomain &domain, const std::vector<Rigidbody *> &bodies,
                                 const std::shared_ptr<const JoltGpuScene::Publication> &publication);
    bool EnsureStaticBoundary(PhysicsDomain &domain, uint32_t bodyId, Backend consumer);
    void QueueStaticFixedMoves(PhysicsDomain &domain);
    void SealStaticFixedMoves(PhysicsDomain &domain);
    void AcceptStaticFixedMoves(PhysicsDomain &domain);
    void RecordStaticPose(PhysicsDomain &domain, uint32_t bodyId, const glm::vec3 *position, const glm::quat *rotation,
                          bool cancelMove);
    bool EnsureStaticBoundaries(PhysicsDomain &domain, Backend consumer);
    bool AdvanceGpuJob(PhysicsDomain &domain, bool synchronous) const;
    bool BeginGpuControl(PhysicsDomain &domain, bool numericalChange, const JoltGpuScene::Consumer &consumer = {},
                         bool requirePublication = false) const;
    bool AdvanceGpuControl(PhysicsDomain &domain, bool synchronous) const;
    void RetryCapacityBodyCreations(PhysicsDomain &domain);
    void RequireGpu(PhysicsDomain &domain) const;
    [[nodiscard]] bool IsFixedPublicationHeld(const PhysicsDomain &domain) const noexcept;
    void FlushAuthoring(PhysicsDomain &domain) const;
    void DrainGpuSource(PhysicsDomain &domain) const;
    void ObserveGpuPublication(PhysicsDomain &domain, bool numericalChange) const;
    [[nodiscard]] JoltGpuScene::Consumer PrepareRenderConsumer(PhysicsDomain &domain, bool numericalChange) const;
    void RunControl(PhysicsDomain &domain, const JoltGpuScene::Consumer &consumer, bool publicationRead = false) const;
    void DrainQueryGpuCapacity(PhysicsDomain &domain) const;
    void AcceptQueryGpuPublication(PhysicsDomain &domain, bool refresh) const;
    void SealQuerySources(PhysicsDomain &domain, bool includeStaticBoundaries) const;
    void SealSpatialQuery(PhysicsDomain &domain) const;
    [[nodiscard]] std::shared_ptr<const JoltGpuScene::Publication> ReadGpuQueryPublication(PhysicsDomain &) const;
    [[nodiscard]] std::shared_ptr<const JoltGpuScene::Publication>
    ReadGpuSpatialQueryPublication(PhysicsDomain &) const;
    void ObserveSpatialGpuQuery(PhysicsDomain &, const std::shared_ptr<const JoltGpuScene::Publication> &,
                                const JoltGpuScene::Consumer &) const;
    [[nodiscard]] JoltCpuQueries::Filter
    CpuSpatialFilter(const JPH::GPU::QueryInput &, const std::shared_ptr<const JoltGpuScene::Publication> &) const;
    [[nodiscard]] std::shared_ptr<JoltGpuQueryResult>
    RunQuery(PhysicsDomain &, const std::shared_ptr<const JoltGpuScene::Publication> &,
             const std::vector<JPH::GPU::QueryInput> &, uint64_t sweepColliderId = 0,
             uint64_t sweepColliderGeneration = 0, uint32_t sweepBodyId = UINT32_MAX) const;
    void ReadJointMotion(PhysicsDomain &domain, const std::vector<uint64_t> &ids, float *output, uint32_t kind) const;
    [[nodiscard]] std::shared_ptr<JoltGpuQueryResult> RunGeometryQuery(PhysicsDomain &domain,
                                                                       const JPH::GPU::QueryInput &request,
                                                                       const JPH::GPU::QueryPrimitive &other,
                                                                       JoltGpuQueryGeometry geometry = {}) const;
    [[nodiscard]] std::vector<Collider *> QueryColliders(PhysicsDomain &, const JPH::GPU::QueryInput &,
                                                         const JoltCpuQueries::Shape &) const;
    [[nodiscard]] bool QueryClosestCast(PhysicsDomain &, const JPH::GPU::QueryInput &, const JoltCpuQueries::Shape &,
                                        RaycastHit &) const;
    [[nodiscard]] std::vector<Rigidbody *>
    QueryRigidbodyCandidatesInBounds(PhysicsDomain &, const std::shared_ptr<const JoltGpuScene::Publication> &,
                                     const glm::vec3 &minimum, const glm::vec3 &maximum, uint32_t layers,
                                     bool triggers) const;
    [[nodiscard]] PhysicsRigidbodyStateBatch
    GatherRigidbodyStateRows(const std::vector<Rigidbody *> &, PhysicsStateFields, bool includeBoxes, bool triggers,
                             const std::vector<JoltGpuQueryExportBinding> &, bool sealed,
                             const std::shared_ptr<const JoltGpuScene::Publication> &) const;
    [[nodiscard]] RaycastHit DecodeQueryHit(const JoltGpuQueryResult &result, const JPH::GPU::QueryHit &record) const;
    [[nodiscard]] bool CanQueueRigidbodyCommand(uint32_t bodyId, const Rigidbody &) const;
    [[nodiscard]] Backend SelectBodyBackend(const PhysicsActorData &, const BodyRegistration *) const;
    [[nodiscard]] bool NeedsBodyBindingChange(const PhysicsActorData &) const;
    [[nodiscard]] bool RebindBody(PhysicsActorData &);
    [[nodiscard]] bool CanStageNewGpuBody(PhysicsDomain &, size_t count = 1);
    void StageNewGpuBody(PhysicsDomain &, uint32_t bodyId, PhysicsActorHandle, const PhysicsActorData &, bool resident,
                         PreparedGpuActor);
    [[nodiscard]] uint32_t StageNewBody(const PhysicsActorData &, PhysicsActorHandle, Backend, bool resident);
    void RetireBodyReplicas(PhysicsDomain &, uint32_t bodyId, const BodyRegistration &);
    void StageActorModel(PhysicsActorData &actor, const PhysicsActorData &author,
                         const std::function<void(PreparedCpuActor &)> &editCpu = {},
                         const std::function<void(PreparedGpuActor &)> &editGpu = {});
    PhysicsTargetSource QueueKinematicBodyCommand(uint32_t bodyId, const JPH::GPU::BodyCommand &,
                                                  const PhysicsTargetAuthor * = nullptr);
    void RefreshGpuActor(PhysicsDomain &domain, uint32_t bodyId,
                         const std::function<void(PreparedGpuActor &)> &edit = {});
    [[nodiscard]] std::vector<std::pair<Transform *, Rigidbody *>>
    CollectCpuPoseReads(const std::vector<Transform *> &transforms, bool worldSpace) const;
    [[nodiscard]] PhysicsBodyMotionState ReadMotionState(PhysicsDomain &domain, uint32_t bodyId,
                                                         PhysicsStateFields fields) const;
    [[nodiscard]] JoltGpuControlWork
    PrepareStateBatch(PhysicsDomain &domain, const std::shared_ptr<const JoltGpuScene::Publication> &publication,
                      const std::vector<Rigidbody *> &bodies, PhysicsStateFields fields, bool includeBoxes,
                      bool queryTriggers, const std::vector<JoltGpuQueryExportBinding> &exports,
                      PhysicsRigidbodyStateBatch &batch, std::shared_ptr<JoltGpuQueryResult> &motion,
                      std::shared_ptr<JoltGpuQueryResult> &shapes, std::exception_ptr &requestError,
                      bool deferExport = false) const;
    [[nodiscard]] std::shared_ptr<const PhysicsShapeGeometry>
    ResolveShapeGeometry(PhysicsDomain &domain, const JoltGpuScene::PublishedLeaf &leaf,
                         const JPH::GPU::QueryShapeState &record) const;
    [[nodiscard]] std::shared_ptr<JoltGpuQueryResult> GatherBodyShapes(PhysicsDomain &domain, uint32_t bodyId,
                                                                       bool queryTriggers) const;
    void MaterializeStateBatch(PhysicsDomain &domain, PhysicsRigidbodyStateBatch &batch,
                               const std::shared_ptr<JoltGpuQueryResult> &motion,
                               const std::shared_ptr<JoltGpuQueryResult> &shapes, bool numerical,
                               PhysicsStateFields cachedFields = static_cast<PhysicsStateFields>(0x1ff)) const;

    bool m_initialized = false;
    std::thread::id m_ownerThreadId{};
    std::unique_ptr<CpuServices> m_cpuServices;
    std::unique_ptr<rhi::ComputeHost> m_gpuHost;
    FrameWorkOwner *m_gpuWork = nullptr;
    std::unique_ptr<JoltGpuJoints> m_gpuJoints;
    std::unique_ptr<JoltGpuScene> m_gpuScene;
    std::unique_ptr<JoltGpuQueries> m_gpuQueries;
    std::unique_ptr<JoltGpuRenderPoses> m_gpuRenderProgram;
    std::unique_ptr<JoltGpuContactReports> m_gpuContactReports;
    std::unique_ptr<JoltGpuMixedContacts> m_gpuMixedContacts;
    std::unique_ptr<JoltMixedConstraintSolver> m_mixedConstraintSolver;
    // Live component ownership is independent of a backend's numerical slots
    // and immutable publications, which may outlive a retired component.
    std::unordered_map<uint32_t, BodyRegistration> m_bodyActors;
    std::unordered_map<uint64_t, PhysicsDomain *> m_cpuJointDomains;
    std::unordered_map<uint64_t, PhysicsDomain *> m_mixedJointDomains;
    std::unordered_set<PhysicsDomain *> m_domains;
};

} // namespace infernux
