#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/Core/JobSystemSingleThreaded.h>
#include <Jolt/Physics/Body/BodyActivationListener.h>
#include <Jolt/Physics/Body/BodyID.h>
#include <Jolt/Physics/Collision/ContactListener.h>
#include <Jolt/Physics/Collision/SimShapeFilter.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include "InfernuxJoltJobSystemAdapter.h"
#include "JoltCpuActor.h"
#include "JoltCpuConstraints.h"
#include "JoltCpuTempAllocator.h"
#include "PhysicsBodyState.h"
#include "PhysicsContactTypes.h"
#include "PhysicsTargetReference.h"

#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace JPH
{
class CompoundShape;
}

namespace infernux
{
struct EngineConfig;
struct JoltCpuShapeLeaves;
enum class ForceMode;

/// Shared scratch and worker access, not another pool for each RuntimeWorld.
/// The owner advances CPU worlds serially; Jolt may parallelize a world's step
/// on the engine workers. No GPU device or host-frame budget is involved.
class JoltCpuExecution final
{
  public:
    explicit JoltCpuExecution(const EngineConfig &config);
    JPH::EPhysicsUpdateError Step(JPH::PhysicsSystem &system, float deltaTime, int collisionSteps);

  private:
    JoltCpuTempAllocator m_scratch;
    JPH::JobSystemSingleThreaded m_inlineJobs;
    InfernuxJoltJobSystemAdapter m_workerJobs;
};

/// Native state owned by one logical World. Public identities are supplied by
/// the common actor store; Jolt indices/sequence numbers never escape this
/// backend. The shared layer filters outlive the scene; contact policy is owned here.
class JoltCpuScene final : private JPH::ContactListener,
                           private JPH::SimShapeFilter,
                           private JPH::BodyActivationListener
{
  public:
    JoltCpuScene(const EngineConfig &config, const JPH::BroadPhaseLayerInterface &layers,
                 const JPH::ObjectVsBroadPhaseLayerFilter &broadphaseFilter,
                 const JPH::ObjectLayerPairFilter &pairFilter);
    ~JoltCpuScene() override;
    [[nodiscard]] static JPH::PhysicsSettings CaptureSettings(const EngineConfig &config);

    /// Creation does not enter broadphase. The common actor lifecycle publishes
    /// the body through Jolt's normal single/batch add operations.
    JPH::BodyID CreateBody(uint32_t bodyId, const PreparedCpuActor &actor);
    void AddBodiesBatch(const std::vector<uint32_t> &bodyIds);
    /// The owner cancels any kinematic target and clears its drive-owned velocity before suspension.
    void RemoveBodyFromBroadphase(uint32_t bodyId, bool preserveVelocity);
    [[nodiscard]] bool IsBodyInBroadphase(uint32_t bodyId) const;
    [[nodiscard]] bool NeedsBodyReplacement(uint32_t bodyId, const PreparedCpuActor &actor) const;
    /// Upgrade a collider-only allocation after the owner has explicitly retired the body's constraints.
    /// The owner publishes the accepted geometry and consumes the captured initial velocities.
    JPH::BodyID ReplaceBody(uint32_t bodyId, const PreparedCpuActor &actor);
    /// Install a complete author update between steps without resetting pose,
    /// velocity or pending dynamic forces. Connected constraints receive the
    /// local COM delta. Adding motion storage to an original static allocation
    /// goes through the owner's body/constraint replacement.
    JPH::Vec3 UpdateBody(uint32_t bodyId, const PreparedCpuActor &actor);
    /// Caller has detached this body's constraints before destruction.
    void DestroyBody(uint32_t bodyId);
    [[nodiscard]] JPH::BodyID NativeBody(uint32_t bodyId) const;
    [[nodiscard]] bool HasBody(uint32_t bodyId) const noexcept;
    [[nodiscard]] const PreparedBodyMass &GetAcceptedMass(uint32_t bodyId) const;
    /// One accepted native sample, read between updates. No device query or
    /// live component access is needed, including for a sleeping body.
    [[nodiscard]] PhysicsBodyMotionState ReadMotionState(uint32_t bodyId,
                                                         PhysicsStateFields fields = PhysicsStateFields::All,
                                                         bool *sleeping = nullptr) const;

    /// The owner applies accepted author mutations and commands in order between native updates.
    enum class ForceKind : uint8_t
    {
        Force,
        Torque,
        ForceAtPosition
    };
    void ApplyBodyForce(uint32_t bodyId, ForceKind kind, ForceMode mode, JPH::Vec3Arg value,
                        JPH::RVec3Arg point = JPH::RVec3::sZero());
    void SetBodyOriginPose(uint32_t bodyId, const std::optional<JPH::RVec3> &position,
                           const std::optional<JPH::Quat> &rotation);
    void SetBodyVelocity(uint32_t bodyId, const std::optional<JPH::Vec3> &linear,
                         const std::optional<JPH::Vec3> &angular);
    void SetBodyAwake(uint32_t bodyId, bool awake);
    void ResetBodyMotion(uint32_t bodyId);
    struct MotionReset
    {
        uint32_t bodyId;
        JPH::RVec3 position;
        JPH::Quat rotation;
        JPH::Vec3 linearVelocity, angularVelocity;
    };
    void ResetBodyMotions(const std::vector<MotionReset> &resets);

    /// Returns the bounds before integration when the prescribed body will move.
    /// The fixed-step owner wakes neighbours of both endpoints after integration.
    [[nodiscard]] std::optional<JPH::AABox> StartKinematicDrive(uint32_t bodyId, JPH::RVec3Arg position,
                                                                JPH::QuatArg rotation, float duration, float maxSpeed);

    /// A collider-only actor temporarily supplies contact velocity for one Fixed.
    [[nodiscard]] std::optional<JPH::AABox> StartTemporaryStaticDrive(uint32_t bodyId, JPH::RVec3Arg position,
                                                                      JPH::QuatArg rotation, float duration,
                                                                      float maxSpeed);
    void StopTemporaryStaticDrive(uint32_t bodyId);

    struct PoseUpdate
    {
        PhysicsBodyPoseUpdate pose;
        bool sleeping;
    };
    /// Consume the dirty poses at a completed owner publication boundary.
    [[nodiscard]] std::vector<PoseUpdate> TakePoseUpdates();

    /// Immutable, sorted canonical pairs from the common world policy.
    bool SetIgnoredPairs(std::shared_ptr<const PhysicsColliderPairs> pairs);
    [[nodiscard]] bool UsesIgnoredPairs(const std::shared_ptr<const PhysicsColliderPairs> &pairs) const noexcept
    {
        return m_ignoredPairs == pairs;
    }

    // Captured author values are immutable throughout Update. Jolt workers do
    // not call live Components or resolve their targets while solving contacts.
    using Surface = PreparedCpuActor::Surface;
    [[nodiscard]] const Surface &GetSurface(uint32_t bodyId, JPH::SubShapeID subshape) const;
    [[nodiscard]] size_t GetBodyCount() const noexcept
    {
        return m_bodies.size();
    }
    [[nodiscard]] JoltCpuConstraints &Constraints() noexcept
    {
        return m_constraints;
    }

    [[nodiscard]] const JoltCpuConstraints &Constraints() const noexcept
    {
        return m_constraints;
    }

    [[nodiscard]] JPH::PhysicsSystem &System() noexcept
    {
        return m_system;
    }
    [[nodiscard]] const JPH::PhysicsSystem &System() const noexcept
    {
        return m_system;
    }

    [[nodiscard]] std::shared_mutex &QueryMutex() const noexcept
    {
        return m_queryMutex;
    }

  private:
    friend class JoltCpuContactPublication;
    friend class JoltCpuConstraints;
    friend class JoltCpuControl;
    friend class JoltCpuQueries;
    struct BodyRecord
    {
        JPH::BodyID native;
        uint64_t shapeRevision = 0;
        JPH::RefConst<JPH::Shape> shape;
        JPH::Vec3 rootScale = JPH::Vec3::sOne();
        const JPH::CompoundShape *memberCompound = nullptr;
        std::vector<Surface> surfaces;
        PhysicsTargetReference bodyTarget;
        std::optional<PreparedBodyMass> mass;
        int collisionDetectionMode = 0;
        bool authoredDynamic = false;
        bool awakeWhenDetached = false;
        bool temporaryStaticDrive = false;
        uint32_t poseUpdateIndex = UINT32_MAX;
        mutable std::shared_ptr<JoltCpuShapeLeaves> queryLeaves;

        uint32_t MemberIndex(JPH::SubShapeID subshape) const;
        const JoltCpuShapeLeaves &QueryLeaves() const;
        void ReuseQueryGeometry(const BodyRecord &previous);
    };
    static BodyRecord CaptureBodyRecord(const PreparedCpuActor &actor);
    static PhysicsBodyMotionState SampleMotionState(uint32_t bodyId, const BodyRecord &record, const JPH::Body &body,
                                                    PhysicsStateFields fields, bool *sleeping = nullptr);
    static bool AcceptedAwake(const BodyRecord &record, const JPH::Body &body);
    JPH::BodyID ClearBodyMotion(uint32_t bodyId);
    void WakeBody(BodyRecord &record, JPH::Body &body);
    void WakeBodiesInBounds(JPH::AABox bounds, JPH::BodyID exclude = {});
    void WakeBodiesAtMovedBounds(JPH::BodyID source, const JPH::AABox &before);
    void MarkPoseUpdate(uint32_t bodyId, BodyRecord &record);
    void MarkActivePoseUpdates();
    void OnBodyActivated(const JPH::BodyID &, JPH::uint64) noexcept override
    {
    }
    void OnBodyDeactivated(const JPH::BodyID &native, JPH::uint64 userData) noexcept override;
    bool SurfacesCollide(const Surface &a, const Surface &b) const;
    static float CombineSurface(float a, float b, PhysicsMaterialCombine modeA, PhysicsMaterialCombine modeB);
    JPH::BodyID CreateBody(uint32_t bodyId, const JPH::BodyCreationSettings &settings, BodyRecord record);
    JPH::ValidateResult OnContactValidate(const JPH::Body &a, const JPH::Body &b, JPH::RVec3Arg,
                                          const JPH::CollideShapeResult &hit) override;
    JPH::ValidateResult OnCCDContactValidate(const JPH::Body &a, const JPH::Body &b, JPH::RVec3Arg origin,
                                             const JPH::CollideShapeResult &hit, bool fromEarlierImpact) override;
    void OnContactAdded(const JPH::Body &a, const JPH::Body &b, const JPH::ContactManifold &manifold,
                        JPH::ContactSettings &settings) override;
    void OnContactPersisted(const JPH::Body &a, const JPH::Body &b, const JPH::ContactManifold &manifold,
                            JPH::ContactSettings &settings) override;
    bool ShouldCollide(const JPH::Body &a, const JPH::Shape *shapeA, const JPH::SubShapeID &subshapeA,
                       const JPH::Body &b, const JPH::Shape *shapeB, const JPH::SubShapeID &subshapeB) const override;
    JPH::PhysicsSystem m_system;
    mutable std::shared_mutex m_queryMutex;
    std::unordered_map<uint32_t, BodyRecord> m_bodies;
    std::vector<uint32_t> m_poseUpdates;
    struct DeactivatedPose
    {
        uint32_t bodyId;
        JPH::BodyID native;
    };
    std::mutex m_deactivatedPoseMutex;
    std::vector<DeactivatedPose> m_deactivatedPoses;
    std::exception_ptr m_deactivatedPoseError;
    std::shared_ptr<const PhysicsColliderPairs> m_ignoredPairs;
    JoltCpuConstraints m_constraints{*this};
};
} // namespace infernux
