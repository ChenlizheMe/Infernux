#pragma once

#include <Jolt/Jolt.h>

#include "JoltGpuActor.h"
#include "JoltGpuTick.h"
#include "PhysicsConstraintOwner.h"
#include "PhysicsECSStore.h"
#include "PhysicsTargetReference.h"

#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace infernux
{
class JoltGpuJoints;

class JoltGpuQueryRequestError : public std::invalid_argument
{
  public:
    using std::invalid_argument::invalid_argument;
};

/// One physical domain using the engine's shared compute services. Component
/// ownership remains in PhysicsECSStore; only author transactions, device
/// resources and immutable publication identities belong to this owner.
class JoltGpuScene final
{
  public:
    static constexpr uint32_t InvalidBodyId = UINT32_MAX;

    struct OwnerIdentity
    {
        /// Capture the actor owner's intrinsic Transform before reserving.
        PhysicsTargetReference target;
        uint64_t gameObjectId = 0;
    };

    struct BodyAllocation
    {
        uint32_t bodyId = InvalidBodyId;
        uint32_t slot = UINT32_MAX;
    };

    struct PublishedLeaf
    {
        PhysicsTargetReference surfaceTarget;
        uint64_t surfaceGameObjectId = 0;
        uint64_t colliderId = 0;
        uint32_t subShapeId = 0;
        uint32_t subShapeBits = 0;
        uint32_t memberIndex = 0;
        /// Accepted terminal geometry in leaf COM space, at baked unit scale.
        JPH::RefConst<JPH::Shape> shape;
        bool sensor = false;
    };

    struct LeafAddress
    {
        uint32_t bodySlot = UINT32_MAX;
        uint32_t actorLeaf = UINT32_MAX;
    };

    /// Application identity and authored geometry for exactly one accepted
    /// slot generation. There is no host copy of live motion state here.
    struct PublishedActor
    {
        uint32_t bodyId = InvalidBodyId;
        uint32_t slot = UINT32_MAX;
        /// Accepted author motion type, without the independent residency bit.
        uint32_t acceptedMotionType = JPH::GPU::MotionStatic;
        PhysicsActorHandle actor;
        /// Enabled component generation contributing this accepted author model.
        PhysicsECSStore::RigidbodyHandle rigidbody;
        /// Full component ID captured with rigidbody, not a live pool lookup.
        uint64_t rigidbodyComponentId = 0;
        OwnerIdentity owner;
        std::vector<PublishedLeaf> leaves;
        JPH::GPU::Float4 rootCenterOfMass{};
        JPH::GPU::Float4 rootScale{1, 1, 1, 0};
        uint64_t shapeRevision = 0;
        JPH::RefConst<JPH::Shape> authoredShape;
    };

    struct PublishedJoint
    {
        uint64_t id = 0;
        uint32_t row = 0;
        /// Public endpoints: A is the owner, B the connected body or world.
        uint32_t bodyIdA = InvalidBodyId;
        uint32_t bodyIdB = InvalidBodyId;
        uint32_t kind = 0;
        JPH::GPU::JointAuthor author{};
    };

    struct JointUpdate
    {
        uint64_t id;
        JPH::GPU::JointAuthor author;
    };

    /// All arrays/resources describe the same completed numerical generation.
    /// actors is indexed by GPU slot; null entries are unowned tombstones.
    /// Retained results resolve through these identities after slot reuse.
    struct Publication
    {
        uint64_t simulationEpoch = 0;
        uint64_t fixedStepIndex = 0;
        std::shared_ptr<rhi::ComputeBuffer> bodyStates;
        std::shared_ptr<rhi::ComputeBuffer> bodyModels;
        std::shared_ptr<rhi::ComputeBuffer> shapeModels;
        std::shared_ptr<rhi::ComputeBuffer> worlds;
        std::vector<std::shared_ptr<const PublishedActor>> actors;
        /// Accepted author participation for this publication, not motion state.
        std::vector<uint8_t> residentBodies;
        /// Global leaf ordinal -> immutable actor catalog in this publication.
        /// Addresses own no individual allocation and never refer to live slots.
        std::vector<LeafAddress> leaves;
        std::shared_ptr<rhi::ComputeBuffer> jointStream;
        std::vector<PublishedJoint> joints;
    };
    /// The retained publication travels with every result/continuation. A
    /// consumer must resolve hit identities through this exact shared value.
    using Consumer = std::function<JoltGpuControlWork(const std::shared_ptr<const Publication> &)>;

    /// Targets come from the common compiler. This owner creates no compiler,
    /// device or queue. One fixed world range contains all capacity slots;
    /// exceeding the current fork bound is an explicit error, never paging
    /// colliding bodies into separate physical domains.
    JoltGpuScene(rhi::ComputeHost &host, FrameWorkOwner &work, uint32_t capacity, const ShaderTargetCode &stepTarget,
                 const ShaderTargetCode &mutationTarget, const ShaderTargetCode &commandTarget,
                 const ShaderTargetCode &broadphaseTarget, JoltGpuJoints &joints);
    ~JoltGpuScene();
    JoltGpuScene(const JoltGpuScene &) = delete;
    JoltGpuScene &operator=(const JoltGpuScene &) = delete;

    [[nodiscard]] BodyAllocation Reserve(PhysicsActorHandle actor, OwnerIdentity owner);
    /// Captures every contributing member as lifetime-safe per-leaf identity.
    /// Only a new ownership initializes state; subsequent changes rebase COM
    /// from the previously accepted metadata using actual device rotation.
    void StageActor(uint32_t bodyId, PreparedGpuActor actor);
    /// Removes participation without discarding authored mass/shape or pose.
    /// A leaving boundary cancels target ownership. Ordinary removal clears
    /// velocities; suspension preserves manual velocities and stops a drive.
    void SetResident(uint32_t bodyId, bool resident, bool preserveVelocity = false);
    /// Owns a value copy immediately. The slot and pose-command root COM are
    /// supplied by this owner, matching preceding queued model changes. An
    /// authored Move returns the exact batch/index source to install as the
    /// corresponding channel guard only after this call succeeds.
    PhysicsTargetSource QueueCommand(uint32_t bodyId, JPH::GPU::BodyCommand command,
                                     const PhysicsTargetAuthor *author = nullptr);
    /// Invalidates body lookup immediately. Its slot becomes reusable only
    /// after the ordered retirement transaction has been accepted.
    void StageRetire(uint32_t bodyId);
    uint64_t CreateJoint(PhysicsConstraintOwner &owner, uint32_t ownerBody, uint32_t connectedBody,
                         JPH::GPU::JointAuthor author);
    void UpdateJoint(uint64_t id, const std::function<void(JPH::GPU::JointAuthor &)> &update);
    /// One ordered author transaction for the complete batch.
    void UpdateJoints(std::vector<JointUpdate> updates);
    void DestroyJoint(uint64_t id);
    void DestroyJointsForBody(uint32_t bodyId);
    [[nodiscard]] const PublishedJoint &GetAuthoredJoint(uint64_t id) const;

    [[nodiscard]] BodyAllocation FindBody(uint32_t bodyId) const noexcept;
    [[nodiscard]] PhysicsActorHandle FindActor(uint32_t bodyId) const noexcept;
    [[nodiscard]] size_t GetBodyCount() const noexcept;
    [[nodiscard]] uint32_t GetCapacity() const noexcept;
    [[nodiscard]] bool CanReserve() const noexcept;
    [[nodiscard]] bool HasPendingControl() const noexcept;

    /// Queued author groups and initialization share the existing tick source.
    /// Author calls during a running tick append inputs for the next boundary;
    /// BeginTick itself requires the previous tick to be complete/published.
    void BeginTick(uint64_t simulationEpoch, uint64_t fixedStepIndex, float fixedDeltaTime,
                   const JoltGpuStepParameters &parameters);
    /// The final consumer runs once after the final solver writer is accepted,
    /// with frozen final body states. Its packet shares this tick's source;
    /// publication waits for that packet as well. Empty worlds also call it.
    void BeginTick(uint64_t simulationEpoch, uint64_t fixedStepIndex, float fixedDeltaTime,
                   const JoltGpuStepParameters &parameters, Consumer finalConsumer);
    /// Accept and complete only the queued author/command prefix, without
    /// advancing a solver or a clock. Uses the last published fixed-step
    /// identity (zero before the first tick) within the supplied epoch.
    void BeginControl(uint64_t simulationEpoch);
    /// Borrow explicit-geometry work on this same source without accepting
    /// queued author inputs, capturing or publishing a scene generation. The
    /// source must be idle/complete, with its prior publication already handled.
    /// Callers observe GetTickStatus/WaitForProgress, not PublishCompletedControl.
    void BeginWork(JoltGpuControlWork work);
    /// Prepares a read-only GPU consumer against the immutable post-command
    /// publication, through the same tick source and completion receipts. The
    /// callback and its continuations may prepare GPU work only: they must not
    /// submit, wait, author scene changes or reenter the world. Publication waits
    /// for the entire returned chain as well as the preceding author prefix.
    void BeginControl(uint64_t simulationEpoch, Consumer consumer);
    /// The optional final consumer follows all control/query work. Both
    /// consumers may prepare GPU work only under the callback contract above.
    void BeginControl(uint64_t simulationEpoch, Consumer consumer, Consumer finalConsumer);
    /// Publishes only a completed tick. Returns true exactly once per tick;
    /// pending/failed/cancelled work retains the previous full publication.
    bool PublishCompletedTick();
    bool PublishCompletedControl();
    [[nodiscard]] const std::shared_ptr<const Publication> &GetPublished() const noexcept;

    [[nodiscard]] JoltGpuTick::Status GetTickStatus() const noexcept;
    [[nodiscard]] FrameWorkOwner::SourceId GetSourceId() const noexcept;
    [[nodiscard]] bool AreControlInputsAccepted() const noexcept;
    [[nodiscard]] FrameWorkWaitResult WaitForProgress(uint64_t maxWaitNanoseconds);
    void SetAdmissionEnabled(bool enabled) noexcept;
    [[nodiscard]] bool IsAdmissionEnabled() const noexcept;
    void Cancel() noexcept;
    [[nodiscard]] bool CanRetire() const noexcept;

  private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace infernux
