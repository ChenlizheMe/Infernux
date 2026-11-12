#pragma once

#include <Jolt/Jolt.h>

#include "JoltGpuActor.h"
#include "JoltGpuTick.h"
#include "PhysicsECSStore.h"
#include "PhysicsTargetReference.h"

#include <functional>
#include <memory>
#include <vector>

namespace infernux
{

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

    /// Application identity and authored geometry for exactly one accepted
    /// slot generation. There is no host copy of live motion state here.
    struct PublishedActor
    {
        uint32_t bodyId = InvalidBodyId;
        uint32_t slot = UINT32_MAX;
        PhysicsActorHandle actor;
        /// Component generation contributing the dynamic author model.
        PhysicsECSStore::RigidbodyHandle rigidbody;
        OwnerIdentity owner;
        PhysicsTargetReference surfaceTarget;
        uint64_t surfaceGameObjectId = 0;
        uint64_t colliderId = 0;
        uint32_t subShapeId = 0;
        JPH::GPU::Float4 rootCenterOfMass{};
        uint64_t shapeRevision = 0;
        JPH::RefConst<JPH::Shape> authoredShape;
    };

    /// All arrays/resources describe the same completed numerical generation.
    /// actors is indexed by GPU slot; null entries are unowned tombstones.
    /// Retained results must resolve through this identity array, never through
    /// the current body's lookup table after a slot has been reused.
    struct Publication
    {
        uint64_t simulationEpoch = 0;
        uint64_t fixedStepIndex = 0;
        std::shared_ptr<rhi::ComputeBuffer> bodyStates;
        std::shared_ptr<rhi::ComputeBuffer> bodyModels;
        std::shared_ptr<rhi::ComputeBuffer> shapeModels;
        std::shared_ptr<rhi::ComputeBuffer> worlds;
        std::vector<std::shared_ptr<const PublishedActor>> actors;
    };

    /// Targets come from the common compiler. This owner creates no compiler,
    /// device or queue. One fixed world range contains all capacity slots;
    /// exceeding the current fork bound is an explicit error, never paging
    /// colliding bodies into separate physical domains.
    JoltGpuScene(rhi::ComputeHost &host, FrameWorkOwner &work, uint32_t capacity, const ShaderTargetCode &stepTarget,
                 const ShaderTargetCode &mutationTarget, const ShaderTargetCode &commandTarget);
    ~JoltGpuScene();
    JoltGpuScene(const JoltGpuScene &) = delete;
    JoltGpuScene &operator=(const JoltGpuScene &) = delete;

    [[nodiscard]] BodyAllocation Reserve(PhysicsActorHandle actor, OwnerIdentity owner);
    /// Captures the raw surfaceCollider as a lifetime-safe target immediately.
    /// Only a new ownership initializes state; subsequent changes rebase COM
    /// from the previously accepted metadata using actual device rotation.
    void StageActor(uint32_t bodyId, PreparedGpuActor actor);
    /// Removes participation without discarding authored mass/shape or pose.
    /// Ordinary removal clears velocities through ordered GPU commands;
    /// suspension preserves them. Neither path resets the authored pose.
    void SetResident(uint32_t bodyId, bool resident, bool preserveVelocity = false);
    /// Owns a value copy immediately. The slot and pose-command root COM are
    /// supplied by this owner, matching preceding queued model changes.
    void QueueCommand(uint32_t bodyId, JPH::GPU::BodyCommand command);
    /// Invalidates body lookup immediately. Its slot becomes reusable only
    /// after the ordered retirement transaction has been accepted.
    void StageRetire(uint32_t bodyId);

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
    /// Accept and complete only the queued author/command prefix, without
    /// advancing a solver or a clock. Uses the last published fixed-step
    /// identity (zero before the first tick) within the supplied epoch.
    void BeginControl(uint64_t simulationEpoch);
    /// Prepares a read-only GPU consumer against the immutable post-command
    /// publication, through the same tick source and completion receipts. The
    /// callback may prepare GPU work only: it must not submit, wait, author
    /// scene changes or reenter the world. Control publication waits for its
    /// returned packet to complete as well as the preceding author prefix.
    void BeginControl(uint64_t simulationEpoch,
                      std::function<std::shared_ptr<rhi::PreparedComputeBatch>(const Publication &)> consumer);
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
