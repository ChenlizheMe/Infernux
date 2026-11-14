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
class RuntimeWorld;

class JoltGpuQueryRequestError : public std::invalid_argument
{
  public:
    using std::invalid_argument::invalid_argument;
};

/// One shared packed numerical backend containing real RuntimeWorld partitions.
/// Component/public identity ownership remains in PhysicsECSStore. There is
/// exactly one storage, program set and Tick/source for all partitions.
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
        uint32_t row = 0; // Publication: local compact row. Internal author catalog: global row.
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

    /// Immutable accepted author directory, shared by numerical publications.
    /// Global body/leaf addresses stay unchanged; per-world joint rows are local.
    /// No live numerical storage or RuntimeWorld/Partition pointer is retained.
    struct PublicationDirectory
    {
        std::vector<std::shared_ptr<const PublishedActor>> actors;
        std::vector<uint8_t> residentBodies;
        std::vector<LeafAddress> leaves;
        std::vector<std::vector<PublishedJoint>> worldJoints;
    };

    /// All arrays/resources describe the same completed numerical generation.
    /// actors is indexed by GPU slot; null entries are unowned tombstones.
    /// Retained results resolve through these identities after slot reuse.
    struct Publication
    {
        uint64_t worldId = 0, layoutGeneration = 0;
        uint32_t worldRow = 0, bodyBase = 0, bodyCount = 0, leafBase = 0, leafCount = 0, jointBase = 0, jointCount = 0;
        uint64_t simulationEpoch = 0;
        uint64_t fixedStepIndex = 0;
        std::shared_ptr<rhi::ComputeBuffer> bodyStates;
        std::shared_ptr<rhi::ComputeBuffer> bodyModels;
        std::shared_ptr<rhi::ComputeBuffer> shapeModels;
        std::shared_ptr<rhi::ComputeBuffer> worlds;
        /// Owns exact global body/leaf vectors for this accepted author layout.
        std::shared_ptr<const PublicationDirectory> directory;
        std::shared_ptr<rhi::ComputeBuffer> jointStream;

        /// Joint rows are local to this publication's real accepted world row.
        [[nodiscard]] const std::vector<PublishedJoint> &GetJoints() const
        {
            return directory->worldJoints.at(worldRow);
        }
    };
    /// The retained publication travels with every result/continuation. A
    /// consumer must resolve hit identities through this exact shared value.
    using Consumer = std::function<JoltGpuControlWork(const std::shared_ptr<const Publication> &)>;

    /// One shared numerical arena/source, never one scene per RuntimeWorld.
    JoltGpuScene(rhi::ComputeHost &host, FrameWorkOwner &work, const ShaderTargetCode &stepTarget,
                 const ShaderTargetCode &mutationTarget, const ShaderTargetCode &commandTarget,
                 const ShaderTargetCode &broadphaseTarget, const ShaderTargetCode &constraintTarget,
                 JoltGpuJoints &joints);
    ~JoltGpuScene();
    JoltGpuScene(const JoltGpuScene &) = delete;
    JoltGpuScene &operator=(const JoltGpuScene &) = delete;

    [[nodiscard]] bool RegisterPartition(RuntimeWorld &world, uint32_t initialCapacity);
    /// publicBodyId has already been reserved by PhysicsECSStore for actor.
    /// This backend only installs its address; it never allocates public IDs.
    [[nodiscard]] BodyAllocation Reserve(RuntimeWorld &world, uint32_t publicBodyId, PhysicsActorHandle actor,
                                         OwnerIdentity owner);
    void ReserveCapacity(RuntimeWorld &world, uint32_t capacity);
    void StageActor(uint32_t bodyId, PreparedGpuActor actor);
    void SetResident(uint32_t bodyId, bool resident, bool preserveVelocity = false);
    PhysicsTargetSource QueueCommand(uint32_t bodyId, JPH::GPU::BodyCommand command,
                                     const PhysicsTargetAuthor *author = nullptr);
    void StageRetire(uint32_t bodyId);
    uint64_t CreateJoint(PhysicsConstraintOwner &owner, uint32_t ownerBody, uint32_t connectedBody,
                         JPH::GPU::JointAuthor author);
    void UpdateJoint(uint64_t id, const std::function<void(JPH::GPU::JointAuthor &)> &update);
    void UpdateJoints(std::vector<JointUpdate> updates);
    void DestroyJoint(uint64_t id);
    void DestroyJointsForBody(uint32_t bodyId);
    [[nodiscard]] const PublishedJoint &GetAuthoredJoint(uint64_t id) const;
    [[nodiscard]] BodyAllocation FindBody(uint32_t bodyId) const noexcept;
    [[nodiscard]] PhysicsActorHandle FindActor(uint32_t bodyId) const noexcept;
    [[nodiscard]] size_t GetBodyCount(const RuntimeWorld &world) const;
    [[nodiscard]] uint32_t GetCapacity(const RuntimeWorld &world) const;
    [[nodiscard]] bool CanReserve(const RuntimeWorld &world) const;
    [[nodiscard]] bool HasPendingControl(const RuntimeWorld &world) const;

    /// Busy preserves the complete per-world FIFO and logical tick identity.
    [[nodiscard]] bool TryBeginTick(RuntimeWorld &world, uint64_t epoch, uint64_t step, float dt,
                                    const JoltGpuStepParameters &parameters, Consumer finalConsumer = {});
    [[nodiscard]] bool TryBeginControl(RuntimeWorld &world, uint64_t epoch, Consumer consumer = {},
                                       Consumer finalConsumer = {});
    [[nodiscard]] bool TryBeginWork(RuntimeWorld &world, JoltGpuControlWork &work);
    bool PublishCompletedTick(RuntimeWorld &world);
    bool PublishCompletedControl(RuntimeWorld &world);
    [[nodiscard]] const std::shared_ptr<const Publication> &GetPublished(const RuntimeWorld &world) const;
    [[nodiscard]] bool IsWriterFor(const RuntimeWorld &world) const noexcept;
    [[nodiscard]] JoltGpuTick::Status GetTickStatus(const RuntimeWorld &world);
    [[nodiscard]] FrameWorkOwner::SourceId GetSourceId() const noexcept;
    [[nodiscard]] bool AreControlInputsAccepted(const RuntimeWorld &world) const;
    [[nodiscard]] FrameWorkWaitResult WaitForProgress(RuntimeWorld &world, uint64_t maxWaitNanoseconds);
    [[nodiscard]] FrameWorkWaitResult WaitForSourceProgress(uint64_t maxWaitNanoseconds);
    void SetAdmissionEnabled(RuntimeWorld &world, bool enabled);
    [[nodiscard]] bool IsAdmissionEnabled(const RuntimeWorld &world) const;
    /// Nonblocking, local cancellation. Keep RuntimeWorld alive until CanRetire;
    /// the shared source performs GPU cleanup after necessary receipts settle.
    void Cancel(RuntimeWorld &world);
    void RetirePartition(RuntimeWorld &world);
    /// Services accepted cleanup nonblockingly. A true result for a retiring
    /// World also releases its completed partition and reusable zero-range row.
    /// An ordinary Cancel retains its partition so the same World can reopen.
    /// Retained Publications keep exact frozen outputs independently of the row.
    [[nodiscard]] bool CanRetire(const RuntimeWorld &world);

  private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace infernux
