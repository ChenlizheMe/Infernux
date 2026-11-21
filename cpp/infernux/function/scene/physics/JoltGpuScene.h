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
#include <unordered_map>
#include <vector>

namespace infernux
{
struct PhysicsBodyCommand;
class JoltGpuJoints;
class JoltGpuGeometryStore;
struct JoltGpuGeometryHandle;
struct JoltGpuGeometryLease;
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
        uint64_t colliderId = 0, colliderGeneration = 0;
        uint32_t subShapeId = 0;
        uint32_t subShapeBits = 0;
        uint32_t memberIndex = 0;
        /// Accepted terminal geometry in leaf COM space, at baked unit scale.
        JPH::RefConst<JPH::Shape> shape;
        std::shared_ptr<const JoltGpuGeometryHandle> geometry;
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
        PhysicsTargetReference bodyTarget;
        OwnerIdentity owner;
        std::vector<PublishedLeaf> leaves;
        JPH::GPU::Float4 rootCenterOfMass{};
        JPH::GPU::Float4 rootScale{1, 1, 1, 0};
        // Accepted physical author record; never a numerical state mirror.
        JPH::GPU::BodyModel physicalModel{};
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
        uint32_t runtimeIntent = 0;
    };

    /// Immutable accepted author directory, shared by numerical publications.
    /// Global body/leaf addresses stay unchanged; per-world joint rows are local.
    /// No live numerical storage or RuntimeWorld/Partition pointer is retained.
    struct PublicationDirectory
    {
        std::vector<std::shared_ptr<const PublishedActor>> actors;
        std::vector<uint8_t> residentBodies;
        std::unordered_map<uint32_t, uint32_t> bodySlots;

        [[nodiscard]] uint32_t FindBodySlot(uint32_t bodyId) const noexcept
        {
            const auto found = bodySlots.find(bodyId);
            return found == bodySlots.end() ? UINT32_MAX : found->second;
        }
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
        bool isFixedStep = false;
        std::shared_ptr<rhi::ComputeBuffer> bodyStates;
        // Compact immutable activity from this exact completed publication;
        // never a frozen copy of the mutable 128B sleep histories.
        std::shared_ptr<rhi::ComputeBuffer> sleepStatus;
        std::shared_ptr<rhi::ComputeBuffer> bodyModels;
        std::shared_ptr<rhi::ComputeBuffer> shapeModels;
        std::shared_ptr<rhi::ComputeBuffer> worlds;
        /// Owns exact global body/leaf vectors for this accepted author layout.
        std::shared_ptr<const PublicationDirectory> directory;
        std::shared_ptr<rhi::ComputeBuffer> jointStream;
        std::shared_ptr<const JoltGpuJointLayout> jointLayout;
        std::shared_ptr<const JoltGpuColliderPairs> colliderPairs;
        std::shared_ptr<JoltGpuGeometryStore> geometryStore;
        std::shared_ptr<const JoltGpuGeometryLease> geometry;
        bool hasGenericGeometry = false;

        /// Joint rows are local to this publication's real accepted world row.
        [[nodiscard]] const std::vector<PublishedJoint> &GetJoints() const
        {
            return directory->worldJoints.at(worldRow);
        }
    };
    /// The retained publication travels with every result/continuation. A
    /// consumer must resolve hit identities through this exact shared value.
    using Consumer = std::function<JoltGpuControlWork(const std::shared_ptr<const Publication> &)>;
    /// Synchronous host capture at successful tick writer admission, never on Busy.
    /// Must not submit work, drive the owner or reenter this scene.
    using ConsumerFactory = std::function<Consumer()>;
    struct ContactObserver
    {
        decltype(JoltGpuTickControl::prepareContactReport) phase;
        Consumer finish;
        // A Fixed can retain a static query catalog without native rigid dynamics.
        bool simulateRigidBodies = true;
    };
    // Called once at successful writer admission, before accepted author
    // capture. The publication directory is populated before the first phase.
    using ContactObserverFactory = std::function<ContactObserver(const std::shared_ptr<const Publication> &)>;

    /// Owner bookkeeping on the exact failed writer; must not throw, drive, wait, or
    /// destroy this scene. Partition ownership retains the World until source retirement.
    using FailureObserver = std::function<void(RuntimeWorld &)>;
    using ParticipantFactory = std::function<JoltGpuParticipantStages()>;

    /// One shared numerical arena/source, never one scene per RuntimeWorld.
    JoltGpuScene(rhi::ComputeHost &host, FrameWorkOwner &work, const ShaderTargetCode &stepTarget,
                 const ShaderTargetCode &mutationTarget, const ShaderTargetCode &commandTarget,
                 const ShaderTargetCode &broadphaseTarget, const ShaderTargetCode &narrowphaseTarget,
                 const ShaderTargetCode &constraintTarget, JoltGpuJoints &joints, FailureObserver failureObserver = {},
                 const ShaderTargetCode *captureTarget = nullptr, const ShaderTargetCode *dynamicsTarget = nullptr);
    ~JoltGpuScene();
    [[nodiscard]] bool TryPrepareStartupPrograms(bool hasLocalContacts);
    [[nodiscard]] const std::shared_ptr<JoltGpuGeometryStore> &GetGeometryStore() const noexcept;
    JoltGpuScene(const JoltGpuScene &) = delete;
    JoltGpuScene &operator=(const JoltGpuScene &) = delete;

    [[nodiscard]] bool RegisterPartition(RuntimeWorld &world, uint32_t initialCapacity);
    /// publicBodyId has already been reserved by PhysicsECSStore for actor.
    /// This backend only installs its address; it never allocates public IDs.
    /// initialResident belongs to this new identity, before its first StageActor.
    [[nodiscard]] BodyAllocation Reserve(RuntimeWorld &world, uint32_t publicBodyId, PhysicsActorHandle actor,
                                         OwnerIdentity owner, bool initialResident,
                                         uint32_t expectedAuthorBodyId = UINT32_MAX);
    void ReserveCapacity(RuntimeWorld &world, uint32_t capacity);
    void StageActor(uint32_t bodyId, PreparedGpuActor actor, bool resetMotion = false);
    void SetResident(uint32_t bodyId, bool resident, bool preserveVelocity = false);
    PhysicsTargetSource QueueCommand(uint32_t bodyId, JPH::GPU::BodyCommand command,
                                     const PhysicsTargetAuthor *author = nullptr);
    void QueueBodyCommand(const PhysicsBodyCommand &command);
    void StageRetire(uint32_t bodyId);
    uint64_t CreateJoint(PhysicsConstraintOwner &owner, uint32_t ownerBody, uint32_t connectedBody,
                         JPH::GPU::JointAuthor author);
    void UpdateJoint(uint64_t id, const std::function<void(JPH::GPU::JointAuthor &)> &update,
                     uint32_t runtimeIntent = 0);
    void UpdateJoints(std::vector<JointUpdate> updates);
    struct PoseDriveIntent
    {
        uint64_t id;
        uint32_t kind;
        JPH::GPU::Float4 relative;
        bool clear = false;
    };
    void QueuePoseDrive(RuntimeWorld &world, std::vector<PoseDriveIntent> intents);
    void DestroyJoint(uint64_t id);
    [[nodiscard]] bool HasJointForBody(uint32_t bodyId) const noexcept;
    void SetColliderPairs(RuntimeWorld &world, std::shared_ptr<const PhysicsColliderPairs> pairs);
    void DestroyJointsForBody(uint32_t bodyId);
    [[nodiscard]] const PublishedJoint &GetAuthoredJoint(uint64_t id) const;
    [[nodiscard]] BodyAllocation FindBody(uint32_t bodyId) const noexcept;
    [[nodiscard]] PhysicsActorHandle FindActor(uint32_t bodyId) const noexcept;
    /// Matches the generation in the ordered author FIFO, including an unaccepted model.
    [[nodiscard]] bool HasStagedRigidbody(uint32_t bodyId, PhysicsECSStore::RigidbodyHandle) const noexcept;
    /// Synchronous author metadata at writer admission, including reserved
    /// actors whose first model is still pending. The visitor must not mutate
    /// membership, drive this scene or retain references into its slot table.
    void VisitActorHandles(const RuntimeWorld &world, const std::function<void(PhysicsActorHandle)> &visitor) const;
    [[nodiscard]] size_t GetBodyCount(const RuntimeWorld &world) const;
    [[nodiscard]] JPH::GPU::BodyModel GetAcceptedBodyModel(uint32_t bodyId) const;
    [[nodiscard]] uint32_t GetCapacity(const RuntimeWorld &world) const;
    [[nodiscard]] bool CanReserve(const RuntimeWorld &world, size_t count = 1) const;
    [[nodiscard]] bool HasPendingControl(const RuntimeWorld &world) const;

    /// Resolve only the admitted GPU identities; the public interaction owner checks both permissions.
    /// Complete this before topology/wake/classification, and include GetRows() in external contact bodies.
    [[nodiscard]] JoltGpuControlWork PrepareParticipantBodies(RuntimeWorld &, const JPH::GPU::StepParams &,
                                                              std::vector<JoltGpuParticipantBodyTarget> targets,
                                                              uint32_t portRows,
                                                              std::shared_ptr<JoltGpuParticipantBodies> &out);
    /// Write only mapped GPU rows of scalar f32[portRows * 16], preserving interleaved CPU rows.
    [[nodiscard]] JoltGpuControlWork PrepareParticipantMotion(RuntimeWorld &,
                                                              const std::shared_ptr<JoltGpuParticipantBodies> &,
                                                              std::shared_ptr<rhi::ComputeBuffer> motion);
    /// Consume finite world-space linear/angular-about-COM deltas from scalar f32[portRows * 6].
    /// Native local constraints redistribute candidate speeds before the ordinary drift applies speed limits.
    [[nodiscard]] JoltGpuControlWork PrepareParticipantImpulses(RuntimeWorld &,
                                                                const std::shared_ptr<JoltGpuParticipantBodies> &,
                                                                std::shared_ptr<rhi::ComputeBuffer> impulses);

    /// Busy preserves the complete per-world FIFO and logical tick identity.
    [[nodiscard]] bool TryBeginTick(RuntimeWorld &world, uint64_t epoch, uint64_t step, float dt,
                                    const JoltGpuStepParameters &parameters, ConsumerFactory prepareFinalConsumer = {},
                                    ContactObserverFactory prepareContacts = {},
                                    ParticipantFactory prepareParticipants = {});
    [[nodiscard]] bool TryBeginControl(RuntimeWorld &world, uint64_t epoch, Consumer consumer = {},
                                       Consumer finalConsumer = {});
    [[nodiscard]] bool TryBeginWork(RuntimeWorld &world, JoltGpuControlWork &work, bool needsSnapshot = true);
    // Consume a published serial exactly once; polling a writer is the completion boundary.
    bool ConsumeCompletedTick(RuntimeWorld &world);
    bool ConsumeCompletedControl(RuntimeWorld &world);
    [[nodiscard]] const std::shared_ptr<const Publication> &GetPublished(const RuntimeWorld &world) const;
    [[nodiscard]] JoltGpuControlWork PrepareSleepQuery(const std::shared_ptr<const Publication> &publication,
                                                       std::vector<uint32_t> relativeBodies,
                                                       std::function<void(std::vector<JPH::GPU::UInt4>)> completed);
    [[nodiscard]] bool IsWriterFor(const RuntimeWorld &world) const noexcept;
    [[nodiscard]] JoltGpuTick::Status GetTickStatus(const RuntimeWorld &world);
    [[nodiscard]] FrameWorkOwner::SourceId GetSourceId() const noexcept;
    [[nodiscard]] bool AreControlInputsAccepted(const RuntimeWorld &world) const;
    [[nodiscard]] FrameWorkWaitResult WaitForProgress(RuntimeWorld &world, uint64_t maxWaitNanoseconds);
    [[nodiscard]] FrameWorkWaitResult WaitForSourceProgress(uint64_t maxWaitNanoseconds);
    RuntimeWorld *SetRequiredWorld(RuntimeWorld *world);
    void SetAdmissionEnabled(RuntimeWorld &world, bool enabled);
    [[nodiscard]] bool IsAdmissionEnabled(const RuntimeWorld &world) const;
    /// Nonblocking, local cancellation. Keep RuntimeWorld alive until CanRetire;
    /// the shared source performs GPU cleanup after necessary receipts settle.
    void Cancel(RuntimeWorld &world);
    void RetirePartition(RuntimeWorld &world);
    /// Host service for partitions whose World may still be playing on CPU or
    /// may already be leaving the session. Never opens a new numerical tick.
    void ServiceRetirements();
    /// Retained Publications keep frozen outputs independently of the row.
    [[nodiscard]] bool CanRetire(const RuntimeWorld &world);

  private:
    friend class PhysicsWorld;
    friend class SceneManager;
    [[nodiscard]] bool HasRunningFixedTick(const RuntimeWorld &world) const noexcept;
    [[nodiscard]] RuntimeWorld *GetRunningObservationWorld() const noexcept;
    void RequireParticipantBodies(RuntimeWorld &, const std::shared_ptr<JoltGpuParticipantBodies> &) const;
    bool IsCurrentPublication(const RuntimeWorld &world, const std::shared_ptr<const Publication> &publication) const;
    void RunPublicationObservation(RuntimeWorld &world, const std::shared_ptr<const Publication> &publication,
                                   JoltGpuControlWork work);
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace infernux
