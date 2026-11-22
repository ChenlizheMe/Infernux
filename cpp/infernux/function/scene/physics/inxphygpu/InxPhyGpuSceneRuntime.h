// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../PhysicsBodyCommand.h"
#include "../PhysicsBodyState.h"
#include "../PhysicsContactPublication.h"
#include "../PhysicsECSStore.h"
#include "../PhysicsGpuRenderBindings.h"
#include "../PhysicsRaycastHit.h"
#include "../PhysicsShapeState.h"
#include "../PhysicsTargetReference.h"
#include "InxPhyGpuParticleExport.h"
#include "InxPhyGpuSceneModel.h"
#include "InxPhyGpuStep.h"

#include <array>
#include <core/threading/FrameWorkOwner.h>
#include <functional>
#include <variant>

namespace infernux
{
class PhysicsParticipant;
}

namespace infernux::physics::inxphygpu
{

struct InxPhyGpuSceneLayout;
struct InxPhyGpuScenePoseTargets;
struct InxPhyGpuSceneKinematicTargets;

struct InxPhyGpuRegisteredBody
{
    uint32_t bodyId;
    uint64_t shapeRevision;
    PhysicsActorHandle actor{};
    std::vector<InxPhyGpuSceneSurfaceIdentity> surfaces{}; // Author collider order.
    PhysicsTargetReference bodyTarget{};
};

struct InxPhyGpuRegisteredJoint
{
    uint64_t jointId;
    uint32_t authorRow;
    std::optional<InxPhyGpuJointPoseTarget> authoredPose{};
};

struct InxPhyGpuRegisteredWorld
{
    uint64_t worldId;
    std::vector<InxPhyGpuRegisteredBody> bodies; // Scene model author order.
    std::vector<InxPhyGpuRegisteredJoint> joints{};
    uint64_t simulationEpoch = 0;
};

struct InxPhyGpuSceneSetup
{
    InxPhyGpuSceneModel model;
    InxPhyGpuRegisteredWorld world;
};

struct InxPhyGpuSceneImpulseBatch
{
    uint64_t worldId;
    uint32_t substep;
    std::vector<uint32_t> bodyIds;
    PhysicsDeviceBufferRange input;
};

struct InxPhyGpuSceneParticipants
{
    uint64_t worldId;
    std::vector<std::weak_ptr<PhysicsParticipant>> owners;
};

struct InxPhyGpuFixedWorld
{
    ContactStreamIdentity identity; // Identity of the state in bodyStates, even when the attempted step was rejected.
    float time;
    uint32_t status;
    uint32_t acceptedStep;
    uint64_t attemptedFixedStepIndex = 0; // The request that produced status.
    bool projectStateInvalidated = false;
};

struct InxPhyGpuFixedPublication
{
    rhi::SubmissionTicket submission;
    std::shared_ptr<rhi::ComputeBuffer> bodyStates;
    std::vector<InxPhyGpuFixedWorld> worlds;
    std::vector<InxPhyGpuBodyState> hostBodies;
    std::vector<std::vector<InxPhyGpuContactReport>> contacts; // Substep-major, then enrolled world.
    uint32_t bodiesPerWorld = 0;
    uint32_t substeps = 1;

    [[nodiscard]] std::optional<uint32_t> FindAcceptedWorld(ContactStreamIdentity identity) const;
    [[nodiscard]] std::optional<InxPhyGpuSceneJointReference> FindJointReference(uint32_t world,
                                                                                 uint64_t jointId) const;
    [[nodiscard]] PhysicsGpuPoseSource PoseSource(uint32_t world) const;
    [[nodiscard]] std::optional<PhysicsGpuBodyPoseBinding> FindPoseBody(uint32_t world, PhysicsActorHandle actor,
                                                                        uint32_t bodyId) const;
    [[nodiscard]] PhysicsGpuMotionExport GatherMotion(uint32_t world, const std::vector<uint32_t> &bodies) const;
    [[nodiscard]] PhysicsGpuBoxExport GatherBoxes(uint32_t world, const std::vector<uint32_t> &bodies,
                                                  bool includeTriggers) const;
    /// Consume the accepted world's reports in solver substep order.
    void PublishContacts(uint32_t world, PhysicsContactPublication &destination,
                         const PhysicsContactPublication::Policy &policy);

  private:
    friend class InxPhyGpuSceneRuntime;
    std::shared_ptr<const InxPhyGpuSceneLayout> m_layout;
    std::shared_ptr<const InxPhyGpuScenePoseTargets> m_poseTargets;
    std::shared_ptr<const InxPhyGpuSceneKinematicTargets> m_kinematicTargets;
    std::vector<std::pair<uint32_t, uint32_t>> m_pendingWrenchRows;
    std::vector<std::pair<uint32_t, uint32_t>> m_kinematicVelocityRows;
    std::shared_ptr<rhi::ComputeBuffer> m_pendingWrenches;
    bool m_integrated = false;
};

struct InxPhyGpuHostBody
{
    PhysicsBodyPoseUpdate pose;
    uint64_t shapeRevision;
    glm::vec3 centerOfMass, linearVelocity, angularVelocity;
    bool sleeping;
};

struct InxPhyGpuSceneBodyRead
{
    ContactStreamIdentity identity;
    std::shared_ptr<rhi::PreparedComputeBatch> packet;

    /// Consume completed motion rows in the caller's requested body order.
    [[nodiscard]] std::vector<InxPhyGpuHostBody> ReadHostBodies() const;

  private:
    friend class InxPhyGpuSceneRuntime;
    PhysicsGpuMotionRead m_motion;
};

struct InxPhyGpuSceneShapeRead
{
    ContactStreamIdentity identity;
    std::shared_ptr<rhi::PreparedComputeBatch> packet;

    [[nodiscard]] std::vector<PhysicsShapeState> ReadHostShapes() const;

  private:
    friend class InxPhyGpuSceneRuntime;
    std::shared_ptr<const InxPhyGpuSceneLayout> m_layout;
    PhysicsGpuMotionRead m_motion;
    std::vector<uint32_t> m_leaves;
    uint32_t m_world = 0, m_body = 0;
};

struct InxPhyGpuSceneRayBatch
{
    ContactStreamIdentity identity;
    InxPhyGpuPreparedRayBatch query;

    /// Materialize the ordinary scene hits on the owner thread after submission.
    /// Consumes the host readback. The batch retains the queried world's layout.
    [[nodiscard]] std::vector<RaycastHit> ReadHostHits(uint32_t *total = nullptr) const;

  private:
    friend class InxPhyGpuSceneRuntime;
    struct RayLine
    {
        glm::vec3 origin, direction;
    };
    std::shared_ptr<const InxPhyGpuSceneLayout> m_layout;
    std::vector<RayLine> m_lines;
    uint32_t m_world = 0, m_allHitCapacity = 0;
};

struct InxPhyGpuSceneBodyBoundsRead
{
    ContactStreamIdentity identity;
    InxPhyGpuPreparedBodyBounds query;

    [[nodiscard]] std::vector<uint32_t> ReadHostBodyIds() const;

  private:
    friend class InxPhyGpuSceneRuntime;
    std::shared_ptr<const InxPhyGpuSceneLayout> m_layout;
    uint32_t m_world = 0;
};

struct InxPhyGpuSceneVolumeRead
{
    std::shared_ptr<rhi::PreparedComputeBatch> packet;

    [[nodiscard]] std::vector<Collider *> ReadOverlaps() const;
    [[nodiscard]] RaycastHit ReadCast() const;

  private:
    friend class InxPhyGpuSceneRuntime;
    std::shared_ptr<const InxPhyGpuSceneLayout> m_layout;
    uint32_t m_world = 0;
};

struct InxPhyGpuSceneSweepSource
{
    uint32_t bodyId;
    uint64_t colliderId, generation, gameObjectId;
};

/// Fixed-step source for one compiled layout and its enrolled scene worlds.
/// Admission and receipt collection use the engine's FrameWorkOwner.
class InxPhyGpuSceneRuntime final
{
  public:
    InxPhyGpuSceneRuntime(FrameWorkOwner &work, rhi::ComputeHost &host, ShaderCodeTarget target,
                          const InxPhyGpuSceneModel &scene, std::vector<InxPhyGpuRegisteredWorld> worlds,
                          float timestep, const ModelVector &gravity, InxPhyGpuCollisionCapacity collisionCapacity,
                          const InxPhyGpuContactSolver &solver, uint32_t commandCapacity, bool contactReports = false,
                          uint32_t substeps = 1, InxPhyGpuRayBatchLayout rayLayout = {},
                          const InxPhyGpuFixedPublication *previous = nullptr);
    ~InxPhyGpuSceneRuntime();
    InxPhyGpuSceneRuntime(const InxPhyGpuSceneRuntime &) = delete;
    InxPhyGpuSceneRuntime &operator=(const InxPhyGpuSceneRuntime &) = delete;

    [[nodiscard]] FrameWorkOwner::SourceId SourceId() const noexcept
    {
        return m_source;
    }
    [[nodiscard]] std::optional<uint32_t> BodyRow(uint32_t world, uint32_t engineBody) const;

    /// Take initialized poses before stepping. A refreshed scene begins at
    /// its previous accepted fixed index. Initialization requires no readback.
    [[nodiscard]] std::optional<InxPhyGpuFixedPublication> TakeInitial();

    struct PublicationWork
    {
        std::shared_ptr<rhi::PreparedComputeBatch> packet;
        std::function<void()> seal;
        std::function<PublicationWork(const InxPhyGpuFixedPublication &)> continueAfterReceipt{};
    };
    using PublicationConsumer = std::function<PublicationWork(const InxPhyGpuFixedPublication &completed)>;
    using ParticleConsumer = std::function<void(std::shared_ptr<particle::GpuParticleCollisionDevicePayload>)>;

    /// Own the fixed identity until its publication is consumed. The device
    /// profile reports status only, while the host profile includes motion.
    /// Joint edits change authored settings at this boundary and remain in
    /// effect until changed again. They do not rewind with simulated motion.
    /// A required consumer prepares device work after the numerical receipt.
    /// In a mixed result it prepares work for the accepted worlds (status zero).
    /// A fully rejected batch remains available through TakePublication.
    /// The whole fixed publication becomes available once that work completes.
    /// This source performs admission and submission, then seals the consumer's
    /// outputs after its receipt and before releasing the fixed publication.
    /// Fixed input rows follow the enrolled world order and are captured at admission.
    void BeginFixed(std::vector<ContactStreamIdentity> identities, const std::vector<PhysicsSceneBodyCommand> &commands,
                    bool host, bool captureContacts = false,
                    const std::vector<InxPhyGpuSceneJointUpdate> &jointUpdates = {}, PublicationConsumer consumer = {},
                    std::vector<PhysicsFixedInputs> fixedInputs = {},
                    const std::vector<InxPhyGpuSceneImpulseBatch> &deviceImpulses = {},
                    const std::vector<InxPhyGpuSceneParticipants> &participants = {});
    /// Apply world-scoped body and joint edits at each world's accepted simulation time.
    /// Forces and kinematic targets wait for the next accepted fixed step, while
    /// pose edits publish immediately. Joint changes wake their endpoints on integration.
    void BeginEdits(const std::vector<PhysicsSceneBodyCommand> &commands,
                    const std::vector<InxPhyGpuSceneJointUpdate> &jointUpdates = {}, PublicationConsumer consumer = {});
    [[nodiscard]] bool HasPublication() const noexcept
    {
        return m_publication.has_value();
    }
    [[nodiscard]] std::optional<InxPhyGpuFixedPublication> TakePublication();
    [[nodiscard]] std::optional<InxPhyGpuHostBody> ReadHostBody(const InxPhyGpuFixedPublication &publication,
                                                                uint32_t world, uint32_t engineBody) const;
    /// Read requested bodies from a frozen device publication. A duplicate body
    /// uses the same readback and retains its position in the requested order.
    [[nodiscard]] InxPhyGpuSceneBodyRead PrepareBodyRead(const InxPhyGpuFixedPublication &publication, uint32_t world,
                                                         const std::vector<uint32_t> &engineBodies) const;
    /// Read the body motion and resolve its cooked leaves against the same publication.
    [[nodiscard]] InxPhyGpuSceneShapeRead PrepareBodyShapes(const InxPhyGpuFixedPublication &publication,
                                                            uint32_t world, uint32_t body, bool triggers) const;
    /// Resolve stable joint IDs in the selected accepted world and keep the
    /// requested order. The result can feed a device consumer without readback.
    [[nodiscard]] PhysicsGpuJointMotionRead PrepareJointMotion(const InxPhyGpuFixedPublication &publication,
                                                               uint32_t world, const std::vector<uint64_t> &jointIds,
                                                               bool readback) const;
    /// Prepare against the chosen completed world. The result retains device
    /// rows and can optionally carry an exact host hit readback.
    [[nodiscard]] InxPhyGpuSceneRayBatch PrepareRaycast(const InxPhyGpuFixedPublication &publication, uint32_t world,
                                                        const std::vector<InxPhyGpuRay> &rays, bool readback,
                                                        const PhysicsDeviceRayOutputs *outputs = nullptr,
                                                        uint32_t allHitCapacity = 0) const;
    [[nodiscard]] InxPhyGpuSceneRayBatch PrepareColliderRaycast(const InxPhyGpuFixedPublication &publication,
                                                                uint32_t world, uint32_t bodyId, uint64_t colliderId,
                                                                uint64_t generation, InxPhyGpuRay ray) const;
    /// Pack the selected float3 ranges on-device and retain the hits with this publication identity.
    [[nodiscard]] InxPhyGpuSceneRayBatch PrepareDeviceRaycast(const InxPhyGpuFixedPublication &publication,
                                                              uint32_t world, const InxPhyGpuDeviceRays &rays,
                                                              const PhysicsDeviceRayOutputs *outputs = nullptr) const;
    [[nodiscard]] InxPhyGpuSceneBodyBoundsRead PrepareBodyBounds(const InxPhyGpuFixedPublication &publication,
                                                                 uint32_t world, const ModelVector &minimum,
                                                                 const ModelVector &maximum, uint32_t layers,
                                                                 bool triggers) const;
    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch>
    PrepareClosestPoint(const InxPhyGpuFixedPublication &publication, uint32_t world, uint32_t bodyId,
                        uint64_t colliderId, uint64_t generation, const ModelVector &point, bool closedMesh) const;
    [[nodiscard]] InxPhyGpuSceneVolumeRead PrepareOverlap(const InxPhyGpuFixedPublication &publication, uint32_t world,
                                                          const InxPhyGpuQueryVolume &volume) const;
    [[nodiscard]] InxPhyGpuSceneVolumeRead PrepareShapeCast(const InxPhyGpuFixedPublication &publication,
                                                            uint32_t world, const InxPhyGpuQueryVolume &volume,
                                                            const ModelVector &motion, float maximum,
                                                            const InxPhyGpuSceneSweepSource *swept,
                                                            const PhysicsColliderPairs &ignored) const;
    [[nodiscard]] PublicationWork
    PrepareParticleCollision(const InxPhyGpuFixedPublication &publication, uint32_t world, const Scene *scene,
                             const std::shared_ptr<const particle::GpuParticleCollisionDevicePayload> &previous,
                             uint64_t revision, ParticleConsumer publish);
    [[nodiscard]] FrameWorkWaitResult WaitForProgress(uint64_t nanoseconds);
    void Retire() noexcept;
    [[nodiscard]] bool CanRetire() const noexcept;

  private:
    enum class Packet : uint8_t
    {
        Initialize,
        Preserve,
        Fixed,
        Edit,
        Reports,
        Consumer,
    };
    struct Operation
    {
        std::vector<ContactStreamIdentity> identities;
        std::vector<InxPhyGpuBodyCommand> commands;
        std::vector<InxPhyGpuJointParameterUpdate> jointUpdates;
        bool host;
        bool captureContacts;
        PublicationConsumer consumer;
        bool integrate;
        std::vector<PhysicsFixedInputs> fixedInputs;
        std::shared_ptr<const InxPhyGpuScenePoseTargets> poseTargets{};
        std::vector<std::pair<uint32_t, uint32_t>> pendingWrenchRows{};
        std::shared_ptr<const InxPhyGpuSceneKinematicTargets> kinematicTargets{};
        std::vector<std::pair<uint32_t, uint32_t>> kinematicVelocityRows{};
        std::vector<InxPhyGpuFixedStageWork> projectStages{};
        std::vector<std::function<void()>> acceptProjectStages{};
        std::vector<uint32_t> projectWorlds{};
        struct TargetCompletion
        {
            uint32_t world;
            PhysicsTargetSource target;
            uint8_t channel;
        };
        std::vector<TargetCompletion> targetCompletions{};
    };
    struct Receipt
    {
        rhi::ComputeCompletion completion;
        Packet kind;
        uint64_t opportunity, gpuGrant, transferGrant, transferBytes;
        bool ready = false;
    };

    [[nodiscard]] FrameWorkFacts Describe() const;
    void EncodeBodyCommand(uint32_t world, const PhysicsBodyCommand &command,
                           std::vector<InxPhyGpuBodyCommand> &out) const;
    void EncodeSceneBodyCommands(const std::vector<PhysicsSceneBodyCommand> &commands, Operation &operation) const;
    void EncodeJointUpdates(const std::vector<InxPhyGpuSceneJointUpdate> &jointUpdates,
                            std::vector<InxPhyGpuJointParameterUpdate> &updates,
                            std::shared_ptr<InxPhyGpuScenePoseTargets> &poses) const;
    void EncodeFixedParticipants(Operation &operation, const std::vector<InxPhyGpuSceneParticipants> &groups) const;
    [[nodiscard]] std::vector<std::pair<uint32_t, uint32_t>>
    KinematicVelocityRows(const std::vector<InxPhyGpuBodyCommand> &commands) const;
    void BeginOperation(Operation operation);
    bool Advance(bool gpu, uint64_t token);
    void Observe();
    bool Submit();
    void Finish();
    [[nodiscard]] InxPhyGpuFixedPublication CapturePublication(rhi::SubmissionTicket submission) const;
    void RetainWrenches(InxPhyGpuFixedPublication &publication, std::vector<std::pair<uint32_t, uint32_t>> rows) const;
    void FinishFixed(const rhi::PreparedComputeBatch &packet);
    void FinishReports(const rhi::PreparedComputeBatch &packet);
    void PrepareConsumer(InxPhyGpuFixedPublication publication);
    void ContinueConsumer(PublicationWork work);
    void PublishCompleted(InxPhyGpuFixedPublication publication);

    FrameWorkOwner &m_work;
    rhi::ComputeHost m_host;
    ShaderCodeTarget m_target;
    FrameWorkOwner::SourceId m_source = 0;
    std::unique_ptr<InxPhyGpuStep> m_step;
    std::unique_ptr<InxPhyGpuParticleExport> m_particles;
    std::shared_ptr<const InxPhyGpuSceneLayout> m_layout;
    std::shared_ptr<const InxPhyGpuScenePoseTargets> m_poseTargets;
    std::shared_ptr<const InxPhyGpuSceneKinematicTargets> m_kinematicTargets;
    std::vector<std::array<PhysicsTargetSource, 2>> m_reboundTargets;
    std::vector<std::pair<uint32_t, uint32_t>> m_pendingWrenchRows;
    std::vector<std::pair<uint32_t, uint32_t>> m_kinematicVelocityRows;
    uint32_t m_bodyCount;
    bool m_contactReportsConfigured;
    std::shared_ptr<rhi::PreparedComputeBatch> m_packet, m_preservation;
    std::optional<Receipt> m_receipt;
    std::optional<Operation> m_operation;
    std::vector<InxPhyGpuFixedWorld> m_acceptedWorlds;
    std::optional<InxPhyGpuFixedPublication> m_initialPublication;
    std::optional<InxPhyGpuFixedPublication> m_pendingPublication;
    std::optional<InxPhyGpuFixedPublication> m_publication;
    std::function<void()> m_sealConsumer;
    std::function<PublicationWork(const InxPhyGpuFixedPublication &)> m_continueConsumer;
    uint64_t m_generation = 1;
    uint64_t m_fixedGpuNanoseconds = 0;
    Packet m_packetKind = Packet::Initialize;
    bool m_retiring = false;
    bool m_projectStepFailed = false;
    std::vector<PhysicsFixedInputs> m_activeFixedInputs;
};

} // namespace infernux::physics::inxphygpu
