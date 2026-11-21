// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../PhysicsBodyCommand.h"
#include "../PhysicsBodyState.h"
#include "../PhysicsContactPublication.h"
#include "../PhysicsECSStore.h"
#include "../PhysicsGpuRenderBindings.h"
#include "../PhysicsRaycastHit.h"
#include "InxPhyGpuSceneModel.h"
#include "InxPhyGpuStep.h"

#include <core/threading/FrameWorkOwner.h>
#include <functional>
#include <variant>

namespace infernux::physics::inxphygpu
{

struct InxPhyGpuSceneLayout;
struct InxPhyGpuScenePoseTargets;

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

struct InxPhyGpuJointPoseTarget
{
    // Child rotation relative to the connected body. Empty restores the authored motor target.
    std::optional<glm::quat> relative;
};

struct InxPhyGpuSceneJointUpdate
{
    uint64_t worldId;
    uint64_t jointId;
    std::variant<InxPhyGpuJointLimits, InxPhyGpuHingeDrive, InxPhyGpuDistanceSpring, InxPhyGpuSwingTwistParameters,
                 InxPhyGpuSwingTwistTargets, ModelSixDOFControls, InxPhyGpuJointPoseTarget>
        value;
};

struct InxPhyGpuSceneBodyCommand
{
    uint64_t worldId;
    PhysicsBodyCommand command;
};

struct InxPhyGpuFixedWorld
{
    ContactStreamIdentity identity;
    float time;
    uint32_t status;
    uint32_t acceptedStep;
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
    std::vector<std::pair<uint32_t, uint32_t>> m_pendingWrenchRows;
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

struct InxPhyGpuSceneRayBatch
{
    ContactStreamIdentity identity;
    InxPhyGpuPreparedRayBatch query;

    /// Materialize the ordinary scene hits on the owner thread after submission.
    /// Consumes the host readback. The batch retains the queried world's layout.
    [[nodiscard]] std::vector<RaycastHit> ReadHostHits() const;

  private:
    friend class InxPhyGpuSceneRuntime;
    struct RayLine
    {
        glm::vec3 origin, direction;
    };
    std::shared_ptr<const InxPhyGpuSceneLayout> m_layout;
    std::vector<RayLine> m_lines;
    uint32_t m_world = 0;
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
    };
    using PublicationConsumer = std::function<PublicationWork(const InxPhyGpuFixedPublication &completed)>;

    /// Own the fixed identity until its publication is consumed. The device
    /// profile reports status only, while the host profile includes motion.
    /// Joint edits change authored settings at this boundary and remain in
    /// effect until changed again. They do not rewind with simulated motion.
    /// A required consumer prepares device work after the numerical receipt.
    /// The whole fixed publication becomes available once that work completes.
    /// This source performs admission and submission, then seals the consumer's
    /// outputs after its receipt and before releasing the fixed publication.
    /// Fixed input rows follow the enrolled world order and are captured at admission.
    void BeginFixed(std::vector<ContactStreamIdentity> identities,
                    const std::vector<InxPhyGpuSceneBodyCommand> &commands, bool host, bool captureContacts = false,
                    const std::vector<InxPhyGpuSceneJointUpdate> &jointUpdates = {}, PublicationConsumer consumer = {},
                    std::vector<PhysicsFixedInputs> fixedInputs = {});
    /// Apply body edits at the last accepted simulation time. Force and
    /// acceleration accumulate device wrenches for the next accepted fixed step.
    void BeginEdits(uint64_t worldId, const std::vector<PhysicsBodyCommand> &commands,
                    PublicationConsumer consumer = {});
    [[nodiscard]] std::optional<InxPhyGpuFixedPublication> TakePublication();
    [[nodiscard]] std::optional<InxPhyGpuHostBody> ReadHostBody(const InxPhyGpuFixedPublication &publication,
                                                                uint32_t world, uint32_t engineBody) const;
    /// Read requested bodies from a frozen device publication. A duplicate body
    /// uses the same readback and retains its position in the requested order.
    [[nodiscard]] InxPhyGpuSceneBodyRead PrepareBodyRead(const InxPhyGpuFixedPublication &publication, uint32_t world,
                                                         const std::vector<uint32_t> &engineBodies) const;
    /// Resolve stable joint IDs in the selected accepted world and keep the
    /// requested order. The result can feed a device consumer without readback.
    [[nodiscard]] InxPhyGpuPreparedJointMotion PrepareJointMotion(const InxPhyGpuFixedPublication &publication,
                                                                  uint32_t world, const std::vector<uint64_t> &jointIds,
                                                                  bool readback) const;
    /// Prepare against the chosen completed world. The result retains device
    /// rows and can optionally carry an exact host hit readback.
    [[nodiscard]] InxPhyGpuSceneRayBatch PrepareRaycast(const InxPhyGpuFixedPublication &publication, uint32_t world,
                                                        const std::vector<InxPhyGpuRay> &rays, bool readback) const;
    /// Pack the selected float3 ranges on-device and retain the hits with this publication identity.
    [[nodiscard]] InxPhyGpuSceneRayBatch PrepareDeviceRaycast(const InxPhyGpuFixedPublication &publication,
                                                              uint32_t world, const InxPhyGpuDeviceRays &rays,
                                                              const PhysicsDeviceRayOutputs *outputs = nullptr) const;
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
    void BeginOperation(Operation operation);
    bool Advance(bool gpu, uint64_t token);
    void Observe();
    bool Submit();
    void Finish();
    [[nodiscard]] InxPhyGpuFixedPublication CapturePublication(const rhi::PreparedComputeBatch &packet) const;
    void RetainWrenches(InxPhyGpuFixedPublication &publication, std::vector<std::pair<uint32_t, uint32_t>> rows) const;
    void FinishFixed(const rhi::PreparedComputeBatch &packet);
    void FinishReports(const rhi::PreparedComputeBatch &packet);
    void PrepareConsumer(InxPhyGpuFixedPublication publication);
    void PublishCompleted(InxPhyGpuFixedPublication publication);

    FrameWorkOwner &m_work;
    rhi::ComputeHost m_host;
    FrameWorkOwner::SourceId m_source = 0;
    std::unique_ptr<InxPhyGpuStep> m_step;
    std::shared_ptr<const InxPhyGpuSceneLayout> m_layout;
    std::shared_ptr<const InxPhyGpuScenePoseTargets> m_poseTargets;
    std::vector<std::pair<uint32_t, uint32_t>> m_pendingWrenchRows;
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
    uint64_t m_generation = 1;
    uint64_t m_fixedGpuNanoseconds = 0;
    Packet m_packetKind = Packet::Initialize;
    bool m_retiring = false;
    std::vector<PhysicsFixedInputs> m_activeFixedInputs;
};

} // namespace infernux::physics::inxphygpu
