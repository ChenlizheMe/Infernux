// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../PhysicsBodyState.h"
#include "../PhysicsContactTypes.h"
#include "../PhysicsECSStore.h"
#include "../PhysicsGpuPoseSource.h"
#include "../PhysicsRaycastHit.h"
#include "WarpSceneModel.h"
#include "WarpStep.h"

#include <core/threading/FrameWorkOwner.h>
#include <functional>

namespace infernux::physics::warp
{

struct WarpSceneLayout;

struct WarpRegisteredBody
{
    uint32_t bodyId;
    uint64_t shapeRevision;
    PhysicsActorHandle actor{};
    std::vector<WarpSceneSurfaceIdentity> surfaces{}; // Author collider order.
};

struct WarpRegisteredHinge
{
    uint64_t jointId;
    uint32_t authorRow;
};

struct WarpRegisteredWorld
{
    uint64_t worldId;
    std::vector<WarpRegisteredBody> bodies; // Scene model author order.
    std::vector<WarpRegisteredHinge> hinges{};
    uint64_t simulationEpoch = 0;
};

struct WarpSceneDriveUpdate
{
    uint32_t world;
    uint64_t jointId;
    WarpHingeDrive drive;
};

struct WarpFixedWorld
{
    ContactStreamIdentity identity;
    float time;
    uint32_t status;
    uint32_t acceptedStep;
};

struct WarpFixedPublication
{
    rhi::SubmissionTicket submission;
    std::shared_ptr<rhi::ComputeBuffer> bodyStates;
    std::vector<WarpFixedWorld> worlds;
    std::vector<WarpBodyState> hostBodies;
    std::vector<std::vector<WarpContactReport>> contacts; // Substep-major, then enrolled world.
    uint32_t bodiesPerWorld = 0;
    uint32_t substeps = 1;

    [[nodiscard]] PhysicsGpuPoseSource PoseSource(uint32_t world) const;
    [[nodiscard]] std::optional<PhysicsGpuBodyPoseBinding> FindPoseBody(uint32_t world, PhysicsActorHandle actor,
                                                                        uint32_t bodyId) const;

  private:
    friend class WarpSceneRuntime;
    std::shared_ptr<const WarpSceneLayout> m_layout;
};

struct WarpHostBody
{
    PhysicsBodyPoseUpdate pose;
    uint64_t shapeRevision;
    glm::vec3 centerOfMass, linearVelocity, angularVelocity;
};

struct WarpSceneBodyRead
{
    ContactStreamIdentity identity;
    std::shared_ptr<rhi::PreparedComputeBatch> packet;

    /// Consume completed motion rows in the caller's requested body order.
    [[nodiscard]] std::vector<WarpHostBody> ReadHostBodies() const;

  private:
    friend class WarpSceneRuntime;
    struct Body
    {
        uint32_t id, packedRow;
        uint64_t shapeRevision;
    };
    std::vector<Body> m_bodies;
    std::vector<uint32_t> m_rangeRows;
    uint32_t m_uniqueRows = 0;
};

struct WarpSceneRayBatch
{
    ContactStreamIdentity identity;
    WarpPreparedRayBatch query;

    /// Materialize the ordinary scene hits on the owner thread after submission.
    /// Consumes the host readback. The batch retains the queried world's layout.
    [[nodiscard]] std::vector<RaycastHit> ReadHostHits() const;

  private:
    friend class WarpSceneRuntime;
    struct RayLine
    {
        glm::vec3 origin, direction;
    };
    std::shared_ptr<const WarpSceneLayout> m_layout;
    std::vector<RayLine> m_lines;
    uint32_t m_world = 0;
};

/// Fixed-step source for one compiled layout and its enrolled scene worlds.
/// Admission and receipt collection use the engine's FrameWorkOwner.
class WarpSceneRuntime final
{
  public:
    WarpSceneRuntime(FrameWorkOwner &work, rhi::ComputeHost &host, ShaderCodeTarget target, const WarpSceneModel &scene,
                     std::vector<WarpRegisteredWorld> worlds, float timestep, const ModelVector &gravity,
                     WarpCollisionCapacity collisionCapacity, const WarpContactSolver &solver, uint32_t commandCapacity,
                     bool contactReports = false, uint32_t substeps = 1, WarpRayBatchLayout rayLayout = {});
    ~WarpSceneRuntime();
    WarpSceneRuntime(const WarpSceneRuntime &) = delete;
    WarpSceneRuntime &operator=(const WarpSceneRuntime &) = delete;

    [[nodiscard]] FrameWorkOwner::SourceId SourceId() const noexcept
    {
        return m_source;
    }
    [[nodiscard]] std::optional<uint32_t> BodyRow(uint32_t world, uint32_t engineBody) const;

    /// Take the initialized poses before admitting the first fixed step.
    /// The startup receipt has fixed index zero and requires no readback.
    [[nodiscard]] std::optional<WarpFixedPublication> TakeInitial();

    using PublicationConsumer =
        std::function<std::shared_ptr<rhi::PreparedComputeBatch>(const WarpFixedPublication &completed)>;

    /// Own the fixed identity until its publication is consumed. The device
    /// profile reports status only, while the host profile includes motion.
    /// Drive edits change authored settings at this boundary and remain in
    /// effect until changed again. They do not rewind with simulated motion.
    /// A required consumer prepares device work after the numerical receipt.
    /// The whole fixed publication becomes available once that work completes.
    /// The callback only prepares the packet. This source performs admission and submission.
    void BeginFixed(std::vector<ContactStreamIdentity> identities, std::vector<WarpBodyCommand> commands, bool host,
                    bool captureContacts = false, std::vector<WarpSceneDriveUpdate> drives = {},
                    PublicationConsumer consumer = {});
    /// Apply immediate body edits at the last accepted simulation time.
    /// Reuses the same completion and optional consumer as a fixed step.
    void BeginEdits(std::vector<WarpBodyCommand> commands, PublicationConsumer consumer = {});
    [[nodiscard]] std::optional<WarpFixedPublication> TakePublication();
    [[nodiscard]] std::optional<WarpHostBody> ReadHostBody(const WarpFixedPublication &publication, uint32_t world,
                                                           uint32_t engineBody) const;
    /// Read only requested bodies from a frozen device publication. One packet
    /// copies adjacent rows together and reads each selected row at most once.
    [[nodiscard]] WarpSceneBodyRead PrepareBodyRead(const WarpFixedPublication &publication, uint32_t world,
                                                    const std::vector<uint32_t> &engineBodies) const;
    /// Prepare against the chosen completed world. The result retains device
    /// rows and can optionally carry an exact host hit readback.
    [[nodiscard]] WarpSceneRayBatch PrepareRaycast(const WarpFixedPublication &publication, uint32_t world,
                                                   const std::vector<WarpRay> &rays, bool readback) const;
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
        std::vector<WarpBodyCommand> commands;
        std::vector<WarpHingeDriveUpdate> drives;
        bool host;
        bool captureContacts;
        PublicationConsumer consumer;
        bool integrate;
    };
    struct Receipt
    {
        rhi::ComputeCompletion completion;
        Packet kind;
        uint64_t opportunity, gpuGrant, transferGrant, transferBytes;
        bool ready = false;
    };

    [[nodiscard]] FrameWorkFacts Describe() const;
    void BeginOperation(Operation operation);
    bool Advance(bool gpu, uint64_t token);
    void Observe();
    bool Submit();
    void Finish();
    [[nodiscard]] WarpFixedPublication CapturePublication(const rhi::PreparedComputeBatch &packet) const;
    void FinishFixed(const rhi::PreparedComputeBatch &packet);
    void FinishReports(const rhi::PreparedComputeBatch &packet);
    void PrepareConsumer(WarpFixedPublication publication);
    void PublishCompleted(WarpFixedPublication publication);

    FrameWorkOwner &m_work;
    rhi::ComputeHost m_host;
    FrameWorkOwner::SourceId m_source = 0;
    std::unique_ptr<WarpStep> m_step;
    std::shared_ptr<const WarpSceneLayout> m_layout;
    uint32_t m_bodyCount;
    bool m_contactReportsConfigured;
    std::shared_ptr<rhi::PreparedComputeBatch> m_packet, m_preservation;
    std::optional<Receipt> m_receipt;
    std::optional<Operation> m_operation;
    std::vector<WarpFixedWorld> m_acceptedWorlds;
    std::optional<WarpFixedPublication> m_initialPublication;
    std::optional<WarpFixedPublication> m_pendingPublication;
    std::optional<WarpFixedPublication> m_publication;
    uint64_t m_generation = 1;
    uint64_t m_fixedGpuNanoseconds = 0;
    Packet m_packetKind = Packet::Initialize;
    bool m_retiring = false;
};

} // namespace infernux::physics::warp
