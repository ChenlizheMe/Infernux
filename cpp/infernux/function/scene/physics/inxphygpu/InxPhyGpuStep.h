// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../PhysicsDeviceBufferRange.h"
#include "../PhysicsFixedInputs.h"
#include "../PhysicsGpuJointMotion.h"
#include "../PhysicsGpuPoseSource.h"
#include "../PhysicsParticipantStage.h"
#include "InxPhyGpuModel.h"

#include <function/renderer/rhi/RhiComputeKernel.h>
#include <functional>
#include <unordered_map>
#include <variant>

namespace infernux
{
enum class ForceMode;
}

namespace infernux::physics::inxphygpu
{

class InxPhyGpuStep;
struct InxPhyGpuCollisionModel;

struct InxPhyGpuCollisionCapacity
{
    uint32_t candidates = 0; // Shared across the batch.
    uint32_t contactsPerWorld = 0;
};

struct InxPhyGpuContactSolver
{
    uint32_t rowsPerWorld = 0;
    uint32_t columnsPerWorld = 0;
    uint32_t iterations = 12;
    uint32_t lineIterations = 10;
    float tolerance = 1.0e-6f;
    float lineTolerance = 0.01f;
    float minimumRestitutionSpeed = 2.0f;
};

struct InxPhyGpuSleepSettings
{
    float quietTime = 0.5f;       // Seconds of continuous quiet required.
    float pointVelocity = 0.005f; // Maximum speed in m/s at a shape point.
};

struct InxPhyGpuContactStream
{
    // Counts retain the requested sizes even when a stream runs out of capacity.
    std::shared_ptr<rhi::ComputeBuffer> candidateCount;
    std::shared_ptr<rhi::ComputeBuffer> contactCounts;
    std::shared_ptr<rhi::ComputeBuffer> records;
    std::shared_ptr<rhi::ComputeBuffer> stepStatus;
    std::shared_ptr<rhi::ComputeBuffer> topology;
    std::shared_ptr<rhi::ComputeBuffer> narrowphaseStatus;
};

struct InxPhyGpuContactReport
{
    ModelVector pointSeparation, normal, relativeVelocity, impulse;
    ModelWords endpoints, identity;
};

enum InxPhyGpuRayFlags : uint32_t
{
    InxPhyGpuRayIncludeStatic = 1,
    InxPhyGpuRayIncludeTriggers = 2,
    InxPhyGpuRayHitBackFaces = 4,
    InxPhyGpuRayTargetGeometry = 8,
};

struct InxPhyGpuRay
{
    ModelVector originLimit{}; // xyz origin and maximum distance.
    ModelVector direction{};   // Unit direction in xyz.
    ModelWords filter{UINT32_MAX, NoModelIndex, InxPhyGpuRayIncludeStatic | InxPhyGpuRayIncludeTriggers, 0};
};

struct InxPhyGpuRayHit
{
    ModelVector normalDistance; // World normal and distance (-1 for a miss).
    ModelWords identity;        // Geometry row, body row, subshape ID, asset face index.
};

struct InxPhyGpuRayBatchLayout
{
    uint32_t raysPerWorld = 0;
    bool sharedRays = false; // One input row broadcast to every world.
};

struct InxPhyGpuRayBuffers
{
    std::shared_ptr<rhi::ComputeBuffer> inputs;
    std::shared_ptr<rhi::ComputeBuffer> hits;
};

struct InxPhyGpuPreparedRayBatch
{
    std::shared_ptr<rhi::PreparedComputeBatch> packet;
    std::shared_ptr<rhi::ComputeBuffer> hits;
    uint32_t count = 0;
};

struct InxPhyGpuPreparedBodyBounds
{
    std::shared_ptr<rhi::PreparedComputeBatch> packet;
    std::shared_ptr<const std::vector<uint32_t>> rows;

    [[nodiscard]] std::vector<uint32_t> ReadMatchingRows() const;
};

[[nodiscard]] std::vector<uint32_t> ReadInxPhyGpuSelection(const rhi::PreparedComputeBatch &packet, uint32_t count);

enum InxPhyGpuGeometryType : uint32_t;

struct InxPhyGpuQueryVolume
{
    ModelPose pose;
    ModelVector size{};
    InxPhyGpuGeometryType type;
    uint32_t layers;
    bool triggers;
};

struct InxPhyGpuShapeCastHit
{
    InxPhyGpuRayHit surface;
    ModelVector point; // World witness on the target, with numerical status in w.
};

struct InxPhyGpuDeviceRays
{
    PhysicsDeviceBufferRange origins, directions;
    uint32_t count;
    float maxDistance;
    ModelWords filter{UINT32_MAX, NoModelIndex, InxPhyGpuRayIncludeStatic | InxPhyGpuRayIncludeTriggers, 0};
};

struct PreparedInxPhyGpuRayQuery;
struct InxPhyGpuParticleGeometry;

class InxPhyGpuPairQuery final
{
  public:
    InxPhyGpuPairQuery(rhi::ComputeHost &host, ShaderCodeTarget target);
    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch> Prepare(rhi::ComputeHost &host,
                                                                     const std::vector<ModelWords> &data);

  private:
    rhi::ComputeDispatchDesc m_dispatch;
};

/// Queries consume an accepted pose buffer and their immutable geometry.
class InxPhyGpuPublishedRayQuery
{
  public:
    [[nodiscard]] static PreparedInxPhyGpuRayQuery
    Build(rhi::ComputeHost &host, ShaderCodeTarget target, const InxPhyGpuModel &model,
          const InxPhyGpuCollisionModel &collision, uint32_t capacity, uint32_t sourceBodyCount,
          const std::vector<uint32_t> &centerOfMassRows, const std::vector<uint32_t> &bodyBoundsRows);

    [[nodiscard]] InxPhyGpuPreparedRayBatch
    Prepare(rhi::ComputeHost &host, const PhysicsGpuPoseSource &source, uint32_t rayCount, bool readback,
            std::vector<rhi::ComputeBufferUpdate> updates, const InxPhyGpuDeviceRays *deviceRays = nullptr,
            const PhysicsDeviceRayOutputs *outputs = nullptr, uint32_t allHitCapacity = 0,
            const std::vector<InxPhyGpuRay> *hostRays = nullptr) const;
    [[nodiscard]] InxPhyGpuPreparedBodyBounds PrepareBodyBounds(rhi::ComputeHost &host,
                                                                const PhysicsGpuPoseSource &source,
                                                                const ModelVector &minimum, const ModelVector &maximum,
                                                                uint32_t layers, bool triggers,
                                                                std::vector<rhi::ComputeBufferUpdate> updates) const;
    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch>
    PrepareClosestPoint(rhi::ComputeHost &host, const PhysicsGpuPoseSource &source, const ModelVector &point,
                        uint32_t first, uint32_t end, bool closedMesh,
                        std::vector<rhi::ComputeBufferUpdate> updates) const;
    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch>
    PrepareOverlap(rhi::ComputeHost &host, const PhysicsGpuPoseSource &source, const InxPhyGpuQueryVolume &query,
                   std::vector<rhi::ComputeBufferUpdate> updates) const;
    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch>
    PrepareShapeCast(rhi::ComputeHost &host, ShaderCodeTarget target, const PhysicsGpuPoseSource &source,
                     const InxPhyGpuQueryVolume &volume, const ModelVector &direction, float maximum,
                     const std::vector<ModelWords> &exclusions, const std::function<uint32_t(uint32_t)> &bodyId,
                     std::vector<rhi::ComputeBufferUpdate> updates) const;
    [[nodiscard]] const std::shared_ptr<rhi::ComputeBuffer> &Inputs() const noexcept
    {
        return m_inputs;
    }
    [[nodiscard]] uint32_t Capacity() const noexcept
    {
        return m_capacity;
    }
    [[nodiscard]] InxPhyGpuParticleGeometry ParticleGeometry() const;
    [[nodiscard]] std::vector<rhi::ComputeDispatchDesc> TakeInitialization() const;

  private:
    friend class InxPhyGpuStep;
    void AppendGeometry(std::vector<rhi::ComputeDispatchDesc> &dispatches, const PhysicsGpuPoseSource &source) const;
    std::shared_ptr<rhi::ComputeBuffer> m_inputs, m_settings, m_inputSettings, m_outputSettings;
    std::shared_ptr<rhi::ComputeBuffer> m_boundsRequest, m_boundsMask, m_closestRequest, m_closestResult;
    std::shared_ptr<rhi::ComputeBuffer> m_overlapRequest, m_overlapMask;
    std::shared_ptr<rhi::ComputeBuffer> m_particlePoses, m_particleShapes, m_particleCold;
    rhi::ComputeDispatchDesc m_geometry, m_pack, m_project, m_traceAll;
    rhi::ComputeDispatchDesc m_selectBounds, m_closest, m_overlap;
    mutable rhi::ComputeDispatchDesc m_shapeCast;
    mutable std::unordered_map<uint64_t, std::shared_ptr<rhi::ComputeBuffer>> m_shapeCastRequests;
    std::vector<rhi::ComputeDispatchDesc> m_rayWork;
    mutable std::vector<rhi::ComputeDispatchDesc> m_initialize;
    mutable std::shared_ptr<rhi::ComputeBuffer> m_inputScratch, m_output;
    uint32_t m_geometryCount = 0, m_bodyCount = 0, m_capacity = 0;
    std::shared_ptr<const std::vector<uint32_t>> m_bodyBoundsRows;
    PhysicsGpuPoseSource::Layout m_layout = PhysicsGpuPoseSource::Layout::OriginWXYZ;
};

struct PreparedInxPhyGpuRayQuery
{
    InxPhyGpuPublishedRayQuery query;
    std::vector<rhi::ComputeBufferUpdate> initialize;
};

enum class InxPhyGpuBodyCommandKind : uint32_t
{
    SetPosition,
    SetRotation,
    SetLinearVelocity,
    SetAngularVelocity,
    Force,
    Torque,
    ForceAtPosition,
    MovePosition,
    MoveRotation,
    ResetMotion,
    Sleep,
    Wake,
    ApplyPendingWrench,
    AdvanceKinematicVelocity,
};

enum class InxPhyGpuPoseCommandFlags : uint32_t
{
    None,
    PreserveDrive,
    TemporaryStatic,
};

struct InxPhyGpuBodyCommand
{
    uint32_t world;
    uint32_t body;
    InxPhyGpuBodyCommandKind kind;
    union {
        ForceMode mode{};
        InxPhyGpuPoseCommandFlags poseFlags;
    };
    ModelVector value{}; // World xyz, or wxyz for rotations.
    ModelVector point{}; // Application point for force, or (duration, maxSpeed) for kinematic targets.
};

struct InxPhyGpuJointParameterUpdate
{
    uint32_t world;
    uint32_t offset; // In ModelWords records within one world.
    std::variant<ModelWords, ModelJointVectors, InxPhyGpuDistanceSpring, ModelSwingTwistControls, ModelSixDOFControls>
        value;
    std::array<uint32_t, 2> dynamicEndpoints{NoModelIndex, NoModelIndex};
};

struct InxPhyGpuDeviceBodyExchange
{
    uint32_t world;
    std::vector<uint32_t> bodies;
    // World COM, xyzw rotation and world linear/angular velocity in four float4 blocks.
    PhysicsDeviceBufferRange motion;
    // Linear and angular impulse about the COM, six float32 values per target body.
    PhysicsDeviceBufferRange impulses;
};

struct InxPhyGpuFixedStageWork
{
    uint32_t substep;
    PhysicsParticipantStage stage;
    std::vector<rhi::ComputeBufferUpdate> updates{};
    std::vector<rhi::ComputeDispatchDesc> dispatches{};
    std::optional<InxPhyGpuDeviceBodyExchange> bodies{};
};

enum class InxPhyGpuBodyPublication
{
    None,
    Device,
    Cpu,
    Status, // Keep body rows device-resident and read only the per-world headers.
};

struct InxPhyGpuBodyState
{
    ModelPose pose;
    ModelVector centerOfMass;    // xyz center of mass, w is 1.0 for a sleeping body.
    ModelVector linearVelocity;  // w carries dynamic quiet time or the kinematic velocity mode.
    ModelVector angularVelocity; // w carries the support count as raw bits.
};

struct PreparedInxPhyGpuStep
{
    std::unique_ptr<InxPhyGpuStep> step;
    // Submit once before the first step. The queue orders the initial values with later steps.
    std::shared_ptr<rhi::PreparedComputeBatch> initialize;
};

struct InxPhyGpuStateHandoff
{
    struct PoseTarget
    {
        uint32_t world, offset;
        ModelWords orientation;
    };

    std::shared_ptr<rhi::ComputeBuffer> states;
    // World-major, one source uint4 address per destination body. Body zero
    // maps the fixed-step header. NoModelIndex keeps the authored seed.
    std::vector<uint32_t> addresses;
    std::vector<PoseTarget> poseTargets{};
    std::shared_ptr<rhi::ComputeBuffer> pendingWrenches;
};

/// Resident state and fixed-step work for worlds that share one compiled model.
class InxPhyGpuStep
{
  public:
    /// Resident rigid dynamics with optional collision detection and constraint response.
    [[nodiscard]] static PreparedInxPhyGpuStep
    Build(rhi::ComputeHost &host, ShaderCodeTarget target, const InxPhyGpuModel &model, uint32_t worlds, float timestep,
          const ModelVector &gravity, const InxPhyGpuCollisionModel *collision = nullptr,
          InxPhyGpuCollisionCapacity capacity = {}, const InxPhyGpuContactSolver *solver = nullptr,
          InxPhyGpuRayBatchLayout rayLayout = {}, uint32_t bodyCommandCapacity = 0, bool contactReports = false,
          uint32_t substeps = 1, const InxPhyGpuStateHandoff &handoff = {}, InxPhyGpuSleepSettings sleep = {});

    /// The ordered snapshot holds qpos, qvel, clocks and, with contact response, a uint32 status per world.
    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch>
    Prepare(bool publish, std::vector<rhi::ComputeBufferUpdate> updates = {}) const;

    /// Commands for one body retain their submitted order. A rejected world
    /// restores its previous motions along with its fixed clock.
    /// A supplied input row per world applies to this step and remains until replaced.
    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch>
    PrepareControlled(std::vector<InxPhyGpuBodyCommand> commands, InxPhyGpuBodyPublication publication,
                      std::vector<rhi::ComputeBufferUpdate> updates = {}, bool captureContacts = false,
                      std::vector<InxPhyGpuJointParameterUpdate> jointUpdates = {},
                      const std::vector<PhysicsFixedInputs> &fixedInputs = {},
                      std::vector<InxPhyGpuFixedStageWork> projectStages = {}) const;

    /// Absolute edits and impulses publish new poses at the current fixed time.
    /// Time-dependent force and movement commands are admitted by the fixed step.
    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch>
    PrepareEdits(std::vector<InxPhyGpuBodyCommand> commands,
                 std::vector<InxPhyGpuJointParameterUpdate> jointUpdates = {}) const;

    /// A reporting step appends the world counts to its readbacks. Read only
    /// accepted worlds' populated ranges before issuing another step.
    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch>
    PrepareContactReports(const std::vector<uint32_t> &acceptedCounts) const;
    [[nodiscard]] bool HasContactReports() const noexcept
    {
        return bool(m_reports.records);
    }
    [[nodiscard]] uint32_t Substeps() const noexcept
    {
        return m_substeps.count;
    }

    /// World-major body rows (including the world row), followed by one
    /// ModelWords record per world with the clock bits, solver status and accepted step ordinal.
    [[nodiscard]] std::shared_ptr<rhi::ComputeBuffer> BodyStates() const noexcept;
    [[nodiscard]] InxPhyGpuParticleGeometry PublishedParticleGeometry() const;
    [[nodiscard]] const std::shared_ptr<rhi::ComputeBuffer> &PendingWrenches() const noexcept
    {
        return m_controls.pendingWrenches;
    }

    [[nodiscard]] PhysicsGpuJointMotionRead PrepareJointMotion(const PhysicsGpuPoseSource &source,
                                                               const std::vector<uint32_t> &rows, bool readback) const;

    /// Enqueue after the state to query. Device consumers enqueue before the next
    /// batch that writes the same hit buffer. A CPU publication pins its own snapshot.
    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch>
    PrepareRays(bool publish, std::vector<rhi::ComputeBufferUpdate> updates = {}) const;

    /// Query a frozen scene publication. Only the selected world is refit
    /// and only the requested ray prefix is dispatched or read back.
    [[nodiscard]] InxPhyGpuPreparedRayBatch
    PreparePublishedRays(const PhysicsGpuPoseSource &source, uint32_t rayCount, bool publish,
                         std::vector<rhi::ComputeBufferUpdate> updates = {},
                         const InxPhyGpuDeviceRays *deviceRays = nullptr,
                         const PhysicsDeviceRayOutputs *outputs = nullptr, uint32_t allHitCapacity = 0,
                         const std::vector<InxPhyGpuRay> *hostRays = nullptr) const;
    [[nodiscard]] InxPhyGpuPreparedBodyBounds PreparePublishedBodyBounds(const PhysicsGpuPoseSource &source,
                                                                         const ModelVector &minimum,
                                                                         const ModelVector &maximum, uint32_t layers,
                                                                         bool triggers) const;

    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch>
    PreparePublishedClosestPoint(const PhysicsGpuPoseSource &source, const ModelVector &point, uint32_t first,
                                 uint32_t end, bool closedMesh) const;
    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch>
    PreparePublishedOverlap(const PhysicsGpuPoseSource &source, const InxPhyGpuQueryVolume &volume) const;
    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch>
    PreparePublishedShapeCast(ShaderCodeTarget target, const PhysicsGpuPoseSource &source,
                              const InxPhyGpuQueryVolume &volume, const ModelVector &direction, float maximum,
                              const std::vector<ModelWords> &exclusions,
                              const std::function<uint32_t(uint32_t)> &bodyId) const;
    [[nodiscard]] const InxPhyGpuRayBuffers &Rays() const noexcept
    {
        return m_rays;
    }

    [[nodiscard]] const std::shared_ptr<rhi::ComputeBuffer> &Positions() const noexcept;
    [[nodiscard]] const std::shared_ptr<rhi::ComputeBuffer> &Velocities() const noexcept;
    [[nodiscard]] const std::shared_ptr<rhi::ComputeBuffer> &AppliedForces() const noexcept;
    [[nodiscard]] const InxPhyGpuContactStream &Contacts() const noexcept
    {
        return m_contacts;
    }

  private:
    struct BodyCommandStreams
    {
        std::shared_ptr<rhi::ComputeBuffer> commands, spans, saved;
        uint32_t capacity = 0;
    };
    struct BodyControls
    {
        BodyCommandStreams streams;
        std::shared_ptr<rhi::ComputeBuffer> states, activity, pendingWrenches;
        rhi::ComputeDispatchDesc apply, edit, finish, pose, publish, publishWithStatus, deviceMotion, deviceImpulse;
        std::vector<uint8_t> motion;
        uint32_t worlds = 0;
    };
    struct ContactReports
    {
        std::shared_ptr<rhi::ComputeBuffer> records, counts;
        rhi::ComputeDispatchDesc capture;
        uint64_t countsOffset = 0, countsBytes = 0;
        uint32_t capacity = 0;
    };
    struct SceneJoints
    {
        std::shared_ptr<rhi::ComputeBuffer> parameters, descriptors;
        std::shared_ptr<rhi::ComputeKernel> motionKernel;
        uint32_t stride = 0;

        void Append(std::vector<rhi::ComputeBufferUpdate> &updates,
                    std::vector<InxPhyGpuJointParameterUpdate> values) const;
    };
    struct FixedSubsteps
    {
        uint32_t count = 1;
        size_t collisionPasses = 0;
        std::vector<rhi::ComputeDispatchDesc> begin;
        rhi::ComputeDispatchDesc drive, advance, restore, restoreActivity, clock;
    };
    struct FixedInputs
    {
        std::shared_ptr<rhi::ComputeBuffer> timestep, substep, gravity, constraints;
        ModelWords constraintDefaults{};

        void Append(std::vector<rhi::ComputeBufferUpdate> &updates, const std::vector<PhysicsFixedInputs> &values,
                    uint32_t substeps) const;
    };

    using SubstepStages = std::array<std::vector<rhi::ComputeDispatchDesc>, 4>;
    void AppendFixedWork(std::vector<rhi::ComputeDispatchDesc> &, std::vector<rhi::ComputeReplayRange> &,
                         uint32_t spans, bool captureContacts, const BodyCommandStreams &streams,
                         std::vector<SubstepStages> stages = {}) const;
    [[nodiscard]] std::vector<SubstepStages> PrepareFixedStages(std::vector<InxPhyGpuFixedStageWork>,
                                                                std::vector<InxPhyGpuBodyCommand> &,
                                                                std::vector<rhi::ComputeBufferUpdate> &) const;
    [[nodiscard]] const BodyCommandStreams &BodyCommandStorage(size_t count) const;
    void BindBodyCommandStreams(rhi::ComputeDispatchDesc &dispatch, const BodyCommandStreams &streams,
                                uint32_t commands, uint32_t spans) const;
    uint32_t AppendBodyCommands(std::vector<InxPhyGpuBodyCommand> commands, bool integrate,
                                const BodyCommandStreams &streams, std::vector<rhi::ComputeBufferUpdate> &updates,
                                std::vector<rhi::ComputeDispatchDesc> &dispatches) const;
    void AppendJointEdits(std::vector<rhi::ComputeBufferUpdate> &updates,
                          std::vector<InxPhyGpuJointParameterUpdate> jointUpdates) const;
    void AppendBodyPublication(std::vector<rhi::ComputeDispatchDesc> &dispatches, bool includeStatus) const;

    InxPhyGpuStep(std::shared_ptr<rhi::ComputeWorkInstance> work, rhi::ComputeDispatchDesc snapshot, uint32_t positions,
                  uint32_t velocities, uint32_t appliedForces, InxPhyGpuContactStream contacts,
                  InxPhyGpuRayBuffers rays, std::vector<rhi::ComputeDispatchDesc> rayWork, BodyControls controls,
                  ContactReports reports, FixedSubsteps substeps, SceneJoints sceneJoints,
                  InxPhyGpuPublishedRayQuery publishedRays, FixedInputs fixedInputs);

    std::shared_ptr<rhi::ComputeWorkInstance> m_work;
    rhi::ComputeDispatchDesc m_snapshot;
    uint32_t m_positions;
    uint32_t m_velocities;
    uint32_t m_appliedForces;
    InxPhyGpuContactStream m_contacts;
    InxPhyGpuRayBuffers m_rays;
    std::vector<rhi::ComputeDispatchDesc> m_rayWork;
    BodyControls m_controls;
    mutable BodyCommandStreams m_expandedBodyCommands;
    ContactReports m_reports;
    FixedSubsteps m_substeps;
    std::shared_ptr<rhi::ComputeReplayCache> m_fixedReplay = rhi::MakeComputeReplayCache();
    std::shared_ptr<rhi::ComputeReplayCache> m_collisionReplay = rhi::MakeComputeReplayCache();
    std::shared_ptr<rhi::ComputeReplayCache> m_dynamicsReplay = rhi::MakeComputeReplayCache();
    SceneJoints m_sceneJoints;
    InxPhyGpuPublishedRayQuery m_publishedRays;
    FixedInputs m_fixedInputs;
};

} // namespace infernux::physics::inxphygpu
