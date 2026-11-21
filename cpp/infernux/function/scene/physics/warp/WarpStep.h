// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "WarpModel.h"

#include <function/renderer/rhi/RhiComputeKernel.h>

namespace infernux
{
enum class ForceMode;
}

namespace infernux::physics::warp
{

class WarpStep;
struct WarpCollisionModel;

struct WarpCollisionCapacity
{
    uint32_t candidates = 0; // Shared across the batch.
    uint32_t contactsPerWorld = 0;
};

struct WarpContactSolver
{
    uint32_t rowsPerWorld = 0;
    uint32_t columnsPerWorld = 0;
    uint32_t iterations = 12;
    uint32_t lineIterations = 10;
    float tolerance = 1.0e-6f;
    float lineTolerance = 0.01f;
};

struct WarpContactStream
{
    // Counts retain the requested sizes even when a stream runs out of capacity.
    std::shared_ptr<rhi::ComputeBuffer> candidateCount;
    std::shared_ptr<rhi::ComputeBuffer> contactCounts;
    std::shared_ptr<rhi::ComputeBuffer> records;
    std::shared_ptr<rhi::ComputeBuffer> stepStatus;
    std::shared_ptr<rhi::ComputeBuffer> topology;
    std::shared_ptr<rhi::ComputeBuffer> narrowphaseStatus;
};

struct WarpContactReport
{
    ModelVector pointSeparation, normal, relativeVelocity, impulse;
    ModelWords endpoints, identity;
};

enum WarpRayFlags : uint32_t
{
    WarpRayIncludeStatic = 1,
    WarpRayIncludeTriggers = 2,
    WarpRayHitBackFaces = 4,
};

struct WarpRay
{
    ModelVector originLimit{}; // xyz origin and maximum distance.
    ModelVector direction{};   // Unit direction in xyz.
    ModelWords filter{UINT32_MAX, NoModelIndex, WarpRayIncludeStatic | WarpRayIncludeTriggers, 0};
};

struct WarpRayHit
{
    ModelVector normalDistance; // World normal and distance (-1 for a miss).
    ModelWords identity;        // Geometry row, body row, subshape ID, asset face index.
};

struct WarpRayBatchLayout
{
    uint32_t raysPerWorld = 0;
    bool sharedRays = false; // One input row broadcast to every world.
};

struct WarpRayBuffers
{
    std::shared_ptr<rhi::ComputeBuffer> inputs;
    std::shared_ptr<rhi::ComputeBuffer> hits;
};

enum class WarpBodyCommandKind : uint32_t
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
};

struct WarpBodyCommand
{
    uint32_t world;
    uint32_t body;
    WarpBodyCommandKind kind;
    ForceMode mode{};
    ModelVector value{}; // World xyz, or wxyz for rotations.
    ModelVector point{}; // World point for ForceAtPosition.
};

enum class WarpBodyPublication
{
    None,
    Device,
    Cpu,
    Status, // Keep body rows device-resident and read only the per-world headers.
};

struct WarpBodyState
{
    ModelPose pose;
    ModelVector centerOfMass;
    ModelVector linearVelocity;
    ModelVector angularVelocity;
};

struct PreparedWarpStep
{
    std::unique_ptr<WarpStep> step;
    // Submit once before the first step. The queue orders the initial values with later steps.
    std::shared_ptr<rhi::PreparedComputeBatch> initialize;
};

/// Resident state and fixed-step work for worlds that share one compiled model.
class WarpStep
{
  public:
    /// Resident rigid dynamics with optional collision detection and constraint response.
    [[nodiscard]] static PreparedWarpStep Build(rhi::ComputeHost &host, ShaderCodeTarget target, const WarpModel &model,
                                                uint32_t worlds, float timestep, const ModelVector &gravity,
                                                const WarpCollisionModel *collision = nullptr,
                                                WarpCollisionCapacity capacity = {},
                                                const WarpContactSolver *solver = nullptr,
                                                WarpRayBatchLayout rayLayout = {}, uint32_t bodyCommandCapacity = 0,
                                                bool contactReports = false, uint32_t substeps = 1);

    /// The ordered snapshot holds qpos, qvel, clocks and, with contact response, a uint32 status per world.
    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch>
    Prepare(bool publish, std::vector<rhi::ComputeBufferUpdate> updates = {}) const;

    /// Commands for one body retain their submitted order. A rejected world
    /// restores its previous motions along with its fixed clock.
    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch>
    PrepareControlled(std::vector<WarpBodyCommand> commands, WarpBodyPublication publication,
                      std::vector<rhi::ComputeBufferUpdate> updates = {}, bool captureContacts = false) const;

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

    /// Enqueue after the state to query. Device consumers enqueue before the next
    /// batch that writes the same hit buffer. A CPU publication pins its own snapshot.
    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch>
    PrepareRays(bool publish, std::vector<rhi::ComputeBufferUpdate> updates = {}) const;

    [[nodiscard]] const WarpRayBuffers &Rays() const noexcept
    {
        return m_rays;
    }

    [[nodiscard]] const std::shared_ptr<rhi::ComputeBuffer> &Positions() const noexcept;
    [[nodiscard]] const std::shared_ptr<rhi::ComputeBuffer> &Velocities() const noexcept;
    [[nodiscard]] const std::shared_ptr<rhi::ComputeBuffer> &AppliedForces() const noexcept;
    [[nodiscard]] const WarpContactStream &Contacts() const noexcept
    {
        return m_contacts;
    }

  private:
    struct BodyControls
    {
        std::shared_ptr<rhi::ComputeBuffer> commands, spans, states;
        rhi::ComputeDispatchDesc apply, finish;
        std::vector<rhi::ComputeDispatchDesc> publication;
        std::vector<uint8_t> motion;
        uint32_t capacity = 0;
        uint32_t worlds = 0;
        uint32_t timestep = 0;
    };
    struct ContactReports
    {
        std::shared_ptr<rhi::ComputeBuffer> records, counts;
        rhi::ComputeDispatchDesc capture;
        uint64_t countsOffset = 0, countsBytes = 0;
        uint32_t capacity = 0;
    };
    struct FixedSubsteps
    {
        uint32_t count = 1;
        std::vector<rhi::ComputeDispatchDesc> begin;
        rhi::ComputeDispatchDesc drive, advance, restore, clock;
    };

    void AppendFixedWork(std::vector<rhi::ComputeDispatchDesc> &, uint32_t spans, bool captureContacts) const;

    WarpStep(std::shared_ptr<rhi::ComputeWorkInstance> work, rhi::ComputeDispatchDesc snapshot, uint32_t positions,
             uint32_t velocities, uint32_t appliedForces, WarpContactStream contacts, WarpRayBuffers rays,
             std::vector<rhi::ComputeDispatchDesc> rayWork, BodyControls controls, ContactReports reports,
             FixedSubsteps substeps);

    std::shared_ptr<rhi::ComputeWorkInstance> m_work;
    rhi::ComputeDispatchDesc m_snapshot;
    uint32_t m_positions;
    uint32_t m_velocities;
    uint32_t m_appliedForces;
    WarpContactStream m_contacts;
    WarpRayBuffers m_rays;
    std::vector<rhi::ComputeDispatchDesc> m_rayWork;
    BodyControls m_controls;
    ContactReports m_reports;
    FixedSubsteps m_substeps;
};

} // namespace infernux::physics::warp
