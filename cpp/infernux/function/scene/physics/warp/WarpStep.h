// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "WarpModel.h"

#include <function/renderer/rhi/RhiComputeKernel.h>

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
                                                const WarpContactSolver *solver = nullptr);

    /// The ordered snapshot holds qpos, qvel, clocks and, with contact response, a uint32 status per world.
    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch>
    Prepare(bool publish, std::vector<rhi::ComputeBufferUpdate> updates = {}) const;

    [[nodiscard]] const std::shared_ptr<rhi::ComputeBuffer> &Positions() const noexcept;
    [[nodiscard]] const std::shared_ptr<rhi::ComputeBuffer> &Velocities() const noexcept;
    [[nodiscard]] const std::shared_ptr<rhi::ComputeBuffer> &AppliedForces() const noexcept;
    [[nodiscard]] const WarpContactStream &Contacts() const noexcept
    {
        return m_contacts;
    }

  private:
    WarpStep(std::shared_ptr<rhi::ComputeWorkInstance> work, rhi::ComputeDispatchDesc snapshot, uint32_t positions,
             uint32_t velocities, uint32_t appliedForces, WarpContactStream contacts);

    std::shared_ptr<rhi::ComputeWorkInstance> m_work;
    rhi::ComputeDispatchDesc m_snapshot;
    uint32_t m_positions;
    uint32_t m_velocities;
    uint32_t m_appliedForces;
    WarpContactStream m_contacts;
};

} // namespace infernux::physics::warp
