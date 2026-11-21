// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "WarpModel.h"

#include <function/renderer/rhi/RhiComputeKernel.h>

namespace infernux::physics::warp
{

class WarpStep;

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
    /// Smooth rigid dynamics and Euler integration. Contact stages attach before integration.
    [[nodiscard]] static PreparedWarpStep Build(rhi::ComputeHost &host, ShaderCodeTarget target, const WarpModel &model,
                                                uint32_t worlds, float timestep, const ModelVector &gravity);

    /// An optional tail copy publishes qpos, qvel, and clocks in that order, each in world order.
    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch>
    Prepare(bool publish, std::vector<rhi::ComputeBufferUpdate> updates = {}) const;

    [[nodiscard]] const std::shared_ptr<rhi::ComputeBuffer> &Positions() const noexcept;
    [[nodiscard]] const std::shared_ptr<rhi::ComputeBuffer> &Velocities() const noexcept;
    [[nodiscard]] const std::shared_ptr<rhi::ComputeBuffer> &AppliedForces() const noexcept;

  private:
    WarpStep(std::shared_ptr<rhi::ComputeWorkInstance> work, rhi::ComputeDispatchDesc snapshot, uint32_t positions,
             uint32_t velocities, uint32_t appliedForces);

    std::shared_ptr<rhi::ComputeWorkInstance> m_work;
    rhi::ComputeDispatchDesc m_snapshot;
    uint32_t m_positions;
    uint32_t m_velocities;
    uint32_t m_appliedForces;
};

} // namespace infernux::physics::warp
