#pragma once

#include "JoltGpuTick.h"

#include <array>

namespace infernux
{
struct JoltGpuContactTopologyStart;

enum class JoltGpuSweepOutcome : uint32_t
{
    NoImpact = JPH::GPU::SweepNoImpact,
    LastImpact = JPH::GPU::SweepLastImpact,
    MoreMotion = JPH::GPU::SweepMoreMotion
};

struct JoltGpuSweepResult
{
    JoltGpuSweepOutcome outcome = JoltGpuSweepOutcome::NoImpact;
    uint32_t integratedBodies = 0;
    uint32_t reportedCandidates = 0;
};

/// Completes generic candidates after the Tick's original analytic initializer.
/// Analytic families retain their existing kernels and budget planner; convex jobs keep
/// their real support/manifold cursor until all candidates are complete.
/// This producer only prepares work, never submits or waits on a queue.
class JoltGpuNarrowphase final
{
  public:
    struct Workspace
    {
        static constexpr uint32_t DispatchesPerPacket = 5u;
        static constexpr uint32_t AdvancesPerPacket = 8u;
        std::shared_ptr<rhi::ComputeBuffer> scratch, jobs, arguments;
        std::array<std::shared_ptr<rhi::ComputeBuffer>, DispatchesPerPacket> ranges;
        std::array<std::shared_ptr<rhi::ComputeBuffer>, AdvancesPerPacket> advanceRanges;
        double advanceNanoseconds = 0;
    };

    JoltGpuNarrowphase(rhi::ComputeHost &host, const ShaderTargetCode &target);

    [[nodiscard]] JoltGpuControlWork Prepare(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
                                             FrameWorkOwner &budget, Workspace &resident,
                                             JoltGpuContactTopologyStart topology, std::function<void()> completed);
    [[nodiscard]] JoltGpuControlWork PrepareSweeps(const JoltGpuStorage &storage, JoltGpuStepInputs &inputs,
                                                   JoltGpuPrograms &programs, FrameWorkOwner &budget, bool continuing,
                                                   bool reportImpacts,
                                                   std::function<JoltGpuControlWork()> retryCandidates,
                                                   std::function<void(JoltGpuSweepResult)> completed);

  private:
    class Impl;
    std::shared_ptr<Impl> m_impl;
};

} // namespace infernux
