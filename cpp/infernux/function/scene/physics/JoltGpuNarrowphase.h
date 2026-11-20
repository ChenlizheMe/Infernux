#pragma once

#include "JoltGpuTick.h"

#include <array>
#include <optional>

namespace infernux
{

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
                                             std::function<void()> completed);
    [[nodiscard]] JoltGpuControlWork PrepareSweeps(const JoltGpuStorage &storage, JoltGpuStepInputs &inputs,
                                                   FrameWorkOwner &budget, float remainingSeconds, bool continuing,
                                                   std::function<void(std::optional<float>)> completed);
    [[nodiscard]] JoltGpuControlWork PrepareSweepEvent(const JoltGpuStorage &storage, JoltGpuStepInputs &inputs,
                                                       FrameWorkOwner &budget, float fraction, float remainingSeconds,
                                                       bool reportImpacts, std::function<void()> completed);

  private:
    class Impl;
    std::shared_ptr<Impl> m_impl;
};

} // namespace infernux
