#pragma once

#include "JoltGpuTick.h"

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
    JoltGpuNarrowphase(rhi::ComputeHost &host, const ShaderTargetCode &target);

    [[nodiscard]] JoltGpuControlWork Prepare(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
                                             FrameWorkOwner &budget, std::function<void()> completed);
    [[nodiscard]] JoltGpuControlWork PrepareSweeps(const JoltGpuStorage &storage, JoltGpuStepInputs &inputs,
                                                   FrameWorkOwner &budget, float remainingSeconds, bool continuing,
                                                   bool reportImpacts,
                                                   std::function<void(std::optional<float>)> completed);

  private:
    class Impl;
    std::shared_ptr<Impl> m_impl;
};

} // namespace infernux
