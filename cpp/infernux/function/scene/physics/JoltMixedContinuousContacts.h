#pragma once

#include "JoltMixedContacts.h"

namespace infernux
{
/// Shares the selected world's retained continuous motion. All device work
/// stays on its original Tick and all CPU updates finish the same native step.
class JoltMixedContinuousContacts final
{
  public:
    JoltMixedContinuousContacts(rhi::ComputeHost &, const ShaderTargetCode &);
    [[nodiscard]] JoltGpuControlWork Prepare(const JoltGpuStorage &, JoltGpuStepInputs &, const JPH::GPU::StepParams &,
                                             uint32_t substep, const std::shared_ptr<const JoltGpuScene::Publication> &,
                                             JoltGpuMixedContacts &, JoltMixedContactStep &,
                                             JoltCpuContactPublication &, const JoltMixedContactStep::Policy &,
                                             std::function<void()>);

  private:
    struct Job;
    struct Program
    {
        std::shared_ptr<rhi::ComputeKernel> kernel;
        std::vector<uint32_t> ports;
    };
    rhi::ComputeHost m_host;
    std::array<Program, 4> m_programs;
};
} // namespace infernux
