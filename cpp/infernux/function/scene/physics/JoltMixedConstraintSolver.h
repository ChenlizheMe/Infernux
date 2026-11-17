#pragma once

#include "JoltMixedContacts.h"

namespace infernux
{
class JoltMixedConstraintSolver final
{
  public:
    JoltMixedConstraintSolver(rhi::ComputeHost &, const ShaderTargetCode &);
    [[nodiscard]] bool TryPreparePrograms(bool velocity, bool position, bool wake);
    [[nodiscard]] decltype(JoltGpuTickControl::prepareCoupledContacts)
    Prepare(const std::shared_ptr<const JoltGpuScene::Publication> &, JoltGpuMixedContacts &, JoltMixedContactStep &,
            JoltCpuContactPublication &, const JoltMixedContactStep::Policy &);

  private:
    struct Job;
    struct Program
    {
        std::shared_ptr<rhi::ComputeKernel> kernel;
        std::vector<uint32_t> ports;
    };
    rhi::ComputeHost m_host;
    Program m_velocity, m_position, m_wake;
};
} // namespace infernux
