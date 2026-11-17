#pragma once

#include "JoltMixedContacts.h"

namespace infernux
{
class JoltMixedConstraintSolver final
{
  public:
    JoltMixedConstraintSolver(rhi::ComputeHost &, const ShaderTargetCode &);
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
    Program m_exchange, m_wake;
};
} // namespace infernux
