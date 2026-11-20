#pragma once

#include <function/scene/physics/JoltGpuTick.h>

namespace infernux
{
struct JoltGpuConstraintWorkset;

/// Builds one substep topology, then resumes that SAME validated CC execution
/// after wake and numerical Setup to publish the active solver schedule.
/// No submission, polling loop, private ComputeHost or CPU graph reconstruction.
class JoltGpuConstraints final
{
  public:
    JoltGpuConstraints(rhi::ComputeHost &host, const ShaderTargetCode &target);

    [[nodiscard]] JoltGpuControlWork
    PrepareTopology(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
                    const JPH::GPU::StepParams &parameters, const std::shared_ptr<rhi::ComputeBuffer> &stepStatus,
                    std::function<void(std::shared_ptr<JoltGpuConstraintTopology>)> completed);
    [[nodiscard]] JoltGpuControlWork
    PrepareNumericalPlan(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
                         std::shared_ptr<JoltGpuConstraintTopology> topology,
                         std::function<void(std::shared_ptr<const JoltGpuConstraintWorkset>)> completed);

  private:
    class Impl;
    std::shared_ptr<Impl> m_impl;
};
} // namespace infernux
