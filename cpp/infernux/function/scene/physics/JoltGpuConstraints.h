#pragma once

#include <function/scene/physics/JoltGpuTick.h>

namespace infernux
{
struct JoltGpuConstraintWorkset;

/// Builds one substep's structural schedule after SetupContacts/SetupJoints.
/// No submission, polling loop, private ComputeHost or CPU graph reconstruction.
class JoltGpuConstraints final
{
  public:
    JoltGpuConstraints(rhi::ComputeHost &host, const ShaderTargetCode &target);

    /// Inputs must identify the accepted substep, with contacts installed and
    /// all setup writers already accepted on this same dependency-ordered source.
    /// Their receipts need not be CPU-observed before Prepare; queue dependencies
    /// complete those producers before this chain reads their frozen leases.
    /// The same tick retains numerical endpoints exclusively through completion.
    /// A null packet with continuation is managed capacity deferral; the owner
    /// retries at a new host opportunity or actual resource progress.
    [[nodiscard]] JoltGpuControlWork
    Prepare(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs, const JPH::GPU::StepParams &parameters,
            std::function<void(std::shared_ptr<const JoltGpuConstraintWorkset>)> completed);

  private:
    class Impl;
    std::shared_ptr<Impl> m_impl;
};
} // namespace infernux
