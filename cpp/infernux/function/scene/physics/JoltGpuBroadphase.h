#pragma once

#include "JoltGpuTick.h"

namespace infernux
{

/// Sparse candidate preparation on the tick's existing source and receipts.
/// This object prepares packets only; it never submits or waits on a queue.
/// Managed capacity pressure returns a null packet with a retained continuation.
class JoltGpuBroadphase final
{
  public:
    JoltGpuBroadphase(rhi::ComputeHost &host, const ShaderTargetCode &target);

    [[nodiscard]] JoltGpuControlWork
    Prepare(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs, const JPH::GPU::StepParams &parameters,
            std::function<void(std::shared_ptr<const JoltGpuContactWorkset>)> completed);

  private:
    class Impl;
    std::shared_ptr<Impl> m_impl;
};

} // namespace infernux
