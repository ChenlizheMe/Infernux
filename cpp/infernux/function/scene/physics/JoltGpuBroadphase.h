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
    /// One discrete collision workspace retained by the fixed-step owner.
    struct Workspace
    {
        uint32_t patchCapacity = 0, nextCapacity = 0;
        std::shared_ptr<rhi::ComputeBuffer> arena, arguments, parameters, patches, contacts, pairCount;
    };

    enum class Mode
    {
        Discrete,
        DiscreteWithHostCount,
        Swept
    };

    JoltGpuBroadphase(rhi::ComputeHost &host, const ShaderTargetCode &target);
    [[nodiscard]] bool TryPrepareStartupPrograms();

    [[nodiscard]] JoltGpuControlWork
    Prepare(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs, JoltGpuPrograms &programs,
            const JPH::GPU::StepParams &parameters, const std::shared_ptr<rhi::ComputeBuffer> &stepStatus,
            Workspace &workspace, std::function<void(std::shared_ptr<const JoltGpuContactWorkset>)> completed,
            Mode mode = Mode::Discrete);

    // The discrete workspace belongs to its fixed-step owner.
    void ResetCapacityHint() noexcept;

  private:
    class Impl;
    std::shared_ptr<Impl> m_impl;
};

} // namespace infernux
