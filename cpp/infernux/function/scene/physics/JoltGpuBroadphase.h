#pragma once

#include "JoltGpuTick.h"

#include <array>

namespace infernux
{

/// Sparse candidate preparation on the tick's existing source and receipts.
/// This object prepares packets only; it never submits or waits on a queue.
/// Managed capacity pressure returns a null packet with a retained continuation.
class JoltGpuBroadphase final
{
  public:
    /// Each fixed-step owner keeps separate discrete and swept workspaces.
    struct Workspace
    {
        static constexpr uint32_t DispatchesPerPacket = 16u;
        uint32_t patchCapacity = 0, nextCapacity = 0;
        uint64_t sweepJobs = 0;
        std::shared_ptr<rhi::ComputeBuffer> arena, arguments, parameters, patches, contacts, pairCount;
        std::array<std::shared_ptr<rhi::ComputeBuffer>, DispatchesPerPacket> parameterSlots, workSlots;
    };

    enum class Mode
    {
        Discrete,
        DiscreteWithHostCount,
        Swept
    };

    using SweepRetry = std::function<JoltGpuControlWork()>;
    using SweepConsumer = std::function<JoltGpuControlWork(std::shared_ptr<const JoltGpuContactWorkset>, SweepRetry)>;

    JoltGpuBroadphase(rhi::ComputeHost &host, const ShaderTargetCode &target);
    [[nodiscard]] bool TryPrepareStartupPrograms();

    [[nodiscard]] JoltGpuControlWork
    Prepare(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs, JoltGpuPrograms &programs,
            const JPH::GPU::StepParams &parameters, const std::shared_ptr<rhi::ComputeBuffer> &stepStatus,
            Workspace &workspace, std::function<void(std::shared_ptr<const JoltGpuContactWorkset>)> completed,
            Mode mode = Mode::Discrete);

    [[nodiscard]] JoltGpuControlWork PrepareSweep(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
                                                  JoltGpuPrograms &programs, const JPH::GPU::StepParams &parameters,
                                                  const std::shared_ptr<rhi::ComputeBuffer> &stepStatus,
                                                  Workspace &workspace, SweepConsumer consumer);

  private:
    class Impl;
    std::shared_ptr<Impl> m_impl;
};

} // namespace infernux
