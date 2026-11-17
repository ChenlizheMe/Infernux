#pragma once

#include <Jolt/GPU/SleepTypes.h>
#include "JoltGpuTick.h"

namespace infernux
{
struct JoltGpuPrescribedSleepBounds
{
    uint32_t count = 0;
    std::shared_ptr<rhi::ComputeBuffer> sources, bounds;
};
// Bounded control producers on the existing Tick source. No queue, wait,
// public setting, graph reconstruction or independent numerical executor.
class JoltGpuSleep final
{
  public:
    struct BoundsValidation
    {
        std::shared_ptr<rhi::ComputeBuffer> status;
    };

    JoltGpuSleep(rhi::ComputeHost &host, const ShaderTargetCode &stepTarget);
    [[nodiscard]] JoltGpuControlWork PrepareInitialize(const JoltGpuStorage &, const JPH::GPU::StepParams &,
                                                       std::function<void()> completed);
    [[nodiscard]] JoltGpuControlWork PrepareWake(const JoltGpuStorage &, const JoltGpuStepInputs &,
                                                 const JoltGpuConstraintTopology &, std::function<void()> completed);
    [[nodiscard]] JoltGpuControlWork PrepareHistory(const JoltGpuStorage &, const JoltGpuStepInputs &,
                                                    const JPH::GPU::BodySleepParameters &, bool lastSubstep,
                                                    std::function<void()> completed);
    // Called ONLY after a genuinely new accepted force-input producer. An
    // absent CPU upload is not evidence of a new external GPU force write.
    [[nodiscard]] JoltGpuControlWork PrepareForceWake(const JoltGpuStorage &, const JPH::GPU::StepParams &,
                                                      std::function<void()> completed);
    [[nodiscard]] JoltGpuControlWork
    PreparePrescribedCapture(const JoltGpuStorage &, const JPH::GPU::StepParams &, std::vector<uint32_t> sourceSlots,
                             std::function<void(std::shared_ptr<const JoltGpuPrescribedSleepBounds>)> completed);
    [[nodiscard]] JoltGpuControlWork PreparePrescribedWake(const JoltGpuStorage &, const JPH::GPU::StepParams &,
                                                           std::shared_ptr<const JoltGpuPrescribedSleepBounds>,
                                                           std::function<void()> completed);
    [[nodiscard]] JoltGpuControlWork
    PrepareSnapshot(const JoltGpuStorage &, const JPH::GPU::StepParams &,
                    std::function<void(std::shared_ptr<rhi::ComputeBuffer>)> completed);
    // Old bounds are captured before the accepted source mutation. The later
    // wake retains the exact 64B bound/pose witness even if the source is removed.
    [[nodiscard]] JoltGpuControlWork
    PrepareOldBounds(const JoltGpuStorage &, const JPH::GPU::StepParams &, uint32_t source,
                     std::shared_ptr<BoundsValidation>,
                     std::function<void(std::shared_ptr<rhi::ComputeBuffer>)> completed);
    [[nodiscard]] JoltGpuControlWork PrepareBoundsWake(const JoltGpuStorage &, const JPH::GPU::StepParams &,
                                                       std::shared_ptr<rhi::ComputeBuffer> oldBounds,
                                                       uint32_t newSource, std::shared_ptr<BoundsValidation>,
                                                       bool requirePoseChange = false);
    [[nodiscard]] JoltGpuControlWork PrepareAuthoredBoundsWake(const JoltGpuStorage &, const JPH::GPU::StepParams &,
                                                               JPH::GPU::Float4 low, JPH::GPU::Float4 high,
                                                               std::shared_ptr<BoundsValidation>);
    [[nodiscard]] JoltGpuControlWork ValidateBoundsEdits(const BoundsValidation &);
    [[nodiscard]] JoltGpuControlWork PrepareQuery(std::shared_ptr<rhi::ComputeBuffer> snapshot, uint32_t bodyCount,
                                                  std::vector<uint32_t> requests,
                                                  std::function<void(std::vector<JPH::GPU::UInt4>)> completed);

  private:
    class Impl;
    std::shared_ptr<Impl> m_impl;
};
} // namespace infernux
