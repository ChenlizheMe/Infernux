#pragma once

#include "JoltGpuTick.h"
#include <function/scene/physics/gpu/SleepTypes.h>

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
    [[nodiscard]] bool TryPrepareStartupPrograms();
    static void InitializeStepStatus(const std::shared_ptr<rhi::ComputeBuffer> &,
                                     std::vector<rhi::ComputeBufferUpdate> &);
    static void ValidateStepStatus(rhi::ComputeReadback &);
    static void ValidateStepStatus(const JPH::GPU::PhysicsStepStatus &);
    [[nodiscard]] JoltGpuControlWork PrepareWake(const JoltGpuStorage &, const JoltGpuStepInputs &,
                                                 const JoltGpuConstraintTopology &,
                                                 const std::shared_ptr<rhi::ComputeBuffer> &status,
                                                 std::function<void()> completed);
    // Record the same wake passes after a topology producer in the current
    // packet. The caller discards an incomplete, still unaccepted append.
    [[nodiscard]] bool AppendWake(const JoltGpuStorage &, const JoltGpuStepInputs &, const JoltGpuConstraintTopology &,
                                  const std::shared_ptr<rhi::ComputeBuffer> &status, size_t dispatchLimit,
                                  std::vector<rhi::ComputeBufferUpdate> &, std::vector<rhi::ComputeDispatchDesc> &);
    [[nodiscard]] JoltGpuControlWork PrepareHistory(const JoltGpuStorage &, const JoltGpuStepInputs &,
                                                    const JPH::GPU::BodySleepParameters &,
                                                    const std::shared_ptr<rhi::ComputeBuffer> &status,
                                                    std::function<void()> completed);
    // Called ONLY after a genuinely new accepted force-input producer. An
    // absent CPU upload is not evidence of a new external GPU force write.
    [[nodiscard]] JoltGpuControlWork PrepareForceWake(const JoltGpuStorage &, const JPH::GPU::StepParams &,
                                                      std::function<void()> completed);
    [[nodiscard]] JoltGpuControlWork
    PreparePrescribedCapture(const JoltGpuStorage &, const JPH::GPU::StepParams &, std::vector<uint32_t> sourceSlots,
                             const std::shared_ptr<rhi::ComputeBuffer> &status,
                             std::function<void(std::shared_ptr<const JoltGpuPrescribedSleepBounds>)> completed);
    [[nodiscard]] JoltGpuControlWork PreparePrescribedWake(const JoltGpuStorage &, const JPH::GPU::StepParams &,
                                                           const JoltGpuPrescribedSleepBounds &,
                                                           const std::shared_ptr<rhi::ComputeBuffer> &status,
                                                           std::function<void()> completed);
    static constexpr uint64_t MinimumSnapshotUploadBytes =
        sizeof(JPH::GPU::SleepWorkParameters) + sizeof(JPH::GPU::StageWork);
    // Pack motion and activity together. Return the number of local body rows recorded.
    [[nodiscard]] uint32_t AppendPublishedMotion(const JoltGpuStorage &, const JPH::GPU::StepParams &,
                                                 const std::shared_ptr<rhi::ComputeBuffer> &snapshot, uint32_t first,
                                                 uint64_t &uploadBudget, std::vector<rhi::ComputeBufferUpdate> &,
                                                 std::vector<rhi::ComputeDispatchDesc> &);
    // Old bounds are captured before the accepted source mutation. The later
    // wake retains the exact 64B bound/pose witness even if the source is removed.
    [[nodiscard]] JoltGpuControlWork
    PrepareOldBounds(const JoltGpuStorage &, const JPH::GPU::StepParams &, uint32_t source,
                     std::shared_ptr<BoundsValidation>,
                     std::function<void(std::shared_ptr<rhi::ComputeBuffer>)> completed);
    [[nodiscard]] JoltGpuControlWork PrepareBoundsWake(const JoltGpuStorage &, const JPH::GPU::StepParams &,
                                                       std::shared_ptr<rhi::ComputeBuffer> oldBounds,
                                                       uint32_t newSource, std::shared_ptr<BoundsValidation>);
    [[nodiscard]] JoltGpuControlWork
    PrepareSourceBoundsWake(const JoltGpuStorage &, const JPH::GPU::StepParams &, std::vector<uint32_t> sources,
                            std::shared_ptr<BoundsValidation>,
                            std::vector<std::pair<uint32_t, uint32_t>> exclusions = {});
    [[nodiscard]] JoltGpuControlWork PrepareAuthoredBoundsWake(const JoltGpuStorage &, const JPH::GPU::StepParams &,
                                                               JPH::GPU::Float4 low, JPH::GPU::Float4 high,
                                                               std::shared_ptr<BoundsValidation>);
    [[nodiscard]] JoltGpuPoseWake PreparePoseWake(const JoltGpuStorage &, const JPH::GPU::StepParams &,
                                                  BoundsValidation &);
    [[nodiscard]] JoltGpuControlWork ValidateBoundsEdits(const BoundsValidation &);

  private:
    class Impl;
    std::shared_ptr<Impl> m_impl;
};
} // namespace infernux
