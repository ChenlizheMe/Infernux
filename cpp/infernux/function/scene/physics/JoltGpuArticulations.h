#pragma once

#include "JoltGpuTick.h"
#include <function/scene/physics/gpu/ArticulationControlTypes.h>
#include <function/scene/physics/gpu/ArticulationTypes.h>

namespace infernux
{
struct JoltGpuArticulationNumericState;
// Author-only metadata. Numeric tail is intentionally NOT initialized/copied
// until a real capture/dynamics producer owns that range in the next slice.
struct JoltGpuArticulationState
{
    uint32_t worldRow = 0;         // accepted address only, not public ID or numerical state
    bool resourceDeclined = false; // this accepted author generation; no hot retry
    std::shared_ptr<const void> authorKey;
    JPH::GPU::ArticulationControlHeader header{};
    std::vector<JPH::GPU::ArticulationLink> links;
    std::vector<JPH::GPU::ArticulationTree> trees;
    std::vector<uint8_t> prefix;
    std::shared_ptr<rhi::ComputeBuffer> control, linkBuffer, treeBuffer;
    std::shared_ptr<JoltGpuArticulationNumericState> numeric;
};
struct JoltGpuArticulationCapture
{
    std::shared_ptr<JoltGpuArticulationState> value;
    bool preparing = false;
};

// Existing Tick producer only: no queue, work source, wait, readback or public API.
class JoltGpuArticulations final
{
  public:
    JoltGpuArticulations(rhi::ComputeHost &, const ShaderTargetCode &stepTarget,
                         std::shared_ptr<rhi::ComputeBuffer> canonicalEmpty,
                         const ShaderTargetCode *captureTarget = nullptr,
                         const ShaderTargetCode *dynamicsTarget = nullptr);
    [[nodiscard]] JoltGpuControlWork
    PrepareAccepted(const JoltGpuStorage &, uint32_t world, std::shared_ptr<const void> authorKey,
                    std::shared_ptr<JoltGpuArticulationState> previous,
                    std::function<void(std::shared_ptr<JoltGpuArticulationState>)> completed,
                    bool acceptedStateIntent = false);
    [[nodiscard]] JoltGpuControlWork PrepareRotors(const JoltGpuStorage &, const JoltGpuStepInputs &,
                                                   std::function<void()> completed);
    [[nodiscard]] JoltGpuControlWork
    PreparePressureDeclined(const JoltGpuStorage &, std::shared_ptr<JoltGpuArticulationState> previous,
                            bool capturedWitness,
                            std::function<void(std::shared_ptr<JoltGpuArticulationState>)> completed);
    [[nodiscard]] JoltGpuControlWork PrepareClassification(const JoltGpuStorage &, const JoltGpuStepInputs &,
                                                           const std::vector<uint32_t> *participantBodies,
                                                           std::function<void()> completed);
    [[nodiscard]] JoltGpuControlWork PreparePose(const JoltGpuStorage &, const JoltGpuStepInputs &,
                                                 std::function<void()> completed);

  private:
    class Impl;
    std::shared_ptr<Impl> m_impl;
};
} // namespace infernux
