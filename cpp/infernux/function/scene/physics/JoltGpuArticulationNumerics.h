#pragma once
#include "JoltGpuArticulations.h"
#include <function/scene/physics/gpu/ArticulationDynamicsTypes.h>
#include <function/scene/physics/gpu/StateFkTypes.h>

namespace infernux
{
struct JoltGpuArticulationNumericState
{
    struct Window
    {
        uint32_t first, count, maxLinks;
    };
    // Exact accepted stable joint identity/kind -> previous tree; no CPU mode.
    struct Origin
    {
        uint64_t id;
        uint32_t kind, tree;
    };
    std::vector<Window> windows;
    std::vector<Origin> origin;
    std::vector<JPH::GPU::UInt4> transitions;
    std::vector<JPH::GPU::StateFkTree> captureMetadata;
    std::shared_ptr<rhi::ComputeBuffer> previousControl;
    uint32_t previousPrefixBytes = 48;
    bool pendingOrigin = true, rowsNeedRefresh = false, initialized = false;
    std::shared_ptr<rhi::ComputeBuffer> roots, coordinates, stagingRoots, stagingCoordinates;
    std::shared_ptr<rhi::ComputeBuffer> captureTrees, rows, witness, summary;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, 8> uniforms;
};
// Same Tick Work source; resident GPU facts alone select representation.
class JoltGpuArticulationNumerics final
{
  public:
    JoltGpuArticulationNumerics(rhi::ComputeHost &, const ShaderTargetCode &step, const ShaderTargetCode &capture,
                                const ShaderTargetCode &dynamics);
    [[nodiscard]] JoltGpuControlWork PrepareInitialize(const JoltGpuStorage &,
                                                       std::shared_ptr<JoltGpuArticulationState>,
                                                       std::shared_ptr<JoltGpuArticulationState> previous,
                                                       std::function<JoltGpuControlWork(bool)> admitted);
    [[nodiscard]] JoltGpuControlWork PrepareDeclined(const JoltGpuStorage &, std::shared_ptr<JoltGpuArticulationState>,
                                                     std::shared_ptr<JoltGpuArticulationState> previous,
                                                     std::function<void()> completed, bool capturedWitness = false);
    [[nodiscard]] JoltGpuControlWork PrepareBeforeForces(const JoltGpuStorage &, const JoltGpuStepInputs &,
                                                         std::function<void()> completed);
    [[nodiscard]] JoltGpuControlWork PrepareCandidate(const JoltGpuStorage &, const JoltGpuStepInputs &,
                                                      std::function<void()> completed);
    [[nodiscard]] JoltGpuControlWork PreparePose(const JoltGpuStorage &, const JoltGpuStepInputs &,
                                                 std::function<void()> completed);
    [[nodiscard]] JoltGpuControlWork PrepareDirty(std::shared_ptr<JoltGpuArticulationState>,
                                                  std::function<void()> completed);

  private:
    class Impl;
    std::shared_ptr<Impl> m_impl;
};
} // namespace infernux
