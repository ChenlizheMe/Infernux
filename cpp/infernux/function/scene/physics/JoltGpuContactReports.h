#pragma once
#include <Jolt/GPU/ContactReportTypes.h>
#include "JoltGpuScene.h"
#include <array>

namespace infernux
{
struct JoltGpuContactReportResult
{
    struct Frame
    {
        uint32_t substep = 0, capacity = 0;
        bool seeded = false, finished = false;
        std::shared_ptr<rhi::ComputeBuffer> interest, indices, words;
        // Admitted by BeforeVelocity and reused by AfterVelocity, so reporting
        // cannot require a new DeviceLocal parameter after reduced commit.
        std::shared_ptr<rhi::ComputeBuffer> parameters;
        std::vector<JPH::GPU::ContactReportPatch> records;
    };
    std::shared_ptr<const JoltGpuScene::Publication> publication;
    std::vector<Frame> frames;
    uint32_t expectedSubsteps = 0;
    bool complete = false;
};

/// Prepares packets on the existing Tick source. Never submits, polls or waits.
/// Inputs with runtime writers are borrowed under that Tick's exclusive lease.
class JoltGpuContactReports final
{
  public:
    using Interest = std::function<std::vector<uint32_t>(const JoltGpuScene::Publication &)>;
    struct Capture
    {
        std::shared_ptr<JoltGpuContactReportResult> result;
        std::function<JoltGpuControlWork(const JoltGpuStorage &, const JoltGpuStepInputs &,
                                         const JPH::GPU::StepParams &, uint32_t, JoltGpuContactReportStage,
                                         std::function<void()>)>
            phase;
        JoltGpuScene::Consumer finish;
    };
    JoltGpuContactReports(rhi::ComputeHost &, const ShaderTargetCode &);
    [[nodiscard]] Capture Prepare(std::shared_ptr<const JoltGpuScene::Publication>, uint32_t substeps,
                                  uint32_t streamFlags, Interest) const;

  private:
    struct Impl;
    std::shared_ptr<Impl> m_impl;
};
} // namespace infernux
