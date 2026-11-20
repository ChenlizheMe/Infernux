#pragma once

#include <function/scene/physics/JoltGpuTick.h>

#include <array>

namespace infernux
{
struct JoltGpuConstraintWorkset;

struct JoltGpuConstraintSetup
{
    std::vector<JoltGpuStageWork> stages;
    // Empty when a contact consumer needs the fully set-up rows first.
    std::function<void(std::shared_ptr<const JoltGpuConstraintWorkset>)> assembled;
    // Record ordered stages against the proposed schedule. Advance the real
    // cursor only after that guarded prefix actually executed.
    std::function<std::function<void()>(
        const JoltGpuStepInputs &, const std::shared_ptr<const JoltGpuConstraintWorkset> &, size_t,
        std::vector<rhi::ComputeBufferUpdate> &, std::vector<rhi::ComputeDispatchDesc> &)>
        appendSolverPrefix;
};

// A topology cursor can be recorded on its own or appended to its producer.
// In-progress convex work gates its topology prefix with the device result.
struct JoltGpuContactTopologyStart
{
    using ObservePrefix = std::function<JoltGpuControlWork(bool collisionComplete)>;
    using AcceptPrefix = std::function<JoltGpuControlWork()>;
    std::function<JoltGpuControlWork()> prepare;
    std::function<ObservePrefix(const std::shared_ptr<rhi::ComputeBuffer> &completion,
                                std::vector<rhi::ComputeBufferUpdate> &uploads,
                                std::vector<rhi::ComputeDispatchDesc> &dispatches)>
        appendAfterCollision;
    std::function<AcceptPrefix(size_t dispatchLimit, std::vector<rhi::ComputeBufferUpdate> &uploads,
                               std::vector<rhi::ComputeDispatchDesc> &dispatches)>
        appendReady;
};

/// Builds one substep topology, then resumes that SAME validated CC execution
/// after wake and numerical Setup to publish the active solver schedule.
/// No submission, polling loop, private ComputeHost or CPU graph reconstruction.
class JoltGpuConstraints final
{
  public:
    /// Private, queue-ordered storage for one fixed-step owner.
    struct Workspace
    {
        static constexpr size_t MaxDispatches = 16;
        struct Uniforms
        {
            std::shared_ptr<rhi::ComputeBuffer> parameters, work;
        };
        std::shared_ptr<rhi::ComputeBuffer> schedule, parents, rootVotes, arguments;
        std::shared_ptr<rhi::ComputeBuffer> gateSpans, gatedArguments;
        std::array<Uniforms, MaxDispatches> uniforms;
    };

    JoltGpuConstraints(rhi::ComputeHost &host, const ShaderTargetCode &target);
    [[nodiscard]] bool TryPrepareStartupPrograms();

    [[nodiscard]] JoltGpuContactTopologyStart
    PrepareTopology(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs, JoltGpuPrograms &programs,
                    const JPH::GPU::StepParams &parameters, const std::shared_ptr<rhi::ComputeBuffer> &stepStatus,
                    Workspace &workspace, std::optional<JoltGpuConstraintSetup> setup,
                    std::function<void(std::shared_ptr<JoltGpuConstraintTopology>, bool setupAccepted)> completed);
    [[nodiscard]] JoltGpuControlWork
    PrepareNumericalPlan(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
                         std::shared_ptr<JoltGpuConstraintTopology> topology,
                         std::function<void(std::shared_ptr<const JoltGpuConstraintWorkset>)> completed);

  private:
    class Impl;
    std::shared_ptr<Impl> m_impl;
};
} // namespace infernux
