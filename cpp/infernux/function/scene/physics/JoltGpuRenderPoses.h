#pragma once

#include "JoltGpuPrograms.h"

namespace JPH::GPU
{
struct PoseNode;
}

namespace infernux
{

struct JoltGpuRenderPoseBatch
{
    std::shared_ptr<rhi::PreparedComputeBatch> batch;
    /// Float32 buffers: four model columns and two world AABB records per mesh.
    std::shared_ptr<rhi::ComputeBuffer> models;
    std::shared_ptr<rhi::ComputeBuffer> bounds;
    uint32_t count = 0;
};

/// Topological node levels and a final mesh dispatch share one immutable state. The
/// caller supplies generated author bindings and publishes RenderPoseBuffer
/// only after its owning scene reaches the corresponding complete boundary.
/// Compilation, submission, budgets and retirement remain shared engine work.
class JoltGpuRenderPoses final
{
  public:
    JoltGpuRenderPoses(rhi::ComputeHost &host, const ShaderTargetCode &target);

    /// Prepare only: no submission, wait or state readback. Generated nodes
    /// contain finite author affine data and authored quaternion products;
    /// signed/zero scales are valid. This boundary checks snapshot ownership,
    /// sizes and bounded node/index domains, not dynamic GPU numerical state.
    /// Empty bindings return no packet or output allocation.
    [[nodiscard]] JoltGpuRenderPoseBatch Prepare(const std::shared_ptr<rhi::ComputeBuffer> &frozenBodyStates,
                                                 const std::vector<JPH::GPU::PoseNode> &nodes,
                                                 const std::vector<JPH::GPU::PoseBinding> &bindings) const;

  private:
    struct Entry
    {
        std::shared_ptr<rhi::ComputeKernel> kernel;
        std::vector<uint8_t> ports;
    };
    struct InputCache
    {
        std::vector<uint8_t> bytes;
        std::shared_ptr<rhi::ComputeBuffer> buffer;
    };
    rhi::ComputeHost m_host;
    Entry m_nodes, m_meshes;
    mutable InputCache m_parameters, m_nodeInputs, m_bindingInputs;
    mutable std::vector<InputCache> m_levelParameters;
    mutable std::shared_ptr<rhi::ComputeBuffer> m_nodeModels, m_models, m_bounds;
    mutable std::weak_ptr<rhi::PreparedComputeBatch> m_outputProducer;
};

} // namespace infernux
