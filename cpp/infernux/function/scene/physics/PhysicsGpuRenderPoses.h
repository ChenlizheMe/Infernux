#pragma once

#include "PhysicsDeviceQueryOutput.h"
#include "PhysicsGpuPoseSource.h"
#include "inxphygpu/render_pose_layout.slang"
#include <function/renderer/rhi/RhiComputeKernel.h>
#include <function/renderer/shader/ShaderProgramCode.h>

namespace infernux
{
struct PhysicsGpuMotionExport;

struct PhysicsGpuRenderPoseBatch
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
class PhysicsGpuRenderPoses final
{
  public:
    PhysicsGpuRenderPoses(rhi::ComputeHost &host, const ShaderTargetCode &target);

    /// Prepare only: no submission, wait or state readback. Generated nodes
    /// contain finite author affine data and authored quaternion products;
    /// signed/zero scales are valid. This boundary checks snapshot ownership,
    /// sizes and bounded node/index domains, not dynamic GPU numerical state.
    /// Empty bindings return no packet or output allocation.
    [[nodiscard]] PhysicsGpuRenderPoseBatch
    Prepare(const PhysicsGpuPoseSource &output, const std::vector<physics::gpu_render::PoseNode> &nodes,
            const std::vector<physics::gpu_render::PoseBinding> &bindings) const;

    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch>
    PrepareMotion(const PhysicsGpuMotionExport &source, const std::vector<PhysicsDeviceQueryOutput> &outputs,
                  const std::vector<size_t> &scatter = {}) const;

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
    Entry m_nodes, m_meshes, m_motion;
    mutable InputCache m_parameters, m_nodeInputs, m_bindingInputs;
    mutable std::vector<InputCache> m_levelParameters;
    mutable std::shared_ptr<rhi::ComputeBuffer> m_nodeModels, m_models, m_bounds;
    mutable std::weak_ptr<rhi::PreparedComputeBatch> m_outputProducer;
};

} // namespace infernux
