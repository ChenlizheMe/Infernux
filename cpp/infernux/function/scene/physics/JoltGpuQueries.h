#pragma once

#include "JoltGpuPrograms.h"

namespace infernux
{

/// All three frozen values must come from ONE immutable scene publication.
/// The publication's identity array must remain with results to resolve slots;
/// current mutable scene/body lookup tables cannot identify an older hit.
struct JoltGpuQuerySnapshot
{
    std::shared_ptr<rhi::ComputeBuffer> bodyModels;
    std::shared_ptr<rhi::ComputeBuffer> shapeModels;
    std::shared_ptr<rhi::ComputeBuffer> bodyStates;
    uint32_t firstBody = 0;
    uint32_t bodyCount = 0;
    uint32_t shapeCount = 0;
};

struct JoltGpuRaycastBatch
{
    std::shared_ptr<rhi::PreparedComputeBatch> batch;
    std::shared_ptr<rhi::ComputeBuffer> hits;
    uint32_t rayCount = 0;
};

/// Typed consumer of the common compiler's raycast_closest artifact. One GPU
/// lane visits the admitted body's leaf ranges; no CPU physics/query fallback.
class JoltGpuQueries final
{
  public:
    JoltGpuQueries(rhi::ComputeHost &host, const ShaderTargetCode &target);

    /// Does not submit or wait. Optional readback is in this SAME packet at
    /// index zero; callers read it only after the actual receipt completes.
    /// Empty rays return an empty result. Empty worlds still produce GPU misses.
    /// Invalid ray records throw before preparing any work. Defaults/configured
    /// masks belong to the public facade; an explicit zero mask means no hit.
    /// Convex starts inside report distance zero. Sphere center normal is local
    /// +Y; box interior normals use the nearest outer face, X/Y/Z tie order.
    /// Rounded box edges/corners use their actual Minkowski geometry/normals;
    /// this deliberately corrects the CPU Jolt ray path's outer-AABox shortcut.
    [[nodiscard]] JoltGpuRaycastBatch Prepare(const JoltGpuQuerySnapshot &snapshot,
                                              const std::vector<JPH::GPU::RaycastInput> &rays,
                                              bool readback = false) const;

  private:
    rhi::ComputeHost m_host;
    std::shared_ptr<rhi::ComputeKernel> m_kernel;
    std::vector<uint8_t> m_ports;
};

} // namespace infernux
