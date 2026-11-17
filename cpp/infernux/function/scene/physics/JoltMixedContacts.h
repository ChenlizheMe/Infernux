#pragma once

#include <Jolt/Jolt.h>

#include <Jolt/Geometry/AABox.h>
#include "JoltGpuScene.h"
#include "PhysicsContactPublication.h"
#include <core/math/Bvh.h>

namespace infernux
{
class JoltCpuScene;

/// One accepted Fixed step's CPU geometry and substep positions. The CPU
/// world still owns integration, and this capture owns no solver bodies.
class JoltMixedContactStep final
{
  public:
    using Policy = PhysicsContactPublication::Policy;
    using Frame = PhysicsContactPublication::Frame;
    struct Candidate
    {
        JPH::GPU::UInt4 source; // GPU leaf index, GPU layer, GPU collision mask, reserved.
        JPH::GPU::Float4 center, rotation, linear, angular, sweepMin, sweepMax;
    };
    static_assert(sizeof(Candidate) == 112);

    void Begin(const JoltCpuScene &, std::vector<uint32_t> bodyIds, uint32_t substeps);
    void CaptureCpuBoundary(uint32_t boundary);
    [[nodiscard]] const std::vector<JPH::GPU::UInt4> &EncodeTree(uint32_t substep);
    void Collect(uint32_t substep, const JoltGpuScene::Publication &, const Policy &, const Candidate &);
    void AppendFrame(uint32_t substep, Frame &);
    void ClearStep();

  private:
    struct CpuProxy
    {
        uint32_t id;
        JPH::RefConst<JPH::Shape> shape;
        PhysicsTargetReference target;
        bool sensor;
    };
    struct Pose
    {
        JPH::RVec3 center;
        JPH::Quat rotation;
        JPH::Vec3 linear, angular;
        JPH::AABox bounds;
    };
    const JoltCpuScene *m_scene = nullptr;
    std::vector<CpuProxy> m_proxies;
    std::vector<std::vector<Pose>> m_pose;
    std::vector<std::vector<JPH::GPU::UInt4>> m_treeWords;
    std::vector<Frame> m_frames;
    std::vector<uint32_t> m_ids, m_escape;
    spatial::Bvh m_tree;
};

/// Device traversal of the accepted CPU BVH against resident GPU leaf bounds.
/// This uses the owning physics cursor's packets throughout.
class JoltGpuMixedContacts final
{
  public:
    JoltGpuMixedContacts(rhi::ComputeHost &, const ShaderTargetCode &);
    [[nodiscard]] JoltGpuScene::ContactObserver Observe(const std::shared_ptr<const JoltGpuScene::Publication> &,
                                                        JoltMixedContactStep &, const JoltMixedContactStep::Policy &);

  private:
    struct Job;
    rhi::ComputeHost m_host;
    std::shared_ptr<rhi::ComputeKernel> m_kernel;
    std::vector<uint32_t> m_ports;
};
} // namespace infernux
