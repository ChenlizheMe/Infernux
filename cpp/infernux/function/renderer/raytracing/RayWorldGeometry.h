#pragma once

#include <core/threading/FrameWorkOwner.h>
#include <core/threading/JobSystem.h>

#include <memory>
#include <vector>

namespace infernux
{
class RenderWorldFrame;
namespace rhi
{
class BufferResource;
class Device;
} // namespace rhi

namespace raytracing
{
class RayWorldGeometryPublication;
class RayWorldGeometryBuild;

struct RayWorldGeometryBinding
{
    std::shared_ptr<const rhi::BufferResource> buffer;
    std::shared_ptr<const rhi::BufferResource> meshes;
    bool pending = false;
};

/// Device-local presentation of immutable World geometry requested by a view.
/// All request, upload and publication operations run on the frame owner.
class RayWorldGeometry final
{
  public:
    RayWorldGeometry(FrameWorkOwner &owner, rhi::Device &device);
    ~RayWorldGeometry();

    [[nodiscard]] RayWorldGeometryBinding Acquire(const std::shared_ptr<const RenderWorldFrame> &world,
                                                  std::shared_ptr<const RayWorldGeometryPublication> &publication);

  private:
    void Observe();
    void CollectReceipt();
    [[nodiscard]] FrameWorkFacts Describe();
    bool Advance();
    [[nodiscard]] std::shared_ptr<RayWorldGeometryPublication> FindPending() const;

    FrameWorkOwner &m_owner;
    rhi::Device &m_device;
    FrameWorkOwner::SourceId m_source = 0;
    std::vector<std::weak_ptr<RayWorldGeometryPublication>> m_publications;
    std::shared_ptr<RayWorldGeometryBuild> m_active;
    std::shared_ptr<const rhi::BufferResource> m_unavailableBuffer, m_emptyMeshBuffer;
    JobHandle m_job;
    FrameWorkResource m_workResource = FrameWorkResource::Cpu;
    uint64_t m_submittedOpportunity = 0;
    uint64_t m_workGrant = 0;
};
} // namespace raytracing
} // namespace infernux
