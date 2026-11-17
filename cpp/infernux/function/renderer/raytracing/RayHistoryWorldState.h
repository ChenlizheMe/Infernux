#pragma once

#include <function/renderer/RenderWorld.h>

namespace infernux::rhi
{
class BufferResource;
}

namespace infernux::raytracing
{
struct RayWorldMaterialPublication;

/// A ray estimate depends on the published lighting, material pages and BLAS.
/// Transform-only TLAS changes stay eligible for per-pixel reprojection.
class RayHistoryWorldState
{
  public:
    bool Update(const RenderWorldFrame &world, const std::shared_ptr<const rhi::BufferResource> &meshes,
                const std::shared_ptr<const RayWorldMaterialPublication> &materials)
    {
        const auto lighting = world.Lighting() ? world.Lighting()->generation : 0;
        if (m_world == world.WorldId() && m_epoch == world.SimulationEpoch() && m_lighting == lighting &&
            !m_meshes.owner_before(meshes) && !meshes.owner_before(m_meshes) && !m_materials.owner_before(materials) &&
            !materials.owner_before(m_materials))
            return false;
        m_world = world.WorldId();
        m_epoch = world.SimulationEpoch();
        m_lighting = lighting;
        m_meshes = meshes;
        m_materials = materials;
        return true;
    }

  private:
    uint64_t m_world = 0;
    uint64_t m_epoch = 0;
    uint64_t m_lighting = 0;
    std::weak_ptr<const rhi::BufferResource> m_meshes;
    std::weak_ptr<const RayWorldMaterialPublication> m_materials;
};

template <class Histories> void InvalidateRayHistories(Histories &histories)
{
    for (auto &[name, history] : histories)
        if (name.rfind("_ray_", 0) == 0)
            history.Invalidate();
}
} // namespace infernux::raytracing
