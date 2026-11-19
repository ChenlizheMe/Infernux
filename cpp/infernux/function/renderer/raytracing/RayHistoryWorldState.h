#pragma once

#include "RayWorldGeometry.h"

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
    bool Update(const RenderWorldFrame &world, const RayWorldGeometryBinding &geometry,
                const std::shared_ptr<const RayWorldMaterialPublication> &materials)
    {
        m_candidateGeometry = geometry.visibilitySerial;
        const auto lighting = world.Lighting() ? world.Lighting()->generation : 0;
        // Geometry prepared for another light publication cannot establish
        // which receivers changed under this view's current lights.
        const bool matchedVisibility =
            geometry.visibilityLighting == lighting || m_candidateGeometry == m_acceptedGeometry;
        if (m_world == world.WorldId() && m_epoch == world.SimulationEpoch() && m_lighting == lighting &&
            matchedVisibility && !m_meshes.owner_before(geometry.meshes) && !geometry.meshes.owner_before(m_meshes) &&
            !m_materials.owner_before(materials) && !materials.owner_before(m_materials))
            return false;
        m_world = world.WorldId();
        m_epoch = world.SimulationEpoch();
        m_lighting = lighting;
        m_meshes = geometry.meshes;
        m_materials = materials;
        return true;
    }

    void AcceptGeometry() noexcept
    {
        m_acceptedGeometry = m_candidateGeometry;
    }

    [[nodiscard]] uint32_t AcceptedGeometry() const noexcept
    {
        return m_acceptedGeometry;
    }

  private:
    uint64_t m_world = 0;
    uint64_t m_epoch = 0;
    uint64_t m_lighting = 0;
    uint32_t m_candidateGeometry = 0;
    uint32_t m_acceptedGeometry = 0;
    std::weak_ptr<const rhi::BufferResource> m_meshes;
    std::weak_ptr<const RayWorldMaterialPublication> m_materials;
};

template <class Histories> void InvalidateRayHistories(Histories &histories)
{
    for (auto &[key, history] : histories)
        if (history.dependsOnRayWorld)
            history.Invalidate();
}
} // namespace infernux::raytracing
