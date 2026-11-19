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

enum class RayHistoryChange
{
    None,
    Lighting,
    Reset
};

/// A changed light can be remeasured. A new world, material publication or
/// mesh allocation invalidates the underlying evidence as well.
class RayHistoryWorldState
{
  public:
    RayHistoryChange Update(const RenderWorldFrame &world, const RayWorldGeometryBinding &geometry,
                            const std::shared_ptr<const RayWorldMaterialPublication> &materials)
    {
        m_candidateGeometry = geometry.visibilitySerial;
        const auto lighting = world.Lighting() ? world.Lighting()->generation : 0;
        const auto cut = world.Lighting() ? world.Lighting()->diffuseCutGeneration : 0;
        m_proposalCut = world.Lighting() ? world.Lighting()->proposalCutGeneration : 0;
        // Geometry prepared for another light publication cannot establish
        // which receivers changed under this view's current lights.
        const bool matchedVisibility =
            geometry.visibilityLighting == lighting || m_candidateGeometry == m_acceptedGeometry;
        const bool sameScene = m_world == world.WorldId() && m_epoch == world.SimulationEpoch() &&
                               !m_meshes.owner_before(geometry.meshes) && !geometry.meshes.owner_before(m_meshes) &&
                               !m_materials.owner_before(materials) && !materials.owner_before(m_materials);
        m_lightingChanged = lighting != m_acceptedLighting || !matchedVisibility;
        const bool newLighting = m_lighting != lighting || !matchedVisibility;
        const bool reset = !sameScene || cut != m_lightCut;
        if (!sameScene) {
            m_world = world.WorldId();
            m_epoch = world.SimulationEpoch();
            m_meshes = geometry.meshes;
            m_materials = materials;
        }
        m_lighting = lighting;
        m_lightCut = cut;
        if (reset)
            return RayHistoryChange::Reset;
        return newLighting ? RayHistoryChange::Lighting : RayHistoryChange::None;
    }

    void Accept() noexcept
    {
        m_acceptedGeometry = m_candidateGeometry;
        m_acceptedLighting = m_lighting;
        m_acceptedProposalCut = m_proposalCut;
    }

    [[nodiscard]] uint32_t ViewFlags() const noexcept
    {
        return (static_cast<uint32_t>(m_lightingChanged) << 4u) |
               (static_cast<uint32_t>(m_proposalCut != m_acceptedProposalCut) << 5u);
    }

    [[nodiscard]] uint32_t AcceptedGeometry() const noexcept
    {
        return m_acceptedGeometry;
    }

  private:
    uint64_t m_world = 0;
    uint64_t m_epoch = 0;
    uint64_t m_lighting = 0;
    uint64_t m_lightCut = 0;
    uint64_t m_acceptedLighting = 0;
    uint64_t m_proposalCut = 0;
    uint64_t m_acceptedProposalCut = 0;
    bool m_lightingChanged = false;
    uint32_t m_candidateGeometry = 0;
    uint32_t m_acceptedGeometry = 0;
    std::weak_ptr<const rhi::BufferResource> m_meshes;
    std::weak_ptr<const RayWorldMaterialPublication> m_materials;
};

template <class Histories> void InvalidateRayHistories(Histories &histories, RayHistoryChange change)
{
    if (change == RayHistoryChange::None)
        return;
    for (auto &[key, history] : histories)
        if (history.dependsOnRayWorld && (change == RayHistoryChange::Reset || !history.respondsToLighting))
            history.Invalidate();
}
} // namespace infernux::raytracing
