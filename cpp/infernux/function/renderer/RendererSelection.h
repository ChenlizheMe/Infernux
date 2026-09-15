#pragma once

#include "RenderIdentity.h"
#include "RendererParameterBlock.h"
#include <function/resources/InxMaterial/InxMaterial.h>
#include <limits>
#include <stdexcept>

namespace infernux
{

/// Author-owned selection for a graph material pass. Entries retain identities,
/// not geometry: the pass consumes the camera's current RenderWorld publication.
/// Mutations belong to the owner-thread update phase, before native recording.
class RendererSelection
{
  public:
    explicit RendererSelection(std::shared_ptr<InxMaterial> material) : m_material(std::move(material))
    {
        if (!m_material)
            throw std::invalid_argument("RendererSelection requires a material");
    }

    void Set(const RenderProxyHandle &renderer, int submesh = -1, const DrawParameterBlock *parameters = nullptr)
    {
        const auto key = Key(renderer, submesh);
        auto captured = parameters ? parameters->Capture(*m_material) : nullptr;
        m_entries.insert_or_assign(key, std::move(captured));
        ++m_revision;
    }

    bool Remove(const RenderProxyHandle &renderer, int submesh = -1)
    {
        const bool removed = m_entries.erase(Key(renderer, submesh)) != 0;
        if (removed)
            ++m_revision;
        return removed;
    }

    void Clear()
    {
        if (!m_entries.empty()) {
            m_entries.clear();
            ++m_revision;
        }
    }

    /// A non-null pointer means selected, even when its parameter block is null.
    /// An exact submesh entry takes precedence over the all-submeshes entry.
    [[nodiscard]] const std::shared_ptr<const RendererParameterBlock> *Find(RenderDrawIdentity draw) const noexcept
    {
        auto found = m_entries.find(draw);
        if (found == m_entries.end()) {
            draw.primitiveIndex = std::numeric_limits<uint32_t>::max();
            found = m_entries.find(draw);
        }
        return found == m_entries.end() ? nullptr : &found->second;
    }

    [[nodiscard]] const std::shared_ptr<InxMaterial> &Material() const noexcept
    {
        return m_material;
    }
    [[nodiscard]] size_t Size() const noexcept
    {
        return m_entries.size();
    }
    [[nodiscard]] uint64_t Revision() const noexcept
    {
        return m_revision;
    }

  private:
    static RenderDrawIdentity Key(const RenderProxyHandle &renderer, int submesh)
    {
        if (!renderer.IsValid() || !renderer.IsSceneBacked())
            throw std::invalid_argument("RendererSelection requires a live scene renderer");
        if (submesh < -1)
            throw std::invalid_argument("submesh must be -1 (all) or a non-negative index");
        return renderer.MakeDrawIdentity(static_cast<uint32_t>(submesh));
    }

    const std::shared_ptr<InxMaterial> m_material;
    std::unordered_map<RenderDrawIdentity, std::shared_ptr<const RendererParameterBlock>, RenderDrawIdentityHash>
        m_entries;
    uint64_t m_revision = 0;
};

} // namespace infernux
