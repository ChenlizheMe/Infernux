#pragma once

#include <cstdint>
#include <function/resources/InxMaterial/MaterialProperty.h>
#include <memory>
#include <unordered_map>

namespace infernux::rhi
{
class ComputeBuffer;
}

namespace infernux
{

class InxMaterial;

/// Immutable per-renderer material-domain parameter publication (descriptor
/// set 0).
///
/// MeshRenderer replaces the shared publication whenever one authored or
/// runtime override changes. RenderWorld/DrawCall can therefore retain a
/// complete value snapshot without locking or cloning the source material.
/// Resolution stays inside this domain: shared Material defaults, persistent
/// Renderer values, then the latest actual runtime-owner write. View state,
/// World environment is resolved into the active View snapshot; View, engine
/// frame state and pass payloads use their own ABI and are never looked up by
/// matching this map's names.
struct RendererParameterBlock
{
    std::unordered_map<std::string, MaterialProperty> properties;
    std::unordered_map<std::string, std::shared_ptr<rhi::ComputeBuffer>> buffers;
    uint64_t revision = 0;
};

/// Mutable authoring object for one explicit draw. Recording the draw validates
/// these values against its material's reflected shader contract and stores a
/// new immutable RendererParameterBlock. Later edits to this object therefore
/// belong to later draw commands.
class DrawParameterBlock
{
  public:
    void SetFloat(const std::string &name, float value);
    void SetVector2(const std::string &name, const glm::vec2 &value);
    void SetVector3(const std::string &name, const glm::vec3 &value);
    void SetVector4(const std::string &name, const glm::vec4 &value);
    void SetColor(const std::string &name, const glm::vec4 &value);
    void SetInt(const std::string &name, int value);
    void SetMatrix(const std::string &name, const glm::mat4 &value);
    void SetFloatArray(const std::string &name, const std::vector<float> &values);
    void SetVector4Array(const std::string &name, const std::vector<glm::vec4> &values);
    void SetTexture(const std::string &name, const std::string &textureGuid);
    void SetBuffer(const std::string &name, std::shared_ptr<rhi::ComputeBuffer> buffer);
    bool Remove(const std::string &name);
    void Clear();

    [[nodiscard]] size_t Size() const noexcept
    {
        return m_properties.size() + m_buffers.size();
    }

    [[nodiscard]] std::shared_ptr<const RendererParameterBlock> Capture(const InxMaterial &material) const;

  private:
    void Set(MaterialProperty property);
    std::unordered_map<std::string, MaterialProperty> m_properties;
    std::unordered_map<std::string, std::shared_ptr<rhi::ComputeBuffer>> m_buffers;
    // Capture reuses an immutable publication while its reflected values are
    // unchanged. Besides avoiding per-draw allocations, pointer identity is
    // the renderer's batching key, so this keeps repeated draws using one
    // DrawParameterBlock on the instanced path.
    mutable std::weak_ptr<const RendererParameterBlock> m_cachedPublication;
};

} // namespace infernux
