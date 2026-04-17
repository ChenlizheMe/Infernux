#include "SpriteRenderer.h"
#include "ComponentFactory.h"
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/scene/PrimitiveMeshes.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace infernux
{

INFERNUX_REGISTER_COMPONENT("SpriteRenderer", SpriteRenderer)

SpriteRenderer::SpriteRenderer()
{
    // Auto-set a Quad mesh — the sprite always renders on a quad.
    SetSharedPrimitiveMesh(PrimitiveMeshes::GetQuadVertices(), PrimitiveMeshes::GetQuadIndices(), "Quad");
    SetCastShadows(false);
}

std::shared_ptr<InxMaterial> SpriteRenderer::GetEffectiveMaterial(uint32_t slot) const
{
    auto mat = GetMaterial(slot);
    if (mat) {
        if (!mat->IsDeleted())
            return mat;
        auto &registry = AssetRegistry::Instance();
        auto err = registry.GetBuiltinMaterial("ErrorMaterial");
        return err ? err : registry.GetBuiltinMaterial("DefaultUnlit");
    }
    // Sprites default to unlit, not lit
    return AssetRegistry::Instance().GetBuiltinMaterial("DefaultUnlit");
}

std::string SpriteRenderer::Serialize() const
{
    // Start with the base MeshRenderer JSON, then append sprite fields.
    std::string baseJson = MeshRenderer::Serialize();
    json j = json::parse(baseJson);

    // Override the type tag so we deserialize back as SpriteRenderer.
    j["type"] = "SpriteRenderer";

    // Sprite-specific fields
    if (!m_spriteGuid.empty())
        j["spriteGuid"] = m_spriteGuid;
    j["frameIndex"] = m_frameIndex;
    j["spriteColor"] = {m_color.r, m_color.g, m_color.b, m_color.a};
    j["flipX"] = m_flipX;
    j["flipY"] = m_flipY;

    return j.dump(2);
}

bool SpriteRenderer::Deserialize(const std::string &jsonStr)
{
    if (!MeshRenderer::Deserialize(jsonStr))
        return false;

    try {
        json j = json::parse(jsonStr);

        if (j.contains("spriteGuid") && j["spriteGuid"].is_string())
            m_spriteGuid = j["spriteGuid"].get<std::string>();

        if (j.contains("frameIndex"))
            m_frameIndex = j["frameIndex"].get<int>();

        if (j.contains("spriteColor") && j["spriteColor"].is_array() && j["spriteColor"].size() == 4) {
            m_color.r = j["spriteColor"][0].get<float>();
            m_color.g = j["spriteColor"][1].get<float>();
            m_color.b = j["spriteColor"][2].get<float>();
            m_color.a = j["spriteColor"][3].get<float>();
        }

        if (j.contains("flipX"))
            m_flipX = j["flipX"].get<bool>();
        if (j.contains("flipY"))
            m_flipY = j["flipY"].get<bool>();

        // Ensure Quad mesh is set after deserialization
        if (!HasInlineMesh()) {
            SetSharedPrimitiveMesh(PrimitiveMeshes::GetQuadVertices(), PrimitiveMeshes::GetQuadIndices(), "Quad");
        }

        return true;
    } catch (const std::exception &) {
        return false;
    }
}

std::unique_ptr<Component> SpriteRenderer::Clone() const
{
    // Clone the MeshRenderer base via its Clone, then cast and copy sprite fields.
    auto baseClone = MeshRenderer::Clone();
    // MeshRenderer::Clone returns a MeshRenderer — we need a SpriteRenderer.
    auto clone = std::make_unique<SpriteRenderer>();
    // Copy base MeshRenderer state by re-deserializing
    clone->Deserialize(Serialize());
    return clone;
}

} // namespace infernux
