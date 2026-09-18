#include "InxMesh.h"

#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <stdexcept>

namespace infernux
{
namespace
{
constexpr std::array<const char *, ModelTextureCount> TextureProperties{
    "texSampler", "normalMap", "metallicMap", "smoothnessMap", "aoMap", "emissionMap"};

std::string TextureIdentity(const MaterialSlotData &data, size_t index)
{
    return data.textureGuids[index].empty()
               ? (index == static_cast<size_t>(ModelTexture::Normal) ? "normal" : "white")
               : data.textureGuids[index];
}

glm::vec4 MetallicChannels(const MaterialSlotData &data)
{
    return data.packedMetallicRoughness ? glm::vec4(0, 0, 1, 0) : glm::vec4(1, 0, 0, 0);
}

glm::vec4 RoughnessChannels(const MaterialSlotData &data)
{
    return data.packedMetallicRoughness ? glm::vec4(0, 1, 0, 0) : glm::vec4(1, 0, 0, 0);
}

float UsesRoughnessMap(const MaterialSlotData &data)
{
    return data.textureGuids[static_cast<size_t>(ModelTexture::Roughness)].empty() ? 0.0f : 1.0f;
}
glm::vec4 SourceBaseColor(const MaterialSlotData &data)
{
    auto baseColor = data.baseColor;
    if (data.alphaMode == ModelAlphaMode::Opaque)
        baseColor.a = 1.0f;
    return baseColor;
}

void ApplySourceSurface(RenderState &state, const MaterialSlotData &data)
{
    state.cullMode = data.doubleSided ? MaterialCullMode::None : MaterialCullMode::Back;
    state.blendEnable = data.alphaMode == ModelAlphaMode::Blend;
    state.depthWriteEnable = !state.blendEnable;
    state.alphaClipEnabled = data.alphaMode == ModelAlphaMode::Mask;
    state.alphaClipThreshold = data.alphaCutoff;
    state.renderQueue = state.blendEnable ? 3000 : (state.alphaClipEnabled ? 2450 : 2000);
    if (state.blendEnable) {
        state.srcColorBlendFactor = MaterialBlendFactor::SourceAlpha;
        state.dstColorBlendFactor = MaterialBlendFactor::OneMinusSourceAlpha;
        state.srcAlphaBlendFactor = MaterialBlendFactor::One;
        state.dstAlphaBlendFactor = MaterialBlendFactor::OneMinusSourceAlpha;
    }
}
} // namespace

bool InxMesh::MatchesMaterialCopy(uint32_t slot, const InxMaterial &material) const
{
    if (slot >= m_materialSlotData.size())
        return false;
    const auto &data = m_materialSlotData[slot];
    const auto matches = [&material](const char *key, const auto &expected) {
        const auto *property = material.GetProperty(key);
        const auto *value = property ? std::get_if<std::decay_t<decltype(expected)>>(&property->value) : nullptr;
        return value && *value == expected;
    };
    auto state = material.GetRenderState();
    ApplySourceSurface(state, data);
    const auto name = slot < m_materialSlotNames.size() && !m_materialSlotNames[slot].empty()
                          ? m_materialSlotNames[slot] : "EmbeddedMaterial_" + std::to_string(slot);
    const auto path = m_filePath.empty() ? std::string() : m_filePath + "::submat:" + std::to_string(slot);
    for (size_t index = 0; index < ModelTextureCount; ++index)
        if (!matches(TextureProperties[index], TextureIdentity(data, index)))
            return false;
    return material.GetName() == name && material.GetFilePath() == path && state == material.GetRenderState() &&
           matches("baseColor", SourceBaseColor(data)) && matches("emissionColor", data.emissionColor) &&
           matches("normalScale", data.normalScale) && matches("occlusionStrength", data.occlusionStrength) &&
           matches("metallicChannels", MetallicChannels(data)) && matches("smoothnessChannels", RoughnessChannels(data)) &&
           matches("smoothnessFromRoughness", UsesRoughnessMap(data)) &&
           matches("metallic", data.metallic) && matches("smoothness", data.smoothness);
}

std::shared_ptr<InxMaterial> InxMesh::CreateMaterialCopy(uint32_t slot) const
{
    if (slot >= m_materialSlotData.size())
        throw std::out_of_range("model has no imported material at this slot");
    auto base = AssetRegistry::Instance().GetBuiltinMaterial("DefaultLit");
    if (!base)
        base = InxMaterial::CreateDefaultLit();
    auto material = base->Clone();
    const auto &data = m_materialSlotData[slot];
    material->SetColor("baseColor", SourceBaseColor(data));
    material->SetColor("emissionColor", data.emissionColor);
    material->SetFloat("metallic", data.metallic);
    material->SetFloat("smoothness", data.smoothness);
    for (size_t index = 0; index < ModelTextureCount; ++index)
        material->SetTextureGuid(TextureProperties[index], TextureIdentity(data, index));
    material->SetFloat("normalScale", data.normalScale);
    material->SetFloat("occlusionStrength", data.occlusionStrength);
    material->SetVector4("metallicChannels", MetallicChannels(data));
    material->SetVector4("smoothnessChannels", RoughnessChannels(data));
    material->SetFloat("smoothnessFromRoughness", UsesRoughnessMap(data));
    auto state = material->GetRenderState();
    ApplySourceSurface(state, data);
    material->SetRenderState(state);
    material->SyncAlphaClipProperty();
    material->SetName(slot < m_materialSlotNames.size() && !m_materialSlotNames[slot].empty()
                          ? m_materialSlotNames[slot]
                          : "EmbeddedMaterial_" + std::to_string(slot));
    if (!m_filePath.empty())
        material->SetFilePath(m_filePath + "::submat:" + std::to_string(slot));
    return material;
}
} // namespace infernux
