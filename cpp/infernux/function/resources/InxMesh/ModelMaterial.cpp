#include "InxMesh.h"

#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <stdexcept>

namespace infernux
{
std::shared_ptr<InxMaterial> InxMesh::CreateMaterialCopy(uint32_t slot) const
{
    if (slot >= m_materialSlotData.size())
        throw std::out_of_range("model has no imported material at this slot");
    auto base = AssetRegistry::Instance().GetBuiltinMaterial("DefaultLit");
    if (!base)
        base = InxMaterial::CreateDefaultLit();
    auto material = base->Clone();
    const auto &data = m_materialSlotData[slot];
    material->SetColor("baseColor", data.baseColor);
    material->SetColor("emissionColor", data.emissionColor);
    material->SetFloat("metallic", data.metallic);
    material->SetFloat("smoothness", data.smoothness);
    material->SetName(slot < m_materialSlotNames.size() && !m_materialSlotNames[slot].empty()
                          ? m_materialSlotNames[slot]
                          : "EmbeddedMaterial_" + std::to_string(slot));
    if (!m_filePath.empty())
        material->SetFilePath(m_filePath + "::submat:" + std::to_string(slot));
    return material;
}
} // namespace infernux
