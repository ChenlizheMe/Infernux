#include "SkinnedMeshRenderer.h"
#include "ComponentFactory.h"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace infernux
{

INFERNUX_REGISTER_COMPONENT("SkinnedMeshRenderer", SkinnedMeshRenderer)

std::string SkinnedMeshRenderer::Serialize() const
{
    json j = json::parse(MeshRenderer::Serialize());
    if (!m_sourceModelGuid.empty())
        j["sourceModelGuid"] = m_sourceModelGuid;
    if (!m_sourceModelPath.empty())
        j["sourceModelPath"] = m_sourceModelPath;
    if (!m_animationTakeNames.empty())
        j["animationTakeNames"] = m_animationTakeNames;
    if (!m_activeTakeName.empty())
        j["activeTakeName"] = m_activeTakeName;
    return j.dump(2);
}

bool SkinnedMeshRenderer::Deserialize(const std::string &jsonStr)
{
    if (!MeshRenderer::Deserialize(jsonStr))
        return false;

    try {
        json j = json::parse(jsonStr);
        m_sourceModelGuid = j.value("sourceModelGuid", std::string());
        m_sourceModelPath = j.value("sourceModelPath", std::string());
        m_activeTakeName = j.value("activeTakeName", std::string());
        m_animationTakeNames.clear();
        if (j.contains("animationTakeNames") && j["animationTakeNames"].is_array()) {
            for (const auto &v : j["animationTakeNames"]) {
                if (v.is_string())
                    m_animationTakeNames.push_back(v.get<std::string>());
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

std::unique_ptr<Component> SkinnedMeshRenderer::Clone() const
{
    auto clone = std::make_unique<SkinnedMeshRenderer>();
    const uint64_t newId = clone->GetComponentID();
    clone->Deserialize(Serialize());
    clone->SetComponentID(newId);
    return clone;
}

} // namespace infernux
