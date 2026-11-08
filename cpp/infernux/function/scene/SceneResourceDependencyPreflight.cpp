#include "SceneResourceDependencyPreflight.h"
#include "SerializedResourceReferences.h"

#include <algorithm>
#include <core/types/InxFwdType.h>
#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/AssetDependencyGraph.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxResource/InxResourceMeta.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace infernux
{
namespace
{

const char *ResourceTypeName(ResourceType type)
{
    switch (type) {
    case ResourceType::Meta:
        return "Meta";
    case ResourceType::Shader:
        return "Shader";
    case ResourceType::Texture:
        return "Texture";
    case ResourceType::Mesh:
        return "Mesh";
    case ResourceType::Material:
        return "Material";
    case ResourceType::Script:
        return "Script";
    case ResourceType::Audio:
        return "Audio";
    case ResourceType::DefaultText:
        return "DefaultText";
    case ResourceType::DefaultBinary:
        return "DefaultBinary";
    case ResourceType::PhysicMaterial:
        return "PhysicMaterial";
    case ResourceType::RenderEffect:
        return "RenderEffect";
    case ResourceType::ParticleGraph:
        return "ParticleGraph";
    case ResourceType::DataAsset:
        return "DataAsset";
    case ResourceType::RenderTexture:
        return "RenderTexture";
    }
    return "Unknown";
}

class ResourcePreflight
{
  public:
    void Validate(const nlohmann::json &document)
    {
        VisitSceneResourceReferences(
            document, [this](const auto &guid, auto type, const auto &path) { RequireAsset(guid, type, path); });
        ExpandAssetDependencies();
    }

    [[nodiscard]] std::vector<std::pair<std::string, std::string>> Dependencies() const
    {
        std::vector<std::pair<std::string, std::string>> result;
        result.reserve(m_assetTypes.size());
        for (const auto &[guid, type] : m_assetTypes)
            result.emplace_back(guid, ResourceTypeName(type));
        std::sort(result.begin(), result.end());
        return result;
    }

  private:
    void ExpandAssetDependencies()
    {
        AssetDatabase &database = GetDatabase("Scene dependencies");
        std::vector<std::string> pending;
        pending.reserve(m_assetTypes.size());
        for (const auto &[guid, _] : m_assetTypes)
            pending.push_back(guid);

        for (size_t index = 0; index < pending.size(); ++index) {
            for (const auto &dependencyGuid : AssetDependencyGraph::Instance().GetDependencies(pending[index])) {
                if (dependencyGuid.empty() || m_assetTypes.find(dependencyGuid) != m_assetTypes.end())
                    continue;
                const auto metadata = database.GetMetaByGuid(dependencyGuid);
                if (!metadata)
                    continue;
                m_assetTypes.emplace(dependencyGuid, metadata->GetResourceType());
                pending.push_back(dependencyGuid);
            }
        }
    }

    void RequireAsset(const std::string &guid, ResourceType expectedType, const std::string &path)
    {
        if (guid.empty())
            throw std::invalid_argument(path + " must not be empty");
        if (const auto cached = m_assetTypes.find(guid); cached != m_assetTypes.end()) {
            if (cached->second != expectedType)
                throw std::invalid_argument(path + " expects " + ResourceTypeName(expectedType) + " but GUID '" + guid +
                                            "' was already validated as " + ResourceTypeName(cached->second));
            return;
        }

        AssetDatabase &database = GetDatabase(path);

        const auto metadata = database.GetMetaByGuid(guid);
        if (!metadata) {
            // Missing references are valid authoring state.  The field itself
            // provides the expected type and the stable GUID allows Undo or a
            // later import to reconnect without rewriting the scene.
            m_assetTypes.emplace(guid, expectedType);
            return;
        }
        const ResourceType actualType = metadata->GetResourceType();
        if (actualType != expectedType) {
            throw std::invalid_argument(path + " expects " + ResourceTypeName(expectedType) + " but GUID '" + guid +
                                        "' is " + ResourceTypeName(actualType));
        }
        m_assetTypes.emplace(guid, actualType);
    }

    AssetDatabase &GetDatabase(const std::string &path)
    {
        if (m_database != nullptr)
            return *m_database;
        auto &registry = AssetRegistry::Instance();
        m_database = registry.GetAssetDatabase();
        if (!registry.IsInitialized() || m_database == nullptr)
            throw std::logic_error(path + " requires an initialized AssetDatabase");
        if (!m_database->IsOwnerThread())
            throw std::logic_error("scene resource dependency preflight must run on the AssetDatabase owner thread");
        return *m_database;
    }

    AssetDatabase *m_database = nullptr;
    std::unordered_map<std::string, ResourceType> m_assetTypes;
};

} // namespace

void PreflightSceneResourceDependencies(const nlohmann::json &document)
{
    ResourcePreflight{}.Validate(document);
}

std::vector<std::pair<std::string, std::string>> CollectSceneResourceDependencies(const nlohmann::json &document)
{
    ResourcePreflight preflight;
    preflight.Validate(document);
    return preflight.Dependencies();
}

} // namespace infernux
