#include "Infernux.h"

#include <function/renderer/InxRenderer.h>
#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/AssetDependencyGraph.h>
#include <function/resources/AssetFormatRegistry.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/ShaderAsset/ShaderAsset.h>
#include <function/resources/ShaderAsset/ShaderLoader.h>

#include <algorithm>
#include <stdexcept>

namespace infernux
{

std::string Infernux::ReloadShaderAuthorSource(const std::string &shaderPath)
{
    auto &registry = AssetRegistry::Instance();
    auto *database = registry.GetAssetDatabase();
    try {
        const auto guid = database->GetGuidFromPath(shaderPath);
        if (guid.empty())
            throw std::invalid_argument("GPU author source is not imported: " + shaderPath);
        std::vector<std::string> roots{guid};
        for (const auto &dependent : database->GetDependencyGraph()->GetDependents(guid))
            if (asset_formats::FindShaderSourceKind(database->GetPathFromGuid(dependent)))
                roots.push_back(dependent);
        std::sort(roots.begin(), roots.end());
        roots.erase(std::unique(roots.begin(), roots.end()), roots.end());

        m_shaderReloadSources.clear();
        for (const auto &root : roots)
            m_shaderReloadSources.push_back(database->GetPathFromGuid(root));

        ShaderLoader loader;
        std::vector<std::pair<std::string, std::shared_ptr<ShaderAsset>>> candidates;
        for (const auto &root : roots) {
            const auto path = database->GetPathFromGuid(root);
            candidates.emplace_back(root, loader.Load(path, root, database).Get<ShaderAsset>());
        }

        // All frontend and target work succeeds before any program generation
        // becomes visible. Source dependencies are published by the loader at
        // this same owner boundary; device readers retain immutable code.
        for (auto &[root, asset] : candidates) {
            const bool executable = asset->shaderType != "library";
            registry.PublishShader(root, std::move(asset));
            if (m_renderer && executable)
                m_renderer->InvalidateShaderCache(root);
        }
        return {};
    } catch (const std::exception &error) {
        return error.what();
    }
}

} // namespace infernux
