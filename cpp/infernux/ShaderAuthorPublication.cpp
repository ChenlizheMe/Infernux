#include "Infernux.h"

#include <function/renderer/InxRenderer.h>
#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/AssetDependencyGraph.h>
#include <function/resources/AssetFormatRegistry.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/ShaderAsset/ShaderAsset.h>
#include <function/resources/ShaderAsset/ShaderLoader.h>
#include <function/resources/ShaderAsset/SlangCompiler.h>

#include <algorithm>
#include <stdexcept>

namespace infernux
{

Infernux::LinkedShaderProgramPreparation Infernux::EnsureShaderAuthorMaterial(const ShaderStagePair &stages,
                                                                              const std::string &fragmentPath)
{
    auto &registry = AssetRegistry::Instance();
    const auto guid = GetAssetDatabase()->GetGuidFromPath(fragmentPath);
    const auto asset = registry.LoadAsset<ShaderAsset>(guid, ResourceType::Shader);
    if (!asset || !asset->materialProgram)
        return {true, false, "This GPU author is not a mesh material: " + fragmentPath};
    const auto &program = *asset->materialProgram;
    if (stages != program.key.stages)
        return {true, false, "Mesh fragment requires its declared standard vertex program"};
    if (!m_renderer->HasShaderProgramArtifact(program.key)) {
        if (!m_renderer->PublishShaderProgramArtifact(program))
            return {true, false, "Mesh author program could not be published"};
        const auto &meta = asset->renderMeta;
        m_renderer->StoreShaderRenderMeta(program.key.stages.fragmentKey, meta.cullMode, meta.depthWrite,
                                          meta.depthTest, meta.blend, meta.queue, meta.passTag, meta.stencil,
                                          meta.alphaClip);
    }
    return {true, true, {}};
}

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

        std::vector<ShaderProgramArtifact> materials;
        std::vector<std::pair<std::string, ShaderProgramKey>> replacedRecipes;
        for (const auto &candidate : candidates) {
            if (candidate.second->materialProgram) {
                materials.push_back(*candidate.second->materialProgram);
                if (registry.IsLoaded(candidate.first)) {
                    const auto resident = registry.LoadAsset<ShaderAsset>(candidate.first, ResourceType::Shader);
                    if (resident->materialProgram &&
                        resident->materialProgram->key.stages != candidate.second->materialProgram->key.stages)
                        replacedRecipes.emplace_back(candidate.first, resident->materialProgram->key);
                }
            }
        }
        if (m_renderer && !materials.empty() && !m_renderer->PublishShaderProgramArtifacts(materials))
            throw std::runtime_error("Mesh author candidate programs could not be published");

        // All frontend and target work succeeds before any program generation
        // becomes visible. Source dependencies are published by the loader at
        // this same owner boundary; device readers retain immutable code.
        for (auto &[root, asset] : candidates) {
            const bool fullscreen = asset->programDomain == ShaderProgramDomain::Fullscreen;
            if (m_renderer && asset->materialProgram) {
                const auto &meta = asset->renderMeta;
                m_renderer->StoreShaderRenderMeta(asset->materialProgram->key.stages.fragmentKey, meta.cullMode,
                                                  meta.depthWrite, meta.depthTest, meta.blend, meta.queue, meta.passTag,
                                                  meta.stencil, meta.alphaClip);
            }
            registry.PublishShader(root, std::move(asset));
            if (m_renderer && fullscreen)
                m_renderer->InvalidateShaderCache(root);
        }
        if (!replacedRecipes.empty()) {
            for (const auto &material : registry.GetAllMaterials()) {
                const auto &fragment = material->GetFragShaderReference().guid;
                if (std::any_of(replacedRecipes.begin(), replacedRecipes.end(),
                                [&](const auto &recipe) { return recipe.first == fragment; }))
                    material->PrepareShaderDefaults();
            }
            if (m_renderer)
                for (const auto &[root, key] : replacedRecipes)
                    m_renderer->RetireShaderProgramArtifact(key);
        }
        return {};
    } catch (const ShaderCompilationError &error) {
        m_shaderReloadDiagnostics = error.diagnostics;
        return error.what();
    } catch (const std::exception &error) {
        return error.what();
    }
}

} // namespace infernux
