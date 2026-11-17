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

Infernux::MaterialShaderPreparation Infernux::EnsureShaderAuthorMaterial(const ShaderStagePair &stages,
                                                                         const std::string &vertexGuid,
                                                                         const std::string &fragmentGuid)
{
    auto &registry = AssetRegistry::Instance();
    auto *database = GetAssetDatabase();
    const auto asset = registry.LoadAsset<ShaderAsset>(fragmentGuid, ResourceType::Shader);
    if (!asset)
        return {false, "Material fragment author is unavailable: " + fragmentGuid};
    const auto vertex =
        vertexGuid.empty() ? nullptr : registry.LoadAsset<ShaderAsset>(vertexGuid, ResourceType::Shader);
    const auto prepared = ShaderLoader::PrepareMaterialProgram(asset, vertex, stages, database);
    auto &entry = m_linkedShaderProgramCache[stages];
    if (!prepared->program) {
        entry.lastError = prepared->error;
        return {false, prepared->error};
    }
    const auto &program = *prepared->program;
    if (!m_renderer->HasShaderProgramArtifact(program.key)) {
        if (!m_renderer->PublishShaderProgramArtifact(program))
            return {false, "Material author program could not be published"};
        const auto &meta = asset->renderMeta;
        m_renderer->StoreShaderRenderMeta(program.key.stages.fragmentKey, meta.cullMode, meta.depthWrite,
                                          meta.depthTest, meta.blend, meta.queue, meta.passTag, meta.stencil,
                                          meta.alphaClip);
    }
    entry.programKey = program.key;
    entry.lastError.clear();
    return {true, {}};
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

        std::vector<std::string> sourcePaths;
        sourcePaths.reserve(roots.size());
        for (const auto &root : roots)
            sourcePaths.push_back(database->GetPathFromGuid(root));
        m_shaderReloadSources = sourcePaths;
        RetireShaderPreparations(sourcePaths);

        ShaderLoader loader;
        const auto changedPath = database->GetPathFromGuid(guid);
        auto changed = loader.Load(changedPath, guid, database).Get<ShaderAsset>();
        if (roots.size() > 1 && (changed->shaderType == "library" || changed->shaderType == "shadingmodel")) {
            // The edited dependency compiled on its own. A later rejection
            // belongs to its consumers, whose separate repairs can resolve it.
            m_shaderReloadSources.erase(
                std::remove(m_shaderReloadSources.begin(), m_shaderReloadSources.end(), changedPath),
                m_shaderReloadSources.end());
        }
        std::vector<std::pair<std::string, std::shared_ptr<ShaderAsset>>> candidates;
        candidates.reserve(roots.size());
        for (size_t index = 0; index < roots.size(); ++index) {
            const auto &root = roots[index];
            auto asset =
                root == guid ? std::move(changed) : loader.Load(sourcePaths[index], root, database).Get<ShaderAsset>();
            candidates.emplace_back(root, std::move(asset));
        }

        std::vector<ShaderProgramArtifact> materials;
        for (const auto &candidate : candidates) {
            if (candidate.second->materialProgram)
                materials.push_back(*candidate.second->materialProgram);
        }

        // A stage is an independent asset. Validate every live combination
        // affected by this edit, using the same source snapshots that will be
        // published; fragment source never chooses or changes a vertex asset.
        const auto snapshot = [&](const std::string &key) -> std::shared_ptr<ShaderAsset> {
            const auto candidate = std::find_if(candidates.begin(), candidates.end(),
                                                [&](const auto &value) { return value.first == key; });
            return candidate != candidates.end() ? candidate->second
                                                 : registry.LoadAsset<ShaderAsset>(key, ResourceType::Shader);
        };
        struct MaterialCandidate
        {
            std::shared_ptr<InxMaterial> material;
            std::shared_ptr<ShaderAsset> fragment;
            std::shared_ptr<const ShaderMaterialPreparation> prepared;
        };
        std::vector<MaterialCandidate> combinations;
        for (const auto &material : registry.GetAllMaterials()) {
            const auto stages = material->GetShaderStages();
            if (!std::binary_search(roots.begin(), roots.end(), stages.vertexKey) &&
                !std::binary_search(roots.begin(), roots.end(), stages.fragmentKey))
                continue;
            if (!asset_formats::FindShaderSourceKind(database->GetPathFromGuid(stages.fragmentKey)))
                continue;
            const auto fragment = snapshot(stages.fragmentKey);
            const auto vertex = material->GetVertShaderReference().guid.empty() ? nullptr : snapshot(stages.vertexKey);
            const auto previous = std::find_if(combinations.begin(), combinations.end(), [&](const auto &value) {
                return value.prepared->program->key.stages == stages;
            });
            const auto prepared = previous != combinations.end()
                                      ? previous->prepared
                                      : ShaderLoader::CompileMaterialProgram(fragment, vertex, stages, database);
            if (!prepared->program) {
                m_shaderReloadDiagnostics = prepared->diagnostics;
                return prepared->error;
            }
            const auto already = std::find_if(materials.begin(), materials.end(),
                                              [&](const auto &value) { return value.key.stages == stages; });
            if (already == materials.end())
                materials.push_back(*prepared->program);
            combinations.push_back({material, fragment, prepared});
        }
        if (m_renderer && !materials.empty() && !m_renderer->PublishShaderProgramArtifacts(materials))
            throw std::runtime_error("Mesh author candidate programs could not be published");

        // PublishShader moves accepted candidate contents into the resident
        // asset. Populate the candidate cache before that move; entries retain
        // the immutable vertex compilation rather than its mutable asset slot.
        for (const auto &combination : combinations) {
            const auto &program = *combination.prepared->program;
            const std::lock_guard lock(combination.fragment->materialPrograms->mutex);
            combination.fragment->materialPrograms->entries[program.key.stages.vertexKey] = combination.prepared;
        }

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
        for (const auto &combination : combinations) {
            const auto &program = *combination.prepared->program;
            combination.material->SynchronizeShaderPropertyDefaults(program);
            m_linkedShaderProgramCache.erase(program.key.stages);
        }
        m_shaderReloadSources = std::move(sourcePaths);
        return {};
    } catch (const ShaderCompilationError &error) {
        m_shaderReloadDiagnostics = error.diagnostics;
        return error.what();
    } catch (const std::exception &error) {
        return error.what();
    }
}

} // namespace infernux
