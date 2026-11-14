#include "ShaderLoader.h"

#include <core/log/InxLog.h>
#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/AssetDependencyGraph.h>
#include <function/resources/AssetFormatRegistry.h>
#include <function/resources/InxResource/InxResourceMeta.h>
#include <function/resources/ShaderAsset/ShaderAsset.h>
#include <function/resources/ShaderAsset/ShaderAuthorMaterial.h>
#include <function/resources/ShaderAsset/ShaderAuthorMetadata.h>
#include <function/resources/ShaderAsset/SlangCompiler.h>

#include <nlohmann/json.hpp>
namespace infernux
{
std::shared_ptr<const ShaderMaterialPreparation>
ShaderLoader::CompileMaterialProgram(const std::shared_ptr<ShaderAsset> &fragment,
                                     const std::shared_ptr<ShaderAsset> &vertex, const ShaderStagePair &stages,
                                     AssetDatabase *database)
{
    auto result = std::make_shared<ShaderMaterialPreparation>();
    result->vertex = vertex ? vertex->program : nullptr;
    try {
        if (!fragment->materialProgram)
            throw std::invalid_argument("Material fragment must declare a mesh or particle shader: " +
                                        fragment->filePath);
        if (fragment->programDomain == ShaderProgramDomain::ParticleSprite) {
            if (!stages.vertexKey.empty())
                throw std::invalid_argument(
                    "Particle geometry is provided by its output; remove the material vertex selection: " +
                    fragment->filePath);
            result->program = fragment->materialProgram;
            return result;
        }
        const bool standard = stages.vertexKey == infernux::kStandardVertexShaderId ||
                              (vertex && vertex->shaderId == infernux::kStandardVertexShaderId);
        if (standard) {
            if (fragment->materialProgram->key.stages == stages)
                result->program = fragment->materialProgram;
            else {
                auto program = std::make_shared<ShaderProgramArtifact>(*fragment->materialProgram);
                program->key.stages = stages;
                result->program = std::move(program);
            }
        } else {
            if (!vertex || vertex->shaderType != "vertex" || !vertex->program)
                throw std::invalid_argument("Select a Python vertex shader or infernux/standard for " +
                                            fragment->filePath);
            auto &compiler = SlangCompiler::Get();
            const auto prepared = compiler.PrepareAuthorSource(
                {"material_" + stages.fragmentKey, fragment->filePath, fragment->authorSourceText},
                [database](const std::string &path) -> std::optional<std::string> {
                    std::vector<char> bytes;
                    if (!database->ReadFile(path, bytes))
                        return std::nullopt;
                    return std::string(bytes.begin(), bytes.end());
                },
                ShaderAuthorPhase::Program, {"vertex_" + stages.vertexKey, vertex->filePath, vertex->authorSourceText});
            const auto compilation = std::make_shared<const ShaderCompilation>(compiler.CompileAuthorProgram(prepared));
            result->program = std::make_shared<const ShaderProgramArtifact>(
                CreateShaderAuthorMaterial(stages.fragmentKey, compilation, prepared.schema, stages.vertexKey));
        }
    } catch (const ShaderCompilationError &error) {
        result->error = error.what();
        result->diagnostics = error.diagnostics;
    } catch (const std::invalid_argument &error) {
        result->error = error.what();
    }
    return result;
}

std::shared_ptr<const ShaderMaterialPreparation>
ShaderLoader::PrepareMaterialProgram(const std::shared_ptr<ShaderAsset> &fragment,
                                     const std::shared_ptr<ShaderAsset> &vertex, const ShaderStagePair &stages,
                                     AssetDatabase *database)
{
    auto &cache = *fragment->materialPrograms;
    {
        const std::lock_guard lock(cache.mutex);
        const auto found = cache.entries.find(stages.vertexKey);
        if (found != cache.entries.end() && found->second->vertex == (vertex ? vertex->program : nullptr))
            return found->second;
    }
    // The static frontend can acquire the interpreter; do not hold a cache
    // mutex across Python compilation or native Slang work.
    auto prepared = CompileMaterialProgram(fragment, vertex, stages, database);
    const std::lock_guard lock(cache.mutex);
    cache.entries[stages.vertexKey] = prepared;
    return prepared;
}

void ShaderLoader::OnRelocated(const RuntimeAssetPayload &payload, const std::string &filePath)
{
    const auto shader = payload.Get<ShaderAsset>();
    shader->filePath = filePath;
    shader->descriptor.filePath = filePath;
}

// =============================================================================
// Internal helper — compile a shader file into a ShaderAsset
// =============================================================================

std::shared_ptr<ShaderAsset> ShaderLoader::CompileAuthorAsset(const ShaderSourceModule &source,
                                                              const ShaderAuthorSource &prepared,
                                                              const std::string &guid,
                                                              std::vector<std::string> dependencies)
{
    const auto *kind = asset_formats::FindShaderSourceKind(source.path);
    const auto schema = nlohmann::json::parse(prepared.schema);
    const auto &info = schema.at("info");
    auto asset = std::make_shared<ShaderAsset>();
    asset->shaderId = info.at("shader_id");
    asset->shaderType = kind->stage;
    asset->filePath = source.path;
    asset->descriptor.shaderId = asset->shaderId;
    asset->descriptor.isFragmentShader = kind->stage == "fragment";
    asset->descriptor.isVertexShader = kind->stage == "vertex";
    asset->descriptor.filePath = source.path;
    asset->descriptor.fileExtension = kind->suffix;
    asset->descriptor.hidden = info.at("hidden").get<bool>();
    asset->authoringSchema = prepared.schema;
    asset->authorSourceText = source.source;
    asset->renderMeta = ReadShaderAuthorRenderMeta(info);
    if (schema.at("domain") == "fullscreen")
        asset->programDomain = ShaderProgramDomain::Fullscreen;
    else if (schema.at("domain") == "particle")
        asset->programDomain = ShaderProgramDomain::ParticleSprite;
    else if (schema.at("domain") == "mesh" || schema.at("domain") == "vertex")
        asset->programDomain = ShaderProgramDomain::Mesh;
    asset->sourceDependencies = std::move(dependencies);
    asset->program = std::make_shared<const ShaderCompilation>(SlangCompiler::Get().CompileAuthorProgram(prepared));
    if (schema.at("domain") == "mesh" || schema.at("domain") == "particle") {
        asset->materialProgram = std::make_shared<const ShaderProgramArtifact>(CreateShaderAuthorMaterial(
            guid, asset->program, prepared.schema,
            asset->programDomain == ShaderProgramDomain::ParticleSprite ? "" : kStandardVertexShaderId));
    }
    return asset;
}

static std::shared_ptr<ShaderAsset> CompileShaderAsset(const std::string &filePath, const std::string &guid,
                                                       AssetDatabase *database)
{
    if (!asset_formats::FindShaderSourceKind(filePath))
        throw std::invalid_argument("Shader assets use typed Python GPU source: " + filePath);
    std::vector<char> bytes;
    if (!database->ReadFile(filePath, bytes))
        throw std::runtime_error("GPU author source could not be read: " + filePath);
    const ShaderSourceModule source{"asset_" + guid, filePath, std::string(bytes.begin(), bytes.end())};
    const auto prepared = SlangCompiler::Get().PrepareAuthorSource(
        source, [database](const std::string &path) -> std::optional<std::string> {
            std::vector<char> contents;
            if (!database->ReadFile(path, contents))
                return std::nullopt;
            return std::string(contents.begin(), contents.end());
        });
    std::vector<std::string> dependencies;
    for (const auto &library : prepared.libraries) {
        if (!asset_formats::FindShaderSourceKind(library.path))
            continue;
        auto dependency = database->GetGuidFromPath(library.path);
        if (!dependency.empty())
            dependencies.push_back(std::move(dependency));
    }
    return ShaderLoader::CompileAuthorAsset(source, prepared, guid, std::move(dependencies));
}

// =============================================================================
// Load
// =============================================================================

RuntimeAssetPayload ShaderLoader::Load(const std::string &filePath, const std::string &guid, AssetDatabase *adb)
{
    return CompileShaderAsset(filePath, guid, adb);
}

void ShaderLoader::OnPublished(const RuntimeAssetPayload &payload, const std::string &guid, AssetDatabase *adb)
{
    const auto asset = payload.Get<ShaderAsset>();
    if (!asset->program)
        return;
    auto &graph = *adb->GetDependencyGraph();
    graph.ClearRuntimeDependenciesOf(guid);
    for (const auto &dependency : asset->sourceDependencies)
        graph.AddRuntimeDependency(guid, dependency);
}

// =============================================================================
// Reload — recompile and replace in-place
// =============================================================================

bool ShaderLoader::Reload(const RuntimeAssetPayload &existing, const std::string &filePath, const std::string &guid,
                          AssetDatabase *adb)
{
    auto oldAsset = existing.Get<ShaderAsset>();
    if (!oldAsset) {
        INXLOG_WARN("ShaderLoader::Reload: null existing instance");
        return false;
    }

    auto newAsset = CompileShaderAsset(filePath, guid, adb);
    if (!newAsset) {
        return false;
    }

    // Replace data in-place (preserving shared_ptr identity)
    *oldAsset = std::move(*newAsset);
    return true;
}

// =============================================================================
// Prepared source dependencies are published with the accepted program.
// =============================================================================

size_t ShaderLoader::EstimateRuntimeBytes(const RuntimeAssetPayload &payload) const
{
    const auto shader = payload.Get<ShaderAsset>();
    if (!shader)
        throw std::invalid_argument("ShaderLoader cannot estimate an empty runtime payload");
    return shader->GetRuntimeMemoryBytes();
}

std::set<std::string> ShaderLoader::ScanDependencies(const std::string & /*filePath*/, AssetDatabase * /*adb*/)
{
    return {};
}

// =============================================================================
// CreateMeta — use the same static author declarations as compilation
// =============================================================================

void ShaderLoader::CreateMeta(AssetSource &source, InxResourceMeta &metaData) const
{
    CreateShaderAuthorMetadata(source, metaData);
}

} // namespace infernux
