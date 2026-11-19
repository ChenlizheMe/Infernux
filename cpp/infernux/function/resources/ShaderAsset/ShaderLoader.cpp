#include "ShaderLoader.h"

#include <core/log/InxLog.h>
#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/AssetDependencyGraph.h>
#include <function/resources/AssetFormatRegistry.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxResource/InxResourceMeta.h>
#include <function/resources/ShaderAsset/ShaderAsset.h>
#include <function/resources/ShaderAsset/ShaderAuthorMaterial.h>
#include <function/resources/ShaderAsset/ShaderAuthorMetadata.h>
#include <function/resources/ShaderAsset/ShaderAuthorProgramCache.h>
#include <function/resources/ShaderAsset/SlangCompiler.h>

#include <nlohmann/json.hpp>
namespace infernux
{
namespace
{
ShaderTextReader TrackAuthorReads(const AssetDatabase &database, std::vector<ShaderCacheInput> &inputs)
{
    return [&database, &inputs](const std::string &path) -> std::optional<std::string> {
        std::vector<char> bytes;
        if (!database.ReadFile(path, bytes))
            return std::nullopt;
        std::string text(bytes.begin(), bytes.end());
        inputs.push_back(MakeShaderCacheInput(ShaderAuthorInputId(database, path), text, path));
        return text;
    };
}

ShaderSourceModule ReadAuthorSource(const AssetDatabase &database, const std::string &path, std::string name)
{
    std::vector<char> bytes;
    if (!database.ReadFile(path, bytes))
        throw std::runtime_error("GPU author source could not be read: " + path);
    return {std::move(name), path, std::string(bytes.begin(), bytes.end())};
}

ShaderAuthorProgramRecord CompileAuthorRecord(const ShaderAuthorSource &prepared, ShaderCodeTarget target,
                                              std::vector<std::string> dependencies)
{
    return {prepared.schema,
            std::make_shared<const ShaderCompilation>(SlangCompiler::Get().CompileAuthorProgram(prepared, {target})),
            std::move(dependencies)};
}
} // namespace

void ShaderLoader::PrepareMaterial(InxMaterial &material)
{
    if (material.IsRenderPublication()) {
        if (!material.GetSelectedShaderArtifact())
            throw ShaderProgramSelectionError("Render publication has no prepared shader author contract");
        return;
    }
    try {
        auto &registry = AssetRegistry::Instance();
        auto *database = registry.GetAssetDatabase();
        const auto resolve = [database](const ShaderAssetReference &reference, const char *stage) {
            if (!reference.guid.empty())
                return reference.guid;
            return reference.pathHint.empty() ? database->FindShaderGuidById(reference.shaderId, stage) : std::string{};
        };
        const auto fragmentGuid = resolve(material.GetFragShaderReference(), "fragment");
        const auto vertexGuid = resolve(material.GetVertShaderReference(), "vertex");
        const auto fragment = registry.LoadAsset<ShaderAsset>(fragmentGuid, ResourceType::Shader);
        if (!fragment)
            throw ShaderProgramSelectionError("Material fragment author is unavailable: " + fragmentGuid);
        const auto vertex =
            vertexGuid.empty() ? nullptr : registry.LoadAsset<ShaderAsset>(vertexGuid, ResourceType::Shader);
        const auto prepared = PrepareMaterialProgram(fragment, vertex, material.GetShaderStages(), database);
        if (!prepared->program)
            throw ShaderProgramSelectionError(prepared->error);
        if (material.m_selectedShaderArtifact != prepared->program) {
            material.InvalidateShaderSelection();
            material.SynchronizeShaderPropertyDefaults(*prepared->program);
            const auto &meta = fragment->renderMeta;
            material.ApplyShaderRenderMeta(meta.cullMode, meta.depthWrite, meta.depthTest, meta.blend, meta.queue,
                                           meta.passTag, meta.stencil, meta.alphaClip);
            material.m_selectedShaderArtifact = prepared->program;
        }
        material.m_shaderProgramDiagnostic.reset();
    } catch (const ShaderProgramSelectionError &error) {
        material.m_selectedShaderArtifact.reset();
        material.ReportShaderProgramError(error.what());
        throw;
    } catch (...) {
        material.InvalidateShaderSelection();
        throw;
    }
}

std::shared_ptr<const ShaderMaterialPreparation>
ShaderLoader::CompileMaterialProgram(const std::shared_ptr<ShaderAsset> &fragment,
                                     const std::shared_ptr<ShaderAsset> &vertex, const ShaderStagePair &stages,
                                     AssetDatabase *database)
{
    auto result = std::make_shared<ShaderMaterialPreparation>();
    result->vertex = vertex ? vertex->program : nullptr;
    try {
        if (fragment->programDomain != ShaderProgramDomain::Mesh && !stages.vertexKey.empty())
            throw std::invalid_argument("Mesh shader vertex requires a Mesh-domain fragment: " + fragment->filePath);
        if (!fragment->materialProgram)
            throw std::invalid_argument("Material requires a graphics fragment author: " + fragment->filePath);
        if (fragment->programDomain != ShaderProgramDomain::Mesh) {
            if (fragment->materialProgram->key.stages == stages)
                result->program = fragment->materialProgram;
            else {
                auto program = std::make_shared<ShaderProgramArtifact>(*fragment->materialProgram);
                program->key.stages = stages;
                result->program = std::move(program);
            }
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
            const auto fragmentId = ShaderAuthorInputId(*database, fragment->filePath);
            const auto vertexId = ShaderAuthorInputId(*database, vertex->filePath);
            std::vector<ShaderCacheInput> inputs{
                MakeShaderCacheInput(fragmentId, fragment->authorSourceText, fragment->filePath),
                MakeShaderCacheInput(vertexId, vertex->authorSourceText, vertex->filePath)};
            const auto surfaces = ShaderProjectSurfaceReader(*database);
            const auto selection =
                ShaderAuthorProgramSelection(database->GetGuidFromPath(fragment->filePath),
                                             database->GetGuidFromPath(vertex->filePath), RuntimeShaderAuthorTarget());
            if (auto record = FindShaderAuthorProgram(*database, selection, surfaces, inputs)) {
                result->program = std::make_shared<const ShaderProgramArtifact>(CreateShaderAuthorMaterial(
                    stages.fragmentKey, record->compilation, record->schema, stages.vertexKey));
                return result;
            }
            auto &compiler = SlangCompiler::Get();
            const auto prepared = compiler.PrepareAuthorSource(
                {"material_" + stages.fragmentKey, fragment->filePath, fragment->authorSourceText},
                TrackAuthorReads(*database, inputs), ShaderAuthorPhase::Program,
                {"vertex_" + stages.vertexKey, vertex->filePath, vertex->authorSourceText}, surfaces);
            auto record =
                CompileAuthorRecord(prepared, RuntimeShaderAuthorTarget(), ShaderAuthorDependencies(*database, inputs));
            result->program = std::make_shared<const ShaderProgramArtifact>(
                CreateShaderAuthorMaterial(stages.fragmentKey, record.compilation, record.schema, stages.vertexKey));
            StoreShaderAuthorProgram(ProjectCacheDirectory(database->GetProjectRoot()), selection, inputs, record);
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
}

// =============================================================================
// Internal helper — compile a shader file into a ShaderAsset
// =============================================================================

std::shared_ptr<ShaderAsset> ShaderLoader::CompileAuthorAsset(const ShaderSourceModule &source,
                                                              const ShaderAuthorSource &prepared,
                                                              const std::string &guid,
                                                              std::vector<std::string> dependencies)
{
    return RestoreAuthorAsset(source, guid,
                              CompileAuthorRecord(prepared, RuntimeShaderAuthorTarget(), std::move(dependencies)));
}

std::shared_ptr<ShaderAsset> ShaderLoader::RestoreAuthorAsset(const ShaderSourceModule &source, const std::string &guid,
                                                              ShaderAuthorProgramRecord record)
{
    const auto *kind = asset_formats::FindShaderSourceKind(source.path);
    const auto schema = nlohmann::json::parse(record.schema);
    const auto &info = schema.at("info");
    auto asset = std::make_shared<ShaderAsset>();
    asset->shaderId = info.at("shader_id");
    asset->shaderType = kind->stage;
    asset->filePath = source.path;
    asset->authoringSchema = std::move(record.schema);
    asset->authorSourceText = source.source;
    asset->renderMeta = ReadShaderAuthorRenderMeta(info);
    if (schema.at("domain") == "fullscreen")
        asset->programDomain = ShaderProgramDomain::Fullscreen;
    else if (schema.at("domain") == "screen_ui")
        asset->programDomain = ShaderProgramDomain::ScreenUI;
    else if (schema.at("domain") == "world_ui")
        asset->programDomain = ShaderProgramDomain::WorldUI;
    else if (schema.at("domain") == "particle")
        asset->programDomain = ShaderProgramDomain::ParticleSprite;
    else if (schema.at("domain") == "mesh" || schema.at("domain") == "vertex")
        asset->programDomain = ShaderProgramDomain::Mesh;
    asset->sourceDependencies = std::move(record.dependencies);
    asset->program = std::move(record.compilation);
    if (schema.at("domain") == "mesh" || schema.at("domain") == "particle" || schema.at("domain") == "screen_ui" ||
        schema.at("domain") == "world_ui") {
        asset->materialProgram = std::make_shared<const ShaderProgramArtifact>(CreateShaderAuthorMaterial(
            guid, asset->program, asset->authoringSchema,
            asset->programDomain == ShaderProgramDomain::Mesh ? kStandardVertexShaderId : ""));
    }
    return asset;
}

static std::shared_ptr<ShaderAsset> CompileShaderAsset(const std::string &filePath, const std::string &guid,
                                                       AssetDatabase *database)
{
    if (!asset_formats::FindShaderSourceKind(filePath))
        throw std::invalid_argument("Shader assets use typed Python GPU source: " + filePath);
    const auto source = ReadAuthorSource(*database, filePath, "asset_" + guid);
    std::vector<ShaderCacheInput> inputs{
        MakeShaderCacheInput(ShaderAuthorInputId(*database, filePath), source.source, filePath)};
    const auto surfaces = ShaderProjectSurfaceReader(*database);
    const auto selection = ShaderAuthorProgramSelection(guid, {}, RuntimeShaderAuthorTarget());
    if (auto cached = FindShaderAuthorProgram(*database, selection, surfaces, inputs))
        return ShaderLoader::RestoreAuthorAsset(source, guid, std::move(*cached));
    const auto prepared = SlangCompiler::Get().PrepareAuthorSource(source, TrackAuthorReads(*database, inputs),
                                                                   ShaderAuthorPhase::Program, {}, surfaces);
    auto asset = ShaderLoader::CompileAuthorAsset(source, prepared, guid, ShaderAuthorDependencies(*database, inputs));
    StoreShaderAuthorProgram(ProjectCacheDirectory(database->GetProjectRoot()), selection, inputs,
                             {asset->authoringSchema, asset->program, asset->sourceDependencies});
    return asset;
}

void ShaderLoader::CookAuthorProgram(const AssetDatabase &database, const std::string &authorGuid,
                                     const std::string &vertexGuid, ShaderCodeTarget target,
                                     const std::string &buildCache, const std::string &library)
{
    const auto surfaces = ShaderProjectSurfaceReader(database);
    const auto selection = ShaderAuthorProgramSelection(authorGuid, vertexGuid, target);
    std::vector<ShaderCacheInput> inputs;
    auto record = FindShaderAuthorProgram(database, selection, surfaces, {}, &inputs, buildCache);
    if (!record) {
        const auto source = ReadAuthorSource(database, database.GetPathFromGuid(authorGuid),
                                             (vertexGuid.empty() ? "asset_" : "material_") + authorGuid);
        inputs.push_back(MakeShaderCacheInput(ShaderAuthorInputId(database, source.path), source.source, source.path));
        ShaderSourceModule vertex;
        if (!vertexGuid.empty()) {
            vertex = ReadAuthorSource(database, database.GetPathFromGuid(vertexGuid), "vertex_" + vertexGuid);
            inputs.push_back(
                MakeShaderCacheInput(ShaderAuthorInputId(database, vertex.path), vertex.source, vertex.path));
        }
        const auto prepared = SlangCompiler::Get().PrepareAuthorSource(source, TrackAuthorReads(database, inputs),
                                                                       ShaderAuthorPhase::Program, vertex, surfaces);
        record = CompileAuthorRecord(prepared, target, ShaderAuthorDependencies(database, inputs));
        // Use the same publication contract as a real material before shipping a cache entry.
        if (vertexGuid.empty())
            (void)RestoreAuthorAsset(source, authorGuid, *record);
        else
            (void)CreateShaderAuthorMaterial(authorGuid, record->compilation, record->schema, vertexGuid);
        StoreShaderAuthorProgram(buildCache, selection, inputs, *record);
    }
    if (!StoreShaderAuthorProgram(library, selection, inputs, *record))
        throw std::runtime_error("Could not package shader program for " + authorGuid + " " + vertexGuid);
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
