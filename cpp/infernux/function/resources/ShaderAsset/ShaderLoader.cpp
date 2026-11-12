#include "ShaderLoader.h"

#include <core/log/InxLog.h>
#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/AssetDependencyGraph.h>
#include <function/resources/AssetFormatRegistry.h>
#include <function/resources/InxFileLoader/InxShaderLoader.hpp>
#include <function/resources/InxResource/InxResourceMeta.h>
#include <function/resources/ShaderAsset/ShaderAsset.h>
#include <function/resources/ShaderAsset/ShaderAuthorMaterial.h>
#include <function/resources/ShaderAsset/ShaderAuthorMetadata.h>
#include <function/resources/ShaderAsset/SlangCompiler.h>

#include <platform/filesystem/InxPath.h>

#include <filesystem>
#include <nlohmann/json.hpp>
namespace infernux
{
void ShaderLoader::OnRelocated(const RuntimeAssetPayload &payload, const std::string &filePath)
{
    const auto shader = payload.Get<ShaderAsset>();
    shader->filePath = filePath;
    shader->descriptor.filePath = filePath;
}

// =============================================================================
// Internal helper — compile a shader file into a ShaderAsset
// =============================================================================

static std::shared_ptr<ShaderAsset> CompileShaderAsset(const std::string &filePath, const std::string &guid,
                                                       AssetDatabase *adb)
{
    if (const auto *kind = asset_formats::FindShaderSourceKind(filePath)) {
        std::vector<char> bytes;
        if (!adb->ReadFile(filePath, bytes))
            throw std::runtime_error("GPU author source could not be read: " + filePath);
        const std::string source(bytes.begin(), bytes.end());
        auto &compiler = SlangCompiler::Get();
        const ShaderSourceModule module{"asset_" + guid, filePath, source};
        const auto prepared =
            compiler.PrepareAuthorSource(module, [adb](const std::string &path) -> std::optional<std::string> {
                std::vector<char> contents;
                if (!adb->ReadFile(path, contents))
                    return std::nullopt;
                return std::string(contents.begin(), contents.end());
            });
        auto asset = std::make_shared<ShaderAsset>();
        const auto meta = adb->GetMetaByGuid(guid);
        asset->shaderId = meta->GetDataAs<std::string>("shader_id");
        asset->shaderType = kind->stage;
        asset->filePath = filePath;
        asset->descriptor.shaderId = asset->shaderId;
        asset->descriptor.isFragmentShader = kind->stage == "fragment";
        asset->descriptor.isVertexShader = kind->stage == "vertex";
        asset->descriptor.filePath = filePath;
        asset->descriptor.fileExtension = kind->suffix;
        asset->authoringSchema = prepared.schema;
        const auto schema = nlohmann::json::parse(prepared.schema);
        const auto &info = schema.at("info");
        asset->descriptor.hidden = info.at("hidden").get<bool>();
        asset->renderMeta.cullMode = info.at("cull_mode");
        asset->renderMeta.depthWrite = info.at("depth_write").is_null()     ? std::string{}
                                       : info.at("depth_write").get<bool>() ? "on"
                                                                            : "off";
        asset->renderMeta.depthTest = info.at("depth_test");
        asset->renderMeta.blend = info.at("blend");
        asset->renderMeta.queue = info.at("render_queue");
        asset->renderMeta.passTag = info.at("pass_tag");
        asset->renderMeta.stencil = info.at("stencil");
        asset->renderMeta.alphaClip = info.at("alpha_clip").is_null() ? std::string{} : info.at("alpha_clip").dump();
        if (schema.at("domain") == "fullscreen")
            asset->programDomain = ShaderProgramDomain::Fullscreen;
        else if (schema.at("domain") == "mesh")
            asset->programDomain = ShaderProgramDomain::Mesh;
        for (const auto &library : prepared.libraries) {
            if (!asset_formats::FindShaderSourceKind(library.path))
                continue;
            auto dependency = adb->GetGuidFromPath(library.path);
            if (!dependency.empty())
                asset->sourceDependencies.push_back(std::move(dependency));
        }
        asset->program = std::make_shared<ShaderCompilation>(
            compiler.Compile(prepared.source, prepared.libraries, {ShaderCodeTarget::SpirV, ShaderCodeTarget::Wgsl},
                             false, prepared.sourceRanges));
        if (asset->programDomain == ShaderProgramDomain::Mesh)
            asset->materialProgram = std::make_shared<const ShaderProgramArtifact>(
                CreateShaderAuthorMaterial(guid, asset->program, prepared.schema));
        return asset;
    }

    // Share one compiler/cache lock with linked-program prewarming and editor
    // hot reload. Publication into AssetRegistry still occurs on the owner
    // thread through AssetLoadTicket.
    const InxShaderLoader::CompilationGuard compileGuard;
    if (filePath.empty() || guid.empty()) {
        INXLOG_WARN("ShaderLoader: empty filePath or guid");
        return nullptr;
    }

    if (!adb) {
        INXLOG_ERROR("ShaderLoader: no AssetDatabase");
        return nullptr;
    }

    // Read shader source
    std::vector<char> content;
    if (!adb->ReadFile(filePath, content)) {
        INXLOG_ERROR("ShaderLoader: failed to read '", filePath, "'");
        return nullptr;
    }
    if (content.empty()) {
        INXLOG_ERROR("ShaderLoader: empty file '", filePath, "'");
        return nullptr;
    }

    // Runtime shader metadata remains authoritative; the source extension is
    // a fallback for loose Editor files and packed Player GLSL alike.
    std::filesystem::path fsPath = ToFsPath(filePath);
    std::string ext = FromFsPath(fsPath.extension());

    // Read metadata for shader_id
    const auto meta = adb->GetMetaByGuid(guid);
    std::string shaderId;
    if (meta && meta->HasKey("shader_id")) {
        shaderId = meta->GetDataAs<std::string>("shader_id");
    }
    if (shaderId.empty()) {
        shaderId = FromFsPath(fsPath.stem());
    }
    std::string shaderType;
    if (meta && meta->HasKey("type"))
        shaderType = meta->GetDataAs<std::string>("type");
    if (shaderType != "vertex" && shaderType != "fragment") {
        if (ext == ".vert")
            shaderType = "vertex";
        else if (ext == ".frag")
            shaderType = "fragment";
    }

    if (shaderType != "vertex" && shaderType != "fragment") {
        INXLOG_ERROR("ShaderLoader: shader stage metadata is invalid for '", filePath, "'");
        return nullptr;
    }

    // Ensure null-terminated for the GLSL compiler.
    if (content.back() != '\0')
        content.push_back('\0');

    const std::string compilePath = InxShaderLoader::StageQualifiedVirtualPath(filePath, shaderType);

    // Use InxShaderLoader to compile (it manages glslang, preprocessing, etc.)
    InxShaderLoader compiler(true, false, false, false, false, false, false, false, false, false);

    // Asset import already created the .meta; use it for Load().
    InxResourceMeta loadMeta;
    if (meta) {
        loadMeta = *meta;
        loadMeta.UpdateFilePath(compilePath);
    } else {
        // Build minimal meta for compilation
        loadMeta.AddMetadata("file_path", InxResourceMeta::NormalizeFilePath(compilePath));
        loadMeta.AddMetadata("type", shaderType);
        loadMeta.AddMetadata("shader_id", shaderId);
    }

    InxShaderLoader::s_lastCompileError.clear();

    auto compiledPtr = compiler.Compile(content.data(), content.size(), loadMeta);
    if (!compiledPtr || compiledPtr->empty()) {
        if (!InxShaderLoader::AreSourceDiagnosticsCaptured())
            INXLOG_ERROR("ShaderLoader: compilation failed for '", filePath, "'");
        return nullptr;
    }

    // Build ShaderAsset
    auto asset = std::make_shared<ShaderAsset>();
    asset->shaderId = shaderId;
    asset->shaderType = shaderType;
    asset->filePath = filePath;
    asset->descriptor = compiler.ParseShaderSource(std::string(content.data(), content.size() - 1), compilePath);
    if (!asset->SetVariant(ShaderCompileTarget::Forward, std::move(*compiledPtr))) {
        INXLOG_ERROR("ShaderLoader: compiler returned invalid Forward SPIR-V for '", filePath, "'");
        return nullptr;
    }

    // Extract variant SPIR-V from InxShaderLoader's static caches
    // Use the meta's file_path as cache key (matches InxShaderLoader::CompileVariant)
    const std::string cacheKey = compilePath;

    auto variants = InxShaderLoader::TakeCompiledVariants(cacheKey);
    for (auto &[target, spirv] : variants)
        asset->SetVariant(target, std::move(spirv));

    if (shaderType == "fragment") {
        // Extract render-state annotations from meta
        if (meta) {
            if (meta->HasKey("shader_cull_mode"))
                asset->renderMeta.cullMode = meta->GetDataAs<std::string>("shader_cull_mode");
            if (meta->HasKey("shader_depth_write"))
                asset->renderMeta.depthWrite = meta->GetDataAs<std::string>("shader_depth_write");
            if (meta->HasKey("shader_depth_test"))
                asset->renderMeta.depthTest = meta->GetDataAs<std::string>("shader_depth_test");
            if (meta->HasKey("shader_blend"))
                asset->renderMeta.blend = meta->GetDataAs<std::string>("shader_blend");
            if (meta->HasKey("shader_queue"))
                asset->renderMeta.queue = meta->GetDataAs<int>("shader_queue");
            if (meta->HasKey("shader_pass_tag"))
                asset->renderMeta.passTag = meta->GetDataAs<std::string>("shader_pass_tag");
            if (meta->HasKey("shader_stencil"))
                asset->renderMeta.stencil = meta->GetDataAs<std::string>("shader_stencil");
            if (meta->HasKey("shader_alpha_test"))
                asset->renderMeta.alphaClip = meta->GetDataAs<std::string>("shader_alpha_test");
        }
    }

    INXLOG_INFO("ShaderLoader: compiled '", shaderId, "' (", asset->shaderType, ") from '", filePath, "'");
    return asset;
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
// CreateMeta — delegate to InxShaderLoader (the shader compiler)
// =============================================================================

void ShaderLoader::CreateMeta(AssetSource &source, InxResourceMeta &metaData) const
{
    if (asset_formats::FindShaderSourceKind(source.Path())) {
        CreateShaderAuthorMetadata(source, metaData);
        return;
    }
    InxShaderLoader compiler(true, false, false, false, false, false, false, false, false, false);
    compiler.CreateMeta(source, metaData);
}

} // namespace infernux
