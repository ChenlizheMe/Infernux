#include "ShaderLoader.h"

#include <core/log/InfLog.h>
#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/InfFileLoader/InfShaderLoader.hpp>
#include <function/resources/InfResource/InfResourceMeta.h>
#include <function/resources/ShaderAsset/ShaderAsset.h>

#include <platform/filesystem/InfPath.h>

#include <filesystem>

namespace infengine
{

// =============================================================================
// Internal helper — compile a shader file into a ShaderAsset
// =============================================================================

static std::shared_ptr<ShaderAsset> CompileShaderAsset(const std::string &filePath, const std::string &guid,
                                                       AssetDatabase *adb)
{
    if (filePath.empty() || guid.empty()) {
        INFLOG_WARN("ShaderLoader: empty filePath or guid");
        return nullptr;
    }

    if (!adb) {
        INFLOG_ERROR("ShaderLoader: no AssetDatabase");
        return nullptr;
    }

    // Read shader source
    std::vector<char> content;
    if (!adb->ReadFile(filePath, content)) {
        INFLOG_ERROR("ShaderLoader: failed to read '", filePath, "'");
        return nullptr;
    }
    if (content.empty()) {
        INFLOG_ERROR("ShaderLoader: empty file '", filePath, "'");
        return nullptr;
    }
    // Ensure null-terminated
    if (content.back() != '\0')
        content.push_back('\0');

    // Determine shader type from extension
    std::filesystem::path fsPath = ToFsPath(filePath);
    std::string ext = fsPath.extension().string();

    // Read metadata for shader_id
    const InfResourceMeta *meta = adb->GetMetaByGuid(guid);
    std::string shaderId;
    if (meta && meta->HasKey("shader_id")) {
        shaderId = meta->GetDataAs<std::string>("shader_id");
    }
    if (shaderId.empty()) {
        shaderId = FromFsPath(fsPath.stem());
    }

    // Use InfShaderLoader to compile (it manages glslang, preprocessing, etc.)
    InfShaderLoader compiler(true, false, false, false, false, false, false, false, false, false);

    // RegisterResource already created the .meta — use it for Load()
    InfResourceMeta loadMeta;
    if (meta) {
        loadMeta = *meta;
    } else {
        // Build minimal meta for compilation
        loadMeta.AddMetadata("file_path", filePath);
        loadMeta.AddMetadata("type", ext == ".vert" ? std::string("vertex") : std::string("fragment"));
        loadMeta.AddMetadata("shader_id", shaderId);
    }

    InfShaderLoader::s_lastCompileError.clear();

    auto compiledPtr = compiler.Compile(content.data(), content.size(), loadMeta);
    if (!compiledPtr || compiledPtr->empty()) {
        INFLOG_ERROR("ShaderLoader: compilation failed for '", filePath, "'");
        return nullptr;
    }

    // Build ShaderAsset
    auto asset = std::make_shared<ShaderAsset>();
    asset->shaderId = shaderId;
    asset->shaderType = (ext == ".vert") ? "vertex" : "fragment";
    asset->filePath = filePath;
    asset->spirvForward = std::move(*compiledPtr);

    // Extract variant SPIR-V from InfShaderLoader's static caches
    // Use the meta's file_path as cache key (matches InfShaderLoader::CompileVariant)
    std::string cacheKey = filePath;
    if (meta && meta->HasKey("file_path")) {
        cacheKey = meta->GetDataAs<std::string>("file_path");
    }

    if (ext == ".vert") {
        auto it = InfShaderLoader::s_shadowVertexVariantCache.find(cacheKey);
        if (it != InfShaderLoader::s_shadowVertexVariantCache.end() && !it->second.empty()) {
            asset->spirvShadowVertex = std::move(it->second);
            InfShaderLoader::s_shadowVertexVariantCache.erase(it);
        }
    }

    if (ext == ".frag") {
        auto sit = InfShaderLoader::s_shadowVariantCache.find(cacheKey);
        if (sit != InfShaderLoader::s_shadowVariantCache.end() && !sit->second.empty()) {
            asset->spirvShadow = std::move(sit->second);
            InfShaderLoader::s_shadowVariantCache.erase(sit);
        }

        auto git = InfShaderLoader::s_gbufferVariantCache.find(cacheKey);
        if (git != InfShaderLoader::s_gbufferVariantCache.end() && !git->second.empty()) {
            asset->spirvGBuffer = std::move(git->second);
            InfShaderLoader::s_gbufferVariantCache.erase(git);
        }

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

    INFLOG_INFO("ShaderLoader: compiled '", shaderId, "' (", asset->shaderType, ") from '", filePath, "'");
    return asset;
}

// =============================================================================
// Load
// =============================================================================

std::shared_ptr<void> ShaderLoader::Load(const std::string &filePath, const std::string &guid, AssetDatabase *adb)
{
    return CompileShaderAsset(filePath, guid, adb);
}

// =============================================================================
// Reload — recompile and replace in-place
// =============================================================================

bool ShaderLoader::Reload(std::shared_ptr<void> existing, const std::string &filePath, const std::string &guid,
                          AssetDatabase *adb)
{
    auto oldAsset = std::static_pointer_cast<ShaderAsset>(existing);
    if (!oldAsset) {
        INFLOG_WARN("ShaderLoader::Reload: null existing instance");
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
// ScanDependencies — shaders have no outgoing asset dependencies
// =============================================================================

std::set<std::string> ShaderLoader::ScanDependencies(const std::string & /*filePath*/, AssetDatabase * /*adb*/)
{
    return {};
}

// =============================================================================
// LoadMeta / CreateMeta — delegate to InfShaderLoader (the shader compiler)
// =============================================================================

bool ShaderLoader::LoadMeta(const char *content, const std::string &filePath, InfResourceMeta &metaData)
{
    InfShaderLoader compiler(true, false, false, false, false, false, false, false, false, false);
    return compiler.LoadMeta(content, filePath, metaData);
}

void ShaderLoader::CreateMeta(const char *content, size_t contentSize, const std::string &filePath,
                              InfResourceMeta &metaData)
{
    InfShaderLoader compiler(true, false, false, false, false, false, false, false, false, false);
    compiler.CreateMeta(content, contentSize, filePath, metaData);
}

} // namespace infengine
