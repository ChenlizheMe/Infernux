#include "MaterialLoader.h"

#include <core/log/InxLog.h>
#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/AssetDependencyGraph.h>
#include <function/resources/InxMaterial/InxMaterial.h>

#include <platform/filesystem/InxPath.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unordered_set>

namespace infernux
{

namespace
{

bool IsBuiltinTextureToken(const std::string &value)
{
    return value == "white" || value == "black" || value == "normal";
}

ShaderAssetReference EnrichShaderReference(ShaderAssetReference reference, AssetDatabase *database,
                                         const char *expectedStage)
{
    if (!database)
        return reference;

    // GUID is the only durable identity. pathHint is refreshed from the GUID
    // for display, but is never used to recover identity or locate an asset.
    // A GUID-less shader_id denotes a symbolic built-in shader program and
    // deliberately contributes no AssetDatabase dependency edge.
    const std::string resolvedPath = reference.guid.empty() ? std::string{} : database->GetPathFromGuid(reference.guid);

    if (!resolvedPath.empty()) {
        reference.pathHint = resolvedPath;

        const auto meta = database->GetMetaByPath(resolvedPath);
        if (!meta || meta->GetResourceType() != ResourceType::Shader)
            throw std::runtime_error("Material shader GUID resolves to a non-shader asset: " + resolvedPath);
        if (!meta->HasKey("type") || meta->GetDataAs<std::string>("type") != expectedStage)
            throw std::runtime_error("Material shader GUID does not identify the required " +
                                     std::string(expectedStage) + " stage: " + resolvedPath);
        if (!meta->HasKey("shader_id") || meta->GetDataAs<std::string>("shader_id").empty())
            throw std::runtime_error("Material shader GUID resolves to metadata without ShaderInfo Name: " +
                                     resolvedPath);
        // A saved shader_id is a derived label and can predate a source rename.
        // Never reject or rebind that durable reference because its label changed.
        reference.shaderId = meta->GetDataAs<std::string>("shader_id");
    }
    return reference;
}

} // namespace

bool MaterialLoader::PrepareDocument(InxMaterial &staged, const std::string &filePath, AssetDatabase *adb)
{
    std::ifstream file(ToFsPath(filePath));
    if (!file.is_open()) {
        INXLOG_WARN("MaterialLoader: cannot open '", filePath, "'");
        return false;
    }
    try {
        if (!staged.ApplyDocument(nlohmann::json::parse(file)))
            return false;
        // Resolve on the unpublished candidate. Public shader setters would
        // notify routing and reset authorship before the complete edit is valid.
        staged.m_vertexShader = EnrichShaderReference(staged.m_vertexShader, adb, "vertex");
        staged.m_fragmentShader = EnrichShaderReference(staged.m_fragmentShader, adb, "fragment");
        return true;
    } catch (const std::exception &error) {
        INXLOG_ERROR("MaterialLoader: invalid document '", filePath, "': ", error.what());
        return false;
    }
}

// =============================================================================
// Load — create a brand-new InxMaterial from a .mat file
// =============================================================================

RuntimeAssetPayload MaterialLoader::Load(const std::string &filePath, const std::string &guid, AssetDatabase *adb)
{
    if (filePath.empty() || guid.empty()) {
        INXLOG_WARN("MaterialLoader::Load: empty filePath or guid");
        return nullptr;
    }

    InxMaterial staged;
    if (!PrepareDocument(staged, filePath, adb))
        return nullptr;
    auto material = std::make_shared<InxMaterial>();
    // Disk identity is known before publication, including on a worker. It must
    // never be enrolled in the owner-thread-only runtime material registry.
    material->SetGuid(guid);
    material->PublishDocument(std::move(staged));

    // Identity — authoritative source is .meta / AssetDatabase, NOT JSON
    material->SetFilePath(filePath);
    material->SetName(FromFsPath(ToFsPath(filePath).stem()));

    // Dependency graph edges (textures, shaders)
    RegisterDependencies(guid, *material, adb);

    // INXLOG_INFO("MaterialLoader: loaded '", material->GetName(), "' (GUID: ", guid, ")");
    return material;
}

// =============================================================================
// Reload — hot-refresh into an existing instance (pointer identity preserved)
// =============================================================================

bool MaterialLoader::Reload(const RuntimeAssetPayload &existing, const std::string &filePath, const std::string &guid,
                            AssetDatabase *adb)
{
    auto mat = existing.Get<InxMaterial>();
    if (!mat) {
        INXLOG_WARN("MaterialLoader::Reload: null existing instance");
        return false;
    }

    InxMaterial staged(*mat);
    if (!PrepareDocument(staged, filePath, adb))
        return false;
    staged.m_name = mat->GetName();
    mat->PublishDocument(std::move(staged));

    // Re-wire dependency graph (texture/shader deps may have changed)
    RegisterDependencies(mat->GetGuid(), *mat, adb);

    return true;
}

// =============================================================================
// ScanDependencies — enumerate outgoing GUIDs for the dependency graph
// =============================================================================

size_t MaterialLoader::EstimateRuntimeBytes(const RuntimeAssetPayload &payload) const
{
    const auto material = payload.Get<InxMaterial>();
    if (!material)
        throw std::invalid_argument("MaterialLoader cannot estimate an empty runtime payload");
    return material->GetRuntimeMemoryBytes();
}

std::set<std::string> MaterialLoader::ScanDependencies(const std::string &filePath, AssetDatabase *adb)
{
    std::set<std::string> deps;

    std::ifstream file = OpenInputFile(filePath);
    if (!file.is_open())
        return deps;

    std::string jsonStr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    // Temporary material just for parsing — lightweight, no GPU resources
    InxMaterial tmp;
    if (!tmp.Deserialize(jsonStr))
        return deps;

    // Texture GUIDs
    for (const auto &[propName, prop] : tmp.GetAllProperties()) {
        if (prop.type != MaterialPropertyType::Texture2D)
            continue;
        const auto *val = std::get_if<std::string>(&prop.value);
        if (val && !val->empty() && !IsBuiltinTextureToken(*val))
            deps.insert(*val);
    }

    // Asset dependencies are GUID-only. shader_id may identify a symbolic
    // built-in program, while path_hint is never an identity fallback.
    (void)adb;
    const auto addShaderDep = [&](const ShaderAssetReference &reference) {
        if (!reference.guid.empty())
            deps.insert(reference.guid);
    };
    addShaderDep(tmp.GetVertShaderReference());
    addShaderDep(tmp.GetFragShaderReference());

    return deps;
}

// =============================================================================
// RegisterDependencies — wire up AssetDependencyGraph edges
// =============================================================================
// CreateMeta — material-specific .meta creation
// =============================================================================

void MaterialLoader::CreateMeta(const char *content, size_t contentSize, const std::string &filePath,
                                InxResourceMeta &metaData) const
{
    if (!content) {
        INXLOG_ERROR("Invalid material content for metadata creation");
        return;
    }

    metaData.Init(content, contentSize, filePath, ResourceType::Material);

    // Use filename stem as material name — authoritative name is set by
    // MaterialLoader::Load() at runtime, so parsing JSON here is unnecessary.
    std::filesystem::path path = ToFsPath(filePath);
    metaData.AddMetadata("material_name", FromFsPath(path.stem()));
}

// =============================================================================

void MaterialLoader::RegisterDependencies(const std::string &materialGuid, const InxMaterial &mat, AssetDatabase *adb)
{
    if (materialGuid.empty())
        return;

    std::unordered_set<std::string> dependencies;

    // Texture property GUIDs
    for (const auto &[propName, prop] : mat.GetAllProperties()) {
        if (prop.type != MaterialPropertyType::Texture2D)
            continue;
        const auto *val = std::get_if<std::string>(&prop.value);
        if (val && !val->empty() && !IsBuiltinTextureToken(*val))
            dependencies.insert(*val);
    }

    // Shader asset edges are GUID-only. Do not infer them from path_hint or
    // shader_id; symbolic built-in programs are not AssetDatabase assets.
    (void)adb;
    const auto addShaderDep = [&](const ShaderAssetReference &reference) {
        if (!reference.guid.empty())
            dependencies.insert(reference.guid);
    };
    addShaderDep(mat.GetVertShaderReference());
    addShaderDep(mat.GetFragShaderReference());
    AssetDependencyGraph::Instance().SetAssetDependencies(materialGuid, dependencies);
}

} // namespace infernux
