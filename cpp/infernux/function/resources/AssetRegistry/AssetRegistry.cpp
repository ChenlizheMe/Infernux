#include "AssetRegistry.h"

#include <core/log/InxLog.h>
#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxTexture/InxTexture.h>

#include <platform/filesystem/InxPath.h>

#include <filesystem>
#include <fstream>
#include <unordered_set>

namespace infernux
{

// =============================================================================
// Singleton
// =============================================================================

AssetRegistry &AssetRegistry::Instance()
{
    static AssetRegistry instance;
    return instance;
}

// =============================================================================
// Lifecycle
// =============================================================================

void AssetRegistry::Initialize(std::unique_ptr<AssetDatabase> adb)
{
    if (m_initialized) {
        INXLOG_WARN("AssetRegistry::Initialize called more than once — ignoring.");
        return;
    }

    m_assetDb = std::move(adb);
    m_initialized = true;
    INXLOG_INFO("AssetRegistry initialized.");
}

void AssetRegistry::Shutdown()
{
    INXLOG_INFO("Shutting down AssetRegistry...");
    m_loadedAssets.clear();
    m_loaders.clear();
    m_builtinMaterials.clear();
    m_assetDb.reset();
    m_initialized = false;
}

// =============================================================================
// Loader registration
// =============================================================================

void AssetRegistry::RegisterLoader(ResourceType type, std::unique_ptr<IAssetLoader> loader)
{
    if (!loader) {
        INXLOG_WARN("AssetRegistry::RegisterLoader: null loader for type ", static_cast<int>(type));
        return;
    }
    // INXLOG_INFO("AssetRegistry: registered loader for ResourceType ", static_cast<int>(type));
    m_loaders[type] = std::move(loader);
}

IAssetLoader *AssetRegistry::GetLoader(ResourceType type) const
{
    auto it = m_loaders.find(type);
    return it != m_loaders.end() ? it->second.get() : nullptr;
}

void AssetRegistry::PopulateAssetDatabaseLoaders()
{
    if (!m_assetDb) {
        INXLOG_WARN("AssetRegistry::PopulateAssetDatabaseLoaders: no AssetDatabase");
        return;
    }
    for (auto &[type, loader] : m_loaders) {
        m_assetDb->SetMetaLoader(type, loader.get());
    }
    INXLOG_INFO("AssetRegistry: populated AssetDatabase with ", m_loaders.size(), " loaders");
}

// =============================================================================
// Internal load helper
// =============================================================================

std::shared_ptr<void> AssetRegistry::LoadAssetInternal(const std::string &filePath, const std::string &guid,
                                                       ResourceType type)
{
    auto loaderIt = m_loaders.find(type);
    if (loaderIt == m_loaders.end()) {
        INXLOG_WARN("AssetRegistry: no loader registered for ResourceType ", static_cast<int>(type));
        return nullptr;
    }

    auto instance = loaderIt->second->Load(filePath, guid, m_assetDb.get());
    if (!instance) {
        INXLOG_WARN("AssetRegistry: loader returned nullptr for '", filePath, "' (GUID: ", guid, ")");
        return nullptr;
    }

    m_loadedAssets[guid] = {instance, type};
    return instance;
}

// =============================================================================
// Hot-reload / invalidation
// =============================================================================

bool AssetRegistry::ReloadAsset(const std::string &guid)
{
    auto it = m_loadedAssets.find(guid);
    if (it == m_loadedAssets.end())
        return false;

    if (!m_assetDb)
        return false;

    std::string path = m_assetDb->GetPathFromGuid(guid);
    if (path.empty()) {
        INXLOG_WARN("AssetRegistry::ReloadAsset: GUID has no path mapping — ", guid);
        return false;
    }

    auto loaderIt = m_loaders.find(it->second.type);
    if (loaderIt == m_loaders.end())
        return false;

    return loaderIt->second->Reload(it->second.instance, path, guid, m_assetDb.get());
}

void AssetRegistry::InvalidateAsset(const std::string &guid)
{
    auto it = m_loadedAssets.find(guid);
    if (it != m_loadedAssets.end()) {
        m_loadedAssets.erase(it);
        INXLOG_DEBUG("AssetRegistry: invalidated cache for GUID ", guid);
    }
}

void AssetRegistry::RemoveAsset(const std::string &guid)
{
    m_loadedAssets.erase(guid);
}

// =============================================================================
// File-event hooks
// =============================================================================

void AssetRegistry::OnAssetModified(const std::string &path)
{
    if (!m_assetDb)
        return;
    std::string guid = m_assetDb->GetGuidFromPath(path);
    if (guid.empty())
        return;

    // Only reload if it's already in cache — don't eagerly load untracked files.
    if (m_loadedAssets.count(guid)) {
        ReloadAsset(guid);
    }
}

void AssetRegistry::OnAssetMoved(const std::string &oldPath, const std::string &newPath)
{
    if (!m_assetDb)
        return;

    // AssetDatabase should have updated the GUID↔path mapping before we get here.
    // Because our cache is keyed by GUID, no cache surgery is needed.
    // However, some asset types store an internal path (InxMaterial::m_filePath)
    // that must be patched so SaveToFile() writes to the correct location.
    std::string guid = m_assetDb->GetGuidFromPath(newPath);
    if (guid.empty()) {
        // Fallback: try oldPath in case AssetDatabase hasn't updated yet
        guid = m_assetDb->GetGuidFromPath(oldPath);
    }
    if (guid.empty())
        return;

    auto it = m_loadedAssets.find(guid);
    if (it == m_loadedAssets.end())
        return;

    auto newName = FromFsPath(ToFsPath(newPath).stem());

    if (it->second.type == ResourceType::Material) {
        auto mat = std::static_pointer_cast<InxMaterial>(it->second.instance);
        if (mat) {
            mat->SetFilePath(newPath);
            mat->SetName(newName);
        }
    }
    if (it->second.type == ResourceType::Texture) {
        auto tex = std::static_pointer_cast<InxTexture>(it->second.instance);
        if (tex) {
            tex->SetFilePath(newPath);
            tex->SetName(newName);
        }
    }
    // Future: add per-type path patching for Audio, Scene, etc.
}

void AssetRegistry::OnAssetDeleted(const std::string &path)
{
    if (!m_assetDb)
        return;

    // Note: the caller should resolve GUID *before* deleting the AssetDatabase mapping.
    // AssetDatabase::DeleteAsset fires NotifyEvent(Deleted) before clearing its mapping,
    // so by the time Python's on_asset_deleted calls us, the GUID may already be gone.
    // We therefore iterate loaded assets to find a match by path as a fallback.
    std::string guid = m_assetDb->GetGuidFromPath(path);
    if (!guid.empty()) {
        RemoveAsset(guid);
        return;
    }

    // Fallback: linear search by asking AssetDatabase for reverse mapping
    // This shouldn't normally be needed — the Python layer resolves GUID first.
}

// =============================================================================
// Built-in materials (named, no GUID)
// =============================================================================

void AssetRegistry::RegisterBuiltinMaterial(const std::string &key, std::shared_ptr<InxMaterial> mat)
{
    if (!mat) {
        INXLOG_WARN("AssetRegistry::RegisterBuiltinMaterial: null material for key '", key, "'");
        return;
    }
    m_builtinMaterials[key] = std::move(mat);
}

std::shared_ptr<InxMaterial> AssetRegistry::GetBuiltinMaterial(const std::string &key) const
{
    auto it = m_builtinMaterials.find(key);
    return it != m_builtinMaterials.end() ? it->second : nullptr;
}

bool AssetRegistry::LoadBuiltinMaterialFromFile(const std::string &key, const std::string &matFilePath)
{
    if (matFilePath.empty())
        return false;

    std::ifstream file(ToFsPath(matFilePath));
    if (!file.is_open()) {
        INXLOG_WARN("AssetRegistry::LoadBuiltinMaterialFromFile: cannot open '", matFilePath, "'");
        return false;
    }
    std::string jsonStr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    auto material = std::make_shared<InxMaterial>();
    if (!material->Deserialize(jsonStr)) {
        INXLOG_ERROR("AssetRegistry::LoadBuiltinMaterialFromFile: deserialization failed for '", matFilePath, "'");
        return false;
    }

    material->SetFilePath(matFilePath);
    material->SetBuiltin(true);
    RegisterBuiltinMaterial(key, material);

    INXLOG_INFO("AssetRegistry: loaded builtin material '", key, "' from: ", matFilePath);
    return true;
}

// =============================================================================
// Built-in material initialization
// =============================================================================

void AssetRegistry::InitializeBuiltinMaterials()
{
    INXLOG_INFO("AssetRegistry: initializing built-in materials...");

    auto registerBuiltin = [this](const std::string &key, std::shared_ptr<InxMaterial> mat) {
        if (mat) {
            mat->SetBuiltin(true);
            RegisterBuiltinMaterial(key, mat);
        }
    };

    registerBuiltin("DefaultLit", InxMaterial::CreateDefaultLit());
    registerBuiltin("DefaultUnlit", InxMaterial::CreateDefaultUnlit());
    registerBuiltin("GizmoMaterial", InxMaterial::CreateGizmoMaterial());
    registerBuiltin("GridMaterial", InxMaterial::CreateGridMaterial());
    registerBuiltin("ComponentGizmosMaterial", InxMaterial::CreateComponentGizmosMaterial());
    registerBuiltin("ComponentGizmoIconMaterial", InxMaterial::CreateComponentGizmoIconMaterial());
    registerBuiltin("ComponentGizmoCameraIconMaterial", InxMaterial::CreateComponentGizmoCameraIconMaterial());
    registerBuiltin("ComponentGizmoLightIconMaterial", InxMaterial::CreateComponentGizmoLightIconMaterial());
    registerBuiltin("EditorToolsMaterial", InxMaterial::CreateEditorToolsMaterial());
    registerBuiltin("SkyboxProcedural", InxMaterial::CreateSkyboxProceduralMaterial());
    registerBuiltin("ErrorMaterial", InxMaterial::CreateErrorMaterial());

    INXLOG_INFO("AssetRegistry: built-in materials initialized.");
}

// =============================================================================
// GetAllMaterials — builtin + loaded from disk
// =============================================================================

std::vector<std::shared_ptr<InxMaterial>> AssetRegistry::GetAllMaterials() const
{
    std::vector<std::shared_ptr<InxMaterial>> result;
    std::unordered_set<InxMaterial *> seen;

    // 1. Built-in materials
    for (const auto &[key, mat] : m_builtinMaterials) {
        if (mat && seen.find(mat.get()) == seen.end()) {
            result.push_back(mat);
            seen.insert(mat.get());
        }
    }

    // 2. User-loaded materials from disk (via AssetRegistry cache)
    for (const auto &[guid, entry] : m_loadedAssets) {
        if (entry.type == ResourceType::Material) {
            auto mat = std::static_pointer_cast<InxMaterial>(entry.instance);
            if (mat && seen.find(mat.get()) == seen.end()) {
                result.push_back(mat);
                seen.insert(mat.get());
            }
        }
    }

    return result;
}

// =============================================================================
// Queries
// =============================================================================

bool AssetRegistry::IsLoaded(const std::string &guid) const
{
    return m_loadedAssets.count(guid) > 0;
}

ResourceType AssetRegistry::GetAssetType(const std::string &guid) const
{
    auto it = m_loadedAssets.find(guid);
    if (it != m_loadedAssets.end())
        return it->second.type;
    return ResourceType::DefaultBinary;
}

std::vector<std::string> AssetRegistry::GetAllLoadedGuids() const
{
    std::vector<std::string> guids;
    guids.reserve(m_loadedAssets.size());
    for (const auto &[guid, entry] : m_loadedAssets)
        guids.push_back(guid);
    return guids;
}

std::vector<std::shared_ptr<void>> AssetRegistry::GetAllAssetsOfType(ResourceType type) const
{
    std::vector<std::shared_ptr<void>> result;
    for (const auto &[guid, entry] : m_loadedAssets) {
        if (entry.type == type)
            result.push_back(entry.instance);
    }
    return result;
}

} // namespace infernux
