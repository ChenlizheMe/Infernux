#include "AssetDatabase.h"

#include <function/resources/AssetDependencyGraph.h>
#include <function/resources/AssetImporter/ConcreteImporters.h>

#include <core/log/InxLog.h>
#include <platform/filesystem/InxPath.h>

#include <algorithm>
#include <fstream>

namespace infernux
{

AssetDatabase::AssetDatabase()
{
    // Loaders are populated later by AssetRegistry::PopulateAssetDatabaseLoaders()
    // after all IAssetLoader plug-ins have been registered.
    INXLOG_DEBUG("AssetDatabase created (loaders pending)");
}

void AssetDatabase::Initialize(const std::string &projectRoot)
{
    m_projectRoot = FromFsPath(ToFsPath(projectRoot));
    if (!m_projectRoot.empty() && m_projectRoot.back() == '/') {
        m_projectRoot.pop_back();
    }

    std::filesystem::path assetsPath = ToFsPath(m_projectRoot) / "Assets";
    if (std::filesystem::exists(assetsPath)) {
        m_assetsRoot = FromFsPath(assetsPath);
    } else {
        m_assetsRoot = m_projectRoot;
    }

    // Register built-in importers
    m_importerRegistry.Register(std::make_unique<TextureImporter>());
    m_importerRegistry.Register(std::make_unique<ShaderImporter>());
    {
        auto matImp = std::make_unique<MaterialImporter>();
        matImp->SetAssetDatabase(this);
        m_importerRegistry.Register(std::move(matImp));
    }
    m_importerRegistry.Register(std::make_unique<ScriptImporter>());
    m_importerRegistry.Register(std::make_unique<AudioImporter>());
    m_importerRegistry.Register(std::make_unique<ModelImporter>());

    INXLOG_DEBUG("AssetDatabase initialized. ProjectRoot=", m_projectRoot, ", AssetsRoot=", m_assetsRoot);
}

void AssetDatabase::AddScanRoot(const std::string &path)
{
    auto norm = FromFsPath(ToFsPath(path));
    for (const auto &existing : m_extraScanRoots) {
        if (existing == norm)
            return;
    }
    m_extraScanRoots.push_back(std::move(norm));
    INXLOG_DEBUG("AssetDatabase: added extra scan root: ", m_extraScanRoots.back());
}

void AssetDatabase::Refresh()
{
    m_guidToPath.clear();
    m_pathToGuid.clear();

    if (m_assetsRoot.empty()) {
        INXLOG_WARN("AssetDatabase.Refresh: assets root not set");
        return;
    }

    std::filesystem::path assetsRootPath = ToFsPath(m_assetsRoot);
    if (!std::filesystem::exists(assetsRootPath)) {
        INXLOG_WARN("AssetDatabase.Refresh: assets root does not exist: ", m_assetsRoot);
        return;
    }

    // ── Pass 1: Register all assets (GUID ↔ path) without running importers ──
    struct PendingAsset
    {
        std::string guid;
        std::string path;
    };
    std::vector<PendingAsset> pendingImports;

    // Collect all directories to scan: Assets root + extra roots (e.g. Library/Resources)
    std::vector<std::filesystem::path> scanRoots;
    scanRoots.push_back(assetsRootPath);
    for (const auto &extra : m_extraScanRoots) {
        std::filesystem::path ep = ToFsPath(extra);
        if (std::filesystem::exists(ep))
            scanRoots.push_back(ep);
    }

    for (const auto &rootPath : scanRoots) {
        for (const auto &entry : std::filesystem::recursive_directory_iterator(rootPath)) {
            if (!entry.is_regular_file())
                continue;

            const std::filesystem::path filePath = entry.path();

            // Remove orphaned .tmp files left by interrupted meta saves or OS copies
            if (filePath.extension() == ".tmp") {
                std::error_code ec;
                std::filesystem::remove(filePath, ec);
                if (!ec) {
                    INXLOG_DEBUG("AssetDatabase.Refresh: cleaned up orphaned temp file: ", FromFsPath(filePath));
                }
                continue;
            }

            if (IsMetaFile(filePath))
                continue;

            std::string pathStr = FromFsPath(filePath);
            ResourceType type = GetResourceTypeForPath(pathStr);
            if (type == ResourceType::Meta)
                continue;

            std::string guid = RegisterResource(pathStr, type);
            if (guid.empty())
                continue;

            UpdateMapping(guid, pathStr);
            pendingImports.push_back({guid, pathStr});
        }
    } // end scan roots

    // ── Pass 2: Run importers (ScanDependencies can now resolve all paths) ──
    for (const auto &asset : pendingImports) {
        RunImporter(asset.guid, asset.path, false);
    }

    INXLOG_INFO("AssetDatabase.Refresh completed. Total assets: ", m_guidToPath.size());
}

std::string AssetDatabase::ImportAsset(const std::string &path)
{
    std::filesystem::path fsPath = ToFsPath(path);
    if (!std::filesystem::exists(fsPath) || !std::filesystem::is_regular_file(fsPath)) {
        return "";
    }

    if (IsMetaFile(fsPath)) {
        return "";
    }

    ResourceType type = GetResourceTypeForPath(path);
    if (type == ResourceType::Meta) {
        return "";
    }

    std::string guid = RegisterResource(path, type);
    if (guid.empty()) {
        return "";
    }

    UpdateMapping(guid, path);
    RunImporter(guid, path, false);
    return guid;
}

bool AssetDatabase::DeleteAsset(const std::string &path)
{
    std::string guid = GetGuidFromPath(path);
    ResourceType type = GetResourceTypeForPath(path);

    // Notify dependents BEFORE removing from maps (they need to resolve guid→path)
    if (!guid.empty()) {
        AssetDependencyGraph::Instance().NotifyEvent(guid, type, AssetEvent::Deleted);
        AssetDependencyGraph::Instance().RemoveAsset(guid);
    }

    DeleteResource(path);

    if (!guid.empty()) {
        RemoveMappingByGuid(guid);
    } else {
        RemoveMappingByPath(path);
    }
    return true;
}

bool AssetDatabase::MoveAsset(const std::string &oldPath, const std::string &newPath)
{
    std::string guid = GetGuidFromPath(oldPath);
    if (guid.empty()) {
        // Try to recover guid from old .meta file
        const std::string metaPath = InxResourceMeta::GetMetaFilePath(oldPath);
        if (std::filesystem::exists(ToFsPath(metaPath))) {
            InxResourceMeta meta;
            if (meta.LoadFromFile(metaPath)) {
                guid = meta.GetGuid();
            }
        }
    }
    MoveResource(oldPath, newPath);

    if (!guid.empty()) {
        UpdateMapping(guid, newPath);
        RemoveMappingByPath(oldPath);
        // Notify dependents — GUID unchanged, but path changed
        ResourceType type = GetResourceTypeForPath(newPath);
        AssetDependencyGraph::Instance().NotifyEvent(guid, type, AssetEvent::Moved);
        return true;
    }

    // If GUID not found, attempt to re-import
    std::string newGuid = ImportAsset(newPath);
    return !newGuid.empty();
}

bool AssetDatabase::ContainsGuid(const std::string &guid) const
{
    return m_guidToPath.find(guid) != m_guidToPath.end();
}

bool AssetDatabase::ContainsPath(const std::string &path) const
{
    std::string norm = NormalizePath(path);
    return m_pathToGuid.find(norm) != m_pathToGuid.end();
}

std::string AssetDatabase::GetGuidFromPath(const std::string &path) const
{
    std::string norm = NormalizePath(path);
    auto it = m_pathToGuid.find(norm);
    if (it != m_pathToGuid.end()) {
        return it->second;
    }

    // Try to load from meta file directly
    const std::string metaPath = InxResourceMeta::GetMetaFilePath(path);
    if (std::filesystem::exists(ToFsPath(metaPath))) {
        InxResourceMeta meta;
        if (meta.LoadFromFile(metaPath)) {
            return meta.GetGuid();
        }
    }

    // Try meta cache lookup by path
    if (const InxResourceMeta *meta = GetMetaByPath(path)) {
        return meta->GetGuid();
    }

    return "";
}

std::string AssetDatabase::GetPathFromGuid(const std::string &guid) const
{
    auto it = m_guidToPath.find(guid);
    if (it != m_guidToPath.end()) {
        return it->second;
    }

    // Fallback: look up in meta cache
    if (const InxResourceMeta *meta = GetMetaByGuid(guid)) {
        if (meta->HasKey("file_path")) {
            return meta->GetDataAs<std::string>("file_path");
        }
    }

    return "";
}

const InxResourceMeta *AssetDatabase::GetMetaByGuid(const std::string &guid) const
{
    auto it = m_metas.find(guid);
    if (it != m_metas.end()) {
        return it->second.get();
    }
    return nullptr;
}

const InxResourceMeta *AssetDatabase::GetMetaByPath(const std::string &path) const
{
    for (const auto &[guid, meta] : m_metas) {
        if (meta->HasKey("file_path")) {
            std::string metaPath = meta->GetDataAs<std::string>("file_path");
            if (metaPath == path) {
                return meta.get();
            }
        }
    }
    return nullptr;
}

std::vector<std::string> AssetDatabase::GetAllGuids() const
{
    std::vector<std::string> result;
    result.reserve(m_guidToPath.size());
    for (const auto &pair : m_guidToPath) {
        result.push_back(pair.first);
    }
    return result;
}

bool AssetDatabase::IsAssetPath(const std::string &path) const
{
    if (m_assetsRoot.empty())
        return false;

    std::string norm = NormalizePath(path);
    std::string assetsNorm = NormalizePath(m_assetsRoot);

    if (assetsNorm.empty())
        return false;

    if (norm.size() < assetsNorm.size())
        return false;

    return norm.rfind(assetsNorm, 0) == 0;
}

void AssetDatabase::OnAssetCreated(const std::string &path)
{
    ImportAsset(path);
}

void AssetDatabase::OnAssetModified(const std::string &path)
{
    ModifyResource(path);

    std::string guid = GetGuidFromPath(path);
    if (guid.empty()) {
        ImportAsset(path);
    } else {
        UpdateMapping(guid, path);
        // Re-scan dependencies (material might have changed its textures)
        RunImporter(guid, path, true);
        // Notify dependents that this asset was modified
        ResourceType type = GetResourceTypeForPath(path);
        AssetDependencyGraph::Instance().NotifyEvent(guid, type, AssetEvent::Modified);
    }
}

void AssetDatabase::OnAssetDeleted(const std::string &path)
{
    DeleteAsset(path);
}

void AssetDatabase::OnAssetMoved(const std::string &oldPath, const std::string &newPath)
{
    MoveAsset(oldPath, newPath);
}

std::string AssetDatabase::NormalizePath(const std::string &path) const
{
    if (path.empty())
        return "";

    try {
        std::filesystem::path fsPath = ToFsPath(path);
        std::filesystem::path normPath;
        if (std::filesystem::exists(fsPath)) {
            normPath = std::filesystem::weakly_canonical(fsPath);
        } else {
            normPath = fsPath.lexically_normal();
        }
        std::string result = FromFsPath(normPath);

#ifdef INX_PLATFORM_WINDOWS
        // Only lowercase ASCII bytes — UTF-8 multi-byte sequences (>= 0x80) are
        // left untouched.  Using ::tolower on raw UTF-8 bytes is UB for non-ASCII.
        for (auto &ch : result) {
            if (ch >= 'A' && ch <= 'Z')
                ch = static_cast<char>(ch + ('a' - 'A'));
        }
#endif

        return result;
    } catch (...) {
        std::string result = FromFsPath(ToFsPath(path));
#ifdef INX_PLATFORM_WINDOWS
        for (auto &ch : result) {
            if (ch >= 'A' && ch <= 'Z')
                ch = static_cast<char>(ch + ('a' - 'A'));
        }
#endif
        return result;
    }
}

bool AssetDatabase::IsMetaFile(const std::filesystem::path &path) const
{
    return path.extension().string() == ".meta";
}

void AssetDatabase::UpdateMapping(const std::string &guid, const std::string &path)
{
    if (guid.empty() || path.empty())
        return;

    std::string norm = NormalizePath(path);
    m_guidToPath[guid] = path;
    m_pathToGuid[norm] = guid;
}

void AssetDatabase::RemoveMappingByGuid(const std::string &guid)
{
    auto it = m_guidToPath.find(guid);
    if (it != m_guidToPath.end()) {
        RemoveMappingByPath(it->second);
        m_guidToPath.erase(it);
    }
}

void AssetDatabase::RemoveMappingByPath(const std::string &path)
{
    std::string norm = NormalizePath(path);
    auto it = m_pathToGuid.find(norm);
    if (it != m_pathToGuid.end()) {
        m_pathToGuid.erase(it);
    }
}

void AssetDatabase::RunImporter(const std::string &guid, const std::string &path, bool isReimport)
{
    if (guid.empty() || path.empty())
        return;

    std::string ext = ToFsPath(path).extension().string();
    if (ext.empty())
        return;

    AssetImporter *importer = m_importerRegistry.GetImporterForExtension(ext);
    if (!importer)
        return;

    // Build the import context
    ImportContext ctx;
    ctx.sourcePath = path;
    ctx.guid = guid;
    ctx.resourceType = GetResourceTypeForPath(path);
    ctx.isReimport = isReimport;

    // Fetch meta (may be null if not yet created — that's OK for lightweight importers)
    InxResourceMeta *meta = nullptr;
    auto metaIt = m_metas.find(guid);
    if (metaIt != m_metas.end())
        meta = metaIt->second.get();
    ctx.meta = meta;

    if (isReimport)
        importer->Reimport(ctx);
    else
        importer->Import(ctx);

    // Persist metadata written by the importer (e.g. mesh_count, vertex_count)
    if (meta) {
        std::string metaPath = InxResourceMeta::GetMetaFilePath(path);
        if (!metaPath.empty())
            meta->SaveToFile(metaPath);
    }
}

// ============================================================================
// Resource management
// ============================================================================

bool AssetDatabase::ReadFile(const std::string &filePath, std::vector<char> &content) const
{
    bool isBinary = IsBinaryFile(filePath);

    std::ios_base::openmode mode = std::ios::in;
    if (isBinary) {
        mode |= std::ios::binary;
    }

    std::ifstream file(ToFsPath(filePath), mode);
    if (!file.is_open()) {
        INXLOG_ERROR("Failed to open file: ", filePath);
        content.clear();
        return false;
    }

    try {
        file.seekg(0, std::ios::end);
        if (file.fail()) {
            INXLOG_ERROR("Failed to seek to end of file: ", filePath);
            content.clear();
            return false;
        }

        std::streampos fileSize = file.tellg();
        if (fileSize == std::streampos(-1)) {
            INXLOG_ERROR("Failed to get file size: ", filePath);
            content.clear();
            return false;
        }

        file.seekg(0, std::ios::beg);
        if (file.fail()) {
            INXLOG_ERROR("Failed to seek to beginning of file: ", filePath);
            content.clear();
            return false;
        }

        content.reserve(static_cast<size_t>(fileSize));
        content.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        if (file.bad() || file.fail()) {
            INXLOG_ERROR("Error occurred while reading file: ", filePath);
            content.clear();
            return false;
        }

        return true;
    } catch (const std::exception &e) {
        INXLOG_ERROR("Exception while reading file: ", filePath, " - ", e.what());
        content.clear();
        return false;
    }
}

bool AssetDatabase::IsBinaryFile(const std::string &filePath) const
{
    std::filesystem::path path = ToFsPath(filePath);
    std::string extension = path.extension().string();

    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    static const std::unordered_set<std::string> textExtensions = {
        ".txt", ".md",  ".json", ".xml",  ".html", ".htm",  ".css", ".js",   ".ts",    ".cpp",  ".c",     ".h",
        ".hpp", ".py",  ".java", ".cs",   ".php",  ".rb",   ".go",  ".rs",   ".swift", ".kt",   ".scala", ".pl",
        ".lua", ".r",   ".sql",  ".yaml", ".yml",  ".toml", ".ini", ".cfg",  ".conf",  ".log",  ".csv",   ".tsv",
        ".rtf", ".tex", ".bib",  ".sh",   ".bat",  ".ps1",  ".cmd", ".vert", ".frag",  ".glsl", ".hlsl",  ".shader"};

    if (textExtensions.find(extension) != textExtensions.end()) {
        return false;
    }

    static const std::unordered_set<std::string> binaryExtensions = {
        ".exe", ".dll",  ".so",  ".dylib", ".bin", ".dat",  ".db",   ".sqlite", ".jpg", ".jpeg",
        ".png", ".gif",  ".bmp", ".tiff",  ".ico", ".webp", ".mp3",  ".wav",    ".ogg", ".flac",
        ".aac", ".m4a",  ".wma", ".mp4",   ".avi", ".mkv",  ".mov",  ".wmv",    ".flv", ".webm",
        ".zip", ".rar",  ".7z",  ".tar",   ".gz",  ".bz2",  ".xz",   ".pdf",    ".doc", ".docx",
        ".xls", ".xlsx", ".ppt", ".pptx",  ".ttf", ".otf",  ".woff", ".woff2",  ".eot"};

    if (binaryExtensions.find(extension) != binaryExtensions.end()) {
        return true;
    }

    return DetectBinaryByContent(filePath);
}

bool AssetDatabase::DetectBinaryByContent(const std::string &filePath) const
{
    std::ifstream file(ToFsPath(filePath), std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    const size_t sampleSize = 512;
    std::vector<char> buffer(sampleSize);
    file.read(buffer.data(), sampleSize);
    size_t bytesRead = file.gcount();

    size_t nullBytes = 0;
    size_t nonAsciiBytes = 0;

    for (size_t i = 0; i < bytesRead; ++i) {
        unsigned char byte = static_cast<unsigned char>(buffer[i]);

        if (byte == 0) {
            nullBytes++;
        } else if (byte > 127) {
            nonAsciiBytes++;
        }
    }

    if (nullBytes > 0) {
        return true;
    }

    double nonAsciiRatio = static_cast<double>(nonAsciiBytes) / bytesRead;
    return nonAsciiRatio > 0.3;
}

ResourceType AssetDatabase::GetResourcesType(const std::string &extensionName) const
{
    std::string ext = extensionName;
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".vert" || ext == ".frag" || ext == ".geom" || ext == ".comp" || ext == ".tesc" || ext == ".tese") {
        return ResourceType::Shader;
    }
    if (ext == ".mat") {
        return ResourceType::Material;
    }
    if (ext == ".meta") {
        return ResourceType::Meta;
    }
    if (ext == ".py") {
        return ResourceType::Script;
    }
    static const std::unordered_set<std::string> textureExtensions = {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".gif",
                                                                      ".psd", ".hdr", ".pic",  ".pnm", ".pgm", ".ppm"};
    if (textureExtensions.find(ext) != textureExtensions.end()) {
        return ResourceType::Texture;
    }
    static const std::unordered_set<std::string> audioExtensions = {".wav"};
    if (audioExtensions.find(ext) != audioExtensions.end()) {
        return ResourceType::Audio;
    }
    static const std::unordered_set<std::string> meshExtensions = {".fbx", ".obj", ".gltf", ".glb",
                                                                   ".dae", ".3ds", ".ply",  ".stl"};
    if (meshExtensions.find(ext) != meshExtensions.end()) {
        return ResourceType::Mesh;
    }
    static const std::unordered_set<std::string> textExtensions = {
        ".txt", ".md",  ".json", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".html",
        ".htm", ".css", ".js",   ".ts",  ".lua",  ".cs",  ".cpp",  ".c",   ".h",   ".hpp"};
    if (textExtensions.find(ext) != textExtensions.end()) {
        return ResourceType::DefaultText;
    }
    static const std::unordered_set<std::string> binaryExtensions = {
        ".exe", ".dll", ".so",  ".dylib", ".bin", ".dat", ".wav", ".mp3", ".ogg", ".flac", ".mp4", ".avi",
        ".mkv", ".mov", ".zip", ".rar",   ".7z",  ".tar", ".gz",  ".pdf", ".ttf", ".otf",  ".woff"};
    if (binaryExtensions.find(ext) != binaryExtensions.end()) {
        return ResourceType::DefaultBinary;
    }
    return ResourceType::DefaultText;
}

ResourceType AssetDatabase::GetResourceTypeForPath(const std::string &filePath) const
{
    std::filesystem::path path = ToFsPath(filePath);
    std::string ext = path.extension().string();
    return GetResourcesType(ext);
}

std::string AssetDatabase::RegisterResource(const std::string &filePath, ResourceType type)
{
    INXLOG_DEBUG("Registering resource: filePath = ", filePath, ", type = ", static_cast<int>(type));

    if (filePath.empty()) {
        INXLOG_ERROR("Received empty filePath!");
        return "";
    }

    auto loader = m_loaders.find(type);
    if (loader == m_loaders.end()) {
        INXLOG_ERROR("Resource type not supported: ", static_cast<int>(type));
        return "";
    }

    std::vector<char> content;
    if (!ReadFile(filePath, content)) {
        INXLOG_ERROR("Failed to read file for resource registration: ", filePath);
        return "";
    }

    if (content.size() == 0)
        content.emplace_back(0);
    const char *contentPtr = content.data();

    InxResourceMeta metaFile;
    std::string metaFilePath = InxResourceMeta::GetMetaFilePath(filePath);

    if (!metaFile.LoadFromFile(metaFilePath)) {
        if (!m_loaders[type]->LoadMeta(contentPtr, filePath, metaFile)) {
            m_loaders[type]->CreateMeta(contentPtr, content.size(), filePath, metaFile);
            metaFile.SaveToFile(metaFilePath);
        } else {
            metaFile.SaveToFile(metaFilePath);
        }
    }

    std::string guid = metaFile.GetGuid();
    m_metas[guid] = std::make_unique<InxResourceMeta>(metaFile);
    UpdateMapping(guid, filePath);
    INXLOG_DEBUG("Resource metadata registered with GUID: ", guid);

    return guid;
}

void AssetDatabase::RemoveResourceMeta(const std::string &uid)
{
    auto metaIt = m_metas.find(uid);
    if (metaIt != m_metas.end()) {
        m_metas.erase(metaIt);
        INXLOG_INFO("Resource meta removed successfully with UID: ", uid);
    } else {
        INXLOG_ERROR("Resource meta not found for UID: ", uid);
    }
}

void AssetDatabase::ModifyResource(const std::string &path)
{
    namespace fs = std::filesystem;
    fs::path filePath = ToFsPath(path);

    if (!fs::exists(filePath)) {
        INXLOG_WARN("ModifyResource: file does not exist: ", path);
        return;
    }

    std::string ext = FromFsPath(filePath.extension());
    ResourceType type = GetResourcesType(ext);

    if (type == ResourceType::Meta) {
        return;
    }

    std::string metaPath = InxResourceMeta::GetMetaFilePath(path);

    std::vector<char> content;
    if (!ReadFile(path, content)) {
        INXLOG_ERROR("ModifyResource: failed to read file: ", path);
        return;
    }
    if (content.empty()) {
        content.emplace_back(0);
    }

    InxResourceMeta meta;
    std::string existingGuid;

    fs::path fsMetaPath = ToFsPath(metaPath);

    if (fs::exists(fsMetaPath) && meta.LoadFromFile(metaPath)) {
        existingGuid = meta.GetGuid();
    }

    auto loaderIt = m_loaders.find(type);
    if (loaderIt == m_loaders.end()) {
        INXLOG_ERROR("ModifyResource: no loader for type: ", static_cast<int>(type));
        return;
    }

    InxResourceMeta newMeta;
    loaderIt->second->CreateMeta(content.data(), content.size(), path, newMeta);

    if (fs::exists(fsMetaPath) && meta.GetMetadata().size() > 0) {
        for (const auto &[key, metaPair] : meta.GetMetadata()) {
            if (key == "guid") {
                continue;
            }
            if (!newMeta.HasKey(key)) {
                newMeta.AddMetadata(key, metaPair.second);
            }
        }
    }

    if (!existingGuid.empty()) {
        newMeta.AddMetadata("guid", existingGuid);
    }

    newMeta.SaveToFile(metaPath);

    std::string guid = newMeta.GetGuid();
    m_metas[guid] = std::make_unique<InxResourceMeta>(newMeta);
}

void AssetDatabase::DeleteResource(const std::string &path)
{
    namespace fs = std::filesystem;

    std::string guidToRemove;
    for (const auto &[guid, meta] : m_metas) {
        if (meta->HasKey("file_path")) {
            std::string metaPath = meta->GetDataAs<std::string>("file_path");
            if (metaPath == path) {
                guidToRemove = guid;
                break;
            }
        }
    }

    if (!guidToRemove.empty()) {
        m_metas.erase(guidToRemove);
    }

    std::string metaPath = InxResourceMeta::GetMetaFilePath(path);
    auto metaFsPath = ToFsPath(metaPath);
    if (fs::exists(metaFsPath)) {
        fs::remove(metaFsPath);
        INXLOG_DEBUG("DeleteResource: deleted meta file: ", metaPath);
    }
}

void AssetDatabase::MoveResource(const std::string &oldPath, const std::string &newPath)
{
    namespace fs = std::filesystem;

    std::string oldMetaPath = InxResourceMeta::GetMetaFilePath(oldPath);
    std::string newMetaPath = InxResourceMeta::GetMetaFilePath(newPath);

    InxResourceMeta meta;
    std::string existingGuid;

    auto oldMetaFsPath = ToFsPath(oldMetaPath);
    if (fs::exists(oldMetaFsPath) && meta.LoadFromFile(oldMetaPath)) {
        existingGuid = meta.GetGuid();

        meta.UpdateFilePath(newPath);
        meta.SaveToFile(newMetaPath);
        fs::remove(oldMetaFsPath);

        auto it = m_metas.find(existingGuid);
        if (it != m_metas.end()) {
            it->second->UpdateFilePath(newPath);
        }

        INXLOG_INFO("MoveResource: ", oldPath, " -> ", newPath, " (guid preserved: ", existingGuid, ")");
    } else {
        std::string ext = ToFsPath(newPath).extension().string();
        ResourceType type = GetResourcesType(ext);

        if (type != ResourceType::Meta) {
            RegisterResource(newPath, type);
            INXLOG_INFO("MoveResource: registered new resource at: ", newPath);
        }
    }
}

std::vector<std::string> AssetDatabase::GetAllResourceGuids() const
{
    std::vector<std::string> guids;
    guids.reserve(m_metas.size());
    for (const auto &[guid, meta] : m_metas) {
        guids.push_back(guid);
    }
    return guids;
}

std::string AssetDatabase::FindShaderPathById(const std::string &shaderId, const std::string &shaderType) const
{
    std::string expectedExt;
    if (shaderType == "vertex" || shaderType == ".vert" || shaderType == "vert") {
        expectedExt = ".vert";
    } else if (shaderType == "fragment" || shaderType == ".frag" || shaderType == "frag") {
        expectedExt = ".frag";
    } else {
        return "";
    }

    for (const auto &[guid, meta] : m_metas) {
        if (!meta)
            continue;

        if (!meta->HasKey("type"))
            continue;
        std::string type = meta->GetDataAs<std::string>("type");
        bool matchesType =
            (expectedExt == ".vert" && type == "vertex") || (expectedExt == ".frag" && type == "fragment");
        if (!matchesType)
            continue;

        if (meta->HasKey("shader_id")) {
            std::string metaShaderId = meta->GetDataAs<std::string>("shader_id");
            if (metaShaderId == shaderId) {
                if (meta->HasKey("file_path")) {
                    return meta->GetDataAs<std::string>("file_path");
                }
            }
        }
    }

    return "";
}

} // namespace infernux
