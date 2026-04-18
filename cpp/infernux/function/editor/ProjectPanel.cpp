#include "ProjectPanel.h"

#include <function/renderer/gui/InxResourcePreviewer.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <imgui_internal.h>

#ifdef INX_PLATFORM_WINDOWS
#include <ShlObj.h> // CF_HDROP, DragQueryFileW
#include <strsafe.h>
#endif

namespace fs = std::filesystem;

// ImGui key constants
static constexpr int kKeyLeftCtrl = ImGuiKey_LeftCtrl;
static constexpr int kKeyRightCtrl = ImGuiKey_RightCtrl;
static constexpr int kKeyLeftShift = ImGuiKey_LeftShift;
static constexpr int kKeyRightShift = ImGuiKey_RightShift;
static constexpr int kKeyF2 = ImGuiKey_F2;
static constexpr int kKeyDelete = ImGuiKey_Delete;
static constexpr int kKeyEnter = ImGuiKey_Enter;
static constexpr int kKeyEscape = ImGuiKey_Escape;
static constexpr int kKeyC = ImGuiKey_C;
static constexpr int kKeyV = ImGuiKey_V;
static constexpr int kKeyX = ImGuiKey_X;
static constexpr int kKeyN = ImGuiKey_N;

// Unicode characters
static constexpr const char *kExpandedArrow = "\xe2\x96\xbc";   // ▼
static constexpr const char *kCollapsedArrow = "\xe2\x96\xb6";  // ▶
static constexpr const char *kSubAssetPrefix = "\xe2\x86\xb3 "; // ↳
static constexpr const char *kEllipsis = "\xe2\x80\xa6";        // …

namespace infernux
{

// ════════════════════════════════════════════════════════════════════
// Static extension sets
// ════════════════════════════════════════════════════════════════════

static const std::unordered_set<std::string> sImageExtensions = {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".gif"};

static const std::unordered_set<std::string> sMaterialExtensions = {".mat"};

static const std::unordered_set<std::string> sModelExtensions = {".fbx", ".obj", ".gltf", ".glb",
                                                                 ".dae", ".3ds", ".ply",  ".stl"};

static const std::unordered_set<std::string> sHiddenExtensions = {".meta", ".pyc", ".pyo", ".tmp"};

static const std::unordered_set<std::string> sHiddenFiles = {"imgui.ini"};

bool ProjectPanel::IsImageExt(const std::string &ext)
{
    return sImageExtensions.count(ext) > 0;
}
bool ProjectPanel::IsMaterialExt(const std::string &ext)
{
    return sMaterialExtensions.count(ext) > 0;
}
bool ProjectPanel::IsModelExt(const std::string &ext)
{
    return sModelExtensions.count(ext) > 0;
}

// ════════════════════════════════════════════════════════════════════
// Static data: Icon map
// ════════════════════════════════════════════════════════════════════

const std::unordered_map<std::string, std::string> &ProjectPanel::GetIconMap()
{
    static const std::unordered_map<std::string, std::string> map = {
        {"__dir__", "folder"},    {".py", "script_py"},     {".lua", "script_lua"},   {".cs", "script_cs"},
        {".cpp", "script_cpp"},   {".c", "script_cpp"},     {".h", "script_cpp"},     {".vert", "shader_vert"},
        {".frag", "shader_frag"}, {".glsl", "shader_glsl"}, {".hlsl", "shader_hlsl"}, {".mat", "material"},
        {".png", "image"},        {".jpg", "image"},        {".jpeg", "image"},       {".bmp", "image"},
        {".tga", "image"},        {".gif", "image"},        {".fbx", "model_3d"},     {".obj", "model_3d"},
        {".gltf", "model_3d"},    {".glb", "model_3d"},     {".wav", "audio"},        {".ttf", "font"},
        {".otf", "font"},         {".txt", "text"},         {".md", "readme"},        {".json", "config"},
        {".yaml", "config"},      {".yml", "config"},       {".xml", "config"},       {".scene", "scene"},
        {".prefab", "prefab"},
    };
    return map;
}

// ════════════════════════════════════════════════════════════════════
// Static data: Drag-drop maps
// ════════════════════════════════════════════════════════════════════

const std::unordered_map<std::string, ProjectPanel::DragDropInfo> &ProjectPanel::GetDragDropMap()
{
    static const std::unordered_map<std::string, DragDropInfo> map = {
        {".py", {"SCRIPT_FILE", "Script"}},     {".mat", {"MATERIAL_FILE", "Material"}},
        {".vert", {"SHADER_FILE", "Shader"}},   {".frag", {"SHADER_FILE", "Shader"}},
        {".glsl", {"SHADER_FILE", "Shader"}},   {".hlsl", {"SHADER_FILE", "Shader"}},
        {".png", {"TEXTURE_FILE", "Texture"}},  {".jpg", {"TEXTURE_FILE", "Texture"}},
        {".jpeg", {"TEXTURE_FILE", "Texture"}}, {".bmp", {"TEXTURE_FILE", "Texture"}},
        {".tga", {"TEXTURE_FILE", "Texture"}},  {".gif", {"TEXTURE_FILE", "Texture"}},
        {".psd", {"TEXTURE_FILE", "Texture"}},  {".hdr", {"TEXTURE_FILE", "Texture"}},
        {".pic", {"TEXTURE_FILE", "Texture"}},  {".wav", {"AUDIO_FILE", "Audio"}},
        {".ttf", {"FONT_FILE", "Font"}},        {".otf", {"FONT_FILE", "Font"}},
        {".scene", {"SCENE_FILE", "Scene"}},
        {".animclip2d", {"ANIMCLIP_FILE", "2D AnimClip"}},
        {".animfsm", {"ANIMFSM_FILE", "AnimFSM"}},
    };
    return map;
}

const std::unordered_map<std::string, ProjectPanel::GuidDragDropInfo> &ProjectPanel::GetGuidDragDropMap()
{
    static const std::unordered_map<std::string, GuidDragDropInfo> map = {
        {".prefab", {"PREFAB_GUID", "PREFAB_FILE", "Prefab"}}, {".fbx", {"MODEL_GUID", "MODEL_FILE", "Model"}},
        {".obj", {"MODEL_GUID", "MODEL_FILE", "Model"}},       {".gltf", {"MODEL_GUID", "MODEL_FILE", "Model"}},
        {".glb", {"MODEL_GUID", "MODEL_FILE", "Model"}},       {".dae", {"MODEL_GUID", "MODEL_FILE", "Model"}},
        {".3ds", {"MODEL_GUID", "MODEL_FILE", "Model"}},       {".ply", {"MODEL_GUID", "MODEL_FILE", "Model"}},
        {".stl", {"MODEL_GUID", "MODEL_FILE", "Model"}},
    };
    return map;
}

const std::vector<std::string> &ProjectPanel::GetMoveAcceptTypes()
{
    static const std::vector<std::string> types = [] {
        std::unordered_set<std::string> pathTypes;
        std::unordered_set<std::string> guidTypes;
        for (auto &[_, info] : GetDragDropMap())
            pathTypes.insert(info.payloadType);
        for (auto &[_, info] : GetGuidDragDropMap()) {
            pathTypes.insert(info.pathPayloadType);
            guidTypes.insert(info.guidPayloadType);
        }
        pathTypes.insert(DRAG_TYPE_PROJECT_ITEM);
        std::vector<std::string> result;
        result.reserve(pathTypes.size() + guidTypes.size());
        for (auto &t : pathTypes)
            result.push_back(t);
        for (auto &t : guidTypes)
            result.push_back(t);
        return result;
    }();
    return types;
}

// ════════════════════════════════════════════════════════════════════
// File type tag for text fallback
// ════════════════════════════════════════════════════════════════════

const char *ProjectPanel::GetFileTypeTag(const std::string &filename)
{
    auto dot = filename.rfind('.');
    if (dot == std::string::npos)
        return "[FILE]";
    std::string ext = filename.substr(dot);
    for (auto &c : ext)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (ext == ".py" || ext == ".lua" || ext == ".cs")
        return "[PY]";
    if (ext == ".mat")
        return "[MAT]";
    if (ext == ".vert" || ext == ".frag" || ext == ".glsl" || ext == ".hlsl")
        return "[SHDR]";
    if (IsImageExt(ext))
        return "[IMG]";
    if (IsModelExt(ext))
        return "[3D]";
    if (ext == ".scene")
        return "[SCN]";
    if (ext == ".prefab")
        return "[PRE]";
    if (ext == ".wav")
        return "[AUD]";
    if (ext == ".ttf" || ext == ".otf")
        return "[FNT]";
    if (ext == ".json" || ext == ".yaml" || ext == ".yml" || ext == ".xml")
        return "[CFG]";
    if (ext == ".txt" || ext == ".md")
        return "[TXT]";
    return "[FILE]";
}

// ════════════════════════════════════════════════════════════════════
// Filtering
// ════════════════════════════════════════════════════════════════════

bool ProjectPanel::ShouldShow(const std::string &name)
{
    if (name.empty())
        return false;
    if (sHiddenFiles.count(name) > 0)
        return false;
    // Check hidden prefixes: '.', '__'
    if (name[0] == '.')
        return false;
    if (name.size() >= 2 && name[0] == '_' && name[1] == '_')
        return false;
    // Check extension
    auto dot = name.rfind('.');
    if (dot != std::string::npos) {
        std::string ext = name.substr(dot);
        for (auto &c : ext)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (sHiddenExtensions.count(ext) > 0)
            return false;
    }
    return true;
}

// ════════════════════════════════════════════════════════════════════
// Path utilities
// ════════════════════════════════════════════════════════════════════

std::string ProjectPanel::NormalizePath(const std::string &path)
{
    std::error_code ec;
    auto canonical = fs::weakly_canonical(fs::u8path(path), ec);
    if (ec)
        return path;
    auto str = canonical.string();
    // Lowercase on Windows for case-insensitive comparison
#ifdef _WIN32
    for (auto &c : str)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
#endif
    return str;
}

bool ProjectPanel::IsPathWithin(const std::string &path, const std::string &parent)
{
    auto np = NormalizePath(path);
    auto npar = NormalizePath(parent);
    if (np.size() < npar.size())
        return false;
    if (np.substr(0, npar.size()) != npar)
        return false;
    // Must be exactly parent or parent + separator
    if (np.size() == npar.size())
        return true;
    char sep = np[npar.size()];
    return sep == '/' || sep == '\\';
}

int64_t ProjectPanel::GetMtimeNs(const std::string &path)
{
    std::error_code ec;
    auto ftime = fs::last_write_time(fs::u8path(path), ec);
    if (ec)
        return 0;
    // Use the file_clock duration directly (epoch-independent but stable for comparison)
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(ftime.time_since_epoch()).count();
    return static_cast<int64_t>(ns);
}

// ════════════════════════════════════════════════════════════════════
// Construction
// ════════════════════════════════════════════════════════════════════

ProjectPanel::ProjectPanel() : EditorPanel("Project", "project")
{
}

// ════════════════════════════════════════════════════════════════════
// Translation helper
// ════════════════════════════════════════════════════════════════════

const std::string &ProjectPanel::Tr(const std::string &key)
{
    auto it = m_trCache.find(key);
    if (it != m_trCache.end())
        return it->second;
    if (translate)
        m_trCache[key] = translate(key);
    else
        m_trCache[key] = key;
    return m_trCache[key];
}

// ════════════════════════════════════════════════════════════════════
// Public API
// ════════════════════════════════════════════════════════════════════

void ProjectPanel::SetRootPath(const std::string &path)
{
    m_rootPath = path;
    InvalidateDirCache();
    auto assetsPath = (fs::u8path(path) / "Assets").string();
    std::error_code ec;
    if (fs::is_directory(assetsPath, ec))
        m_currentPath = assetsPath;
    else
        m_currentPath = path;
}

void ProjectPanel::SetRenderer(InxRenderer *renderer)
{
    if (m_renderer == renderer)
        return;
    m_renderer = renderer;
    m_typeIconCache.clear();
    m_typeIconsLoaded = false;
}
void ProjectPanel::SetAssetDatabase(AssetDatabase *adb)
{
    m_assetDatabase = adb;
}
void ProjectPanel::SetIconsDirectory(const std::string &dir)
{
    if (m_iconsDir == dir && m_typeIconsLoaded)
        return;
    m_iconsDir = dir;
    m_typeIconCache.clear();
    m_typeIconsLoaded = false;
}

void ProjectPanel::SetCurrentPath(const std::string &path)
{
    std::error_code ec;
    if (!path.empty() && fs::is_directory(path, ec))
        m_currentPath = path;
}

void ProjectPanel::ClearSelection()
{
    if (!m_selectedFile.empty() || !m_selectedFiles.empty()) {
        m_selectedFile.clear();
        m_selectedFiles.clear();
        m_selectedSet.clear();
        NotifySelectionChanged();
    }
}

void ProjectPanel::SetSelectedFile(const std::string &path)
{
    if (path.empty()) {
        ClearSelection();
        return;
    }

    std::error_code ec;
    fs::path selectedPath = fs::u8path(path);
    fs::path parent = selectedPath.parent_path();
    if (!parent.empty() && fs::is_directory(parent, ec))
        m_currentPath = parent.string();

    m_selectedFile = path;
    m_selectedFiles = {path};
    m_selectedSet.clear();
    m_selectedSet.insert(path);
    NotifySelectionChanged();
}

void ProjectPanel::InvalidateMaterialThumbnail(const std::string &filePath)
{
    if (filePath.empty() || !m_renderer)
        return;
    auto normTarget = NormalizePath(filePath);

    std::vector<std::string> toRemove;
    for (auto &[path, _] : m_thumbnailCache) {
        if (NormalizePath(path) == normTarget)
            toRemove.push_back(path);
    }
    for (auto &path : toRemove) {
        std::string thumbName = "__mat_thumb__" + path;
        if (m_renderer->HasImGuiTexture(thumbName))
            m_renderer->RemoveImGuiTexture(thumbName);
        m_thumbnailCache.erase(path);
    }

    std::vector<std::string> mtimeToRemove;
    for (auto &[path, _] : m_materialMtimeCache) {
        if (NormalizePath(path) == normTarget)
            mtimeToRemove.push_back(path);
    }
    for (auto &path : mtimeToRemove)
        m_materialMtimeCache.erase(path);
}

// ════════════════════════════════════════════════════════════════════
// Notification helpers
// ════════════════════════════════════════════════════════════════════

void ProjectPanel::NotifySelectionChanged()
{
    if (!onFileSelected)
        return;
    if (m_selectedFiles.size() == 1)
        onFileSelected(m_selectedFiles[0]);
    else
        onFileSelected("");
}

void ProjectPanel::NotifyEmptyAreaClicked()
{
    if (onEmptyAreaClicked)
        onEmptyAreaClicked();
}

std::vector<std::string> ProjectPanel::GetSelectedPaths() const
{
    std::vector<std::string> result;
    std::error_code ec;
    for (auto &p : m_selectedFiles) {
        if (!p.empty() && fs::exists(fs::u8path(p), ec))
            result.push_back(p);
    }
    if (result.empty() && !m_selectedFile.empty() && fs::exists(fs::u8path(m_selectedFile), ec))
        result.push_back(m_selectedFile);
    return result;
}

// ════════════════════════════════════════════════════════════════════
// Directory snapshot cache
// ════════════════════════════════════════════════════════════════════

void ProjectPanel::InvalidateDirCache()
{
    m_dirCache.clear();
    m_dirTreeMetaCache.clear();
    m_augmentedCache.clear();
    m_labelCache.clear();
    ClearThumbnailQueue();
}

ProjectPanel::DirSnapshot *ProjectPanel::GetDirSnapshot(const std::string &path)
{
    if (path.empty())
        return nullptr;

    // Fast path: return cached snapshot if TTL hasn't expired (avoids syscalls)
    auto it = m_dirCache.find(path);
    if (it != m_dirCache.end()) {
        if ((m_frameTimeNow - it->second.lastValidatedAt) < DIR_CACHE_TTL)
            return &it->second;
        int64_t mtimeNs = GetMtimeNs(path);
        it->second.lastValidatedAt = m_frameTimeNow;
        if (it->second.mtimeNs == mtimeNs)
            return &it->second;
    }

    std::error_code ec;
    if (!fs::is_directory(fs::u8path(path), ec))
        return nullptr;

    int64_t mtimeNs = GetMtimeNs(path);

    DirSnapshot snap;
    snap.mtimeNs = mtimeNs;
    snap.lastValidatedAt = m_frameTimeNow;

    for (auto &entry : fs::directory_iterator(fs::u8path(path), ec)) {
        if (ec)
            break;
        auto name = entry.path().filename().string();
        if (!ShouldShow(name))
            continue;

        bool isDir = entry.is_directory(ec);
        if (ec) {
            ec.clear();
            continue;
        }

        FileItem item;
        item.name = std::move(name);
        item.path = entry.path().string();

        if (isDir) {
            item.type = FileItem::Dir;
            snap.dirs.push_back(std::move(item));
        } else {
            item.type = FileItem::File;
            auto ext = entry.path().extension().string();
            for (auto &c : ext)
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            item.ext = std::move(ext);

            if (IsImageExt(item.ext) || IsMaterialExt(item.ext)) {
                auto ftime = entry.last_write_time(ec);
                if (!ec) {
                    item.mtimeNs =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(ftime.time_since_epoch()).count();
                }
            }
            snap.files.push_back(std::move(item));
        }
    }

    // Sort case-insensitive
    auto cmpName = [](const FileItem &a, const FileItem &b) {
        auto la = a.name, lb = b.name;
        for (auto &c : la)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        for (auto &c : lb)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return la < lb;
    };
    std::sort(snap.dirs.begin(), snap.dirs.end(), cmpName);
    std::sort(snap.files.begin(), snap.files.end(), cmpName);

    snap.items.reserve(snap.dirs.size() + snap.files.size());
    snap.items.insert(snap.items.end(), snap.dirs.begin(), snap.dirs.end());
    snap.items.insert(snap.items.end(), snap.files.begin(), snap.files.end());

    // Update tree meta
    m_dirTreeMetaCache[path] = {!snap.dirs.empty()};

    auto &stored = m_dirCache[path] = std::move(snap);
    return &stored;
}

ProjectPanel::DirTreeMeta *ProjectPanel::GetDirTreeMeta(const std::string &path)
{
    if (path.empty())
        return nullptr;

    auto it = m_dirTreeMetaCache.find(path);
    if (it != m_dirTreeMetaCache.end())
        return &it->second;

    std::error_code ec;
    bool hasSubdirs = false;
    for (auto &entry : fs::directory_iterator(fs::u8path(path), ec)) {
        if (ec)
            break;
        auto name = entry.path().filename().string();
        if (!ShouldShow(name))
            continue;
        if (entry.is_directory(ec) && !ec) {
            hasSubdirs = true;
            break;
        }
        ec.clear();
    }

    auto &meta = m_dirTreeMetaCache[path];
    meta.hasSubdirs = hasSubdirs;
    return &meta;
}

std::vector<ProjectPanel::FileItem> *ProjectPanel::GetProjectItems(const std::string &path, DirSnapshot *snapshot)
{
    if (!snapshot)
        snapshot = GetDirSnapshot(path);
    if (!snapshot)
        return nullptr;

    if (m_expandedModels.empty())
        return &snapshot->items;

    // Check if any expanded models in current items
    std::vector<std::string> expandedPaths;
    for (auto &item : snapshot->items) {
        if (item.type == FileItem::File && IsModelExt(item.ext) && m_expandedModels.count(item.path) > 0) {
            expandedPaths.push_back(item.path);
        }
    }
    if (expandedPaths.empty())
        return &snapshot->items;

    auto cacheIt = m_augmentedCache.find(path);
    if (cacheIt != m_augmentedCache.end() && cacheIt->second.mtimeNs == snapshot->mtimeNs &&
        cacheIt->second.expandedPaths == expandedPaths) {
        return &cacheIt->second.items;
    }

    // Build augmented list — model sub-assets are handled via callback
    // For now just return snapshot items (sub-asset expansion requires
    // AssetRegistry callback which is Python-side)
    auto &cached = m_augmentedCache[path];
    cached.mtimeNs = snapshot->mtimeNs;
    cached.expandedPaths = expandedPaths;
    cached.items = snapshot->items; // Copy base items
    return &cached.items;
}

// ════════════════════════════════════════════════════════════════════
// Thumbnail system
// ════════════════════════════════════════════════════════════════════

void ProjectPanel::ClearThumbnailQueue()
{
    m_thumbQueue.clear();
    m_thumbQueueKeys.clear();
}

void ProjectPanel::QueueThumbnailRequest(const std::string &kind, const std::string &filePath)
{
    if (filePath.empty())
        return;

    std::string retryKey = kind + "|" + filePath;
    auto retryIt = m_thumbRetryAfter.find(retryKey);
    if (retryIt != m_thumbRetryAfter.end()) {
        if (retryIt->second > m_frameTimeNow)
            return;
    }

    ThumbnailRequest req{m_currentPath, kind, filePath};
    if (m_thumbQueueKeys.count(req) > 0)
        return;

    m_thumbQueue.push_back(req);
    m_thumbQueueKeys.insert(req);
}

bool ProjectPanel::DownsampleTexture(const std::string &filePath, int maxPx, std::vector<unsigned char> &outPixels,
                                     int &outWidth, int &outHeight)
{
    auto texData = InxTextureLoader::LoadFromFile(filePath);
    if (!texData.IsValid())
        return false;

    int srcW = texData.width;
    int srcH = texData.height;
    const auto &srcPixels = texData.pixels;

    if (srcW > maxPx || srcH > maxPx) {
        float scale = static_cast<float>(maxPx) / static_cast<float>(std::max(srcW, srcH));
        int w = std::max(1, static_cast<int>(srcW * scale));
        int h = std::max(1, static_cast<int>(srcH * scale));
        outPixels.resize(w * h * 4);
        int rowStride = srcW * 4;
        for (int dy = 0; dy < h; ++dy) {
            int sy = std::min(static_cast<int>((dy + 0.5f) * srcH / h), srcH - 1);
            int rowOff = sy * rowStride;
            for (int dx = 0; dx < w; ++dx) {
                int sx = std::min(static_cast<int>((dx + 0.5f) * srcW / w), srcW - 1);
                int srcIdx = rowOff + sx * 4;
                int dstIdx = (dy * w + dx) * 4;
                outPixels[dstIdx + 0] = srcPixels[srcIdx + 0];
                outPixels[dstIdx + 1] = srcPixels[srcIdx + 1];
                outPixels[dstIdx + 2] = srcPixels[srcIdx + 2];
                outPixels[dstIdx + 3] = srcPixels[srcIdx + 3];
            }
        }
        outWidth = w;
        outHeight = h;
    } else {
        outPixels = srcPixels;
        outWidth = srcW;
        outHeight = srcH;
    }
    return true;
}

int64_t ProjectPanel::GetMaterialMtimeNs(const std::string &filePath)
{
    if (filePath.empty())
        return 0;

    double now = m_frameTimeNow;

    auto it = m_materialMtimeCache.find(filePath);
    if (it != m_materialMtimeCache.end() && (now - it->second.second) < 1.0)
        return it->second.first;

    std::error_code ec;
    if (!fs::exists(fs::u8path(filePath), ec))
        return 0;

    int64_t mtimeNs = GetMtimeNs(filePath);
    m_materialMtimeCache[filePath] = {mtimeNs, now};
    return mtimeNs;
}

uint64_t ProjectPanel::GetThumbnail(const std::string &filePath, int64_t cachedMtimeNs)
{
    if (filePath.empty() || !m_renderer)
        return 0;

    if (cachedMtimeNs == 0) {
        std::error_code ec;
        if (!fs::exists(fs::u8path(filePath), ec))
            return 0;
        cachedMtimeNs = GetMtimeNs(filePath);
    }

    // Fast path: C++ cache hit — no string allocation needed
    auto it = m_thumbnailCache.find(filePath);
    if (it != m_thumbnailCache.end() && it->second.mtimeNs == cachedMtimeNs && it->second.texId != 0)
        return it->second.texId;

    // Slow path: check renderer GPU cache (string allocation only on miss)
    std::string thumbName = "__thumb__" + filePath;
    if (m_renderer->HasImGuiTexture(thumbName)) {
        auto texId = m_renderer->GetImGuiTextureId(thumbName);
        if (texId != 0) {
            m_thumbnailCache[filePath] = {texId, cachedMtimeNs};
            return texId;
        }
    }

    QueueThumbnailRequest("image", filePath);
    return 0;
}

uint64_t ProjectPanel::GetMaterialThumbnail(const std::string &filePath)
{
    if (filePath.empty() || !m_renderer)
        return 0;

    int64_t mtimeNs = GetMaterialMtimeNs(filePath);
    if (mtimeNs == 0)
        return 0;

    // Fast path: C++ cache hit — no string allocation
    auto it = m_thumbnailCache.find(filePath);
    if (it != m_thumbnailCache.end()) {
        if (it->second.mtimeNs == mtimeNs && it->second.texId != 0)
            return it->second.texId;
        if (it->second.mtimeNs != mtimeNs) {
            std::string thumbName = "__mat_thumb__" + filePath;
            if (m_renderer->HasImGuiTexture(thumbName))
                m_renderer->RemoveImGuiTexture(thumbName);
            m_thumbnailCache.erase(it);
        }
    }

    QueueThumbnailRequest("material", filePath);
    return 0;
}

void ProjectPanel::ProcessPendingThumbnails()
{
    if (!m_renderer)
        return;

    if (m_thumbQueuePath != m_currentPath) {
        m_thumbQueuePath = m_currentPath;
        ClearThumbnailQueue();
        return;
    }

    int remaining = std::max(THUMBS_PER_FRAME - m_thumbsLoadedThisFrame, 0);
    double now = m_frameTimeNow;

    while (!m_thumbQueue.empty() && remaining > 0) {
        auto req = m_thumbQueue.front();
        m_thumbQueue.pop_front();
        m_thumbQueueKeys.erase(req);

        if (req.dirPath != m_currentPath)
            continue;

        std::error_code ec;
        if (!fs::exists(fs::u8path(req.filePath), ec))
            continue;

        std::string retryKey = req.kind + "|" + req.filePath;
        auto retryIt = m_thumbRetryAfter.find(retryKey);
        if (retryIt != m_thumbRetryAfter.end() && retryIt->second > now)
            continue;

        if (req.kind == "image") {
            int64_t mtimeNs = GetMtimeNs(req.filePath);
            if (mtimeNs == 0)
                continue;

            std::string thumbName = "__thumb__" + req.filePath;
            auto cIt = m_thumbnailCache.find(req.filePath);
            if (cIt != m_thumbnailCache.end() && cIt->second.mtimeNs == mtimeNs && cIt->second.texId != 0)
                continue;

            if (m_renderer->HasImGuiTexture(thumbName)) {
                auto texId = m_renderer->GetImGuiTextureId(thumbName);
                if (texId != 0) {
                    m_thumbnailCache[req.filePath] = {texId, mtimeNs};
                    continue;
                }
            }

            std::vector<unsigned char> pixels;
            int w = 0, h = 0;
            if (!DownsampleTexture(req.filePath, THUMBNAIL_MAX_PX, pixels, w, h)) {
                m_thumbRetryAfter[retryKey] = now + THUMB_RETRY_DELAY;
                continue;
            }

            auto texId = m_renderer->UploadTextureForImGui(thumbName, pixels.data(), w, h);
            if (texId == 0) {
                m_thumbRetryAfter[retryKey] = now + THUMB_RETRY_DELAY;
                continue;
            }

            m_thumbnailCache[req.filePath] = {texId, mtimeNs};
        } else if (req.kind == "material") {
            int64_t mtimeNs = GetMaterialMtimeNs(req.filePath);
            if (mtimeNs == 0)
                continue;

            std::string thumbName = "__mat_thumb__" + req.filePath;
            auto cIt = m_thumbnailCache.find(req.filePath);
            if (cIt != m_thumbnailCache.end()) {
                if (cIt->second.mtimeNs == mtimeNs && cIt->second.texId != 0)
                    continue;
            }

            std::vector<unsigned char> pixels;
            if (!MaterialPreviewer::RenderToPixels(req.filePath, THUMBNAIL_MAX_PX, pixels, m_assetDatabase,
                                                   m_renderer)) {
                m_thumbRetryAfter[retryKey] = now + THUMB_RETRY_DELAY;
                continue;
            }

            int sq = THUMBNAIL_MAX_PX;
            auto texId = m_renderer->UploadTextureForImGui(thumbName, pixels.data(), sq, sq);
            if (texId == 0) {
                m_thumbRetryAfter[retryKey] = now + THUMB_RETRY_DELAY;
                continue;
            }

            m_thumbnailCache[req.filePath] = {texId, mtimeNs};
        } else {
            continue;
        }

        m_thumbRetryAfter.erase(retryKey);
        ++m_thumbsLoadedThisFrame;
        --remaining;
    }
}

// ════════════════════════════════════════════════════════════════════
// File-type icon system
// ════════════════════════════════════════════════════════════════════

void ProjectPanel::EnsureTypeIconsLoaded()
{
    if (m_typeIconsLoaded || !m_renderer || m_iconsDir.empty())
        return;

    std::unordered_set<std::string> needed;
    for (auto &[_, iconName] : GetIconMap())
        needed.insert(iconName);
    needed.insert("file"); // generic fallback

    std::error_code ec;
    for (auto &iconKey : needed) {
        std::string texName = "__typeicon__" + iconKey;
        if (m_renderer->HasImGuiTexture(texName)) {
            m_typeIconCache[iconKey] = m_renderer->GetImGuiTextureId(texName);
            continue;
        }

        auto iconPath = (fs::u8path(m_iconsDir) / (iconKey + ".png")).string();
        if (!fs::is_regular_file(iconPath, ec))
            continue;

        auto texData = InxTextureLoader::LoadFromFile(iconPath);
        if (!texData.IsValid())
            continue;

        auto tid = m_renderer->UploadTextureForImGui(texName, texData.pixels.data(), texData.width, texData.height);
        if (tid != 0)
            m_typeIconCache[iconKey] = tid;
    }

    m_typeIconsLoaded = true;
}

uint64_t ProjectPanel::GetTypeIconId(const FileItem &item) const
{
    const std::string *key = nullptr;
    static const std::string fallbackKey = "file";
    auto &iconMap = GetIconMap();

    if (item.type == FileItem::Dir) {
        auto sit = iconMap.find("__dir__");
        key = sit != iconMap.end() ? &sit->second : &fallbackKey;
    } else if (item.type == FileItem::SubMesh) {
        auto sit = iconMap.find(".fbx");
        key = sit != iconMap.end() ? &sit->second : &fallbackKey;
    } else if (item.type == FileItem::SubMaterial) {
        auto sit = iconMap.find(".mat");
        key = sit != iconMap.end() ? &sit->second : &fallbackKey;
    } else {
        auto mapIt = iconMap.find(item.ext);
        key = mapIt != iconMap.end() ? &mapIt->second : &fallbackKey;
    }

    auto it = m_typeIconCache.find(*key);
    return it != m_typeIconCache.end() ? it->second : 0;
}

// ════════════════════════════════════════════════════════════════════
// Label layout cache
// ════════════════════════════════════════════════════════════════════

float ProjectPanel::GetGridTextLineHeight(InxGUIContext *ctx)
{
    if (m_gridTextLineHeight <= 0.0f)
        m_gridTextLineHeight = std::max(ImGui::GetTextLineHeight(), 14.0f);
    return m_gridTextLineHeight;
}

const ProjectPanel::LabelEntry &ProjectPanel::GetCachedItemLabel(InxGUIContext *ctx, const FileItem &item,
                                                                 float textRegionW)
{
    bool isExpanded = (item.type == FileItem::File && IsModelExt(item.ext) && m_expandedModels.count(item.path) > 0);

    LabelCacheKey key;
    key.path = item.path;
    key.name = item.name;
    key.type = static_cast<uint8_t>(item.type);
    key.expanded = isExpanded;
    key.widthPx = static_cast<int>(textRegionW);

    auto it = m_labelCache.find(key);
    if (it != m_labelCache.end())
        return it->second;

    // Build display name
    std::string nameDisplay = item.name;
    if (item.type == FileItem::File) {
        auto dot = nameDisplay.rfind('.');
        if (dot != std::string::npos)
            nameDisplay = nameDisplay.substr(0, dot);
        if (IsModelExt(item.ext)) {
            nameDisplay += "  ";
            nameDisplay += isExpanded ? kExpandedArrow : kCollapsedArrow;
        }
    } else if (item.type == FileItem::SubMesh || item.type == FileItem::SubMaterial) {
        nameDisplay = std::string(kSubAssetPrefix) + item.name;
    }

    float maxTextW = textRegionW - 4.0f;
    float textW = ctx->CalcTextWidth(nameDisplay);
    if (textW > maxTextW) {
        // Truncate with ellipsis
        std::string truncated = nameDisplay;
        while (truncated.size() > 1) {
            truncated.pop_back();
            float tw = ctx->CalcTextWidth(truncated + kEllipsis);
            if (tw <= maxTextW) {
                nameDisplay = truncated + kEllipsis;
                textW = tw;
                break;
            }
        }
        if (truncated.size() <= 1) {
            nameDisplay = kEllipsis;
            textW = ctx->CalcTextWidth(nameDisplay);
        }
    }

    float offsetX = std::max((textRegionW - textW) * 0.5f, 0.0f);

    LabelEntry entry{std::move(nameDisplay), offsetX};
    if (m_labelCache.size() > 4096)
        m_labelCache.clear();
    auto [insertIt, _] = m_labelCache.emplace(std::move(key), std::move(entry));
    return insertIt->second;
}

// ════════════════════════════════════════════════════════════════════
// Grid layout — virtual scrolling
// ════════════════════════════════════════════════════════════════════

ProjectPanel::GridRange ProjectPanel::GetVisibleGridRange(InxGUIContext *ctx, int itemCount, int cols, float rowHeight,
                                                          float startY) const
{
    if (itemCount <= 0 || cols <= 0 || rowHeight <= 0.0f)
        return {0, itemCount, 0.0f, 0.0f};

    float scrollY = std::max(ctx->GetScrollY() - startY, 0.0f);
    float viewportH = std::max(ctx->GetContentRegionAvailHeight(), rowHeight);
    int totalRows = (itemCount + cols - 1) / cols;
    int firstRow = std::max(static_cast<int>(scrollY / rowHeight), 0);
    int visibleRows = std::max(static_cast<int>(viewportH / rowHeight) + 2, 1);
    int lastRow = std::min(totalRows, firstRow + visibleRows);

    GridRange r;
    r.topSpacer = firstRow * rowHeight;
    r.bottomSpacer = std::max(totalRows - lastRow, 0) * rowHeight;
    r.startIndex = firstRow * cols;
    r.endIndex = std::min(itemCount, lastRow * cols);
    return r;
}

// ════════════════════════════════════════════════════════════════════
// Click & keyboard input
// ════════════════════════════════════════════════════════════════════

bool ProjectPanel::IsCtrl(InxGUIContext *ctx) const
{
    return ctx->IsKeyDown(kKeyLeftCtrl) || ctx->IsKeyDown(kKeyRightCtrl);
}

bool ProjectPanel::IsShift(InxGUIContext *ctx) const
{
    return ctx->IsKeyDown(kKeyLeftShift) || ctx->IsKeyDown(kKeyRightShift);
}

void ProjectPanel::HandleItemClick(const FileItem &item, InxGUIContext *ctx)
{
    double now = m_frameTimeNow;
    bool doubleClicked = (m_lastClickedFile == item.path && (now - m_lastClickTime) < 0.4);
    m_lastClickedFile = item.path;
    m_lastClickTime = now;

    bool ctrl = IsCtrl(ctx);
    bool shift = IsShift(ctx);

    if (ctrl && !doubleClicked) {
        auto it = std::find(m_selectedFiles.begin(), m_selectedFiles.end(), item.path);
        if (it != m_selectedFiles.end()) {
            m_selectedFiles.erase(it);
            m_selectedFile = m_selectedFiles.empty() ? "" : m_selectedFiles.back();
        } else {
            m_selectedFiles.push_back(item.path);
            m_selectedFile = item.path;
        }
        m_selectedSet.clear();
        m_selectedSet.insert(m_selectedFiles.begin(), m_selectedFiles.end());
        NotifySelectionChanged();
        return;
    }

    if (shift && !doubleClicked && !m_selectedFile.empty() && m_visibleItems) {
        int anchorIdx = -1, targetIdx = -1;
        for (int i = 0; i < static_cast<int>(m_visibleItems->size()); ++i) {
            auto &vi = (*m_visibleItems)[i];
            if (vi.path == m_selectedFile)
                anchorIdx = i;
            if (vi.path == item.path)
                targetIdx = i;
        }
        if (anchorIdx >= 0 && targetIdx >= 0) {
            int lo = std::min(anchorIdx, targetIdx);
            int hi = std::max(anchorIdx, targetIdx);
            m_selectedFiles.clear();
            for (int i = lo; i <= hi; ++i)
                m_selectedFiles.push_back((*m_visibleItems)[i].path);
            m_selectedSet.clear();
            m_selectedSet.insert(m_selectedFiles.begin(), m_selectedFiles.end());
            NotifySelectionChanged();
            return;
        }
    }

    // Normal single select
    m_selectedFiles = {item.path};
    m_selectedFile = item.path;
    m_selectedSet = {item.path};
    NotifySelectionChanged();

    if (item.type == FileItem::Dir) {
        if (doubleClicked) {
            m_currentPath = item.path;
            m_lastClickedFile.clear();
        }
    } else if (item.type == FileItem::SubMesh || item.type == FileItem::SubMaterial) {
        // Sub-assets: select only
    } else if (doubleClicked) {
        if (IsModelExt(item.ext)) {
            if (m_expandedModels.count(item.path) > 0)
                m_expandedModels.erase(item.path);
            else
                m_expandedModels.insert(item.path);
        } else if (item.ext == ".scene") {
            if (openScene)
                openScene(item.path);
        } else if (item.ext == ".prefab") {
            if (openPrefabMode)
                openPrefabMode(item.path);
        } else if (item.ext == ".animclip2d") {
            if (openAnimClip)
                openAnimClip(item.path);
        } else if (item.ext == ".animfsm") {
            if (openAnimFsm)
                openAnimFsm(item.path);
        } else {
            if (openFile)
                openFile(item.path);
        }
    }
}

void ProjectPanel::HandleKeyboardShortcuts(InxGUIContext *ctx)
{
    if (!m_renamingPath.empty())
        return;

    bool ctrl = IsCtrl(ctx);
    bool shift = IsShift(ctx);

    // Ctrl+Shift+N: create new folder (no selection required)
    if (ctrl && shift && ctx->IsKeyPressed(kKeyN)) {
        CreateAndRename("NewFolder", "", [this](const std::string &name) {
            if (createFolder)
                return createFolder(m_currentPath, name);
            return std::make_pair(false, std::string("No callback"));
        });
        return;
    }

    // Early out: avoid GetSelectedPaths() syscalls when no key is pressed
    bool anyRelevantKey = ctx->IsKeyPressed(kKeyF2) || ctx->IsKeyPressed(kKeyDelete) ||
                          (ctrl && (ctx->IsKeyPressed(kKeyC) || ctx->IsKeyPressed(kKeyX) || ctx->IsKeyPressed(kKeyV)));
    if (!anyRelevantKey)
        return;

    auto selected = GetSelectedPaths();
    bool hasSel = !selected.empty();
    bool singleSel = (selected.size() == 1 && !m_selectedFile.empty());

    if (hasSel) {
        if (ctx->IsKeyPressed(kKeyF2) && singleSel)
            BeginRename(m_selectedFile);
        else if (ctx->IsKeyPressed(kKeyDelete)) {
            if (deleteItems)
                deleteItems(selected);
            m_pendingCacheInvalidation = true;
            m_selectedFile.clear();
            m_selectedFiles.clear();
            m_selectedSet.clear();
            NotifySelectionChanged();
        } else if (ctrl && ctx->IsKeyPressed(kKeyC))
            ClipboardCopy(selected);
        else if (ctrl && ctx->IsKeyPressed(kKeyX))
            ClipboardCut(selected);
        else if (ctrl && ctx->IsKeyPressed(kKeyV))
            ClipboardPaste();
    } else {
        if (ctrl && ctx->IsKeyPressed(kKeyV))
            ClipboardPaste();
    }
}

void ProjectPanel::HandleExternalFileDrops()
{
    // This is handled via callback from Python's InputManager binding
    // since InputManager singleton access is simpler from Python.
    // The Python bootstrap wiring handles this.
}

void ProjectPanel::ReceiveDroppedFiles(const std::vector<std::string> &paths)
{
    if (paths.empty() || m_currentPath.empty())
        return;

    std::error_code ec;
    std::vector<std::string> copiedPaths;

    for (auto &src : paths) {
        if (src.empty() || !fs::exists(fs::u8path(src), ec))
            continue;

        auto name = fs::u8path(src).filename().string();
        auto dst = (fs::u8path(m_currentPath) / name).string();

        // If destination already exists, use unique name
        if (fs::exists(fs::u8path(dst), ec)) {
            if (!getUniqueName)
                continue;
            auto stem = fs::u8path(name).stem().string();
            auto ext = fs::u8path(name).extension().string();
            if (fs::is_directory(fs::u8path(src), ec)) {
                ext = "";
                stem = name;
            }
            auto uniqueName = getUniqueName(m_currentPath, stem, ext);
            dst = (fs::u8path(m_currentPath) / (uniqueName + ext)).string();
        }

        try {
            if (fs::is_directory(fs::u8path(src), ec)) {
                fs::copy(fs::u8path(src), fs::u8path(dst), fs::copy_options::recursive, ec);
            } else {
                fs::copy_file(fs::u8path(src), fs::u8path(dst), ec);
            }
            if (!ec)
                copiedPaths.push_back(dst);
        } catch (...) {
            continue;
        }
    }

    if (copiedPaths.empty())
        return;

    m_pendingCacheInvalidation = true;
    m_selectedFiles = copiedPaths;
    m_selectedFile = copiedPaths.back();
    m_selectedSet.clear();
    m_selectedSet.insert(copiedPaths.begin(), copiedPaths.end());
    NotifySelectionChanged();
}

// ════════════════════════════════════════════════════════════════════
// Rename
// ════════════════════════════════════════════════════════════════════

void ProjectPanel::BeginRename(const std::string &path)
{
    m_renamingPath = path;
    auto name = fs::u8path(path).filename().string();
    std::error_code ec;
    if (fs::is_regular_file(path, ec)) {
        auto stem = fs::u8path(path).stem().string();
        if (!stem.empty())
            name = stem;
    }
    std::strncpy(m_renameBuf, name.c_str(), sizeof(m_renameBuf) - 1);
    m_renameBuf[sizeof(m_renameBuf) - 1] = '\0';
    m_renameFocusRequested = true;
}

void ProjectPanel::CommitRename()
{
    if (m_renamingPath.empty() || !doRename) {
        CancelRename();
        return;
    }

    std::string newName(m_renameBuf);
    if (newName.empty()) {
        CancelRename();
        return;
    }

    std::string newPath = doRename(m_renamingPath, newName);
    if (!newPath.empty()) {
        m_pendingCacheInvalidation = true;
        if (m_selectedFile == m_renamingPath) {
            m_selectedFile = newPath;
            m_selectedFiles = {newPath};
            m_selectedSet = {newPath};
        }
        if (m_currentPath == m_renamingPath)
            m_currentPath = newPath;
    }
    m_renamingPath.clear();
}

void ProjectPanel::CancelRename()
{
    m_renamingPath.clear();
}

void ProjectPanel::CreateAndRename(const std::string &baseName, const std::string &extension,
                                   std::function<std::pair<bool, std::string>(const std::string &)> createFn)
{
    if (!getUniqueName || !createFn)
        return;

    std::string name = getUniqueName(m_currentPath, baseName, extension);
    auto [ok, result] = createFn(name);
    if (!ok)
        return;

    std::string fileName = name;
    if (!extension.empty() && fileName.find(extension) == std::string::npos)
        fileName += extension;
    auto newPath = (fs::u8path(m_currentPath) / fileName).string();

    m_selectedFile = newPath;
    m_selectedFiles = {newPath};
    m_selectedSet = {newPath};
    NotifySelectionChanged();

    BeginRename(newPath);
    InvalidateDirCache();
}

// ════════════════════════════════════════════════════════════════════
// Clipboard
// ════════════════════════════════════════════════════════════════════

/// Retrieve file paths from the OS clipboard (CF_HDROP on Windows).
static std::vector<std::string> GetOSClipboardFiles()
{
    std::vector<std::string> result;
#ifdef INX_PLATFORM_WINDOWS
    if (!OpenClipboard(nullptr))
        return result;

    HANDLE hData = GetClipboardData(CF_HDROP);
    if (hData) {
        HDROP hDrop = static_cast<HDROP>(hData);
        UINT fileCount = DragQueryFileW(hDrop, 0xFFFFFFFF, nullptr, 0);
        for (UINT i = 0; i < fileCount; ++i) {
            UINT len = DragQueryFileW(hDrop, i, nullptr, 0);
            if (len == 0)
                continue;
            std::wstring wpath(len + 1, L'\0');
            DragQueryFileW(hDrop, i, wpath.data(), len + 1);
            wpath.resize(len);
            // Convert wstring to UTF-8 via filesystem
            std::error_code ec;
            auto u8str = fs::path(wpath).string();
            if (!u8str.empty())
                result.push_back(std::move(u8str));
        }
    }
    CloseClipboard();
#endif
    return result;
}

void ProjectPanel::ClipboardCopy(const std::vector<std::string> &paths)
{
    m_clipboardPaths.clear();
    std::error_code ec;
    for (auto &p : paths)
        if (!p.empty() && fs::exists(fs::u8path(p), ec))
            m_clipboardPaths.push_back(p);
    m_clipboardIsCut = false;
}

void ProjectPanel::ClipboardCut(const std::vector<std::string> &paths)
{
    m_clipboardPaths.clear();
    std::error_code ec;
    for (auto &p : paths)
        if (!p.empty() && fs::exists(fs::u8path(p), ec))
            m_clipboardPaths.push_back(p);
    m_clipboardIsCut = true;
}

bool ProjectPanel::HasClipboardItems() const
{
    std::error_code ec;
    for (auto &p : m_clipboardPaths)
        if (fs::exists(fs::u8path(p), ec))
            return true;
    return false;
}

void ProjectPanel::ClipboardPaste()
{
    std::error_code ec;
    std::vector<std::string> sources;
    bool isCut = m_clipboardIsCut;

    // Try internal clipboard first
    for (auto &p : m_clipboardPaths)
        if (fs::exists(fs::u8path(p), ec))
            sources.push_back(p);

    // Fall back to OS clipboard (always copy, never cut)
    if (sources.empty()) {
        sources = GetOSClipboardFiles();
        isCut = false;
    }

    if (sources.empty()) {
        m_clipboardPaths.clear();
        return;
    }

    std::vector<std::string> pastedPaths;
    for (auto &src : sources) {
        auto name = fs::u8path(src).filename().string();
        auto dst = (fs::u8path(m_currentPath) / name).string();
        bool samePath = (NormalizePath(src) == NormalizePath(dst));

        if (samePath && isCut)
            continue;

        if (samePath || fs::exists(dst, ec)) {
            if (!getUniqueName)
                continue;
            auto stem = fs::u8path(name).stem().string();
            auto ext = fs::u8path(name).extension().string();
            if (fs::is_directory(src, ec)) {
                ext = "";
                stem = name;
            }
            auto uniqueName = getUniqueName(m_currentPath, stem, ext);
            dst = (fs::u8path(m_currentPath) / (uniqueName + ext)).string();
        }

        try {
            if (isCut) {
                if (moveItemToDirectory) {
                    auto result = moveItemToDirectory(src, m_currentPath);
                    if (!result.empty())
                        pastedPaths.push_back(result);
                } else {
                    fs::rename(fs::u8path(src), fs::u8path(dst), ec);
                    if (!ec)
                        pastedPaths.push_back(dst);
                }
            } else if (fs::is_directory(src, ec)) {
                fs::copy(fs::u8path(src), fs::u8path(dst), fs::copy_options::recursive, ec);
                if (!ec)
                    pastedPaths.push_back(dst);
            } else {
                fs::copy_file(fs::u8path(src), fs::u8path(dst), ec);
                if (!ec)
                    pastedPaths.push_back(dst);
            }
        } catch (...) {
            continue;
        }
    }

    if (pastedPaths.empty())
        return;

    if (isCut)
        m_clipboardPaths.clear();

    m_pendingCacheInvalidation = true;
    m_selectedFiles = pastedPaths;
    m_selectedFile = pastedPaths.back();
    m_selectedSet.clear();
    m_selectedSet.insert(pastedPaths.begin(), pastedPaths.end());
    NotifySelectionChanged();
}

// ════════════════════════════════════════════════════════════════════
// Move helpers
// ════════════════════════════════════════════════════════════════════

std::vector<std::string> ProjectPanel::GetDragMoveSources(const std::string &draggedPath) const
{
    auto selected = GetSelectedPaths();
    std::vector<std::string> candidates;
    if (std::find(selected.begin(), selected.end(), draggedPath) != selected.end())
        candidates = selected;
    else
        candidates = {draggedPath};

    // Remove ancestors
    std::sort(candidates.begin(), candidates.end(), [](const std::string &a, const std::string &b) {
        auto sepA = std::count(a.begin(), a.end(), '\\') + std::count(a.begin(), a.end(), '/');
        auto sepB = std::count(b.begin(), b.end(), '\\') + std::count(b.begin(), b.end(), '/');
        if (sepA != sepB)
            return sepA < sepB;
        return a.size() < b.size();
    });

    std::vector<std::string> kept;
    for (auto &p : candidates) {
        bool subsumed = false;
        for (auto &k : kept) {
            if (IsPathWithin(p, k)) {
                subsumed = true;
                break;
            }
        }
        if (!subsumed)
            kept.push_back(p);
    }
    return kept;
}

std::string ProjectPanel::ResolveMovePayloadPath(const std::string &payloadType, const std::string &payload) const
{
    if (payload.empty())
        return "";

    // Check if it's a path-based type
    auto &ddMap = GetDragDropMap();
    auto &gdMap = GetGuidDragDropMap();
    bool isPathType = (payloadType == DRAG_TYPE_PROJECT_ITEM);
    if (!isPathType) {
        for (auto &[_, info] : ddMap)
            if (info.payloadType == payloadType) {
                isPathType = true;
                break;
            }
    }
    if (!isPathType) {
        for (auto &[_, info] : gdMap)
            if (info.pathPayloadType == payloadType) {
                isPathType = true;
                break;
            }
    }

    if (isPathType) {
        std::error_code ec;
        return fs::exists(fs::u8path(payload), ec) ? payload : "";
    }

    // GUID-based type
    bool isGuidType = false;
    for (auto &[_, info] : gdMap)
        if (info.guidPayloadType == payloadType) {
            isGuidType = true;
            break;
        }

    if (isGuidType) {
        std::string path;
        if (getPathFromGuid)
            path = getPathFromGuid(payload);
        else if (m_assetDatabase)
            path = m_assetDatabase->GetPathFromGuid(payload);

        std::error_code ec;
        return (!path.empty() && fs::exists(fs::u8path(path), ec)) ? path : "";
    }

    return "";
}

void ProjectPanel::MoveProjectItemsToFolder(const std::string &targetDir, const std::string &payloadType,
                                            const std::string &payload)
{
    std::error_code ec;
    if (targetDir.empty() || !fs::is_directory(targetDir, ec))
        return;

    auto draggedPath = ResolveMovePayloadPath(payloadType, payload);
    if (draggedPath.empty())
        return;

    auto sources = GetDragMoveSources(draggedPath);
    // Remove items targeting self
    sources.erase(std::remove_if(sources.begin(), sources.end(),
                                 [&](const std::string &s) { return NormalizePath(s) == NormalizePath(targetDir); }),
                  sources.end());

    if (sources.empty())
        return;

    std::vector<std::string> movedPaths;
    for (auto &source : sources) {
        if (fs::is_directory(source, ec) && IsPathWithin(targetDir, source))
            continue; // Can't move folder into itself

        if (moveItemToDirectory) {
            auto newPath = moveItemToDirectory(source, targetDir);
            if (!newPath.empty())
                movedPaths.push_back(newPath);
        }
    }

    if (movedPaths.empty())
        return;

    m_pendingCacheInvalidation = true;
    m_thumbnailCache.clear();
    m_selectedFiles = movedPaths;
    m_selectedFile = movedPaths.back();
    m_selectedSet.clear();
    m_selectedSet.insert(movedPaths.begin(), movedPaths.end());
    NotifySelectionChanged();
}

// ════════════════════════════════════════════════════════════════════
// PreRender
// ════════════════════════════════════════════════════════════════════

void ProjectPanel::PreRender(InxGUIContext *ctx)
{
    m_frameTimeNow = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    m_thumbsLoadedThisFrame = 0;
    EnsureTypeIconsLoaded();
    ProcessPendingThumbnails();
    GetGridTextLineHeight(ctx);

    if (m_currentPath != m_lastNotifiedPath) {
        m_lastNotifiedPath = m_currentPath;
        if (onStateChanged)
            onStateChanged();
    }
}

// ════════════════════════════════════════════════════════════════════
// OnRenderContent — main entry
// ════════════════════════════════════════════════════════════════════

void ProjectPanel::OnRenderContent(InxGUIContext *ctx)
{
    // Process any deferred cache invalidation from the previous frame
    // (CommitRename, Delete, Paste, Move set this flag to avoid invalidating
    // the items pointer mid-iteration in RenderFileGrid).
    if (m_pendingCacheInvalidation) {
        m_pendingCacheInvalidation = false;
        InvalidateDirCache();
    }

    RenderBreadcrumb(ctx);
    ctx->Separator();

    // Left panel: folder tree (200px)
    if (ctx->BeginChild("FolderTree", 200, 0, false)) {
        RenderFolderTree(ctx);
    }
    ctx->EndChild();

    ctx->SameLine();

    // Right panel: file grid
    ctx->PushStyleVarVec2(ImGuiStyleVar_WindowPadding, 12.0f, 8.0f);
    ctx->PushStyleColor(ImGuiCol_Border, 0.0f, 0.0f, 0.0f, 0.0f); // transparent border
    if (ctx->BeginChild("FileGrid", 0, 0, true)) {
        RenderFileGrid(ctx);
    }
    ctx->EndChild();
    ctx->PopStyleColor(1); // Border
    ctx->PopStyleVar(1);   // WindowPadding

    bool hasSelection = !m_selectedFile.empty() || !m_selectedFiles.empty();
    bool clickedOutsideProject =
        hasSelection &&
        (ImGui::IsMouseClicked(0) || ImGui::IsMouseClicked(1) || ImGui::IsMouseClicked(2)) &&
        !ImGui::IsWindowHovered(ImGuiHoveredFlags_RootAndChildWindows) &&
        !ImGui::IsAnyItemActive();
    if (clickedOutsideProject)
        ClearSelection();
}

// ════════════════════════════════════════════════════════════════════
// Breadcrumb
// ════════════════════════════════════════════════════════════════════

void ProjectPanel::RenderBreadcrumb(InxGUIContext *ctx)
{
    if (m_currentPath != m_breadcrumbPath) {
        m_breadcrumbPath = m_currentPath;
        if (!m_rootPath.empty()) {
            auto rel = fs::relative(fs::u8path(m_currentPath), fs::u8path(m_rootPath));
            auto relStr = rel.string();
            if (relStr == ".")
                relStr = fs::u8path(m_rootPath).filename().string();
            m_breadcrumbText = "Path: " + relStr;
        } else {
            m_breadcrumbText = "Path: " + m_currentPath;
        }
    }
    ctx->Label(m_breadcrumbText);
}

// ════════════════════════════════════════════════════════════════════
// Folder tree
// ════════════════════════════════════════════════════════════════════

void ProjectPanel::RenderFolderTree(InxGUIContext *ctx)
{
    auto *rootSnap = m_rootPath.empty() ? nullptr : GetDirSnapshot(m_rootPath);
    if (rootSnap) {
        auto projectName = fs::u8path(m_rootPath).filename().string();
        int rootFlags =
            ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_FramePadding;
        if (m_currentPath == m_rootPath)
            rootFlags |= ImGuiTreeNodeFlags_Selected;

        ctx->SetNextItemOpen(true, ImGuiCond_FirstUseEver);
        bool nodeOpen = ctx->TreeNodeEx((projectName + "###" + m_rootPath).c_str(), rootFlags);
        if (ctx->IsItemClicked())
            m_currentPath = m_rootPath;
        if (nodeOpen) {
            RenderFolderTreeRecursive(ctx, m_rootPath, rootSnap);
            ctx->TreePop();
        }
    } else {
        ctx->Label(Tr("project.no_project_path"));
    }

    float remainH = ctx->GetContentRegionAvailHeight();
    if (remainH > 4.0f) {
        ctx->InvisibleButton("##folder_tree_empty_area", ctx->GetContentRegionAvailWidth(), remainH);
        if (ctx->IsItemClicked(0)) {
            ClearSelection();
            NotifyEmptyAreaClicked();
        }
    }
}

void ProjectPanel::RenderFolderTreeRecursive(InxGUIContext *ctx, const std::string &path, DirSnapshot *snapshot)
{
    if (!snapshot)
        snapshot = GetDirSnapshot(path);
    if (!snapshot)
        return;

    for (auto &d : snapshot->dirs) {
        int flags =
            ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_FramePadding;
        if (m_currentPath == d.path)
            flags |= ImGuiTreeNodeFlags_Selected;

        auto *meta = GetDirTreeMeta(d.path);
        bool hasSubdirs = meta && meta->hasSubdirs;
        if (!hasSubdirs)
            flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;

        ImGui::PushID(d.path.c_str());
        bool open = ctx->TreeNodeEx(d.name.c_str(), flags);
        if (ctx->IsItemClicked())
            m_currentPath = d.path;
        if (hasSubdirs && open) {
            RenderFolderTreeRecursive(ctx, d.path);
            ctx->TreePop();
        }
        ImGui::PopID();
    }
}

// ════════════════════════════════════════════════════════════════════
// File grid
// ════════════════════════════════════════════════════════════════════

void ProjectPanel::RenderFileGrid(InxGUIContext *ctx)
{
    RenderContextMenu(ctx);

    auto *snapshot = m_currentPath.empty() ? nullptr : GetDirSnapshot(m_currentPath);
    if (!snapshot) {
        ctx->Label(Tr("project.invalid_path"));
        return;
    }

    auto *items = GetProjectItems(m_currentPath, snapshot);
    if (!items)
        return;

    // Back button
    auto parent = fs::u8path(m_currentPath).parent_path().string();
    if (m_currentPath != m_rootPath && parent != m_rootPath) {
        if (ctx->Selectable("[..]", false))
            m_currentPath = parent;
    }

    // Grid config
    float iconSize = static_cast<float>(ICON_SIZE);
    float avail_w = ctx->GetContentRegionAvailWidth();
    int cols = std::max(static_cast<int>(avail_w / CELL_WIDTH), 1);
    float rowHeight = iconSize + GetGridTextLineHeight(ctx) + GRID_PADDING + 8.0f;

    if (items->empty() && m_currentPath == m_rootPath) {
        ctx->Label(Tr("project.empty_folder"));
        ctx->Label(Tr("project.right_click_hint"));
    }

    // Keyboard shortcuts
    HandleKeyboardShortcuts(ctx);

    // Virtual scrolling
    float gridStartX = ctx->GetCursorPosX();
    float gridStartY = ctx->GetCursorPosY();
    int itemCount = static_cast<int>(items->size());
    auto range = GetVisibleGridRange(ctx, itemCount, cols, rowHeight, gridStartY);
    const ImGuiPayload *dragPayload = ImGui::GetDragDropPayload();
    bool hasDragPayload = (dragPayload != nullptr);
    bool hasHierarchyDragPayload = hasDragPayload && dragPayload->IsDataType(DRAG_TYPE_HIERARCHY_GO);

    if (ctx->BeginTable("FileGrid", cols, 0, 0.0f)) {
        m_visibleItems = items;

        if (range.topSpacer > 0.0f) {
            ctx->TableNextRow();
            ctx->TableSetColumnIndex(0);
            ctx->Dummy(1.0f, range.topSpacer);
            ctx->TableNextRow();
        }

        ImDrawList *drawList = ImGui::GetWindowDrawList();
        float cellW = static_cast<float>(CELL_WIDTH);

        for (int i = range.startIndex; i < range.endIndex; ++i) {
            auto &item = (*items)[i];
            ctx->TableNextColumn();
            ImGui::PushID(i);

            bool isSelected = m_selectedSet.count(item.path) > 0;
            // Record cell start position for full-cell drop overlay later
            // ── Resolve display texture (inline for speed) ──
            uint64_t displayTexId = 0;
            if (item.type == FileItem::SubMesh) {
                auto tic = m_typeIconCache.find("model_3d");
                displayTexId = tic != m_typeIconCache.end() ? tic->second : 0;
            } else if (item.type == FileItem::SubMaterial) {
                auto tic = m_typeIconCache.find("material");
                displayTexId = tic != m_typeIconCache.end() ? tic->second : 0;
            } else if (item.type == FileItem::File) {
                if (IsImageExt(item.ext))
                    displayTexId = GetThumbnail(item.path, item.mtimeNs);
                else if (IsMaterialExt(item.ext))
                    displayTexId = GetMaterialThumbnail(item.path);
                if (displayTexId == 0)
                    displayTexId = GetTypeIconId(item);
            } else {
                displayTexId = GetTypeIconId(item);
            }

            // ── Render icon ──
            if (displayTexId != 0) {
                // InvisibleButton for hit-testing; AddImage for drawing
                // Much cheaper than ImageButton (no style/border processing)
                ImGui::InvisibleButton("##ic", ImVec2(iconSize, iconSize));
                ImVec2 rMin = ImGui::GetItemRectMin();
                ImVec2 rMax = ImGui::GetItemRectMax();
                drawList->AddImage(ImTextureRef(static_cast<ImTextureID>(displayTexId)), rMin, rMax);
                // Select on mouse RELEASE (not press) so that press-and-drag
                // initiates drag-drop instead of changing the selection.
                if (ImGui::IsItemHovered() && ImGui::IsMouseReleased(0) && !hasDragPayload)
                    HandleItemClick(item, ctx);
                if (ImGui::IsItemClicked(1)) {
                    m_selectedFile = item.path;
                    if (m_selectedSet.count(item.path) == 0) {
                        m_selectedFiles = {item.path};
                        m_selectedSet = {item.path};
                    }
                    NotifySelectionChanged();
                }
                if (isSelected)
                    drawList->AddRectFilled(rMin, rMax, IM_COL32(235, 87, 87, 140));
            } else {
                const char *tag = (item.type != FileItem::Dir) ? GetFileTypeTag(item.name) : "[DIR]";
                ctx->Selectable(tag, isSelected, 0, iconSize, iconSize);
                if (ImGui::IsItemHovered() && ImGui::IsMouseReleased(0) && !hasDragPayload)
                    HandleItemClick(item, ctx);
                if (ctx->IsItemClicked(1)) {
                    m_selectedFile = item.path;
                    if (m_selectedSet.count(item.path) == 0) {
                        m_selectedFiles = {item.path};
                        m_selectedSet = {item.path};
                    }
                    NotifySelectionChanged();
                }
            }

            // ── Drag-drop source (must always run to detect drag start) ──
            RenderDragDropSource(ctx, item);

            // ── Drop targets (only when a drag is active) ──
            if (hasDragPayload && item.type == FileItem::Dir) {
                RenderFolderDropTarget(ctx, item.path);
            }

            // ── Label ──
            {
                float cellStartX = ImGui::GetItemRectMin().x - ImGui::GetWindowPos().x + ImGui::GetScrollX();
                RenderItemLabel(ctx, item, iconSize, cellStartX);
            }

            ImGui::PopID();
        }

        if (range.bottomSpacer > 0.0f) {
            ctx->TableNextRow();
            ctx->TableSetColumnIndex(0);
            ctx->Dummy(1.0f, range.bottomSpacer);
        }

        ctx->EndTable();
    }

    // Full-grid hierarchy drop target (covers entire FileGrid child window)
    if (hasHierarchyDragPayload) {
        ImGuiWindow *win = ImGui::GetCurrentWindow();
        if (ImGui::BeginDragDropTargetCustom(win->InnerRect, win->ID)) {
            uint64_t objId = 0;
            if (ctx->AcceptDragDropPayload(DRAG_TYPE_HIERARCHY_GO, &objId)) {
                if (createPrefabFromHierarchy)
                    createPrefabFromHierarchy(objId, m_currentPath);
            }
            ImGui::EndDragDropTarget();
        }
    }

    // Bottom empty area: click to deselect + accept hierarchy drops
    float remainH = ctx->GetContentRegionAvailHeight();
    if (remainH > 10.0f) {
        ctx->InvisibleButton("##drop_prefab_area", ctx->GetContentRegionAvailWidth(), remainH);
        if (ctx->IsItemClicked(0)) {
            ClearSelection();
            NotifyEmptyAreaClicked();
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// Context menu
// ════════════════════════════════════════════════════════════════════

void ProjectPanel::RenderContextMenu(InxGUIContext *ctx)
{
    if (!ctx->BeginPopupContextWindow("ProjectContextMenu", 1))
        return;

    if (ctx->BeginMenu(Tr("project.create_menu"))) {
        if (ctx->Selectable(Tr("project.create_folder"), false)) {
            CreateAndRename("NewFolder", "", [this](const std::string &name) {
                if (createFolder)
                    return createFolder(m_currentPath, name);
                return std::make_pair(false, std::string("No callback"));
            });
        }
        ctx->Separator();
        if (ctx->Selectable(Tr("project.create_script"), false)) {
            CreateAndRename("NewComponent", ".py", [this](const std::string &name) {
                if (createScript)
                    return createScript(m_currentPath, name);
                return std::make_pair(false, std::string("No callback"));
            });
        }
        ctx->Separator();
        if (ctx->Selectable(Tr("project.create_vert_shader"), false)) {
            CreateAndRename("NewShader", ".vert", [this](const std::string &name) {
                if (createShader)
                    return createShader(m_currentPath, name, "vert");
                return std::make_pair(false, std::string("No callback"));
            });
        }
        if (ctx->Selectable(Tr("project.create_frag_shader"), false)) {
            CreateAndRename("NewShader", ".frag", [this](const std::string &name) {
                if (createShader)
                    return createShader(m_currentPath, name, "frag");
                return std::make_pair(false, std::string("No callback"));
            });
        }
        ctx->Separator();
        if (ctx->Selectable(Tr("project.create_material"), false)) {
            CreateAndRename("NewMaterial", ".mat", [this](const std::string &name) {
                if (createMaterial)
                    return createMaterial(m_currentPath, name);
                return std::make_pair(false, std::string("No callback"));
            });
        }
        ctx->Separator();
        if (ctx->Selectable(Tr("project.create_scene"), false)) {
            CreateAndRename("NewScene", ".scene", [this](const std::string &name) {
                if (createScene)
                    return createScene(m_currentPath, name);
                return std::make_pair(false, std::string("No callback"));
            });
        }
        ctx->EndMenu();
    }

    std::error_code ec;
    if (!m_selectedFile.empty() && fs::exists(fs::u8path(m_selectedFile), ec)) {
        ctx->Separator();
        if (ctx->Selectable(Tr("project.reveal_in_explorer"), false)) {
            if (revealInExplorer)
                revealInExplorer(m_selectedFile);
        }
        ctx->Separator();
        auto selectedPaths = GetSelectedPaths();
        if (ctx->Selectable(Tr("project.copy"), false))
            ClipboardCopy(selectedPaths);
        if (ctx->Selectable(Tr("project.cut"), false))
            ClipboardCut(selectedPaths);
        if (HasClipboardItems()) {
            if (ctx->Selectable(Tr("project.paste"), false))
                ClipboardPaste();
        }
        ctx->Separator();
        bool canRename = (selectedPaths.size() == 1);
        if (!canRename)
            ctx->BeginDisabled();
        if (ctx->Selectable(Tr("project.rename"), false))
            BeginRename(m_selectedFile);
        if (!canRename)
            ctx->EndDisabled();
        if (ctx->Selectable(Tr("project.delete"), false)) {
            if (deleteItems)
                deleteItems(selectedPaths);
            InvalidateDirCache();
            if (std::find(m_selectedFiles.begin(), m_selectedFiles.end(), m_selectedFile) != m_selectedFiles.end()) {
                m_selectedFile.clear();
                m_selectedFiles.clear();
                m_selectedSet.clear();
                NotifySelectionChanged();
            }
        }
    } else {
        ctx->Separator();
        if (ctx->Selectable(Tr("project.reveal_in_explorer"), false)) {
            if (revealInExplorer)
                revealInExplorer(m_currentPath);
        }
        if (HasClipboardItems()) {
            if (ctx->Selectable(Tr("project.paste"), false))
                ClipboardPaste();
        }
    }

    ctx->EndPopup();
}

void ProjectPanel::RenderDragDropSource(InxGUIContext *ctx, const FileItem &item)
{
    if (item.type != FileItem::Dir && item.type != FileItem::File)
        return;

    // BeginDragDropSource is cheap (~1µs) — returns false 99.9% of the time.
    // All map lookups and string formatting are deferred to the rare drag-active path.
    if (!ctx->BeginDragDropSource(0))
        return;

    if (item.type == FileItem::Dir) {
        ctx->SetDragDropPayload(DRAG_TYPE_PROJECT_ITEM, item.path);
        ctx->Label("Folder: " + item.name);
        ctx->EndDragDropSource();
        return;
    }

    // File type — resolve drag payload only when actually dragging
    auto &ddMap = GetDragDropMap();
    auto &gdMap = GetGuidDragDropMap();
    auto ddIt = ddMap.find(item.ext);
    auto gdIt = gdMap.find(item.ext);

    if (ddIt != ddMap.end()) {
        const char *pType = ddIt->second.payloadType;
        const char *labelPfx = ddIt->second.label;

        if (item.ext == ".py" && validateScriptComponent) {
            if (!validateScriptComponent(item.path)) {
                pType = DRAG_TYPE_PROJECT_ITEM;
                labelPfx = "Item (script file not attachable)";
            }
        }

        ctx->SetDragDropPayload(pType, item.path);
        ctx->Label(std::string(labelPfx) + ": " + item.name);
    } else if (gdIt != gdMap.end()) {
        std::string guid;
        if (getGuidFromPath)
            guid = getGuidFromPath(item.path);
        else if (m_assetDatabase)
            guid = m_assetDatabase->GetGuidFromPath(item.path);

        if (!guid.empty())
            ctx->SetDragDropPayload(gdIt->second.guidPayloadType, guid);
        else
            ctx->SetDragDropPayload(gdIt->second.pathPayloadType, item.path);

        ctx->Label(std::string(gdIt->second.label) + ": " + item.name);
    } else {
        ctx->SetDragDropPayload(DRAG_TYPE_PROJECT_ITEM, item.path);
        ctx->Label("Item: " + item.name);
    }

    ctx->EndDragDropSource();
}

void ProjectPanel::RenderFolderDropTarget(InxGUIContext *ctx, const std::string &folderPath)
{
    ImGui::PushStyleColor(ImGuiCol_DragDropTarget, ImVec4(0, 0, 0, 0));
    if (ctx->BeginDragDropTarget()) {
        bool handled = false;
        uint64_t objId = 0;
        if (ctx->AcceptDragDropPayload(DRAG_TYPE_HIERARCHY_GO, &objId)) {
            if (createPrefabFromHierarchy)
                createPrefabFromHierarchy(objId, folderPath);
            handled = true;
        }
        if (!handled) {
            auto &acceptTypes = GetMoveAcceptTypes();
            for (auto &dt : acceptTypes) {
                std::string payload;
                if (ctx->AcceptDragDropPayload(dt, &payload)) {
                    MoveProjectItemsToFolder(folderPath, dt, payload);
                    break;
                }
            }
        }
        ctx->EndDragDropTarget();
    }
    ImGui::PopStyleColor(1);
}

void ProjectPanel::RenderItemLabel(InxGUIContext *ctx, const FileItem &item, float iconSize, float cellStartX)
{
    if (m_renamingPath == item.path) {
        if (m_renameFocusRequested) {
            ctx->SetKeyboardFocusHere();
            m_renameFocusRequested = false;
        }

        ctx->SetCursorPosX(cellStartX);
        ctx->SetNextItemWidth(iconSize);
        ctx->TextInput("##rename_" + item.path, m_renameBuf, sizeof(m_renameBuf));

        if (ctx->IsKeyPressed(kKeyEnter))
            CommitRename();
        else if (ctx->IsKeyPressed(kKeyEscape))
            CancelRename();
        else if (ctx->IsItemDeactivated())
            CommitRename();
    } else {
        auto &entry = GetCachedItemLabel(ctx, item, iconSize);
        ctx->SetCursorPosX(cellStartX + entry.offsetX);
        ctx->Label(entry.displayText);
    }
}

} // namespace infernux
