#include "ProjectPanel.h"

#include "Infernux.h"

#include <function/renderer/gui/InxResourcePreviewer.h>
#include <platform/filesystem/InxPath.h>

#include <algorithm>
#include <any>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <imgui_internal.h>
#include <unordered_set>

#ifdef INX_PLATFORM_WINDOWS
#include <ShlObj.h> // CF_HDROP, DragQueryFileW
#include <shellapi.h>
#include <strsafe.h>
#endif

namespace fs = std::filesystem;

namespace
{
constexpr const char *kSubMatToken = "::submat:";
constexpr const char *kSubBoneToken = "::subbone:";
constexpr const char *kSubAnimToken = "::subanim:";

static std::string TrimCopy(std::string s)
{
    auto notSpace = [](unsigned char ch) { return !std::isspace(ch); };
    while (!s.empty() && !notSpace(static_cast<unsigned char>(s.front())))
        s.erase(s.begin());
    while (!s.empty() && !notSpace(static_cast<unsigned char>(s.back())))
        s.pop_back();
    return s;
}

/// "Armature | Action" from importers → display / row id "Armature" only.
static std::string StripPipeDisplaySuffix(const std::string &s)
{
    auto pos = s.find(" | ");
    if (pos == std::string::npos)
        return s;
    return TrimCopy(s.substr(0, pos));
}

static std::vector<std::string> SplitCommaList(const std::string &csv)
{
    std::vector<std::string> out;
    std::string cur;
    for (char ch : csv) {
        if (ch == ',') {
            auto t = TrimCopy(cur);
            if (!t.empty())
                out.push_back(std::move(t));
            cur.clear();
        } else {
            cur.push_back(ch);
        }
    }
    auto t = TrimCopy(cur);
    if (!t.empty())
        out.push_back(std::move(t));
    return out;
}

static std::string MakeSubAssetVirtualPath(const std::string &basePath, const char *token, int index)
{
    return basePath + token + std::to_string(index);
}

static bool IsVirtualSubAssetPath(const std::string &path)
{
    return path.find(kSubMatToken) != std::string::npos || path.find(kSubBoneToken) != std::string::npos ||
           path.find(kSubAnimToken) != std::string::npos;
}

static std::string ResolveRealAssetPath(const std::string &path)
{
    if (path.empty())
        return path;
    for (const char *tok : {kSubMatToken, kSubBoneToken, kSubAnimToken}) {
        auto pos = path.find(tok);
        if (pos != std::string::npos)
            return path.substr(0, pos);
    }
    return path;
}

#ifdef INX_PLATFORM_WINDOWS
static std::string Utf8FromWidePath(const std::wstring &wpath)
{
    return infernux::FromFsPath(fs::path(wpath));
}
#endif

static std::string SelectionPathForInspector(const std::string &path)
{
    if (path.empty())
        return path;
    // Embedded material slots use the material inspector (Python + virtual path).
    if (path.find(kSubMatToken) != std::string::npos)
        return path;
    // Embedded animation takes use the 3D clip inspector (Python + virtual path).
    if (path.find(kSubAnimToken) != std::string::npos)
        return path;
    return ResolveRealAssetPath(path);
}

/// True if the mouse is over the docked/floating Inspector window (screen space).
/// Prevents Project panel from clearing file selection when clicking empty Inspector space.
bool IsMouseOverInspectorWindow()
{
    ImGuiWindow *win = ImGui::FindWindowByName("Inspector###inspector");
    if (win == nullptr || win->Hidden)
        return false;
    const ImVec2 mp = ImGui::GetIO().MousePos;
    const float x0 = win->Pos.x;
    const float y0 = win->Pos.y;
    const float x1 = x0 + win->SizeFull.x;
    const float y1 = y0 + win->SizeFull.y;
    return mp.x >= x0 && mp.x <= x1 && mp.y >= y0 && mp.y <= y1;
}
} // namespace

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

namespace infernux
{

namespace
{
inline ImU32 ProjectSelectionOutlineColor()
{
    return IM_COL32(static_cast<int>(EditorTheme::ACCENT_R * 255.0f), static_cast<int>(EditorTheme::ACCENT_G * 255.0f),
                    static_cast<int>(EditorTheme::ACCENT_B * 255.0f), 255);
}

constexpr float kProjectSelectionOutlineThickness = 2.0f;
/// Full-height click strip on the right of the model icon (image drawn with aspect preserved).
constexpr float kModelExpandStripW = 12.0f;
/// Pixel size of `model_expand_*.png` in repo (square); used only for correct scaling in the strip.
constexpr float kModelExpandIconSrcPx = 32.0f;

} // namespace

// ════════════════════════════════════════════════════════════════════
// Static extension sets
// ════════════════════════════════════════════════════════════════════

static const std::unordered_set<std::string> sImageExtensions = {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".gif",
                                                                 ".psd", ".hdr", ".pic",  ".pnm", ".pgm", ".ppm"};

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
        {"__dir__", "folder"},
        {".py", "script_py"},
        {".vert", "shader_vert"},
        {".frag", "shader_frag"},
        {".hlsl", "shader_hlsl"},
        {".fbx", "model_3d"},
        {".obj", "model_3d"},
        {".gltf", "model_3d"},
        {".glb", "model_3d"},
        {".wav", "audio"},
        {".ttf", "font"},
        {".otf", "font"},
        {".txt", "text"},
        {".md", "readme"},
        {".scene", "scene"},
        {".prefab", "prefab"},
        {".animclip2d", "animclip2d"},
        {".animclip3d", "animclip3d"},
        {".animfsm", "animfsm"},
    };
    return map;
}

// ════════════════════════════════════════════════════════════════════
// Static data: Drag-drop maps
// ════════════════════════════════════════════════════════════════════

const std::unordered_map<std::string, ProjectPanel::DragDropInfo> &ProjectPanel::GetDragDropMap()
{
    static const std::unordered_map<std::string, DragDropInfo> map = {
        {".py", {"SCRIPT_FILE", "Script"}},
        {".mat", {"MATERIAL_FILE", "Material"}},
        {".vert", {"SHADER_FILE", "Shader"}},
        {".frag", {"SHADER_FILE", "Shader"}},
        {".glsl", {"SHADER_FILE", "Shader"}},
        {".hlsl", {"SHADER_FILE", "Shader"}},
        {".png", {"TEXTURE_FILE", "Texture"}},
        {".jpg", {"TEXTURE_FILE", "Texture"}},
        {".jpeg", {"TEXTURE_FILE", "Texture"}},
        {".bmp", {"TEXTURE_FILE", "Texture"}},
        {".tga", {"TEXTURE_FILE", "Texture"}},
        {".gif", {"TEXTURE_FILE", "Texture"}},
        {".psd", {"TEXTURE_FILE", "Texture"}},
        {".hdr", {"TEXTURE_FILE", "Texture"}},
        {".pic", {"TEXTURE_FILE", "Texture"}},
        {".pnm", {"TEXTURE_FILE", "Texture"}},
        {".pgm", {"TEXTURE_FILE", "Texture"}},
        {".ppm", {"TEXTURE_FILE", "Texture"}},
        {".wav", {"AUDIO_FILE", "Audio"}},
        {".ttf", {"FONT_FILE", "Font"}},
        {".otf", {"FONT_FILE", "Font"}},
        {".scene", {"SCENE_FILE", "Scene"}},
        {".animclip2d", {"ANIMCLIP_FILE", "2D AnimClip"}},
        {".animclip3d", {"ANIMCLIP3D_FILE", "3D AnimClip"}},
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
    if (ext == ".animclip3d")
        return "[A3]";
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

uint64_t ProjectPanel::GetMtimeNs(const std::string &path)
{
    std::error_code ec;
    auto ftime = fs::last_write_time(fs::u8path(path), ec);
    if (ec)
        return 0;
    // Store raw bit pattern as uint64_t.  On MSVC, file_clock ns since epoch
    // can exceed INT64_MAX; we only need the value for change-detection, not
    // as a signed timestamp, so reinterpret the bits.
    const auto rawNs = std::chrono::duration_cast<std::chrono::nanoseconds>(ftime.time_since_epoch()).count();
    uint64_t bits;
    std::memcpy(&bits, &rawNs, sizeof(bits));
    return bits;
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

void ProjectPanel::SetEngine(Infernux *engine)
{
    m_engine = engine;
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
    if (filePath.empty())
        return;
    auto normTarget = NormalizePath(filePath);

    std::vector<std::string> mtimeToRemove;
    for (auto &[path, _] : m_materialMtimeCache) {
        if (NormalizePath(path) == normTarget)
            mtimeToRemove.push_back(path);
    }
    for (auto &path : mtimeToRemove)
        m_materialMtimeCache.erase(path);
}

void ProjectPanel::InvalidateTextureThumbnail(const std::string &filePath)
{
    if (filePath.empty())
        return;
    auto normTarget = NormalizePath(filePath);

    std::vector<std::string> mtimeToRemove;
    for (auto &[path, _] : m_textureMtimeCache) {
        if (NormalizePath(path) == normTarget)
            mtimeToRemove.push_back(path);
    }
    for (auto &path : mtimeToRemove)
        m_textureMtimeCache.erase(path);
}

// ════════════════════════════════════════════════════════════════════
// Notification helpers
// ════════════════════════════════════════════════════════════════════

void ProjectPanel::NotifySelectionChanged()
{
    if (!onFileSelected)
        return;
    if (m_selectedFiles.size() == 1)
        onFileSelected(SelectionPathForInspector(m_selectedFiles[0]));
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
    std::unordered_set<std::string> seen;
    for (auto &p : m_selectedFiles) {
        if (p.empty())
            continue;
        std::string real = ResolveRealAssetPath(p);
        if (real.empty())
            continue;
        if (!fs::exists(fs::u8path(real), ec))
            continue;
        if (seen.insert(real).second)
            result.push_back(real);
    }
    if (result.empty() && !m_selectedFile.empty()) {
        std::string real = ResolveRealAssetPath(m_selectedFile);
        if (!real.empty() && fs::exists(fs::u8path(real), ec))
            result.push_back(real);
    }
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
        uint64_t mtimeNs = GetMtimeNs(path);
        it->second.lastValidatedAt = m_frameTimeNow;
        if (it->second.mtimeNs == mtimeNs)
            return &it->second;
    }

    std::error_code ec;
    if (!fs::is_directory(fs::u8path(path), ec))
        return nullptr;

    uint64_t mtimeNs = GetMtimeNs(path);

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
                    const auto rawNs =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(ftime.time_since_epoch()).count();
                    std::memcpy(&item.mtimeNs, &rawNs, sizeof(item.mtimeNs));
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

namespace
{
std::string TryGetMetaString(const infernux::InxResourceMeta *meta, const std::string &key)
{
    if (!meta || key.empty())
        return {};
    const auto &map = meta->GetMetadata();
    auto it = map.find(key);
    if (it == map.end())
        return {};
    const auto &typeName = it->second.first;
    const auto &value = it->second.second;
    try {
        if (typeName == "string")
            return std::any_cast<std::string>(value);
    } catch (...) {
    }
    return {};
}

int TryGetMetaInt(const infernux::InxResourceMeta *meta, const std::string &key, int defaultValue)
{
    if (!meta || key.empty())
        return defaultValue;
    const auto &map = meta->GetMetadata();
    auto it = map.find(key);
    if (it == map.end())
        return defaultValue;
    const auto &typeName = it->second.first;
    const auto &value = it->second.second;
    try {
        if (typeName == "int")
            return std::any_cast<int>(value);
        if (typeName == "size_t")
            return static_cast<int>(std::any_cast<size_t>(value));
        if (typeName == "float")
            return static_cast<int>(std::lround(std::any_cast<float>(value)));
    } catch (...) {
    }
    return defaultValue;
}
} // namespace

void ProjectPanel::AppendModelSubAssets(std::vector<FileItem> &out, AssetDatabase *adb, const FileItem &modelItem)
{
    const std::string &modelPath = modelItem.path;
    const infernux::InxResourceMeta *meta = nullptr;
    if (adb)
        meta = adb->GetMetaByPath(modelPath);

    const uint64_t childMtime = modelItem.mtimeNs;

    // ── Materials (material slots) ────────────────────────────────────
    std::vector<std::string> matNames = SplitCommaList(TryGetMetaString(meta, "material_slots"));
    int matCount = TryGetMetaInt(meta, "material_slot_count", -1);
    if (matNames.empty() && matCount > 0) {
        matNames.reserve(static_cast<size_t>(matCount));
        for (int i = 0; i < matCount; ++i)
            matNames.push_back("Material_" + std::to_string(i));
    }

    if (!matNames.empty()) {
        for (int i = 0; i < static_cast<int>(matNames.size()); ++i) {
            FileItem sub{};
            sub.type = FileItem::SubMaterial;
            sub.name = matNames[static_cast<size_t>(i)];
            sub.path = MakeSubAssetVirtualPath(modelPath, kSubMatToken, i);
            sub.ext = ".mat";
            sub.parentPath = modelPath;
            sub.mtimeNs = childMtime;
            sub.slotIndex = i;
            out.push_back(std::move(sub));
        }
    } else {
        FileItem sub{};
        sub.type = FileItem::SubMaterial;
        sub.name = "(No materials in meta — reimport model)";
        sub.path = MakeSubAssetVirtualPath(modelPath, kSubMatToken, -1);
        sub.ext = ".mat";
        sub.parentPath = modelPath;
        sub.mtimeNs = childMtime;
        sub.slotIndex = -1;
        out.push_back(std::move(sub));
    }

    // ── Embedded animation takes (one row per take; virtual id = model GUID when available) ──
    std::string animVirtualBase = modelPath;
    if (adb) {
        std::string g = adb->GetGuidFromPath(modelPath);
        if (!g.empty())
            animVirtualBase = std::move(g);
    }
    std::vector<std::string> animNames = SplitCommaList(TryGetMetaString(meta, "animation_names_csv"));
    int animCount = TryGetMetaInt(meta, "animation_count", -1);
    if (!animNames.empty()) {
        const int maxShow = 24;
        const int total = static_cast<int>(animNames.size());
        const int show = std::min(total, maxShow);
        for (int i = 0; i < show; ++i) {
            FileItem sub{};
            sub.type = FileItem::SubMesh;
            const std::string &takeName = animNames[static_cast<size_t>(i)];
            sub.name = StripPipeDisplaySuffix(takeName) + ".animclip3d";
            sub.path = MakeSubAssetVirtualPath(animVirtualBase, kSubAnimToken, i);
            sub.ext = ".animclip3d";
            sub.parentPath = modelPath;
            sub.mtimeNs = childMtime;
            sub.slotIndex = i;
            out.push_back(std::move(sub));
        }
        if (total > show) {
            FileItem sub{};
            sub.type = FileItem::SubMesh;
            sub.name = std::string("... ") + std::to_string(total - show) + " more animation takes";
            sub.path = MakeSubAssetVirtualPath(animVirtualBase, kSubAnimToken, 999999);
            sub.ext = ".animclip3d";
            sub.parentPath = modelPath;
            sub.mtimeNs = childMtime;
            sub.slotIndex = -1;
            out.push_back(std::move(sub));
        }
    } else if (animCount > 0) {
        FileItem sub{};
        sub.type = FileItem::SubMesh;
        sub.name = std::string("Animations: ") + std::to_string(animCount) + " take(s) (reimport for names)";
        sub.path = MakeSubAssetVirtualPath(animVirtualBase, kSubAnimToken, 0);
        sub.ext = ".animclip3d";
        sub.parentPath = modelPath;
        sub.mtimeNs = childMtime;
        sub.slotIndex = -1;
        out.push_back(std::move(sub));
    }
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

    auto &cached = m_augmentedCache[path];
    cached.mtimeNs = snapshot->mtimeNs;
    cached.expandedPaths = expandedPaths;
    cached.items.clear();
    cached.items.reserve(snapshot->items.size() + 8);

    std::unordered_set<std::string> expandedSet(expandedPaths.begin(), expandedPaths.end());
    for (const auto &item : snapshot->items) {
        cached.items.push_back(item);
        if (item.type == FileItem::File && IsModelExt(item.ext) && expandedSet.count(item.path) > 0)
            AppendModelSubAssets(cached.items, m_assetDatabase, item);
    }
    return &cached.items;
}

// ════════════════════════════════════════════════════════════════════
// Thumbnail system
// ════════════════════════════════════════════════════════════════════

uint64_t ProjectPanel::GetMaterialMtimeNs(const std::string &filePath)
{
    if (filePath.empty())
        return 0;

    double now = m_frameTimeNow;

    auto it = m_materialMtimeCache.find(filePath);
    if (it != m_materialMtimeCache.end() && (now - it->second.second) < 1.0)
        return it->second.first;

    std::string diskPath = filePath;
    const auto subPos = filePath.find(kSubMatToken);
    if (subPos != std::string::npos)
        diskPath = filePath.substr(0, subPos);

    std::error_code ec;
    if (!fs::exists(fs::u8path(diskPath), ec))
        return 0;

    uint64_t mtimeNs = GetMtimeNs(diskPath);
    m_materialMtimeCache[filePath] = {mtimeNs, now};
    return mtimeNs;
}

uint64_t ProjectPanel::GetTextureMtimeNs(const std::string &filePath)
{
    if (filePath.empty())
        return 0;

    double now = m_frameTimeNow;

    auto it = m_textureMtimeCache.find(filePath);
    if (it != m_textureMtimeCache.end() && (now - it->second.second) < 1.0)
        return it->second.first;

    std::error_code ec;
    if (!fs::exists(fs::u8path(filePath), ec))
        return 0;

    uint64_t imageMtime = GetMtimeNs(filePath);

    // Also watch the .meta file so that import setting changes (filter_mode, max_size, srgb…)
    // invalidate the cached thumbnail.
    uint64_t metaMtime = 0;
    std::string metaPath = InxResourceMeta::GetMetaFilePath(filePath);
    if (!metaPath.empty() && fs::exists(fs::u8path(metaPath), ec))
        metaMtime = GetMtimeNs(metaPath);

    // Combine both mtimes into a single fingerprint that changes when either changes.
    uint64_t combined = imageMtime ^ (metaMtime * UINT64_C(2654435761));
    m_textureMtimeCache[filePath] = {combined, now};
    return combined;
}

uint64_t ProjectPanel::GetThumbnail(const std::string &filePath, uint64_t cachedMtimeNs)
{
    if (filePath.empty() || !m_engine)
        return 0;

    uint64_t texMtime = GetTextureMtimeNs(filePath);
    if (texMtime == 0)
        return 0;

    // Read import settings from .meta for nearest/srgb.
    bool nearest = false;
    bool srgb = false;
    if (m_assetDatabase) {
        const InxResourceMeta *meta = m_assetDatabase->GetMetaByPath(filePath);
        if (meta) {
            if (meta->HasKey("filter_mode")) {
                std::string fm = meta->GetDataAs<std::string>("filter_mode");
                nearest = (fm == "point" || fm == "nearest");
            }
            if (meta->HasKey("srgb"))
                srgb = meta->GetDataAs<bool>("srgb");
        }
    }

    const std::string resourceKey = std::string("tex|") + filePath;
    // pump=false: PreRender already pumped once this frame.
    auto [texId, w, h] = m_engine->QueryOrScheduleTexturePreview(resourceKey, filePath, texMtime, nearest, srgb, false);
    return texId;
}

uint64_t ProjectPanel::GetMaterialThumbnail(const std::string &filePath)
{
    if (filePath.empty() || !m_engine)
        return 0;

    uint64_t mtimeNs = GetMaterialMtimeNs(filePath);
    if (mtimeNs == 0)
        return 0;

    const std::string resourceKey = std::string("mat|") + filePath;
    return m_engine->QueryOrScheduleMaterialPreview(resourceKey, filePath, "", mtimeNs);
}

uint64_t ProjectPanel::GetModelThumbnail(const std::string &filePath)
{
    if (filePath.empty() || !m_engine)
        return 0;

    double now = m_frameTimeNow;

    uint64_t mtimeNs = 0;
    auto it = m_modelMtimeCache.find(filePath);
    if (it != m_modelMtimeCache.end() && (now - it->second.second) < 1.0) {
        mtimeNs = it->second.first;
    } else {
        std::error_code ec;
        if (!fs::exists(fs::u8path(filePath), ec))
            return 0;
        mtimeNs = GetMtimeNs(filePath);
        m_modelMtimeCache[filePath] = {mtimeNs, now};
    }
    if (mtimeNs == 0)
        return 0;

    const std::string resourceKey = std::string("mesh|") + filePath;
    return m_engine->QueryOrScheduleMeshPreview(resourceKey, filePath, mtimeNs);
}

uint64_t ProjectPanel::GetPrefabThumbnail(const std::string &filePath)
{
    return GetModelThumbnail(filePath);
}

void ProjectPanel::ProcessPendingThumbnails()
{
    if (!m_engine)
        return;
    m_engine->PumpPreviewTasks();
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
    needed.insert("model_expand_open");
    needed.insert("model_expand_closed");

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
    auto &iconMap = GetIconMap();

    if (item.type == FileItem::Dir) {
        auto sit = iconMap.find("__dir__");
        key = sit != iconMap.end() ? &sit->second : nullptr;
    } else if (item.type == FileItem::SubMesh) {
        auto mapIt = iconMap.find(item.ext.empty() ? ".fbx" : item.ext);
        key = mapIt != iconMap.end() ? &mapIt->second : nullptr;
    } else if (item.type == FileItem::SubMaterial) {
        auto sit = iconMap.find(".mat");
        key = sit != iconMap.end() ? &sit->second : nullptr;
    } else {
        auto mapIt = iconMap.find(item.ext);
        key = mapIt != iconMap.end() ? &mapIt->second : nullptr;
    }

    if (!key)
        return 0;
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
    LabelCacheKey key;
    key.path = item.path;
    key.name = item.name;
    key.type = static_cast<uint8_t>(item.type);
    // Model expand/collapse is shown with a side button, not in the string (avoids missing glyphs).
    key.expanded = false;
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
    } else if (item.type == FileItem::SubMesh) {
        std::string subLabel = item.name;
        if (item.ext == ".animclip3d" && !subLabel.empty()) {
            auto d = subLabel.rfind('.');
            if (d != std::string::npos) {
                std::string stem = subLabel.substr(0, d);
                std::string ext = subLabel.substr(d);
                stem = StripPipeDisplaySuffix(stem);
                subLabel = std::move(stem) + ext;
            }
        }
        nameDisplay = std::string("  ") + subLabel;
    } else if (item.type == FileItem::SubMaterial) {
        nameDisplay = std::string("  ") + item.name;
    }

    static constexpr const char *kEllipsisAscii = "...";
    float maxTextW = textRegionW - 4.0f;
    float textW = ctx->CalcTextWidth(nameDisplay);
    if (textW > maxTextW) {
        // Truncate with ASCII ellipsis
        std::string truncated = nameDisplay;
        while (truncated.size() > 1) {
            truncated.pop_back();
            float tw = ctx->CalcTextWidth(std::string(truncated) + kEllipsisAscii);
            if (tw <= maxTextW) {
                nameDisplay = truncated + kEllipsisAscii;
                textW = tw;
                break;
            }
        }
        if (truncated.size() <= 1) {
            nameDisplay = kEllipsisAscii;
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
            if (openFile)
                openFile(item.path);
        } else if (item.ext == ".scene") {
            if (openScene)
                openScene(item.path);
        } else if (item.ext == ".prefab") {
            if (openPrefabMode)
                openPrefabMode(item.path);
        } else if (item.ext == ".animclip2d") {
            if (openAnimClip)
                openAnimClip(item.path);
        } else if (item.ext == ".animclip3d") {
            // 3D clips are edited via the Inspector (Python asset_details_renderer).
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
    bool singleSel = (selected.size() == 1 && !m_selectedFile.empty() && !IsVirtualSubAssetPath(m_selectedFile));

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

        auto name = FromFsPath(fs::u8path(src).filename());
        auto dst = FromFsPath(fs::u8path(m_currentPath) / fs::u8path(name));

        // If destination already exists, use unique name
        if (fs::exists(fs::u8path(dst), ec)) {
            if (!getUniqueName)
                continue;
            auto stem = FromFsPath(fs::u8path(name).stem());
            auto ext = FromFsPath(fs::u8path(name).extension());
            if (fs::is_directory(fs::u8path(src), ec)) {
                ext = "";
                stem = name;
            }
            auto uniqueName = getUniqueName(m_currentPath, stem, ext);
            dst = FromFsPath(fs::u8path(m_currentPath) / fs::u8path(uniqueName + ext));
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
    if (path.empty())
        return;
    // Virtual sub-assets are not real files — renaming them would be meaningless.
    if (IsVirtualSubAssetPath(path))
        return;

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
            auto u8str = Utf8FromWidePath(wpath);
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
        auto name = FromFsPath(fs::u8path(src).filename());
        auto dst = FromFsPath(fs::u8path(m_currentPath) / fs::u8path(name));
        bool samePath = (NormalizePath(src) == NormalizePath(dst));

        if (samePath && isCut)
            continue;

        if (samePath || fs::exists(fs::u8path(dst), ec)) {
            if (!getUniqueName)
                continue;
            auto stem = FromFsPath(fs::u8path(name).stem());
            auto ext = FromFsPath(fs::u8path(name).extension());
            if (fs::is_directory(fs::u8path(src), ec)) {
                ext = "";
                stem = name;
            }
            auto uniqueName = getUniqueName(m_currentPath, stem, ext);
            dst = FromFsPath(fs::u8path(m_currentPath) / fs::u8path(uniqueName + ext));
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
            } else if (fs::is_directory(fs::u8path(src), ec)) {
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
    bool clickedOutsideProject = hasSelection &&
                                 (ImGui::IsMouseClicked(0) || ImGui::IsMouseClicked(1) || ImGui::IsMouseClicked(2)) &&
                                 !ImGui::IsWindowHovered(ImGuiHoveredFlags_RootAndChildWindows) &&
                                 !ImGui::IsAnyItemActive() && !IsMouseOverInspectorWindow();
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
                displayTexId = GetTypeIconId(item);
            } else if (item.type == FileItem::SubMaterial) {
                displayTexId = GetMaterialThumbnail(item.path);
                if (displayTexId == 0)
                    displayTexId = GetTypeIconId(item);
            } else if (item.type == FileItem::File) {
                if (IsImageExt(item.ext))
                    displayTexId = GetThumbnail(item.path, item.mtimeNs);
                else if (IsMaterialExt(item.ext))
                    displayTexId = GetMaterialThumbnail(item.path);
                else if (IsModelExt(item.ext))
                    displayTexId = GetModelThumbnail(item.path);
                else if (item.ext == ".prefab")
                    displayTexId = GetPrefabThumbnail(item.path);
                if (displayTexId == 0)
                    displayTexId = GetTypeIconId(item);
            } else {
                displayTexId = GetTypeIconId(item);
            }

            // ── Render icon (model: thumbnail on the left + narrow expand strip, same height) ──
            const bool isModelFile = (item.type == FileItem::File && IsModelExt(item.ext));
            const float stripW = isModelFile ? kModelExpandStripW : 0.0f;
            const float thumbW = (stripW > 0.0f) ? (iconSize - stripW) : iconSize;

            if (displayTexId != 0) {
                int srcW = 0;
                int srcH = 0;
                if (item.type == FileItem::File) {
                    if (IsImageExt(item.ext) && m_engine) {
                        const std::string resourceKey = std::string("tex|") + item.path;
                        auto [readyW, readyH] = m_engine->GetTexturePreviewSize(resourceKey);
                        srcW = readyW;
                        srcH = readyH;
                    } else if (IsMaterialExt(item.ext) || IsModelExt(item.ext) || item.ext == ".prefab") {
                        srcW = 256;
                        srcH = 256;
                    }
                } else if (item.type == FileItem::SubMaterial) {
                    srcW = 256;
                    srcH = 256;
                }

                ImGui::BeginGroup();
                // InvisibleButton for hit-testing; AddImage for drawing
                ImGui::InvisibleButton("##ic", ImVec2(thumbW, iconSize));
                const bool thumbHovered = ImGui::IsItemHovered();
                const bool thumbRmb = ImGui::IsItemClicked(1);
                ImVec2 rMin = ImGui::GetItemRectMin();
                ImVec2 rMax = ImGui::GetItemRectMax();
                ImVec2 drawMin = rMin;
                ImVec2 drawMax = rMax;
                if (srcW > 0 && srcH > 0) {
                    const float scale =
                        std::min(thumbW / static_cast<float>(srcW), iconSize / static_cast<float>(srcH));
                    const float drawW = std::max(1.0f, static_cast<float>(srcW) * scale);
                    const float drawH = std::max(1.0f, static_cast<float>(srcH) * scale);
                    drawMin.x += (thumbW - drawW) * 0.5f;
                    drawMin.y += (iconSize - drawH) * 0.5f;
                    drawMax = ImVec2(drawMin.x + drawW, drawMin.y + drawH);
                }
                drawList->AddImage(ImTextureRef(static_cast<ImTextureID>(displayTexId)), drawMin, drawMax);
                if (isModelFile) {
                    ImGui::SameLine(0.0f, 0.0f);
                    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
                    const bool ex = m_expandedModels.count(item.path) > 0;
                    uint64_t expandTex = 0;
                    if (ex) {
                        auto eit = m_typeIconCache.find("model_expand_open");
                        expandTex = eit != m_typeIconCache.end() ? eit->second : 0;
                    } else {
                        auto eit = m_typeIconCache.find("model_expand_closed");
                        expandTex = eit != m_typeIconCache.end() ? eit->second : 0;
                    }
                    bool expandClicked = false;
                    if (expandTex != 0) {
                        // ImageButton with ImVec2(stripW, iconSize) distorts a square art asset; keep hit area
                        // full strip x icon height but draw the glyph with aspect preserved and centered.
                        ImGui::InvisibleButton("##mdlexpand", ImVec2(stripW, iconSize));
                        expandClicked = ImGui::IsItemHovered() && ImGui::IsMouseReleased(0) && !hasDragPayload;
                        const ImVec2 ex0 = ImGui::GetItemRectMin();
                        const ImVec2 ex1 = ImGui::GetItemRectMax();
                        const float exW = ex1.x - ex0.x;
                        const float exH = ex1.y - ex0.y;
                        const float srcW = kModelExpandIconSrcPx;
                        const float srcH = kModelExpandIconSrcPx;
                        const float gScale = std::min(exW / srcW, exH / srcH);
                        const float dW = std::max(1.0f, srcW * gScale);
                        const float dH = std::max(1.0f, srcH * gScale);
                        const float cx = (ex0.x + ex1.x) * 0.5f;
                        const float cy = (ex0.y + ex1.y) * 0.5f;
                        const ImVec2 dmin(cx - dW * 0.5f, cy - dH * 0.5f);
                        const ImVec2 dmax(cx + dW * 0.5f, cy + dH * 0.5f);
                        drawList->AddImage(ImTextureRef(static_cast<ImTextureID>(expandTex)), dmin, dmax);
                    } else {
                        expandClicked = ImGui::Button(ex ? "v" : ">", ImVec2(stripW, iconSize));
                    }
                    if (expandClicked) {
                        if (ex)
                            m_expandedModels.erase(item.path);
                        else
                            m_expandedModels.insert(item.path);
                    }
                    ImGui::PopStyleVar();
                }
                ImGui::EndGroup();

                if (isSelected) {
                    const ImVec2 g0 = ImGui::GetItemRectMin();
                    const ImVec2 g1 = ImGui::GetItemRectMax();
                    drawList->AddRect(g0, g1, ProjectSelectionOutlineColor(), 0.0f, 0,
                                      kProjectSelectionOutlineThickness);
                }
                if (thumbHovered && ImGui::IsMouseReleased(0) && !hasDragPayload)
                    HandleItemClick(item, ctx);
                if (thumbRmb) {
                    m_selectedFile = item.path;
                    if (m_selectedSet.count(item.path) == 0) {
                        m_selectedFiles = {item.path};
                        m_selectedSet = {item.path};
                    }
                    NotifySelectionChanged();
                }
            } else {
                const char *tag = (item.type != FileItem::Dir) ? GetFileTypeTag(item.name) : "[DIR]";
                ctx->Selectable(tag, isSelected, 0, iconSize, iconSize);
                ImVec2 rMin = ImGui::GetItemRectMin();
                ImVec2 rMax = ImGui::GetItemRectMax();
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
                if (isSelected)
                    drawList->AddRect(rMin, rMax, ProjectSelectionOutlineColor(), 0.0f, 0,
                                      kProjectSelectionOutlineThickness);
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
    const std::string selectedReal = m_selectedFile.empty() ? std::string() : ResolveRealAssetPath(m_selectedFile);
    if (!m_selectedFile.empty() && !selectedReal.empty() && fs::exists(fs::u8path(selectedReal), ec)) {
        ctx->Separator();
        if (ctx->Selectable(Tr("project.reveal_in_explorer"), false)) {
            if (revealInExplorer)
                revealInExplorer(selectedReal);
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
        bool canRename = (selectedPaths.size() == 1) && !IsVirtualSubAssetPath(m_selectedFile);
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
    // Embedded model materials are browse-only (no drag — use a standalone .mat to assign).
    if (item.type != FileItem::Dir && item.type != FileItem::File && item.type != FileItem::SubMesh)
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

    if (item.type == FileItem::SubMesh) {
        if (item.parentPath.empty()) {
            ctx->EndDragDropSource();
            return;
        }

        // Embedded animation take (model.fbx::subanim:i) — drag as 3D clip, same as a .animclip3d file.
        // slotIndex >= 0 marks a real take row; overflow / placeholder rows use -1.
        if (item.path.find(kSubAnimToken) != std::string::npos && item.slotIndex >= 0) {
            auto &ddMap = GetDragDropMap();
            auto ddIt = ddMap.find(".animclip3d");
            if (ddIt != ddMap.end()) {
                ctx->SetDragDropPayload(ddIt->second.payloadType, item.path);
                ctx->Label(std::string(ddIt->second.label) + ": " + item.name);
            } else {
                ctx->SetDragDropPayload("ANIMCLIP3D_FILE", item.path);
                ctx->Label("3D AnimClip: " + item.name);
            }
            ctx->EndDragDropSource();
            return;
        }

        // Other embedded FBX sub-entries: parent model asset (GUID when possible).
        std::string ext = fs::u8path(item.parentPath).extension().string();
        for (auto &c : ext)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

        auto &gdMap = GetGuidDragDropMap();
        auto gdIt = gdMap.find(ext);

        if (gdIt != gdMap.end()) {
            std::string guid;
            if (getGuidFromPath)
                guid = getGuidFromPath(item.parentPath);
            else if (m_assetDatabase)
                guid = m_assetDatabase->GetGuidFromPath(item.parentPath);

            if (!guid.empty())
                ctx->SetDragDropPayload(gdIt->second.guidPayloadType, guid);
            else
                ctx->SetDragDropPayload(gdIt->second.pathPayloadType, item.parentPath);

            ctx->Label(std::string("Model") + ": " + item.name);
        } else {
            ctx->SetDragDropPayload(DRAG_TYPE_PROJECT_ITEM, item.parentPath);
            ctx->Label("Model: " + item.name);
        }

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
