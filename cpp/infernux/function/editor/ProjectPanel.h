#pragma once

#include <function/editor/EditorPanel.h>
#include <function/editor/EditorTheme.h>
#include <function/renderer/InxRenderer.h>
#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/InxFileLoader/InxTextureLoader.hpp>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace infernux
{

class Infernux;

/// C++ native Project panel — Unity-style asset browser with folder tree,
/// file grid, thumbnails, drag-drop, inline rename, and virtual scrolling.
///
/// Heavy-lift rendering (grid loop, folder tree, thumbnail management) is
/// entirely in C++.  Python-only managers (scene open, prefab creation,
/// script validation, asset inspector) are reached via std::function
/// callbacks set from the bootstrap layer.
class ProjectPanel : public EditorPanel
{
  public:
    ProjectPanel();

    // ── Public API (called from Python bootstrap / other panels) ─────

    void SetRootPath(const std::string &path);
    void SetEngine(Infernux *engine);
    void SetRenderer(InxRenderer *renderer);
    void SetAssetDatabase(AssetDatabase *adb);
    void SetIconsDirectory(const std::string &dir);

    void ClearSelection();
    void SetSelectedFile(const std::string &path);

    void InvalidateMaterialThumbnail(const std::string &filePath);
    void InvalidateTextureThumbnail(const std::string &filePath);

    /// Invalidate the directory cache so listing refreshes next frame.
    void InvalidateDirCache();

    /// Accept files dropped from the OS (e.g. Windows Explorer).
    /// Copies each file/directory into the current directory.
    void ReceiveDroppedFiles(const std::vector<std::string> &paths);

    /// State persistence
    std::string GetCurrentPath() const
    {
        return m_currentPath;
    }
    void SetCurrentPath(const std::string &path);

    // ── Notification callbacks ───────────────────────────────────────

    /// Called when file selection changes (receives single path or empty).
    std::function<void(const std::string &)> onFileSelected;
    /// Called when empty area is clicked.
    std::function<void()> onEmptyAreaClicked;
    /// Called when current_path changes between frames.
    std::function<void()> onStateChanged;

    // ── File operation callbacks (delegated to Python) ───────────────

    /// Create folder: (currentPath, name) → (ok, errorMsg)
    std::function<std::pair<bool, std::string>(const std::string &, const std::string &)> createFolder;
    /// Create script: (currentPath, name) → (ok, errorMsg)
    std::function<std::pair<bool, std::string>(const std::string &, const std::string &)> createScript;
    /// Create shader: (currentPath, name, type) → (ok, errorMsg)
    std::function<std::pair<bool, std::string>(const std::string &, const std::string &, const std::string &)>
        createShader;
    /// Create material: (currentPath, name) → (ok, errorMsg)
    std::function<std::pair<bool, std::string>(const std::string &, const std::string &)> createMaterial;
    /// Create scene: (currentPath, name) → (ok, errorMsg)
    std::function<std::pair<bool, std::string>(const std::string &, const std::string &)> createScene;
    /// Create animation clip: (currentPath, name) → (ok, errorMsg)
    std::function<std::pair<bool, std::string>(const std::string &, const std::string &)> createAnimClip;
    /// Create 3D animation clip: (currentPath, name) → (ok, errorMsg)
    std::function<std::pair<bool, std::string>(const std::string &, const std::string &)> createAnimClip3D;
    /// Create animation state machine: (currentPath, name) → (ok, errorMsg)
    std::function<std::pair<bool, std::string>(const std::string &, const std::string &)> createAnimFsm;
    /// Create prefab from hierarchy gameobject: (objId, currentPath)
    std::function<void(uint64_t, const std::string &)> createPrefabFromHierarchy;

    /// Delete items: (paths) — shows confirmation dialog
    std::function<void(const std::vector<std::string> &)> deleteItems;

    /// Rename: (oldPath, newName) → newPath or empty on failure
    std::function<std::string(const std::string &, const std::string &)> doRename;
    /// Get unique name: (currentPath, baseName, extension) → uniqueName
    std::function<std::string(const std::string &, const std::string &, const std::string &)> getUniqueName;

    /// Move item to directory: (itemPath, destDir) → newPath or empty
    std::function<std::string(const std::string &, const std::string &)> moveItemToDirectory;

    /// Open file: (filePath)
    std::function<void(const std::string &)> openFile;
    /// Open scene: (filePath)
    std::function<void(const std::string &)> openScene;
    /// Open prefab mode: (filePath)
    std::function<void(const std::string &)> openPrefabMode;
    /// Open animation clip: (filePath)
    std::function<void(const std::string &)> openAnimClip;
    /// Open animation state machine: (filePath)
    std::function<void(const std::string &)> openAnimFsm;
    /// Reveal in file explorer: (path)
    std::function<void(const std::string &)> revealInExplorer;

    /// Validate Python script for drag-drop: (filePath) → true if component
    std::function<bool(const std::string &)> validateScriptComponent;

    /// Get GUID from path (delegates to AssetDatabase if callback not set)
    std::function<std::string(const std::string &)> getGuidFromPath;
    /// Get path from GUID
    std::function<std::string(const std::string &)> getPathFromGuid;

    /// Invalidate asset inspector cache
    std::function<void(const std::string &)> invalidateAssetInspector;

    // ── Translation ──────────────────────────────────────────────────

    std::function<std::string(const std::string &)> translate;

    // ── Drag-drop payload types ──────────────────────────────────────

    static constexpr const char *DRAG_TYPE_PROJECT_ITEM = "PROJECT_PANEL_ITEM_PATH";
    static constexpr const char *DRAG_TYPE_HIERARCHY_GO = "HIERARCHY_GAMEOBJECT";

  protected:
    void OnRenderContent(InxGUIContext *ctx) override;
    void PreRender(InxGUIContext *ctx) override;

  private:
    // ── Translation cache ────────────────────────────────────────────
    const std::string &Tr(const std::string &key);
    std::unordered_map<std::string, std::string> m_trCache;

    // ── Item representation ──────────────────────────────────────────
    struct FileItem
    {
        enum Type : uint8_t
        {
            Dir,
            File,
            SubMesh,
            SubMaterial
        };
        Type type = File;
        std::string name;
        std::string path;
        std::string ext;
        std::string parentPath; // for sub-assets
        uint64_t mtimeNs = 0;
        int slotIndex = -1; // for SubMaterial
    };

    // ── Directory snapshot cache ─────────────────────────────────────
    struct DirSnapshot
    {
        uint64_t mtimeNs = 0;
        double lastValidatedAt = 0.0; // steady-clock seconds
        std::vector<FileItem> dirs;
        std::vector<FileItem> files;
        std::vector<FileItem> items; // dirs + files
    };

    struct DirTreeMeta
    {
        bool hasSubdirs = false;
    };

    DirSnapshot *GetDirSnapshot(const std::string &path);
    DirTreeMeta *GetDirTreeMeta(const std::string &path);
    std::vector<FileItem> *GetProjectItems(const std::string &path, DirSnapshot *snapshot = nullptr);
    static void AppendModelSubAssets(std::vector<FileItem> &out, AssetDatabase *adb, const FileItem &modelItem);

    std::unordered_map<std::string, DirSnapshot> m_dirCache;
    std::unordered_map<std::string, DirTreeMeta> m_dirTreeMetaCache;

    // Augmented items (with model sub-assets)
    struct AugmentedCache
    {
        uint64_t mtimeNs = 0;
        std::vector<std::string> expandedPaths;
        std::vector<FileItem> items;
    };
    std::unordered_map<std::string, AugmentedCache> m_augmentedCache;

    // ── Label layout cache ───────────────────────────────────────────
    struct LabelCacheKey
    {
        std::string path;
        std::string name;
        uint8_t type;
        bool expanded;
        int widthPx;

        bool operator==(const LabelCacheKey &o) const
        {
            return type == o.type && expanded == o.expanded && widthPx == o.widthPx && path == o.path && name == o.name;
        }
    };
    struct LabelCacheKeyHash
    {
        size_t operator()(const LabelCacheKey &k) const
        {
            size_t h = std::hash<std::string>{}(k.path);
            h ^= std::hash<int>{}(k.widthPx) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };
    struct LabelEntry
    {
        std::string displayText;
        float offsetX = 0.0f;
    };
    std::unordered_map<LabelCacheKey, LabelEntry, LabelCacheKeyHash> m_labelCache;
    float m_gridTextLineHeight = 0.0f;

    // ── Thumbnail system ─────────────────────────────────────────────
    std::unordered_map<std::string, std::pair<uint64_t, double>> m_materialMtimeCache;
    std::unordered_map<std::string, std::pair<uint64_t, double>> m_textureMtimeCache;
    std::unordered_map<std::string, std::pair<uint64_t, double>> m_modelMtimeCache;

    void ProcessPendingThumbnails();
    uint64_t GetThumbnail(const std::string &filePath, uint64_t cachedMtimeNs);
    uint64_t GetMaterialThumbnail(const std::string &filePath);
    uint64_t GetModelThumbnail(const std::string &filePath);
    uint64_t GetPrefabThumbnail(const std::string &filePath);
    uint64_t GetMaterialMtimeNs(const std::string &filePath);
    uint64_t GetTextureMtimeNs(const std::string &filePath);

    // ── File-type icon cache ─────────────────────────────────────────
    std::unordered_map<std::string, uint64_t> m_typeIconCache;
    bool m_typeIconsLoaded = false;
    std::string m_iconsDir;

    void EnsureTypeIconsLoaded();
    uint64_t GetTypeIconId(const FileItem &item) const;

    // ── Drag-drop maps ───────────────────────────────────────────────
    struct DragDropInfo
    {
        const char *payloadType;
        const char *label;
    };
    struct GuidDragDropInfo
    {
        const char *guidPayloadType;
        const char *pathPayloadType;
        const char *label;
    };
    static const std::unordered_map<std::string, DragDropInfo> &GetDragDropMap();
    static const std::unordered_map<std::string, GuidDragDropInfo> &GetGuidDragDropMap();
    static const std::vector<std::string> &GetMoveAcceptTypes();

    // ── Filtering ────────────────────────────────────────────────────
    static bool ShouldShow(const std::string &name);

    // File type tag for text fallback
    static const char *GetFileTypeTag(const std::string &filename);

    // Icon map: ext/key → icon filename
    static const std::unordered_map<std::string, std::string> &GetIconMap();

    // ── Panel state ──────────────────────────────────────────────────
    std::string m_rootPath;
    std::string m_currentPath;
    std::string m_lastNotifiedPath;

    // Breadcrumb
    std::string m_breadcrumbPath;
    std::string m_breadcrumbText;

    // Selection
    std::string m_selectedFile;
    std::vector<std::string> m_selectedFiles;
    std::unordered_set<std::string> m_selectedSet; // for O(1) lookup

    void NotifySelectionChanged();
    void NotifyEmptyAreaClicked();
    std::vector<std::string> GetSelectedPaths() const;

    // Double-click detection
    std::string m_lastClickedFile;
    double m_lastClickTime = 0.0;

    // Rename
    std::string m_renamingPath;
    char m_renameBuf[256] = {};
    bool m_renameFocusRequested = false;

    // Deferred cache invalidation — set by operations that modify the filesystem
    // mid-render (CommitRename, Delete, Paste, Move) so that the file grid's item
    // pointer stays valid for the remainder of the frame.
    bool m_pendingCacheInvalidation = false;

    // Clipboard
    std::vector<std::string> m_clipboardPaths;
    bool m_clipboardIsCut = false;

    // Model expansion
    std::unordered_set<std::string> m_expandedModels;

    // Visible items for shift-range select
    std::vector<FileItem> *m_visibleItems = nullptr;

    // External refs
    Infernux *m_engine = nullptr;
    InxRenderer *m_renderer = nullptr;
    AssetDatabase *m_assetDatabase = nullptr;

    // ── Extension sets ───────────────────────────────────────────────
    static bool IsImageExt(const std::string &ext);
    static bool IsMaterialExt(const std::string &ext);
    static bool IsModelExt(const std::string &ext);

    // ── Rendering helpers ────────────────────────────────────────────
    void RenderBreadcrumb(InxGUIContext *ctx);
    void RenderFolderTree(InxGUIContext *ctx);
    void RenderFolderTreeRecursive(InxGUIContext *ctx, const std::string &path, DirSnapshot *snapshot = nullptr);
    void RenderFileGrid(InxGUIContext *ctx);
    void RenderContextMenu(InxGUIContext *ctx);
    void RenderDragDropSource(InxGUIContext *ctx, const FileItem &item);
    void RenderFolderDropTarget(InxGUIContext *ctx, const std::string &folderPath);
    void RenderItemLabel(InxGUIContext *ctx, const FileItem &item, float iconSize, float cellStartX);

    // ── Click & keyboard handling ────────────────────────────────────
    void HandleItemClick(const FileItem &item, InxGUIContext *ctx);
    void HandleKeyboardShortcuts(InxGUIContext *ctx);
    void HandleExternalFileDrops();

    [[nodiscard]] bool IsCtrl(InxGUIContext *ctx) const;
    [[nodiscard]] bool IsShift(InxGUIContext *ctx) const;

    // ── Rename helpers ───────────────────────────────────────────────
    void BeginRename(const std::string &path);
    void CommitRename();
    void CancelRename();
    void CreateAndRename(const std::string &baseName, const std::string &extension,
                         std::function<std::pair<bool, std::string>(const std::string &)> createFn);

    // ── Clipboard helpers ────────────────────────────────────────────
    void ClipboardCopy(const std::vector<std::string> &paths);
    void ClipboardCut(const std::vector<std::string> &paths);
    void ClipboardPaste();
    bool HasClipboardItems() const;

    // ── Move helpers ─────────────────────────────────────────────────
    std::vector<std::string> GetDragMoveSources(const std::string &draggedPath) const;
    std::string ResolveMovePayloadPath(const std::string &payloadType, const std::string &payload) const;
    void MoveProjectItemsToFolder(const std::string &targetDir, const std::string &payloadType,
                                  const std::string &payload);

    // ── Path utility ─────────────────────────────────────────────────
    static std::string NormalizePath(const std::string &path);
    static bool IsPathWithin(const std::string &path, const std::string &parent);
    static uint64_t GetMtimeNs(const std::string &path);

    // ── Grid layout ──────────────────────────────────────────────────
    static constexpr int ICON_SIZE = 64;
    static constexpr int GRID_PADDING = 10;
    static constexpr int CELL_WIDTH = ICON_SIZE + GRID_PADDING;
    static constexpr double DIR_CACHE_TTL = 0.5; // seconds before re-checking mtime

    double m_frameTimeNow = 0.0; // steady_clock time cached once per frame

    struct GridRange
    {
        int startIndex = 0;
        int endIndex = 0;
        float topSpacer = 0.0f;
        float bottomSpacer = 0.0f;
    };
    GridRange GetVisibleGridRange(InxGUIContext *ctx, int itemCount, int cols, float rowHeight,
                                  float startY = 0.0f) const;
    float GetGridTextLineHeight(InxGUIContext *ctx);
    const LabelEntry &GetCachedItemLabel(InxGUIContext *ctx, const FileItem &item, float textRegionW);
};

} // namespace infernux
