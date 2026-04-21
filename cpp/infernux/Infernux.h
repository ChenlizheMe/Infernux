#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <core/types/InxFwdType.h>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <function/renderer/InxRenderer.h>
#include <function/scene/EditorCameraController.h>
#include <function/scene/SceneManager.h>

#include <core/error/InxError.h>
#include <core/log/InxLog.h>
#include <platform/filesystem/InxExtLoad.h>
#include <platform/filesystem/InxPath.h>
#include <platform/input/InputManager.h>

#include <function/resources/AssetDatabase/AssetDatabase.h>

namespace infernux
{

class Infernux
{
  public:
    Infernux(std::string dllPath);
    ~Infernux();

    // Prevent copying
    Infernux(const Infernux &) = delete;
    Infernux &operator=(const Infernux &) = delete;
    Infernux(Infernux &&) = delete;
    Infernux &operator=(Infernux &&) = delete;

    void Run();
    void Exit();
    void Cleanup();

    // renderer
    void InitRenderer(int width, int height, const std::string &projectPath,
                      const std::string &builtinResourcePath = "");
    void ResetImGuiLayout();
    void SelectDockedWindow(const std::string &windowId);

    // resources manager
    void ModifyResources(const std::string &filePath);
    void DeleteResources(const std::string &filePath);
    void MoveResources(const std::string &oldFilePath, const std::string &newFilePath);

    /// @brief Get the asset database instance
    /// @return Pointer to AssetDatabase, or nullptr if not initialized
    AssetDatabase *GetAssetDatabase() const;

    // ========================================================================
    // Scene Camera Control API - for Scene View with Unity-style controls
    // ========================================================================

    /// @brief Get the editor camera controller (property-based access).
    /// @return Pointer to EditorCameraController, or nullptr if not valid
    EditorCameraController *GetEditorCamera();

    /// @brief Process scene view input (call from Python when scene view is hovered/focused)
    /// @param deltaTime Time since last frame
    /// @param rightMouseDown Is right mouse button held
    /// @param middleMouseDown Is middle mouse button held
    /// @param mouseDeltaX Mouse movement X
    /// @param mouseDeltaY Mouse movement Y
    /// @param scrollDelta Mouse wheel scroll
    /// @param keyW W key held
    /// @param keyA A key held
    /// @param keyS S key held
    /// @param keyD D key held
    /// @param keyQ Q key held
    /// @param keyE E key held
    /// @param keyShift Shift key held
    void ProcessSceneViewInput(float deltaTime, bool rightMouseDown, bool middleMouseDown, float mouseDeltaX,
                               float mouseDeltaY, float scrollDelta, bool keyW, bool keyA, bool keyS, bool keyD,
                               bool keyQ, bool keyE, bool keyShift);

    // ========================================================================
    // Scene Picking API - for editor selection (screen-space to object)
    // ========================================================================

    /// @brief Pick a scene object by screen-space coordinates in the scene view
    /// @param screenX Mouse X in pixels relative to the scene viewport
    /// @param screenY Mouse Y in pixels relative to the scene viewport
    /// @param viewportWidth Scene viewport width in pixels
    /// @param viewportHeight Scene viewport height in pixels
    /// @return Picked GameObject ID, or 0 if none
    uint64_t PickSceneObjectId(float screenX, float screenY, float viewportWidth, float viewportHeight);

    /// @brief Pick all feasible scene objects under screen-space coordinates, nearest first.
    /// @param screenX Mouse X in pixels relative to the scene viewport
    /// @param screenY Mouse Y in pixels relative to the scene viewport
    /// @param viewportWidth Scene viewport width in pixels
    /// @param viewportHeight Scene viewport height in pixels
    /// @return Ordered candidate GameObject IDs (nearest first), deduplicated.
    std::vector<uint64_t> PickSceneObjectIds(float screenX, float screenY, float viewportWidth, float viewportHeight);

    /// @brief Lightweight gizmo-only handle proximity test (no scene raycast).
    /// Used every frame for hover highlighting — tests axis and plane handles.
    /// @return Gizmo handle ID, or 0 if not hovering any handle.
    uint64_t PickGizmoAxis(float screenX, float screenY, float viewportWidth, float viewportHeight);

    // ========================================================================
    // Editor Tools API — highlight + ray + mode for Python-side gizmo interaction
    // ========================================================================

    /// @brief Set the highlighted gizmo handle. 0=None, 1=X, 2=Y, 3=Z, 4=XY, 5=XZ, 6=YZ.
    void SetEditorToolHighlight(int axis);

    /// @brief Set the active tool mode. 0=None, 1=Translate, 2=Rotate, 3=Scale.
    void SetEditorToolMode(int mode);

    /// @brief Get the active tool mode. 0=None, 1=Translate, 2=Rotate, 3=Scale.
    int GetEditorToolMode() const;

    /// @brief Set local coordinate mode for editor tools (gizmo aligns to object rotation).
    void SetEditorToolLocalMode(bool local);

    /// @brief Build a world-space ray from screen coordinates (same math as picking).
    /// @return (originX, originY, originZ, dirX, dirY, dirZ)
    std::tuple<float, float, float, float, float, float> ScreenToWorldRay(float screenX, float screenY,
                                                                          float viewportWidth, float viewportHeight);

    // ========================================================================
    // Editor Gizmos API
    // ========================================================================

    /// @brief Set selection outline for a game object (Unity-style orange wireframe)
    /// @param objectId The ID of the object to show selection for, or 0 to clear
    void SetSelectionOutline(uint64_t objectId);

    /// @brief Set combined selection outline for multiple objects
    /// @param objectIds List of object IDs to show combined outline for
    void SetSelectionOutlines(const std::vector<uint64_t> &objectIds);

    /// @brief Get the currently selected object ID (0 if none)
    [[nodiscard]] uint64_t GetSelectedObjectId() const
    {
        return m_selectedObjectId;
    }

    /// @brief Clear selection outline
    void ClearSelectionOutline();

    // ========================================================================
    // Material Pipeline API
    // ========================================================================

    /// @brief Refresh a material's pipeline by reloading shaders
    /// @param material The material to refresh
    /// @return true if successful, false otherwise
    bool RefreshMaterialPipeline(std::shared_ptr<InxMaterial> material);

    /// @brief Reload a shader from file (hot-reload support)
    /// @param shaderPath The path to the shader file (.vert or .frag)
    /// @return true if successful, false otherwise
    /// @brief Reload a shader file and refresh materials using it.
    /// @return Empty string on success, or error message on failure.
    std::string ReloadShader(const std::string &shaderPath);

    /// @brief Invalidate and reload a texture after import settings change
    /// @param texturePath The texture file path whose .meta was updated
    void ReloadTexture(const std::string &texturePath);

    /// @brief Reload a mesh asset after import settings change and notify dependents
    /// @param meshPath The mesh file path whose .meta was updated
    void ReloadMesh(const std::string &meshPath);

    /// @brief Reload an audio clip asset and notify dependents
    /// @param audioPath The audio file path to reload
    void ReloadAudio(const std::string &audioPath);

    // ========================================================================
    // Preview Task System API (C++ thread pool, Python-scheduled)
    // ========================================================================

    /// @brief Initialize preview task worker threads.
    /// @param workerCount Number of workers; 0 means default(1).
    void InitPreviewTaskSystem(uint32_t workerCount = 1);

    /// @brief Shutdown preview task workers and clear pending/completed tasks.
    void ShutdownPreviewTaskSystem();

    /// @brief Schedule a material preview generation task.
    /// @param resourceKey Stable cache key (e.g. "mat|<norm_path>")
    /// @param matFilePath Material file path
    /// @param stamp Revision stamp from caller
    /// @return true if task accepted or already in-flight/ready for same/newer stamp.
    bool ScheduleMaterialPreviewTask(const std::string &resourceKey, const std::string &matFilePath, uint64_t stamp);

    /// @brief Combined query + schedule for material preview.
    ///
    /// Returns the current ImGui texture id (stale-return for anti-flicker).
    /// Internally manages a monotonic generation counter; re-renders only
    /// when the content actually changes (JSON hash or file mtime).
    ///
    /// @param materialJson  If non-empty, render from this JSON (Inspector live edits).
    /// @param fileMtimeHint If != 0 and materialJson is empty, used to detect file changes (ProjectPanel).
    uint64_t QueryOrScheduleMaterialPreview(const std::string &resourceKey, const std::string &matFilePath,
                                            const std::string &materialJson = "", uint64_t fileMtimeHint = 0);

    /// @brief Schedule a texture preview generation task (legacy wrapper).
    bool ScheduleTexturePreviewTask(const std::string &resourceKey, const std::string &textureFilePath, uint64_t stamp,
                                    bool nearest = false, bool srgb = false);

    /// @brief Pump completed preview tasks on main thread and upload textures.
    void PumpPreviewTasks();

    /// @brief Synchronously render + upload ALL queued material previews.
    /// Intended for bootstrap prewarm — ignores per-frame budget and cooldown.
    void FlushAllMaterialPreviews();

    /// @brief Get uploaded texture id for a material preview key.
    /// @return Non-zero ImGui texture id when available (stale-return for anti-flicker).
    uint64_t GetMaterialPreviewTextureId(const std::string &resourceKey) const;

    /// @brief Get uploaded texture id for a texture preview key.
    /// @return Non-zero ImGui texture id when available (stale-return for anti-flicker).
    uint64_t GetTexturePreviewTextureId(const std::string &resourceKey) const;

    /// @brief Get uploaded texture preview dimensions.
    /// @return (width,height); (0,0) when not ready.
    std::pair<int, int> GetTexturePreviewSize(const std::string &resourceKey) const;

    /// @brief Invalidate one material preview task/cache entry.
    void InvalidateMaterialPreviewTask(const std::string &resourceKey);

    /// @brief Invalidate one texture preview task/cache entry.
    void InvalidateTexturePreviewTask(const std::string &resourceKey);

    /// @brief Combined query + schedule for texture preview.
    ///
    /// Returns (textureId, width, height).  Internally manages a monotonic
    /// generation counter; re-renders only when content changes.
    ///
    /// @param contentStampHint Caller-provided content hash (mtime combo, etc.).
    ///        C++ uses this to detect changes and bump the generation counter.
    std::tuple<uint64_t, int, int> QueryOrScheduleTexturePreview(const std::string &resourceKey,
                                                                 const std::string &textureFilePath,
                                                                 uint64_t contentStampHint, bool nearest, bool srgb,
                                                                 bool pump);

    /// @brief Schedule texture preview from in-memory data (JPEG/PNG/etc.).
    ///
    /// The image bytes are decoded on a worker thread via stb_image.
    /// @param resourceKey Stable cache key
    /// @param imageData Raw encoded image bytes (JPEG, PNG, etc.)
    /// @param stamp Revision stamp
    /// @param nearest Use point filter for uploaded texture
    /// @return true if task accepted
    bool ScheduleTexturePreviewFromMemory(const std::string &resourceKey, const std::vector<unsigned char> &imageData,
                                          uint64_t stamp, bool nearest);

    /// @brief Combined query + schedule for mesh/model preview.
    ///
    /// Returns the current ImGui texture id (stale-return for anti-flicker).
    /// Internally manages a monotonic generation counter; re-renders only
    /// when the content changes (file mtime).
    ///
    /// @param resourceKey    Stable cache key (e.g. "mesh|<norm_path>")
    /// @param meshFilePath   Path to the model file (.fbx, .obj, .gltf, ...)
    /// @param fileMtimeHint  File mtime for change detection (0 = unknown)
    /// @return ImGui texture id (0 if not ready yet)
    uint64_t QueryOrScheduleMeshPreview(const std::string &resourceKey, const std::string &meshFilePath,
                                        uint64_t fileMtimeHint = 0);

    /// @brief Schedule async material save from JSON snapshot.
    /// @param key Coalescing key (usually file path)
    /// @param filePath Target .mat path
    /// @param jsonSnapshot Serialized material JSON snapshot
    /// @return true if task accepted
    bool ScheduleMaterialSaveSnapshotTask(const std::string &key, const std::string &filePath,
                                          const std::string &jsonSnapshot);

    // debug
    void SetLogLevel(LogLevel engineLevel);

    // State check
    [[nodiscard]] bool IsCleanedUp() const
    {
        return m_isCleanedUp;
    }
    [[nodiscard]] bool IsCleaningUp() const
    {
        return m_isCleaningUp;
    }
    [[nodiscard]] bool IsInitialized() const
    {
        return m_renderer != nullptr && !m_isCleanedUp;
    }

    /// @brief Get the renderer subsystem for direct access.
    /// @return Pointer to InxRenderer, or nullptr if not initialized / cleaned up
    [[nodiscard]] InxRenderer *GetRenderer() const
    {
        return (m_isCleanedUp || !m_renderer) ? nullptr : m_renderer.get();
    }

  private:
    /// @brief Check if engine is valid for operations
    [[nodiscard]] bool CheckEngineValid(const char *operation) const;

    /// @brief Ensure a shader is loaded in the renderer
    /// @param shaderId The shader_id to check/load
    /// @param shaderType "vertex" or "fragment"
    /// @return true if shader is loaded, false otherwise
    bool EnsureShaderLoaded(const std::string &shaderId, const std::string &shaderType);

    /// @brief Load shaders from a directory via AssetRegistry and register them with the renderer.
    /// @param dir Directory to scan for .vert/.frag files
    /// @param recursive If true, scan subdirectories recursively
    void LoadAndRegisterShaders(const std::string &dir, bool recursive);

    /// @brief Push a compiled ShaderAsset into the renderer (SPIR-V + variants + render meta).
    void RegisterShaderToRenderer(const struct ShaderAsset &asset);

    struct PreviewTaskItem
    {
        std::function<void()> fn;
    };

    struct MaterialPreviewCompleted
    {
        std::string resourceKey;
        uint64_t stamp = 0;
        int size = 0;
        bool success = false;
        std::vector<unsigned char> pixels;
    };

    struct TexturePreviewCompleted
    {
        std::string resourceKey;
        uint64_t generation = 0;
        int width = 0;
        int height = 0;
        bool nearest = false;
        bool success = false;
        std::vector<unsigned char> pixels;
    };

    struct MaterialPreviewRequest
    {
        std::string resourceKey;
        std::string matFilePath;
        uint64_t generation = 0;
        std::string materialJson; ///< If non-empty, render from this JSON instead of reading matFilePath from disk.
    };

    struct TexturePreviewRequest
    {
        std::string resourceKey;
        std::string textureFilePath;
        uint64_t generation = 0;
        bool nearest = false;
        bool srgb = false;
    };

    struct MeshPreviewCompleted
    {
        std::string resourceKey;
        uint64_t generation = 0;
        int size = 0;
        bool success = false;
        std::vector<unsigned char> pixels;
    };

    struct MeshPreviewRequest
    {
        std::string resourceKey;
        std::string meshFilePath;
        uint64_t generation = 0;
    };

    struct MaterialPreviewState
    {
        uint64_t generation = 0;      ///< Monotonic counter, bumped on detected content change
        uint64_t readyGeneration = 0; ///< Generation of last completed render
        uint64_t lastJsonHash = 0;    ///< std::hash of last JSON string seen
        uint64_t lastFileMtime = 0;   ///< Last file mtime seen from ProjectPanel
        bool inFlight = false;
        int readySize = 0;
        std::string textureName;
        uint64_t textureId = 0;
    };

    struct TexturePreviewState
    {
        uint64_t generation = 0;       ///< Monotonic counter, bumped on detected content change
        uint64_t readyGeneration = 0;  ///< Generation of last completed render
        uint64_t lastContentStamp = 0; ///< Last content stamp seen from caller
        bool inFlight = false;
        int readyWidth = 0;
        int readyHeight = 0;
        std::string textureName;
        uint64_t textureId = 0;
        bool nearest = false;
        bool srgb = false;
    };

    struct MeshPreviewState
    {
        uint64_t generation = 0;
        uint64_t readyGeneration = 0;
        uint64_t lastFileMtime = 0;
        bool inFlight = false;
        int readySize = 0;
        std::string textureName;
        uint64_t textureId = 0;
        std::string meshFilePath; ///< Absolute path to model file
    };

    void EnqueuePreviewTask(std::function<void()> fn);
    static std::string BuildPreviewTextureName(const std::string &resourceKey);
    static std::string BuildTexturePreviewTextureName(const std::string &resourceKey);
    static std::string BuildMeshPreviewTextureName(const std::string &resourceKey);

    InxAppMetadata m_metadata{"Infernux", 0, 1, 0, "com.infrenderer.Infernux"};

    std::unique_ptr<InxExtLoad> m_extLoader;
    std::unique_ptr<AssetDatabase> m_assetDatabase;
    std::unique_ptr<InxRenderer> m_renderer;

    LogLevel m_logLevel = LogLevel::LOG_INFO;
    bool m_isCleanedUp = false;
    bool m_isCleaningUp = false;

    // Selection tracking for outline updates
    uint64_t m_selectedObjectId = 0;
    std::vector<uint64_t> m_cachedOutlineIds; ///< Last set of IDs passed to SetSelectionOutlines

    // ImGui ini file path — stored as std::filesystem::path so that
    // wide-char paths (e.g. Chinese usernames) work correctly on Windows.
    std::filesystem::path m_imguiIniPath;

    // Preview task system
    std::vector<std::thread> m_previewWorkers;
    std::queue<PreviewTaskItem> m_previewTaskQueue;
    mutable std::mutex m_previewTaskMutex;
    std::condition_variable m_previewTaskCv;
    bool m_previewStopRequested = false;
    bool m_previewTaskSystemInitialized = false;

    mutable std::mutex m_previewResultMutex;
    std::queue<MaterialPreviewCompleted> m_previewCompletedQueue;
    std::queue<TexturePreviewCompleted> m_texturePreviewCompletedQueue;
    std::queue<MeshPreviewCompleted> m_meshPreviewCompletedQueue;
    std::queue<MaterialPreviewRequest> m_previewRequestQueue;
    std::queue<MeshPreviewRequest> m_meshPreviewRequestQueue;
    std::unordered_map<std::string, MaterialPreviewState> m_materialPreviewStates;
    std::unordered_map<std::string, TexturePreviewState> m_texturePreviewStates;
    std::unordered_map<std::string, MeshPreviewState> m_meshPreviewStates;
    int m_lastPumpFrame = -1;
    std::chrono::steady_clock::time_point m_lastMaterialRenderTime{};

    void LoadImGuiLayout();
    void SaveImGuiLayout();
};
} // namespace infernux