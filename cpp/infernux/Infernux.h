#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <core/threading/FrameWorkOwner.h>
#include <core/threading/JobSystem.h>
#include <core/types/InxFwdType.h>
#include <core/types/ShaderProgramArtifact.h>
#include <deque>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <function/renderer/InxRenderer.h>
#include <function/scene/EditorCameraController.h>
#include <function/scene/SceneManager.h>

#include <core/error/InxError.h>
#include <core/log/InxLog.h>
#include <platform/filesystem/InxPath.h>
#include <platform/input/InputManager.h>

#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/AssetRuntimeRecord.h>

namespace infernux
{

class AssetLoadTicket;
struct MeshPreviewPreparation;
struct TimelineCubePreviewCandidate;
class LinkedShaderProgramLoadTicket;

struct PreviewPixelSummary
{
    uint64_t hash = 0;
    uint32_t nonTransparentPixelCount = 0;
    uint8_t minRgb = 0;
    uint8_t maxRgb = 0;
};

enum class RuntimeMode
{
    Graphical,
    Headless,
};

struct PresentationReceipt
{
    uint64_t sequence = 0;
    uint64_t hostOpportunity = 0;
    uint64_t acceptedNs = 0;
};

/// Two-phase linked-program preload. Worker jobs only produce CPU artifacts;
/// TryCommitLinkedShaderPrograms performs every renderer mutation on the
/// engine owner thread.
class LinkedShaderProgramLoadTicket final
{
  public:
    [[nodiscard]] bool IsComplete() const noexcept;
    [[nodiscard]] bool IsCommitted() const noexcept;
    [[nodiscard]] bool IsSuperseded() const noexcept;
    [[nodiscard]] bool WasProducedOnWorker() const noexcept;
    bool Cancel() noexcept;

  private:
    struct SourceIdentity;
    struct State;
    friend class Infernux;
    std::shared_ptr<State> m_state;
};

class Infernux
{
  public:
    Infernux(std::string dllPath, RuntimeMode mode = RuntimeMode::Graphical);
    ~Infernux();

    // Prevent copying
    Infernux(const Infernux &) = delete;
    Infernux &operator=(const Infernux &) = delete;
    Infernux(Infernux &&) = delete;
    Infernux &operator=(Infernux &&) = delete;

    void Run();
    void Tick(float deltaTime);
    void SetPreSceneUpdateCallback(std::function<void(float)> callback);
    void SetEndFrameCallback(std::function<void(bool)> callback);
    void SetRuntimeFrameCallbacks(std::function<void(float)> beforeWork, std::function<void()> afterWork);
    [[nodiscard]] FrameWorkOwner &GetFrameWorkOwner() noexcept
    {
        return m_frameWork;
    }
    [[nodiscard]] uint64_t GetSubmittedRenderFrameCount() const noexcept
    {
        return m_submittedRenderFrames;
    }
    [[nodiscard]] const PresentationReceipt &GetLastPresentation() const noexcept
    {
        return m_lastPresentation;
    }
    void Exit();
    [[nodiscard]] bool IsExitRequested() const
    {
        return m_exitRequested.load(std::memory_order_acquire);
    }
    void Cleanup();
    void RequireCleanupBoundary() const;
    void RequireFrameInactive() const;
    [[nodiscard]] std::unique_ptr<rhi::ComputeHost> AcquireComputeHost();
    [[nodiscard]] bool HasGpuCompute() const noexcept;
    [[nodiscard]] std::shared_ptr<rhi::RenderTexture> CreateRenderTexture(const rhi::RenderTextureDesc &description);
    [[nodiscard]] std::shared_ptr<rhi::RenderTexture> LoadRenderTexture(const std::string &guid);
    void RequireComputeHostsReleased() const;

    void InitHeadless(const std::string &projectPath, const std::string &builtinResourcePath = "",
                      std::function<void()> prepareSettings = {}, bool gpu = false);

    [[nodiscard]] RuntimeMode GetRuntimeMode() const noexcept
    {
        return m_runtimeMode;
    }

    /// Wall-clock timings captured during the most recent graphical startup.
    [[nodiscard]] const std::unordered_map<std::string, double> &GetStartupPhaseTimingsMs() const noexcept
    {
        return m_startupPhaseTimingsMs;
    }

    // renderer
    void InitRenderer(int width, int height, const std::string &projectPath,
                      const std::string &builtinResourcePath = "", std::function<void()> prepareSettings = {});
    void ResetImGuiLayout();
    void SelectDockedWindow(const std::string &windowId, bool allowDuringModal = false);

    // Automation input is queued and consumed by the graphical event loop.
    // It is kept separate from gameplay's InputManager query API.
    uint64_t QueueSyntheticKeyInput(int scancode, bool pressed, bool repeat = false);
    uint64_t QueueSyntheticMouseButtonInput(int button, bool pressed, float x, float y);
    uint64_t QueueSyntheticMouseMotionInput(float x, float y, float deltaX, float deltaY);
    uint64_t QueueSyntheticMouseWheelInput(float horizontal, float vertical);
    uint64_t QueueSyntheticTextInput(const std::string &text);
    uint64_t QueueSyntheticCloseRequest();
    [[nodiscard]] uint64_t GetLastProcessedSyntheticInputSequence() const;
    [[nodiscard]] size_t GetPendingSyntheticInputCount() const;

    /// @brief Get the asset database instance
    /// @return Pointer to AssetDatabase, or nullptr if not initialized
    AssetDatabase *GetAssetDatabase() const;
    [[nodiscard]] std::vector<AssetRuntimeRecord> GetAssetRuntimeRecords() const;

    /// Compile linked shader programs referenced by already-loaded materials
    /// on the engine JobSystem, then publish them explicitly on the owner
    /// thread before a scene becomes live.
    [[nodiscard]] std::shared_ptr<LinkedShaderProgramLoadTicket>
    BeginPrepareLinkedShaderPrograms(const std::vector<std::string> &materialGuids);
    bool TryCommitLinkedShaderPrograms(const std::shared_ptr<LinkedShaderProgramLoadTicket> &ticket);

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

    /// @brief Return the icon owners whose rendered Scene-view quads contain
    /// the cursor, nearest first. This lets asynchronous GPU mesh picking
    /// preserve an intentional click on a component icon owned by a mesh.
    std::vector<uint64_t> PickSceneIconObjectIds(float screenX, float screenY, float viewportWidth,
                                                 float viewportHeight);

    /// @brief Lightweight gizmo-only handle proximity test (no scene raycast).
    /// Used every frame for hover highlighting — tests axis and plane handles.
    /// @return Gizmo handle ID, or 0 if not hovering any handle.
    uint64_t PickGizmoAxis(float screenX, float screenY, float viewportWidth, float viewportHeight);

    // ========================================================================
    // Editor Tools API — highlight + ray + mode for Python-side gizmo interaction
    // ========================================================================

    /// @brief Set the highlighted gizmo handle. 0=None, 1=X, 2=Y, 3=Z, 4=XY, 5=XZ, 6=YZ.
    void SetEditorToolHighlight(int axis);

    /// @brief Set the active tool mode. 0=None, 1=Translate, 2=Rotate, 3=Scale, 4=Rect.
    void SetEditorToolMode(int mode);

    /// @brief Get the active tool mode. 0=None, 1=Translate, 2=Rotate, 3=Scale, 4=Rect.
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

    /// Query published standalone stages and linked material programs without loading anything.
    [[nodiscard]] bool IsShaderLoaded(const std::string &shaderId, const std::string &shaderType) const;

    /// @brief Reload a shader from file (hot-reload support)
    /// @param shaderPath The path to the shader file (.vert or .frag)
    /// @return true if successful, false otherwise
    /// @brief Reload a shader file and refresh materials using it.
    /// @return Empty string on success, or error message on failure.
    std::string ReloadShaderRuntime(const std::string &shaderPath, const std::string &previousShaderId,
                                    const std::string &previousSourcePath = "");
    const std::vector<std::string> &GetShaderReloadSources() const
    {
        return m_shaderReloadSources;
    }

    /// @brief Invalidate and reload a texture after import settings change
    /// @param texturePath The texture file path whose .meta was updated
    void ReloadTexture(const std::string &texturePath);

    /// @brief Reload a mesh asset after import settings change and notify dependents
    /// @param meshPath The mesh file path whose .meta was updated
    void ReloadMesh(const std::string &meshPath);

    /// @brief Reload an audio clip asset and notify dependents
    /// @param audioPath The audio file path to reload
    void ReloadAudio(const std::string &audioPath);

    /// @brief Combined query + schedule for material preview.
    ///
    /// The last published image stays visible while the current content is prepared.
    ///
    /// @param matFilePath Imported material or model material slot when materialJson is empty.
    /// @param materialJson Inline document for live edits.
    /// @param fileMtimeHint Optional file change hint for the imported source.
    uint64_t QueryOrScheduleMaterialPreview(const std::string &resourceKey, const std::string &matFilePath,
                                            const std::string &materialJson = "", uint64_t fileMtimeHint = 0,
                                            bool authoring = false);

    /// @brief Release Inspector ownership while preserving the shared preview texture.
    void ReleasePreviewAuthoring(const std::string &resourceKey);

    struct PreviewTaskSnapshot
    {
        std::string kind;
        std::string resourceKey;
        std::string textureName;
        uint64_t generation = 0;
        uint64_t readyGeneration = 0;
        uint64_t pendingUploadVersion = 0;
        uint64_t pendingPreviewGeneration = 0;
        uint64_t publishedUploadVersion = 0;
        uint64_t failedUploadVersion = 0;
        uint64_t textureId = 0;
        bool inFlight = false;
        bool authoring = false;
        bool hasRenderTicket = false;
        bool renderTicketDone = false;
        int pendingWidth = 0;
        int pendingHeight = 0;
        int readyWidth = 0;
        int readyHeight = 0;
        uint64_t pixelGeneration = 0;
        uint64_t pixelHash = 0;
        uint32_t nonTransparentPixelCount = 0;
        uint8_t minRgb = 0;
        uint8_t maxRgb = 0;
        uint32_t imguiDrawCommandCount = 0;
    };

    /// @brief Read-only diagnostics for active preview state machines.
    std::vector<PreviewTaskSnapshot> GetPreviewTaskSnapshots() const;

    /// @brief Pump completed preview tasks on main thread and upload textures.
    void PumpPreviewTasks();

    /// @brief Poll asynchronous GPU completion without waiting for the device.
    void PollGpuCompletions();
    [[nodiscard]] bool IsGpuSubmissionCompleted(rhi::SubmissionTicket ticket) const noexcept;

    /// @brief Get uploaded texture id for a material preview key.
    /// @return Non-zero ImGui texture id when available (stale-return for anti-flicker).
    uint64_t GetMaterialPreviewTextureId(const std::string &resourceKey) const;

    /// @brief Check whether the uploaded material preview matches the latest requested generation.
    bool IsMaterialPreviewReady(const std::string &resourceKey) const;

    /// @brief Get uploaded texture id for a texture preview key.
    /// @return Non-zero ImGui texture id when available (stale-return for anti-flicker).
    uint64_t GetTexturePreviewTextureId(const std::string &resourceKey) const;

    /// @brief Get uploaded texture preview dimensions.
    /// @return (width,height); (0,0) when not ready.
    std::pair<int, int> GetTexturePreviewSize(const std::string &resourceKey) const;

    /// @brief Get an already uploaded mesh preview without scheduling work.
    /// Model thumbnails are imported asynchronously; this accessor is intentionally
    /// a cache lookup so FileManager can stay responsive while an FBX is loading.
    uint64_t GetMeshPreviewTextureId(const std::string &resourceKey) const;

    /// @brief Invalidate one material preview task/cache entry.
    void InvalidateMaterialPreviewTask(const std::string &resourceKey);

    /// @brief Invalidate one texture preview task/cache entry.
    void InvalidateTexturePreviewTask(const std::string &resourceKey);

    /// @brief Retire decoded/uploaded texture data and pending publications.
    /// Retains only a generation tombstone so late jobs cannot revive the key.
    void ReleaseTexturePreviewTask(const std::string &resourceKey);
    /// Retire queued/rendering material work and its published thumbnail.
    void ReleaseMaterialPreviewTask(const std::string &resourceKey);
    /// Retire all previews (including embedded model products) for a deleted source path.
    void ReleaseAssetPreviewTasks(const std::string &assetPath);

    /// @brief Combined query + schedule for texture preview.
    ///
    /// Returns (textureId, width, height).  Internally manages a monotonic
    /// generation counter; re-renders only when content changes.
    ///
    /// @param contentStampHint Caller-provided content hash (mtime combo, etc.).
    ///        C++ uses this to detect changes and bump the generation counter.
    /// @param useImportedTexture Use the game asset publication. Document
    ///        illustrations set false to preserve the source image's pixels.
    std::tuple<uint64_t, int, int> QueryOrScheduleTexturePreview(const std::string &resourceKey,
                                                                 const std::string &textureFilePath,
                                                                 uint64_t contentStampHint, bool nearest, bool srgb,
                                                                 int maxSize, const std::string &textureFormat,
                                                                 const std::string &textureType, bool authoring,
                                                                 bool pump, bool useImportedTexture = true);

    /// @brief Schedule texture preview from in-memory data (JPEG/PNG/etc.).
    ///
    /// The image bytes are decoded on a worker thread via stb_image.
    /// @param resourceKey Stable cache key
    /// @param imageData Raw encoded image bytes (JPEG, PNG, etc.)
    /// @param stamp Revision stamp
    /// @param nearest Use point filter for uploaded texture
    /// @return true if task accepted
    bool ScheduleTexturePreviewFromMemory(const std::string &resourceKey, std::vector<unsigned char> imageData,
                                          uint64_t stamp, bool nearest);

    /// @brief Combined query + schedule for mesh/model preview.
    ///
    /// Returns the current ImGui texture id (stale-return for anti-flicker).
    /// Internally manages a monotonic generation counter; re-renders only
    /// when the asset or any of its published dependencies changes.
    ///
    /// @param resourceKey    Stable cache key (e.g. "mesh|<norm_path>")
    /// @param meshFilePath   Path to the model file (.fbx, .obj, .gltf, ...)
    /// @param fileMtimeHint  File mtime for change detection (0 = unknown)
    /// @return ImGui texture id (0 if not ready yet)
    uint64_t QueryOrScheduleMeshPreview(const std::string &resourceKey, const std::string &meshFilePath,
                                        uint64_t fileMtimeHint = 0);
    /// Read the revision of an imported catalog path (including model subresources), without disk IO.
    [[nodiscard]] uint64_t GetPreviewDependencyRevision(const std::string &path) const;

    /// @brief Queue a Timeline cube preview render (non-blocking). Returns the latest
    ///        ImGui texture id immediately; GPU work runs in PumpPreviewTasks().
    uint64_t RenderTimelineCubePreview(float px, float py, float pz, float rx, float ry, float rz, float sx, float sy,
                                       float sz, float camYaw, float camPitch, float camDistance, int size);

    /// @brief Execute a pending Timeline cube preview render if one was queued this frame.
    bool PumpTimelineCubePreviewIfDirty();
    uint64_t RenderModelAnimationPreview(const std::shared_ptr<InxMesh> &mesh, const std::string &take, float seconds,
                                         int size, uint64_t dependencyRevision);
    int AdvanceMaterialPreview(bool gpu, int uploadBudget, bool &progressed);
    bool AdvancePreviewTasks(bool gpu, uint64_t lane);
    FrameWorkFacts DescribePreviewWork();

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
        return m_isInitialized && !m_isCleanedUp;
    }

    /// @brief Get the renderer subsystem for direct access.
    /// @return Pointer to InxRenderer, or nullptr if not initialized / cleaned up
    [[nodiscard]] InxRenderer *GetRenderer() const
    {
        return (m_isCleanedUp || !m_renderer) ? nullptr : m_renderer.get();
    }

  private:
    void StartFrameDriver();
    void TickFrame(float deltaTime);
    /// @brief Check if engine is valid for operations
    [[nodiscard]] bool CheckEngineValid(const char *operation) const;
    void RegisterAssetLoaders();
    void RegisterAssetCallbacks();

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

    struct LinkedShaderProgramPreparation
    {
        bool usesLinkedArtifact = false;
        bool success = true;
        std::string error;
    };

    std::string ReloadShaderDependencies(const std::string &shaderPath, const std::string &previousSourcePath);
    std::string ReloadShaderAuthorSource(const std::string &shaderPath);
    std::string ReloadShaderSourceBatch(const std::vector<std::string> &roots);
    // Sources checked by the last synchronous candidate publication. Diagnostic
    // owners use this receipt to retire only errors resolved by that result.
    std::vector<std::string> m_shaderReloadSources;

    struct LinkedShaderProgramCacheEntry
    {
        uint64_t sourceStamp = 0;
        ShaderProgramKey programKey;
        uint64_t failedSourceStamp = 0;
        std::string lastError;
        bool failureReported = false;
        // Compiled UI candidates wait for an actual draw to acquire GPU ownership.
        std::shared_ptr<const ShaderProgramArtifact> preparedArtifact;
        uint64_t sourceEnvironmentRevision = 0;
    };

    [[nodiscard]] LinkedShaderProgramPreparation EnsureLinkedShaderProgramArtifact(const ShaderStagePair &stages);
    [[nodiscard]] LinkedShaderProgramPreparation EnsureLinkedShaderProgramArtifact(const ShaderStagePair &stages,
                                                                                   const std::string &vertexPath,
                                                                                   const std::string &fragmentPath,
                                                                                   bool requireCurrentSource = false);
    [[nodiscard]] LinkedShaderProgramPreparation
    EnsureLinkedShaderProgramArtifact(const std::shared_ptr<InxMaterial> &material, bool requireCurrentSource = false);
    [[nodiscard]] std::optional<ShaderProgramDomain>
    InspectMaterialShaderDomain(const std::shared_ptr<InxMaterial> &material) const;

    std::unordered_map<ShaderStagePair, LinkedShaderProgramCacheEntry, ShaderStagePairHash> m_linkedShaderProgramCache;
    // Concurrent scene loads share the same immutable source identity; a
    // changed identity retires old work without retaining jobs/artifacts.
    std::unordered_map<ShaderStagePair, std::shared_ptr<const LinkedShaderProgramLoadTicket::SourceIdentity>,
                       ShaderStagePairHash>
        m_linkedShaderPreparations;

    struct TexturePreviewCompleted
    {
        std::string resourceKey;
        uint64_t generation = 0;
        bool nearest = false;
        bool displayEncoding = false;
        std::shared_ptr<FrameCpuStorageClaim> storage;
        std::shared_ptr<const TextureCpuData> pixels;
        PreviewPixelSummary summary;
    };

    struct MaterialPreviewRaster;

    struct MaterialPreviewRequest
    {
        std::string resourceKey;
        std::string matFilePath;
        uint64_t generation = 0;
        std::string materialJson; ///< Live document when the asset is being edited.
        std::shared_ptr<AssetLoadTicket> loadTicket;
        std::shared_ptr<InxMaterial> material;
        std::shared_ptr<MaterialPreviewRaster> raster;
        std::string assetGuid;
        uint64_t assetContentGeneration = 0;
        uint64_t materialVersion = 0;
        std::vector<std::pair<std::string, uint64_t>> inputRevisions;
    };

    struct TexturePreviewRequest
    {
        std::string resourceKey;
        std::string textureFilePath;
        uint64_t generation = 0;
        bool nearest = false;
        bool srgb = false;
        int maxSize = 256;
        std::string textureFormat = "auto";
        std::string textureType = "default";
    };

    struct TexturePreviewSource;

    struct MeshPreviewRequest
    {
        std::string resourceKey;
        std::string meshFilePath;
        uint64_t generation = 0;
        std::shared_ptr<MeshPreviewPreparation> preparation;
        std::shared_ptr<PreviewGeometryUpload> geometry;
    };

    struct PendingPreviewUpload
    {
        uint64_t version = 0;
        uint64_t generation = 0;
        int width = 0;
        int height = 0;
        PreviewPixelSummary summary;
    };

    struct MaterialPreviewState
    {
        uint64_t generation = 0;       ///< Monotonic counter, bumped on detected content change
        uint64_t readyGeneration = 0;  ///< Generation of last completed render
        uint64_t failedGeneration = 0; ///< Rejected source is retried only after a change
        uint64_t lastFileMtime = 0;    ///< Last file mtime seen from ProjectPanel
        uint64_t lastDependencyRevision = 0;
        PendingPreviewUpload upload;
        uint64_t pendingGeneration = 0;
        bool authoring = false;
        uint64_t renderGeneration = 0;
        std::shared_ptr<PreviewImage> renderTicket;
        std::shared_ptr<const std::vector<std::pair<std::string, uint64_t>>> submittedInputs;
        std::string textureName;
        std::optional<PreviewPixelSummary> readyPixels;
        // Latest accepted source for this key. A newer authoring revision may
        // arrive while an older GPU readback/upload is still in flight; keep
        // the source here so the preview pump can immediately continue with
        // the newest generation instead of waiting for another UI query.
        std::string latestMatFilePath;
        std::string latestMaterialJson;

        bool HasPendingWork() const noexcept
        {
            return pendingGeneration || renderTicket || upload.version;
        }
    };

    struct TexturePreviewState
    {
        uint64_t generation = 0;       ///< Monotonic counter, bumped on detected content change
        uint64_t readyGeneration = 0;  ///< Generation of last completed render
        uint64_t failedGeneration = 0; ///< Retry decode only when the source/settings change
        uint64_t lastContentStamp = 0; ///< Last content stamp seen from caller
        uint64_t decodeGeneration = 0;
        std::shared_ptr<const TexturePreviewRequest> fileRequest;
        PendingPreviewUpload upload;
        bool authoring = false;
        std::string textureName;
        std::string sourcePath;
        std::string importedTextureGuid;
        uint64_t importedContentGeneration = 0;
        uint64_t importedRuntimeVersion = 0;
        bool importedGpuPending = false;
        bool nearest = false;
        bool srgb = false;
        int maxSize = 256;
        std::string textureFormat = "auto";
        std::string textureType = "default";
        std::optional<PreviewPixelSummary> readyPixels;

        bool HasPendingWork() const noexcept
        {
            return decodeGeneration || upload.version || importedGpuPending;
        }
    };

    struct MeshPreviewState
    {
        uint64_t generation = 0;
        uint64_t readyGeneration = 0;
        uint64_t failedGeneration = 0;
        uint64_t lastFileMtime = 0;
        uint64_t lastDependencyRevision = 0;
        uint64_t pendingGeneration = 0;
        uint64_t renderGeneration = 0;
        std::shared_ptr<PreviewImage> renderTicket;
        std::string textureName;
        std::string meshFilePath; ///< Absolute path to model file
    };

    enum class PreviewStep
    {
        Wait,
        Host,
        Gpu
    };
    // Caller holds m_previewResultMutex.
    PreviewStep DescribeMaterialPreviewRequest(const MaterialPreviewRequest &request) const;
    PreviewStep DescribeMeshPreviewRequest(const MeshPreviewRequest &request) const;
    bool IsMaterialPreviewSourceCurrent(const MaterialPreviewRequest &request) const;
    void BumpMaterialPreviewGeneration(MaterialPreviewState &state);
    bool RefreshMaterialPreviewDependency(MaterialPreviewState &state);
    bool RefreshMeshPreviewDependency(MeshPreviewState &state) const;
    void ObserveSubmittedPreviews();
    bool PrepareMaterialPreview(MaterialPreviewRequest &request, bool &progressed);
    bool FinishMeshPreviewPreparation(const MeshPreviewRequest &request, std::shared_ptr<PreviewImage> image);
    bool PrepareMeshPreview(MeshPreviewRequest &request, bool gpu, std::shared_ptr<InxMesh> &mesh,
                            std::vector<std::shared_ptr<InxMaterial>> &materials, bool &progressed);
    void EnqueuePreviewTask(std::function<void()> fn, std::optional<std::weak_ptr<const void>> owner = std::nullopt);
    void EnqueuePreviewTask(std::function<void(std::shared_ptr<FrameCpuStorageClaim>)> fn, uint64_t storageBytes,
                            std::optional<std::weak_ptr<const void>> owner);
    void QueueTextureFilePreview(const std::shared_ptr<const TexturePreviewRequest> &request);
    void DecodeTextureFilePreview(const TexturePreviewRequest &request, int decodeSize,
                                  const TexturePreviewSource *source = nullptr,
                                  std::shared_ptr<FrameCpuStorageClaim> storage = {},
                                  const std::function<bool()> &retired = {});
    void CompleteTexturePreview(TexturePreviewCompleted completed);
    void RetireTexturePreviewDecode(const std::string &key, TexturePreviewState &state);
    FrameWorkSource GetPreviewIoWorkSource();
    void ObservePreviewJob();
    static std::string BuildPreviewTextureName(const std::string &resourceKey);
    static std::string BuildTexturePreviewTextureName(const std::string &resourceKey);
    static std::string BuildMeshPreviewTextureName(const std::string &resourceKey);
    bool CommitPublishedPreviewTextures();
    // Caller holds m_previewResultMutex. Each key has one pending preparation.
    void ScheduleDirtyMaterialPreview(const std::string &key, MaterialPreviewState &state);
    void ScheduleDirtyMeshPreview(const std::string &key, MeshPreviewState &state);
    void DrainPreviewJobs();
    void AcquireRuntimeServices();
    void StartRuntimeServices();
    void ReleaseRuntimeServices() noexcept;
    void CreateRenderer();
    void FinishFrame(const RenderFrameResult &result);
    void BeginRuntimeFrame(float deltaTime);
    [[nodiscard]] FrameGpuAvailability BackgroundGpuAvailability(bool foregroundReady);

    InxAppMetadata m_metadata{"Infernux", 0, 1, 0, "com.infrenderer.Infernux"};
    RuntimeMode m_runtimeMode = RuntimeMode::Graphical;

    std::unique_ptr<AssetDatabase> m_assetDatabase;
    std::unique_ptr<vk::VulkanBackendContext> m_gpuServices;
    std::unique_ptr<InxRenderer> m_renderer;

    LogLevel m_logLevel = LogLevel::LOG_INFO;
    bool m_isCleanedUp = false;
    bool m_isCleaningUp = false;
    bool m_isInitialized = false;
    bool m_ownsRuntimeServices = false;
    bool m_ownsJobSystem = false;
    std::unordered_map<std::string, double> m_startupPhaseTimingsMs;
    std::atomic<bool> m_exitRequested{false};
    /// True while Run() owns the frame loop, including its headless Tick calls.
    std::atomic<bool> m_runLoopActive{false};
    bool m_tickActive = false;
    std::mutex m_runMutex;
    std::condition_variable m_runCv;
    std::function<void(float)> m_preSceneUpdateCallback;
    std::function<void(bool)> m_endFrameCallback;
    std::function<void(float)> m_runtimeBeforeWork;
    std::function<void()> m_runtimeAfterWork;
    FrameWorkOwner m_frameWork;
    uint64_t m_submittedRenderFrames = 0;
    PresentationReceipt m_lastPresentation;

    // Selection tracking for outline updates
    uint64_t m_selectedObjectId = 0;

    // ImGui ini file path — stored as std::filesystem::path so that
    // wide-char paths (e.g. Chinese usernames) work correctly on Windows.
    std::filesystem::path m_imguiIniPath;
    std::filesystem::path m_imguiLayoutMetadataPath;

    struct PendingPreviewJob
    {
        std::function<void(std::shared_ptr<FrameCpuStorageClaim>)> run;
        std::optional<std::weak_ptr<const void>> owner;
        uint64_t storageBytes = 0;
        bool IsRetired() const noexcept
        {
            return owner && owner->expired();
        }
    };
    std::deque<PendingPreviewJob>::iterator FindReadyPreviewJob();
    std::deque<PendingPreviewJob> m_previewJobs;
    JobHandle m_previewDispatcherJob;
    std::mutex m_previewJobMutex;
    bool m_acceptPreviewJobs = true;
    uint64_t m_previewExecutionNanoseconds = 0;
    uint64_t m_previewJobFrame = 0, m_previewJobGrant = 0;

    mutable std::mutex m_previewResultMutex;
    std::deque<TexturePreviewCompleted> m_texturePreviewCompletedQueue;
    std::deque<MaterialPreviewRequest> m_materialPreviewRequestQueue;
    std::deque<MeshPreviewRequest> m_meshPreviewRequestQueue;
    std::unordered_map<std::string, MaterialPreviewState> m_materialPreviewStates;
    std::unordered_map<std::string, TexturePreviewState> m_texturePreviewStates;
    std::unordered_map<std::string, MeshPreviewState> m_meshPreviewStates;
    std::atomic_bool m_hasPendingPreviewUploads{false};
    std::atomic_bool m_hasPreviewPumpWork{false};
    FrameWorkOwner::SourceId m_previewWorkSource = 0;
    FrameWorkOwner::SourceId m_previewIoWorkSource = 0;
    FrameWorkOwner::SourceId m_textureWorkSource = 0;
    FrameWorkOwner::SourceId m_assetLoadWorkSource = 0;
    FrameWorkOwner::SourceId m_assetCatalogWorkSource = 0;
    FrameWorkOwner::SourceId m_documentWriteWorkSource = 0;
    uint64_t m_nextPreviewLane = 0;
    uint64_t m_nextPreviewHostLane = 0;
    std::string m_lastImportedPreviewKey;

    // Timeline-editor cube preview: query schedules, PumpTimelineCubePreviewIfDirty renders.
    uint64_t m_cubePreviewTexId = 0;
    uint64_t m_lastCubePreviewHash = 0;
    uint64_t m_pendingCubePreviewHash = 0;
    bool m_timelineCubeDirty = false;
    float m_pendingCubePx = 0, m_pendingCubePy = 0, m_pendingCubePz = 0;
    float m_pendingCubeRx = 0, m_pendingCubeRy = 0, m_pendingCubeRz = 0;
    float m_pendingCubeSx = 1, m_pendingCubeSy = 1, m_pendingCubeSz = 1;
    float m_pendingCubeCamYaw = 0, m_pendingCubeCamPitch = 0, m_pendingCubeCamDist = 6;
    int m_pendingCubeSize = 240;
    std::shared_ptr<InxMaterial> m_cubePreviewCubeMat;
    std::shared_ptr<InxMaterial> m_cubePreviewFloorMat;
    std::unique_ptr<TimelineCubePreviewCandidate> m_cubePreviewCandidate;
    std::vector<Vertex> m_cubePreviewFloorVerts;
    std::vector<uint32_t> m_cubePreviewFloorIndices;
    bool m_cubePreviewFloorBuilt = false;

    bool ExecuteTimelineCubePreviewRender(float px, float py, float pz, float rx, float ry, float rz, float sx,
                                          float sy, float sz, float camYaw, float camPitch, float camDistance, int size,
                                          uint64_t hash, bool &progressed);

    void LoadImGuiLayout();
    void SaveImGuiLayout();
};
} // namespace infernux
