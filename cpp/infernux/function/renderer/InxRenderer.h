#pragma once

// Minimal includes required for public API value types and POD members
#include "InxRenderStruct.h"
#include "ProfileConfig.h"
#include <array>
#include <chrono>
#include <core/log/InxLog.h>           // LogLevel enum (used in SetLogLevel)
#include <core/types/InxApplication.h> // InxAppMetadata value type (public methods)
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <vector>

namespace infernux
{
// ============================================================================
// Forward declarations for private subsystem types.
// Full definitions are in InxRenderer.cpp. Keeping them out of this header
// eliminates the transitive include fan-out that made every InxRenderer.h
// consumer recompile when any subsystem changed.
// ============================================================================
class EditorGizmos;
class EditorTools;
class GizmosDrawCallBuffer;
class InxGUI;
class InxGUIRenderable;
class InxMaterial;
class InxVkCoreModular;
class InxView;
class OutlineRenderer;
class RenderPipelineCallback;
class ResourcePreviewManager;
class SceneRenderGraph;
class SceneRenderTarget;
class TransientResourcePool;
class InxGUIContext;
class InxScreenUIRenderer;

class InxRenderer
{
  public:
    InxRenderer();
    ~InxRenderer();

    InxRenderer(const InxRenderer &) = delete;
    InxRenderer &operator=(const InxRenderer &) = delete;
    InxRenderer &operator=(InxRenderer &&) = delete;
    InxRenderer(InxRenderer &&) = delete;

    void SetCameraPos(float x, float y, float z);
    void SetCameraLookAt(float x, float y, float z);
    void SetCameraUp(float x, float y, float z);

    float *GetCameraPos();
    float *GetCameraLookAt();
    float *GetCameraUp();

    void TranslateCamera(float x, float y, float z);

    void SetAppMetadata(InxAppMetadata appMetaData);
    InxAppMetadata GetAppMetadata();
    InxAppMetadata GetRendererMetadata();

    void Init(int width, int height, InxAppMetadata appMetaData);
    void PreparePipeline();
    void DrawFrame();

    /// @brief Drain GPU work before destructive scene/resource replacement.
    void WaitForGpuIdle();

    void LoadShader(const char *name, const std::vector<char> &code, const char *type);
    bool HasShader(const std::string &name, const std::string &type) const;

    /// @brief Store shader render-state annotations (forwarded to InxVkCoreModular)
    void StoreShaderRenderMeta(const std::string &shaderId, const std::string &cullMode, const std::string &depthWrite,
                               const std::string &depthTest, const std::string &blend, int queue,
                               const std::string &passTag = "", const std::string &stencil = "",
                               const std::string &alphaClip = "");
    bool GetUserEvent();
    void ShowWindow();
    void HideWindow();
    void SetWindowIcon(const std::string &iconPath);
    void SetWindowFullscreen(bool fullscreen);
    void SetWindowTitle(const std::string &title);
    void SetWindowMaximized(bool maximized);
    void SetWindowResizable(bool resizable);

    // Close-request interception (delegates to InxView)
    bool IsCloseRequested() const;
    void ConfirmClose();
    void CancelClose();

    void SetGUIFont(const char *fontPath, float fontSize);
    float GetDisplayScale() const;
    void RegisterGUIRenderable(const char *name, std::shared_ptr<InxGUIRenderable> renderable);
    void UnregisterGUIRenderable(const char *name);
    void QueueDockTabSelection(const char *windowId);

    // ImGui texture management
    uint64_t UploadTextureForImGui(const std::string &name, const unsigned char *pixels, int width, int height);
    void RemoveImGuiTexture(const std::string &name);
    bool HasImGuiTexture(const std::string &name) const;
    uint64_t GetImGuiTextureId(const std::string &name) const;

    // Resource preview manager
    ResourcePreviewManager *GetResourcePreviewManager();

    void SetLogLevel(LogLevel level);

    // Scene system integration
    void InitializeDefaultScene();
    void UpdateSceneLighting();

    // Get the material from the first MeshRenderer in the scene
    std::shared_ptr<InxMaterial> GetFirstMeshRendererMaterial();

    // Scene render target for offscreen rendering
    uint64_t GetSceneTextureId() const;
    void ResizeSceneRenderTarget(uint32_t width, uint32_t height);

    // Editor gizmos
    void SetShowGrid(bool show);
    bool IsShowGrid() const;
    EditorGizmos &GetEditorGizmos();

    /// @brief Access editor tools (translate/rotate/scale gizmo)
    EditorTools *GetEditorTools();

    /// @brief Access the component gizmos draw call buffer (Python-driven)
    GizmosDrawCallBuffer *GetGizmosDrawCallBuffer();

    /// @brief Set the selected object ID for outline tracking
    void SetSelectedObjectId(uint64_t objectId)
    {
        m_selectedObjectId = objectId;
    }
    [[nodiscard]] uint64_t GetSelectedObjectId() const
    {
        return m_selectedObjectId;
    }

    // Material pipeline refresh - call this after modifying material shader paths
    bool RefreshMaterialPipeline(std::shared_ptr<InxMaterial> material);

    /// @brief Render a material preview sphere using the material's real GPU pipeline.
    /// @return true if GPU rendering succeeded and outPixels was filled.
    bool RenderMaterialPreviewGPU(std::shared_ptr<InxMaterial> material, int size,
                                  std::vector<unsigned char> &outPixels);

    // Refresh all materials using a specific shader
    bool RefreshMaterialsUsingShader(const std::string &shaderId);

    // Invalidate shader cache for hot-reload (must call before loading new shader code)
    void InvalidateShaderCache(const std::string &shaderId);

    // Invalidate cached GPU texture and force materials to re-resolve it
    void InvalidateTextureCache(const std::string &texturePath);

    // Remove pipeline render data for a specific material (releases shared_ptr)
    void RemoveMaterialPipeline(const std::string &materialName);

    // ========================================================================
    // Render Graph Access (for Python/ML integration)
    // ========================================================================

    /// @brief Get the scene render graph for pass configuration and output access
    /// @return Pointer to SceneRenderGraph, or nullptr if not initialized
    SceneRenderGraph *GetSceneRenderGraph();

    /// @brief Set a custom render pipeline (Python-driven rendering via SRC).
    /// Pass nullptr to revert to the default C++ rendering path.
    void SetRenderPipeline(std::shared_ptr<RenderPipelineCallback> pipeline);

    // ========================================================================
    // Game Camera Render Target (for Game View panel)
    // ========================================================================

    /// @brief Get Game View texture ID for ImGui display
    uint64_t GetGameTextureId() const;

    /// @brief Resize the game render target to match Game View panel size
    void ResizeGameRenderTarget(uint32_t width, uint32_t height);

    /// @brief Enable/disable game camera rendering
    void SetGameCameraEnabled(bool enabled);

    /// @brief Enable/disable scene view rendering (called by Python panel visibility)
    void SetSceneViewVisible(bool visible)
    {
        m_sceneViewVisible = visible;
    }
    [[nodiscard]] bool IsSceneViewVisible() const
    {
        return m_sceneViewVisible;
    }

    /// @brief Check if game camera rendering is enabled
    [[nodiscard]] bool IsGameCameraEnabled() const
    {
        return m_gameCameraEnabled;
    }

    /// @brief Get the screen UI renderer for GPU-based 2D screen-space UI
    /// @return Pointer to InxScreenUIRenderer, or nullptr if not initialized
    InxScreenUIRenderer *GetScreenUIRenderer();

    // ========================================================================
    // MSAA Configuration
    // ========================================================================

    /// @brief Set MSAA sample count for both scene and game render targets.
    /// Valid values: 1 (off), 2, 4, 8.  Triggers Vulkan resource recreation.
    void SetMsaaSamples(int samples);

    /// @brief Get current MSAA sample count (1 = off).
    int GetMsaaSamples() const;

    // ========================================================================
    // Present Mode
    // ========================================================================

    /// @brief Set present mode: 0=IMMEDIATE, 1=MAILBOX, 2=FIFO, 3=FIFO_RELAXED
    void SetPresentMode(int mode);

    /// @brief Set a callback invoked each frame before GUI::BuildFrame().
    /// Scene-mutating deferred tasks run here to prevent stale-reference hangs.
    void SetPreGuiCallback(std::function<void()> callback)
    {
        m_preGuiCallback = std::move(callback);
    }

    /// @brief Set a callback invoked each frame AFTER VkCore::DrawFrame() + EndFrame().
    /// Heavy scene-loading work runs here so that it occurs between frames.
    /// The Python callback calls engine.pump_events() internally when needed.
    void SetPostDrawCallback(std::function<void()> callback)
    {
        m_postDrawCallback = std::move(callback);
    }

    /// @brief Get current present mode (0=IMMEDIATE, 1=MAILBOX, 2=FIFO, 3=FIFO_RELAXED)
    int GetPresentMode() const;

    // ========================================================================
    // Editor Power-Save / Idle Mode
    // ========================================================================

    /// @brief Enable/disable editor idle mode (reduced FPS when no input).
    void SetEditorIdleEnabled(bool enabled);

    /// @brief Check if editor idle mode is enabled.
    bool IsEditorIdleEnabled() const;

    /// @brief Set the idle-mode target FPS (e.g. 10).  0 disables idling.
    void SetEditorIdleFps(float fps);

    /// @brief Get the current idle-mode target FPS.
    float GetEditorIdleFps() const;

    /// @brief Check if the editor is currently in idle (reduced-FPS) state.
    bool IsEditorIdling() const;

    /// @brief Set the editor-mode FPS cap (e.g. 60). 0 = uncapped.
    /// Only applies outside play mode.
    void SetEditorFpsCap(float fps);

    /// @brief Get the editor-mode FPS cap.
    float GetEditorFpsCap() const;

    /// @brief Tell the renderer whether the engine is in play mode.
    /// In play mode, the frame-rate cap and idle sleep are both disabled.
    void SetPlayModeRendering(bool play);

    /// @brief Check if the renderer is in play-mode (uncapped FPS).
    bool IsPlayModeRendering() const;

    /// @brief Force full-speed rendering for the next few frames (e.g. after
    /// a programmatic scene change that doesn't generate SDL events).
    void RequestFullSpeedFrame();

  private:
    InxAppMetadata m_appMetadata;
    InxAppMetadata m_rendererMetadata;

    float m_cameraPos[3] = {2.0f, 2.0f, 2.0f};
    float m_cameraLookAt[3] = {0.0f, 0.0f, 0.0f};
    float m_cameraUp[3] = {0.0f, 1.0f, 0.0f};

    // Delta time tracking
    std::chrono::high_resolution_clock::time_point m_lastFrameTime;
    float m_deltaTime = 0.016f;
    float m_totalTime = 0.0f;
    float m_smoothDeltaTime = 0.016f;
    uint64_t m_frameCount = 0;

    std::unique_ptr<InxVkCoreModular> m_vkCore;
    std::unique_ptr<InxGUI> m_gui;
    std::unique_ptr<InxView> m_view;
    std::unique_ptr<SceneRenderTarget> m_sceneRenderTarget;
    std::unique_ptr<SceneRenderGraph> m_sceneRenderGraph;
    std::unique_ptr<EditorGizmos> m_editorGizmos;
    std::unique_ptr<EditorTools> m_editorTools;
    std::unique_ptr<GizmosDrawCallBuffer> m_componentGizmos;
    std::unique_ptr<OutlineRenderer> m_outlineRenderer;
    std::unique_ptr<TransientResourcePool> m_transientResourcePool;

    // Game Camera: separate render target + graph for Game View
    std::unique_ptr<SceneRenderTarget> m_gameRenderTarget;
    std::unique_ptr<SceneRenderGraph> m_gameRenderGraph;
    std::unique_ptr<InxScreenUIRenderer> m_screenUIRenderer;
    bool m_gameCameraEnabled = false;
    bool m_sceneViewVisible = true; ///< Set from Python scene_view_panel visibility

    /// Per-frame cached game camera pointer, lazily resolved once per frame
    /// by FindGameCameraCached() and cleared at the start of each DrawFrame.
    class Camera *m_cachedGameCamera = nullptr;
    bool m_gameCameraCacheValid = false;

    // Per-camera shadow VP data for multi-camera shadow isolation.
    // Editor camera shadow data goes into m_lightCollector (default path).
    // Game camera shadow data is stored here and patched into the lighting
    // UBO inline before the game render graph executes.
    bool m_hasGameShadowData = false;
    std::array<glm::mat4, 4> m_gameShadowVPs{};
    std::array<float, 4> m_gameShadowSplits{};
    uint32_t m_gameShadowCascadeCount = 0;
    float m_gameShadowMapResolution = 0.0f;

    // SRP: Python-driven render pipeline (nullptr = default C++ path)
    std::shared_ptr<RenderPipelineCallback> m_renderPipeline;

    // Selection tracking for auto-update of outline transforms
    uint64_t m_selectedObjectId = 0;

    // Executor sub-timing (accumulated during the render-graph executor callback)
#if INFERNUX_FRAME_PROFILE
    struct ExecutorSubTiming
    {
        double sceneExecMs = 0;
        double sceneMsaaMs = 0;
        double gameSetupMs = 0;
        double gameExecMs = 0;
        double gameMsaaMs = 0;
        double gameRestoreMs = 0;
    };
    ExecutorSubTiming m_executorTiming{};
#endif

    /// @brief Find effective game camera via Scene::FindGameCamera().
    /// Returns the highest-priority active Camera (by depth), excluding the editor camera.
    class Camera *FindGameCamera();

    /// @brief Per-frame cached version of FindGameCamera().
    /// First call per frame does the actual discovery; subsequent calls return cached result.
    class Camera *FindGameCameraCached();

    /// Callback invoked once per frame BEFORE GUI::BuildFrame().
    /// Used by Python to tick DeferredTaskRunner so that scene-mutating
    /// operations (deserialize, scene load) complete before any ImGui
    /// panel renders — preventing stale-reference hangs.
    std::function<void()> m_preGuiCallback;

    /// Callback invoked once per frame AFTER VkCore::DrawFrame() + EndFrame().
    /// Used by Python to run heavy scene loads between frames, avoiding
    /// Windows "Not Responding" by running between SDL_PumpEvents() calls.
    std::function<void()> m_postDrawCallback;
};
} // namespace infernux