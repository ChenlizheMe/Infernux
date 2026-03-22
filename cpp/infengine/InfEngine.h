#pragma once
#include <core/types/InfFwdType.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include <function/renderer/InfRenderer.h>
#include <function/scene/EditorCameraController.h>
#include <function/scene/SceneManager.h>

#include <core/error/InfError.h>
#include <core/log/InfLog.h>
#include <platform/filesystem/InfExtLoad.h>
#include <platform/filesystem/InfPath.h>
#include <platform/input/InputManager.h>

#include <function/resources/AssetDatabase/AssetDatabase.h>

namespace infengine
{

class InfEngine
{
  public:
    InfEngine(std::string dllPath);
    ~InfEngine();

    // Prevent copying
    InfEngine(const InfEngine &) = delete;
    InfEngine &operator=(const InfEngine &) = delete;
    InfEngine(InfEngine &&) = delete;
    InfEngine &operator=(InfEngine &&) = delete;

    void Run();
    void Exit();
    void Cleanup();

    // renderer
    void InitRenderer(int width, int height, const std::string &projectPath,
                      const std::string &builtinResourcePath = "");
    void ResetImGuiLayout();

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

    /// @brief Lightweight gizmo-only axis proximity test (no scene raycast).
    /// Used every frame for hover highlighting — only tests the 3 axis handles.
    /// @return Gizmo axis ID, or 0 if not hovering any axis.
    uint64_t PickGizmoAxis(float screenX, float screenY, float viewportWidth, float viewportHeight);

    // ========================================================================
    // Editor Tools API — highlight + ray + mode for Python-side gizmo interaction
    // ========================================================================

    /// @brief Set the highlighted (hovered) gizmo axis. 0=None, 1=X, 2=Y, 3=Z.
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
    bool RefreshMaterialPipeline(std::shared_ptr<InfMaterial> material);

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
    /// @return Pointer to InfRenderer, or nullptr if not initialized / cleaned up
    [[nodiscard]] InfRenderer *GetRenderer() const
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

    InfAppMetadata m_metadata{"InfEngine", 0, 1, 0, "com.infrenderer.InfEngine"};

    std::unique_ptr<InfExtLoad> m_extLoader;
    std::unique_ptr<AssetDatabase> m_assetDatabase;
    std::unique_ptr<InfRenderer> m_renderer;

    LogLevel m_logLevel = LogLevel::LOG_INFO;
    bool m_isCleanedUp = false;
    bool m_isCleaningUp = false;

    // Selection tracking for outline updates
    uint64_t m_selectedObjectId = 0;
    std::vector<uint64_t> m_cachedOutlineIds; ///< Last set of IDs passed to SetSelectionOutlines

    // ImGui ini file path — stored as std::filesystem::path so that
    // wide-char paths (e.g. Chinese usernames) work correctly on Windows.
    std::filesystem::path m_imguiIniPath;

    void LoadImGuiLayout();
    void SaveImGuiLayout();
};
} // namespace infengine