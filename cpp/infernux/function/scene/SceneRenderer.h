#pragma once

#include "../scene/SceneSystem.h"
#include <function/renderer/Frustum.h>
#include <function/renderer/InxRenderStruct.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <functional>
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

namespace infernux
{

/**
 * @brief RenderableObject - data needed to render one object.
 *
 * This is a lightweight struct extracted from GameObject+MeshRenderer
 * for efficient rendering without traversing the scene graph during draw calls.
 */
struct RenderableObject
{
    uint64_t objectId;
    glm::mat4 worldMatrix;
    MeshRef mesh;
    std::shared_ptr<InxMaterial> renderMaterial; // Actual material for rendering
    MeshRenderer *meshRenderer = nullptr;        // Direct pointer to avoid re-lookup
    AABB worldBounds;                            // World-space bounding box for culling
    bool visible;
};

/**
 * @brief SceneRenderer - bridges Scene system with Vulkan rendering.
 *
 * Responsibilities:
 * - Collect renderable objects from the active scene
 * - Provide camera matrices to the renderer
 * - Sort objects for rendering (by material, depth, etc.)
 * - Culling (frustum, occlusion)
 */
class SceneRenderer
{
  public:
    SceneRenderer() = default;
    ~SceneRenderer() = default;

    // ========================================================================
    // Frame preparation
    // ========================================================================

    /// @brief Prepare render data for the current frame using editor camera.
    /// Call this before DrawFrame()
    void PrepareFrame();

    /// @brief Prepare render data for a specific camera (independent culling).
    /// Used by Game View to cull against the game camera's frustum.
    void PrepareFrame(Camera *camera);

    /// @brief Get the view matrix for rendering
    [[nodiscard]] glm::mat4 GetViewMatrix() const;

    /// @brief Get the projection matrix for rendering
    [[nodiscard]] glm::mat4 GetProjectionMatrix() const;

    /// @brief Get camera position for shaders
    [[nodiscard]] glm::vec3 GetCameraPosition() const;

    /// @brief Get camera forward direction
    [[nodiscard]] glm::vec3 GetCameraForward() const;

    /// @brief Get camera up vector
    [[nodiscard]] glm::vec3 GetCameraUp() const;

    // ========================================================================
    // Renderable access
    // ========================================================================

    /// @brief Get all renderable objects for this frame
    [[nodiscard]] const std::vector<RenderableObject> &GetRenderables() const
    {
        return m_renderables;
    }

    /// @brief Get number of visible objects after culling
    [[nodiscard]] size_t GetVisibleCount() const
    {
        return m_visibleCount;
    }

    /// @brief Build draw calls from visible renderables.
    /// Converts the culled/sorted RenderableObject list into DrawCall + combined vertex/index data.
    /// @note Vertices remain in model/local space; per-object world transform is applied on GPU via push constants.
    [[nodiscard]] DrawCallResult BuildDrawCalls();

    /// @brief Build draw calls by re-culling existing renderables against a different camera.
    /// Reuses renderables collected by PrepareFrame() to avoid re-collecting world matrices,
    /// bounds, and materials. Only re-applies frustum culling with the given camera.
    [[nodiscard]] DrawCallResult BuildDrawCallsForCamera(Camera *camera);

    // ========================================================================
    // Settings
    // ========================================================================

    /// @brief Enable/disable frustum culling
    void SetFrustumCullingEnabled(bool enabled)
    {
        m_frustumCulling = enabled;
    }
    [[nodiscard]] bool IsFrustumCullingEnabled() const
    {
        return m_frustumCulling;
    }

    /// @brief Set aspect ratio (from window)
    void SetAspectRatio(float aspect);

  private:
    void CollectRenderables(uint32_t cullingMask = 0xFFFFFFFF);
    void PerformCulling();
    void SortRenderables();

    /// @brief Fast-path: update only world matrices and bounds in cached renderables.
    void UpdateCachedRenderableTransforms();

    /// @brief Shared draw-call emission logic used by both BuildDrawCalls() and BuildDrawCallsForCamera().
    void EmitDrawCallsForRenderable(DrawCallResult &result, const RenderableObject &renderable, bool visible,
                                    bool bufferDirty) const;

    std::vector<RenderableObject> m_renderables;
    size_t m_visibleCount = 0;

    bool m_frustumCulling = true;

    // Cached camera state for the frame
    Camera *m_activeCamera = nullptr;

    // ── Renderable cache ─────────────────────────────────────────────
    // When the renderer set hasn't changed (same MeshRenderers, same
    // enable/disable state), we skip full CollectRenderables/Sort and
    // only update world matrices + bounds in-place.
    uint64_t m_cachedMeshRendererVersion = 0;

    // Draw call cache: reused when renderables are cached.
    DrawCallResult m_cachedDrawCalls;
    bool           m_drawCallsCacheValid = false;
};

/**
 * @brief Helper to integrate SceneRenderer with existing InxVkCore.
 *
 * This provides the DrawScene callback that reads from SceneManager.
 */
class SceneRenderBridge
{
  public:
    static SceneRenderBridge &Instance();

    SceneRenderBridge(const SceneRenderBridge &) = delete;
    SceneRenderBridge &operator=(const SceneRenderBridge &) = delete;

    /// @brief Get the scene renderer
    [[nodiscard]] SceneRenderer &GetSceneRenderer()
    {
        return m_sceneRenderer;
    }

    /// @brief Update camera data (call before DrawFrame)
    /// Returns camera data in the format expected by InxVkCore::DrawFrame
    void UpdateCameraData(float *outPos, float *outLookAt, float *outUp);

    /// @brief Set aspect ratio from window dimensions
    void OnWindowResize(uint32_t width, uint32_t height);

    /// @brief Prepare editor camera rendering (call once per frame before draw calls)
    void PrepareFrame();

    /// @brief Prepare rendering for a specific camera (independent culling).
    /// @param camera The camera to cull and collect renderables for.
    /// @return DrawCallResult with draw calls visible to this camera.
    [[nodiscard]] DrawCallResult PrepareAndBuildForCamera(Camera *camera);

    /// @brief Build draw calls for a camera reusing the editor camera's renderables.
    /// Avoids re-collecting world matrices, bounds, and materials (from PrepareFrame).
    /// Only re-applies frustum culling and layer filtering for the given camera.
    [[nodiscard]] DrawCallResult CullAndBuildForCamera(Camera *camera);

    /// @brief Build draw calls from the current frame's visible renderables.
    /// Delegates to SceneRenderer::BuildDrawCalls().
    [[nodiscard]] DrawCallResult BuildDrawCalls();

    /// @brief Get the editor camera.
    [[nodiscard]] Camera *GetEditorCamera() const;

  private:
    SceneRenderBridge() = default;
    ~SceneRenderBridge() = default;

    SceneRenderer m_sceneRenderer;
};

} // namespace infernux
