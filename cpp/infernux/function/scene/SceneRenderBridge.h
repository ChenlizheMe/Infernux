#pragma once

#include "SceneRenderExtractor.h"
#include "SceneRenderer.h"
#include "function/renderer/GameScreenOutput.h"
#include "function/renderer/gui/InxScreenUIRenderer.h"
#include <functional>
#include <optional>

namespace infernux
{

class Camera;
class RuntimeWorld;
class AssetDependencyGraph;
struct RenderViewPlan;
using RenderViewPlans = std::unordered_map<uint64_t, std::shared_ptr<const RenderViewPlan>>;

/// One atomic publication includes World values and every view's CPU plan.
struct RenderWorldPublication
{
    std::shared_ptr<const RenderWorldFrame> world;
    RenderViewPlans plans;
    std::unordered_map<uint64_t, bool> viewParticipation;
    InxScreenUIRenderer::Snapshot worldUI;
    std::unordered_map<uint64_t, InxScreenUIRenderer::Snapshot> preparedUI;
    std::optional<GameScreenSourceConfig> screenConfig;
    GameScreenColorContract screenColor;
    uint64_t renderPolicyRevision = 0;

    [[nodiscard]] bool Participates(uint64_t cameraId) const;
    [[nodiscard]] bool HasScreenViews() const;
};

/// Scene-side adapter that publishes extracted frames to the renderer
/// frontend. It is the only owner that connects mutable Scene state to the
/// immutable RenderWorld boundary.
class SceneRenderBridge final
{
  public:
    static SceneRenderBridge &Instance();

    SceneRenderBridge(const SceneRenderBridge &) = delete;
    SceneRenderBridge &operator=(const SceneRenderBridge &) = delete;

    [[nodiscard]] SceneRenderer &GetSceneRenderer()
    {
        return m_sceneRenderer;
    }

    void UpdateCameraData(float *outPos, float *outLookAt, float *outUp);
    void OnWindowResize(uint32_t width, uint32_t height);
    void PrepareFrame(bool useActiveCameraCulling = true);
    void SetEditorViewedWorld(uint64_t worldId) noexcept
    {
        m_editorViewedWorld = worldId;
    }
    [[nodiscard]] uint64_t GetEditorViewedWorldId() const noexcept
    {
        return m_editorViewedWorld;
    }
    [[nodiscard]] RuntimeWorld *GetEditorViewedWorld() const noexcept;
    [[nodiscard]] const std::shared_ptr<const RenderWorldFrame> &AcquirePreparedEditorFrame() const noexcept
    {
        return m_preparedEditor;
    }
    [[nodiscard]] std::shared_ptr<const RenderWorldFrame> AcquirePreparedGameFrame() const;
    [[nodiscard]] std::shared_ptr<const RenderWorldFrame> CapturePreview(Camera &camera);
    using CaptureViews = std::function<bool(RenderWorldPublication &, const RenderWorldPublication *)>;
    [[nodiscard]] bool PublishRuntimeFrame(RuntimeWorld &world, const CaptureViews &capture,
                                           uint64_t renderPolicyRevision,
                                           const std::optional<GameScreenSourceConfig> &screenConfig);
    [[nodiscard]] bool PublishPausedAuthorPresentation(RuntimeWorld &world, const CaptureViews &capture,
                                                       uint64_t renderPolicyRevision,
                                                       const std::optional<GameScreenSourceConfig> &screenConfig);
    [[nodiscard]] std::shared_ptr<const RenderWorldPublication> AcquireLatestDisplayPublication(uint64_t worldId) const;
    /// Rebuild only render work from a complete World, retaining its exact
    /// geometry, camera, pose, lighting and UI owners. Never extracts or steps.
    [[nodiscard]] bool RefreshRuntimeViewPlans(const std::shared_ptr<const RenderWorldPublication> &source,
                                               uint64_t renderPolicyRevision,
                                               const GameScreenSourceConfig &screenConfig, const CaptureViews &capture);
    void RetireRuntimeWorld(uint64_t worldId);
    /// Latest complete producer value, before this Host's prepared view index.
    [[nodiscard]] std::shared_ptr<const RenderWorldPublication> AcquireLatestRuntimePublication(uint64_t worldId) const;
    [[nodiscard]] std::shared_ptr<const RenderWorldFrame> AcquirePreparedRuntimeFrame() const noexcept
    {
        return m_preparedRuntime;
    }
    [[nodiscard]] const std::shared_ptr<const RenderWorldPublication> &
    AcquirePreparedRuntimePublication() const noexcept
    {
        return m_preparedRuntimePublication;
    }
    [[nodiscard]] std::shared_ptr<const RenderWorldPublication>
    AcquirePreparedRuntimePublication(uint64_t worldId) const noexcept;
    [[nodiscard]] const std::vector<std::shared_ptr<const RenderWorldPublication>> &
    AcquirePreparedRuntimePublications() const noexcept
    {
        return m_preparedRuntimePublications;
    }
    [[nodiscard]] std::shared_ptr<const RenderWorldFrame> AcquirePreparedFrameForCamera(uint64_t cameraId) const;
    [[nodiscard]] std::shared_ptr<const RenderWorldFrame> AcquirePreparedFrameForCamera(uint64_t worldId,
                                                                                        uint64_t cameraId) const;
    [[nodiscard]] CameraDrawCallResult CullAndBuildForView(const RenderViewData &view,
                                                           std::shared_ptr<const RenderWorldFrame> owner,
                                                           bool includeShadowDrawCalls);
    [[nodiscard]] const DrawCallResult &BuildDrawCalls();
    [[nodiscard]] Camera *GetEditorCamera() const;

#if INFERNUX_FRAME_PROFILE
    [[nodiscard]] SceneRendererProfileSnapshot GetProfileSnapshot() const;
    void ResetProfileSnapshot();
#endif

  private:
    SceneRenderBridge() = default;
    ~SceneRenderBridge() = default;

    struct RuntimePublication
    {
        SceneRenderExtractor extractor;
        std::shared_ptr<AssetDependencyGraph> assetGraph;
        RenderWorldSnapshot staging;
        std::shared_ptr<RenderWorldPublication> pending;
        std::shared_ptr<const RenderWorldPublication> completed;
        std::shared_ptr<RenderWorldPublication> pausedPending;
        std::shared_ptr<const RenderWorldPublication> display;
    };
    SceneRenderExtractor m_extractor;
    SceneRenderExtractor m_previewExtractor;
    RenderWorldSnapshot m_previewPublication;
    RenderWorldSnapshot m_editorPublication;
    std::unordered_map<uint64_t, std::shared_ptr<RuntimePublication>> m_runtimePublications;
    std::shared_ptr<const RenderWorldFrame> m_preparedEditor;
    std::shared_ptr<const RenderWorldFrame> m_preparedRuntime;
    std::shared_ptr<const RenderWorldPublication> m_preparedRuntimePublication;
    std::vector<std::shared_ptr<const RenderWorldPublication>> m_preparedRuntimePublications;
    std::unordered_map<uint64_t, std::shared_ptr<const RenderWorldPublication>> m_preparedRuntimeByWorld;
    uint64_t m_editorCameraId = 0;
    uint64_t m_editorViewedWorld = 0;
    bool m_runtimePrepared = false;
    SceneRenderer m_sceneRenderer;
};

} // namespace infernux
