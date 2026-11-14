#pragma once

#include "SceneRenderExtractor.h"
#include "SceneRenderer.h"
#include <functional>

namespace infernux
{

class Camera;
class RuntimeWorld;
struct RenderViewPlan;
using RenderViewPlans = std::unordered_map<uint64_t, std::shared_ptr<const RenderViewPlan>>;

/// One atomic publication includes World values and every view's CPU plan.
struct RenderWorldPublication
{
    std::shared_ptr<const RenderWorldFrame> world;
    RenderViewPlans plans;
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
    void SetMaterialPreparation(SceneRenderExtractor::PrepareMaterial prepare)
    {
        m_prepareMaterial = std::move(prepare);
    }
    [[nodiscard]] std::shared_ptr<const RenderWorldFrame> AcquirePreparedGameFrame() const;
    [[nodiscard]] std::shared_ptr<const RenderWorldFrame> CapturePreview(Camera &camera);
    using CaptureViews = std::function<bool(const std::shared_ptr<const RenderWorldFrame> &, RenderViewPlans &)>;
    [[nodiscard]] bool PublishRuntimeFrame(RuntimeWorld &world, const CaptureViews &capture);
    void RetireRuntimeWorld(uint64_t worldId);
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
        RenderWorldSnapshot staging;
        std::shared_ptr<RenderWorldPublication> pending;
        std::shared_ptr<const RenderWorldPublication> completed;
    };
    SceneRenderExtractor m_extractor;
    SceneRenderExtractor m_previewExtractor;
    RenderWorldSnapshot m_previewPublication;
    SceneRenderExtractor::PrepareMaterial m_prepareMaterial;
    RenderWorldSnapshot m_editorPublication;
    std::unordered_map<uint64_t, std::unique_ptr<RuntimePublication>> m_runtimePublications;
    std::shared_ptr<const RenderWorldFrame> m_preparedEditor;
    std::shared_ptr<const RenderWorldFrame> m_preparedRuntime;
    std::shared_ptr<const RenderWorldPublication> m_preparedRuntimePublication;
    std::vector<std::shared_ptr<const RenderWorldPublication>> m_preparedRuntimePublications;
    std::unordered_map<uint64_t, std::shared_ptr<const RenderWorldPublication>> m_preparedRuntimeByWorld;
    uint64_t m_editorCameraId = 0;
    bool m_runtimePrepared = false;
    SceneRenderer m_sceneRenderer;
};

} // namespace infernux
