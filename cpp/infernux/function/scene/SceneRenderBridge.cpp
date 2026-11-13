#include "SceneRenderBridge.h"

#include "SceneManager.h"
#include <core/types/ShaderTypes.h>

namespace infernux
{

SceneRenderBridge &SceneRenderBridge::Instance()
{
    // The engine singletons intentionally outlive Python and renderer teardown.
    static SceneRenderBridge *instance = new SceneRenderBridge();
    return *instance;
}

void SceneRenderBridge::UpdateCameraData(float *outPos, float *outLookAt, float *outUp)
{
    static const RenderViewData defaultView;
    const auto world = m_preparedEditor ? m_preparedEditor : m_preparedRuntime;
    const auto &view = world ? world->PrimaryView() : defaultView;
    const glm::vec3 lookAt = view.position + view.forward;

    if (outPos) {
        outPos[0] = view.position.x;
        outPos[1] = view.position.y;
        outPos[2] = view.position.z;
    }
    if (outLookAt) {
        outLookAt[0] = lookAt.x;
        outLookAt[1] = lookAt.y;
        outLookAt[2] = lookAt.z;
    }
    if (outUp) {
        outUp[0] = view.up.x;
        outUp[1] = view.up.y;
        outUp[2] = view.up.z;
    }
}

void SceneRenderBridge::OnWindowResize(uint32_t width, uint32_t height)
{
    if (height == 0)
        return;

    m_extractor.SetAspectRatio(static_cast<float>(width) / static_cast<float>(height));
    if (Camera *editorCamera = GetEditorCamera())
        editorCamera->SetScreenDimensions(width, height);
}

bool SceneRenderBridge::PublishRuntimeFrame(RuntimeWorld &world, const CaptureViews &capture)
{
    WorldExecutionScope scope(&world);
    auto &state = m_runtimePublications[world.id];
    if (!state)
        state = std::make_unique<RuntimePublication>();
    state->extractor.SetMaterialPreparation(m_prepareMaterial);
    try {
        (void)state->extractor.ExtractRuntimeFrame(state->staging, world);
        auto candidate = std::make_shared<RenderWorldPublication>();
        candidate->world = state->staging.Acquire();
        if (!capture(candidate->world, candidate->plans))
            return false;
        std::atomic_store_explicit(&state->completed,
                                   std::shared_ptr<const RenderWorldPublication>(std::move(candidate)),
                                   std::memory_order_release);
    } catch (const ShaderProgramSelectionError &) {
        // The author service owns the source diagnostic. Publish no part of
        // this replacement; a corrected author retries this same logical F.
        return false;
    }
    return true;
}

void SceneRenderBridge::RetireRuntimeWorld(uint64_t worldId)
{
    m_runtimePublications.erase(worldId);
    if (m_preparedRuntime && m_preparedRuntime->WorldId() == worldId) {
        m_preparedRuntime.reset();
        m_preparedRuntimePublication.reset();
    }
    // Editor extraction during Play can also retain this World's pose,
    // material and geometry leases. Shutdown has no later PrepareFrame to
    // recycle them, so explicitly retire these producer/cache owners as well.
    m_preparedEditor.reset();
    m_editorPublication.Reset();
    m_extractor = SceneRenderExtractor{};
    m_previewPublication.Reset();
    m_previewExtractor = SceneRenderExtractor{};
    m_sceneRenderer.m_cameraCullCaches.clear();
    m_sceneRenderer.m_buildOwner.reset();
    m_sceneRenderer.WritableRenderWorld().Reset();
    m_editorCameraId = 0;
}
void SceneRenderBridge::PrepareFrame(bool useActiveCameraCulling)
{
    SceneManager &manager = SceneManager::Instance();
    m_runtimePrepared = manager.IsPlaying();
    m_preparedRuntime.reset();
    m_preparedRuntimePublication.reset();
    if (m_runtimePrepared) {
        const auto found = m_runtimePublications.find(manager.CurrentWorld().id);
        if (found != m_runtimePublications.end()) {
            m_preparedRuntimePublication =
                std::atomic_load_explicit(&found->second->completed, std::memory_order_acquire);
            if (m_preparedRuntimePublication)
                m_preparedRuntime = m_preparedRuntimePublication->world;
        }
    } else {
        // Stop releases producer state. Acquired graphs may still retain the
        // final publication until their in-flight work retires normally.
        m_runtimePublications.clear();
    }
    m_extractor.SetMaterialPreparation(m_prepareMaterial);
    try {
        (void)m_extractor.ExtractEditorFrame(m_editorPublication, useActiveCameraCulling);
    } catch (const ShaderProgramSelectionError &) {
        // The failed working frame is private; the completed editor frame is
        // still valid while its existing source diagnostic is being resolved.
    }
    m_preparedEditor = m_editorPublication.Acquire();
    const Camera *editorCamera = GetEditorCamera();
    m_editorCameraId = editorCamera ? editorCamera->GetComponentID() : 0;
    // Retirement consults acquired publications, never a partial F's Component
    // registry. In particular, F2 destroying a camera cannot retire F1's list.
    const auto retainedBy = [](const auto &frame, uint64_t cameraId, const auto &cache) {
        return frame && frame->FindView(cameraId) && cache.worldId == frame->WorldId() &&
               cache.structuralRevision == frame->StructuralRevision();
    };
    auto &caches = m_sceneRenderer.m_cameraCullCaches;
    for (auto it = caches.begin(); it != caches.end();) {
        if (!retainedBy(m_preparedEditor, it->first, *it->second) &&
            !retainedBy(m_preparedRuntime, it->first, *it->second))
            it = caches.erase(it);
        else
            ++it;
    }
    m_sceneRenderer.m_buildOwner.reset();
    const auto selected = m_preparedEditor && m_editorCameraId ? m_preparedEditor : m_preparedRuntime;
    m_sceneRenderer.WritableRenderWorld().SelectPublication(selected);
    m_sceneRenderer.m_visibleCount.store(selected ? selected->Proxies().size() : 0, std::memory_order_relaxed);
}

std::shared_ptr<const RenderWorldFrame> SceneRenderBridge::AcquirePreparedGameFrame() const
{
    return SceneManager::Instance().IsPlaying() ? m_preparedRuntime : m_preparedEditor;
}

std::shared_ptr<const RenderWorldFrame> SceneRenderBridge::CapturePreview(Camera &camera)
{
    m_previewExtractor.SetMaterialPreparation(m_prepareMaterial);
    (void)m_previewExtractor.ExtractCameraFrame(m_previewPublication, &camera);
    return m_previewPublication.Acquire();
}

std::shared_ptr<const RenderWorldFrame> SceneRenderBridge::AcquirePreparedFrameForCamera(uint64_t cameraId) const
{
    const auto &frame = m_runtimePrepared && cameraId != m_editorCameraId ? m_preparedRuntime : m_preparedEditor;
    return frame && frame->FindView(cameraId) ? frame : nullptr;
}

CameraDrawCallResult SceneRenderBridge::CullAndBuildForView(const RenderViewData &view,
                                                            std::shared_ptr<const RenderWorldFrame> owner,
                                                            bool includeShadowDrawCalls)
{
    const RenderViewData *publishedView = owner ? owner->FindView(view.cameraId) : nullptr;
    if (!publishedView)
        return {};
    m_sceneRenderer.WritableRenderWorld().SelectPublication(std::move(owner));
    return m_sceneRenderer.BuildDrawCallsForCamera(view, includeShadowDrawCalls);
}

const DrawCallResult &SceneRenderBridge::BuildDrawCalls()
{
    return m_sceneRenderer.BuildDrawCalls();
}

Camera *SceneRenderBridge::GetEditorCamera() const
{
#if defined(INFERNUX_RUNTIME_MINIMAL_HOST)
    return nullptr;
#else
    return SceneManager::Instance().GetEditorCameraController().GetCamera();
#endif
}

#if INFERNUX_FRAME_PROFILE
SceneRendererProfileSnapshot SceneRenderBridge::GetProfileSnapshot() const
{
    SceneRendererProfileSnapshot result = m_extractor.GetProfileSnapshot();
    const SceneRendererProfileSnapshot &renderer = m_sceneRenderer.GetProfileSnapshot();
    result.buildMs += renderer.buildMs;
    result.buildCameraMs += renderer.buildCameraMs;
    result.buildCalls += renderer.buildCalls;
    result.buildCameraCalls += renderer.buildCameraCalls;
    result.drawCalls += renderer.drawCalls;
    return result;
}

void SceneRenderBridge::ResetProfileSnapshot()
{
    m_extractor.ResetProfileSnapshot();
    m_sceneRenderer.ResetProfileSnapshot();
}
#endif

} // namespace infernux
