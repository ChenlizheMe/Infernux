#include "SceneRenderBridge.h"

#include "SceneManager.h"
#include <algorithm>
#include <core/types/ShaderTypes.h>
#include <function/renderer/ScriptableRenderContext.h>
#include <stdexcept>

namespace infernux
{

bool RenderWorldPublication::Participates(uint64_t cameraId) const
{
    return plans.at(cameraId)->state == RenderViewPlanState::Submitted;
}

bool RenderWorldPublication::HasScreenViews() const
{
    return std::any_of(world->Views().begin(), world->Views().end(), [this](const auto &view) {
        return !view.editorView && !view.renderToTexture && Participates(view.cameraId);
    });
}

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

bool SceneRenderBridge::PublishRuntimeFrame(RuntimeWorld &world, const CaptureViews &capture,
                                            uint64_t renderPolicyRevision,
                                            const std::optional<GameScreenSourceConfig> &screenConfig)
{
    WorldExecutionScope scope(&world);
    auto &state = m_runtimePublications[world.id];
    if (!state)
        state = std::make_unique<RuntimePublication>();
    state->extractor.SetMaterialPreparation(m_prepareMaterial);
    try {
        if (!state->pending) {
            (void)state->extractor.ExtractRuntimeFrame(state->staging, world);
            state->pending = std::make_shared<RenderWorldPublication>();
            state->pending->world = state->staging.Acquire();
            state->pending->screenConfig = screenConfig;
            state->pending->renderPolicyRevision = renderPolicyRevision;
        }
        std::shared_ptr<const RenderWorldPublication> retained;
        if (state->pending->renderPolicyRevision != renderPolicyRevision ||
            (state->pending->screenConfig && screenConfig &&
             state->pending->screenConfig->samples != screenConfig->samples)) {
            retained = std::make_shared<const RenderWorldPublication>(*state->pending);
            state->pending->plans.clear();
            state->pending->renderPolicyRevision = renderPolicyRevision;
            if (state->pending->screenConfig && screenConfig) {
                auto &raster = *state->pending->screenConfig;
                raster.samples = screenConfig->samples;
                if (raster.config == screenConfig->config)
                    raster.generation = screenConfig->generation;
            }
        }
        if (!capture(*state->pending, retained.get()))
            return false;
        std::atomic_store_explicit(&state->completed,
                                   std::shared_ptr<const RenderWorldPublication>(std::move(state->pending)),
                                   std::memory_order_release);
    } catch (const ShaderProgramSelectionError &) {
        state->pending.reset();
        // The author service owns the source diagnostic. Publish no part of
        // this replacement; a corrected author retries this same logical F.
        return false;
    }
    return true;
}

bool SceneRenderBridge::RefreshRuntimeViewPlans(const std::shared_ptr<const RenderWorldPublication> &source,
                                                uint64_t renderPolicyRevision,
                                                const GameScreenSourceConfig &screenConfig, const CaptureViews &capture)
{
    auto candidate = std::make_shared<RenderWorldPublication>();
    candidate->world = source->world;
    candidate->screenConfig = screenConfig;
    candidate->renderPolicyRevision = renderPolicyRevision;
    try {
        if (!capture(*candidate, source.get()))
            return false;
    } catch (const ShaderProgramSelectionError &) {
        return false;
    }
    // A source may retire while authored render work is prepared. Publish only
    // against that exact complete source, never over a newer World completion.
    const auto found = m_runtimePublications.find(source->world->WorldId());
    if (found == m_runtimePublications.end())
        return false;
    auto expected = source;
    return std::atomic_compare_exchange_strong_explicit(
        &found->second->completed, &expected, std::shared_ptr<const RenderWorldPublication>(std::move(candidate)),
        std::memory_order_acq_rel, std::memory_order_acquire);
}

std::shared_ptr<const RenderWorldPublication> SceneRenderBridge::AcquireLatestRuntimePublication(uint64_t worldId) const
{
    const auto found = m_runtimePublications.find(worldId);
    return found == m_runtimePublications.end()
               ? nullptr
               : std::atomic_load_explicit(&found->second->completed, std::memory_order_acquire);
}

void SceneRenderBridge::RetireRuntimeWorld(uint64_t worldId)
{
    m_runtimePublications.erase(worldId);
    if (m_preparedRuntime && m_preparedRuntime->WorldId() == worldId) {
        m_preparedRuntime.reset();
        m_preparedRuntimePublication.reset();
    }
    m_preparedRuntimeByWorld.erase(worldId);
    m_preparedRuntimePublications.erase(
        std::remove_if(m_preparedRuntimePublications.begin(), m_preparedRuntimePublications.end(),
                       [worldId](const auto &entry) { return entry->world->WorldId() == worldId; }),
        m_preparedRuntimePublications.end());
    if ((m_preparedEditor && m_preparedEditor->WorldId() == worldId) || m_editorPublication.WorldId() == worldId) {
        m_preparedEditor.reset();
        m_editorPublication.Reset();
        m_extractor = SceneRenderExtractor{};
        m_editorCameraId = 0;
    }
    // Authoring extraction is a different owner. Retiring a runtime World
    // cannot discard its Scene View or another World's camera cull cache.
    if (m_previewPublication.WorldId() == worldId) {
        m_previewPublication.Reset();
        m_previewExtractor = SceneRenderExtractor{};
    }
    auto &caches = m_sceneRenderer.m_cameraCullCaches;
    for (auto it = caches.begin(); it != caches.end();) {
        if (it->second->worldId == worldId)
            it = caches.erase(it);
        else
            ++it;
    }
    if (m_sceneRenderer.m_buildOwner && m_sceneRenderer.m_buildOwner->WorldId() == worldId)
        m_sceneRenderer.m_buildOwner.reset();
    if (m_sceneRenderer.WritableRenderWorld().WorldId() == worldId)
        m_sceneRenderer.WritableRenderWorld().Reset();
}
void SceneRenderBridge::PrepareFrame(bool useActiveCameraCulling)
{
    SceneManager &manager = SceneManager::Instance();
    m_runtimePrepared = manager.IsRuntimeWorldManagementActive();
    m_preparedRuntime.reset();
    m_preparedRuntimePublication.reset();
    m_preparedRuntimePublications.clear();
    m_preparedRuntimeByWorld.clear();
    if (m_runtimePrepared) {
        const auto *primary = manager.GetPrimaryRuntimeWorld();
        for (const auto worldId : manager.GetRuntimeWorldIds()) {
            const auto found = m_runtimePublications.find(worldId);
            if (found == m_runtimePublications.end())
                continue;
            auto completed = std::atomic_load_explicit(&found->second->completed, std::memory_order_acquire);
            if (!completed)
                continue;
            if (!completed->world || completed->world->WorldId() != worldId)
                throw std::logic_error("Runtime publication has the wrong World owner: " + std::to_string(worldId));
            m_preparedRuntimePublications.push_back(completed);
            m_preparedRuntimeByWorld.emplace(worldId, completed);
            if (primary && primary->id == worldId) {
                m_preparedRuntimePublication = completed;
                m_preparedRuntime = completed->world;
            }
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
        const auto owner = m_preparedRuntimeByWorld.find(it->second->worldId);
        const bool runtimeRetains =
            owner != m_preparedRuntimeByWorld.end() && retainedBy(owner->second->world, it->first, *it->second);
        if (!retainedBy(m_preparedEditor, it->first, *it->second) && !runtimeRetains)
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
    return SceneManager::Instance().IsRuntimeWorldManagementActive() ? m_preparedRuntime : m_preparedEditor;
}

std::shared_ptr<const RenderWorldFrame> SceneRenderBridge::CapturePreview(Camera &camera)
{
    m_previewExtractor.SetMaterialPreparation(m_prepareMaterial);
    (void)m_previewExtractor.ExtractCameraFrame(m_previewPublication, &camera);
    return m_previewPublication.Acquire();
}

std::shared_ptr<const RenderWorldPublication>
SceneRenderBridge::AcquirePreparedRuntimePublication(uint64_t worldId) const noexcept
{
    const auto found = m_preparedRuntimeByWorld.find(worldId);
    return found == m_preparedRuntimeByWorld.end() ? nullptr : found->second;
}

std::shared_ptr<const RenderWorldFrame> SceneRenderBridge::AcquirePreparedFrameForCamera(uint64_t worldId,
                                                                                         uint64_t cameraId) const
{
    if (m_preparedEditor && m_preparedEditor->WorldId() == worldId && m_preparedEditor->FindView(cameraId))
        return m_preparedEditor;
    const auto publication = AcquirePreparedRuntimePublication(worldId);
    const auto frame = publication ? publication->world : nullptr;
    const auto *view = frame ? frame->FindView(cameraId) : nullptr;
    if (view && view->worldId != worldId)
        throw std::logic_error("Published camera has the wrong World owner: " + std::to_string(worldId) + "/" +
                               std::to_string(cameraId));
    return view ? frame : nullptr;
}

std::shared_ptr<const RenderWorldFrame> SceneRenderBridge::AcquirePreparedFrameForCamera(uint64_t cameraId) const
{
    if (m_preparedEditor && m_preparedEditor->FindView(cameraId))
        return m_preparedEditor;
    // Component IDs are process-unique today; callers carrying a World must
    // use the explicit overload so accidental owner mismatches fail closed.
    for (const auto &entry : m_preparedRuntimePublications)
        if (entry->world->FindView(cameraId))
            return entry->world;
    return nullptr;
}

CameraDrawCallResult SceneRenderBridge::CullAndBuildForView(const RenderViewData &view,
                                                            std::shared_ptr<const RenderWorldFrame> owner,
                                                            bool includeShadowDrawCalls)
{
    const RenderViewData *publishedView = owner ? owner->FindView(view.cameraId) : nullptr;
    if (!publishedView)
        return {};
    if (owner->WorldId() != view.worldId || publishedView->worldId != view.worldId)
        throw std::logic_error("Render view World owner does not match its published source");
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
