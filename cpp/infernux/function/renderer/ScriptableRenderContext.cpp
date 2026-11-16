#include "ScriptableRenderContext.h"
#include <chrono>
#include <core/log/InxLog.h>
#include <function/renderer/rhi/RhiRenderTexture.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxMesh/InxMesh.h>
#include <function/scene/Scene.h>
#include <function/scene/SceneManager.h>
#include <stdexcept>

namespace infernux
{
#if INFERNUX_FRAME_PROFILE
ScriptableRenderContext::ProfileSnapshot g_srcProfileSnapshot;

ScriptableRenderContext::ProfileSnapshot ScriptableRenderContext::GetProfileSnapshot()
{
    return g_srcProfileSnapshot;
}

void ScriptableRenderContext::ResetProfileSnapshot()
{
    g_srcProfileSnapshot = {};
}
#endif

void ScriptableRenderContext::InitializeView(std::shared_ptr<const RenderWorldFrame> world, const RenderViewData &view,
                                             bool capture, bool completedWorld)
{
    const auto *published = world ? world->FindView(view.cameraId) : nullptr;
    if (!published)
        throw std::invalid_argument("Render context requires a view in its retained World publication");
    m_renderWorldOwner = std::move(world);
    m_cachedCameraView = std::make_shared<RenderViewData>(view);

    if (capture) {
        auto &manager = SceneManager::Instance();
        auto *source = manager.AuthoringWorld().id == view.worldId ? &manager.AuthoringWorld()
                                                                   : manager.FindRuntimeWorld(view.worldId);
        if (!source)
            throw std::logic_error("UI capture requires the actual retained view's source World");
        m_captureWorldScope = std::make_unique<WorldExecutionScope>(source);
        m_scene = source->scenes.active;
        m_gizmoCtx.activeScene = source->scenes.active;
        m_recordedPlan = std::make_shared<RenderViewPlan>();
    }
    m_captureCompletedWorld = completedWorld;
    if (completedWorld && (!capture || view.editorView))
        throw std::logic_error("Completed World capture requires a runtime view plan");
}

ScriptableRenderContext::ScriptableRenderContext(std::shared_ptr<const RenderWorldFrame> world,
                                                 const RenderViewData &view, RenderViewCaptureServices services,
                                                 bool completedWorld)
    : m_captureServices(std::move(services))
{
    InitializeView(std::move(world), view, true, completedWorld);
    SetupCameraProperties();
}

void ScriptableRenderContext::SetupCameraProperties()
{
    m_cachedView = m_cachedCameraView->view;
    m_cachedProj = m_cachedCameraView->projection;
    if (!m_recordedPlan)
        m_execution->setup(*this);
}

bool ScriptableRenderContext::ApplyGraph(const RenderGraphDescription &desc)
{
    if (!m_recordedPlan)
        return m_execution->apply(*this, desc);
    m_recordedPlan->graph =
        AcquireRenderGraphPublication(desc, m_captureServices.prepareMaterial, m_captureServices.acquireProgram);
    return true;
}

void ScriptableRenderContext::UpdateParameterBlocks(const std::vector<GraphParameterBlockUpdate> &updates)
{
    if (m_recordedPlan)
        m_recordedPlan->parameters = updates;
    else
        m_execution->update(*this, updates);
}

void ScriptableRenderContext::SubmitCulling(CullingResults &culling)
{
    if (!m_recordedPlan) {
        m_execution->submit(*this, culling);
        return;
    }
    m_recordedPlan->culling = culling;
    m_recordedPlan->draws = std::move(m_orderedDrawCalls);
    m_recordedPlan->commands = std::move(m_pendingCommands);
    m_recordedPlan->state = RenderViewPlanState::Submitted;
    m_submitted = true;
}

bool ScriptableRenderContext::IsGraphRevisionCurrent(uint64_t revision) const
{
    return !m_recordedPlan && m_execution->isCurrent(*this, revision);
}

int ScriptableRenderContext::GetOutputSamples() const noexcept
{
    return m_cachedCameraView->outputGeneration
               ? static_cast<int>(m_cachedCameraView->outputGeneration->description.samples)
               : 0;
}

ScriptableRenderContext::ScriptableRenderContext(std::shared_ptr<const RenderWorldFrame> world)
    : m_renderWorldOwner(std::move(world)), m_captureCompletedWorld(true)
{
    if (!m_renderWorldOwner)
        throw std::logic_error("World UI capture requires its immutable World owner");
    auto &manager = SceneManager::Instance();
    auto *source = manager.FindRuntimeWorld(m_renderWorldOwner->WorldId());
    if (!source)
        throw std::logic_error("World UI capture requires its live source World");
    m_captureWorldScope = std::make_unique<WorldExecutionScope>(source);
    m_scene = source->scenes.active;
    m_recordedPlan = std::make_shared<RenderViewPlan>();
}

void ScriptableRenderContext::SetWorldUISnapshot(CompletedUIFrame snapshot)
{
    if (!m_recordedPlan || !snapshot || !m_renderWorldOwner->OwnsUI(snapshot->World()))
        throw std::logic_error("World UI must belong to the selected shared World source");
    m_recordedPlan->worldUI = std::move(snapshot);
}

CullingResults &ScriptableRenderContext::Cull()
{
    const auto &view = *m_cachedCameraView;
#if INFERNUX_FRAME_PROFILE
    using Clock = std::chrono::high_resolution_clock;
    const auto cullStart = Clock::now();
#endif
    if (m_hasCullData)
        return m_cachedCullingResults;

    const bool editorView = view.editorView;

    // Lighting is staged after RenderPipeline::Render, so the graph's shadow
    // views here belong to the previous frame. Collect the scene-owned caster
    // list independently; the current light views filter it during recording.
    const bool needsShadowDrawCalls = m_captureServices.includeShadowCasters;
    CameraDrawCallResult ownedResult;
    CullingResults results;

    // RenderWorld extraction is camera-neutral. Scene, Game, preview, and
    // future stacked cameras all derive an independent visible list here.
    ownedResult = m_captureServices.culler->CullAndBuildForView(view, m_renderWorldOwner, needsShadowDrawCalls);
    results.visibleListRevision = ownedResult.visibleListRevision;
    results.shadowListRevision = ownedResult.shadowListRevision;
    results.renderWorldOwner = ownedResult.worldOwner;

    m_hasCullData = true;

    if (ownedResult.visibleDrawCallsRef) {
        results.visibleRenderers =
            RendererList::Borrow(*ownedResult.visibleDrawCallsRef, RendererListPurpose::CameraVisible,
                                 RenderDomainBit(RenderDomain::SceneGeometry));
    } else {
        results.visibleRenderers =
            RendererList::Own(std::move(ownedResult.visibleDrawCalls), RendererListPurpose::CameraVisible,
                              RenderDomainBit(RenderDomain::SceneGeometry));
    }
    if (needsShadowDrawCalls) {
        if (ownedResult.shadowDrawCallsRef) {
            results.shadowCasters =
                RendererList::Borrow(*ownedResult.shadowDrawCallsRef, RendererListPurpose::ShadowCasters,
                                     RenderDomainBit(RenderDomain::SceneGeometry));
        } else {
            results.shadowCasters =
                RendererList::Own(std::move(ownedResult.shadowDrawCalls), RendererListPurpose::ShadowCasters,
                                  RenderDomainBit(RenderDomain::SceneGeometry));
        }
    }
    // Populate visible light count from the scene light collector.
    // CollectLights() runs earlier in the frame (InxRenderer::UpdateSceneLighting),
    // so the count is already available.
    results.lightCount =
        m_renderWorldOwner->Lighting() ? static_cast<uint32_t>(m_renderWorldOwner->Lighting()->lights.size()) : 0;
    m_cachedCullingResults = std::move(results);
#if INFERNUX_FRAME_PROFILE
    const double elapsedMs = std::chrono::duration<double, std::milli>(Clock::now() - cullStart).count();
    g_srcProfileSnapshot.cullMs += elapsedMs;
    if (!editorView) {
        g_srcProfileSnapshot.cullGameMs += elapsedMs;
        g_srcProfileSnapshot.cullGameCalls += 1.0;
    } else {
        g_srcProfileSnapshot.cullEditorMs += elapsedMs;
        g_srcProfileSnapshot.cullEditorCalls += 1.0;
    }
    g_srcProfileSnapshot.cullCalls += 1.0;
    g_srcProfileSnapshot.baseDrawCalls += static_cast<double>(m_cachedCullingResults.visibleObjectCount());
#endif
    return m_cachedCullingResults;
}

bool ScriptableRenderContext::RenderWithGraph(const RenderGraphDescription &desc,
                                              const std::vector<GraphParameterBlockUpdate> &updates)
{
    SetupCameraProperties();
    if (!IsGraphRevisionCurrent(desc.sourceRevision) && !ApplyGraph(desc))
        return false;
    UpdateParameterBlocks(updates);
    CullingResults &culling = Cull();
    SubmitCulling(culling);
    return true;
}

bool ScriptableRenderContext::UsesPublishedWorldUI() const
{
    const bool published = m_recordedPlan && static_cast<bool>(m_recordedPlan->worldUI);
    if (!published && (m_captureCompletedWorld || m_renderWorldOwner->IsSharedSceneView()))
        throw std::logic_error("Runtime UI capture requires its shared World UI publication");
    return published;
}

void ScriptableRenderContext::SetScreenUISnapshot(CompletedUIFrame snapshot)
{
    if (!m_recordedPlan || !snapshot || !m_renderWorldOwner->OwnsUI(snapshot->World()))
        throw std::logic_error("Completed UI snapshot must belong to this exact captured World publication");
    m_recordedPlan->ui = std::move(snapshot);
}

void ScriptableRenderContext::ExecuteCommandBuffer(CommandBuffer &cmd)
{
    const auto &commands = cmd.GetCommands();
    m_pendingCommands.insert(m_pendingCommands.end(), commands.begin(), commands.end());
}

RenderDomainMask ScriptableRenderContext::ProcessPendingCommandBuffers()
{
    RenderDomainMask appendedDomains = 0;
    for (const auto &command : m_pendingCommands) {
        switch (command.type) {
        case RenderCommandType::DrawMesh: {
            const auto &params = std::get<DrawMeshParams>(command.data);
            if (!params.geometry || !params.material)
                break;
            DrawCall draw;
            if (params.geometry->subMeshes.empty()) {
                draw.indexStart = 0;
                draw.indexCount = static_cast<uint32_t>(params.geometry->indices.size());
                draw.vertexStart = 0;
            } else {
                const auto &submesh = params.geometry->subMeshes.at(static_cast<size_t>(params.submeshIndex));
                draw.indexStart = submesh.indexStart;
                draw.indexCount = submesh.indexCount;
                draw.vertexStart = static_cast<int32_t>(submesh.vertexStart);
            }
            draw.worldMatrix = params.worldMatrix;
            draw.material = params.material;
            draw.parameterBlock = params.parameterBlock;
            draw.objectId = params.objectId;
            draw.layerMask = 1u;
            draw.frustumVisible = true;
            draw.castsShadows = false;
            draw.worldBounds = params.worldBounds;
            draw.meshVertices = params.vertices;
            draw.meshIndices = params.indices;
            draw.meshIndexFormat = params.meshIndexFormat;
            draw.meshDataOwner = params.geometry;
            draw.meshAssetGuid = params.meshGuid;
            draw.meshRuntimeVersion = params.meshGeneration;
            draw.meshUsesDetachedStorage = params.meshGuid.empty();
            m_orderedDrawCalls.push_back(std::move(draw));
            appendedDomains |= RenderDomainBit(RenderDomain::SceneGeometry);
            break;
        }

        // Public recording rejects these commands. Treat their presence in
        // a stream as a contract violation, never as successful execution.
        case RenderCommandType::ClearRenderTarget:
        case RenderCommandType::SetRenderTarget:
        case RenderCommandType::GetTemporaryRT:
        case RenderCommandType::ReleaseTemporaryRT:
            throw std::logic_error("Unsupported CommandBuffer target command reached execution; "
                                   "use RenderGraph pass write_color/write_depth and set_clear");
        }
    }
    m_pendingCommands.clear();
    return appendedDomains;
}

bool ScriptableRenderContext::IsCommandImplemented(RenderCommandType type) noexcept
{
    switch (type) {
    case RenderCommandType::DrawMesh:
        return true;
    case RenderCommandType::GetTemporaryRT:
    case RenderCommandType::ReleaseTemporaryRT:
    case RenderCommandType::ClearRenderTarget:
    case RenderCommandType::SetRenderTarget:
        return false;
    }
    return false;
}

RenderTargetHandle ScriptableRenderContext::GetCameraTarget() const
{
    // Returns the sentinel CAMERA_TARGET_HANDLE.
    // At execution time, this resolves to the scene render target's resolved color image.
    return CAMERA_TARGET_HANDLE;
}
} // namespace infernux
