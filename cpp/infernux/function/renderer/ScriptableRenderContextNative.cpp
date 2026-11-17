#include "CommandBuffer.h"
#include "EditorGizmos.h"
#include "EditorTools.h"
#include "FullscreenRenderer.h"
#include "GizmosDrawCallBuffer.h"
#include "InxVkCoreModular.h"
#include "SceneRenderGraph.h"
#include "ScriptableRenderContext.h"
#include "vk/RhiVulkanTypes.h"
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxMesh/InxMesh.h>
#include <function/scene/Scene.h>
#include <function/scene/SceneManager.h>
#include <function/scene/SceneRenderBridge.h>

#include <core/log/InxLog.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <stdexcept>

namespace infernux
{

#if INFERNUX_FRAME_PROFILE
extern ScriptableRenderContext::ProfileSnapshot g_srcProfileSnapshot;
#endif

const ScriptableRenderContext::Execution ScriptableRenderContext::s_nativeExecution = {
    [](auto &context) { context.SetupNativeCameraProperties(); },
    [](auto &context, const auto &graph) { return context.ApplyNativeGraph(graph); },
    [](auto &context, const auto &updates) { context.UpdateNativeParameterBlocks(updates); },
    [](auto &context, auto &culling) { context.SubmitNativeCulling(culling); },
    [](const auto &context, uint64_t revision) { return context.IsNativeGraphCurrent(revision); },
};

ScriptableRenderContext::ScriptableRenderContext(InxVkCoreModular *vkCore, SceneRenderGraph *graph, uint64_t draw,
                                                 const EditorGizmosContext &gizmoCtx,
                                                 std::shared_ptr<const RenderWorldFrame> world,
                                                 const RenderViewData &view, bool capture, bool completedWorld)
    : m_vkCore(vkCore), m_graph(graph), m_draw(draw), m_gizmoCtx(gizmoCtx)
{
    m_execution = &s_nativeExecution;
    m_captureServices.culler = &SceneRenderBridge::Instance().GetSceneRenderer();
    m_captureServices.includeShadowCasters =
        !graph || graph->GetRenderViewContext().kind != rhi::RenderViewKind::Preview;
    m_captureServices.prepareMaterial = [vkCore](const auto &material) { vkCore->PrepareMaterialAuthoring(material); };
    m_captureServices.acquireProgram = [vkCore](const auto &name) {
        return vkCore->GetFullscreenPipelineCache()->AcquireProgram(name);
    };
    InitializeView(std::move(world), view, capture, completedWorld);
    SetupCameraProperties();
}

void ScriptableRenderContext::SetupNativeCameraProperties()
{
    const auto &view = *m_cachedCameraView;
    if (m_graph && !m_recordedPlan) {
        m_graph->SetSurfaceRecords(m_renderWorldOwner ? m_renderWorldOwner->SurfaceRecords() : nullptr);
        m_graph->SetCameraInvertCulling(view.invertCulling);
        m_graph->UpdateMainPassClearSettings(view.clearFlags, view.backgroundColor, view.dithering, view.stopNaNs);
    }
}

bool ScriptableRenderContext::ApplyNativeGraph(const RenderGraphDescription &desc)
{

#if INFERNUX_FRAME_PROFILE
    using Clock = std::chrono::high_resolution_clock;
    const auto t0 = Clock::now();
#endif
    if (!m_graph)
        throw std::runtime_error("Render graph publication requires an available render context graph");
    const bool applied = m_graph->ApplyPythonGraph(desc);
#if INFERNUX_FRAME_PROFILE
    g_srcProfileSnapshot.applyGraphMs += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
#endif
    return applied;
}

void ScriptableRenderContext::UpdateNativeParameterBlocks(const std::vector<GraphParameterBlockUpdate> &updates)
{

    if (m_graph && !updates.empty())
        m_graph->UpdateParameterBlocks(updates);
}

void ScriptableRenderContext::SubmitNativeCulling(CullingResults &culling)
{

#if INFERNUX_FRAME_PROFILE
    using Clock = std::chrono::high_resolution_clock;
    const auto submitStart = Clock::now();
    const size_t baseDrawCount = culling.visibleRenderers.Size();
    const bool baseRendererListBorrowed = culling.visibleRenderers.IsBorrowed();
#endif
    if (m_submitted) {
        INXLOG_WARN("ScriptableRenderContext::SubmitNativeCulling() called after already submitted");
        return;
    }

    // Freeze pass-local values at the same authority boundary as the view's
    // renderer list. Later parameter edits belong to the next submission.
    if (m_graph)
        m_graph->CaptureParameterBlocksForSubmission();

    // A game camera with an unchanged borrowed visible set can reuse the
    // RenderGraph's complete submitted list (including its skybox) without
    // copying and then destroying tens of thousands of DrawCalls every frame.
    // Editor-only appenders deliberately stay on the normal path.
    const bool hasEditorAppenders = m_gizmoCtx.gizmos || m_gizmoCtx.editorTools || !m_pendingCommands.empty() ||
                                    (m_gizmoCtx.componentGizmos && (m_gizmoCtx.componentGizmos->HasData() ||
                                                                    m_gizmoCtx.componentGizmos->HasIconData()));
    uint64_t submissionSignature = 1469598103934665603ULL;
    auto mixSubmission = [&](uint64_t value) {
        submissionSignature ^= value;
        submissionSignature *= 1099511628211ULL;
    };
    // Camera culling publishes a monotonic content revision. RenderWorld
    // publications rotate between recyclable frame objects, so pointer values
    // cannot represent draw-list identity.
    mixSubmission(culling.visibleListRevision);
    // RenderWorld publications rotate between recyclable frame objects. The
    // shadow vector address therefore changes even when its contents do not.
    // Use the camera cache revision as the durable identity instead of that
    // transient address, otherwise a static scene can never reuse submission.
    mixSubmission(culling.shadowListRevision);
    const bool drawSkyboxForSignature =
        !m_cachedCameraView->valid || m_cachedCameraView->clearFlags == CameraClearFlags::Skybox;
    mixSubmission(drawSkyboxForSignature ? 1ULL : 0ULL);
    std::shared_ptr<InxMaterial> signatureSkyboxMaterial;
    if (drawSkyboxForSignature) {
        if (m_renderWorldOwner && m_renderWorldOwner->Lighting())
            signatureSkyboxMaterial = m_renderWorldOwner->Lighting()->skyboxMaterial;
        mixSubmission(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(signatureSkyboxMaterial.get())));
    }
    const bool shadowListReusable = culling.shadowCasters.Empty() || culling.shadowCasters.IsBorrowed();
    const bool visibleListReusable = culling.visibleRenderers.IsBorrowed();
    const bool graphAvailable = m_graph != nullptr;
    const uint64_t objectBufferRevision = m_vkCore->GetObjectBufferRevision();
    const bool signatureReusable =
        graphAvailable && m_graph->CanReuseCachedSubmission(submissionSignature, objectBufferRevision);
    if (!hasEditorAppenders && visibleListReusable && shadowListReusable && signatureReusable) {
        // The cached shadow list borrows storage from the same retained world snapshot.
        if (m_cachedCameraView->valid)
            m_graph->SetCachedCameraView(*m_cachedCameraView);
        m_vkCore->SetDrawCalls(&m_graph->GetCachedDrawCalls());
        m_vkCore->SetShadowDrawCalls(m_graph->HasCachedShadowDrawCalls() ? &m_graph->GetCachedShadowDrawCalls()
                                                                         : nullptr);
        MarkSubmitted(m_graph->GetObjectBindingIdentity(), true);
#if INFERNUX_FRAME_PROFILE
        g_srcProfileSnapshot.cachedSubmissionReuses += 1.0;
        g_srcProfileSnapshot.submitMs += std::chrono::duration<double, std::milli>(Clock::now() - submitStart).count();
        g_srcProfileSnapshot.submitCalls += 1.0;
        g_srcProfileSnapshot.finalDrawCalls += static_cast<double>(m_graph->GetCachedDrawCalls().size());
#endif
        return;
    }
#if INFERNUX_FRAME_PROFILE
    g_srcProfileSnapshot.submissionRejectEditorAppenders += hasEditorAppenders ? 1.0 : 0.0;
    g_srcProfileSnapshot.submissionRejectOwnedVisibleList += visibleListReusable ? 0.0 : 1.0;
    g_srcProfileSnapshot.submissionRejectOwnedShadowList += shadowListReusable ? 0.0 : 1.0;
    g_srcProfileSnapshot.submissionRejectMissingGraph += graphAvailable ? 0.0 : 1.0;
    g_srcProfileSnapshot.submissionRejectSignature += signatureReusable ? 0.0 : 1.0;
    if (baseRendererListBorrowed) {
        g_srcProfileSnapshot.borrowedRendererListSubmits += 1.0;
        g_srcProfileSnapshot.materializedDrawCalls += static_cast<double>(baseDrawCount);
    } else {
        g_srcProfileSnapshot.ownedRendererListSubmits += 1.0;
    }
#endif

    // Move or reference draw calls — avoids 1000+ shared_ptr atomic refcount ops.
#if INFERNUX_FRAME_PROFILE
    auto t0 = Clock::now();
#endif
    RenderDomainMask submittedDomains = culling.visibleRenderers.Domains();
    m_orderedDrawCalls = culling.visibleRenderers.Consume();
    const size_t baseOrderedDrawCallCount = m_orderedDrawCalls.size();
    m_orderedDrawCalls.reserve(m_orderedDrawCalls.size() + 16);
    submittedDomains |= ProcessPendingCommandBuffers();
#if INFERNUX_FRAME_PROFILE
    g_srcProfileSnapshot.submitBaseMs += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
#endif

    // Append skybox draw call only when ClearFlags == Skybox (or no camera set)
    const bool drawSkybox = !m_cachedCameraView->valid || m_cachedCameraView->clearFlags == CameraClearFlags::Skybox;

    if (drawSkybox) {
#if INFERNUX_FRAME_PROFILE
        t0 = Clock::now();
#endif
        // Scene environment picks the skybox material (falls back to the
        // builtin procedural sky when unset or unloadable).
        std::shared_ptr<InxMaterial> skyboxMat;
        if (m_renderWorldOwner && m_renderWorldOwner->Lighting())
            skyboxMat = m_renderWorldOwner->Lighting()->skyboxMaterial;
        if (skyboxMat) {
            static constexpr uint64_t SKYBOX_OBJECT_ID = 0xFFFFFFFFFFFFFF00ULL;
            DrawCall dc;
            dc.indexStart = 0;
            dc.indexCount = static_cast<uint32_t>(PrimitiveMeshes::GetSkyboxCubeIndices().size());
            dc.worldMatrix = glm::mat4(1.0f);
            dc.material = skyboxMat;
            dc.objectId = SKYBOX_OBJECT_ID;
            dc.identity = RenderProxyHandle::Synthetic(RenderDomain::Skybox, dc.objectId).MakeDrawIdentity();
            dc.isStatic = true;
            dc.meshVertices = &PrimitiveMeshes::GetSkyboxCubeVertices();
            dc.meshIndices = &PrimitiveMeshes::GetSkyboxCubeIndices();
            m_orderedDrawCalls.push_back(dc);
            submittedDomains |= RenderDomainBit(RenderDomain::Skybox);
        }
#if INFERNUX_FRAME_PROFILE
        g_srcProfileSnapshot.submitEditorAppendMs +=
            std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
#endif
    }

    // Auto-append editor gizmos
    if (m_gizmoCtx.gizmos) {
#if INFERNUX_FRAME_PROFILE
        t0 = Clock::now();
#endif
        DrawCallResult gizmoResult = m_gizmoCtx.gizmos->GetDrawCalls(m_gizmoCtx.gizmoMaterial, m_gizmoCtx.gridMaterial);
        if (!gizmoResult.drawCalls.empty())
            submittedDomains |= RenderDomainBit(RenderDomain::EditorGizmo);
        for (auto &dc : gizmoResult.drawCalls) {
            m_orderedDrawCalls.push_back(dc);
        }
#if INFERNUX_FRAME_PROFILE
        g_srcProfileSnapshot.submitEditorAppendMs +=
            std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
#endif
    }

    // Auto-append component gizmos (script-side, depth-tested)
    if (m_gizmoCtx.componentGizmos && m_gizmoCtx.componentGizmos->HasData()) {
#if INFERNUX_FRAME_PROFILE
        t0 = Clock::now();
#endif
        DrawCallResult compGizmoResult = m_gizmoCtx.componentGizmos->GetDrawCalls(m_gizmoCtx.componentGizmosMaterial);
        if (!compGizmoResult.drawCalls.empty())
            submittedDomains |= RenderDomainBit(RenderDomain::ComponentGizmo);
        for (auto &dc : compGizmoResult.drawCalls) {
            m_orderedDrawCalls.push_back(dc);
        }
#if INFERNUX_FRAME_PROFILE
        g_srcProfileSnapshot.submitEditorAppendMs +=
            std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
#endif
    }

    const bool hasIcons = m_gizmoCtx.componentGizmos && m_gizmoCtx.componentGizmos->HasIconData();
    const glm::mat4 editorCameraToWorld =
        hasIcons || m_gizmoCtx.editorTools ? glm::inverse(m_cachedView) : glm::mat4(1.0f);
    const glm::vec3 editorCameraPosition(editorCameraToWorld[3]);

    // Auto-append component gizmo icons using the Scene camera snapshot.
    if (hasIcons) {
#if INFERNUX_FRAME_PROFILE
        t0 = Clock::now();
#endif
        const GizmosDrawCallBuffer::IconMaterials iconMaterials{
            m_gizmoCtx.componentGizmoIconMaterial, m_gizmoCtx.cameraGizmoIconMaterial,
            m_gizmoCtx.lightGizmoIconMaterial, m_gizmoCtx.particleGizmoIconMaterial};
        DrawCallResult iconResult = m_gizmoCtx.componentGizmos->GetIconDrawCalls(
            iconMaterials, editorCameraPosition, glm::vec3(editorCameraToWorld[0]), glm::vec3(editorCameraToWorld[1]),
            m_cachedProj, m_graph ? m_graph->GetRenderViewContext().height : 1u, m_gizmoCtx.iconDpiScale);
        if (!iconResult.drawCalls.empty())
            submittedDomains |= RenderDomainBit(RenderDomain::ComponentGizmo);
        for (auto &dc : iconResult.drawCalls) {
            m_orderedDrawCalls.push_back(dc);
        }
#if INFERNUX_FRAME_PROFILE
        g_srcProfileSnapshot.submitEditorAppendMs +=
            std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
#endif
    }

    // Auto-append editor tools (translate/rotate/scale handles)
    if (m_gizmoCtx.editorTools) {
#if INFERNUX_FRAME_PROFILE
        t0 = Clock::now();
#endif
        DrawCallResult toolsResult = m_gizmoCtx.editorTools->GetDrawCalls(
            m_gizmoCtx.editorToolsMaterial, m_gizmoCtx.selectedObjectId, m_gizmoCtx.activeScene, editorCameraPosition);
        if (!toolsResult.drawCalls.empty())
            submittedDomains |= RenderDomainBit(RenderDomain::EditorTool);
        for (auto &dc : toolsResult.drawCalls) {
            m_orderedDrawCalls.push_back(dc);
        }
#if INFERNUX_FRAME_PROFILE
        g_srcProfileSnapshot.submitEditorAppendMs +=
            std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
#endif
    }

    const std::vector<DrawCall> *shadowSource =
        culling.shadowCasters.Empty() ? nullptr : &culling.shadowCasters.DrawCalls();

    // Compute a lightweight resource-identity fingerprint. World matrices and
    // visibility intentionally do not participate: neither changes mesh GPU
    // buffers. The fingerprint still detects a same-sized but different
    // visible set after camera movement.
    uint64_t bufferIdentity = 1469598103934665603ULL;
    bool hasForcedBufferUpdate = false;
    auto accumulateBufferIdentity = [&](const DrawCall &dc) {
        auto mix = [&](uint64_t value) {
            bufferIdentity ^= value;
            bufferIdentity *= 1099511628211ULL;
        };
        mix(dc.objectId);
        mix(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dc.meshVertices)));
        mix(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dc.meshIndices)));
        mix(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dc.meshVertexBuffer.get())));
        mix(std::hash<std::string>{}(dc.meshAssetGuid));
        mix(dc.meshRuntimeVersion);
        mix(dc.meshUsesDetachedStorage);
        mix(static_cast<uint64_t>(dc.meshIndexFormat));
        mix(static_cast<uint64_t>(dc.meshGeometryView));
        hasForcedBufferUpdate = hasForcedBufferUpdate || dc.forceBufferUpdate;
    };
    if (shadowSource) {
        for (const DrawCall &dc : *shadowSource)
            accumulateBufferIdentity(dc);
    }
    for (const DrawCall &dc : m_orderedDrawCalls)
        accumulateBufferIdentity(dc);

    const bool reuseObjectBuffers = m_graph && !hasForcedBufferUpdate &&
                                    m_graph->HasReadyObjectBindings(objectBufferRevision) &&
                                    m_graph->GetObjectBindingIdentity() == bufferIdentity;

#if INFERNUX_FRAME_PROFILE
    t0 = Clock::now();
#endif
    bool objectBuffersReady = true;
    if (!reuseObjectBuffers) {
        uint64_t lastEnsuredId = 0;
        const auto ensure = [&](const std::vector<DrawCall> &draws, size_t start) {
            for (size_t index = start; index < draws.size(); ++index) {
                const auto &draw = draws[index];
                if (draw.objectId == lastEnsuredId || !draw.meshVertices || !draw.meshIndices)
                    continue;
                lastEnsuredId = draw.objectId;
                objectBuffersReady &= m_vkCore->PrepareObjectBuffers(draw);
            }
        };
        // The scene's shadow list already contains all scene geometry. Only
        // explicit draws, skybox and editor additions need a second traversal.
        if (shadowSource)
            ensure(*shadowSource, 0);
        ensure(m_orderedDrawCalls, shadowSource ? baseOrderedDrawCallCount : 0);
    }
#if INFERNUX_FRAME_PROFILE
    g_srcProfileSnapshot.ensureBuffersMs += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
#endif

    // Cache draw calls AND camera VP on the associated render graph.
    // Each SceneRenderGraph stores its own draw-call set and VP matrices
    // so the executor lambda can swap them before each graph execution,
    // ensuring full isolation between Scene View and Game View rendering.
    if (m_graph) {
#if INFERNUX_FRAME_PROFILE
        t0 = Clock::now();
#endif
        m_graph->SetCachedRendererList(
            RendererList::Own(std::move(m_orderedDrawCalls), RendererListPurpose::CameraVisible, submittedDomains));
        if (shadowSource) {
            m_graph->SetCachedShadowRendererList(std::move(culling.shadowCasters));
        } else {
            m_graph->ClearCachedShadowDrawCalls();
        }
        m_graph->SetCachedSubmissionSignature(submissionSignature, culling.renderWorldOwner,
                                              objectBuffersReady ? m_vkCore->GetObjectBufferRevision() : 0);
        if (m_cachedCameraView->valid) {
            m_graph->SetCachedCameraView(*m_cachedCameraView);
        }
        // Point VkCore at the graph's cached copy (survives this scope).
        m_vkCore->SetDrawCalls(&m_graph->GetCachedDrawCalls(), true);
        m_vkCore->SetShadowDrawCalls(
            m_graph->HasCachedShadowDrawCalls() ? &m_graph->GetCachedShadowDrawCalls() : nullptr, true);
#if INFERNUX_FRAME_PROFILE
        g_srcProfileSnapshot.cacheGraphMs += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
#endif
    }

    MarkSubmitted(bufferIdentity, reuseObjectBuffers);
#if INFERNUX_FRAME_PROFILE
    g_srcProfileSnapshot.submitMs += std::chrono::duration<double, std::milli>(Clock::now() - submitStart).count();
    g_srcProfileSnapshot.submitCalls += 1.0;
    g_srcProfileSnapshot.finalDrawCalls +=
        static_cast<double>(m_graph ? m_graph->GetCachedDrawCalls().size() : baseDrawCount);
#endif
}

bool ScriptableRenderContext::IsNativeGraphCurrent(uint64_t sourceRevision) const
{
    return !m_recordedPlan && m_graph && m_graph->IsPythonGraphCurrent(sourceRevision);
}

void ScriptableRenderContext::MarkSubmitted(uint64_t objectBindingIdentity, bool reusesObjectBindings)
{
    m_submitted = true;
    if (m_graph)
        m_graph->RequestView(m_draw, objectBindingIdentity, reusesObjectBindings);
}

bool ScriptableRenderContext::CaptureScreenUI(InxScreenUIRenderer &renderer)
{
    if (!m_captureCompletedWorld || !m_recordedPlan)
        throw std::logic_error("World UI preparation requires its owning view capture");
    if (!m_recordedPlan->ui) {
        if (auto snapshot = renderer.CaptureWorldCommands(m_renderWorldOwner))
            SetScreenUISnapshot(std::move(snapshot));
        else
            return false;
    }
    return renderer.PrepareSnapshot(m_recordedPlan->ui);
}

bool ScriptableRenderContext::ExecutePlan(const std::shared_ptr<const RenderViewPlan> &plan)
{
    if (!plan)
        throw std::logic_error("A view execution requires its captured plan");
    if (plan->state != RenderViewPlanState::Submitted)
        return false;
    if (plan->ui && !m_renderWorldOwner->OwnsUI(plan->ui->World()))
        throw std::logic_error("Game UI plan does not own the selected completed World");
    if (!m_cachedCameraView->editorView && m_renderWorldOwner->CompletedWorldFrame() && !plan->ui)
        throw std::logic_error("Completed Game view plan has no UI publication owner");
    if ((m_renderWorldOwner->CompletedWorldFrame() || m_renderWorldOwner->IsPausedAuthorPresentation()) &&
        !plan->worldUI)
        throw std::logic_error("Runtime view requires its shared World UI publication");
    if (plan->worldUI && !m_renderWorldOwner->OwnsUI(plan->worldUI->World()))
        throw std::logic_error("World UI plan lost its shared World owner");
    m_graph->SetScreenUISnapshot(plan->ui, plan->worldUI);
    const auto &previous = m_graph->GetAppliedPlan();
    if (plan->graph &&
        (!previous || previous->graph != plan->graph || !m_graph->IsPythonGraphCurrent(plan->graph->sourceRevision))) {
        if (!ApplyGraph(*plan->graph))
            return false;
    }
    UpdateParameterBlocks(plan->parameters);
    m_graph->SetAppliedPlan(plan);
    m_orderedDrawCalls = plan->draws;
    m_pendingCommands = plan->commands;
    CullingResults culling = plan->culling;
    SubmitCulling(culling);
    return m_submitted;
}

} // namespace infernux
