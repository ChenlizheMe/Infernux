#include "ScriptableRenderContext.h"
#include "CommandBuffer.h"
#include "EditorGizmos.h"
#include "EditorTools.h"
#include "GizmosDrawCallBuffer.h"
#include "InxVkCoreModular.h"
#include "SceneRenderGraph.h"
#include "TransientResourcePool.h"
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/scene/SceneRenderer.h>

#include <core/log/InxLog.h>

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace infernux
{

#if INFERNUX_FRAME_PROFILE
namespace
{
ScriptableRenderContext::ProfileSnapshot g_srcProfileSnapshot;
}

ScriptableRenderContext::ProfileSnapshot ScriptableRenderContext::GetProfileSnapshot()
{
    return g_srcProfileSnapshot;
}

void ScriptableRenderContext::ResetProfileSnapshot()
{
    g_srcProfileSnapshot = {};
}
#endif

// ============================================================================
// Construction
// ============================================================================

ScriptableRenderContext::ScriptableRenderContext(InxVkCoreModular *vkCore, SceneRenderGraph *graph,
                                                 const EditorGizmosContext &gizmoCtx)
    : m_vkCore(vkCore), m_graph(graph), m_gizmoCtx(gizmoCtx)
{
    // Resolve the scene pointer: prefer the gizmo context's activeScene,
    // fall back to SceneManager's active scene.
    m_scene = gizmoCtx.activeScene;
    if (!m_scene) {
        m_scene = SceneManager::Instance().GetActiveScene();
    }
}

// ============================================================================
// SetupCameraProperties
// ============================================================================

void ScriptableRenderContext::SetupCameraProperties(Camera *camera)
{
    m_activeCamera = camera;

    // Snapshot the camera's VP matrices NOW so that the executor lambda
    // uses the exact same values even if the camera transform is modified
    // later in the frame.  The cached matrices are propagated to the
    // associated SceneRenderGraph in SubmitCulling().
    if (camera) {
        m_cachedView = camera->GetViewMatrix();
        m_cachedProj = camera->GetProjectionMatrix();

        // Propagate Camera clear flags / background color to the render graph
        // so the MainColor pass uses the correct clear behaviour this frame.
        if (m_graph) {
            m_graph->UpdateMainPassClearSettings(camera->GetClearFlags(), camera->GetBackgroundColor());
        }
    }
}

// ============================================================================
// Cull
// ============================================================================

CullingResults ScriptableRenderContext::Cull(Camera *camera)
{
#if INFERNUX_FRAME_PROFILE
    using Clock = std::chrono::high_resolution_clock;
    const auto cullStart = Clock::now();
#endif
    if (m_hasCullData) {
        // Multiple Cull() calls in one frame: return cached results with a
        // warning rather than crashing.  Multi-camera rendering within a
        // single context is not yet supported; create a new SRC per camera.
        INXLOG_WARN("ScriptableRenderContext::Cull() called more than once — "
                    "returning cached results. Create a new context per camera "
                    "for multi-camera rendering.");
        return m_cachedCullingResults;
    }

    SceneRenderBridge &bridge = SceneRenderBridge::Instance();
    Camera *editorCam = bridge.GetEditorCamera();

    // Pointer to draw calls — avoids copying the entire vector of DrawCalls
    // (each DrawCall contains a shared_ptr<InxMaterial> whose atomic refcount
    // would be bumped N times on copy).
    const std::vector<DrawCall> *drawCallsPtr = nullptr;
    const bool needsShadowDrawCalls =
        (m_vkCore->GetLightCollector().GetShadowCascadeCount() > 0);
    CameraDrawCallResult ownedResult; // Only used for game camera path

    if (camera && camera != editorCam) {
        // Non-editor camera (e.g. Game View camera): reuse editor camera's
        // already-collected renderables, re-cull with this camera's frustum.
        ownedResult = bridge.CullAndBuildForCamera(camera, needsShadowDrawCalls);
        drawCallsPtr = &ownedResult.visibleDrawCalls;
    } else {
        // Editor camera: reuse the already-prepared frame data from
        // SceneRenderBridge::PrepareFrame() (called earlier in DrawFrame).
        // BuildDrawCalls returns const ref — zero copy.
        const DrawCallResult &cached = bridge.BuildDrawCalls();
        drawCallsPtr = &cached.drawCalls;
    }

    m_hasCullData = true;

    CullingResults results;
    if (camera && camera != editorCam) {
        results.drawCalls = std::move(ownedResult.visibleDrawCalls); // game camera: move visible forward list
        if (needsShadowDrawCalls) {
            if (ownedResult.shadowDrawCallsRef) {
                // All-layers game camera: zero-copy reference to cached draw calls.
                results.shadowDrawCallsRef = ownedResult.shadowDrawCallsRef;
            } else {
                results.shadowDrawCalls = std::move(ownedResult.shadowDrawCalls);
            }
        }
    } else {
        // Editor camera: store a non-owning pointer instead of copying
        // 14,400+ DrawCalls with shared_ptr atomic refcount bumps.
        results.sceneDrawCallsRef = drawCallsPtr;
        if (needsShadowDrawCalls) {
            results.shadowDrawCallsRef = drawCallsPtr;
        }
    }
    // Populate visible light count from the scene light collector.
    // CollectLights() runs earlier in the frame (InxRenderer::UpdateSceneLighting),
    // so the count is already available.
    results.lightCount = m_vkCore->GetLightCollector().GetTotalLightCount();
    m_cachedCullingResults = results;
#if INFERNUX_FRAME_PROFILE
    const double elapsedMs = std::chrono::duration<double, std::milli>(Clock::now() - cullStart).count();
    g_srcProfileSnapshot.cullMs += elapsedMs;
    if (camera && camera != editorCam) {
        g_srcProfileSnapshot.cullGameMs += elapsedMs;
        g_srcProfileSnapshot.cullGameCalls += 1.0;
    } else {
        g_srcProfileSnapshot.cullEditorMs += elapsedMs;
        g_srcProfileSnapshot.cullEditorCalls += 1.0;
    }
    g_srcProfileSnapshot.cullCalls += 1.0;
    g_srcProfileSnapshot.baseDrawCalls += static_cast<double>(results.visibleObjectCount());
#endif
    return results;
}

// ============================================================================
// RenderGraph-driven API
// ============================================================================

void ScriptableRenderContext::ApplyGraph(const RenderGraphDescription &desc)
{
#if INFERNUX_FRAME_PROFILE
    using Clock = std::chrono::high_resolution_clock;
    const auto t0 = Clock::now();
#endif
    if (m_graph) {
        m_graph->ApplyPythonGraph(desc);
    } else {
        INXLOG_WARN("ScriptableRenderContext::ApplyGraph: No SceneRenderGraph available");
    }
#if INFERNUX_FRAME_PROFILE
    g_srcProfileSnapshot.applyGraphMs += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
#endif
}

void ScriptableRenderContext::SubmitCulling(CullingResults culling)
{
#if INFERNUX_FRAME_PROFILE
    using Clock = std::chrono::high_resolution_clock;
    const auto submitStart = Clock::now();
    const size_t baseDrawCount = culling.sceneDrawCallsRef ? culling.sceneDrawCallsRef->size() : culling.drawCalls.size();
#endif
    if (m_submitted) {
        INXLOG_WARN("ScriptableRenderContext::SubmitCulling() called after already submitted");
        return;
    }

    // Move or reference draw calls — avoids 1000+ shared_ptr atomic refcount ops.
#if INFERNUX_FRAME_PROFILE
    auto t0 = Clock::now();
#endif
    if (culling.sceneDrawCallsRef) {
        // Editor camera fast path: reference scene draw calls directly,
        // then move them into the ordered list (one move, zero shared_ptr copies).
        m_orderedDrawCalls = *culling.sceneDrawCallsRef;
    } else {
        m_orderedDrawCalls = std::move(culling.drawCalls);
    }
    const size_t baseOrderedDrawCallCount = m_orderedDrawCalls.size();
    m_orderedDrawCalls.reserve(m_orderedDrawCalls.size() + 16);
#if INFERNUX_FRAME_PROFILE
    g_srcProfileSnapshot.submitBaseMs += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
#endif

    // Append skybox draw call only when ClearFlags == Skybox (or no camera set)
    bool drawSkybox = true;
    if (m_activeCamera) {
        CameraClearFlags flags = m_activeCamera->GetClearFlags();
        drawSkybox = (flags == CameraClearFlags::Skybox);
    }

    if (drawSkybox) {
#if INFERNUX_FRAME_PROFILE
        t0 = Clock::now();
#endif
        auto skyboxMat = AssetRegistry::Instance().GetBuiltinMaterial("SkyboxProcedural");
        if (skyboxMat) {
            static constexpr uint64_t SKYBOX_OBJECT_ID = 0xFFFFFFFFFFFFFF00ULL;
            DrawCall dc;
            dc.indexStart = 0;
            dc.indexCount = static_cast<uint32_t>(PrimitiveMeshes::GetSkyboxCubeIndices().size());
            dc.worldMatrix = glm::mat4(1.0f);
            dc.material = skyboxMat.get();
            dc.objectId = SKYBOX_OBJECT_ID;
            dc.meshVertices = &PrimitiveMeshes::GetSkyboxCubeVertices();
            dc.meshIndices = &PrimitiveMeshes::GetSkyboxCubeIndices();
            m_orderedDrawCalls.push_back(dc);
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
        DrawCallResult gizmoResult =
            m_gizmoCtx.gizmos->GetDrawCalls(m_gizmoCtx.gizmoMaterial, m_gizmoCtx.gridMaterial,
                                            m_gizmoCtx.selectedObjectId, m_gizmoCtx.activeScene, m_gizmoCtx.cameraPos);
        for (auto &dc : gizmoResult.drawCalls) {
            m_orderedDrawCalls.push_back(dc);
        }
#if INFERNUX_FRAME_PROFILE
        g_srcProfileSnapshot.submitEditorAppendMs +=
            std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
#endif
    }

    // Auto-append component gizmos (Python-driven, depth-tested)
    if (m_gizmoCtx.componentGizmos && m_gizmoCtx.componentGizmos->HasData()) {
#if INFERNUX_FRAME_PROFILE
        t0 = Clock::now();
#endif
        DrawCallResult compGizmoResult = m_gizmoCtx.componentGizmos->GetDrawCalls(m_gizmoCtx.componentGizmosMaterial);
        for (auto &dc : compGizmoResult.drawCalls) {
            m_orderedDrawCalls.push_back(dc);
        }
#if INFERNUX_FRAME_PROFILE
        g_srcProfileSnapshot.submitEditorAppendMs +=
            std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
#endif
    }

    // Auto-append component gizmo icons (Python-driven, billboard diamonds)
    if (m_gizmoCtx.componentGizmos && m_gizmoCtx.componentGizmos->HasIconData()) {
#if INFERNUX_FRAME_PROFILE
        t0 = Clock::now();
#endif
        glm::vec3 cameraRight(1.0f, 0.0f, 0.0f);
        glm::vec3 cameraUp(0.0f, 1.0f, 0.0f);
        if (m_activeCamera && m_activeCamera->GetGameObject() && m_activeCamera->GetGameObject()->GetTransform()) {
            Transform *cameraTransform = m_activeCamera->GetGameObject()->GetTransform();
            cameraRight = cameraTransform->GetWorldRight();
            cameraUp = cameraTransform->GetWorldUp();
        }
        DrawCallResult iconResult = m_gizmoCtx.componentGizmos->GetIconDrawCalls(
            m_gizmoCtx.componentGizmoIconMaterial, m_gizmoCtx.cameraGizmoIconMaterial,
            m_gizmoCtx.lightGizmoIconMaterial, m_gizmoCtx.cameraPos, cameraRight, cameraUp);
        static size_t s_lastSubmittedIconDrawCalls = static_cast<size_t>(-1);
        if (s_lastSubmittedIconDrawCalls != iconResult.drawCalls.size()) {
            s_lastSubmittedIconDrawCalls = iconResult.drawCalls.size();
        }
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
            m_gizmoCtx.editorToolsMaterial, m_gizmoCtx.selectedObjectId, m_gizmoCtx.activeScene, m_gizmoCtx.cameraPos);
        for (auto &dc : toolsResult.drawCalls) {
            m_orderedDrawCalls.push_back(dc);
        }
#if INFERNUX_FRAME_PROFILE
        g_srcProfileSnapshot.submitEditorAppendMs +=
            std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
#endif
    }

    const std::vector<DrawCall> *shadowSource = culling.shadowDrawCallsRef;
    if (!shadowSource && !culling.shadowDrawCalls.empty()) {
        shadowSource = &culling.shadowDrawCalls;
    }

    // Ensure per-object GPU buffers
#if INFERNUX_FRAME_PROFILE
    t0 = Clock::now();
#endif
    if (shadowSource) {
        // Consecutive-objectId dedup: draw calls for multi-submesh objects
        // share the same objectId and are adjacent in the array.  Skip
        // redundant hash-map lookups inside EnsureObjectBuffers.
        uint64_t lastEnsuredId = 0;
        for (const DrawCall &dc : *shadowSource) {
            if (dc.objectId == lastEnsuredId) continue;
            lastEnsuredId = dc.objectId;
            if (dc.meshVertices && dc.meshIndices) {
                m_vkCore->EnsureObjectBuffers(dc.objectId, *dc.meshVertices, *dc.meshIndices, dc.forceBufferUpdate);
            }
        }

        for (size_t drawCallIndex = baseOrderedDrawCallCount; drawCallIndex < m_orderedDrawCalls.size(); ++drawCallIndex) {
            const DrawCall &dc = m_orderedDrawCalls[drawCallIndex];
            if (dc.objectId == lastEnsuredId) continue;
            lastEnsuredId = dc.objectId;
            if (dc.meshVertices && dc.meshIndices) {
                m_vkCore->EnsureObjectBuffers(dc.objectId, *dc.meshVertices, *dc.meshIndices, dc.forceBufferUpdate);
            }
        }
    }

    // Build the forward-render list.
    // Game camera path already has a compact visible-only list (move it).
    // Editor camera path: skip visibility pre-filter — DrawSceneFiltered
    // already checks frustumVisible per draw call, so pre-filtering is
    // redundant memcpy of ~2.7k DrawCalls.
    std::vector<DrawCall> forwardDrawCalls = std::move(m_orderedDrawCalls);

    if (!shadowSource) {
        for (const DrawCall &dc : forwardDrawCalls) {
            if (dc.meshVertices && dc.meshIndices) {
                m_vkCore->EnsureObjectBuffers(dc.objectId, *dc.meshVertices, *dc.meshIndices, dc.forceBufferUpdate);
            }
        }
    }

    DrawCallResult result;
    result.drawCalls = std::move(forwardDrawCalls);
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
        m_graph->SetCachedDrawCalls(std::move(result.drawCalls));
        if (shadowSource) {
            if (culling.shadowDrawCallsRef) {
                // Editor camera: shadow source is the scene's cached draw calls.
                // Use a zero-copy reference instead of copying 10k+ DrawCalls.
                m_graph->SetCachedShadowDrawCallsRef(culling.shadowDrawCallsRef);
            } else {
                m_graph->SetCachedShadowDrawCalls(std::move(culling.shadowDrawCalls));
            }
        } else {
            m_graph->ClearCachedShadowDrawCalls();
        }
        if (m_activeCamera) {
            m_graph->SetCachedCameraVP(m_cachedView, m_cachedProj);
        }
        // Point VkCore at the graph's cached copy (survives this scope).
        m_vkCore->SetDrawCalls(&m_graph->GetCachedDrawCalls());
#if INFERNUX_FRAME_PROFILE
        g_srcProfileSnapshot.cacheGraphMs += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
#endif
    }

    // NOTE: CleanupUnusedBuffers is called by InxRenderer::DrawFrame() after
    // all pipeline renders, using the union of all graphs' draw calls.
    // This prevents one graph's cleanup from removing buffers another graph needs.

    // Release transient resources
    if (m_transientPool) {
        m_transientPool->EndFrame();
    }

    m_submitted = true;
#if INFERNUX_FRAME_PROFILE
    g_srcProfileSnapshot.submitMs += std::chrono::duration<double, std::milli>(Clock::now() - submitStart).count();
    g_srcProfileSnapshot.submitCalls += 1.0;
    g_srcProfileSnapshot.finalDrawCalls += static_cast<double>(m_graph ? m_graph->GetCachedDrawCalls().size()
                                                                       : baseDrawCount);
#endif
}

void ScriptableRenderContext::RenderWithGraph(Camera *camera, const RenderGraphDescription &desc)
{
    SetupCameraProperties(camera);
    CullingResults culling = Cull(camera);
    ApplyGraph(desc);
    SubmitCulling(std::move(culling));
}

// ============================================================================
// Phase 2: CommandBuffer Integration
// ============================================================================

void ScriptableRenderContext::ExecuteCommandBuffer(CommandBuffer &cmd)
{
    // Accumulate pending CommandBuffers; they are processed during Submit()
    m_pendingCommandBuffers.push_back(&cmd);
}

void ScriptableRenderContext::ProcessPendingCommandBuffers()
{
    for (CommandBuffer *cmd : m_pendingCommandBuffers) {
        if (!cmd)
            continue;

        for (const auto &command : cmd->GetCommands()) {
            switch (command.type) {
            case RenderCommandType::GetTemporaryRT: {
                // Allocate from transient pool
                if (m_transientPool) {
                    const auto &params = std::get<GetTemporaryRTParams>(command.data);
                    uint32_t slotId =
                        m_transientPool->Acquire(params.width, params.height, params.format, params.samples);
                    m_handleToSlotMap[params.handleId] = slotId;
                }
                break;
            }

            case RenderCommandType::ReleaseTemporaryRT: {
                if (m_transientPool) {
                    const auto &params = std::get<ReleaseTemporaryRTParams>(command.data);
                    auto it = m_handleToSlotMap.find(params.handleId);
                    if (it != m_handleToSlotMap.end()) {
                        m_transientPool->Release(it->second);
                        m_handleToSlotMap.erase(it);
                    }
                }
                break;
            }

            case RenderCommandType::SetGlobalFloat: {
                const auto &params = std::get<SetGlobalFloatParams>(command.data);
                m_globalFloats[params.name] = params.value;
                break;
            }

            case RenderCommandType::SetGlobalVector: {
                const auto &params = std::get<SetGlobalVectorParams>(command.data);
                m_globalVectors[params.name] = {params.x, params.y, params.z, params.w};
                break;
            }

            case RenderCommandType::SetGlobalTexture: {
                const auto &params = std::get<SetGlobalTextureParams>(command.data);
                m_globalTextures[params.name] = params.handleId;
                break;
            }

            case RenderCommandType::ClearRenderTarget:
            case RenderCommandType::SetRenderTarget:
            case RenderCommandType::DrawMesh:
            case RenderCommandType::SetGlobalMatrix:
            case RenderCommandType::RequestAsyncReadback:
                // These commands require actual Vulkan command buffer integration.
                // They are stubbed for now and will be fully implemented when
                // InxVkCoreModular gains multi-RT support.
                // For MVP: log and skip.
                static int stubWarnCount = 0;
                if (stubWarnCount++ < 5) {
                    INXLOG_WARN("CommandBuffer command type ", static_cast<int>(command.type),
                                " is not yet fully implemented in the Vulkan backend");
                }
                break;
            }
        }
    }
    m_pendingCommandBuffers.clear();
}

// ============================================================================
// Phase 2: Camera Target
// ============================================================================

RenderTargetHandle ScriptableRenderContext::GetCameraTarget(Camera * /*camera*/) const
{
    // Returns the sentinel CAMERA_TARGET_HANDLE.
    // At execution time, this resolves to the scene render target's resolved color image.
    return CAMERA_TARGET_HANDLE;
}

// ============================================================================
// Phase 2: Global Shader Parameters (immediate mode)
// ============================================================================

void ScriptableRenderContext::SetGlobalTexture(const std::string &name, RenderTargetHandle handle)
{
    m_globalTextures[name] = handle.id;
}

void ScriptableRenderContext::SetGlobalFloat(const std::string &name, float value)
{
    m_globalFloats[name] = value;
}

void ScriptableRenderContext::SetGlobalVector(const std::string &name, float x, float y, float z, float w)
{
    m_globalVectors[name] = {x, y, z, w};
}

} // namespace infernux
