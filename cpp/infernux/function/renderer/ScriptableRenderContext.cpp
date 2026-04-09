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
#include <stdexcept>

namespace infernux
{

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
    DrawCallResult ownedResult; // Only used for game camera path

    if (camera && camera != editorCam) {
        // Non-editor camera (e.g. Game View camera): reuse editor camera's
        // already-collected renderables, re-cull with this camera's frustum.
        ownedResult = bridge.CullAndBuildForCamera(camera);
        drawCallsPtr = &ownedResult.drawCalls;
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
        results.drawCalls = std::move(ownedResult.drawCalls); // game camera: move
    } else {
        results.drawCalls = *drawCallsPtr; // editor camera: copy (cached data must survive)
    }
    // Populate visible light count from the scene light collector.
    // CollectLights() runs earlier in the frame (InxRenderer::UpdateSceneLighting),
    // so the count is already available.
    results.lightCount = m_vkCore->GetLightCollector().GetTotalLightCount();
    m_cachedCullingResults = results;
    return results;
}

// ============================================================================
// RenderGraph-driven API
// ============================================================================

void ScriptableRenderContext::ApplyGraph(const RenderGraphDescription &desc)
{
    if (m_graph) {
        m_graph->ApplyPythonGraph(desc);
    } else {
        INXLOG_WARN("ScriptableRenderContext::ApplyGraph: No SceneRenderGraph available");
    }
}

void ScriptableRenderContext::SubmitCulling(CullingResults culling)
{
    if (m_submitted) {
        INXLOG_WARN("ScriptableRenderContext::SubmitCulling() called after already submitted");
        return;
    }

    // Move draw calls directly — avoids 1000+ shared_ptr atomic refcount ops.
    m_orderedDrawCalls = std::move(culling.drawCalls);
    m_orderedDrawCalls.reserve(m_orderedDrawCalls.size() + 16);

    // Append skybox draw call only when ClearFlags == Skybox (or no camera set)
    bool drawSkybox = true;
    if (m_activeCamera) {
        CameraClearFlags flags = m_activeCamera->GetClearFlags();
        drawSkybox = (flags == CameraClearFlags::Skybox);
    }

    if (drawSkybox) {
        auto skyboxMat = AssetRegistry::Instance().GetBuiltinMaterial("SkyboxProcedural");
        if (skyboxMat) {
            static constexpr uint64_t SKYBOX_OBJECT_ID = 0xFFFFFFFFFFFFFF00ULL;
            DrawCall dc;
            dc.indexStart = 0;
            dc.indexCount = static_cast<uint32_t>(PrimitiveMeshes::GetSkyboxCubeIndices().size());
            dc.worldMatrix = glm::mat4(1.0f);
            dc.material = skyboxMat;
            dc.objectId = SKYBOX_OBJECT_ID;
            dc.meshVertices = &PrimitiveMeshes::GetSkyboxCubeVertices();
            dc.meshIndices = &PrimitiveMeshes::GetSkyboxCubeIndices();
            m_orderedDrawCalls.push_back(dc);
        }
    }

    // Auto-append editor gizmos
    if (m_gizmoCtx.gizmos) {
        DrawCallResult gizmoResult =
            m_gizmoCtx.gizmos->GetDrawCalls(m_gizmoCtx.gizmoMaterial, m_gizmoCtx.gridMaterial,
                                            m_gizmoCtx.selectedObjectId, m_gizmoCtx.activeScene, m_gizmoCtx.cameraPos);
        for (auto &dc : gizmoResult.drawCalls) {
            m_orderedDrawCalls.push_back(dc);
        }
    }

    // Auto-append component gizmos (Python-driven, depth-tested)
    if (m_gizmoCtx.componentGizmos && m_gizmoCtx.componentGizmos->HasData()) {
        DrawCallResult compGizmoResult = m_gizmoCtx.componentGizmos->GetDrawCalls(m_gizmoCtx.componentGizmosMaterial);
        for (auto &dc : compGizmoResult.drawCalls) {
            m_orderedDrawCalls.push_back(dc);
        }
    }

    // Auto-append component gizmo icons (Python-driven, billboard diamonds)
    if (m_gizmoCtx.componentGizmos && m_gizmoCtx.componentGizmos->HasIconData()) {
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
    }

    // Auto-append editor tools (translate/rotate/scale handles)
    if (m_gizmoCtx.editorTools) {
        DrawCallResult toolsResult = m_gizmoCtx.editorTools->GetDrawCalls(
            m_gizmoCtx.editorToolsMaterial, m_gizmoCtx.selectedObjectId, m_gizmoCtx.activeScene, m_gizmoCtx.cameraPos);
        for (auto &dc : toolsResult.drawCalls) {
            m_orderedDrawCalls.push_back(dc);
        }
    }

    // Build final result
    DrawCallResult result;
    result.drawCalls = std::move(m_orderedDrawCalls);

    // Ensure per-object GPU buffers
    for (const DrawCall &dc : result.drawCalls) {
        if (dc.meshVertices && dc.meshIndices) {
            m_vkCore->EnsureObjectBuffers(dc.objectId, *dc.meshVertices, *dc.meshIndices, dc.forceBufferUpdate);
        }
    }

    // Cache draw calls AND camera VP on the associated render graph.
    // Each SceneRenderGraph stores its own draw-call set and VP matrices
    // so the executor lambda can swap them before each graph execution,
    // ensuring full isolation between Scene View and Game View rendering.
    if (m_graph) {
        m_graph->SetCachedDrawCalls(std::move(result.drawCalls));
        if (m_activeCamera) {
            m_graph->SetCachedCameraVP(m_cachedView, m_cachedProj);
        }
        // Point VkCore at the graph's cached copy (survives this scope).
        m_vkCore->SetDrawCalls(&m_graph->GetCachedDrawCalls());
    }

    // NOTE: CleanupUnusedBuffers is called by InxRenderer::DrawFrame() after
    // all pipeline renders, using the union of all graphs' draw calls.
    // This prevents one graph's cleanup from removing buffers another graph needs.

    // Release transient resources
    if (m_transientPool) {
        m_transientPool->EndFrame();
    }

    m_submitted = true;
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
