#include "SceneRenderer.h"
#include "SceneManager.h"
#include <algorithm>
#include <cstring>
#include <glm/gtc/matrix_transform.hpp>

namespace infernux
{

// ============================================================================
// SceneRenderer Implementation
// ============================================================================

void SceneRenderer::PrepareFrame()
{
    // Always use editor camera for scene rendering (Scene View).
    // Game View handles its own camera independently via PrepareFrame(Camera*).
    SceneManager &sm = SceneManager::Instance();
    m_activeCamera = sm.GetEditorCameraController().GetCamera();

    // Collect and cull renderables (editor camera sees all layers)
    CollectRenderables(0xFFFFFFFF);

    if (m_frustumCulling) {
        PerformCulling();
    }

    SortRenderables();
}

void SceneRenderer::PrepareFrame(Camera *camera)
{
    if (!camera) {
        m_renderables.clear();
        m_visibleCount = 0;
        return;
    }

    m_activeCamera = camera;

    // Collect renderables filtered by camera's culling mask
    CollectRenderables(camera->GetCullingMask());

    if (m_frustumCulling) {
        PerformCulling();
    }

    SortRenderables();
}

glm::mat4 SceneRenderer::GetViewMatrix() const
{
    if (m_activeCamera) {
        return m_activeCamera->GetViewMatrix();
    }
    return glm::mat4{1.0f};
}

glm::mat4 SceneRenderer::GetProjectionMatrix() const
{
    if (m_activeCamera) {
        return m_activeCamera->GetProjectionMatrix();
    }
    return glm::perspective(glm::radians(60.0f), 16.0f / 9.0f, 0.1f, 1000.0f);
}

glm::vec3 SceneRenderer::GetCameraPosition() const
{
    if (m_activeCamera && m_activeCamera->GetGameObject()) {
        return m_activeCamera->GetGameObject()->GetTransform()->GetWorldPosition();
    }
    return glm::vec3{0.0f, 0.0f, 5.0f};
}

glm::vec3 SceneRenderer::GetCameraForward() const
{
    if (m_activeCamera && m_activeCamera->GetGameObject()) {
        return m_activeCamera->GetGameObject()->GetTransform()->GetWorldForward();
    }
    return glm::vec3{0.0f, 0.0f, 1.0f};
}

glm::vec3 SceneRenderer::GetCameraUp() const
{
    if (m_activeCamera && m_activeCamera->GetGameObject()) {
        return m_activeCamera->GetGameObject()->GetTransform()->GetWorldUp();
    }
    return glm::vec3{0.0f, 1.0f, 0.0f};
}

void SceneRenderer::SetAspectRatio(float aspect)
{
    if (m_activeCamera) {
        m_activeCamera->SetAspectRatio(aspect);
    }
}

void SceneRenderer::CollectRenderables(uint32_t cullingMask)
{
    m_renderables.clear();

    Scene *activeScene = SceneManager::Instance().GetActiveScene();
    if (!activeScene)
        return;

    // Use the MeshRenderer registry — O(N) over active renderers only,
    // no GetAllObjects() tree walk, no dynamic_cast, no vector allocation.
    const auto &meshRenderers = SceneManager::Instance().GetActiveMeshRenderers();

    m_renderables.reserve(meshRenderers.size());

    for (MeshRenderer *renderer : meshRenderers) {
        if (!renderer || !renderer->IsEnabled())
            continue;

        GameObject *obj = renderer->GetGameObject();
        if (!obj || !obj->IsActiveInHierarchy())
            continue;

        // Layer-based culling: skip objects not in the camera's culling mask
        uint32_t objectLayerBit = 1u << static_cast<uint32_t>(obj->GetLayer());
        if ((cullingMask & objectLayerBit) == 0)
            continue;

        // Check if renderer has inline mesh, asset mesh, or mesh reference
        if (!renderer->HasMeshAsset() && !renderer->HasInlineMesh() && !renderer->GetMesh().IsValid())
            continue;

        RenderableObject renderable;
        renderable.objectId = obj->GetID();
        renderable.worldMatrix = obj->GetTransform()->GetWorldMatrix();
        renderable.mesh = renderer->GetMesh();
        renderable.renderMaterial = renderer->GetEffectiveMaterial(); // Get actual InxMaterial
        renderable.meshRenderer = renderer;                           // Store direct pointer

        // Get world-space bounding box for frustum culling
        glm::vec3 boundsMin, boundsMax;
        renderer->GetWorldBounds(boundsMin, boundsMax);
        renderable.worldBounds = AABB(boundsMin, boundsMax);
        renderable.visible = true; // Will be set by culling

        m_renderables.push_back(std::move(renderable));
    }

    m_visibleCount = m_renderables.size();
}

void SceneRenderer::PerformCulling()
{
    if (!m_activeCamera) {
        // No camera, mark all as visible
        m_visibleCount = m_renderables.size();
        return;
    }

    // Extract frustum from view-projection matrix
    glm::mat4 viewProj = m_activeCamera->GetViewProjectionMatrix();
    Frustum frustum;
    frustum.ExtractFromMatrix(viewProj);

    // Test each object against frustum
    m_visibleCount = 0;
    for (auto &renderable : m_renderables) {
        // Use AABB-frustum intersection test
        renderable.visible = frustum.IntersectsAABB(renderable.worldBounds);
        if (renderable.visible) {
            ++m_visibleCount;
        }
    }
}

void SceneRenderer::SortRenderables()
{
    // Sort by material render queue (for proper render order)
    // - Lower queue = render first (opaque before transparent)
    // - Within same queue, sort by material pointer (minimize state changes)

    std::sort(m_renderables.begin(), m_renderables.end(), [](const RenderableObject &a, const RenderableObject &b) {
        // Get render queues (default 2000 for opaque)
        int32_t queueA = a.renderMaterial ? a.renderMaterial->GetRenderQueue() : 2000;
        int32_t queueB = b.renderMaterial ? b.renderMaterial->GetRenderQueue() : 2000;

        if (queueA != queueB) {
            return queueA < queueB; // Lower queue first
        }

        // Same queue: sort by material pointer to minimize state changes
        return a.renderMaterial.get() < b.renderMaterial.get();
    });
}

// ============================================================================
// Shared draw-call emission (eliminates duplication between the two Build paths)
// ============================================================================
void SceneRenderer::EmitDrawCallsForRenderable(DrawCallResult &result, const RenderableObject &renderable, bool visible,
                                               bool bufferDirty) const
{
    MeshRenderer *renderer = renderable.meshRenderer;
    if (!renderer)
        return;

    if (renderer->HasMeshAsset()) {
        auto meshPtr = renderer->GetMeshAssetRef().Get();
        if (!meshPtr)
            return;
        const auto &objVertices = meshPtr->GetVertices();
        const auto &objIndices = meshPtr->GetIndices();
        if (objVertices.empty() || objIndices.empty())
            return;

        const glm::mat4 &worldMatrix = renderable.worldMatrix;

        uint32_t subMeshCount = meshPtr->GetSubMeshCount();
        int32_t submeshFilter = renderer->GetSubmeshIndex();
        int32_t nodeGroup = renderer->GetNodeGroup();
        if (subMeshCount == 0) {
            // Fallback: single draw call for entire mesh
            DrawCall dc;
            dc.indexStart = 0;
            dc.indexCount = static_cast<uint32_t>(objIndices.size());
            dc.vertexStart = 0;
            dc.worldMatrix = worldMatrix;
            dc.material = renderer->GetEffectiveMaterial(0);
            dc.objectId = renderable.objectId;
            dc.frustumVisible = visible;
            dc.worldBounds = renderable.worldBounds;
            dc.meshVertices = &objVertices;
            dc.meshIndices = &objIndices;
            dc.forceBufferUpdate = bufferDirty;
            result.drawCalls.push_back(dc);
        } else if (submeshFilter >= 0 && static_cast<uint32_t>(submeshFilter) < subMeshCount) {
            // Single submesh mode — render only the specified submesh
            const auto &sub = meshPtr->GetSubMesh(static_cast<uint32_t>(submeshFilter));
            glm::mat4 effectiveMatrix = worldMatrix;
            const glm::vec3 &pivot = renderer->GetMeshPivotOffset();
            if (pivot != glm::vec3(0.0f)) {
                effectiveMatrix = worldMatrix * glm::translate(glm::mat4(1.0f), pivot);
            }
            DrawCall dc;
            dc.indexStart = sub.indexStart;
            dc.indexCount = sub.indexCount;
            dc.vertexStart = 0;
            dc.worldMatrix = effectiveMatrix;
            dc.material = renderer->GetEffectiveMaterial(0);
            dc.objectId = renderable.objectId;
            dc.frustumVisible = visible;
            dc.worldBounds = renderable.worldBounds;
            dc.meshVertices = &objVertices;
            dc.meshIndices = &objIndices;
            dc.forceBufferUpdate = bufferDirty;
            result.drawCalls.push_back(dc);
        } else {
            // One DrawCall per submesh with its own material slot
            // Build local slot remap for nodeGroup so material indices are contiguous
            constexpr uint32_t SLOT_REMAP_CAP = 32;
            uint32_t slotRemap[SLOT_REMAP_CAP];
            std::memset(slotRemap, 0xFF, sizeof(slotRemap));
            uint32_t nextSlotIdx = 0;
            if (nodeGroup >= 0) {
                for (uint32_t si = 0; si < subMeshCount; ++si) {
                    const auto &s = meshPtr->GetSubMesh(si);
                    if (static_cast<int32_t>(s.nodeGroup) != nodeGroup)
                        continue;
                    if (s.materialSlot < SLOT_REMAP_CAP && slotRemap[s.materialSlot] == 0xFFFFFFFF)
                        slotRemap[s.materialSlot] = nextSlotIdx++;
                }
            }
            bool firstDirty = true;
            for (uint32_t si = 0; si < subMeshCount; ++si) {
                const auto &sub = meshPtr->GetSubMesh(si);
                if (nodeGroup >= 0 && static_cast<int32_t>(sub.nodeGroup) != nodeGroup)
                    continue;
                DrawCall dc;
                dc.indexStart = sub.indexStart;
                dc.indexCount = sub.indexCount;
                dc.vertexStart = 0;
                dc.worldMatrix = worldMatrix;
                uint32_t matSlot = sub.materialSlot;
                if (nodeGroup >= 0 && matSlot < SLOT_REMAP_CAP && slotRemap[matSlot] != 0xFFFFFFFF)
                    matSlot = slotRemap[matSlot];
                dc.material = renderer->GetEffectiveMaterial(matSlot);
                dc.objectId = renderable.objectId;
                dc.frustumVisible = visible;
                dc.worldBounds = renderable.worldBounds;
                dc.meshVertices = &objVertices;
                dc.meshIndices = &objIndices;
                dc.forceBufferUpdate = firstDirty ? bufferDirty : false;
                firstDirty = false;
                result.drawCalls.push_back(dc);
            }
        }
    } else if (renderer->HasInlineMesh()) {
        const auto &objVertices = renderer->GetInlineVertices();
        const auto &objIndices = renderer->GetInlineIndices();
        if (objVertices.empty() || objIndices.empty())
            return;

        const glm::mat4 &worldMatrix = renderable.worldMatrix;
        DrawCall dc;
        dc.indexStart = 0;
        dc.indexCount = static_cast<uint32_t>(objIndices.size());
        dc.vertexStart = 0;
        dc.worldMatrix = worldMatrix;
        dc.material = renderer->GetEffectiveMaterial(0);
        dc.objectId = renderable.objectId;
        dc.frustumVisible = visible;
        dc.worldBounds = renderable.worldBounds;
        dc.meshVertices = &objVertices;
        dc.meshIndices = &objIndices;
        dc.forceBufferUpdate = bufferDirty;
        result.drawCalls.push_back(dc);
    }
}

DrawCallResult SceneRenderer::BuildDrawCalls() const
{
    DrawCallResult result;
    result.drawCalls.reserve(m_renderables.size());

    for (const auto &renderable : m_renderables) {
        MeshRenderer *renderer = renderable.meshRenderer;
        if (!renderer)
            continue;

        bool bufferDirty = renderer->ConsumeMeshBufferDirty();
        EmitDrawCallsForRenderable(result, renderable, renderable.visible, bufferDirty);
    }

    return result;
}

DrawCallResult SceneRenderer::BuildDrawCallsForCamera(Camera *camera) const
{
    DrawCallResult result;
    if (!camera || m_renderables.empty())
        return result;

    const uint32_t cullingMask = camera->GetCullingMask();
    Frustum frustum;
    frustum.ExtractFromMatrix(camera->GetViewProjectionMatrix());

    result.drawCalls.reserve(m_renderables.size());

    for (const auto &renderable : m_renderables) {
        MeshRenderer *renderer = renderable.meshRenderer;
        if (!renderer)
            continue;

        // Layer mask filter
        GameObject *obj = renderer->GetGameObject();
        if (obj) {
            uint32_t layerBit = 1u << static_cast<uint32_t>(obj->GetLayer());
            if ((cullingMask & layerBit) == 0)
                continue;
        }

        // Frustum test using pre-computed world bounds
        const bool visible = m_frustumCulling ? frustum.IntersectsAABB(renderable.worldBounds) : true;

        // Don't consume dirty flag — scene view already did
        EmitDrawCallsForRenderable(result, renderable, visible, false);
    }

    return result;
}

// ============================================================================
// SceneRenderBridge Implementation
// ============================================================================

SceneRenderBridge &SceneRenderBridge::Instance()
{
    static SceneRenderBridge instance;
    return instance;
}

void SceneRenderBridge::UpdateCameraData(float *outPos, float *outLookAt, float *outUp)
{
    glm::vec3 pos = m_sceneRenderer.GetCameraPosition();
    glm::vec3 forward = m_sceneRenderer.GetCameraForward();
    glm::vec3 up = m_sceneRenderer.GetCameraUp();

    // LookAt is position + forward direction
    glm::vec3 lookAt = pos + forward;

    if (outPos) {
        outPos[0] = pos.x;
        outPos[1] = pos.y;
        outPos[2] = pos.z;
    }

    if (outLookAt) {
        outLookAt[0] = lookAt.x;
        outLookAt[1] = lookAt.y;
        outLookAt[2] = lookAt.z;
    }

    if (outUp) {
        outUp[0] = up.x;
        outUp[1] = up.y;
        outUp[2] = up.z;
    }
}

void SceneRenderBridge::OnWindowResize(uint32_t width, uint32_t height)
{
    if (height > 0) {
        float aspect = static_cast<float>(width) / static_cast<float>(height);
        m_sceneRenderer.SetAspectRatio(aspect);

        // Sync screen dimensions for ScreenToWorld/WorldToScreen
        Camera *editorCam = GetEditorCamera();
        if (editorCam) {
            editorCam->SetScreenDimensions(width, height);
        }
    }
}

void SceneRenderBridge::PrepareFrame()
{
    m_sceneRenderer.PrepareFrame();
}

DrawCallResult SceneRenderBridge::PrepareAndBuildForCamera(Camera *camera)
{
    // Use a temporary SceneRenderer for independent camera culling
    // so we don't disturb the editor camera's cached state.
    SceneRenderer tempRenderer;
    tempRenderer.SetFrustumCullingEnabled(m_sceneRenderer.IsFrustumCullingEnabled());
    tempRenderer.PrepareFrame(camera);
    return tempRenderer.BuildDrawCalls();
}

DrawCallResult SceneRenderBridge::CullAndBuildForCamera(Camera *camera) const
{
    // Reuse editor camera's already-collected renderables.
    // Only re-cull with the given camera's frustum + layer mask.
    return m_sceneRenderer.BuildDrawCallsForCamera(camera);
}

DrawCallResult SceneRenderBridge::BuildDrawCalls() const
{
    return m_sceneRenderer.BuildDrawCalls();
}

Camera *SceneRenderBridge::GetEditorCamera() const
{
    return SceneManager::Instance().GetEditorCameraController().GetCamera();
}

} // namespace infernux
