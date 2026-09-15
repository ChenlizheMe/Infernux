#include "SceneRenderExtractor.h"
#include "SceneManager.h"
#include "SkinnedMeshRenderer.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>

namespace infernux
{

namespace
{
struct InlineMeshSnapshot
{
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

std::shared_ptr<const InlineMeshSnapshot> CaptureInlineMesh(const MeshRenderer &renderer)
{
    auto owner = std::make_shared<InlineMeshSnapshot>();
    owner->vertices = renderer.GetInlineVertices();
    owner->indices = renderer.GetInlineIndices();
    return owner;
}

void CaptureOrReferenceInlineMesh(const MeshRenderer &renderer, const std::vector<Vertex> *&vertices,
                                  const std::vector<uint32_t> *&indices, std::shared_ptr<const void> &owner)
{
    if (renderer.HasSharedInlineMesh()) {
        // Primitive streams live in PrimitiveMeshes static storage. Retaining
        // the renderer's direct references avoids N identical allocations and
        // copies when a scene contains thousands of ordinary Cube objects.
        vertices = &renderer.GetInlineVertices();
        indices = &renderer.GetInlineIndices();
        owner.reset();
        return;
    }

    auto inlineOwner = CaptureInlineMesh(renderer);
    vertices = &inlineOwner->vertices;
    indices = &inlineOwner->indices;
    owner = std::move(inlineOwner);
}
} // namespace

size_t SceneRenderExtractor::ExtractEditorFrame(RenderWorldSnapshot &world, bool useActiveCameraCulling)
{
#if defined(INFERNUX_RUNTIME_MINIMAL_HOST)
    // Minimal platform players intentionally omit the editor-camera service.
    // Keeping this symbol available lets the portable extractor compile while
    // making accidental editor-view use fail closed.
    (void)useActiveCameraCulling;
    world.Clear();
    m_activeCamera = nullptr;
    m_sceneSources.clear();
    m_visibleCount = 0;
    return 0;
#else
    // Extraction is deliberately camera-neutral. Scene, Game, previews, and
    // future stacked Game cameras all consume this same immutable world and
    // derive their own visible lists in SceneRenderer.
    (void)useActiveCameraCulling;
#if INFERNUX_FRAME_PROFILE
    using Clock = std::chrono::high_resolution_clock;
    const auto prepareStart = Clock::now();
#endif
    SceneManager &sm = SceneManager::Instance();
    m_activeCamera = sm.GetEditorCameraController().GetCamera();

    const uint64_t currentVersion = sm.GetMeshRendererVersion();
    const uint64_t currentTransformRevision = sm.GetRenderTransformRevision();
    const uint64_t currentContentRevision = sm.GetRenderContentRevision();
    Scene *activeScene = sm.GetActiveScene();
    const uint64_t currentWorldId = activeScene ? activeScene->GetWorldId() : 0;
    RenderWorldFrame &frame = world.BeginFrame(currentWorldId, currentVersion);
    CapturePrimaryView(frame, m_activeCamera);
    const bool fastPath =
        frame.MatchesSource(currentWorldId, currentVersion) && frame.m_proxies.size() == m_sceneSources.size();
    if (fastPath) {
        // Fast path: renderer set unchanged — fuse transform/bounds/draw-call patch.
#if INFERNUX_FRAME_PROFILE
        const auto t0 = Clock::now();
#endif
        if (!(m_allRenderersStatic && currentTransformRevision == frame.m_transformRevision &&
              currentContentRevision == frame.m_contentRevision))
            UpdateCachedRenderableTransforms(frame);
#if INFERNUX_FRAME_PROFILE
        m_profileSnapshot.updateMs += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        m_profileSnapshot.prepareFastCalls += 1.0;
#endif
    } else {
        // Slow path: full rebuild.
#if INFERNUX_FRAME_PROFILE
        const auto t0 = Clock::now();
#endif
        CollectRenderables(frame);
#if INFERNUX_FRAME_PROFILE
        m_profileSnapshot.collectMs += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        m_profileSnapshot.prepareSlowCalls += 1.0;
#endif
    }

    if (!fastPath) {
#if INFERNUX_FRAME_PROFILE
        const auto t0 = Clock::now();
#endif
        SortRenderables(frame);
        RebuildDrawCalls(frame);
#if INFERNUX_FRAME_PROFILE
        m_profileSnapshot.sortMs += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
#endif
    }

#if INFERNUX_FRAME_PROFILE
    m_profileSnapshot.prepareMs += std::chrono::duration<double, std::milli>(Clock::now() - prepareStart).count();
    m_profileSnapshot.prepareCalls += 1.0;
    m_profileSnapshot.renderables += static_cast<double>(frame.m_proxies.size());
    m_profileSnapshot.visible += static_cast<double>(m_visibleCount);
#endif
    frame.m_worldId = currentWorldId;
    frame.m_structuralRevision = currentVersion;
    frame.m_transformRevision = currentTransformRevision;
    frame.m_contentRevision = currentContentRevision;
    world.Publish();
    return m_visibleCount;
#endif
}

size_t SceneRenderExtractor::ExtractCameraFrame(RenderWorldSnapshot &world, Camera *camera)
{
    if (!camera) {
        world.Clear();
        m_sceneSources.clear();
        m_visibleCount = 0;
        return 0;
    }

#if INFERNUX_FRAME_PROFILE
    using Clock = std::chrono::high_resolution_clock;
    const auto prepareStart = Clock::now();
#endif
    m_activeCamera = camera;

    SceneManager &sm = SceneManager::Instance();
    const uint64_t currentVersion = sm.GetMeshRendererVersion();
    const uint64_t currentTransformRevision = sm.GetRenderTransformRevision();
    const uint64_t currentContentRevision = sm.GetRenderContentRevision();
    Scene *activeScene = sm.GetActiveScene();
    const uint64_t currentWorldId = activeScene ? activeScene->GetWorldId() : 0;
    RenderWorldFrame &frame = world.BeginFrame(currentWorldId, currentVersion);
    CapturePrimaryView(frame, m_activeCamera);
    const bool fastPath =
        frame.MatchesSource(currentWorldId, currentVersion) && frame.m_proxies.size() == m_sceneSources.size();
    if (fastPath) {
#if INFERNUX_FRAME_PROFILE
        const auto t0 = Clock::now();
#endif
        if (!(m_allRenderersStatic && currentTransformRevision == frame.m_transformRevision &&
              currentContentRevision == frame.m_contentRevision))
            UpdateCachedRenderableTransforms(frame);
#if INFERNUX_FRAME_PROFILE
        m_profileSnapshot.updateMs += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        m_profileSnapshot.prepareFastCalls += 1.0;
#endif
    } else {
#if INFERNUX_FRAME_PROFILE
        const auto t0 = Clock::now();
#endif
        CollectRenderables(frame);
#if INFERNUX_FRAME_PROFILE
        m_profileSnapshot.collectMs += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        m_profileSnapshot.prepareSlowCalls += 1.0;
#endif
    }

    if (!fastPath) {
#if INFERNUX_FRAME_PROFILE
        const auto t0 = Clock::now();
#endif
        SortRenderables(frame);
        RebuildDrawCalls(frame);
#if INFERNUX_FRAME_PROFILE
        m_profileSnapshot.sortMs += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
#endif
    }

#if INFERNUX_FRAME_PROFILE
    m_profileSnapshot.prepareMs += std::chrono::duration<double, std::milli>(Clock::now() - prepareStart).count();
    m_profileSnapshot.prepareCalls += 1.0;
    m_profileSnapshot.renderables += static_cast<double>(frame.m_proxies.size());
    m_profileSnapshot.visible += static_cast<double>(m_visibleCount);
#endif
    frame.m_worldId = currentWorldId;
    frame.m_structuralRevision = currentVersion;
    frame.m_transformRevision = currentTransformRevision;
    frame.m_contentRevision = currentContentRevision;
    world.Publish();
    return m_visibleCount;
}

void SceneRenderExtractor::SetAspectRatio(float aspect)
{
    if (m_activeCamera) {
        m_activeCamera->SetAspectRatio(aspect);
    }
}

void SceneRenderExtractor::CapturePrimaryView(RenderWorldFrame &frame, Camera *camera)
{
    frame.m_primaryView = {};
    if (!camera)
        return;

    auto &view = frame.m_primaryView;
    view.view = camera->GetViewMatrix();
    view.projection = camera->GetProjectionMatrix();
    view.viewProjection = camera->GetViewProjectionMatrix();
    view.cullingMask = camera->GetCullingMask();
    view.cameraId = camera->GetComponentID();
    view.valid = true;
    const auto cameraToWorld = camera->GetCameraToWorldMatrix();
    view.position = glm::vec3(cameraToWorld[3]);
    view.forward = glm::normalize(glm::vec3(cameraToWorld[2]));
    view.up = glm::normalize(glm::vec3(cameraToWorld[1]));
}

void SceneRenderExtractor::CollectRenderables(RenderWorldFrame &frame)
{
    frame.m_proxies.clear();
    frame.m_drawCalls = {};
    m_sceneSources.clear();
    m_allRenderersStatic = true;

    Scene *activeScene = SceneManager::Instance().GetActiveScene();
    if (!activeScene)
        return;

    // Use the MeshRenderer registry — O(N) over active renderers only,
    // no GetAllObjects() tree walk, no dynamic_cast, no vector allocation.
    const auto &meshRenderers = SceneManager::Instance().GetActiveMeshRenderers();

    frame.m_proxies.reserve(meshRenderers.size());
    m_sceneSources.reserve(meshRenderers.size());

    for (MeshRenderer *renderer : meshRenderers) {
        if (!renderer || !renderer->IsEnabled())
            continue;

        GameObject *obj = renderer->GetGameObject();
        if (!obj || !obj->IsActiveInHierarchy())
            continue;

        if (!obj->IsStatic() || dynamic_cast<SkinnedMeshRenderer *>(renderer) != nullptr)
            m_allRenderersStatic = false;

        uint32_t objectLayerBit = 1u << static_cast<uint32_t>(obj->GetLayer());

        renderer->RefreshProceduralGeometry(obj->GetTransform()->GetWorldMatrix());

        // Check if renderer has inline mesh, asset mesh, or mesh reference
        if (!renderer->HasMeshAsset() && !renderer->HasInlineMesh() && !renderer->GetMesh().IsValid())
            continue;

        RenderProxy renderable;
        renderable.structural.objectId = obj->GetID();
        renderable.structural.layerMask = objectLayerBit;
        renderable.structural.isStatic = obj->IsStatic() && dynamic_cast<SkinnedMeshRenderer *>(renderer) == nullptr;
        renderable.structural.identity = RenderProxyHandle::FromScene(obj->GetHandle(), renderer->GetHandle());
        const glm::mat4 objectWorldMatrix = obj->GetTransform()->GetWorldMatrix();
        renderable.frame.worldMatrix = renderer->ResolveRenderWorldMatrix(objectWorldMatrix);
        renderable.frame.boundsWorldMatrix = renderer->ResolveBoundsWorldMatrix(objectWorldMatrix);
        renderable.structural.renderMaterial = renderer->GetEffectiveMaterial();
        renderable.structural.renderQueue =
            renderable.structural.renderMaterial ? renderable.structural.renderMaterial->GetRenderQueue() : 2000;
        renderable.structural.materialSortKey = reinterpret_cast<uintptr_t>(renderable.structural.renderMaterial.get());

        SceneRenderSource source;
        source.meshRenderer = renderer;
        source.transform = obj->GetTransform();
        source.skinnedRenderer = dynamic_cast<SkinnedMeshRenderer *>(renderer);
        if (renderer->HasInlineMesh()) {
            CaptureOrReferenceInlineMesh(*renderer, source.inlineVertices, source.inlineIndices,
                                         source.inlineMeshOwner);
            source.inlineMeshVersion = renderer->GetInlineMeshVersion();
        }

        // Get world-space bounding box for frustum culling — reuse the world matrix
        glm::vec3 boundsMin, boundsMax;
        renderer->ComputeWorldBounds(renderable.frame.boundsWorldMatrix, boundsMin, boundsMax);
        renderable.frame.worldBounds = AABB(boundsMin, boundsMax);
        // RenderWorld is shared by every camera. Visibility is intentionally
        // not stored here; each camera derives and caches its own renderer list.
        renderable.frame.visible = true;

        frame.m_proxies.push_back(std::move(renderable));
        m_sceneSources.push_back(std::move(source));
    }

    m_visibleCount = frame.m_proxies.size();
}

void SceneRenderExtractor::UpdateCachedRenderableTransforms(RenderWorldFrame &frame)
{
    // Fast path: renderer set unchanged. Refresh transforms, bounds, and
    // cached draw calls in one O(N) pass. Culling is per-camera downstream.
    // Optimization: skip bounds recomputation and heavy draw-call patching
    // for objects whose world transform has not changed since last frame.
    m_visibleCount = 0;

    for (size_t renderableIndex = 0; renderableIndex < frame.m_proxies.size(); ++renderableIndex) {
        auto &renderable = frame.m_proxies[renderableIndex];
        auto &frameData = renderable.frame;
        auto &cache = renderable.cache;
        SceneRenderSource &source = m_sceneSources[renderableIndex];
        MeshRenderer *mr = source.meshRenderer;
        if (!mr)
            continue;
        Transform *transform = source.transform;
        if (!transform)
            continue;

        mr->RefreshProceduralGeometry(transform->GetWorldMatrix());

        const glm::mat4 objectWorldMatrix = transform->GetWorldMatrix();
        const glm::mat4 worldMatrix = mr->ResolveRenderWorldMatrix(objectWorldMatrix);
        const glm::mat4 boundsWorldMatrix = mr->ResolveBoundsWorldMatrix(objectWorldMatrix);

        // World-space resident vertices deliberately render with an identity
        // matrix, while their conservative bounds still follow the object's
        // Transform anchor. Track those two matrices independently so the
        // fast path cannot freeze culling, picking, or editor handles.
        const bool renderTransformChanged = std::memcmp(&worldMatrix, &frameData.worldMatrix, sizeof(glm::mat4)) != 0;
        const bool boundsTransformChanged =
            std::memcmp(&boundsWorldMatrix, &frameData.boundsWorldMatrix, sizeof(glm::mat4)) != 0;
        const bool dynamicSkinBounds = source.skinnedRenderer && source.skinnedRenderer->HasRuntimeSkinnedMesh();
        const bool bufferDirty = mr->ConsumeMeshBufferDirty();
        const size_t cachedDrawCallEnd =
            std::min(cache.drawCallStart + cache.drawCallCount, frame.m_drawCalls.drawCalls.size());
        bool inlinePublicationChanged = false;
        if (mr->HasInlineMesh()) {
            const auto &resident = mr->GetVertexBuffer();
            const uint64_t version = mr->GetInlineMeshVersion();
            inlinePublicationChanged = source.inlineMeshVersion != version;
            for (size_t drawCallIndex = cache.drawCallStart;
                 !inlinePublicationChanged && drawCallIndex < cachedDrawCallEnd; ++drawCallIndex) {
                const DrawCall &draw = frame.m_drawCalls.drawCalls[drawCallIndex];
                inlinePublicationChanged =
                    draw.meshRuntimeVersion != version || draw.meshVertexBuffer.get() != resident.get();
            }
        }
        const bool geometryPublicationChanged = bufferDirty || inlinePublicationChanged;
        if (geometryPublicationChanged && mr->HasInlineMesh()) {
            CaptureOrReferenceInlineMesh(*mr, source.inlineVertices, source.inlineIndices, source.inlineMeshOwner);
            source.inlineMeshVersion = mr->GetInlineMeshVersion();
        }

        // Procedural meshes can move their vertices while the owning Transform
        // remains unchanged (LineRenderer world-space trails are the common
        // case). Their cached bounds must follow the content publication or
        // per-camera frustum culling will keep testing the previous shape.
        if (boundsTransformChanged || geometryPublicationChanged || dynamicSkinBounds) {
            const bool translationOnly =
                boundsTransformChanged && !geometryPublicationChanged &&
                std::memcmp(&boundsWorldMatrix[0], &frameData.boundsWorldMatrix[0], sizeof(glm::vec4) * 3) == 0;
            const glm::vec3 translationDelta = glm::vec3(boundsWorldMatrix[3] - frameData.boundsWorldMatrix[3]);
            frameData.boundsWorldMatrix = boundsWorldMatrix;

            if (translationOnly && !dynamicSkinBounds) {
                frameData.worldBounds.min += translationDelta;
                frameData.worldBounds.max += translationDelta;
            } else {
                glm::vec3 bmin, bmax;
                mr->ComputeWorldBounds(boundsWorldMatrix, bmin, bmax);
                frameData.worldBounds = AABB(bmin, bmax);
            }
        }
        if (renderTransformChanged)
            frameData.worldMatrix = worldMatrix;

        frameData.visible = true;

        if (cache.drawCallCount > 0) {
            const size_t drawCallEnd = cachedDrawCallEnd;
            std::shared_ptr<const std::vector<glm::mat4>> skinBoneMatricesOwner;
            const std::vector<glm::mat4> *skinBoneMatricesPtr = nullptr;
            std::shared_ptr<const std::vector<glm::mat4>> previousSkinBoneMatricesOwner;
            const std::vector<glm::mat4> *previousSkinBoneMatricesPtr = nullptr;
            const bool isSkinnedRenderer = source.skinnedRenderer != nullptr;
            if (auto *skinned = source.skinnedRenderer; skinned && skinned->HasRuntimeSkinnedMesh()) {
                const auto pose = skinned->GetRuntimeSkinPoseSnapshot();
                skinBoneMatricesOwner = pose ? pose->current : nullptr;
                skinBoneMatricesPtr = skinBoneMatricesOwner.get();
                previousSkinBoneMatricesOwner = pose ? pose->previous : nullptr;
                previousSkinBoneMatricesPtr = previousSkinBoneMatricesOwner.get();
            }

            auto patchSkinPalette = [&](DrawCall &dc) {
                if (!isSkinnedRenderer)
                    return;
                dc.skinBoneMatricesOwner = skinBoneMatricesOwner;
                dc.skinBoneMatrices = skinBoneMatricesPtr;
                dc.previousSkinBoneMatricesOwner = previousSkinBoneMatricesOwner;
                dc.previousSkinBoneMatrices = previousSkinBoneMatricesPtr;
            };

            for (size_t drawCallIndex = cache.drawCallStart; drawCallIndex < drawCallEnd; ++drawCallIndex) {
                DrawCall &dc = frame.m_drawCalls.drawCalls[drawCallIndex];
                dc.material = mr->GetEffectiveMaterial(dc.materialSlot);
                dc.parameterBlock = mr->GetParameterBlock(dc.materialSlot);
                // A rebuilt procedural mesh frequently reuses the freed heap
                // block of its previous build, so EnsureObjectBuffers' pointer
                // + size fast path cannot detect the content change on its
                // own. bufferDirty is the authoritative content-change signal
                // and must reach the GPU path exactly like RebuildDrawCalls
                // does, otherwise every-frame trails freeze at the last frame
                // whose allocation happened to move (and flicker between the
                // frozen and the fresh publication as sizes drift).
                dc.forceBufferUpdate = geometryPublicationChanged && mr->HasInlineMesh();
                if (geometryPublicationChanged && mr->HasInlineMesh()) {
                    dc.meshDataOwner = source.inlineMeshOwner;
                    dc.meshVertices = source.inlineVertices;
                    dc.meshIndices = source.inlineIndices;
                    dc.meshVertexBuffer = mr->GetVertexBuffer();
                    dc.meshRuntimeVersion = mr->GetInlineMeshVersion();
                    dc.meshAssetGuid = mr->HasSharedInlineMesh() ? mr->GetSharedInlineMeshGuid() : std::string{};
                    dc.indexStart = 0;
                    dc.indexCount = source.inlineIndices ? static_cast<uint32_t>(source.inlineIndices->size()) : 0;
                    dc.vertexStart = 0;
                }
            }

            if (renderTransformChanged || boundsTransformChanged || geometryPublicationChanged || dynamicSkinBounds) {
                // Full patch: transform/bounds changed or dynamic mesh data
                // needs a GPU re-upload.
                const glm::vec3 &pivot = mr->GetMeshPivotOffset();
                glm::mat4 drawWorldMatrix = worldMatrix;
                if (mr->GetSubmeshIndex() >= 0 && pivot != glm::vec3(0.0f)) {
                    drawWorldMatrix = worldMatrix * glm::translate(glm::mat4(1.0f), pivot);
                }

                bool firstDirty = true;
                for (size_t drawCallIndex = cache.drawCallStart; drawCallIndex < drawCallEnd; ++drawCallIndex) {
                    DrawCall &dc = frame.m_drawCalls.drawCalls[drawCallIndex];
                    dc.worldMatrix = drawWorldMatrix;
                    dc.worldBounds = frameData.worldBounds;
                    dc.frustumVisible = true;
                    dc.forceBufferUpdate = firstDirty ? geometryPublicationChanged : false;
                    patchSkinPalette(dc);
                    firstDirty = false;
                }
            } else if (isSkinnedRenderer) {
                for (size_t drawCallIndex = cache.drawCallStart; drawCallIndex < drawCallEnd; ++drawCallIndex) {
                    patchSkinPalette(frame.m_drawCalls.drawCalls[drawCallIndex]);
                }
            }
            // else: unchanged static draw calls are already correct.
        }

        if (frameData.visible) {
            ++m_visibleCount;
        }
    }
}

void SceneRenderExtractor::SortRenderables(RenderWorldFrame &frame)
{
    // Sort by material render queue (for proper render order)
    // - Lower queue = render first (opaque before transparent)
    // - Within same queue, sort by material pointer (minimize state changes)

    std::vector<size_t> order(frame.m_proxies.size());
    for (size_t i = 0; i < order.size(); ++i)
        order[i] = i;
    std::sort(order.begin(), order.end(), [&frame](size_t lhs, size_t rhs) {
        const auto &a = frame.m_proxies[lhs].structural;
        const auto &b = frame.m_proxies[rhs].structural;
        if (a.renderQueue != b.renderQueue)
            return a.renderQueue < b.renderQueue;
        return a.materialSortKey < b.materialSortKey;
    });

    std::vector<RenderProxy> sortedProxies;
    std::vector<SceneRenderSource> sortedSources;
    sortedProxies.reserve(order.size());
    sortedSources.reserve(order.size());
    for (size_t index : order) {
        sortedProxies.push_back(std::move(frame.m_proxies[index]));
        sortedSources.push_back(std::move(m_sceneSources[index]));
    }
    frame.m_proxies = std::move(sortedProxies);
    m_sceneSources = std::move(sortedSources);
}

// ============================================================================
// Shared draw-call emission (eliminates duplication between the two Build paths)
// ============================================================================
void SceneRenderExtractor::EmitDrawCallsForRenderable(DrawCallResult &result, const RenderProxy &renderable,
                                                      SceneRenderSource &source, bool visible, bool bufferDirty) const
{
    const auto &structural = renderable.structural;
    const auto &frame = renderable.frame;
    MeshRenderer *renderer = source.meshRenderer;
    if (!renderer)
        return;

    if (renderer->HasMeshAsset()) {
        const auto &assetRef = renderer->GetMeshAssetRef();
        auto &registry = AssetRegistry::Instance();
        if (!registry.IsLoaded(assetRef.GetGuid()) ||
            assetRef.GetCachedVersion() != registry.GetAssetVersion(assetRef.GetGuid()))
            return;
        auto meshPtr = assetRef.Get();
        if (!meshPtr)
            return;
        const auto stampAssetIdentity = [&assetRef](DrawCall &drawCall) {
            drawCall.meshAssetGuid = assetRef.GetGuid();
            drawCall.meshRuntimeVersion = assetRef.GetCachedVersion();
        };
        const auto geometry = meshPtr->GetGeometrySnapshot();
        const std::vector<Vertex> *objVerticesPtr = &geometry->vertices;
        const std::vector<uint32_t> *objIndicesPtr = &geometry->indices;
        const std::vector<SubMesh> *subMeshesPtr = &geometry->subMeshes;
        std::shared_ptr<const void> meshDataOwner = geometry;
        std::shared_ptr<const std::vector<glm::mat4>> skinBoneMatricesOwner;
        const std::vector<glm::mat4> *skinBoneMatricesPtr = nullptr;
        std::shared_ptr<const std::vector<glm::mat4>> previousSkinBoneMatricesOwner;
        const std::vector<glm::mat4> *previousSkinBoneMatricesPtr = nullptr;
        if (auto *skinned = source.skinnedRenderer; skinned && skinned->HasRuntimeSkinnedMesh()) {
            objVerticesPtr = &skinned->GetRuntimeSkinnedVertices();
            objIndicesPtr = &skinned->GetRuntimeSkinnedIndices();
            subMeshesPtr = &skinned->GetRuntimeSkinnedSubMeshes();
            meshDataOwner = skinned->GetRuntimeModelSnapshot();
            const auto pose = skinned->GetRuntimeSkinPoseSnapshot();
            skinBoneMatricesOwner = pose ? pose->current : nullptr;
            skinBoneMatricesPtr = skinBoneMatricesOwner.get();
            previousSkinBoneMatricesOwner = pose ? pose->previous : nullptr;
            previousSkinBoneMatricesPtr = previousSkinBoneMatricesOwner.get();
        }
        const auto &objVertices = *objVerticesPtr;
        const auto &objIndices = *objIndicesPtr;
        if (objVertices.empty() || objIndices.empty())
            return;

        const glm::mat4 &worldMatrix = frame.worldMatrix;

        uint32_t subMeshCount = static_cast<uint32_t>(subMeshesPtr ? subMeshesPtr->size() : 0);
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
            dc.parameterBlock = renderer->GetParameterBlock(0);
            dc.materialSlot = 0;
            dc.objectId = structural.objectId;
            dc.layerMask = structural.layerMask;
            dc.identity = structural.identity.MakeDrawIdentity();
            dc.frustumVisible = visible;
            dc.castsShadows = renderer->CastsShadows();
            dc.isStatic = structural.isStatic;
            dc.worldBounds = frame.worldBounds;
            dc.meshVertices = &objVertices;
            dc.meshIndices = &objIndices;
            dc.meshDataOwner = meshDataOwner;
            stampAssetIdentity(dc);
            dc.skinBoneMatricesOwner = skinBoneMatricesOwner;
            dc.skinBoneMatrices = skinBoneMatricesPtr;
            dc.previousSkinBoneMatricesOwner = previousSkinBoneMatricesOwner;
            dc.previousSkinBoneMatrices = previousSkinBoneMatricesPtr;
            dc.forceBufferUpdate = bufferDirty;
            result.drawCalls.push_back(dc);
        } else if (submeshFilter >= 0 && static_cast<uint32_t>(submeshFilter) < subMeshCount) {
            // Single submesh mode — render only the specified submesh
            const auto &sub = (*subMeshesPtr)[static_cast<uint32_t>(submeshFilter)];
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
            dc.parameterBlock = renderer->GetParameterBlock(0);
            dc.materialSlot = 0;
            dc.objectId = structural.objectId;
            dc.layerMask = structural.layerMask;
            dc.identity = structural.identity.MakeDrawIdentity(static_cast<uint32_t>(submeshFilter));
            dc.frustumVisible = visible;
            dc.castsShadows = renderer->CastsShadows();
            dc.isStatic = structural.isStatic;
            dc.worldBounds = frame.worldBounds;
            dc.meshVertices = &objVertices;
            dc.meshIndices = &objIndices;
            dc.meshDataOwner = meshDataOwner;
            stampAssetIdentity(dc);
            dc.skinBoneMatricesOwner = skinBoneMatricesOwner;
            dc.skinBoneMatrices = skinBoneMatricesPtr;
            dc.previousSkinBoneMatricesOwner = previousSkinBoneMatricesOwner;
            dc.previousSkinBoneMatrices = previousSkinBoneMatricesPtr;
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
                    const auto &s = (*subMeshesPtr)[si];
                    if (static_cast<int32_t>(s.nodeGroup) != nodeGroup)
                        continue;
                    if (s.materialSlot < SLOT_REMAP_CAP && slotRemap[s.materialSlot] == 0xFFFFFFFF)
                        slotRemap[s.materialSlot] = nextSlotIdx++;
                }
            }
            bool firstDirty = true;
            for (uint32_t si = 0; si < subMeshCount; ++si) {
                const auto &sub = (*subMeshesPtr)[si];
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
                dc.parameterBlock = renderer->GetParameterBlock(matSlot);
                dc.materialSlot = matSlot;
                dc.objectId = structural.objectId;
                dc.layerMask = structural.layerMask;
                dc.identity = structural.identity.MakeDrawIdentity(si);
                dc.frustumVisible = visible;
                dc.castsShadows = renderer->CastsShadows();
                dc.isStatic = structural.isStatic;
                dc.worldBounds = frame.worldBounds;
                dc.meshVertices = &objVertices;
                dc.meshIndices = &objIndices;
                dc.meshDataOwner = meshDataOwner;
                stampAssetIdentity(dc);
                dc.skinBoneMatricesOwner = skinBoneMatricesOwner;
                dc.skinBoneMatrices = skinBoneMatricesPtr;
                dc.previousSkinBoneMatricesOwner = previousSkinBoneMatricesOwner;
                dc.previousSkinBoneMatrices = previousSkinBoneMatricesPtr;
                dc.forceBufferUpdate = firstDirty ? bufferDirty : false;
                firstDirty = false;
                result.drawCalls.push_back(dc);
            }
        }
    } else if (renderer->HasInlineMesh()) {
        if (!source.inlineVertices || !source.inlineIndices ||
            source.inlineMeshVersion != renderer->GetInlineMeshVersion()) {
            CaptureOrReferenceInlineMesh(*renderer, source.inlineVertices, source.inlineIndices,
                                         source.inlineMeshOwner);
            source.inlineMeshVersion = renderer->GetInlineMeshVersion();
        }
        const auto &objVertices = *source.inlineVertices;
        const auto &objIndices = *source.inlineIndices;
        if (objVertices.empty() || objIndices.empty())
            return;

        const glm::mat4 &worldMatrix = frame.worldMatrix;
        DrawCall dc;
        dc.indexStart = 0;
        dc.indexCount = static_cast<uint32_t>(objIndices.size());
        dc.vertexStart = 0;
        dc.worldMatrix = worldMatrix;
        dc.material = renderer->GetEffectiveMaterial(0);
        dc.parameterBlock = renderer->GetParameterBlock(0);
        dc.materialSlot = 0;
        dc.objectId = structural.objectId;
        dc.layerMask = structural.layerMask;
        dc.identity = structural.identity.MakeDrawIdentity();
        dc.frustumVisible = visible;
        dc.castsShadows = renderer->CastsShadows();
        dc.isStatic = structural.isStatic;
        dc.worldBounds = frame.worldBounds;
        dc.meshVertices = &objVertices;
        dc.meshIndices = &objIndices;
        dc.meshDataOwner = source.inlineMeshOwner;
        dc.meshVertexBuffer = renderer->GetVertexBuffer();
        dc.meshRuntimeVersion = renderer->GetInlineMeshVersion();
        if (renderer->HasSharedInlineMesh()) {
            dc.meshAssetGuid = renderer->GetSharedInlineMeshGuid();
        }
        dc.forceBufferUpdate = bufferDirty;
        result.drawCalls.push_back(dc);
    }
}

void SceneRenderExtractor::RebuildDrawCalls(RenderWorldFrame &frame)
{
    DrawCallResult result;
    result.drawCalls.reserve(frame.m_drawCalls.drawCalls.empty() ? frame.m_proxies.size()
                                                                 : frame.m_drawCalls.drawCalls.size());

    for (size_t index = 0; index < frame.m_proxies.size(); ++index) {
        RenderProxy &renderable = frame.m_proxies[index];
        SceneRenderSource &source = m_sceneSources[index];
        MeshRenderer *renderer = source.meshRenderer;
        if (!renderer)
            continue;

        bool bufferDirty = renderer->ConsumeMeshBufferDirty();
        renderable.cache.drawCallStart = result.drawCalls.size();
        EmitDrawCallsForRenderable(result, renderable, source, renderable.frame.visible, bufferDirty);
        renderable.cache.drawCallCount = result.drawCalls.size() - renderable.cache.drawCallStart;
    }

    frame.m_drawCalls = std::move(result);
}

} // namespace infernux
