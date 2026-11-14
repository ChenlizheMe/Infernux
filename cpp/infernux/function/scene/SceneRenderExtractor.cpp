#include "SceneRenderExtractor.h"
#include "SceneManager.h"
#include "SkinnedMeshRenderer.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <function/renderer/rhi/RhiComputeBuffer.h>
#include <function/renderer/rhi/RhiRenderTexture.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/scene/Light.h>
#include <function/scene/LightingData.h>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <stdexcept>

namespace infernux
{

namespace
{
struct InlineMeshSnapshot
{
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

struct CpuRenderPose
{
    glm::mat4 render;
    glm::mat4 bounds;
};

std::optional<CpuRenderPose> ReadCpuRenderPose(MeshRenderer &renderer, Transform &transform, bool devicePose)
{
    if (devicePose && !renderer.RequiresCpuPoseForGeometry())
        return std::nullopt;
    const auto &world = transform.GetWorldMatrix();
    return CpuRenderPose{renderer.ResolveRenderWorldMatrix(world), renderer.ResolveBoundsWorldMatrix(world)};
}

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

RenderPoseBinding SceneRenderExtractor::CaptureDevicePose(const MeshRenderer &renderer)
{
    auto binding = renderer.GetDevicePose();
    if (!binding.value)
        return binding;
    const auto *source = renderer.GetPoseBuffer().get();
    const auto [entry, inserted] = m_framePoseValues.emplace(source, binding.value);
    binding.value = entry->second;
    if (inserted) {
        const auto previous = m_previousPoseValues.find(source);
        if (previous == m_previousPoseValues.end() || previous->second != binding.value)
            ++m_poseRevision;
    }
    return binding;
}

size_t SceneRenderExtractor::ExtractEditorFrame(RenderWorldSnapshot &world, bool useActiveCameraCulling)
{
    (void)useActiveCameraCulling;
#if defined(INFERNUX_RUNTIME_MINIMAL_HOST)
    world.Clear();
    m_sceneSources.clear();
    m_activeCamera = nullptr;
    m_visibleCount = 0;
    return 0;
#else
    SceneManager &manager = SceneManager::Instance();
    auto &authoring = manager.AuthoringWorld();
    const WorldExecutionScope ownerScope(&authoring);
    // Scene View reads the authoring World's registries and skeleton poses,
    // independently of the runtime World that most recently executed.
    manager.PublishSkeletonPoses();
    return ExtractSourceFrame(world, manager.GetEditorCameraController().GetCamera(), authoring.id, 0, 0, true);
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
    auto *object = camera->GetGameObject();
    auto *scene = object ? object->GetScene() : nullptr;
    auto *owner = scene ? scene->GetRuntimeWorld() : nullptr;
    if (!owner)
        throw std::logic_error("Scene camera preview requires its camera's scene World");
    const WorldExecutionScope ownerScope(owner);
    SceneManager &manager = SceneManager::Instance();
    manager.PublishSkeletonPoses();
    return ExtractSourceFrame(world, camera, owner->id, 0, 0, false);
}

size_t SceneRenderExtractor::ExtractRuntimeFrame(RenderWorldSnapshot &publication, RuntimeWorld &world)
{
    WorldExecutionScope scope(&world);
    Scene *scene = world.scenes.active;
    Camera *editorCamera = nullptr;
#if !defined(INFERNUX_RUNTIME_MINIMAL_HOST)
    editorCamera = SceneManager::Instance().GetEditorCameraController().GetCamera();
#endif
    Camera *primary = scene ? scene->FindGameCamera(editorCamera) : nullptr;
    // The caller has already completed skeleton/transform publication. Do not
    // perform another simulation update or borrow a later Host's live state.
    return ExtractSourceFrame(publication, primary, world.id, world.simulation.frameTime.frameCount,
                              world.simulation.simulationEpoch, false);
}

size_t SceneRenderExtractor::ExtractSourceFrame(RenderWorldSnapshot &publication, Camera *primary, uint64_t worldId,
                                                uint64_t completedWorldFrame, uint64_t simulationEpoch,
                                                bool includeEditorView)
{
#if INFERNUX_FRAME_PROFILE
    using Clock = std::chrono::high_resolution_clock;
    const auto prepareStart = Clock::now();
#endif
    m_activeCamera = primary;
    SceneManager &manager = SceneManager::Instance();
    // Procedural rebuilds and material defaults can change source revisions.
    // Finish author preparation before selecting structural/content caches.
    std::unordered_set<const InxMaterial *> prepared;
    for (auto *renderer : manager.GetActiveMeshRenderers()) {
        if (!renderer || !renderer->IsEnabled())
            continue;
        auto *object = renderer->GetGameObject();
        if (!object || !object->IsActiveInHierarchy())
            continue;
        if (!renderer->GetPoseBuffer() || renderer->RequiresCpuPoseForGeometry())
            renderer->RefreshProceduralGeometry(object->GetTransform()->GetWorldMatrix());
        for (uint32_t slot = 0, count = std::max(1u, renderer->GetMaterialCount()); slot < count; ++slot) {
            const auto material = renderer->GetEffectiveMaterial(slot);
            if (m_prepareMaterial && material && prepared.insert(material.get()).second)
                m_prepareMaterial(material);
        }
    }
    const uint64_t sourceStructure = manager.GetMeshRendererVersion();
    const uint64_t sourceRouting = InxMaterial::GetRoutingPublicationRevision();
    if (m_sourceWorldId != worldId || m_sourceStructureRevision != sourceStructure ||
        m_sourceRoutingRevision != sourceRouting) {
        ++m_structurePublicationRevision;
        m_sourceWorldId = worldId;
        m_sourceStructureRevision = sourceStructure;
        m_sourceRoutingRevision = sourceRouting;
    }
    const uint64_t currentVersion = m_structurePublicationRevision;
    const uint64_t transformRevision = manager.GetRenderTransformRevision();
    const uint64_t contentRevision = manager.GetRenderContentRevision();
    RenderWorldFrame &frame = publication.BeginFrame(worldId, currentVersion);
    CaptureViews(frame, primary, includeEditorView);
    frame.m_completedWorldFrame = completedWorldFrame;
    frame.m_simulationEpoch = simulationEpoch;
    for (auto &view : frame.m_views) {
        view.worldId = worldId;
        view.completedWorldFrame = completedWorldFrame;
        view.simulationEpoch = simulationEpoch;
    }
    if (const auto *primaryView = primary ? frame.FindView(primary->GetComponentID()) : nullptr)
        frame.m_primaryView = *primaryView;
    m_framePoseValues.clear();
    const bool fastPath =
        frame.MatchesSource(worldId, currentVersion) && frame.m_proxies.size() == m_sceneSources.size();
    if (fastPath) {
        if (!(m_allRenderersStatic && transformRevision == frame.m_transformRevision &&
              contentRevision == frame.m_sourceContentRevision))
            UpdateCachedRenderableTransforms(frame);
    } else {
        CollectRenderables(frame);
        SortRenderables(frame);
        RebuildDrawCalls(frame);
    }
    const bool meshPublicationsChanged = CaptureMeshPublications(frame);
    CaptureMaterialPublications(frame);
    CaptureLighting(frame);
    if (m_sourceContentRevision != contentRevision || m_materialPublicationsChanged || meshPublicationsChanged) {
        ++m_contentPublicationRevision;
        m_sourceContentRevision = contentRevision;
    }
    frame.m_worldId = worldId;
    frame.m_structuralRevision = currentVersion;
    frame.m_transformRevision = transformRevision;
    frame.m_contentRevision = m_contentPublicationRevision;
    frame.m_sourceContentRevision = contentRevision;
    frame.m_poseRevision = m_poseRevision;
    m_previousPoseValues.swap(m_framePoseValues);
#if INFERNUX_FRAME_PROFILE
    m_profileSnapshot.prepareMs += std::chrono::duration<double, std::milli>(Clock::now() - prepareStart).count();
    m_profileSnapshot.prepareCalls += 1.0;
    m_profileSnapshot.prepareFastCalls += fastPath ? 1.0 : 0.0;
    m_profileSnapshot.prepareSlowCalls += fastPath ? 0.0 : 1.0;
    m_profileSnapshot.renderables += static_cast<double>(frame.m_proxies.size());
    m_profileSnapshot.visible += static_cast<double>(m_visibleCount);
#endif
    publication.Publish();
    return m_visibleCount;
}
void SceneRenderExtractor::SetAspectRatio(float aspect)
{
    if (m_activeCamera) {
        m_activeCamera->UpdateAutomaticAspectRatio(aspect);
    }
}

RenderViewData SceneRenderExtractor::CaptureCameraView(const Camera &camera)
{
    RenderViewData view;
    view.view = camera.GetViewMatrix();
    view.projection = camera.GetProjectionMatrix();
    view.viewProjection = view.projection * view.view;
    view.cullingMask = camera.GetCullingMask();
    view.cameraId = camera.GetComponentID();
    if (const auto *object = camera.GetGameObject()) {
        view.cameraObjectId = object->GetID();
        view.name = object->GetName();
        if (const auto *scene = object->GetScene()) {
            view.sceneWorldId = scene->GetWorldId();
            view.sceneName = scene->GetName();
        }
    }
    view.temporalHistoryRevision = camera.GetTemporalHistoryRevision();
    view.nearClip = camera.GetNearClip();
    view.farClip = camera.GetFarClip();
    view.depth = camera.GetDepth();
    view.rect = camera.GetRect();
    view.pixelWidth = camera.GetPixelWidth();
    view.pixelHeight = camera.GetPixelHeight();
    view.clearFlags = camera.GetClearFlags();
    view.backgroundColor = camera.GetBackgroundColor();
    view.dithering = camera.GetDithering();
    view.stopNaNs = camera.GetStopNaNs();
    view.invertCulling = camera.GetInvertCulling();
    view.renderToTexture = camera.GetRenderToTexture();
    view.renderOnDemand = camera.GetRenderOnDemand();
    view.wantsMemoryOutput = camera.WantsMemoryOutput();
    view.memoryOutputSize = camera.GetMemoryOutputSize();
    view.hasAuthoredTargetTexture = camera.HasTargetTexture();
    view.outputTextureGuid = camera.GetTargetTextureGuid();
    view.outputTexture = camera.GetOutputTexture();
    view.outputGeneration = view.outputTexture ? view.outputTexture->Acquire() : nullptr;
    view.valid = true;
    const auto cameraToWorld = camera.GetCameraToWorldMatrix();
    view.position = glm::vec3(cameraToWorld[3]);
    view.forward = glm::normalize(glm::vec3(cameraToWorld[2]));
    view.up = glm::normalize(glm::vec3(cameraToWorld[1]));
    return view;
}

void SceneRenderExtractor::CapturePrimaryView(RenderWorldFrame &frame, Camera *camera)
{
    frame.m_primaryView = camera ? CaptureCameraView(*camera) : RenderViewData{};
}

void SceneRenderExtractor::CaptureViews(RenderWorldFrame &frame, Camera *primary, bool includeEditorView)
{
    frame.m_views.clear();
    frame.m_viewIndices.clear();
    CapturePrimaryView(frame, primary);
    Camera *editorCamera = nullptr;
#if !defined(INFERNUX_RUNTIME_MINIMAL_HOST)
    editorCamera = SceneManager::Instance().GetEditorCameraController().GetCamera();
#endif
    Scene *scene = SceneManager::Instance().GetActiveScene();
    const auto *gamePrimary = scene ? scene->FindGameCamera(editorCamera) : nullptr;
    const auto append = [&](Camera &camera, bool editor) {
        RenderViewData view = CaptureCameraView(camera);
        view.editorView = editor;
        view.isMainCamera = !editor && &camera == gamePrimary;
        frame.m_viewIndices.emplace(view.cameraId, frame.m_views.size());
        frame.m_views.push_back(std::move(view));
    };
    if (scene) {
        // Includes all loaded/persistent Scene cameras, deterministically
        // ordered by depth + component ID. Output cameras remain continuous.
        for (Camera *camera : scene->GetActiveGameCameras(editorCamera))
            append(*camera, false);
    }
    if (includeEditorView && editorCamera)
        append(*editorCamera, true);
    if (primary && !frame.FindView(primary->GetComponentID()))
        append(*primary, primary == editorCamera);
    if (const auto *captured = primary ? frame.FindView(primary->GetComponentID()) : nullptr)
        frame.m_primaryView = *captured;
}
void SceneRenderExtractor::CaptureLighting(RenderWorldFrame &frame)
{
    auto next = std::make_shared<RenderLightingSource>();
    SceneManager &manager = SceneManager::Instance();
    next->lights = CaptureRenderLights();
    Scene *scene = manager.GetActiveScene();
    const SceneEnvironmentSettings environment = scene ? scene->GetEnvironment() : SceneEnvironmentSettings{};
    auto sourceSky =
        scene ? scene->ResolveSkyboxMaterial() : AssetRegistry::Instance().GetBuiltinMaterial("SkyboxProcedural");
    if (sourceSky && m_prepareMaterial)
        m_prepareMaterial(sourceSky);
    auto capturedSky = sourceSky ? sourceSky->CaptureRenderPublication() : nullptr;
    const auto baseSky = capturedSky;
    if (environment.skyboxMaterialGuid.empty() && sourceSky) {
        const auto previous = m_lighting ? m_lighting->skyboxMaterial : nullptr;
        const auto matchesColor = [&](const char *name, const glm::vec3 &color) {
            const auto *property = previous ? previous->GetProperty(name) : nullptr;
            return property && std::holds_alternative<glm::vec4>(property->value) &&
                   std::get<glm::vec4>(property->value) == glm::vec4(color, 1.0f);
        };
        const auto *exposure = previous ? previous->GetProperty("exposure") : nullptr;
        const bool unchanged = m_baseSkybox == capturedSky && matchesColor("sky_top_color", environment.skyTopColor) &&
                               matchesColor("sky_horizon_color", environment.skyHorizonColor) &&
                               matchesColor("ground_color", environment.skyGroundColor) && exposure &&
                               std::holds_alternative<float>(exposure->value) &&
                               std::get<float>(exposure->value) == environment.skyExposure;
        if (unchanged) {
            capturedSky = previous;
        } else {
            // A private authoring copy prepares this environment override;
            // published render values are then sealed through the same cache.
            auto authoredSky = sourceSky->Clone();
            authoredSky->SetColor("sky_top_color", glm::vec4(environment.skyTopColor, 1.0f));
            authoredSky->SetColor("sky_horizon_color", glm::vec4(environment.skyHorizonColor, 1.0f));
            authoredSky->SetColor("ground_color", glm::vec4(environment.skyGroundColor, 1.0f));
            authoredSky->SetFloat("exposure", environment.skyExposure);
            capturedSky = authoredSky->CaptureRenderPublication();
        }
        m_baseSkybox = baseSky;
    } else {
        m_baseSkybox.reset();
    }
    next->skyboxMaterial = capturedSky;
    SceneLightCollector ambient;
    ambient.Clear();
    using AmbientSource = SceneEnvironmentSettings::AmbientSource;
    if (static_cast<AmbientSource>(environment.ambientSource) == AmbientSource::Color) {
        ambient.SetAmbientColor(environment.ambientColor, environment.ambientIntensity);
    } else if (static_cast<AmbientSource>(environment.ambientSource) == AmbientSource::Gradient) {
        ambient.SetAmbientGradient(environment.ambientSkyColor, environment.ambientEquatorColor,
                                   environment.ambientGroundColor, environment.ambientIntensity);
    } else {
        const auto *top = capturedSky ? capturedSky->GetProperty("sky_top_color") : nullptr;
        const auto *horizon = capturedSky ? capturedSky->GetProperty("sky_horizon_color") : nullptr;
        const auto *ground = capturedSky ? capturedSky->GetProperty("ground_color") : nullptr;
        const auto *exposure = capturedSky ? capturedSky->GetProperty("exposure") : nullptr;
        if (top && ground && std::holds_alternative<glm::vec4>(top->value) &&
            std::holds_alternative<glm::vec4>(ground->value)) {
            const glm::vec3 topColor(std::get<glm::vec4>(top->value));
            const glm::vec3 groundColor(std::get<glm::vec4>(ground->value));
            const glm::vec3 horizonColor = horizon && std::holds_alternative<glm::vec4>(horizon->value)
                                               ? glm::vec3(std::get<glm::vec4>(horizon->value))
                                               : glm::mix(groundColor, topColor, 0.5f);
            const float scale =
                exposure && std::holds_alternative<float>(exposure->value) ? std::get<float>(exposure->value) : 1.0f;
            ambient.SetAmbientGradient(topColor, horizonColor, groundColor, scale * environment.ambientIntensity);
        } else {
            ambient.SetAmbientGradient(glm::vec3(0.5f), glm::vec3(0.4f), glm::vec3(0.25f),
                                       environment.ambientIntensity);
        }
    }
    const auto &values = ambient.GetLightingUBO();
    next->ambientSky = values.ambientSkyColor;
    next->ambientEquator = values.ambientEquatorColor;
    next->ambientGround = values.ambientGroundColor;
    if (!m_lighting || !next->SameValues(*m_lighting)) {
        next->generation = ++m_lightingGeneration;
        m_lighting = std::move(next);
    }
    frame.m_lighting = m_lighting;
}
bool SceneRenderExtractor::CaptureMeshPublications(RenderWorldFrame &frame)
{
    bool changed = false;
    for (size_t index = 0; index < frame.m_proxies.size(); ++index) {
        auto &source = m_sceneSources[index];
        const auto &live = source.meshRenderer->GetVertexBuffer();
        auto &value = source.vertexPublication;
        const auto sameValue = [&] {
            if (!live || !value)
                return !live && !value;
            const auto &a = live->GetDesc();
            const auto &b = value->GetDesc();
            return live->GetAllocation() == value->GetAllocation() && a.elementCount == b.elementCount &&
                   a.scalarType == b.scalarType && a.lanes == b.lanes;
        };
        if (!sameValue()) {
            value = live ? live->Freeze() : nullptr;
            changed = true;
        }
        const auto &range = frame.m_proxies[index].cache;
        const auto end = range.drawCallStart + range.drawCallCount;
        for (size_t draw = range.drawCallStart; draw < end; ++draw)
            frame.m_drawCalls.drawCalls[draw].meshVertexBuffer = value;
    }
    return changed;
}

void SceneRenderExtractor::CaptureMaterialPublications(RenderWorldFrame &frame)
{
    std::unordered_map<const InxMaterial *, std::weak_ptr<InxMaterial>> materials;
    std::unordered_map<const RendererParameterBlock *, ParameterPublication> parameters;
    m_materialPublicationsChanged = false;
    const auto captureMaterial = [&](const std::shared_ptr<InxMaterial> &source) {
        if (!source)
            return std::shared_ptr<InxMaterial>{};
        const auto sameFrame = materials.find(source.get());
        if (sameFrame != materials.end())
            return sameFrame->second.lock();
        auto value = source->CaptureRenderPublication();
        const auto previous = m_lastMaterialPublications.find(source.get());
        if (previous == m_lastMaterialPublications.end() || previous->second.lock() != value)
            m_materialPublicationsChanged = true;
        materials[source.get()] = value;
        return value;
    };
    const auto captureParameters = [&](const std::shared_ptr<const RendererParameterBlock> &source) {
        if (!source || source->buffers.empty())
            return source;
        if (const auto sameFrame = parameters.find(source.get()); sameFrame != parameters.end())
            return sameFrame->second.frozen;
        const auto previous = m_parameterPublications.find(source.get());
        auto frozen = previous == m_parameterPublications.end() ? nullptr : previous->second.frozen;
        bool unchanged = frozen && previous->second.source.lock() == source;
        for (const auto &[name, buffer] : source->buffers) {
            if (!unchanged)
                break;
            const auto found = frozen->buffers.find(name);
            unchanged = found != frozen->buffers.end() && bool(found->second) == bool(buffer) &&
                        (!buffer || found->second->GetAllocation() == buffer->GetAllocation());
        }
        if (!unchanged) {
            auto next = std::make_shared<RendererParameterBlock>(*source);
            for (auto &[name, buffer] : next->buffers)
                if (buffer && !buffer->GetAllocation()->IsGraphWritable())
                    buffer = buffer->Freeze();
            frozen = std::move(next);
            m_materialPublicationsChanged = true;
        }
        parameters.emplace(source.get(), ParameterPublication{source, frozen});
        return frozen;
    };
    for (size_t index = 0; index < frame.m_proxies.size(); ++index) {
        auto &proxy = frame.m_proxies[index];
        MeshRenderer *renderer = m_sceneSources[index].meshRenderer;
        proxy.structural.renderMaterial = captureMaterial(renderer->GetEffectiveMaterial());
        proxy.structural.renderQueue =
            proxy.structural.renderMaterial ? proxy.structural.renderMaterial->GetRenderQueue() : 2000;
        proxy.structural.materialSortKey = reinterpret_cast<uintptr_t>(proxy.structural.renderMaterial.get());
        const size_t end =
            std::min(proxy.cache.drawCallStart + proxy.cache.drawCallCount, frame.m_drawCalls.drawCalls.size());
        for (size_t drawIndex = proxy.cache.drawCallStart; drawIndex < end; ++drawIndex) {
            auto &draw = frame.m_drawCalls.drawCalls[drawIndex];
            draw.material = captureMaterial(renderer->GetEffectiveMaterial(draw.materialSlot));
            draw.parameterBlock = captureParameters(renderer->GetParameterBlock(draw.materialSlot));
        }
    }
    m_lastMaterialPublications.swap(materials);
    m_parameterPublications.swap(parameters);
}
void SceneRenderExtractor::CollectRenderables(RenderWorldFrame &frame)
{
    std::unordered_map<RenderDrawIdentity, std::shared_ptr<rhi::ComputeBuffer>, RenderDrawIdentityHash> vertexValues;
    for (auto &source : m_sceneSources)
        if (source.vertexPublication)
            vertexValues.emplace(source.identity, std::move(source.vertexPublication));
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

        if (!obj->IsStatic() || renderer->GetPoseBuffer() || dynamic_cast<SkinnedMeshRenderer *>(renderer) != nullptr)
            m_allRenderersStatic = false;

        uint32_t objectLayerBit = 1u << static_cast<uint32_t>(obj->GetLayer());

        RenderProxy renderable;
        renderable.frame.devicePose = CaptureDevicePose(*renderer);
        const auto cpuPose = ReadCpuRenderPose(*renderer, *obj->GetTransform(), renderable.frame.devicePose.HasValue());

        // Check if renderer has inline mesh, asset mesh, or mesh reference
        if (!renderer->HasMeshAsset() && !renderer->HasInlineMesh() && !renderer->GetMesh().IsValid())
            continue;

        if (auto *skin = dynamic_cast<SkinnedMeshRenderer *>(renderer); skin && !skin->HasRuntimeSkinnedMesh())
            continue;
        renderable.structural.objectId = obj->GetID();
        renderable.structural.layerMask = objectLayerBit;
        renderable.structural.isStatic =
            obj->IsStatic() && !renderer->GetPoseBuffer() && dynamic_cast<SkinnedMeshRenderer *>(renderer) == nullptr;
        renderable.structural.identity = RenderProxyHandle::FromScene(obj->GetHandle(), renderer->GetHandle());
        if (cpuPose) {
            renderable.frame.worldMatrix = cpuPose->render;
            renderable.frame.boundsWorldMatrix = cpuPose->bounds;
        }
        renderable.structural.renderMaterial = renderer->GetEffectiveMaterial();
        renderable.structural.renderQueue =
            renderable.structural.renderMaterial ? renderable.structural.renderMaterial->GetRenderQueue() : 2000;
        renderable.structural.materialSortKey = reinterpret_cast<uintptr_t>(renderable.structural.renderMaterial.get());

        SceneRenderSource source;
        source.identity = renderable.structural.identity.MakeDrawIdentity();
        if (const auto previous = vertexValues.find(source.identity); previous != vertexValues.end())
            source.vertexPublication = std::move(previous->second);
        source.meshRenderer = renderer;
        source.transform = obj->GetTransform();
        source.skinnedRenderer = dynamic_cast<SkinnedMeshRenderer *>(renderer);
        if (renderer->HasInlineMesh()) {
            CaptureOrReferenceInlineMesh(*renderer, source.inlineVertices, source.inlineIndices,
                                         source.inlineMeshOwner);
            source.inlineMeshVersion = renderer->GetInlineMeshVersion();
        }

        // Get world-space bounding box for frustum culling — reuse the world matrix
        if (cpuPose) {
            glm::vec3 boundsMin, boundsMax;
            renderer->ComputeWorldBounds(cpuPose->bounds, boundsMin, boundsMax);
            renderable.frame.worldBounds = AABB(boundsMin, boundsMax);
        }
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
    m_allRenderersStatic = true;

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

        const bool hadDevicePose = frameData.devicePose.HasValue();
        frameData.devicePose = CaptureDevicePose(*mr);
        const bool poseAuthorityChanged = hadDevicePose != frameData.devicePose.HasValue();
        renderable.structural.isStatic =
            mr->GetGameObject()->IsStatic() && !source.skinnedRenderer && !frameData.devicePose.HasValue();
        m_allRenderersStatic &= renderable.structural.isStatic;

        const auto cpuPose = ReadCpuRenderPose(*mr, *transform, frameData.devicePose.HasValue());
        const glm::mat4 worldMatrix = cpuPose ? cpuPose->render : glm::mat4(1);
        const glm::mat4 boundsWorldMatrix = cpuPose ? cpuPose->bounds : glm::mat4(1);

        // World-space resident vertices deliberately render with an identity
        // matrix, while their conservative bounds still follow the object's
        // Transform anchor. Track those two matrices independently so the
        // fast path cannot freeze culling, picking, or editor handles.
        const bool renderTransformChanged =
            poseAuthorityChanged || std::memcmp(&worldMatrix, &frameData.worldMatrix, sizeof(glm::mat4)) != 0;
        const bool boundsTransformChanged =
            poseAuthorityChanged ||
            std::memcmp(&boundsWorldMatrix, &frameData.boundsWorldMatrix, sizeof(glm::mat4)) != 0;
        const bool dynamicSkinBounds = source.skinnedRenderer && source.skinnedRenderer->HasRuntimeSkinnedMesh();
        const bool bufferDirty = mr->ConsumeMeshBufferDirty();
        const size_t cachedDrawCallEnd =
            std::min(cache.drawCallStart + cache.drawCallCount, frame.m_drawCalls.drawCalls.size());
        bool inlinePublicationChanged = false;
        if (mr->HasInlineMesh()) {
            const uint64_t version = mr->GetInlineMeshVersion();
            inlinePublicationChanged = source.inlineMeshVersion != version;
            for (size_t drawCallIndex = cache.drawCallStart;
                 !inlinePublicationChanged && drawCallIndex < cachedDrawCallEnd; ++drawCallIndex) {
                const DrawCall &draw = frame.m_drawCalls.drawCalls[drawCallIndex];
                inlinePublicationChanged = draw.meshRuntimeVersion != version;
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
        if (!cpuPose) {
            // Device model/AABB are the authoritative publication. These CPU
            // placeholders are never used to reject or sort its draw calls.
            frameData.boundsWorldMatrix = boundsWorldMatrix;
            frameData.worldBounds = {};
        } else if (boundsTransformChanged || geometryPublicationChanged || dynamicSkinBounds) {
            const bool translationOnly =
                boundsTransformChanged && !poseAuthorityChanged && !geometryPublicationChanged &&
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
            auto *skinned = source.skinnedRenderer;
            const auto pose =
                skinned && skinned->HasRuntimeSkinnedMesh() ? skinned->GetRuntimeSkinPoseSnapshot() : nullptr;

            auto patchSkinPalette = [&](DrawCall &dc) {
                if (!skinned)
                    return;
                dc.skinBoneMatrices = pose ? pose->current : nullptr;
                dc.skinContinuity = pose ? pose->continuity : 0;
            };

            for (size_t drawCallIndex = cache.drawCallStart; drawCallIndex < drawCallEnd; ++drawCallIndex) {
                DrawCall &dc = frame.m_drawCalls.drawCalls[drawCallIndex];
                dc.devicePose = frameData.devicePose;
                dc.isStatic = renderable.structural.isStatic;
                dc.material = mr->GetEffectiveMaterial(dc.materialSlot);
                dc.parameterBlock = mr->GetParameterBlock(dc.materialSlot);
                dc.castsShadows = mr->CastsShadows();
                dc.receivesShadows = mr->ReceivesShadows();
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
                    dc.meshIndexFormat = MeshIndexFormat::Auto;
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
                const glm::mat4 drawWorldMatrix = mr->ResolveSubmeshWorldMatrix(worldMatrix);

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
            } else if (skinned) {
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
        const auto geometry = renderer->GetAssetGeometry();
        if (!geometry)
            return;
        const std::vector<Vertex> *objVerticesPtr = &geometry->vertices;
        const std::vector<uint32_t> *objIndicesPtr = &geometry->indices;
        const std::vector<SubMesh> *subMeshesPtr = &geometry->subMeshes;
        std::shared_ptr<const void> meshDataOwner = geometry;
        SkinPoseHistory::SnapshotPtr skinPose;
        MeshIndexFormat drawIndexFormat = meshPtr->GetIndexFormat();
        if (auto *skinned = source.skinnedRenderer; skinned && skinned->HasRuntimeSkinnedMesh()) {
            objVerticesPtr = &skinned->GetRuntimeSkinnedVertices();
            objIndicesPtr = &skinned->GetRuntimeSkinnedIndices();
            subMeshesPtr = &skinned->GetRuntimeSkinnedSubMeshes();
            const auto runtimeModel = skinned->GetRuntimeModelSnapshot();
            meshDataOwner = runtimeModel;
            if (runtimeModel)
                drawIndexFormat = runtimeModel->indexFormat;
            skinPose = skinned->GetRuntimeSkinPoseSnapshot();
        }
        const auto stampGeometry = [&](DrawCall &draw) {
            draw.meshAssetGuid = assetRef.GetGuid();
            draw.meshRuntimeVersion = assetRef.GetCachedVersion();
            draw.meshGeometryView =
                renderer->IsModelNodeLocal() ? MeshGeometryView::NodeLocal : MeshGeometryView::MergedModelSpace;
            draw.meshDataOwner = meshDataOwner;
            if (skinPose) {
                draw.skinBoneMatrices = skinPose->current;
                draw.skinContinuity = skinPose->continuity;
            }
        };
        const auto &objVertices = *objVerticesPtr;
        const auto &objIndices = *objIndicesPtr;
        const bool gpuResidentOnly = !source.skinnedRenderer && !meshPtr->HasCpuGeometry();
        if ((objVertices.empty() || objIndices.empty()) && !gpuResidentOnly)
            return;

        const glm::mat4 &worldMatrix = frame.worldMatrix;

        uint32_t subMeshCount = static_cast<uint32_t>(subMeshesPtr ? subMeshesPtr->size() : 0);
        int32_t submeshFilter = renderer->GetSubmeshIndex();
        int32_t nodeGroup = renderer->GetNodeGroup();
        if (subMeshCount == 0) {
            // Fallback: single draw call for entire mesh
            DrawCall dc;
            dc.indexStart = 0;
            dc.indexCount = gpuResidentOnly ? geometry->indexCount : static_cast<uint32_t>(objIndices.size());
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
            dc.receivesShadows = renderer->ReceivesShadows();
            dc.isStatic = structural.isStatic;
            dc.worldBounds = frame.worldBounds;
            dc.devicePose = frame.devicePose;
            dc.meshVertices = &objVertices;
            dc.meshIndices = &objIndices;
            dc.meshIndexFormat = drawIndexFormat;
            stampGeometry(dc);
            dc.forceBufferUpdate = bufferDirty;
            result.drawCalls.push_back(dc);
        } else if (submeshFilter >= 0 && static_cast<uint32_t>(submeshFilter) < subMeshCount) {
            // Single submesh mode — render only the specified submesh
            const auto &sub = (*subMeshesPtr)[static_cast<uint32_t>(submeshFilter)];
            const glm::mat4 effectiveMatrix = renderer->ResolveSubmeshWorldMatrix(worldMatrix);
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
            dc.receivesShadows = renderer->ReceivesShadows();
            dc.isStatic = structural.isStatic;
            dc.worldBounds = frame.worldBounds;
            dc.devicePose = frame.devicePose;
            dc.meshVertices = &objVertices;
            dc.meshIndices = &objIndices;
            dc.meshIndexFormat = drawIndexFormat;
            stampGeometry(dc);
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
                dc.receivesShadows = renderer->ReceivesShadows();
                dc.isStatic = structural.isStatic;
                dc.worldBounds = frame.worldBounds;
                dc.devicePose = frame.devicePose;
                dc.meshVertices = &objVertices;
                dc.meshIndices = &objIndices;
                dc.meshIndexFormat = drawIndexFormat;
                stampGeometry(dc);
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
        dc.receivesShadows = renderer->ReceivesShadows();
        dc.isStatic = structural.isStatic;
        dc.worldBounds = frame.worldBounds;
        dc.devicePose = frame.devicePose;
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
