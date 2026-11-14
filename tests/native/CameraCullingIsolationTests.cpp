#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/AssetDependencyGraph.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxMesh/InxMesh.h>
#include <function/resources/InxMesh/MeshLoader.h>
#include <function/scene/Camera.h>
#include <function/scene/GameObject.h>
#include <function/scene/Light.h>
#include <function/scene/LightingData.h>
#include <function/scene/LineRenderer.h>
#include <function/scene/MeshRenderer.h>
#include <function/scene/PrimitiveMeshes.h>
#include <function/scene/Scene.h>
#include <function/scene/SceneManager.h>
#include <function/scene/SceneRenderBridge.h>
#include <function/scene/SceneRenderExtractor.h>
#include <function/scene/SkeletonInstance.h>
#include <function/scene/SkinnedMeshRenderer.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <limits>
#include <memory>
#include <optional>

using namespace infernux;

static CameraDrawCallResult CullView(SceneRenderBridge &bridge, Camera *camera, bool shadows = false)
{
    auto frame = bridge.AcquirePreparedFrameForCamera(camera->GetComponentID());
    assert(frame);
    const auto *view = frame->FindView(camera->GetComponentID());
    assert(view);
    return bridge.CullAndBuildForView(*view, std::move(frame), shadows);
}

static void TestAnimatedBoundsInvalidateBothCameraCaches(AssetRegistry &registry, SceneManager &manager)
{
    const auto mesh = registry.CreateRuntimeMesh("AnimatedCulling");
    const std::string guid = mesh->GetGuid();
    auto skin = std::make_shared<InxSkinnedMesh>();
    skin->guid = guid;
    skin->scaleFactor = 1.0f;
    SkinnedRuntimeNode node;
    node.name = "Joint";
    skin->skeleton.nodes.push_back(node);
    skin->skeleton.nodeByName.emplace(node.name, 0);
    SkinnedRuntimeBone bone;
    bone.name = node.name;
    bone.nodeIndex = 0;
    skin->skeleton.bones.push_back(bone);
    skin->skeleton.boneByName.emplace(bone.name, 0);
    for (const auto position : {glm::vec3(99.5f, -0.5f, 0), glm::vec3(100.5f, -0.5f, 0), glm::vec3(100.0f, 0.5f, 0)}) {
        skin->baseVertices.push_back(Vertex::Create(position, {0, 0, -1}, {0, 0}));
        SkinInfluence influence;
        influence.weight[0] = 1.0f;
        skin->influences.push_back(influence);
    }
    skin->indices = {0, 1, 2};
    SubMesh sub;
    sub.vertexCount = 3;
    sub.indexCount = 3;
    sub.boundsMin = {99.5f, -0.5f, 0};
    sub.boundsMax = {100.5f, 0.5f, 0};
    skin->subMeshes.push_back(sub);
    SkinnedRuntimeAnimation animation;
    animation.name = "Move";
    animation.id = "culling-move";
    animation.durationTicks = 1.0;
    animation.ticksPerSecond = 1.0;
    animation.defaultLoop = false;
    SkinnedRuntimeTrack track;
    track.nodeIndex = 0;
    track.positions = {{0.0, glm::vec3(0)}, {1.0, glm::vec3(-100, 0, 0)}};
    track.rotations = {{0.0, glm::quat(1, 0, 0, 0)}};
    track.scales = {{0.0, glm::vec3(1)}};
    animation.tracks.push_back(track);
    animation.trackByNodeIndex = {0};
    skin->animations.push_back(animation);
    skin->NormalizeInfluences();
    assert(skin->IsAssetPayloadValid() && skin->skeleton.IsValid());
    InxMesh replacement("AnimatedCulling");
    replacement.SetData(skin->baseVertices, skin->indices, skin->subMeshes);
    replacement.SetSkinnedData(skin);
    registry.PublishMesh(guid, std::move(replacement));
    auto *scene = manager.CreateScene("AnimatedBounds");
    auto *object = scene->CreateGameObject("AnimatedTriangle");
    auto *renderer = object->AddComponent<SkinnedMeshRenderer>();
    renderer->SetSourceModelGuid(guid);
    auto makeCamera = [&](const char *name, float x) {
        auto *cameraObject = scene->CreateGameObject(name);
        cameraObject->GetTransform()->SetPosition({x, 0, -5});
        auto *camera = cameraObject->AddComponent<Camera>();
        camera->SetAspectRatio(1.0f);
        return camera;
    };
    Camera *nearCamera = makeCamera("Near", 0);
    Camera *farCamera = makeCamera("Far", 100);
    auto &bridge = SceneRenderBridge::Instance();
    auto contains = [object](const CameraDrawCallResult &result) {
        const auto &draws = result.visibleDrawCallsRef ? *result.visibleDrawCallsRef : result.visibleDrawCalls;
        return std::any_of(draws.begin(), draws.end(),
                           [object](const DrawCall &draw) { return draw.objectId == object->GetID(); });
    };
    auto check = [&](bool visibleFromNear) {
        glm::vec3 localMin, localMax, repeatedMin, repeatedMax, translatedMin, translatedMax;
        renderer->ComputeWorldBounds(glm::mat4(1), localMin, localMax);
        renderer->ComputeWorldBounds(glm::mat4(1), repeatedMin, repeatedMax);
        assert(localMin == repeatedMin && localMax == repeatedMax);
        const glm::vec3 offset(3, -4, 2);
        renderer->ComputeWorldBounds(glm::translate(glm::mat4(1), offset), translatedMin, translatedMax);
        assert(glm::length(translatedMin - localMin - offset) < 1.0e-5f);
        assert(glm::length(translatedMax - localMax - offset) < 1.0e-5f);
        bridge.PrepareFrame(false);
        const auto nearResult = CullView(bridge, nearCamera, false);
        const auto farResult = CullView(bridge, farCamera, false);
        assert(contains(nearResult) == visibleFromNear);
        assert(contains(farResult) == !visibleFromNear);
        assert(contains(nearResult) == visibleFromNear); // The second camera does not modify the first list.
        const auto cached = CullView(bridge, nearCamera, false);
        assert(cached.visibleListRevision == nearResult.visibleListRevision);
        assert(contains(cached) == visibleFromNear);
    };
    renderer->SubmitAnimationPose("Move", 0, 0, "", 0, 0, false);
    check(false);
    manager.CommitSkinPoseHistories();
    renderer->SubmitAnimationPose("Move", 1, 1, "", 0, 0, false);
    check(true); // No object or camera transform changed: only the skeleton moved.
    manager.CommitSkinPoseHistories();
    const auto stoppedRevision = manager.GetRenderTransformRevision();
    check(true);
    check(true); // A stopped pose remains visible without repeated submissions.
    assert(manager.GetRenderTransformRevision() == stoppedRevision);
    renderer->SubmitAnimationPose("Move", 0, 0, "", 0, 0, false);
    check(false);
    manager.CommitSkinPoseHistories();
    check(false);
    manager.UnloadAllScenes();
    registry.DestroyRuntimeMesh(guid);
}

static void TestResidentLightShadows(SceneManager &manager)
{
    for (auto type : {LightType::Directional, LightType::Spot, LightType::Point, LightType::Area}) {
        // Keep both loaded Scenes, their camera and the persistent light in
        // one real runtime owner, including the final post-Stop shadow check.
        auto &runtimeWorld = manager.BeginRuntimeWorld();
        std::optional<WorldExecutionScope> runtimeScope(std::in_place, &runtimeWorld);
        Scene *sceneA = manager.CreateScene("ShadowWorldA");
        Scene *sceneB = manager.CreateScene("ShadowWorldB");
        Scene *preview = manager.CreatePreviewScene("ShadowPreview");
        auto *cameraObject = sceneA->CreateGameObject("Camera");
        cameraObject->GetTransform()->SetPosition({0, 0, -5});
        auto *camera = cameraObject->AddComponent<Camera>();
        camera->SetClipPlanes(0.1f, 100.0f);
        camera->SetAspectRatio(1.0f);
        auto makeLight = [type](Scene *scene, uint32_t mask) {
            auto *object = scene->CreateGameObject("Light");
            object->GetTransform()->SetPosition({0, 3, -2});
            auto *light = object->AddComponent<Light>();
            light->SetLightType(type);
            light->SetShadows(LightShadows::Hard);
            light->SetCullingMask(mask);
            return light;
        };
        Light *lightA = makeLight(sceneA, 1u);
        Light *lightB = makeLight(sceneB, 2u);
        makeLight(preview, 4u);
        const bool published = manager.PublishRuntimeWorld(runtimeWorld.id);
        assert(published);
        const uint64_t idA = lightA->GetGameObject()->GetID();
        const uint64_t idB = lightB->GetGameObject()->GetID();
        const uint32_t viewCount = type == LightType::Directional ? 4u : type == LightType::Spot ? 1u : 6u;
        SceneLightCollector collector;
        auto check = [&](bool presentA, bool presentB, bool shadowB = true) {
            auto *active = manager.GetActiveScene();
            collector.CollectLights(active, {0, 0, -5});
            collector.ComputeShadowVP(active, {0, 0, -5}, 4096, camera, {1, 20});
            collector.BuildShaderLightingUBO();
            const auto &snapshot = collector.GetCanonicalLightSnapshot();
            const auto &lights = type == LightType::Directional ? snapshot.directionalLights : snapshot.localLights;
            assert(lights.size() == size_t(presentA) + size_t(presentB));
            const auto &frame = collector.GetShadowFrame();
            assert(frame.assignments.size() == size_t(presentA) + size_t(presentB && shadowB));
            assert(frame.views.size() == frame.assignments.size() * viewCount);
            for (const auto &light : lights) {
                const uint64_t id = uint64_t(light.identityAndShadow.x) | (uint64_t(light.identityAndShadow.y) << 32u);
                assert((presentA && id == idA) || (presentB && id == idB));
                const auto *assignment = frame.Find(id);
                const bool casts = id == idA || shadowB;
                assert((assignment != nullptr) == casts);
                assert(light.identityAndShadow.w == (casts ? viewCount : 0u));
                if (!assignment)
                    continue;
                assert(assignment->viewCount == viewCount);
                assert(light.identityAndShadow.z == assignment->firstView);
                for (uint32_t index = 0; index < viewCount; ++index) {
                    const auto &view = frame.views[assignment->firstView + index];
                    assert(view.lightId == id && view.subView == index);
                    assert(view.cullingMask == (id == idA ? 1u : 2u));
                    assert(view.atlas.IsValid());
                    for (int column = 0; column < 4; ++column)
                        for (int row = 0; row < 4; ++row)
                            assert(std::isfinite(view.viewProjection[column][row]));
                }
            }
            return frame;
        };
        manager.SetActiveScene(sceneA);
        const auto first = check(true, true);
        manager.SetActiveScene(sceneB);
        const auto switched = check(true, true);
        for (size_t index = 0; index < first.views.size(); ++index) {
            const auto &left = first.views[index];
            const auto &right = switched.views[index];
            assert(left.lightId == right.lightId && left.viewProjection == right.viewProjection);
            assert(left.atlas.x == right.atlas.x && left.atlas.y == right.atlas.y &&
                   left.atlas.size == right.atlas.size);
        }
        manager.SetActiveScene(sceneA);
        lightB->SetEnabled(false);
        check(true, false);
        lightB->SetEnabled(true);
        lightB->GetGameObject()->SetActive(false);
        check(true, false);
        lightB->GetGameObject()->SetActive(true);
        lightA->SetEnabled(false);
        check(false, true);
        lightA->SetEnabled(true);
        lightB->SetShadows(LightShadows::None);
        check(true, true, false);
        lightB->SetShadows(LightShadows::Hard);
        manager.ClosePreviewScene(preview);
        check(true, true);

        // A persistent light stays in the same render world after its source
        // scene is unloaded; Stop must retire it from both lighting and shadows.
        manager.Play();
        manager.DontDestroyOnLoad(lightB->GetGameObject());
        manager.PrepareActiveSceneReplacement();
        assert(lightB->GetGameObject()->GetScene() == manager.GetRuntimePersistentScene());
        manager.UnloadScene(sceneB);
        check(true, true);
        lightB->SetEnabled(false);
        check(true, false);
        lightB->SetEnabled(true);
        check(true, true);
        manager.Stop();
        check(true, false);
        manager.UnloadAllScenes();
        assert(manager.GetActiveLights().empty());
        runtimeScope.reset();
        const bool retired = manager.CollectRetiredRuntimeWorlds();
        assert(retired);
    }
}

static void TestEffectiveProjectionShadowCoverage(SceneManager &manager)
{
    auto *scene = manager.CreateScene("EffectiveShadowProjection");
    auto *camera = scene->CreateGameObject("Camera")->AddComponent<Camera>();
    auto *light = scene->CreateGameObject("Sun")->AddComponent<Light>();
    light->SetShadows(LightShadows::Hard);
    light->GetTransform()->SetEulerAngles(30, 45, 0);
    SceneLightCollector collector;
    auto collect = [&]() {
        collector.CollectLights(scene, {});
        collector.ComputeShadowVP(scene, {}, 4096, camera, {2, 20});
        return collector.GetShadowFrame();
    };
    auto covered = [&](const lighting::ShadowFrame &frame) {
        assert(frame.views.size() == lighting::DirectionalCascadeCount);
        const glm::dmat4 projection(camera->GetProjectionMatrix());
        const auto inverse = glm::inverse(projection);
        const glm::dmat4 cameraToWorld(camera->GetCameraToWorldMatrix());
        size_t tested = 0;
        for (const auto &view : frame.views) {
            // Independent receiver oracle: interior pixels at three depths
            // selected by the shader's cascade ranges, intersected with real
            // projection rays. Ignore points clipped by an oblique near plane.
            for (double fraction : {0.1, 0.5, 0.9})
                for (double x : {-0.85, 0.0, 0.85})
                    for (double y : {-0.85, 0.0, 0.85}) {
                        const double depth = glm::mix(double(view.splitNear), double(view.splitFar), fraction);
                        const auto h0 = inverse * glm::dvec4(x, y, 1, 1);
                        const auto h1 = inverse * glm::dvec4(x, y, 0.5, 1);
                        assert(h0.w != 0 && h1.w != 0);
                        const auto p0 = glm::dvec3(h0) / h0.w;
                        const auto p1 = glm::dvec3(h1) / h1.w;
                        const auto receiver = p0 + (p1 - p0) * ((depth - p0.z) / (p1.z - p0.z));
                        const auto clip = projection * glm::dvec4(receiver, 1);
                        if (clip.w <= 0 || clip.z < 0 || clip.z > clip.w)
                            continue;
                        ++tested;
                        const auto world = cameraToWorld * glm::dvec4(receiver, 1);
                        const auto shadow = glm::dmat4(view.viewProjection) * world;
                        assert(std::isfinite(shadow.x) && std::isfinite(shadow.y) && std::isfinite(shadow.z));
                        assert(std::abs(shadow.x / shadow.w) <= 1.002);
                        assert(std::abs(shadow.y / shadow.w) <= 1.002);
                    }
        }
        assert(tested >= 36);
    };
    for (bool affineView : {false, true}) {
        glm::mat4 view(1.0f);
        if (affineView) {
            view[0][0] = 0.5f;
            view[1][0] = 0.3f;
            view[2][2] = 2.0f;
            view[3] = {4, -3, 1, 1};
        }
        camera->SetViewMatrix(view);
        for (int kind = 0; kind < 5; ++kind) {
            camera->ResetProjectionMatrix();
            camera->SetClipPlanes(0.1f, 100.0f);
            camera->SetAspectRatio(1.3f);
            camera->SetFieldOfView(60.0f);
            camera->SetProjectionMode(kind == 1 ? CameraProjection::Orthographic : CameraProjection::Perspective);
            camera->SetOrthographicSize(5.0f);
            auto projection = camera->GetProjectionMatrix();
            if (kind == 2) { // Asymmetric lens shift.
                projection[2][0] += 0.6f;
                projection[2][1] -= 0.3f;
            } else if (kind == 3) {
                projection = camera->CalculateObliqueMatrix({0.3f, 0.1f, 1.0f, -1.0f});
            } else if (kind == 4) { // Left-handed ZO infinite far plane.
                projection[2][2] = 0.0f;
                projection[3][2] = 0.1f;
            }
            const auto ordinary = collect();
            if (kind <= 1)
                covered(ordinary);
            camera->SetProjectionMatrix(projection);
            const auto original = collect();
            covered(original);
            if (kind <= 1)
                for (size_t index = 0; index < original.views.size(); ++index)
                    assert(original.views[index].viewProjection == ordinary.views[index].viewProjection);

            // These parameters are dormant until ResetProjectionMatrix. None
            // may affect cascade geometry, depth selection or atlas placement.
            camera->SetClipPlanes(0.01f, 5000.0f);
            camera->SetFieldOfView(20.0f);
            camera->SetAspectRatio(2.0f);
            camera->SetProjectionMode(CameraProjection::Orthographic);
            camera->SetOrthographicSize(20.0f);
            assert(camera->GetProjectionMatrix() == projection);
            const auto changed = collect();
            covered(changed);
            for (size_t index = 0; index < changed.views.size(); ++index) {
                const auto &a = original.views[index];
                const auto &b = changed.views[index];
                assert(a.splitNear == b.splitNear && a.splitFar == b.splitFar);
                assert(a.viewProjection == b.viewProjection);
                assert(a.atlas.x == b.atlas.x && a.atlas.y == b.atlas.y && a.atlas.size == b.atlas.size);
            }
        }
    }
    manager.UnloadAllScenes();
}

static void TestRetiredCameraMaterials(SceneManager &manager)
{
    auto &bridge = SceneRenderBridge::Instance();
    for (int retirement = 0; retirement != 3; ++retirement) {
        auto *scene = manager.CreateScene("CameraMaterialLifetime");
        auto *cameraObject = scene->CreateGameObject("TransientCamera");
        cameraObject->GetTransform()->SetPosition({0, 0, -5});
        auto *camera = cameraObject->AddComponent<Camera>();
        camera->SetAspectRatio(1.0f);
        auto *object = scene->CreateGameObject("TransientGeometry");
        auto *renderer = object->AddComponent<MeshRenderer>();
        renderer->SetSharedPrimitiveMesh(PrimitiveMeshes::GetCubeVertices(), PrimitiveMeshes::GetCubeIndices(), "Cube");
        auto material = std::make_shared<InxMaterial>("TransientCameraMaterial");
        std::weak_ptr<InxMaterial> retiredAuthor = material;
        renderer->SetMaterial(0, material);
        material.reset();
        bridge.PrepareFrame(false);
        auto result = CullView(bridge, camera, true);
        assert(result.visibleDrawCallsRef && result.visibleDrawCallsRef->size() == 1);
        std::weak_ptr<InxMaterial> retiredMaterial = result.visibleDrawCallsRef->front().material;
        assert(retiredMaterial.lock() != retiredAuthor.lock());
        // An explicitly retained immutable publication must remain usable.
        auto consumer = result.worldOwner;
        const auto *borrowedVisible = result.visibleDrawCallsRef;
        const auto *borrowedShadows = result.shadowDrawCallsRef;
        result = {};
        if (retirement == 2) {
            manager.UnloadAllScenes();
        } else {
            scene->DestroyGameObject(object);
            if (retirement == 0)
                scene->DestroyGameObject(cameraObject);
            else
                camera->SetEnabled(false);
            scene->ProcessPendingDestroys();
        }
        for (int frame = 0; frame != 16; ++frame) {
            bridge.PrepareFrame(false);
            (void)bridge.BuildDrawCalls();
        }
        assert(retiredAuthor.expired());
        assert(!retiredMaterial.expired());
        assert(consumer->DrawCalls().drawCalls.front().material == retiredMaterial.lock());
        assert(borrowedVisible->size() == 1 && borrowedVisible->front().material == retiredMaterial.lock());
        assert(borrowedShadows->size() == 1 && borrowedShadows->front().material == retiredMaterial.lock());
        consumer.reset();
        for (int frame = 0; frame != 16; ++frame) {
            bridge.PrepareFrame(false);
            (void)bridge.BuildDrawCalls();
        }
        // A destroyed or no-longer-rendered camera must not pin old geometry.
        assert(retiredMaterial.expired());
        manager.UnloadAllScenes();
    }
}

int main()
{
    auto &registry = AssetRegistry::Instance();
    registry.Initialize(std::make_unique<AssetDatabase>());
    registry.RegisterLoader(ResourceType::Mesh, std::make_unique<MeshLoader>());

    // CPU scene decoding/clone keeps authored output identity with no renderer,
    // GPU device or RenderTexture loader. The renderer resolves it later.
    {
        Camera authored;
        const std::string guid = "0123456789abcdef0123456789abcdef";
        authored.SetTargetTextureGuid(guid);
        assert(authored.HasTargetTexture() && !authored.GetTargetTexture());
        assert(AssetDependencyGraph::Instance().HasDependency(authored.GetInstanceGuid(), guid));
        const auto document = authored.SerializeDocument();
        assert(document.at("targetTextureGuid") == guid);
        auto cloned = authored.Clone();
        auto *cameraClone = static_cast<Camera *>(cloned.get());
        assert(cameraClone->GetTargetTextureGuid() == guid && !cameraClone->GetTargetTexture());
        assert(AssetDependencyGraph::Instance().HasDependency(cameraClone->GetInstanceGuid(), guid));
        authored.SetTargetTextureGuid("");
        assert(!authored.HasTargetTexture());
        assert(authored.DeserializeDocument(document));
        assert(authored.GetTargetTextureGuid() == guid && !authored.GetTargetTexture());
        auto legacy = document;
        legacy.erase("targetTextureGuid");
        assert(authored.DeserializeDocument(legacy));
        assert(!authored.HasTargetTexture());
        assert(!AssetDependencyGraph::Instance().HasDependency(authored.GetInstanceGuid(), guid));
    }

    SceneManager &manager = SceneManager::Instance();
    manager.Stop();
    manager.UnloadAllScenes();

    TestRetiredCameraMaterials(manager);

    Scene *scene = manager.CreateScene("IndependentCameraCulling");
    auto createCube = [scene](const char *name, const glm::vec3 &position) {
        GameObject *object = scene->CreateGameObject(name);
        object->GetTransform()->SetPosition(position);
        MeshRenderer *renderer = object->AddComponent<MeshRenderer>();
        renderer->SetSharedPrimitiveMesh(PrimitiveMeshes::GetCubeVertices(), PrimitiveMeshes::GetCubeIndices(), "Cube");
        return object;
    };
    GameObject *leftCube = createCube("LeftCameraCube", glm::vec3(0.0f, 0.0f, 0.0f));
    GameObject *rightCube = createCube("RightCameraCube", glm::vec3(100.0f, 0.0f, 0.0f));

    GameObject *dynamicLineObject = scene->CreateGameObject("WorldSpaceDynamicLine");
    LineRenderer *dynamicLine = dynamicLineObject->AddComponent<LineRenderer>();
    dynamicLine->SetUseWorldSpace(true);
    dynamicLine->SetPositions({glm::vec3(-0.5f, 0.0f, 0.0f), glm::vec3(0.5f, 0.0f, 0.0f)});

    GameObject *leftCameraObject = scene->CreateGameObject("LeftCamera");
    leftCameraObject->GetTransform()->SetPosition(glm::vec3(0.0f, 0.0f, -5.0f));
    Camera *leftCamera = leftCameraObject->AddComponent<Camera>();
    leftCamera->SetAspectRatio(1.0f);

    GameObject *rightCameraObject = scene->CreateGameObject("RightCamera");
    rightCameraObject->GetTransform()->SetPosition(glm::vec3(100.0f, 0.0f, -5.0f));
    Camera *rightCamera = rightCameraObject->AddComponent<Camera>();
    rightCamera->SetAspectRatio(1.0f);

    SceneRenderBridge &bridge = SceneRenderBridge::Instance();
    bridge.PrepareFrame(false);
    CameraDrawCallResult leftResult = CullView(bridge, leftCamera, false);
    CameraDrawCallResult rightResult = CullView(bridge, rightCamera, false);
    assert(leftResult.visibleListRevision != 0);
    assert(rightResult.visibleListRevision != 0);
    assert(leftResult.visibleDrawCallsRef && leftResult.visibleDrawCallsRef->size() == 2);
    assert(rightResult.visibleDrawCallsRef && rightResult.visibleDrawCallsRef->size() == 1);
    assert(std::any_of(leftResult.visibleDrawCallsRef->begin(), leftResult.visibleDrawCallsRef->end(),
                       [leftCube](const DrawCall &draw) { return draw.objectId == leftCube->GetID(); }));
    assert(rightResult.visibleDrawCallsRef->front().objectId == rightCube->GetID());

    // The second camera must not mutate the first camera's cached list.
    assert(std::any_of(leftResult.visibleDrawCallsRef->begin(), leftResult.visibleDrawCallsRef->end(),
                       [leftCube](const DrawCall &draw) { return draw.objectId == leftCube->GetID(); }));
    CameraDrawCallResult leftCachedResult = CullView(bridge, leftCamera, false);
    assert(leftCachedResult.visibleDrawCallsRef && leftCachedResult.visibleDrawCallsRef->size() == 2);
    assert(std::any_of(leftCachedResult.visibleDrawCallsRef->begin(), leftCachedResult.visibleDrawCallsRef->end(),
                       [leftCube](const DrawCall &draw) { return draw.objectId == leftCube->GetID(); }));
    assert(leftCachedResult.visibleListRevision == leftResult.visibleListRevision);

    // Renderer-local material parameters are dynamic draw payload, not camera
    // visibility. A camera-local compact list must patch the immutable block
    // from the newest RenderWorld publication before reusing its cull result.
    MeshRenderer *leftRenderer = leftCube->GetComponent<MeshRenderer>();
    // A custom near plane affects native culling without altering the other
    // camera; the next publication adopts either the override or its reset.
    const auto originalLeftProjection = leftCamera->GetProjectionMatrix();
    leftCamera->SetProjectionMatrix(leftCamera->CalculateObliqueMatrix(glm::vec4(0, 0, 1, -6)));
    bridge.PrepareFrame(false);
    const auto clippedLeft = CullView(bridge, leftCamera, false);
    assert(clippedLeft.visibleDrawCallsRef);
    assert(std::none_of(clippedLeft.visibleDrawCallsRef->begin(), clippedLeft.visibleDrawCallsRef->end(),
                        [leftCube](const DrawCall &draw) { return draw.objectId == leftCube->GetID(); }));
    const auto untouchedRight = CullView(bridge, rightCamera, false);
    assert(untouchedRight.visibleListRevision == rightResult.visibleListRevision);

    auto clonedLeft = leftCamera->Clone();
    const auto *clonedCamera = static_cast<const Camera *>(clonedLeft.get());
    assert(clonedCamera->HasCustomProjectionMatrix());
    assert(clonedCamera->GetProjectionMatrix() == leftCamera->GetProjectionMatrix());
    leftCamera->ResetProjectionMatrix();
    assert(leftCamera->GetProjectionMatrix() == originalLeftProjection);
    bridge.PrepareFrame(false);
    const auto restoredLeft = CullView(bridge, leftCamera, false);
    assert(std::any_of(restoredLeft.visibleDrawCallsRef->begin(), restoredLeft.visibleDrawCallsRef->end(),
                       [leftCube](const DrawCall &draw) { return draw.objectId == leftCube->GetID(); }));

    // A view override can look elsewhere without mutating the scene Transform.
    leftCamera->SetViewMatrix(rightCamera->GetViewMatrix());
    leftCamera->SetInvertCulling(true);
    bridge.PrepareFrame(false);
    const auto overridden = CullView(bridge, leftCamera, false);
    assert(overridden.visibleDrawCallsRef);
    assert(std::any_of(overridden.visibleDrawCallsRef->begin(), overridden.visibleDrawCallsRef->end(),
                       [rightCube](const DrawCall &draw) { return draw.objectId == rightCube->GetID(); }));
    assert(std::none_of(overridden.visibleDrawCallsRef->begin(), overridden.visibleDrawCallsRef->end(),
                        [leftCube](const DrawCall &draw) { return draw.objectId == leftCube->GetID(); }));
    assert(leftCamera->GetCameraToWorldMatrix()[3].x == 100.0f);
    assert(leftCameraObject->GetTransform()->GetWorldPosition().x == 0.0f);
    auto viewClone = leftCamera->Clone();
    const auto *viewCloneCamera = static_cast<const Camera *>(viewClone.get());
    assert(viewCloneCamera->HasCustomViewMatrix() && viewCloneCamera->GetInvertCulling());
    assert(viewCloneCamera->GetViewMatrix() == rightCamera->GetViewMatrix());
    assert(viewCloneCamera->GetCameraToWorldMatrix() == leftCamera->GetCameraToWorldMatrix());
    assert(!rightCamera->HasCustomViewMatrix() && !rightCamera->GetInvertCulling());
    const auto stillRight = CullView(bridge, rightCamera, false);
    assert(stillRight.visibleListRevision == untouchedRight.visibleListRevision);
    leftCamera->ResetViewMatrix();
    leftCamera->SetInvertCulling(false);

    auto sharedMaterial = InxMaterial::CreateDefaultLit();
    // This CPU culling fixture authors its own parameter declaration; shader
    // defaults are supplied by the renderer's compiler boundary in a project.
    sharedMaterial->SetColor("base_color", glm::vec4(1.0f));
    sharedMaterial->SetFloat("metallic", 0.0f);
    leftRenderer->SetMaterial(0, sharedMaterial);
    bridge.PrepareFrame(false);
    CameraDrawCallResult leftMaterialAssigned = CullView(bridge, leftCamera, false);
    const auto findLeftDraw = [leftCube](const CameraDrawCallResult &result) -> const DrawCall * {
        if (!result.visibleDrawCallsRef)
            return nullptr;
        const auto found =
            std::find_if(result.visibleDrawCallsRef->begin(), result.visibleDrawCallsRef->end(),
                         [leftCube](const DrawCall &draw) { return draw.objectId == leftCube->GetID(); });
        return found == result.visibleDrawCallsRef->end() ? nullptr : &*found;
    };
    const DrawCall *beforeParameter = findLeftDraw(leftMaterialAssigned);
    assert(beforeParameter && !beforeParameter->parameterBlock);
    // Shadow switches are live draw payload. They must refresh cached camera
    // lists even when no transform, mesh or material has changed.
    assert(beforeParameter->castsShadows && beforeParameter->receivesShadows);
    leftRenderer->SetCastShadows(false);
    leftRenderer->SetReceivesShadows(false);
    bridge.PrepareFrame(false);
    const auto noShadows = CullView(bridge, leftCamera, false);
    const DrawCall *shadowDisabled = findLeftDraw(noShadows);
    assert(shadowDisabled && !shadowDisabled->castsShadows && !shadowDisabled->receivesShadows);
    leftRenderer->SetCastShadows(true);
    leftRenderer->SetReceivesShadows(true);
    bridge.PrepareFrame(false);
    const auto restoredShadows = CullView(bridge, leftCamera, false);
    const DrawCall *shadowRestored = findLeftDraw(restoredShadows);
    assert(shadowRestored && shadowRestored->castsShadows && shadowRestored->receivesShadows);
    bool rejectedNonFiniteOverride = false;
    try {
        leftRenderer->SetParameter(0, "base_color", glm::vec4(0.1f, std::numeric_limits<float>::infinity(), 0.2f, 1.0f),
                                   false);
    } catch (const std::invalid_argument &) {
        rejectedNonFiniteOverride = true;
    }
    assert(rejectedNonFiniteOverride && !leftRenderer->GetParameterBlock(0));
    leftRenderer->SetParameter(0, "base_color", glm::vec4(0.1f, 0.8f, 0.2f, 1.0f), false);
    const auto stableRuntimePublication = leftRenderer->GetParameterBlock(0);
    assert(stableRuntimePublication);
    leftRenderer->SetParameter(0, "base_color", glm::vec4(0.1f, 0.8f, 0.2f, 1.0f), false);
    assert(leftRenderer->GetParameterBlock(0) == stableRuntimePublication);
    leftRenderer->SetParameter(0, "base_color", glm::vec4(0.8f, 0.2f, 0.1f, 1.0f), false, "animation");
    const auto laterOwnerPublication = leftRenderer->GetParameterBlock(0);
    assert(laterOwnerPublication && laterOwnerPublication != stableRuntimePublication);
    // Repeating script's own value is not a no-op after another owner wrote
    // the field: actual write order remains authoritative.
    leftRenderer->SetParameter(0, "base_color", glm::vec4(0.1f, 0.8f, 0.2f, 1.0f), false);
    assert(leftRenderer->GetParameterBlock(0) != laterOwnerPublication);
    assert(std::get<glm::vec4>(leftRenderer->GetParameter(0, "base_color")->value) ==
           glm::vec4(0.1f, 0.8f, 0.2f, 1.0f));
    bridge.PrepareFrame(false);
    CameraDrawCallResult leftParameterChanged = CullView(bridge, leftCamera, false);
    const DrawCall *afterParameter = findLeftDraw(leftParameterChanged);
    assert(afterParameter && afterParameter->parameterBlock);
    assert(afterParameter->parameterBlock == leftRenderer->GetParameterBlock(0));
    assert(leftParameterChanged.visibleListRevision != leftMaterialAssigned.visibleListRevision);

    // A shared material may feed multiple submeshes while each material slot
    // retains an independent renderer parameter publication. Removing one
    // field must update that draw only, not the sibling slot or material.
    GameObject *twoSlotObject = scene->CreateGameObject("TwoSlotRenderer");
    twoSlotObject->GetTransform()->SetPosition(glm::vec3(0.0f, 2.0f, 0.0f));
    MeshRenderer *twoSlotRenderer = twoSlotObject->AddComponent<MeshRenderer>();
    auto twoSlotMesh = registry.CreateRuntimeMesh("TwoSlotMesh");
    const std::string twoSlotMeshGuid = twoSlotMesh->GetGuid();
    std::vector<Vertex> twoSlotVertices(6);
    twoSlotVertices[0].pos = {-1.0f, -0.5f, 0.0f};
    twoSlotVertices[1].pos = {0.0f, -0.5f, 0.0f};
    twoSlotVertices[2].pos = {-0.5f, 0.5f, 0.0f};
    twoSlotVertices[3].pos = {0.0f, -0.5f, 0.0f};
    twoSlotVertices[4].pos = {1.0f, -0.5f, 0.0f};
    twoSlotVertices[5].pos = {0.5f, 0.5f, 0.0f};
    for (auto &vertex : twoSlotVertices)
        vertex.normal = {0.0f, 0.0f, -1.0f};
    SubMesh leftSubmesh{};
    leftSubmesh.indexCount = 3;
    leftSubmesh.vertexCount = 3;
    leftSubmesh.materialSlot = 0;
    SubMesh rightSubmesh = leftSubmesh;
    rightSubmesh.indexStart = 3;
    rightSubmesh.vertexStart = 3;
    rightSubmesh.materialSlot = 1;
    InxMesh twoSlotReplacement("TwoSlotMesh");
    twoSlotReplacement.SetData(std::move(twoSlotVertices), {0, 1, 2, 3, 4, 5}, {leftSubmesh, rightSubmesh});
    twoSlotReplacement.SetMaterialSlotNames({"Left", "Right"});
    registry.PublishMesh(twoSlotMeshGuid, std::move(twoSlotReplacement));
    twoSlotRenderer->SetMeshAsset(twoSlotMeshGuid, twoSlotMesh);
    twoSlotRenderer->SetMaterial(0, sharedMaterial);
    twoSlotRenderer->SetMaterial(1, sharedMaterial);
    twoSlotRenderer->SetParameter(0, "base_color", glm::vec4(0.8f, 0.1f, 0.1f, 1.0f), true);
    twoSlotRenderer->SetParameter(1, "base_color", glm::vec4(0.1f, 0.8f, 0.1f, 1.0f), true);
    const auto stablePersistentPublication = twoSlotRenderer->GetParameterBlock(1);
    twoSlotRenderer->SetParameter(1, "base_color", glm::vec4(0.1f, 0.8f, 0.1f, 1.0f), true);
    assert(twoSlotRenderer->GetParameterBlock(1) == stablePersistentPublication);
    twoSlotRenderer->SetParameter(1, "metallic", 0.65f, false, "play-runtime");
    auto rendererClone = twoSlotRenderer->Clone();
    auto *clonedRenderer = static_cast<MeshRenderer *>(rendererClone.get());
    assert(clonedRenderer->GetParameter(1, "base_color", true));
    assert(!clonedRenderer->GetParameter(1, "metallic"));
    const auto serializedRenderer = twoSlotRenderer->SerializeDocument();
    assert(serializedRenderer.contains("parameterOverrides"));
    assert(serializedRenderer["parameterOverrides"][1].contains("base_color"));
    assert(!serializedRenderer["parameterOverrides"][1].contains("metallic"));
    MeshRenderer restoredRenderer;
    assert(restoredRenderer.DeserializeDocument(serializedRenderer));
    assert(restoredRenderer.GetParameter(1, "base_color", true));
    assert(!restoredRenderer.GetParameter(1, "metallic"));
    assert(twoSlotRenderer->RemoveParameter(1, "metallic", false, "play-runtime"));
    const auto slotOneBlock = twoSlotRenderer->GetParameterBlock(1);
    bridge.PrepareFrame(false);
    CameraDrawCallResult twoSlotResult = CullView(bridge, leftCamera, false);
    std::vector<const DrawCall *> twoSlotDraws;
    for (const DrawCall &draw : *twoSlotResult.visibleDrawCallsRef) {
        if (draw.objectId == twoSlotObject->GetID())
            twoSlotDraws.push_back(&draw);
    }
    assert(twoSlotDraws.size() == 2);
    std::sort(twoSlotDraws.begin(), twoSlotDraws.end(),
              [](const DrawCall *left, const DrawCall *right) { return left->materialSlot < right->materialSlot; });
    assert(twoSlotDraws[0]->materialSlot == 0 &&
           twoSlotDraws[0]->parameterBlock == twoSlotRenderer->GetParameterBlock(0));
    assert(twoSlotDraws[1]->materialSlot == 1 && twoSlotDraws[1]->parameterBlock == slotOneBlock);
    assert(twoSlotDraws[0]->material == twoSlotDraws[1]->material);
    assert(twoSlotDraws[0]->material != sharedMaterial);
    assert(twoSlotDraws[0]->material->IsRenderPublication());

    auto texturedMaterial = sharedMaterial->Clone();
    texturedMaterial->SetTextureGuid("albedo_map", "white");
    twoSlotRenderer->SetMaterial(0, texturedMaterial);
    twoSlotRenderer->SetParameter(0, "albedo_map", std::string("white"), false, "gameplay");
    const auto textureBlockBeforeReload = twoSlotRenderer->GetParameterBlock(0);
    twoSlotRenderer->OnParameterTextureAssetEvent("white", AssetEvent::Modified);
    const auto textureBlockAfterReload = twoSlotRenderer->GetParameterBlock(0);
    assert(textureBlockAfterReload && textureBlockAfterReload != textureBlockBeforeReload);
    assert(std::get<std::string>(textureBlockAfterReload->properties.at("albedo_map").value) == "white");
    // A delete publishes another generation while already captured draw data
    // retains the prior generation until its GPU submission can retire.
    twoSlotRenderer->OnParameterTextureAssetEvent("white", AssetEvent::Deleted);
    const auto textureBlockAfterDelete = twoSlotRenderer->GetParameterBlock(0);
    assert(textureBlockAfterDelete && textureBlockAfterDelete != textureBlockAfterReload);
    assert(std::get<std::string>(textureBlockBeforeReload->properties.at("albedo_map").value) == "white");
    assert(std::get<std::string>(textureBlockAfterReload->properties.at("albedo_map").value) == "white");
    assert(twoSlotRenderer->RemoveParameter(0, "albedo_map", false, "gameplay"));

    assert(twoSlotRenderer->RemoveParameter(0, "base_color", true));
    bridge.PrepareFrame(false);
    CameraDrawCallResult oneSlotRemoved = CullView(bridge, leftCamera, false);
    twoSlotDraws.clear();
    for (const DrawCall &draw : *oneSlotRemoved.visibleDrawCallsRef) {
        if (draw.objectId == twoSlotObject->GetID())
            twoSlotDraws.push_back(&draw);
    }
    std::sort(twoSlotDraws.begin(), twoSlotDraws.end(),
              [](const DrawCall *left, const DrawCall *right) { return left->materialSlot < right->materialSlot; });
    assert(twoSlotDraws.size() == 2);
    assert(!twoSlotDraws[0]->parameterBlock);
    assert(twoSlotDraws[1]->parameterBlock == slotOneBlock);

    // Skinning changes dynamic draw-call payload without changing camera
    // visibility. A new content publication must still invalidate a graph's
    // cached submission, or Game View will keep the first (bind) pose.
    manager.NotifyMeshRendererContentChanged(leftRenderer);
    bridge.PrepareFrame(false);
    CameraDrawCallResult leftContentChanged = CullView(bridge, leftCamera, false);
    assert(leftContentChanged.visibleDrawCallsRef);
    assert(std::any_of(leftContentChanged.visibleDrawCallsRef->begin(), leftContentChanged.visibleDrawCallsRef->end(),
                       [leftCube](const DrawCall &draw) { return draw.objectId == leftCube->GetID(); }));
    assert(leftContentChanged.visibleListRevision != leftCachedResult.visibleListRevision);

    // A procedural renderer can move only its vertices. The unchanged object
    // Transform must not leave the old world bounds in the camera cache.
    dynamicLine->SetPositions({glm::vec3(99.5f, 0.0f, 0.0f), glm::vec3(100.5f, 0.0f, 0.0f)});
    bridge.PrepareFrame(false);
    CameraDrawCallResult rightAfterLineMove = CullView(bridge, rightCamera, false);
    assert(rightAfterLineMove.visibleDrawCallsRef);
    assert(
        std::any_of(rightAfterLineMove.visibleDrawCallsRef->begin(), rightAfterLineMove.visibleDrawCallsRef->end(),
                    [dynamicLineObject](const DrawCall &draw) { return draw.objectId == dynamicLineObject->GetID(); }));

    // RenderWorld recycles multiple immutable frames. A one-shot geometry
    // dirty flag may update the first recycled frame only; every later frame
    // must still compare its own durable mesh publication identity. Otherwise
    // dynamic resident/procedural meshes alternate between current and stale
    // topology as the snapshots rotate, which presents as whole-surface
    // flicker in Game View.
    GameObject *snapshotMeshObject = scene->CreateGameObject("SnapshotRotationMesh");
    MeshRenderer *snapshotMeshRenderer = snapshotMeshObject->AddComponent<MeshRenderer>();
    snapshotMeshRenderer->SetMesh(PrimitiveMeshes::GetQuadVertices(), PrimitiveMeshes::GetQuadIndices());

    SceneRenderExtractor snapshotExtractor;
    RenderWorldSnapshot snapshotWorld;
    std::vector<std::shared_ptr<const RenderWorldFrame>> retainedFrames;
    for (int i = 0; i < 3; ++i) {
        const size_t visibleCount = snapshotExtractor.ExtractCameraFrame(snapshotWorld, leftCamera);
        assert(visibleCount > 0);
        retainedFrames.push_back(snapshotWorld.Acquire());
    }
    retainedFrames.clear();

    auto changedVertices = PrimitiveMeshes::GetQuadVertices();
    changedVertices.front().pos.y += 0.25f;
    snapshotMeshRenderer->SetProceduralMesh(std::move(changedVertices), PrimitiveMeshes::GetQuadIndices());
    const uint64_t changedVersion = snapshotMeshRenderer->GetInlineMeshVersion();
    for (int i = 0; i < 6; ++i) {
        const size_t visibleCount = snapshotExtractor.ExtractCameraFrame(snapshotWorld, leftCamera);
        assert(visibleCount > 0);
        const auto publication = snapshotWorld.Acquire();
        assert(publication);
        const auto found = std::find_if(
            publication->DrawCalls().drawCalls.begin(), publication->DrawCalls().drawCalls.end(),
            [snapshotMeshObject](const DrawCall &draw) { return draw.objectId == snapshotMeshObject->GetID(); });
        assert(found != publication->DrawCalls().drawCalls.end());
        assert(found->meshRuntimeVersion == changedVersion);
        assert(found->meshVertices && !found->meshVertices->empty());
        assert(found->meshVertices->front().pos.y == PrimitiveMeshes::GetQuadVertices().front().pos.y + 0.25f);
    }

    // Additively loaded Scenes form one render world.  Changing the active
    // authoring Scene must not remove another loaded Scene's draw calls: the
    // selection-outline mask, Scene picking, and Gizmos all consume this same
    // publication and therefore need the selected object even when it belongs
    // to a non-active Scene.
    Scene *additiveScene = manager.CreateScene("RenderWorldAdditive");
    GameObject *additiveObject = additiveScene->CreateGameObject("AdditiveOutlinedObject");
    MeshRenderer *additiveRenderer = additiveObject->AddComponent<MeshRenderer>();
    additiveRenderer->SetMesh(PrimitiveMeshes::GetCubeVertices(), PrimitiveMeshes::GetCubeIndices());
    const uint64_t additiveObjectId = additiveObject->GetID();
    const uint64_t primaryObjectId = snapshotMeshObject->GetID();

    for (Scene *authoringOwner : {scene, additiveScene}) {
        manager.SetActiveScene(authoringOwner);
        const size_t visibleCount = snapshotExtractor.ExtractCameraFrame(snapshotWorld, leftCamera);
        assert(visibleCount > 0);
        const auto publication = snapshotWorld.Acquire();
        assert(publication);
        const auto &draws = publication->DrawCalls().drawCalls;
        const auto contains = [&draws](uint64_t objectId) {
            return std::any_of(draws.begin(), draws.end(),
                               [objectId](const DrawCall &draw) { return draw.objectId == objectId; });
        };
        assert(contains(primaryObjectId));
        assert(contains(additiveObjectId));
    }

    // Each imported skinned child uses its group's deformed bounds in the
    // shared skeleton's frame, including the root's full parent transform.
    auto importedMesh = registry.CreateRuntimeMesh("ImportedSkinnedBounds");
    const std::string importedGuid = importedMesh->GetGuid();
    auto skin = std::make_shared<InxSkinnedMesh>();
    skin->guid = importedGuid;
    skin->scaleFactor = 0.1f;
    skin->skeleton.nodes = {{"Root"}, {"NearJoint", 0}, {"FarJoint", 0}};
    skin->skeleton.nodeByName = {{"Root", 0}, {"NearJoint", 1}, {"FarJoint", 2}};
    skin->skeleton.bones = {{"NearJoint", 1}, {"FarJoint", 2}};
    skin->skeleton.boneByName = {{"NearJoint", 0}, {"FarJoint", 1}};
    const glm::vec3 authoredPositions[] = {{-1.0f, -2.0f, 0.0f},  {1.0f, -2.0f, 0.0f},   {0.0f, 2.0f, 0.0f},
                                           {100.0f, -2.0f, 0.0f}, {102.0f, -2.0f, 0.0f}, {101.0f, 2.0f, 0.0f}};
    std::vector<Vertex> scaledVertices(6);
    for (size_t index = 0; index < scaledVertices.size(); ++index) {
        skin->baseVertices.push_back(Vertex::Create(authoredPositions[index], {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}));
        SkinInfluence influence;
        influence.boneIndex[0] = index < 3 ? 0 : 1;
        influence.weight[0] = 1.0f;
        skin->influences.push_back(influence);
        scaledVertices[index] = skin->baseVertices.back();
        scaledVertices[index].pos *= skin->scaleFactor;
    }
    skin->indices = {0, 1, 2, 3, 4, 5};
    SubMesh nearGroup{};
    nearGroup.vertexCount = nearGroup.indexCount = 3;
    nearGroup.nodeGroup = 0;
    nearGroup.boundsMin = {-0.1f, -0.2f, 0.0f};
    nearGroup.boundsMax = {0.1f, 0.2f, 0.0f};
    SubMesh farGroup = nearGroup;
    farGroup.vertexStart = farGroup.indexStart = 3;
    farGroup.nodeGroup = 1;
    farGroup.boundsMin = {10.0f, -0.2f, 0.0f};
    farGroup.boundsMax = {10.2f, 0.2f, 0.0f};
    skin->subMeshes = {nearGroup, farGroup};
    skin->NormalizeInfluences();
    InxMesh importedReplacement("ImportedSkinnedBounds");
    importedReplacement.SetData(std::move(scaledVertices), skin->indices, skin->subMeshes);
    importedReplacement.SetSkinnedData(skin);
    registry.PublishMesh(importedGuid, std::move(importedReplacement));

    GameObject *importParent = scene->CreateGameObject("ImportedParent");
    importParent->GetTransform()->SetPosition({3.0f, -4.0f, 2.0f});
    importParent->GetTransform()->SetLocalRotation(
        glm::angleAxis(glm::radians(37.0f), glm::normalize(glm::vec3(0.0f, 1.0f, 1.0f))));
    importParent->GetTransform()->SetLocalScale({-2.0f, 3.0f, 0.5f});
    GameObject *importRoot = scene->CreateGameObject("ImportedCharacter");
    importRoot->SetParent(importParent, false);
    importRoot->GetTransform()->SetLocalPosition({1.0f, 0.5f, -0.25f});
    importRoot->GetTransform()->SetLocalRotation(glm::angleAxis(glm::radians(-21.0f), glm::vec3(0.0f, 0.0f, 1.0f)));
    auto *skeleton = importRoot->AddComponent<SkeletonInstance>();
    skeleton->SetSourceModelGuid(importedGuid);
    GameObject *importChild = scene->CreateGameObject("ImportedNearNode");
    importChild->SetParent(importRoot, false);
    importChild->GetTransform()->SetLocalPosition({0.75f, 0.25f, 1.0f});
    auto *skinnedRenderer = importChild->AddComponent<SkinnedMeshRenderer>();
    skinnedRenderer->SetSourceModelGuid(importedGuid);
    skinnedRenderer->SetNodeGroup(0);
    assert(skinnedRenderer->GetSkeletonInstance() == skeleton && skinnedRenderer->HasRuntimeSkinnedMesh());
    const glm::mat4 world = importRoot->GetTransform()->GetWorldMatrix();
    const auto checkGroupBounds = [&](const SubMesh &group) {
        glm::vec3 minimum, maximum;
        skinnedRenderer->GetWorldBounds(minimum, maximum);
        glm::vec3 expectedMin(std::numeric_limits<float>::max());
        glm::vec3 expectedMax(std::numeric_limits<float>::lowest());
        for (int x = 0; x < 2; ++x)
            for (int y = 0; y < 2; ++y)
                for (int z = 0; z < 2; ++z) {
                    const glm::vec3 corner(x ? group.boundsMax.x : group.boundsMin.x,
                                           y ? group.boundsMax.y : group.boundsMin.y,
                                           z ? group.boundsMax.z : group.boundsMin.z);
                    const glm::vec3 position(world * glm::vec4(corner, 1.0f));
                    expectedMin = glm::min(expectedMin, position);
                    expectedMax = glm::max(expectedMax, position);
                }
        assert(glm::length(minimum - expectedMin) < 1.0e-4f);
        assert(glm::length(maximum - expectedMax) < 1.0e-4f);
    };
    checkGroupBounds(nearGroup);
    skinnedRenderer->SetSubmeshIndex(1);
    checkGroupBounds(farGroup);

    // A camera-masked, offscreen caster must remain available to the light's
    // shadow frustum, including when the camera's visible list is cached.
    GameObject *offscreenCaster = createCube("MaskedOffscreenShadowCaster", glm::vec3(-3.0f, 3.0f, -2.5f));
    offscreenCaster->SetLayer(2);
    leftCamera->SetCullingMask(1u);
    bridge.PrepareFrame(false);
    const auto maskedShadows = CullView(bridge, leftCamera, true);
    assert(maskedShadows.visibleDrawCallsRef && maskedShadows.shadowDrawCallsRef);
    assert(std::none_of(maskedShadows.visibleDrawCallsRef->begin(), maskedShadows.visibleDrawCallsRef->end(),
                        [offscreenCaster](const DrawCall &draw) { return draw.objectId == offscreenCaster->GetID(); }));
    assert(std::any_of(maskedShadows.shadowDrawCallsRef->begin(), maskedShadows.shadowDrawCallsRef->end(),
                       [offscreenCaster](const DrawCall &draw) { return draw.objectId == offscreenCaster->GetID(); }));
    const auto cachedMaskedShadows = CullView(bridge, leftCamera, true);
    assert(cachedMaskedShadows.visibleListRevision == maskedShadows.visibleListRevision);
    assert(cachedMaskedShadows.shadowDrawCallsRef == maskedShadows.shadowDrawCallsRef);
    assert(cachedMaskedShadows.shadowListRevision == maskedShadows.shadowListRevision);
    assert(cachedMaskedShadows.shadowListRevision != 0);
    const auto withoutShadows = CullView(bridge, leftCamera, false);
    assert(!withoutShadows.shadowDrawCallsRef && withoutShadows.shadowDrawCalls.empty());
    leftCamera->SetCullingMask(0xFFFFFFFFu);
    scene->DestroyGameObject(offscreenCaster);
    bridge.PrepareFrame(false);

    manager.UnloadAllScenes();
    registry.DestroyRuntimeMesh(twoSlotMeshGuid);
    registry.DestroyRuntimeMesh(importedGuid);
    TestAnimatedBoundsInvalidateBothCameraCaches(registry, manager);
    TestResidentLightShadows(manager);
    TestEffectiveProjectionShadowCoverage(manager);
    registry.Shutdown();
    return 0;
}
