#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/AssetDependencyGraph.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxMesh/InxMesh.h>
#include <function/resources/InxMesh/MeshLoader.h>
#include <function/scene/Camera.h>
#include <function/scene/GameObject.h>
#include <function/scene/LineRenderer.h>
#include <function/scene/MeshRenderer.h>
#include <function/scene/PrimitiveMeshes.h>
#include <function/scene/Scene.h>
#include <function/scene/SceneManager.h>
#include <function/scene/SceneRenderBridge.h>
#include <function/scene/SceneRenderExtractor.h>
#include <function/scene/SkinnedMeshRenderer.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <limits>
#include <memory>

using namespace infernux;

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
    CameraDrawCallResult leftResult = bridge.CullAndBuildForCamera(leftCamera, false);
    CameraDrawCallResult rightResult = bridge.CullAndBuildForCamera(rightCamera, false);
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
    CameraDrawCallResult leftCachedResult = bridge.CullAndBuildForCamera(leftCamera, false);
    assert(leftCachedResult.visibleDrawCallsRef && leftCachedResult.visibleDrawCallsRef->size() == 2);
    assert(std::any_of(leftCachedResult.visibleDrawCallsRef->begin(), leftCachedResult.visibleDrawCallsRef->end(),
                       [leftCube](const DrawCall &draw) { return draw.objectId == leftCube->GetID(); }));
    assert(leftCachedResult.visibleListRevision == leftResult.visibleListRevision);

    // Renderer-local material parameters are dynamic draw payload, not camera
    // visibility. A camera-local compact list must patch the immutable block
    // from the newest RenderWorld publication before reusing its cull result.
    MeshRenderer *leftRenderer = leftCube->GetComponent<MeshRenderer>();
    // A custom near plane affects native culling without altering the other
    // camera; reset restores the ordinary projection immediately.
    const auto originalLeftProjection = leftCamera->GetProjectionMatrix();
    leftCamera->SetProjectionMatrix(leftCamera->CalculateObliqueMatrix(glm::vec4(0, 0, 1, -6)));
    const auto clippedLeft = bridge.CullAndBuildForCamera(leftCamera, false);
    assert(clippedLeft.visibleDrawCallsRef);
    assert(std::none_of(clippedLeft.visibleDrawCallsRef->begin(), clippedLeft.visibleDrawCallsRef->end(),
                        [leftCube](const DrawCall &draw) { return draw.objectId == leftCube->GetID(); }));
    const auto untouchedRight = bridge.CullAndBuildForCamera(rightCamera, false);
    assert(untouchedRight.visibleListRevision == rightResult.visibleListRevision);
    auto clonedLeft = leftCamera->Clone();
    const auto *clonedCamera = static_cast<const Camera *>(clonedLeft.get());
    assert(clonedCamera->HasCustomProjectionMatrix());
    assert(clonedCamera->GetProjectionMatrix() == leftCamera->GetProjectionMatrix());
    leftCamera->ResetProjectionMatrix();
    assert(leftCamera->GetProjectionMatrix() == originalLeftProjection);
    const auto restoredLeft = bridge.CullAndBuildForCamera(leftCamera, false);
    assert(std::any_of(restoredLeft.visibleDrawCallsRef->begin(), restoredLeft.visibleDrawCallsRef->end(),
                       [leftCube](const DrawCall &draw) { return draw.objectId == leftCube->GetID(); }));

    // A view override can look elsewhere without mutating the scene Transform.
    leftCamera->SetViewMatrix(rightCamera->GetViewMatrix());
    leftCamera->SetInvertCulling(true);
    const auto overridden = bridge.CullAndBuildForCamera(leftCamera, false);
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
    const auto stillRight = bridge.CullAndBuildForCamera(rightCamera, false);
    assert(stillRight.visibleListRevision == untouchedRight.visibleListRevision);
    leftCamera->ResetViewMatrix();
    leftCamera->SetInvertCulling(false);

    auto sharedMaterial = InxMaterial::CreateDefaultLit();
    leftRenderer->SetMaterial(0, sharedMaterial);
    bridge.PrepareFrame(false);
    CameraDrawCallResult leftMaterialAssigned = bridge.CullAndBuildForCamera(leftCamera, false);
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
    bool rejectedNonFiniteOverride = false;
    try {
        leftRenderer->SetParameter(0, "baseColor", glm::vec4(0.1f, std::numeric_limits<float>::infinity(), 0.2f, 1.0f),
                                   false);
    } catch (const std::invalid_argument &) {
        rejectedNonFiniteOverride = true;
    }
    assert(rejectedNonFiniteOverride && !leftRenderer->GetParameterBlock(0));
    leftRenderer->SetParameter(0, "baseColor", glm::vec4(0.1f, 0.8f, 0.2f, 1.0f), false);
    const auto stableRuntimePublication = leftRenderer->GetParameterBlock(0);
    assert(stableRuntimePublication);
    leftRenderer->SetParameter(0, "baseColor", glm::vec4(0.1f, 0.8f, 0.2f, 1.0f), false);
    assert(leftRenderer->GetParameterBlock(0) == stableRuntimePublication);
    leftRenderer->SetParameter(0, "baseColor", glm::vec4(0.8f, 0.2f, 0.1f, 1.0f), false, "animation");
    const auto laterOwnerPublication = leftRenderer->GetParameterBlock(0);
    assert(laterOwnerPublication && laterOwnerPublication != stableRuntimePublication);
    // Repeating script's own value is not a no-op after another owner wrote
    // the field: actual write order remains authoritative.
    leftRenderer->SetParameter(0, "baseColor", glm::vec4(0.1f, 0.8f, 0.2f, 1.0f), false);
    assert(leftRenderer->GetParameterBlock(0) != laterOwnerPublication);
    assert(std::get<glm::vec4>(leftRenderer->GetParameter(0, "baseColor")->value) == glm::vec4(0.1f, 0.8f, 0.2f, 1.0f));
    bridge.PrepareFrame(false);
    CameraDrawCallResult leftParameterChanged = bridge.CullAndBuildForCamera(leftCamera, false);
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
    twoSlotRenderer->SetParameter(0, "baseColor", glm::vec4(0.8f, 0.1f, 0.1f, 1.0f), true);
    twoSlotRenderer->SetParameter(1, "baseColor", glm::vec4(0.1f, 0.8f, 0.1f, 1.0f), true);
    const auto stablePersistentPublication = twoSlotRenderer->GetParameterBlock(1);
    twoSlotRenderer->SetParameter(1, "baseColor", glm::vec4(0.1f, 0.8f, 0.1f, 1.0f), true);
    assert(twoSlotRenderer->GetParameterBlock(1) == stablePersistentPublication);
    twoSlotRenderer->SetParameter(1, "metallic", 0.65f, false, "play-runtime");
    auto rendererClone = twoSlotRenderer->Clone();
    auto *clonedRenderer = static_cast<MeshRenderer *>(rendererClone.get());
    assert(clonedRenderer->GetParameter(1, "baseColor", true));
    assert(!clonedRenderer->GetParameter(1, "metallic"));
    const auto serializedRenderer = twoSlotRenderer->SerializeDocument();
    assert(serializedRenderer.contains("parameterOverrides"));
    assert(serializedRenderer["parameterOverrides"][1].contains("baseColor"));
    assert(!serializedRenderer["parameterOverrides"][1].contains("metallic"));
    MeshRenderer restoredRenderer;
    assert(restoredRenderer.DeserializeDocument(serializedRenderer));
    assert(restoredRenderer.GetParameter(1, "baseColor", true));
    assert(!restoredRenderer.GetParameter(1, "metallic"));
    assert(twoSlotRenderer->RemoveParameter(1, "metallic", false, "play-runtime"));
    const auto slotOneBlock = twoSlotRenderer->GetParameterBlock(1);
    bridge.PrepareFrame(false);
    CameraDrawCallResult twoSlotResult = bridge.CullAndBuildForCamera(leftCamera, false);
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
    assert(twoSlotDraws[0]->material == sharedMaterial && twoSlotDraws[1]->material == sharedMaterial);

    auto texturedMaterial = sharedMaterial->Clone();
    texturedMaterial->SetTextureGuid("texSampler", "white");
    twoSlotRenderer->SetMaterial(0, texturedMaterial);
    twoSlotRenderer->SetParameter(0, "texSampler", std::string("white"), false, "gameplay");
    const auto textureBlockBeforeReload = twoSlotRenderer->GetParameterBlock(0);
    twoSlotRenderer->OnParameterTextureAssetEvent("white", AssetEvent::Modified);
    const auto textureBlockAfterReload = twoSlotRenderer->GetParameterBlock(0);
    assert(textureBlockAfterReload && textureBlockAfterReload != textureBlockBeforeReload);
    assert(std::get<std::string>(textureBlockAfterReload->properties.at("texSampler").value) == "white");
    // A delete publishes another generation while already captured draw data
    // retains the prior generation until its GPU submission can retire.
    twoSlotRenderer->OnParameterTextureAssetEvent("white", AssetEvent::Deleted);
    const auto textureBlockAfterDelete = twoSlotRenderer->GetParameterBlock(0);
    assert(textureBlockAfterDelete && textureBlockAfterDelete != textureBlockAfterReload);
    assert(std::get<std::string>(textureBlockBeforeReload->properties.at("texSampler").value) == "white");
    assert(std::get<std::string>(textureBlockAfterReload->properties.at("texSampler").value) == "white");
    assert(twoSlotRenderer->RemoveParameter(0, "texSampler", false, "gameplay"));

    assert(twoSlotRenderer->RemoveParameter(0, "baseColor", true));
    bridge.PrepareFrame(false);
    CameraDrawCallResult oneSlotRemoved = bridge.CullAndBuildForCamera(leftCamera, false);
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
    CameraDrawCallResult leftContentChanged = bridge.CullAndBuildForCamera(leftCamera, false);
    assert(leftContentChanged.visibleDrawCallsRef);
    assert(std::any_of(leftContentChanged.visibleDrawCallsRef->begin(), leftContentChanged.visibleDrawCallsRef->end(),
                       [leftCube](const DrawCall &draw) { return draw.objectId == leftCube->GetID(); }));
    assert(leftContentChanged.visibleListRevision != leftCachedResult.visibleListRevision);

    // A procedural renderer can move only its vertices. The unchanged object
    // Transform must not leave the old world bounds in the camera cache.
    dynamicLine->SetPositions({glm::vec3(99.5f, 0.0f, 0.0f), glm::vec3(100.5f, 0.0f, 0.0f)});
    bridge.PrepareFrame(false);
    CameraDrawCallResult rightAfterLineMove = bridge.CullAndBuildForCamera(rightCamera, false);
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

    // Each imported skinned child draws only its node group. Its selection
    // bound must follow that same group after import scaling and the full
    // parent/child affine transform, including rotation and negative scale.
    auto importedMesh = registry.CreateRuntimeMesh("ImportedSkinnedBounds");
    const std::string importedGuid = importedMesh->GetGuid();
    auto skin = std::make_shared<InxSkinnedMesh>();
    skin->scaleFactor = 0.1f;
    const glm::vec3 authoredPositions[] = {{-1.0f, -2.0f, 0.0f},  {1.0f, -2.0f, 0.0f},   {0.0f, 2.0f, 0.0f},
                                           {100.0f, -2.0f, 0.0f}, {102.0f, -2.0f, 0.0f}, {101.0f, 2.0f, 0.0f}};
    std::vector<Vertex> scaledVertices(6);
    for (size_t index = 0; index < scaledVertices.size(); ++index) {
        skin->baseVertices.push_back(Vertex::Create(authoredPositions[index], {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}));
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
    InxMesh importedReplacement("ImportedSkinnedBounds");
    importedReplacement.SetData(std::move(scaledVertices), skin->indices, skin->subMeshes);
    importedReplacement.SetSkinnedData(skin);
    registry.PublishMesh(importedGuid, std::move(importedReplacement));

    GameObject *importParent = scene->CreateGameObject("ImportedParent");
    importParent->GetTransform()->SetPosition({3.0f, -4.0f, 2.0f});
    importParent->GetTransform()->SetLocalRotation(
        glm::angleAxis(glm::radians(37.0f), glm::normalize(glm::vec3(0.0f, 1.0f, 1.0f))));
    importParent->GetTransform()->SetLocalScale({-2.0f, 3.0f, 0.5f});
    GameObject *importChild = scene->CreateGameObject("ImportedNearNode");
    importChild->SetParent(importParent, false);
    importChild->GetTransform()->SetLocalPosition({1.0f, 0.5f, -0.25f});
    importChild->GetTransform()->SetLocalRotation(glm::angleAxis(glm::radians(-21.0f), glm::vec3(0.0f, 0.0f, 1.0f)));
    auto *skinnedRenderer = importChild->AddComponent<SkinnedMeshRenderer>();
    skinnedRenderer->SetSourceModelGuid(importedGuid);
    skinnedRenderer->SetNodeGroup(0);
    glm::vec3 nearMin, nearMax;
    skinnedRenderer->GetWorldBounds(nearMin, nearMax);
    glm::vec3 expectedMin(std::numeric_limits<float>::max());
    glm::vec3 expectedMax(std::numeric_limits<float>::lowest());
    for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y)
            for (int z = 0; z < 2; ++z) {
                const glm::vec3 corner(x ? nearGroup.boundsMax.x : nearGroup.boundsMin.x,
                                       y ? nearGroup.boundsMax.y : nearGroup.boundsMin.y,
                                       z ? nearGroup.boundsMax.z : nearGroup.boundsMin.z);
                const glm::vec3 position =
                    glm::vec3(importChild->GetTransform()->GetWorldMatrix() * glm::vec4(corner, 1.0f));
                expectedMin = glm::min(expectedMin, position);
                expectedMax = glm::max(expectedMax, position);
            }
    assert(glm::length(nearMin - expectedMin) < 1.0e-4f);
    assert(glm::length(nearMax - expectedMax) < 1.0e-4f);
    skinnedRenderer->SetSubmeshIndex(1);
    glm::vec3 farMin, farMax;
    skinnedRenderer->GetWorldBounds(farMin, farMax);
    assert(glm::length(farMin - nearMin) > 1.0f);
    assert(glm::length(farMax - nearMax) > 1.0f);

    manager.UnloadAllScenes();
    registry.DestroyRuntimeMesh(twoSlotMeshGuid);
    registry.DestroyRuntimeMesh(importedGuid);
    registry.Shutdown();
    return 0;
}
