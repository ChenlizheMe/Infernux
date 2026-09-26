#include <function/renderer/RenderWorld.h>
#include <function/scene/Camera.h>
#include <function/scene/GameObject.h>
#include <function/scene/Light.h>
#include <function/scene/MeshRenderer.h>
#include <function/scene/PrimitiveMeshes.h>
#include <function/scene/Scene.h>
#include <function/scene/SceneManager.h>
#include <function/scene/SceneRenderExtractor.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

using namespace infernux;

namespace
{
struct AuthoredSceneObjects
{
    GameObject *rendered = nullptr;
    Camera *camera = nullptr;
};

AuthoredSceneObjects PopulateScene(Scene *scene, const char *prefix, float exposure)
{
    SceneEnvironmentSettings environment = scene->GetEnvironment();
    environment.skyExposure = exposure;
    scene->SetEnvironment(environment);

    GameObject *rendered = scene->CreateGameObject(std::string(prefix) + " Rendered");
    MeshRenderer *renderer = rendered->AddComponent<MeshRenderer>();
    renderer->SetSharedPrimitiveMesh(PrimitiveMeshes::GetCubeVertices(), PrimitiveMeshes::GetCubeIndices(), "Cube");
    rendered->AddComponent<Light>();

    GameObject *cameraObject = scene->CreateGameObject(std::string(prefix) + " Camera");
    cameraObject->GetTransform()->SetPosition(glm::vec3(0.0f, 0.0f, -5.0f));
    Camera *camera = cameraObject->AddComponent<Camera>();
    scene->SetMainCamera(camera);
    return {rendered, camera};
}

std::vector<uint64_t> ExtractObjectIds(SceneRenderExtractor &extractor, RenderWorldSnapshot &world, Camera *camera)
{
    (void)extractor.ExtractCameraFrame(world, camera);
    const auto frame = world.Acquire();
    assert(frame && frame->PrimaryView().valid);
    std::vector<uint64_t> ids;
    ids.reserve(frame->DrawCalls().drawCalls.size());
    for (const DrawCall &draw : frame->DrawCalls().drawCalls) {
        if (draw.frustumVisible)
            ids.push_back(draw.objectId);
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}

void RequireIds(const std::vector<uint64_t> &actual, std::initializer_list<uint64_t> expected)
{
    std::vector<uint64_t> sorted(expected);
    std::sort(sorted.begin(), sorted.end());
    assert(actual == sorted);
}
} // namespace

int main()
{
    SceneManager &manager = SceneManager::Instance();
    manager.Stop();
    manager.UnloadAllScenes();

    Scene *sceneA = manager.CreateScene("WebWorldA");
    const AuthoredSceneObjects objectsA = PopulateScene(sceneA, "A", 1.25f);
    Scene *sceneB = manager.CreateScene("WebWorldB");
    const AuthoredSceneObjects objectsB = PopulateScene(sceneB, "B", 2.5f);
    objectsB.camera->SetDepth(-100.0f);
    manager.SetActiveScene(sceneA);

    SceneRenderExtractor extractor;
    RenderWorldSnapshot world;
    auto resolveWebCamera = [&manager]() {
        Scene *active = manager.GetActiveScene();
        return active ? active->FindGameCamera(nullptr) : nullptr;
    };

    // The active Scene owns the explicit camera/environment preference, even
    // when an additive Scene has a lower-depth camera. Render extraction and
    // lighting still consume the complete runtime World registries.
    assert(resolveWebCamera() == objectsA.camera);
    assert(manager.GetActiveScene()->GetEnvironment().skyExposure == 1.25f);
    RequireIds(ExtractObjectIds(extractor, world, resolveWebCamera()),
               {objectsA.rendered->GetID(), objectsB.rendered->GetID()});
    assert(manager.GetActiveMeshRenderers().size() == 2);
    assert(manager.GetActiveLights().size() == 2);

    manager.SetActiveScene(sceneB);
    assert(resolveWebCamera() == objectsB.camera);
    assert(manager.GetActiveScene()->GetEnvironment().skyExposure == 2.5f);
    RequireIds(ExtractObjectIds(extractor, world, resolveWebCamera()),
               {objectsA.rendered->GetID(), objectsB.rendered->GetID()});

    // Loading the same authored document again must publish a distinct object
    // identity while retaining all three Scenes in one RenderWorld frame.
    const uint64_t sourceBId = objectsB.rendered->GetID();
    const auto documentB = sceneB->SerializeDocument();
    Scene *duplicateB = manager.CreateScene("WebWorldBDuplicate");
    auto duplicateCommit = duplicateB->CommitDocumentRetainingCurrentWorld(documentB);
    assert(duplicateCommit);
    const auto remapped = duplicateCommit->GetObjectIdRemap().find(sourceBId);
    assert(remapped != duplicateCommit->GetObjectIdRemap().end());
    const uint64_t duplicateBId = remapped->second;
    assert(duplicateBId != sourceBId && duplicateBId != objectsA.rendered->GetID());
    duplicateCommit->Finalize();
    manager.SetActiveScene(sceneA);
    RequireIds(ExtractObjectIds(extractor, world, resolveWebCamera()),
               {objectsA.rendered->GetID(), sourceBId, duplicateBId});
    assert(manager.GetActiveMeshRenderers().size() == 3);
    assert(manager.GetActiveLights().size() == 3);

    // Unloading the active additive Scene retires only that Scene's renderer
    // and light identities. The duplicate load and the original Scene remain.
    manager.SetActiveScene(sceneB);
    manager.UnloadScene(sceneB);
    assert(manager.GetActiveScene() == sceneA);
    RequireIds(ExtractObjectIds(extractor, world, resolveWebCamera()), {objectsA.rendered->GetID(), duplicateBId});
    assert(manager.GetActiveMeshRenderers().size() == 2);
    assert(manager.GetActiveLights().size() == 2);

    manager.SetActiveScene(duplicateB);
    Camera *duplicateCamera = resolveWebCamera();
    assert(duplicateCamera && duplicateCamera->GetGameObject()->GetScene() == duplicateB);
    assert(duplicateB->GetEnvironment().skyExposure == 2.5f);
    manager.UnloadScene(duplicateB);
    assert(manager.GetActiveScene() == sceneA);
    RequireIds(ExtractObjectIds(extractor, world, resolveWebCamera()), {objectsA.rendered->GetID()});

    // Runtime-persistent geometry and lights use the same registries. They
    // survive active Scene replacement and retire together when Play stops.
    const uint64_t persistentObjectId = objectsA.rendered->GetID();
    manager.Play();
    manager.DontDestroyOnLoad(objectsA.rendered);
    manager.PrepareActiveSceneReplacement();
    Scene *persistent = manager.GetRuntimePersistentScene();
    assert(persistent && persistent->FindByID(persistentObjectId));

    Scene *sceneC = manager.CreateScene("WebWorldC");
    SceneEnvironmentSettings environmentC = sceneC->GetEnvironment();
    environmentC.skyExposure = 3.75f;
    sceneC->SetEnvironment(environmentC);
    GameObject *cameraObjectC = sceneC->CreateGameObject("C Camera");
    cameraObjectC->GetTransform()->SetPosition(glm::vec3(0.0f, 0.0f, -5.0f));
    Camera *cameraC = cameraObjectC->AddComponent<Camera>();
    sceneC->SetMainCamera(cameraC);
    manager.SetActiveScene(sceneC);
    manager.UnloadScene(sceneA);
    assert(resolveWebCamera() == cameraC);
    assert(manager.GetActiveScene()->GetEnvironment().skyExposure == 3.75f);
    RequireIds(ExtractObjectIds(extractor, world, resolveWebCamera()), {persistentObjectId});
    assert(manager.GetActiveMeshRenderers().size() == 1);
    assert(manager.GetActiveLights().size() == 1);

    manager.Stop();
    assert(manager.GetRuntimePersistentScene() == nullptr);
    assert(manager.GetActiveMeshRenderers().empty());
    assert(manager.GetActiveLights().empty());
    assert(extractor.ExtractCameraFrame(world, resolveWebCamera()) == 0);
    const auto emptyFrame = world.Acquire();
    assert(emptyFrame && emptyFrame->DrawCalls().drawCalls.empty());

    manager.UnloadAllScenes();
    return 0;
}
