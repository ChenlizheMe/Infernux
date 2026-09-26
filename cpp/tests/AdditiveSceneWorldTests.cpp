#include <function/editor/SelectionOutline.h>
#include <function/scene/Camera.h>
#include <function/scene/Component.h>
#include <function/scene/GameObject.h>
#include <function/scene/Light.h>
#include <function/scene/Scene.h>
#include <function/scene/SceneManager.h>
#include <function/scene/Transform.h>
#include <function/scene/TransformECSStore.h>

#include <cassert>

using namespace infernux;

namespace
{
struct ProbeState
{
    int awake = 0;
    int start = 0;
    int fixed = 0;
    int update = 0;
    int late = 0;
    int destroy = 0;
};

class AdditiveLifecycleProbe final : public Component
{
  public:
    explicit AdditiveLifecycleProbe(ProbeState &state) : m_state(state)
    {
    }

    const char *GetTypeName() const override
    {
        return "AdditiveLifecycleProbe";
    }

    std::string GetConstraintTypeId() const override
    {
        return "test:AdditiveLifecycleProbe";
    }

    const ComponentTypeConstraints &GetComponentTypeConstraints() const override
    {
        static const ComponentTypeConstraints constraints{};
        return constraints;
    }

    bool WantsRuntimeUpdate() const override
    {
        return true;
    }

    bool WantsRuntimeFixedUpdate() const override
    {
        return true;
    }

    bool WantsRuntimeLateUpdate() const override
    {
        return true;
    }

    void Awake() override
    {
        ++m_state.awake;
    }

    void Start() override
    {
        ++m_state.start;
    }

    void FixedUpdate(float) override
    {
        ++m_state.fixed;
    }

    void Update(float) override
    {
        ++m_state.update;
        Transform *transform = GetTransform();
        transform->SetPosition(transform->GetPosition() + glm::vec3(1.0f, 0.0f, 0.0f));
    }

    void LateUpdate(float) override
    {
        ++m_state.late;
    }

    void OnDestroy() override
    {
        ++m_state.destroy;
    }

  private:
    ProbeState &m_state;
};

void RunFrame(SceneManager &manager, float deltaTime)
{
    auto &transforms = TransformECSStore::Instance();
    transforms.BeginFrameCache();
    manager.Update(deltaTime);
    manager.LateUpdate(deltaTime);
    if (transforms.EndFrameCache())
        manager.PublishPhysicsTransformsToRenderer();
    manager.PublishAuthoredTransformsToPhysics();
    manager.EndFrame();
}
} // namespace

int main()
{
    SceneManager &manager = SceneManager::Instance();
    manager.Stop();
    manager.UnloadAllScenes();
    manager.SetFixedTimeStep(0.02f);
    manager.SetMaxFixedDeltaTime(1.0f / 3.0f);
    manager.SetTimeScale(1.0f);

    // Loading the same authored document beside its source must publish a
    // distinct runtime identity graph. The commit token exposes the one
    // authoritative remap used by Python ObjectReference fields.
    Scene *identitySource = manager.CreateScene("IdentitySource");
    GameObject *sourceObject = identitySource->CreateGameObject("SharedDocumentObject");
    sourceObject->AddComponent<Light>();
    const uint64_t documentObjectId = sourceObject->GetID();
    const auto identityDocument = identitySource->SerializeDocument();
    Scene *identityCopy = manager.CreateScene("IdentityCopy");
    auto identityCommit = identityCopy->CommitDocumentRetainingCurrentWorld(identityDocument);
    assert(identityCommit);
    const auto &objectIdRemap = identityCommit->GetObjectIdRemap();
    const auto remapped = objectIdRemap.find(documentObjectId);
    assert(remapped != objectIdRemap.end());
    assert(remapped->second != documentObjectId);
    assert(identitySource->FindByID(documentObjectId) == sourceObject);
    assert(identityCopy->FindByID(documentObjectId) == nullptr);
    assert(identityCopy->FindByID(remapped->second) != nullptr);
    identityCommit->Finalize();
    manager.UnloadScene(identityCopy);
    assert(manager.GetSceneCount() == 1);
    assert(manager.GetActiveLights().size() == 1);

    for (int cycle = 0; cycle < 32; ++cycle) {
        Scene *copy = manager.CreateScene("IdentitySoakCopy");
        auto commit = copy->CommitDocumentRetainingCurrentWorld(identityDocument);
        assert(commit);
        commit->Finalize();
        assert(manager.GetActiveLights().size() == 2);
        manager.UnloadScene(copy);
        assert(manager.GetSceneCount() == 1);
        assert(manager.GetActiveLights().size() == 1);
        assert(identitySource->FindByID(documentObjectId) == sourceObject);
    }
    manager.UnloadAllScenes();

    ProbeState stateA;
    ProbeState stateB;
    Scene *sceneA = manager.CreateScene("AdditiveA");
    Scene *sceneB = manager.CreateScene("AdditiveB");
    manager.SetActiveScene(sceneA);

    GameObject *objectA = sceneA->CreateGameObject("ObjectA");
    GameObject *objectB = sceneB->CreateGameObject("ObjectB");
    GameObject *childB = sceneB->CreateGameObject("SelectedDescendantB");
    childB->SetParent(objectB);
    GameObject *hiddenB = sceneB->CreateGameObject("InactiveDescendantB");
    hiddenB->SetParent(objectB);
    hiddenB->SetActive(false);
    const uint64_t objectAId = objectA->GetID();
    const uint64_t objectBId = objectB->GetID();
    const uint64_t childBId = childB->GetID();
    const std::vector<uint64_t> selectionIds{objectAId, objectBId, childBId, objectBId, 0};
    const std::vector<uint64_t> expectedOutline{objectAId, objectBId, childBId};
    assert(ExpandSelectionOutlineIds(manager, selectionIds) == expectedOutline);
    assert(manager.GetActiveScene() == sceneA); // Selection must not change authoring ownership.
    manager.SetActiveScene(sceneB);
    assert(ExpandSelectionOutlineIds(manager, selectionIds) == expectedOutline);
    manager.SetActiveScene(sceneA);
    assert(ExpandSelectionOutlineIds(manager, {objectBId}) == std::vector<uint64_t>({objectBId, childBId}));
    assert(ExpandSelectionOutlineIds(manager, {}).empty());
    objectA->AddComponent<AdditiveLifecycleProbe>(stateA);
    objectB->AddComponent<AdditiveLifecycleProbe>(stateB);
    objectA->AddComponent<Light>();
    objectB->AddComponent<Light>();
    Camera *cameraA = objectA->AddComponent<Camera>();
    Camera *cameraB = objectB->AddComponent<Camera>();
    cameraA->SetDepth(5.0f);
    cameraB->SetDepth(-2.0f);

    assert(manager.GetSceneCount() == 2);
    assert(manager.GetActiveLights().size() == 2);
    assert(manager.IsRuntimeScene(sceneA));
    assert(manager.IsRuntimeScene(sceneB));
    const auto worldCameras = sceneA->GetActiveGameCameras(nullptr);
    assert(worldCameras.size() == 2);
    assert(worldCameras[0] == cameraB);
    assert(worldCameras[1] == cameraA);

    manager.Play();
    assert(stateA.awake == 1 && stateB.awake == 1);
    assert(stateA.start == 1 && stateB.start == 1);

    // Half a fixed interval is retained across an active-scene switch. Both
    // loaded Scenes keep updating, and both use the same deferred Transform
    // cache even though only one is active for authoring/camera policy.
    RunFrame(manager, 0.01f);
    assert(objectA->GetTransform()->GetPosition().x == 1.0f);
    assert(objectB->GetTransform()->GetPosition().x == 1.0f);
    assert(stateA.fixed == 0 && stateB.fixed == 0);
    manager.SetActiveScene(sceneB);
    RunFrame(manager, 0.01f);
    assert(objectA->GetTransform()->GetPosition().x == 2.0f);
    assert(objectB->GetTransform()->GetPosition().x == 2.0f);
    assert(stateA.fixed == 1 && stateB.fixed == 1);
    assert(stateA.update == 2 && stateB.update == 2);
    assert(stateA.late == 2 && stateB.late == 2);

    // Unloading the active additive Scene promotes another loaded Scene and
    // removes only the unloaded Scene's lifecycle and global registry state.
    manager.UnloadScene(sceneB);
    assert(ExpandSelectionOutlineIds(manager, selectionIds) == std::vector<uint64_t>({objectAId}));
    assert(manager.GetActiveScene() == sceneA);
    assert(manager.GetSceneCount() == 1);
    assert(manager.GetActiveLights().size() == 1);
    assert(stateB.destroy == 1);
    const int updatesB = stateB.update;
    RunFrame(manager, 0.02f);
    assert(stateA.update == 3);
    assert(stateB.update == updatesB);

    manager.Stop();
    manager.UnloadAllScenes();
    assert(stateA.destroy == 1);
    assert(manager.GetActiveLights().empty());
    return 0;
}
