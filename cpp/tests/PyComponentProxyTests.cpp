#include <function/scene/BoxCollider.h>
#include <function/scene/GameObject.h>
#include <function/scene/PyComponentProxy.h>
#include <function/scene/Scene.h>
#include <function/scene/SceneManager.h>
#include <function/scene/physics/PhysicsWorld.h>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <functional>
#include <iostream>
#include <new>
#include <pybind11/embed.h>

namespace py = pybind11;

namespace
{
class TeardownQueryProbe final : public infernux::Component
{
  public:
    explicit TeardownQueryProbe(std::function<void(infernux::GameObject *)> check) : m_check(std::move(check)) {}
    void OnDestroy() override { m_check(GetGameObject()); }
    const char *GetTypeName() const override { return "TeardownQueryProbe"; }
    std::string GetConstraintTypeId() const override { return "test:teardown-query-probe"; }
  private:
    std::function<void(infernux::GameObject *)> m_check;
};

void TestSceneTeardownQueries()
{
    auto &manager = infernux::SceneManager::Instance();
    for (int mode = 0; mode < 5; ++mode) {
        auto *scene = mode == 4 ? manager.CreatePreviewScene("TeardownQueries")
                                : manager.CreateScene("TeardownQueries");
        const auto worldId = scene->GetWorldId();
        const auto emptyDocument = scene->SerializeDocument();
        std::shared_ptr<infernux::SceneCommitToken> token;
        if (mode == 3)
            token = scene->CommitDocumentRetainingCurrentWorld(emptyDocument);
        auto *first = scene->CreateGameObject("EarlierRoot");
        const auto firstId = first->GetID();
        auto *parent = scene->CreateGameObject("LaterRoot");
        auto *child = scene->CreateGameObject("LaterChild");
        child->SetParent(parent, false);
        int callbacks = 0;
        const auto check = [&](infernux::GameObject *owner) {
            ++callbacks;
            assert(scene->GetRootObjects().empty());
            assert(scene->GetAllObjects().empty());
            assert(scene->FindByID(firstId) == nullptr);
            assert(scene->FindByID(owner->GetID()) == owner);
            if (owner->GetParent())
                assert(owner->GetParent()->GetChildren().empty());
            if (mode == 0 || mode == 1 || mode == 4)
                assert(manager.GetSceneByWorldId(worldId) == nullptr);
            bool rejected = false;
            try {
                scene->CreateGameObject("CannotResurrectRetiringWorld");
            } catch (const std::logic_error &) {
                rejected = true;
            }
            assert(rejected);
            rejected = false;
            try {
                owner->AddComponent<infernux::BoxCollider>();
            } catch (const std::logic_error &) {
                rejected = true;
            }
            assert(rejected);
        };
        assert(parent->AddExistingComponent(std::make_unique<TeardownQueryProbe>(check)) != nullptr);
        assert(child->AddExistingComponent(std::make_unique<TeardownQueryProbe>(check)) != nullptr);
        if (mode == 0)
            manager.UnloadScene(scene);
        else if (mode == 1)
            manager.UnloadAllScenes();
        else if (mode == 2)
            assert(scene->DeserializeDocument(emptyDocument));
        else if (mode == 3)
            assert(token && token->Rollback());
        else {
            // Preview scenes follow the same detached ownership contract.
            manager.ClosePreviewScene(scene);
        }
        std::cout << "teardown mode=" << mode << " callbacks=" << callbacks << std::endl;
        assert(callbacks == 2);
        if (mode == 2 || mode == 3) {
            assert(scene->GetAllObjects().empty());
            assert(scene->CreateGameObject("NewWorldIsWritable") != nullptr);
            manager.UnloadScene(scene);
        }
    }
}

void TestSubtreeReplacementTeardown()
{
    infernux::Scene scene("SubtreeReplacement");
    auto *parent = scene.CreateGameObject("Parent");
    const auto emptyDocument = parent->SerializeDocument();
    assert(parent->AddComponent<infernux::BoxCollider>() != nullptr);
    assert(parent->GetComponentsInExecutionOrder().size() == 1);
    auto *first = scene.CreateGameObject("EarlierChild");
    first->SetParent(parent, false);
    auto *second = scene.CreateGameObject("LaterChild");
    second->SetParent(parent, false);
    int callbacks = 0;
    assert(second->AddExistingComponent(std::make_unique<TeardownQueryProbe>([&](auto *owner) {
        ++callbacks;
        assert(owner->GetParent() == parent);
        assert(parent->GetChildren().empty());
        assert(parent->GetAllComponents().empty());
        assert(parent->GetComponentsInExecutionOrder().empty());
        const auto objects = scene.GetAllObjects();
        assert(objects.size() == 1 && objects.front() == parent);
    })) != nullptr);
    assert(parent->DeserializeDocument(emptyDocument));
    assert(callbacks == 1);
    assert(parent->GetChildren().empty());
    assert(parent->AddComponent<infernux::BoxCollider>() != nullptr);
    std::cout << "subtree replacement teardown callbacks=" << callbacks << std::endl;
}

void TestRetainedPhysicsTargets()
{
    // Force exact address AND serialized-ID reuse. The retained value must not
    // resolve the replacement, even if an allocator would usually avoid reuse.
    alignas(infernux::BoxCollider) unsigned char colliderStorage[sizeof(infernux::BoxCollider)];
    auto *first = new (colliderStorage) infernux::BoxCollider();
    const py::object firstWrapper = py::cast(first, py::return_value_policy::reference);
    const uint64_t id = first->GetComponentID();
    const infernux::PhysicsTargetReference retiredCollider(first);
    assert(retiredCollider.GetCollider() == first);
    first->~BoxCollider();
    assert(!firstWrapper.attr("__bool__")().cast<bool>());
    bool argumentRejected = false;
    try {
        (void)firstWrapper.cast<infernux::Component *>();
    } catch (const infernux::InvalidNativeObjectError &) {
        argumentRejected = true;
    }
    assert(argumentRejected);
    auto *second = new (colliderStorage) infernux::BoxCollider();
    second->SetComponentID(id);
    const py::object secondWrapper = py::cast(second, py::return_value_policy::reference);
    assert(!firstWrapper.is(secondWrapper));
    assert(!firstWrapper.attr("__bool__")().cast<bool>());
    assert(secondWrapper.attr("__bool__")().cast<bool>());
    assert(retiredCollider.GetCollider() == nullptr);
    const infernux::PhysicsTargetReference replacementCollider(second);
    assert(replacementCollider.GetCollider() == second);
    second->~BoxCollider();
    assert(replacementCollider.GetCollider() == nullptr);

    alignas(infernux::GameObject) unsigned char ownerStorage[sizeof(infernux::GameObject)];
    auto *owner = new (ownerStorage) infernux::GameObject("NativeRecordOwner");
    auto *collider = owner->AddComponent<infernux::BoxCollider>();
    const uint64_t transformId = owner->GetTransform()->GetComponentID();
    infernux::RaycastHit hit;
    hit.target = infernux::PhysicsTargetReference(collider);
    hit.distance = 7.5f;
    // Conversion executes getters in the extension, while the object and its
    // registry live in this executable. It must preserve the publishing owner.
    const py::object retained = py::cast(hit);
    const py::object retainedOwner = retained.attr("game_object");
    const py::object savedMethod = retainedOwner.attr("get_transform");
    const py::object retainedTransform = savedMethod();
    assert(retainedOwner.cast<infernux::GameObject *>() == owner);
    {
        // Engine.exit releases the GIL before native cleanup. Borrower state
        // and weakref destruction must both reacquire it where necessary.
        py::gil_scoped_release release;
        owner->~GameObject();
    }
    assert(!retainedOwner.attr("__bool__")().cast<bool>());
    assert(!retainedTransform.attr("__bool__")().cast<bool>());
    bool methodRejected = false;
    try {
        (void)savedMethod();
    } catch (const py::error_already_set &error) {
        methodRejected = error.matches(py::module_::import("infernux.lib").attr("InvalidNativeObjectError"));
    }
    assert(methodRejected);
    assert(hit.target.GetGameObject() == nullptr);
    assert(retained.attr("game_object").is_none());
    auto *replacement = new (ownerStorage) infernux::GameObject("ReplacementRecordOwner");
    replacement->GetTransform()->SetComponentID(transformId);
    replacement->AddComponent<infernux::BoxCollider>();
    const py::object replacementWrapper = py::cast(replacement, py::return_value_policy::reference);
    assert(!replacementWrapper.is(retainedOwner));
    assert(replacementWrapper.attr("__bool__")().cast<bool>());
    assert(!retainedOwner.attr("__bool__")().cast<bool>());
    assert(!retainedTransform.attr("__bool__")().cast<bool>());
    assert(hit.target.GetGameObject() == nullptr);
    assert(retained.attr("game_object").is_none());
    assert(retained.attr("collider").is_none());
    assert(retained.attr("distance").cast<float>() == 7.5f);
    replacement->~GameObject();
}

class NativeUpdateProbe final : public infernux::Component
{
  public:
    explicit NativeUpdateProbe(int &updates) : m_updates(updates)
    {
    }

    void Update(float) override
    {
        ++m_updates;
    }

    [[nodiscard]] const char *GetTypeName() const override
    {
        return "NativeUpdateProbe";
    }

    [[nodiscard]] std::string GetConstraintTypeId() const override
    {
        return "test:native-update-probe";
    }

    [[nodiscard]] bool WantsRuntimeUpdate() const override
    {
        return true;
    }

  private:
    int &m_updates;
};
} // namespace

int main()
{
    py::scoped_interpreter interpreter{};
    py::exec(R"PY(
from infernux.components import InxComponent
class CollisionEnterOnlyProbe(InxComponent):
    _uses_component_data_store = False

    def on_collision_enter(self, collision):
        pass
)PY");

    TestRetainedPhysicsTargets();
    TestSceneTeardownQueries();
    TestSubtreeReplacementTeardown();

    const py::object collisionProbe = py::globals()["CollisionEnterOnlyProbe"]();
    infernux::PyComponentProxy collisionProxy(collisionProbe);
    assert(collisionProxy.WantsPhysicsCallbacks());
    assert(collisionProxy.WantsCollisionEnterCallbacks());
    assert(!collisionProxy.WantsCollisionStayCallbacks());
    assert(!collisionProxy.WantsCollisionExitCallbacks());
    assert(!collisionProxy.WantsTriggerEnterCallbacks());
    assert(!collisionProxy.WantsTriggerStayCallbacks());
    assert(!collisionProxy.WantsTriggerExitCallbacks());

    // Script publication can add or remove edit-mode execution without
    // replacing the native proxy. Both native gates and the Python mirror
    // must describe the refreshed class, including a restored old revision.
    for (const bool editMode : {false, true, false}) {
        collisionProbe.attr("__class__").attr("_execute_in_edit_mode_") = py::bool_(editMode);
        collisionProxy.RefreshPythonLifecycleDispatch();
        assert(collisionProxy.WantsEditModeLifecycle() == editMode);
        assert(collisionProxy.WantsEditModeUpdate() == editMode);
        assert(collisionProbe.attr("_execute_in_edit_mode").cast<bool>() == editMode);
        assert(collisionProxy.WantsCollisionEnterCallbacks());
    }

    // Python work availability controls only Python callbacks. Native
    // components remain on the native Scene traversal.
    auto &sceneManager = infernux::SceneManager::Instance();
    infernux::Scene *scene = sceneManager.CreateScene("RuntimeSchedulerOwner");
    infernux::GameObject *owner = scene->CreateGameObject("PythonOwner");
    bool missingSchedulerRejected = false;
    try {
        owner->AddExistingComponent(std::make_unique<infernux::PyComponentProxy>(collisionProbe));
    } catch (const std::runtime_error &) {
        missingSchedulerRejected = true;
    }
    assert(missingSchedulerRejected);

    int nativeUpdates = 0;
    assert(owner->AddExistingComponent(std::make_unique<NativeUpdateProbe>(nativeUpdates)) != nullptr);

    int runtimeUpdates = 0;
    sceneManager.SetRuntimeLifecycleCallbacks([] {}, [](float) {}, [&runtimeUpdates](float) { ++runtimeUpdates; },
                                              [](float) {}, [](float) {}, [] {});
    sceneManager.SetRuntimeLifecycleWorkAvailable(false);
    sceneManager.Play();
    sceneManager.Update(1.0f / 60.0f);
    sceneManager.LateUpdate(1.0f / 60.0f);
    sceneManager.EndFrame();
    assert(runtimeUpdates == 0);
    assert(nativeUpdates == 1);
    sceneManager.Stop();

    // The native frame driver consumes the published phase plan. A scheduler
    // may remain installed while a structural rebuild produces an empty phase;
    // that phase must not cross into Python until a newer plan enables it.
    sceneManager.SetRuntimeLifecycleWorkAvailable(true);
    sceneManager.SetRuntimeLifecyclePlan(1, 0, 0, 0);
    sceneManager.Play();
    sceneManager.Update(1.0f / 60.0f);
    sceneManager.LateUpdate(1.0f / 60.0f);
    sceneManager.EndFrame();
    assert(runtimeUpdates == 0);
    sceneManager.SetRuntimeLifecyclePlan(2, 0, 1, 0);
    sceneManager.Update(1.0f / 60.0f);
    sceneManager.LateUpdate(1.0f / 60.0f);
    sceneManager.EndFrame();
    assert(runtimeUpdates == 1);
    sceneManager.Stop();
    sceneManager.ClearRuntimeLifecycleCallbacks();
    sceneManager.UnloadAllScenes();
    return 0;
}
