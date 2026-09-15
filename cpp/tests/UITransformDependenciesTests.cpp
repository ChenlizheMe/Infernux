#include <cassert>
#include <function/scene/Scene.h>
#include <function/scene/SceneManager.h>
#include <function/scene/UITransformDependencies.h>

using namespace infernux;

int main()
{
    auto &manager = SceneManager::Instance();
    auto *scene = manager.CreateScene("UI dependencies");
    auto *object = scene->CreateGameObject("World UI");
    auto &store = TransformECSStore::Instance();
    UITransformDependencies world({}, {object});
    UITransformDependencies screen({object}, {});
    auto worldRevision = world.Poll();
    auto screenRevision = screen.Poll();
    assert(world.GetChangedEntries() == std::vector<uint32_t>{0});
    assert(screen.GetChangedEntries() == std::vector<uint32_t>{0});

    store.BeginFrameCache();
    const auto serial = store.GetGlobalTransformSerial();
    for (int index = 1; index <= 4; ++index) {
        object->GetTransform()->SetPosition(float(index), 2.0f, 3.0f);
        // World overrides have not yet advanced the committed transform serial.
        assert(store.GetGlobalTransformSerial() == serial);
        const auto current = world.Poll();
        assert(current != worldRevision);
        assert(world.GetChangedEntries() == std::vector<uint32_t>{0});
        assert(world.Poll() == current);
        assert(world.GetChangedEntries().empty());
        worldRevision = current;
        assert(screen.Poll() == screenRevision); // Local pose is still uncommitted.
    }
    (void)store.EndFrameCache();
    assert(world.Poll() == worldRevision);
    assert(screen.Poll() != screenRevision);
    object->GetTransform()->SetPosition(0.0f, 0.0f, 0.0f);
    object->SetLayer(31);
    auto hit = world.ProjectWorldRay({0.25f, -0.1f, 2.0f}, {0, 0, -1}, 0x80000000u)[0];
    assert(std::abs(hit[0] - 0.25) < 1e-6 && std::abs(hit[1] + 0.1) < 1e-6 && hit[2] == 2.0);
    hit = world.ProjectWorldRay({0, 0, 2}, {1, 0, 0}, 0xffffffffu)[0];
    assert(std::isnan(hit[0]));
    hit = world.ProjectWorldRay({0, 0, 2}, {0, 0, 1}, 0xffffffffu)[0];
    assert(std::isnan(hit[0]));
    hit = world.ProjectWorldRay({0, 0, 2}, {0, 0, -1}, 1u)[0];
    assert(std::isnan(hit[0]));
    store.BeginFrameCache();
    for (int index = 1; index <= 4; ++index) {
        object->GetTransform()->SetPosition(float(index), 0, 0);
        hit = world.ProjectWorldRay({0, 0, 2}, {0, 0, -1}, 0xffffffffu)[0];
        assert(hit[0] == -double(index)); // Unbounded, live, pre-commit drag coordinates.
    }
    (void)store.EndFrameCache();
    auto *sibling = scene->CreateGameObject("Unrelated UI");
    UITransformDependencies mixed({object, sibling}, {object, sibling});
    mixed.Poll();
    assert((mixed.GetChangedEntries() == std::vector<uint32_t>{0, 1, 2, 3}));
    auto revision = mixed.Poll();
    assert(mixed.GetChangedEntries().empty());
    sibling->GetTransform()->SetPosition(3, 4, 5);
    assert(mixed.Poll() != revision);
    assert((mixed.GetChangedEntries() == std::vector<uint32_t>{1, 3}));
    revision = mixed.Poll();
    // Layers affect only world geometry, even without a changed pose serial.
    sibling->SetLayer(7);
    assert(mixed.Poll() != revision);
    assert(mixed.GetChangedEntries() == std::vector<uint32_t>{3});
    sibling->GetTransform()->SetPosition(3, 4, 9);
    mixed.Poll();
    assert(mixed.GetChangedEntries() == std::vector<uint32_t>{3});
    sibling->GetTransform()->SetLocalScale(2, 3, 4);
    revision = mixed.Poll();
    assert(mixed.GetChangedEntries().empty());
    assert(mixed.Poll() == revision);
    manager.UnloadAllScenes();
    bool stale = false;
    try {
        world.Poll();
    } catch (const std::runtime_error &) {
        stale = true;
    }
    assert(stale);
    return 0;
}
