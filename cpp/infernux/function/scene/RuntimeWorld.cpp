#include "RuntimeWorld.h"
#include "Scene.h"
#include "physics/PhysicsDomain.h"
#include "physics/PhysicsECSStore.h"
#include <atomic>

namespace infernux
{
namespace
{
std::atomic<uint64_t> s_nextRuntimeWorldId{1};
thread_local RuntimeWorld *s_currentWorld = nullptr;
} // namespace

RuntimeWorld::SceneCollection::SceneCollection() = default;
RuntimeWorld::SceneCollection::~SceneCollection()
{
    Clear();
}

void RuntimeWorld::SceneCollection::Clear()
{
    auto retiringPersistent = std::move(persistent);
    std::vector<std::unique_ptr<Scene>> retiringScenes;
    retiringScenes.swap(loaded);
    active = nullptr;
    loadedSet.clear();
    pendingPersistentRootIds.clear();
    pendingPersistentRootIdSet.clear();
    retiringPersistent.reset();
    retiringScenes.clear();
}
std::shared_ptr<RuntimeWorldHandle> RuntimeWorld::ReserveHandle()
{
    return std::make_shared<RuntimeWorldHandle>(s_nextRuntimeWorldId.fetch_add(1, std::memory_order_relaxed));
}

RuntimeWorld::RuntimeWorld(std::shared_ptr<RuntimeWorldHandle> reserved)
    : handle(reserved ? std::move(reserved) : ReserveHandle()), id(handle->id)
{
    // Input must outlive every World, including the SceneManager default World.
    (void)InputManager::Instance();
    physics = std::make_unique<PhysicsDomain>(*this);
    PhysicsECSStore::Instance().RegisterRuntimeWorld(*this);
}
RuntimeWorld::~RuntimeWorld()
{
    const WorldExecutionScope worldScope(this);
    scenes.Clear();
    PhysicsECSStore::Instance().RetireRuntimeWorld(*this);
    PhysicsWorld::Instance().RetireDomain(*physics);
    physics.reset();
    InputManager::Instance().RetireWorldInput(id);
}

WorldExecutionScope::WorldExecutionScope(RuntimeWorld *world) noexcept : m_previous(s_currentWorld), m_world(world)
{
    if (world) {
        ++world->executionScopes;
        m_randomScope.emplace(world->random);
    }
    if (world && world != m_previous) {
        m_inputScope.emplace(world->frame.input, world->id);
        s_currentWorld = world;
    }
}

WorldExecutionScope::~WorldExecutionScope()
{
    s_currentWorld = m_previous;
    if (m_world)
        --m_world->executionScopes;
}

RuntimeWorld *WorldExecutionScope::Current() noexcept
{
    return s_currentWorld;
}
} // namespace infernux
