#include "RuntimeWorld.h"
#include "Scene.h"
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
RuntimeWorld::RuntimeWorld() : id(s_nextRuntimeWorldId.fetch_add(1, std::memory_order_relaxed))
{
    // Input must outlive every World, including the SceneManager default World.
    (void)InputManager::Instance();
}
RuntimeWorld::~RuntimeWorld()
{
    scenes.Clear();
    InputManager::Instance().RetireWorldInput(id);
}

WorldExecutionScope::WorldExecutionScope(RuntimeWorld *world) noexcept : m_previous(s_currentWorld)
{
    if (world && world != m_previous) {
        m_inputScope.emplace(world->frame.input, world->id);
        s_currentWorld = world;
    }
}

WorldExecutionScope::~WorldExecutionScope()
{
    s_currentWorld = m_previous;
}

RuntimeWorld *WorldExecutionScope::Current() noexcept
{
    return s_currentWorld;
}
} // namespace infernux
