#include "Random.h"
#include <chrono>

namespace infernux
{
namespace
{
thread_local std::shared_ptr<Random::Storage> s_defaultStorage;
thread_local Random::Storage *s_scopedStorage = nullptr;
} // namespace

Random::Storage::Storage()
    : state(Seed(static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())))
{
}

std::shared_ptr<Random::Storage> Random::ToolStorage()
{
    static auto storage = std::make_shared<Storage>();
    return storage;
}

void Random::SetThreadDefaultStorage(std::shared_ptr<Storage> storage)
{
    s_defaultStorage = std::move(storage);
}

Random::StorageScope::StorageScope(std::shared_ptr<Storage> storage) noexcept
    : m_storage(std::move(storage)), m_previous(s_scopedStorage)
{
    s_scopedStorage = m_storage.get();
}

Random::StorageScope::~StorageScope()
{
    s_scopedStorage = m_previous;
}

Random::Snapshot Random::CaptureStorage(Storage &storage)
{
    const std::lock_guard<std::mutex> lock(storage.mutex);
    return {storage.state, storage.scramble};
}

void Random::RestoreStorage(Storage &storage, const Snapshot &snapshot)
{
    const std::lock_guard<std::mutex> lock(storage.mutex);
    storage.state = snapshot.state;
    storage.scramble = snapshot.scramble;
}

Random::Storage &Random::Shared()
{
    if (s_scopedStorage)
        return *s_scopedStorage;
    if (s_defaultStorage)
        return *s_defaultStorage;
    return *ToolStorage();
}
} // namespace infernux
