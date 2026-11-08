#pragma once

#include "core/FoundationApi.h"

#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>

namespace infernux
{
// Process-local names, shared by all shaders and material instances. Entries
// never move or disappear; documents continue to serialize their string names.
class ShaderPropertyIds
{
  public:
    static int32_t Get(const std::string &name)
    {
        auto &registry = Registry();
        const std::lock_guard lock(registry.mutex);
        const auto found = registry.ids.find(name);
        if (found != registry.ids.end())
            return found->second;
        registry.names.push_back(name);
        const auto id = static_cast<int32_t>(registry.names.size());
        registry.ids.emplace(name, id);
        return id;
    }

    static const std::string *Find(int32_t id)
    {
        auto &registry = Registry();
        const std::lock_guard lock(registry.mutex);
        return id > 0 && static_cast<size_t>(id) <= registry.names.size() ? &registry.names[id - 1] : nullptr;
    }

  private:
    struct Names
    {
        std::mutex mutex;
        std::unordered_map<std::string, int32_t> ids;
        std::deque<std::string> names;
    };

    INFERNUX_FOUNDATION_API static Names &Registry();
};
} // namespace infernux
