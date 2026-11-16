#include "RenderGraphSchedule.h"

#include <algorithm>
#include <queue>
#include <unordered_map>

namespace infernux
{

RenderGraphSchedule CompileRenderGraphSchedule(const std::vector<GraphSchedulePass> &passes, GraphResourceHandle output,
                                               const std::vector<std::pair<uint32_t, uint32_t>> &explicitDependencies)
{
    RenderGraphSchedule result;
    const auto count = static_cast<uint32_t>(passes.size());
    result.dependencies.resize(count);
    result.retention.resize(count, GraphPassRetention::Unreachable);
    result.references.resize(count);
    std::unordered_map<GraphResourceHandle, uint32_t, GraphResourceHandleHash> producers;
    for (uint32_t id = 0; id < count; ++id)
        for (const auto write : passes[id].writes)
            producers.emplace(write, id);

    std::queue<uint32_t> retained;
    const auto retain = [&](uint32_t id, GraphPassRetention reason) {
        if (result.retention[id] == GraphPassRetention::Unreachable) {
            result.retention[id] = reason;
            retained.push(id);
        } else if (reason == GraphPassRetention::GraphOutput)
            result.retention[id] = reason;
        ++result.references[id];
    };
    const auto root = producers.find(output);
    if (root != producers.end())
        retain(root->second, GraphPassRetention::GraphOutput);
    for (uint32_t id = 0; id < count; ++id) {
        if (passes[id].sideEffect)
            retain(id, GraphPassRetention::SideEffect);
        else if (passes[id].externalWrite)
            retain(id, GraphPassRetention::ExternalWrite);
    }
    while (!retained.empty()) {
        const auto id = retained.front();
        retained.pop();
        for (const auto &read : passes[id].reads) {
            const auto producer = producers.find(read.resource);
            if (read.retainProducer && producer != producers.end() && producer->second != id)
                retain(producer->second, GraphPassRetention::Dependency);
        }
    }

    const auto active = [&](uint32_t id) { return result.retention[id] != GraphPassRetention::Unreachable; };
    std::unordered_map<GraphResourceHandle, std::vector<uint32_t>, GraphResourceHandleHash> readers;
    uint32_t activeCount = 0;
    for (uint32_t id = 0; id < count; ++id) {
        if (!active(id))
            continue;
        ++activeCount;
        for (const auto &read : passes[id].reads) {
            readers[read.resource].push_back(id);
            const auto producer = producers.find(read.resource);
            if (producer != producers.end() && producer->second != id && active(producer->second))
                result.dependencies[id].push_back(producer->second);
        }
    }
    for (const auto &[version, users] : readers) {
        if (version.version == UINT32_MAX)
            continue;
        auto next = version;
        ++next.version;
        const auto writer = producers.find(next);
        if (writer == producers.end() || !active(writer->second))
            continue;
        for (const auto user : users)
            if (user != writer->second)
                result.dependencies[writer->second].push_back(user);
    }
    for (const auto &[later, earlier] : explicitDependencies)
        if (active(later) && active(earlier))
            result.dependencies[later].push_back(earlier);

    std::vector<std::vector<uint32_t>> successors(count);
    std::vector<uint32_t> remaining(count);
    std::priority_queue<uint32_t, std::vector<uint32_t>, std::greater<uint32_t>> ready;
    for (uint32_t id = 0; id < count; ++id) {
        auto &dependencies = result.dependencies[id];
        std::sort(dependencies.begin(), dependencies.end());
        dependencies.erase(std::unique(dependencies.begin(), dependencies.end()), dependencies.end());
        remaining[id] = static_cast<uint32_t>(dependencies.size());
        for (const auto dependency : dependencies)
            successors[dependency].push_back(id);
        if (active(id) && remaining[id] == 0)
            ready.push(id);
    }
    while (!ready.empty()) {
        const auto id = ready.top();
        ready.pop();
        result.order.push_back(id);
        for (const auto next : successors[id])
            if (--remaining[next] == 0)
                ready.push(next);
    }
    if (result.order.size() != activeCount) {
        result.order.clear();
        for (uint32_t id = 0; id < count; ++id)
            if (active(id) && remaining[id] != 0)
                result.unresolved.push_back(id);
    }
    return result;
}

} // namespace infernux
