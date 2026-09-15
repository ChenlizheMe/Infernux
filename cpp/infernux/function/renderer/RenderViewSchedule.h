#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace infernux
{
// Resource identities are live owners retained by the camera/graph. No content
// hashes or frame-old substitutions participate in the scheduling contract.
struct RenderViewAccess
{
    std::vector<const void *> reads;
    std::vector<const void *> writes;
};

struct RenderViewSchedule
{
    std::vector<uint32_t> order;
    std::vector<std::vector<uint32_t>> predecessors;

    static RenderViewSchedule Build(const std::vector<RenderViewAccess> &views)
    {
        RenderViewSchedule result;
        result.predecessors.resize(views.size());
        std::unordered_map<const void *, uint32_t> lastWriter;
        const auto edge = [&](uint32_t before, uint32_t after) {
            if (before == after)
                throw std::invalid_argument("RenderTexture feedback requires distinct input/output resources");
            auto &dependencies = result.predecessors[after];
            if (std::find(dependencies.begin(), dependencies.end(), before) == dependencies.end())
                dependencies.push_back(before);
        };
        for (uint32_t i = 0; i < views.size(); ++i) {
            for (const void *resource : views[i].writes) {
                if (std::find(views[i].reads.begin(), views[i].reads.end(), resource) != views[i].reads.end())
                    throw std::invalid_argument("RenderTexture feedback requires distinct input/output resources");
                const auto previous = lastWriter.find(resource);
                // Cameras sharing an output preserve author order (depth).
                if (previous != lastWriter.end() && previous->second != i)
                    edge(previous->second, i);
                lastWriter[resource] = i;
            }
        }
        for (uint32_t i = 0; i < views.size(); ++i) {
            for (const void *resource : views[i].reads) {
                const auto producer = lastWriter.find(resource);
                if (producer == lastWriter.end())
                    throw std::invalid_argument("RenderTexture read has no active producer in this frame");
                edge(producer->second, i);
            }
        }
        std::vector<std::vector<uint32_t>> consumers(views.size());
        std::vector<uint32_t> remaining(views.size());
        std::priority_queue<uint32_t, std::vector<uint32_t>, std::greater<uint32_t>> ready;
        for (uint32_t i = 0; i < views.size(); ++i) {
            remaining[i] = static_cast<uint32_t>(result.predecessors[i].size());
            if (remaining[i] == 0)
                ready.push(i);
            for (const auto producer : result.predecessors[i])
                consumers[producer].push_back(i);
        }
        while (!ready.empty()) {
            const auto next = ready.top();
            ready.pop();
            result.order.push_back(next);
            for (const auto consumer : consumers[next])
                if (--remaining[consumer] == 0)
                    ready.push(consumer);
        }
        if (result.order.size() != views.size())
            throw std::invalid_argument("RenderTexture dependencies form a cycle between active views");
        return result;
    }
};
} // namespace infernux
