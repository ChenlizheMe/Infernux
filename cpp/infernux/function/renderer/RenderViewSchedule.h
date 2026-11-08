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
class GraphBufferInput;

struct RenderViewResultDependency
{
    GraphBufferInput *input;
    uint64_t producerView;
};

struct RenderViewRead
{
    const void *texture;
    // Ordinary display samples initialized storage, including before its first
    // camera write. An explicit graph import can require this frame's producer.
    bool requiresProducer;

    RenderViewRead(const void *texture, bool requiresProducer = true)
        : texture(texture), requiresProducer(requiresProducer)
    {
    }
};

// Resource identities are live owners retained by the camera/graph.
struct RenderViewAccess
{
    std::vector<RenderViewRead> reads;
    std::vector<const void *> writes;
    bool admitted = true;
    uint64_t viewId = 0;
    std::vector<RenderViewResultDependency> pendingResults;
};

struct RenderViewSchedule
{
    std::vector<uint32_t> order;
    std::vector<std::vector<uint32_t>> predecessors;
    std::vector<uint32_t> missingProducers;
    std::vector<uint32_t> feedbackViews;
    std::vector<GraphBufferInput *> cyclicInputs;

    static RenderViewSchedule Build(const std::vector<RenderViewAccess> &views)
    {
        RenderViewSchedule result;
        result.predecessors.resize(views.size());
        std::vector<std::vector<uint32_t>> consumers(views.size());
        std::vector<std::vector<uint32_t>> earlierLayers(views.size());
        std::vector<bool> available(views.size(), true);
        std::queue<uint32_t> withheld;
        const auto withhold = [&](uint32_t view) {
            if (available[view]) {
                available[view] = false;
                withheld.push(view);
            }
        };
        std::unordered_map<const void *, uint32_t> lastWriter;
        const auto edge = [&](uint32_t before, uint32_t after) {
            if (before == after)
                throw std::invalid_argument("RenderTexture feedback requires distinct input/output resources");
            auto &dependencies = result.predecessors[after];
            if (std::find(dependencies.begin(), dependencies.end(), before) == dependencies.end()) {
                dependencies.push_back(before);
                consumers[before].push_back(after);
            }
        };
        for (uint32_t i = 0; i < views.size(); ++i) {
            if (!views[i].admitted)
                withhold(i);
            for (const void *resource : views[i].writes) {
                const auto feedback = std::find_if(views[i].reads.begin(), views[i].reads.end(),
                                                   [resource](const auto &read) { return read.texture == resource; });
                if (feedback != views[i].reads.end()) {
                    if (feedback->requiresProducer)
                        throw std::invalid_argument("RenderTexture feedback requires distinct input/output resources");
                    if (available[i]) {
                        result.feedbackViews.push_back(i);
                        withhold(i);
                    }
                }
                const auto previous = lastWriter.find(resource);
                // Cameras sharing an output preserve author order (depth).
                if (previous != lastWriter.end() && previous->second != i) {
                    edge(previous->second, i);
                    earlierLayers[i].push_back(previous->second);
                }
                lastWriter[resource] = i;
            }
        }
        for (uint32_t i = 0; i < views.size(); ++i) {
            bool missing = false;
            for (const auto &read : views[i].reads) {
                if (!read.requiresProducer)
                    continue;
                const auto producer = lastWriter.find(read.texture);
                if (producer == lastWriter.end())
                    missing = true;
                else
                    edge(producer->second, i);
            }
            if (missing) {
                result.missingProducers.push_back(i);
                withhold(i);
            }
        }
        // Keep all contributions to one output together. If a layer cannot
        // render, later layers and texture readers must wait for a full image.
        while (!withheld.empty()) {
            const auto view = withheld.front();
            withheld.pop();
            for (const auto consumer : consumers[view])
                withhold(consumer);
            for (const auto earlier : earlierLayers[view])
                withhold(earlier);
        }
        // Ready views retain author order. Ordinary texture reads use whatever
        // the shared attachment holds at that point in the Graphics queue.
        std::vector<uint32_t> remaining(views.size());
        std::priority_queue<uint32_t, std::vector<uint32_t>, std::greater<uint32_t>> ready;
        for (uint32_t i = 0; i < views.size(); ++i) {
            remaining[i] = static_cast<uint32_t>(result.predecessors[i].size());
            if (remaining[i] == 0)
                ready.push(i);
        }
        size_t ordered = 0;
        while (!ready.empty()) {
            const auto next = ready.top();
            ready.pop();
            ++ordered;
            if (available[next])
                result.order.push_back(next);
            for (const auto consumer : consumers[next])
                if (--remaining[consumer] == 0)
                    ready.push(consumer);
        }
        if (ordered != views.size())
            throw std::invalid_argument("RenderTexture dependencies form a cycle between active views");

        // A pending managed result cannot complete if withholding its input
        // consumer also withholds the graph execution producing the result's
        // source. Include shared target layers and other pending result edges.
        if (std::any_of(views.begin(), views.end(), [](const auto &view) { return !view.pendingResults.empty(); })) {
            std::unordered_map<uint64_t, uint32_t> indices;
            std::vector<std::vector<uint32_t>> resultUsers(views.size());
            for (uint32_t i = 0; i < views.size(); ++i)
                indices.emplace(views[i].viewId, i);
            for (uint32_t i = 0; i < views.size(); ++i)
                for (const auto &dependency : views[i].pendingResults)
                    if (const auto producer = indices.find(dependency.producerView); producer != indices.end())
                        resultUsers[producer->second].push_back(i);

            for (uint32_t i = 0; i < views.size(); ++i) {
                if (views[i].pendingResults.empty())
                    continue;
                std::vector<bool> reachable(views.size());
                std::vector<uint32_t> queue;
                const auto visit = [&](uint32_t view) {
                    if (!reachable[view]) {
                        reachable[view] = true;
                        queue.push_back(view);
                    }
                };
                visit(i);
                for (size_t next = 0; next < queue.size(); ++next) {
                    const auto view = queue[next];
                    for (const auto *edges : {&consumers[view], &earlierLayers[view], &resultUsers[view]})
                        for (const auto destination : *edges)
                            visit(destination);
                }
                for (const auto &dependency : views[i].pendingResults)
                    if (const auto producer = indices.find(dependency.producerView);
                        producer != indices.end() && reachable[producer->second])
                        result.cyclicInputs.push_back(dependency.input);
            }
        }
        return result;
    }
};
} // namespace infernux
