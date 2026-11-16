#pragma once

#include "RenderGraphIdentity.h"
#include "rhi/RenderCoreApi.h"

#include <vector>

namespace infernux
{

enum class GraphPassRetention
{
    Unreachable,
    GraphOutput,
    SideEffect,
    ExternalWrite,
    Dependency
};

struct GraphScheduleRead
{
    GraphResourceHandle resource;
    // A complete attachment clear orders previous users without retaining
    // the contents produced by the previous version.
    bool retainProducer = true;
};

struct GraphSchedulePass
{
    std::vector<GraphScheduleRead> reads;
    std::vector<GraphResourceHandle> writes;
    bool sideEffect = false;
    bool externalWrite = false;
};

struct RenderGraphSchedule
{
    std::vector<uint32_t> order;
    std::vector<std::vector<uint32_t>> dependencies;
    std::vector<GraphPassRetention> retention;
    std::vector<uint32_t> references;
    std::vector<uint32_t> unresolved;
};

/// Compile versioned data dependencies once, independently of the device.
/// Both graphics backends consume the same live-pass order. Resource versions
/// that share storage also induce read-before-overwrite dependencies.
[[nodiscard]] INFERNUX_RENDER_CORE_API RenderGraphSchedule
CompileRenderGraphSchedule(const std::vector<GraphSchedulePass> &passes, GraphResourceHandle output,
                           const std::vector<std::pair<uint32_t, uint32_t>> &explicitDependencies = {});

} // namespace infernux
