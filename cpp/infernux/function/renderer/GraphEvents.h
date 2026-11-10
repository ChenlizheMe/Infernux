#pragma once

#include "rhi/RenderViewContext.h"
#include "rhi/RhiSubmission.h"
#include <cstddef>
#include <glm/glm.hpp>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace infernux
{

struct GraphBufferCapture;

struct GraphEventDesc
{
    std::string label;
    std::string metadata = "null";
    std::map<std::string, std::string> buffers;

    bool operator==(const GraphEventDesc &other) const
    {
        return label == other.label && metadata == other.metadata && buffers == other.buffers;
    }
};

struct GraphEventInterest
{
    bool active = false;
};

struct GraphEventViewSample
{
    uint32_t width;
    uint32_t height;
    glm::vec2 jitterNdc;
    glm::mat4 view;
    glm::mat4 projection;
    glm::mat4 unjitteredProjection;
    std::optional<glm::mat4> previousViewProjection;
};

struct GraphEventRecord
{
    std::string label;
    std::string metadata;
    std::weak_ptr<GraphEventInterest> interest;
    std::string pass;
    size_t ordinal = 0;
    rhi::RenderViewId view = rhi::InvalidRenderViewId;
    uint64_t generation = 0;
    uint64_t occurrence = 0;
    uint64_t historyEpoch = 0;
    rhi::SubmissionTicket submission;
    std::shared_ptr<GraphEventViewSample> viewSample;
    std::vector<std::pair<std::string, std::shared_ptr<GraphBufferCapture>>> sources;
};

using GraphEventInterests = std::unordered_map<std::string, std::weak_ptr<GraphEventInterest>>;
using GraphEventQueue = std::list<GraphEventRecord>;

} // namespace infernux
