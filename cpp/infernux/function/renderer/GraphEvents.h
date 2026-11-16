#pragma once

#include "rhi/RenderViewContext.h"
#include "rhi/RhiDescriptors.h"
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

namespace rhi
{
class ComputeBuffer;
}
struct GameScreenAdmission;
/// A capture of one submitted graph execution, not a mutable latest-value slot.
struct GraphBufferCapture
{
    enum class Status
    {
        Pending,
        Accepted,
        Cancelled
    };
    Status status = Status::Pending;
    std::string error;
    std::vector<uint64_t> shape;
    rhi::ComputeScalarType scalarType = rhi::ComputeScalarType::Float32;
    uint64_t byteOffset = 0;
    uint64_t storageByteSize = 0;
    std::string storageClaim;
    std::shared_ptr<rhi::ComputeBuffer> storageBuffer;
    rhi::RenderViewId view = rhi::InvalidRenderViewId;
    uint64_t generation = 0, occurrence = 0;
    std::string name;
    std::shared_ptr<rhi::ComputeBuffer> value;
    bool selected = false;
    // Actual execution contract, set when this capture is selected.
    std::shared_ptr<const GameScreenAdmission> screenAdmission;
    void Cancel(const std::string &reason)
    {
        if (status == Status::Pending) {
            status = Status::Cancelled;
            error = reason;
            storageBuffer.reset();
        }
    }
};

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
