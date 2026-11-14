#pragma once

#include "rhi/RenderViewContext.h"
#include "rhi/RhiDescriptors.h"
#include "rhi/RhiSubmission.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace infernux
{
class RenderWorldFrame;
struct RenderWorldPublication;
namespace rhi
{
struct RenderTextureGeneration;
}

enum class GameScreenConsumerKind : uint8_t
{
    Presentation,
    Required,
};

/// A display consumer subscribes to a logical World screen-camera stack.
/// Its identity is independent of World order, Camera IDs and allocation slots.
struct GameScreenOutputConfig
{
    uint32_t width = 1920, height = 1080;
    rhi::FilterMode filter = rhi::FilterMode::Linear;
    bool operator==(const GameScreenOutputConfig &other) const noexcept
    {
        return width == other.width && height == other.height && filter == other.filter;
    }
};

/// The actual raster configuration at CPU World/view-plan sealing.
struct GameScreenSourceConfig
{
    rhi::RenderViewId outputId = rhi::InvalidRenderViewId;
    uint64_t generation = 0;
    GameScreenOutputConfig config;
    rhi::SampleCount samples = rhi::SampleCount::One;
};

/// One admitted source/configuration. A desired resize cannot rewrite it.
struct GameScreenAdmission
{
    uint64_t worldId = 0, generation = 0;
    rhi::RenderViewId outputId = rhi::InvalidRenderViewId;
    GameScreenOutputConfig config;
    rhi::SampleCount samples = rhi::SampleCount::One;
    std::shared_ptr<const RenderWorldFrame> source;
    std::shared_ptr<const RenderWorldPublication> publication;
};

/// Immutable accepted pixels. Only published after completion of this output's
/// actual terminal queue ticket. Consumers must retain the whole sample while
/// using its image; owning only copied native handles is not a pixel lease.
struct GameScreenSample
{
    uint64_t worldId = 0, generation = 0, sequence = 0;
    rhi::RenderViewId outputId = rhi::InvalidRenderViewId;
    uint64_t completedWorldFrame = 0, simulationEpoch = 0;
    rhi::SubmissionTicket completion; // Internal dependency identity; never a user timeline API.
    std::shared_ptr<rhi::WorkCompletionReceipt> receipt = std::make_shared<rhi::WorkCompletionReceipt>();
    rhi::SubmissionSerial retirementEpoch = rhi::InvalidSubmissionSerial;
    std::shared_ptr<const RenderWorldFrame> source;
    std::shared_ptr<const GameScreenAdmission> admission;
    std::shared_ptr<const rhi::RenderTextureGeneration> image;
    std::shared_ptr<const void> storageOwner;
};

struct GameScreenOutputStatus
{
    uint64_t worldId = 0, generation = 0;
    rhi::RenderViewId outputId = rhi::InvalidRenderViewId;
    GameScreenOutputConfig requested;
    std::size_t consumers = 0, pendingSamples = 0, allocatedSlots = 0;
    bool hasCompletedSample = false, workWithheld = false, waitingForPublication = false;
    bool sourcePublished = false, hasScreenCamera = false;
    bool hasPresentationConsumer = false, hasRequiredConsumer = false;
    // Snapshot facts for the shared SceneManager Step owner. Desired config
    // above never changes the identity/config of these retained records.
    std::shared_ptr<const GameScreenAdmission> admitted;
    std::shared_ptr<const GameScreenSample> accepted, completed;
    std::string error;
    std::string publicationWaitReason;
};
} // namespace infernux
