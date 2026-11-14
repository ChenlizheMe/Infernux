#pragma once

#include "RhiHandles.h"
#include "RhiTypes.h"

#include <cstdint>

namespace infernux::rhi
{

using SubmissionSerial = uint64_t;
inline constexpr SubmissionSerial InvalidSubmissionSerial = 0;

enum class QueueRole : uint8_t
{
    Graphics,
    Compute,
    Transfer,
    Present,
    Count,
};

enum class SubmissionDomain : uint8_t
{
    Frame,
    AsyncUpload,
    AsyncReadback,
    Preview,
    Background,
};

struct SubmissionTicket
{
    DeviceId device = InvalidDeviceId;
    QueueRole queue = QueueRole::Graphics;
    SubmissionSerial serial = InvalidSubmissionSerial;

    [[nodiscard]] constexpr bool IsValid() const noexcept
    {
        return device != InvalidDeviceId && serial != InvalidSubmissionSerial;
    }
    [[nodiscard]] constexpr bool operator==(const SubmissionTicket &other) const noexcept
    {
        return device == other.device && queue == other.queue && serial == other.serial;
    }
};

/// Owner-thread observation of one actually accepted graphics work batch.
/// No native timeline, device, resource or callback is owned by this value.
struct WorkCompletionReceipt
{
    SubmissionTicket submission;
    bool complete = false;
    bool failed = false;
};

struct SubmissionDependency
{
    SubmissionTicket producer;
    PipelineStage stages = PipelineStage::AllCommands;
};

/// Observed work in the original frame and compute pools, not differences
/// between reserved/submitted serials. The newest submitted frame may belong
/// to an earlier opportunity when presentation is suspended.
struct GpuQueuePressure
{
    uint32_t frameSlots = 0;
    uint32_t pendingFrames = 0;
    uint32_t olderPendingFrames = 0;
    uint32_t recordingFrames = 0;
    uint64_t computeSubmissions = 0;
};

} // namespace infernux::rhi
