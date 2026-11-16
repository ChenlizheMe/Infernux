#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace infernux
{
enum class FrameWorkSourcePhase : uint8_t
{
    Observe,
    Refresh,
    Describe,
    AdvanceHost,
    AdvanceGpu,
    Count
};

struct FrameWorkSourcePhaseProfile
{
    uint64_t calls = 0;
    uint64_t nanoseconds = 0;
};

/// Inclusive callback durations. Nested sources overlap; totals are not additive.
/// Counts cover the existing renderer window's Begin/End interval, not a set of
/// complete host frames; frame-tail work outside that interval is not sampled.
/// These values do not retain callbacks, Worlds, resources or source objects.
struct FrameWorkSourceProfile
{
    uint64_t sourceId = 0;
    std::array<FrameWorkSourcePhaseProfile, static_cast<size_t>(FrameWorkSourcePhase::Count)> phases{};
};
} // namespace infernux
