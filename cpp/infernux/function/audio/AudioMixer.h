#pragma once

#include "AudioBusAutomation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace infernux::audio_mixer
{

struct StereoFrame
{
    float left;
    float right;
};

// Output order: source/track/spatial gain -> bus and master gain (once per
// voice) -> SDL sum/clamp -> this optional postmix stage -> output meters.
// A future processor owns only fixed-size callback-local state. The owner
// publishes enable and one scalar parameter together; the callback reads one
// lock-free snapshot per block. Bypass never touches the interleaved F32 data.
struct OutputDspSettings
{
    bool enabled = false;
    float parameter = 0.0f;
};

struct BypassOutputDsp
{
    void Process(float *, size_t, int, float) noexcept
    {
    }
    void Reset() noexcept
    {
    }
};

template <typename Processor> class OutputDspStage
{
  public:
    static_assert(noexcept(std::declval<Processor &>().Process(nullptr, size_t{}, 0, 0.0f)));
    static_assert(noexcept(std::declval<Processor &>().Reset()));

    // Owner thread only. One release store publishes the complete setting.
    void Publish(OutputDspSettings settings) noexcept
    {
        m_settings.Publish(settings);
    }

    // Audio callback only. Processor::Process/Reset must neither allocate nor
    // lock; their noexcept signatures are checked above. No dynamic dispatch.
    void Process(float *samples, size_t sampleCount, int channels) noexcept
    {
        const auto &settings = m_settings.Consume();
        if (!settings.enabled) {
            if (m_active) {
                m_processor.Reset();
                m_active = false;
            }
            return;
        }
        m_active = true;
        m_processor.Process(samples, sampleCount, channels, settings.parameter);
    }

    // Only after the audio callback has stopped (before start or after close).
    // Clears both authored settings and processor history for the next device.
    void ResetDeviceSession() noexcept
    {
        m_settings.Publish({});
        m_processor.Reset();
        m_active = false;
    }

  private:
    RealtimeValueMailbox<OutputDspSettings> m_settings;
    Processor m_processor{};
    bool m_active = false; // Audio callback state, or after callback shutdown.
};

struct OutputMeter
{
    float peak = 0.0f;
    uint64_t saturatedSamples = 0;
};

inline OutputMeter MeasureOutput(const float *samples, size_t sampleCount) noexcept
{
    OutputMeter result;
    for (size_t sample = 0; sample < sampleCount; ++sample) {
        const float magnitude = std::abs(samples[sample]);
        result.peak = std::max(result.peak, magnitude);
        result.saturatedSamples += magnitude >= 1.0f;
    }
    return result;
}

inline float ApproachParameter(float current, float target, float maximumStep)
{
    const float delta = target - current;
    if (std::abs(delta) <= maximumStep)
        return target;
    return current + std::copysign(maximumStep, delta);
}

/// Linear point-source attenuation with explicit near/far boundaries.
/// A collapsed range remains a hard cutoff at minDistance.
inline float DistanceAttenuation(float distance, float minDistance, float maxDistance)
{
    if (maxDistance <= minDistance)
        return distance <= minDistance ? 1.0f : 0.0f;
    if (distance <= minDistance)
        return 1.0f;
    if (distance >= maxDistance)
        return 0.0f;
    return 1.0f - (distance - minDistance) / (maxDistance - minDistance);
}

/// Mix one decoded stereo frame between a channel-preserving 2D signal and
/// an equal-power point-source signal. Distance attenuation belongs only to
/// the spatial side of the blend.
inline StereoFrame MixStereoFrame(float left, float right, float gain, float spatialGain, float pan, float spatialBlend)
{
    const float blend = std::clamp(spatialBlend, 0.0f, 1.0f);
    const float clampedPan = std::clamp(pan, -1.0f, 1.0f);
    const float pan01 = (clampedPan + 1.0f) * 0.5f;
    constexpr float halfPi = 1.57079632679f;
    const float pointLeftGain = std::cos(pan01 * halfPi);
    const float pointRightGain = std::sin(pan01 * halfPi);
    const float pointSample = 0.5f * (left + right) * std::clamp(spatialGain, 0.0f, 1.0f);
    return {
        gain * (left + (pointSample * pointLeftGain - left) * blend),
        gain * (right + (pointSample * pointRightGain - right) * blend),
    };
}

} // namespace infernux::audio_mixer
