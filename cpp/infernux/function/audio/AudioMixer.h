#pragma once

#include <algorithm>
#include <cmath>

namespace infernux::audio_mixer
{

struct StereoFrame
{
    float left;
    float right;
};

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
