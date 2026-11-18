#pragma once

#include "RenderWorld.h"

#include <algorithm>
#include <cmath>

namespace infernux
{
inline constexpr uint32_t kTemporalJitterSampleCount = 8;

inline bool IsRenderViewCut(const RenderViewData &previous, const RenderViewData &current)
{
    if (previous.cameraId != current.cameraId || previous.worldId != current.worldId ||
        previous.simulationEpoch != current.simulationEpoch ||
        previous.temporalHistoryRevision != current.temporalHistoryRevision)
        return true;
    const glm::mat4 previousInverse = glm::inverse(previous.view);
    const glm::mat4 currentInverse = glm::inverse(current.view);
    const glm::vec3 offset = glm::vec3(currentInverse[3] - previousInverse[3]);
    const float directionCosine =
        glm::dot(glm::normalize(glm::vec3(previousInverse[2])), glm::normalize(glm::vec3(currentInverse[2])));
    if (glm::dot(offset, offset) > 100.0f || directionCosine < 0.70710678f)
        return true;
    for (glm::length_t column = 0; column < 4; ++column)
        for (glm::length_t row = 0; row < 4; ++row)
            if (std::abs(current.projection[column][row] - previous.projection[column][row]) > 1e-3f)
                return true;
    return false;
}

inline glm::vec2 RenderViewJitterNdc(uint32_t sampleIndex, uint32_t width, uint32_t height)
{
    if (width == 0 || height == 0)
        return glm::vec2(0.0f);
    // Each axis visits all eight subpixel strata. Opposite phases cancel exactly,
    // so a completed coverage cycle stays centered on the unjittered pixel.
    static constexpr int offsets[kTemporalJitterSampleCount][2] = {
        {-7, -1}, {3, 5}, {1, -7}, {-5, 3}, {7, 1}, {-3, -5}, {-1, 7}, {5, -3}};
    const auto &offset = offsets[sampleIndex % kTemporalJitterSampleCount];
    return {static_cast<float>(offset[0]) / (8.0f * width), static_cast<float>(offset[1]) / (8.0f * height)};
}

inline glm::mat4 JitterRenderViewProjection(glm::mat4 projection, const glm::vec2 &jitterNdc)
{
    for (glm::length_t column = 0; column < 4; ++column) {
        projection[column][0] += jitterNdc.x * projection[column][3];
        projection[column][1] += jitterNdc.y * projection[column][3];
    }
    return projection;
}
} // namespace infernux
