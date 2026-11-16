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
    const auto halton = [](uint32_t index, uint32_t base) {
        float value = 0.0f;
        float fraction = 1.0f;
        while (index > 0) {
            fraction /= static_cast<float>(base);
            value += fraction * static_cast<float>(index % base);
            index /= base;
        }
        return value;
    };
    const uint32_t index = sampleIndex % kTemporalJitterSampleCount + 1u;
    return {2.0f * (halton(index, 2u) - 0.5f) / width, 2.0f * (halton(index, 3u) - 0.5f) / height};
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
