#pragma once

#include "RenderWorld.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace infernux
{
inline constexpr uint32_t kTemporalJitterSampleCount = 128;

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
    static constexpr auto rasterOffsets = [] {
        auto radicalInverse = [](uint32_t index, uint32_t base) {
            const float inverseBase = 1.0f / static_cast<float>(base);
            float place = inverseBase;
            float result = 0.0f;
            while (index != 0u) {
                result += static_cast<float>(index % base) * place;
                index /= base;
                place *= inverseBase;
            }
            return result;
        };
        std::array<glm::vec2, kTemporalJitterSampleCount> offsets{};
        for (uint32_t phase = 0; phase < kTemporalJitterSampleCount; ++phase) {
            // Center Halton(2, 3) samples and convert to projection displacement in top-left pixels.
            const uint32_t sample = phase + 1u;
            offsets[phase] = {0.5f - radicalInverse(sample, 2u), radicalInverse(sample, 3u) - 0.5f};
        }
        return offsets;
    }();
    return 2.0f * rasterOffsets[sampleIndex % kTemporalJitterSampleCount] / glm::vec2(width, height);
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
