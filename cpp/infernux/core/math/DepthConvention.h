#pragma once

#include <glm/glm.hpp>

namespace infernux::depth
{

inline constexpr float Near = 1.0f;
inline constexpr float Far = 0.0f;

// Perspective eye depth is 1 / (z * deviceDepth + w).
[[nodiscard]] inline glm::vec4 LinearizationParams(float nearClip, float farClip)
{
    const float ratio = farClip / nearClip;
    return {ratio - 1.0f, 1.0f, (ratio - 1.0f) / farClip, 1.0f / farClip};
}

} // namespace infernux::depth
