#pragma once

#include "GizmosDrawCallBuffer.h"

#include <algorithm>
#include <cstdint>
#include <glm/glm.hpp>
#include <limits>
#include <optional>

namespace infernux
{

/// The icon pick area is derived from the same world-space billboard that is
/// rendered in Scene view. Coordinates are in the displayed image's space, so
/// a pending render-target resize can stretch the image without shifting hits.
struct ProjectedSceneIcon
{
    glm::vec2 minimum;
    glm::vec2 maximum;
};

[[nodiscard]] inline std::optional<ProjectedSceneIcon>
ProjectSceneIcon(const glm::vec3 &position, const glm::mat4 &cameraToWorld, const glm::mat4 &view,
                 const glm::mat4 &projection, uint32_t renderHeight, float displayScale, const glm::vec2 &displayedSize)
{
    if (renderHeight == 0 || displayedSize.x <= 0.0f || displayedSize.y <= 0.0f)
        return std::nullopt;

    const glm::vec3 cameraPosition(cameraToWorld[3]);
    const glm::vec3 right = glm::normalize(glm::vec3(cameraToWorld[0]));
    const glm::vec3 up = glm::normalize(glm::vec3(cameraToWorld[1]));
    const glm::vec3 forward = glm::normalize(glm::cross(right, up));
    const float halfWorldSize = GizmosDrawCallBuffer::ComputeIconHalfWorldSize(position, cameraPosition, forward,
                                                                               projection, renderHeight, displayScale);

    const glm::mat4 viewProjection = projection * view;
    const glm::vec4 centerClip = viewProjection * glm::vec4(position, 1.0f);
    // Both perspective and orthographic projections use the Vulkan 0..1
    // depth interval. Do not offer an invisible icon behind the camera.
    if (centerClip.w <= 0.0f || centerClip.z < 0.0f || centerClip.z > centerClip.w)
        return std::nullopt;

    glm::vec2 minimum(std::numeric_limits<float>::max());
    glm::vec2 maximum(std::numeric_limits<float>::lowest());
    for (float y : {-1.0f, 1.0f}) {
        for (float x : {-1.0f, 1.0f}) {
            const glm::vec3 corner = position + halfWorldSize * (x * right + y * up);
            const glm::vec4 clip = viewProjection * glm::vec4(corner, 1.0f);
            if (clip.w <= 0.0f)
                return std::nullopt;
            const glm::vec2 point = (glm::vec2(clip) / clip.w + 1.0f) * 0.5f * displayedSize;
            minimum = glm::min(minimum, point);
            maximum = glm::max(maximum, point);
        }
    }

    // UI hit targets need a little breathing room around anti-aliased edges.
    const float margin = 4.0f * std::max(displayScale, 1.0f);
    return ProjectedSceneIcon{minimum - margin, maximum + margin};
}

[[nodiscard]] inline bool SceneIconContains(const ProjectedSceneIcon &icon, const glm::vec2 &point)
{
    return point.x >= icon.minimum.x && point.x <= icon.maximum.x && point.y >= icon.minimum.y &&
           point.y <= icon.maximum.y;
}

} // namespace infernux
