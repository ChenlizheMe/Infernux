#include "ViewLighting.h"

#include <function/renderer/RenderWorld.h>
#include <limits>

namespace infernux::lighting
{
namespace
{
ShadowDepthRange VisibleShadowDepthRange(const RenderViewData &camera,
                                         std::initializer_list<ViewLighting::ReceiverList> receivers)
{
    ShadowDepthRange result{};
    if (!camera.valid)
        return result;

    // Cascade selection in the shader uses view-space z, including any
    // affine view override. A normalized world forward changes those units.
    const auto &view = camera.view;
    const glm::vec3 depthAxis(view[0][2], view[1][2], view[2][2]);
    const auto absoluteDepthAxis = glm::abs(depthAxis);
    float nearest = std::numeric_limits<float>::max();
    float farthest = 0.0f;
    for (const auto &list : receivers) {
        for (const DrawCall &drawCall : list.get()) {
            // Device-authored receivers cover the camera projection. Their final
            // GPU bounds cannot participate in a CPU-only tightening of cascades.
            if (drawCall.devicePose.value)
                return {};
            if (!drawCall.frustumVisible || !drawCall.worldBounds.IsValid())
                continue;
            const glm::vec3 center = (drawCall.worldBounds.min + drawCall.worldBounds.max) * 0.5f;
            const glm::vec3 extent = (drawCall.worldBounds.max - drawCall.worldBounds.min) * 0.5f;
            const float centerDepth = glm::dot(center, depthAxis) + view[3][2];
            const float depthRadius = glm::dot(absoluteDepthAxis, extent);
            const float objectFar = centerDepth + depthRadius;
            if (objectFar <= 0.0f)
                continue;

            // A huge receiver crossing the camera should not collapse the entire
            // logarithmic distribution onto the near clip plane.
            const float boundedRadius = std::min(depthRadius, std::max(centerDepth * 0.5f, 0.0f));
            nearest = std::min(nearest, std::max(centerDepth - boundedRadius, 0.001f));
            farthest = std::max(farthest, objectFar);
        }
    }
    if (farthest <= 0.0f || nearest == std::numeric_limits<float>::max())
        return result;

    // These are geometry bounds. ComputeShadowVP intersects them with the
    // effective projection; dormant authored clipping planes do not apply.
    const float span = std::max(farthest - nearest, 0.001f);
    result.nearDepth = nearest;
    result.farDepth = farthest + span * 0.05f;
    return result;
}
} // namespace

bool ViewLighting::Prepare(const std::shared_ptr<const RenderLightingSource> &source, const RenderViewData &camera,
                           uint32_t shadowResolution, std::initializer_list<ReceiverList> receivers)
{
    const auto depth = shadowResolution ? VisibleShadowDepthRange(camera, receivers) : ShadowDepthRange{};
    if (m_source == source && m_view == camera.view && m_projection == camera.projection &&
        m_position == camera.position && m_valid == camera.valid && m_shadowResolution == shadowResolution &&
        m_visibleDepth.nearDepth == depth.nearDepth && m_visibleDepth.farDepth == depth.farDepth)
        return false;

    m_lights.CollectLights(*source, camera.position);
    m_lights.ComputeShadowVP(*source, camera, static_cast<float>(shadowResolution), depth);
    m_lights.BuildShaderLightingUBO();
    m_source = source;
    m_view = camera.view;
    m_projection = camera.projection;
    m_position = camera.position;
    m_valid = camera.valid;
    m_shadowResolution = shadowResolution;
    m_visibleDepth = depth;
    ++m_revision;
    return true;
}

} // namespace infernux::lighting
