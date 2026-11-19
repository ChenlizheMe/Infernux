#pragma once

#include <function/scene/LightingData.h>

#include <functional>
#include <initializer_list>

namespace infernux
{
struct DrawCall;

namespace lighting
{

/// Lighting and shadow cascades prepared for one camera from an immutable World publication.
class ViewLighting
{
  public:
    using ReceiverList = std::reference_wrapper<const std::vector<DrawCall>>;

    /// Returns true when light uniforms, shadow views and their bindings need to be republished.
    bool Prepare(const std::shared_ptr<const RenderLightingSource> &source, const RenderViewData &camera,
                 uint32_t shadowResolution, std::initializer_list<ReceiverList> receivers);

    const SceneLightCollector &Lights() const noexcept
    {
        return m_lights;
    }
    uint64_t Revision() const noexcept
    {
        return m_revision;
    }

  private:
    std::shared_ptr<const RenderLightingSource> m_source;
    glm::mat4 m_view{1.0f}, m_projection{1.0f};
    glm::vec3 m_position{0.0f};
    ShadowDepthRange m_visibleDepth{};
    uint32_t m_shadowResolution = 0;
    bool m_valid = false;
    uint64_t m_revision = 1;
    SceneLightCollector m_lights;
};

} // namespace lighting
} // namespace infernux
