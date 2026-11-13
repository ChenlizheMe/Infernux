#pragma once

#include <array>
#include <cstddef>
#include <string_view>
#include <utility>

namespace infernux
{

// The same primary-component order identifies objects in Hierarchy and Inspector.
class EditorObjectIcon
{
  public:
    void Consider(std::string_view type)
    {
        for (std::size_t index = 0; index < m_priority; ++index) {
            if (s_icons[index].first == type) {
                m_priority = index;
                break;
            }
        }
    }

    [[nodiscard]] const char *Resource(bool prefab = false) const
    {
        return prefab                         ? "prefab"
               : m_priority == s_icons.size() ? "components/component_transform"
                                              : s_icons[m_priority].second;
    }

  private:
    static constexpr std::array s_icons{
        std::pair{"Camera", "components/component_camera"},
        std::pair{"Light", "components/component_light"},
        std::pair{"SkinnedMeshRenderer", "components/component_skinnedmeshrenderer"},
        std::pair{"MeshRenderer", "components/component_meshrenderer"},
        std::pair{"SpriteRenderer", "components/component_spriterenderer"},
        std::pair{"LineRenderer", "components/component_linerenderer"},
        std::pair{"TrailRenderer", "components/component_trailrenderer"},
        std::pair{"ParticleSystem", "components/component_particlesystem"},
        std::pair{"UICanvas", "components/component_uicanvas"},
        std::pair{"UIImage", "components/component_uiimage"},
        std::pair{"UIText", "components/component_uitext"},
        std::pair{"UIButton", "components/component_uibutton"},
        std::pair{"Rigidbody", "components/component_rigidbody"},
        std::pair{"BoxCollider", "components/component_boxcollider"},
        std::pair{"SphereCollider", "components/component_spherecollider"},
        std::pair{"AudioSource", "components/component_audiosource"},
        std::pair{"AudioListener", "components/component_audiolistener"},
        std::pair{"Script", "components/component_script"},
    };
    std::size_t m_priority = s_icons.size();
};

} // namespace infernux
