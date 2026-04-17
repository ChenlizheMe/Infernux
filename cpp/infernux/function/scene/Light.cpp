#include "Light.h"
#include "ComponentFactory.h"
#include "GameObject.h"
#include "SceneManager.h"
#include "Transform.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace infernux
{

// Register Light component with factory
INFERNUX_REGISTER_COMPONENT("Light", Light)

Light::~Light()
{
    SceneManager::Instance().UnregisterLight(this);
}

void Light::OnEnable()
{
    // Only register with the global light list if this object belongs to
    // the active scene.  Prefab template cache objects must not leak here.
    if (auto *go = GetGameObject())
        if (go->GetScene() != SceneManager::Instance().GetActiveScene())
            return;
    SceneManager::Instance().RegisterLight(this);
}

void Light::OnDisable()
{
    SceneManager::Instance().UnregisterLight(this);
}

std::string Light::Serialize() const
{
    json j;
    j["schema_version"] = 1;
    j["type"] = GetTypeName();
    j["enabled"] = IsEnabled();
    j["component_id"] = GetComponentID();

    // Light type
    j["lightType"] = static_cast<int>(m_lightType);

    // Color & intensity
    j["color"] = {m_color.r, m_color.g, m_color.b};
    j["intensity"] = m_intensity;

    // Range
    j["range"] = m_range;

    // Spot settings
    j["spotAngle"] = m_spotAngle;
    j["outerSpotAngle"] = m_outerSpotAngle;

    // Shadows
    j["shadows"] = static_cast<int>(m_shadows);
    j["shadowStrength"] = m_shadowStrength;
    j["shadowBias"] = m_shadowBias;
    j["shadowNormalBias"] = m_shadowNormalBias;

    // Rendering
    j["renderMode"] = static_cast<int>(m_renderMode);
    j["cullingMask"] = m_cullingMask;

    // Baking
    j["baked"] = m_baked;

    return j.dump();
}

bool Light::Deserialize(const std::string &jsonStr)
{
    try {
        json j = json::parse(jsonStr);

        if (j.contains("enabled")) {
            SetEnabled(j["enabled"].get<bool>());
        }
        if (j.contains("component_id")) {
            SetComponentID(j["component_id"].get<uint64_t>());
        }

        // Light type
        if (j.contains("lightType")) {
            m_lightType = static_cast<LightType>(j["lightType"].get<int>());
        }

        // Color & intensity
        if (j.contains("color") && j["color"].is_array() && j["color"].size() >= 3) {
            m_color = glm::vec3(j["color"][0].get<float>(), j["color"][1].get<float>(), j["color"][2].get<float>());
        }
        if (j.contains("intensity")) {
            m_intensity = j["intensity"].get<float>();
        }

        // Range
        if (j.contains("range")) {
            m_range = j["range"].get<float>();
        }

        // Spot settings
        if (j.contains("spotAngle")) {
            m_spotAngle = j["spotAngle"].get<float>();
        }
        if (j.contains("outerSpotAngle")) {
            m_outerSpotAngle = j["outerSpotAngle"].get<float>();
        }

        // Shadows
        if (j.contains("shadows")) {
            m_shadows = static_cast<LightShadows>(j["shadows"].get<int>());
        }
        if (j.contains("shadowStrength")) {
            m_shadowStrength = j["shadowStrength"].get<float>();
        }
        if (j.contains("shadowBias")) {
            m_shadowBias = j["shadowBias"].get<float>();
        }
        if (j.contains("shadowNormalBias")) {
            m_shadowNormalBias = j["shadowNormalBias"].get<float>();
        }

        // Rendering
        if (j.contains("renderMode")) {
            m_renderMode = static_cast<LightRenderMode>(j["renderMode"].get<int>());
        }
        if (j.contains("cullingMask")) {
            m_cullingMask = j["cullingMask"].get<uint32_t>();
        }

        // Baking
        if (j.contains("baked")) {
            m_baked = j["baked"].get<bool>();
        }

        return true;
    } catch (const std::exception &e) {
        return false;
    }
}

// ============================================================================
// Shadow mapping — light view/projection helpers
// ============================================================================

glm::mat4 Light::GetLightViewMatrix(const glm::vec3 &shadowCenter) const
{
    // Default: look along -Z
    glm::vec3 lightDir = glm::vec3(0.0f, -1.0f, 0.0f);
    glm::vec3 lightPos = glm::vec3(0.0f, 10.0f, 0.0f);

    // If attached to a GameObject, use its transform
    if (GetGameObject()) {
        Transform *transform = GetGameObject()->GetTransform();
        if (transform) {
            lightDir = transform->GetWorldForward();
            lightPos = transform->GetWorldPosition();
        }
    }

    // For directional lights, center the shadow frustum on shadowCenter
    // (typically the camera position) and place the light far along -lightDir
    if (m_lightType == LightType::Directional) {
        lightPos = shadowCenter - lightDir * 50.0f;
    }

    glm::vec3 target = lightPos + lightDir;
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);

    // Avoid degenerate case when light points straight up/down
    if (std::abs(glm::dot(lightDir, up)) > 0.99f) {
        up = glm::vec3(0.0f, 0.0f, 1.0f);
    }

    return glm::lookAt(lightPos, target, up);
}

glm::mat4 Light::GetLightProjectionMatrix(float shadowExtent, float nearPlane, float farPlane) const
{
    switch (m_lightType) {
    case LightType::Directional:
        // Orthographic projection for directional light shadows
        return glm::ortho(-shadowExtent, shadowExtent, -shadowExtent, shadowExtent, nearPlane, farPlane);

    case LightType::Spot: {
        // Perspective projection matching the spot cone angle
        float fov = glm::radians(m_outerSpotAngle * 2.0f);
        return glm::perspective(fov, 1.0f, nearPlane, m_range);
    }

    case LightType::Point:
    case LightType::Area:
    default:
        // Point light shadow map requires cubemap — not yet supported
        // Return identity as placeholder
        return glm::mat4(1.0f);
    }
}

std::unique_ptr<Component> Light::Clone() const
{
    auto clone = std::make_unique<Light>();
    clone->m_enabled = m_enabled;
    clone->m_executionOrder = m_executionOrder;
    clone->m_lightType = m_lightType;
    clone->m_color = m_color;
    clone->m_intensity = m_intensity;
    clone->m_range = m_range;
    clone->m_spotAngle = m_spotAngle;
    clone->m_outerSpotAngle = m_outerSpotAngle;
    clone->m_shadows = m_shadows;
    clone->m_shadowStrength = m_shadowStrength;
    clone->m_shadowBias = m_shadowBias;
    clone->m_shadowNormalBias = m_shadowNormalBias;
    clone->m_renderMode = m_renderMode;
    clone->m_cullingMask = m_cullingMask;
    clone->m_baked = m_baked;
    return clone;
}

} // namespace infernux
