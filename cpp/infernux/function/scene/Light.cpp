#include "Light.h"
#include "ComponentDocumentValidation.h"
#include "ComponentFactory.h"
#include "GameObject.h"
#include "SceneManager.h"
#include "Transform.h"
#include <cmath>
#include <core/log/InxLog.h>
#include <core/types/ColorSpace.h>
#include <limits>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace infernux
{

namespace
{
glm::vec3 ApproximateBlackbodySrgb(float kelvin)
{
    const float t = glm::clamp(kelvin, 1000.0f, 20000.0f) / 100.0f;
    const float red =
        t <= 66.0f ? 1.0f : glm::clamp(329.698727446f * std::pow(t - 60.0f, -0.1332047592f) / 255.0f, 0.0f, 1.0f);
    const float green = t <= 66.0f
                            ? glm::clamp((99.4708025861f * std::log(t) - 161.1195681661f) / 255.0f, 0.0f, 1.0f)
                            : glm::clamp(288.1221695283f * std::pow(t - 60.0f, -0.0755148492f) / 255.0f, 0.0f, 1.0f);
    const float blue = t >= 66.0f ? 1.0f
                       : t <= 19.0f
                           ? 0.0f
                           : glm::clamp((138.5177312231f * std::log(t - 10.0f) - 305.0447927307f) / 255.0f, 0.0f, 1.0f);
    return {red, green, blue};
}

glm::vec3 NormalizedBlackbodyLinear(float kelvin)
{
    if (kelvin == 6500.0f)
        return glm::vec3(1.0f);
    static const glm::vec3 reference = inx::color::SrgbToLinear(ApproximateBlackbodySrgb(6500.0f));
    return inx::color::SrgbToLinear(ApproximateBlackbodySrgb(kelvin)) / reference;
}

SemanticTypeDescriptor DescribeLight()
{
    SemanticTypeDescriptor type;
    type.typeGuid = "native:infernux.Light";
    type.readableId = "infernux.component.light";
    type.owner = "engine:native";
    type.origin = "native";
    type.displayName = "Light";
    type.runtimeProfiles = {"editor", "player", "headless"};
    const auto add = [&](const char *name, const char *stored, const char *kind, json initial) -> json & {
        type.fields.push_back({std::string("Light.") + name,
                               std::string("FieldType.") + kind,
                               false,
                               {{"field_id", name},
                                {"serialized_name", stored},
                                {"serialized", true},
                                {"hidden", false},
                                {"nullable", false},
                                {"storage_kind", "native_property"},
                                {"display_name_key", std::string("light.") + name},
                                {"tooltip", std::string("light.tooltip.") + name},
                                {"default", std::move(initial)}}});
        return type.fields.back().attributes;
    };
    const auto enumeration = [&](const char *name, const char *stored, const char *enumName,
                                 std::initializer_list<const char *> names, std::initializer_list<const char *> labels,
                                 const char *initial) {
        json members = json::array();
        size_t value = 0;
        for (const char *member : names)
            members.push_back({{"name", member}, {"value", value++}});
        auto &attributes = add(name, stored, "ENUM", {{"$type", "enum"}, {"enum_type", enumName}, {"name", initial}});
        attributes["enum"] = {{"type_id", std::string("native:infernux.") + enumName},
                              {"members", std::move(members)},
                              {"labels", labels}};
    };

    enumeration("light_type", "lightType", "LightType", {"Directional", "Point", "Spot", "Area"},
                {"light.type.directional", "light.type.point", "light.type.spot", "light.type.area"}, "Directional");
    enumeration("color_mode", "colorMode", "LightColorMode", {"Color", "FilterAndTemperature"},
                {"light.color_mode.color", "light.color_mode.filter_and_temperature"}, "Color");
    type.fields.back().attributes["header"] = "light.section.appearance";
    type.fields.back().attributes["serialized"] = false;
    type.fields.back().attributes["setter_owns_document_shape"] = true;
    add("color", "color", "VEC3", {1.0, 1.0, 1.0});
    add("use_color_temperature", "useColorTemperature", "BOOL", false)["hidden"] = true;
    add("color_temperature", "colorTemperature", "FLOAT", 6500.0)["range"] = {1000.0, 20000.0};
    add("intensity", "intensity", "FLOAT", 1.0)["range"] = {0.0, 10.0};
    add("range", "range", "FLOAT", 10.0)["range"] = {0.1, 100.0};
    add("spot_angle", "spotAngle", "FLOAT", 30.0)["range"] = {1.0, 179.0};
    add("outer_spot_angle", "outerSpotAngle", "FLOAT", 45.0)["range"] = {1.0, 179.0};
    add("area_size", "areaSize", "VEC2", {1.6, 1.0});
    add("area_two_sided", "areaTwoSided", "BOOL", false);
    enumeration("shadows", "shadows", "LightShadows", {"NoShadows", "Hard", "Soft"},
                {"light.shadows.none", "light.shadows.hard", "light.shadows.soft"}, "Hard");
    type.fields.back().attributes["header"] = "light.section.shadows";
    add("shadow_strength", "shadowStrength", "FLOAT", 1.0)["range"] = {0.0, 1.0};
    add("shadow_softness", "shadowSoftness", "FLOAT", 1.5)["range"] = {0.25, 8.0};
    enumeration("render_mode", "renderMode", "LightRenderMode", {"Auto", "ForcePixel", "ForceVertex"},
                {"light.render.auto", "light.render.force_pixel", "light.render.force_vertex"}, "Auto");
    type.fields.back().attributes["header"] = "light.section.rendering";
    add("culling_mask", "cullingMask", "INT", 0xffffffffu);
    add("influence_domains", "influenceDomains", "INT", AllLightInfluenceDomains);
    add("baked", "baked", "BOOL", false)["header"] = "light.section.baking";
    return type;
}

const bool registeredLight = ComponentFactory::Register(
    "Light", [] { return std::make_unique<Light>(); }, Light::ValidateSerializedDocument, Light::GetTypeConstraints(),
    DescribeLight);
} // namespace

Light::~Light()
{
    SceneManager::Instance().UnregisterLight(this);
}

void Light::OnEnable()
{
    // Only runtime-resident scenes contribute to the global light list.
    // Prefab/template utility scenes must not leak here, while objects moved
    // to the DontDestroyOnLoad scene must be able to re-enable normally.
    if (auto *go = GetGameObject())
        if (!SceneManager::Instance().IsRuntimeScene(go->GetScene()))
            return;
    SceneManager::Instance().RegisterLight(this);
}

void Light::OnDisable()
{
    SceneManager::Instance().UnregisterLight(this);
}

glm::vec3 Light::GetLinearColor() const
{
    return m_effectiveLinearColor;
}

glm::vec3 Light::GetEffectiveColor() const
{
    return inx::color::LinearToSrgb(GetLinearColor());
}

void Light::UpdateEffectiveColorCache()
{
    const glm::vec3 filter = inx::color::SrgbToLinear(m_color);
    m_effectiveLinearColor = m_useColorTemperature ? filter * NormalizedBlackbodyLinear(m_colorTemperature) : filter;
}

nlohmann::json Light::SerializeDocument() const
{
    json j = Component::SerializeDocument();

    // Light type
    j["lightType"] = static_cast<int>(m_lightType);

    // Color & intensity
    j["color"] = {m_color.r, m_color.g, m_color.b};
    j["useColorTemperature"] = m_useColorTemperature;
    j["colorTemperature"] = m_colorTemperature;
    j["intensity"] = m_intensity;

    // Range
    j["range"] = m_range;

    // Spot settings
    j["spotAngle"] = m_spotAngle;
    j["outerSpotAngle"] = m_outerSpotAngle;
    j["areaSize"] = {m_areaSize.x, m_areaSize.y};
    j["areaTwoSided"] = m_areaTwoSided;

    // Shadows
    j["shadows"] = static_cast<int>(m_shadows);
    j["shadowStrength"] = m_shadowStrength;
    j["shadowSoftness"] = m_shadowSoftness;

    // Rendering
    j["renderMode"] = static_cast<int>(m_renderMode);
    j["cullingMask"] = m_cullingMask;
    j["influenceDomains"] = m_influenceDomains;

    // Baking
    j["baked"] = m_baked;

    return j;
}

void Light::ValidateSerializedDocument(const nlohmann::json &j)
{
    using namespace component_document_validation;
    ValidateComponentDocument(j, "Light",
                              {"lightType", "color", "useColorTemperature", "colorTemperature", "intensity", "range",
                               "spotAngle", "outerSpotAngle", "areaSize", "areaTwoSided", "shadows", "shadowStrength",
                               "shadowSoftness", "renderMode", "cullingMask", "influenceDomains", "baked"});
    const int lightType = RequireInteger(j, "lightType", "Light");
    RequireFiniteVector(j, "color", 3, "Light");
    RequireBoolean(j, "useColorTemperature", "Light");
    const float colorTemperature = RequireFiniteFloat(j, "colorTemperature", "Light");
    const float intensity = RequireFiniteFloat(j, "intensity", "Light");
    const float range = RequireFiniteFloat(j, "range", "Light");
    const float spotAngle = RequireFiniteFloat(j, "spotAngle", "Light");
    const float outerSpotAngle = RequireFiniteFloat(j, "outerSpotAngle", "Light");
    RequireFiniteVector(j, "areaSize", 2, "Light");
    RequireBoolean(j, "areaTwoSided", "Light");
    const int shadows = RequireInteger(j, "shadows", "Light");
    const float shadowStrength = RequireFiniteFloat(j, "shadowStrength", "Light");
    const float shadowSoftness = RequireFiniteFloat(j, "shadowSoftness", "Light");
    const int renderMode = RequireInteger(j, "renderMode", "Light");
    const uint64_t cullingMask = RequireUnsignedInteger(j, "cullingMask", "Light");
    const uint64_t influenceDomains = RequireUnsignedInteger(j, "influenceDomains", "Light");
    RequireBoolean(j, "baked", "Light");

    if (lightType < static_cast<int>(LightType::Directional) || lightType > static_cast<int>(LightType::Area))
        throw std::invalid_argument("Light.lightType is unsupported");
    if (colorTemperature < 1000.0f || colorTemperature > 20000.0f)
        throw std::invalid_argument("Light.colorTemperature must be between 1000 and 20000 K");
    if (intensity < 0.0f || range <= 0.0f)
        throw std::invalid_argument("Light intensity and range are invalid");
    if (spotAngle <= 0.0f || outerSpotAngle < spotAngle || outerSpotAngle >= 180.0f)
        throw std::invalid_argument("Light spot cone angles are invalid");
    if (j["areaSize"][0].get<float>() <= 0.0f || j["areaSize"][1].get<float>() <= 0.0f)
        throw std::invalid_argument("Light area size is invalid");
    if (shadows < static_cast<int>(LightShadows::None) || shadows > static_cast<int>(LightShadows::Soft))
        throw std::invalid_argument("Light.shadows is unsupported");
    if (shadowStrength < 0.0f || shadowStrength > 1.0f || shadowSoftness < 0.25f || shadowSoftness > 8.0f)
        throw std::invalid_argument("Light shadow parameters are invalid");
    if (renderMode < static_cast<int>(LightRenderMode::Auto) ||
        renderMode > static_cast<int>(LightRenderMode::ForceVertex))
        throw std::invalid_argument("Light.renderMode is unsupported");
    if (cullingMask > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("Light.cullingMask exceeds 32 bits");
    if ((influenceDomains & ~static_cast<uint64_t>(AllLightInfluenceDomains)) != 0u)
        throw std::invalid_argument("Light.influenceDomains contains unsupported flags");
}

bool Light::DeserializeDocument(const nlohmann::json &j)
{
    try {
        ValidateSerializedDocument(j);
        if (!Component::DeserializeDocument(j))
            return false;

        m_lightType = static_cast<LightType>(j["lightType"].get<int>());
        m_color = glm::vec3(j["color"][0].get<float>(), j["color"][1].get<float>(), j["color"][2].get<float>());
        m_useColorTemperature = j["useColorTemperature"].get<bool>();
        m_colorTemperature = j["colorTemperature"].get<float>();
        UpdateEffectiveColorCache();
        m_intensity = j["intensity"].get<float>();
        m_range = j["range"].get<float>();
        m_spotAngle = j["spotAngle"].get<float>();
        m_outerSpotAngle = j["outerSpotAngle"].get<float>();
        m_areaSize = glm::vec2(j["areaSize"][0].get<float>(), j["areaSize"][1].get<float>());
        m_areaTwoSided = j["areaTwoSided"].get<bool>();
        m_shadows = static_cast<LightShadows>(j["shadows"].get<int>());
        m_shadowStrength = j["shadowStrength"].get<float>();
        m_shadowSoftness = j["shadowSoftness"].get<float>();
        m_renderMode = static_cast<LightRenderMode>(j["renderMode"].get<int>());
        m_cullingMask = j["cullingMask"].get<uint32_t>();
        m_influenceDomains = j["influenceDomains"].get<uint32_t>();
        m_baked = j["baked"].get<bool>();

        return true;
    } catch (const std::exception &e) {
        INXLOG_ERROR("Light::Deserialize failed: ", e.what());
        return false;
    }
}

std::unique_ptr<Component> Light::Clone() const
{
    auto clone = std::make_unique<Light>();
    clone->m_enabled = m_enabled;
    clone->m_executionOrder = m_executionOrder;
    clone->m_lightType = m_lightType;
    clone->m_color = m_color;
    clone->m_useColorTemperature = m_useColorTemperature;
    clone->m_colorTemperature = m_colorTemperature;
    clone->m_effectiveLinearColor = m_effectiveLinearColor;
    clone->m_intensity = m_intensity;
    clone->m_range = m_range;
    clone->m_spotAngle = m_spotAngle;
    clone->m_outerSpotAngle = m_outerSpotAngle;
    clone->m_areaSize = m_areaSize;
    clone->m_areaTwoSided = m_areaTwoSided;
    clone->m_shadows = m_shadows;
    clone->m_shadowStrength = m_shadowStrength;
    clone->m_shadowSoftness = m_shadowSoftness;
    clone->m_renderMode = m_renderMode;
    clone->m_cullingMask = m_cullingMask;
    clone->m_influenceDomains = m_influenceDomains;
    clone->m_baked = m_baked;
    return clone;
}

} // namespace infernux
