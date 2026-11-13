#include "LightingData.h"
#include "Camera.h"
#include "GameObject.h"
#include "Light.h"
#include "Scene.h"
#include "SceneManager.h"
#include "SceneRenderBridge.h"
#include "Transform.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <core/log/InxLog.h>
#include <core/types/ColorSpace.h>
#include <cstring>
#include <function/renderer/RenderLightingSource.h>
#include <function/renderer/RenderWorld.h>
#include <limits>
#include <stdexcept>

namespace infernux
{

std::vector<RenderLightSource> CaptureRenderLights()
{
    std::vector<RenderLightSource> lights;
    for (const Light *light : SceneManager::Instance().GetActiveLights()) {
        const GameObject *object = light ? light->GetGameObject() : nullptr;
        const Transform *transform = object ? object->GetTransform() : nullptr;
        if (!light || !light->IsEnabled() || !object || !object->IsActiveInHierarchy() || !transform)
            continue;
        RenderLightSource value;
        value.id = object->GetID();
        value.type = static_cast<uint32_t>(light->GetLightType());
        value.position = transform->GetWorldPosition();
        value.direction = transform->GetWorldForward();
        value.right = transform->GetWorldRight();
        value.up = transform->GetWorldUp();
        value.linearColor = light->GetLinearColor();
        value.intensity = light->GetIntensity();
        value.range = light->GetRange();
        value.innerSpotAngle = light->GetSpotAngle();
        value.outerSpotAngle = light->GetOuterSpotAngle();
        value.areaSize = light->GetAreaSize();
        value.areaTwoSided = light->GetAreaTwoSided();
        value.shadows = static_cast<uint32_t>(light->GetShadows());
        value.shadowStrength = light->GetShadowStrength();
        value.shadowBias = light->GetShadowBias();
        value.shadowNormalBias = light->GetShadowNormalBias();
        value.shadowSoftness = light->GetShadowSoftness();
        value.cullingMask = light->GetCullingMask();
        value.influenceDomains = light->GetInfluenceDomains();
        lights.push_back(value);
    }
    return lights;
}

void SceneLightCollector::CollectLights(Scene *scene, const glm::vec3 &cameraPosition)
{
    RenderLightingSource source;
    if (scene)
        source.lights = CaptureRenderLights();
    Clear();
    source.ambientSky = m_lightingUBO.ambientSkyColor;
    source.ambientEquator = m_lightingUBO.ambientEquatorColor;
    source.ambientGround = m_lightingUBO.ambientGroundColor;
    CollectLights(source, cameraPosition);
}

void SceneLightCollector::CollectLights(const RenderLightingSource &source, const glm::vec3 &cameraPosition)
{
    Clear();
    SetAmbientLinear(source.ambientSky, source.ambientEquator, source.ambientGround);
    SetCameraPosition(cameraPosition);
    for (const auto &light : source.lights) {
        const glm::vec3 direction = glm::dot(light.direction, light.direction) > 0.0f ? glm::normalize(light.direction)
                                                                                      : glm::vec3(0.0f, -1.0f, 0.0f);
        CanonicalLightData canonical{};
        canonical.positionRange = glm::vec4(light.position, light.type == 0u ? 0.0f : std::max(light.range, 0.0f));
        canonical.directionOuterCos =
            glm::vec4(direction, light.type == 2u ? std::cos(glm::radians(light.outerSpotAngle * 0.5f)) : -1.0f);
        canonical.colorIntensity = glm::vec4(light.linearColor, std::max(light.intensity, 0.0f));
        canonical.shadowAndInnerCos =
            glm::vec4(light.shadowStrength, light.shadowBias, light.shadowNormalBias,
                      light.type == 2u ? std::cos(glm::radians(light.innerSpotAngle * 0.5f)) : -1.0f);
        if (light.type == 3u) {
            canonical.areaRightWidth = glm::vec4(glm::normalize(light.right), light.areaSize.x);
            canonical.areaUpHeight = glm::vec4(glm::normalize(light.up), light.areaSize.y);
            canonical.directionOuterCos.w = light.areaTwoSided ? 1.0f : 0.0f;
        }
        canonical.metadata = glm::uvec4(light.type, light.cullingMask, light.shadows, light.influenceDomains);
        canonical.identityAndShadow =
            glm::uvec4(static_cast<uint32_t>(light.id), static_cast<uint32_t>(light.id >> 32u), 0u, 0u);
        m_canonicalLightSnapshot.Add(canonical);
        const glm::vec4 color(light.linearColor, light.intensity);
        const glm::vec4 shadow(light.shadowStrength, light.shadowBias, light.shadowNormalBias, float(light.shadows));
        const glm::uvec4 metadata(light.cullingMask, light.influenceDomains, 0u, 0u);
        if (light.type == 0u && m_directionalLightCount < MAX_DIRECTIONAL_LIGHTS) {
            auto &data = m_lightingUBO.directionalLights[m_directionalLightCount];
            data.direction = glm::vec4(direction, 0.0f);
            data.color = color;
            data.shadowParams = shadow;
            data.metadata = metadata;
            m_directionalLightIds[m_directionalLightCount++] = light.id;
        } else if (light.type == 1u) {
            PointLightSortData sorted{};
            sorted.data.position = glm::vec4(light.position, light.range);
            sorted.data.color = color;
            sorted.data.attenuation = glm::vec4(light.range, 0.0f, 0.0f, 0.0f);
            sorted.data.shadowParams = shadow;
            sorted.data.metadata = metadata;
            sorted.lightId = light.id;
            m_pointLightSortBuffer.push_back(sorted);
        } else if (light.type == 2u && m_spotLightCount < MAX_SPOT_LIGHTS) {
            auto &data = m_lightingUBO.spotLights[m_spotLightCount];
            data.position = glm::vec4(light.position, light.range);
            data.direction = glm::vec4(direction, 0.0f);
            data.color = color;
            data.spotParams = glm::vec4(std::cos(glm::radians(light.innerSpotAngle * 0.5f)),
                                        std::cos(glm::radians(light.outerSpotAngle * 0.5f)), 0.0f, 0.0f);
            data.attenuation = glm::vec4(light.range, 0.0f, 0.0f, 0.0f);
            data.shadowParams = shadow;
            data.metadata = metadata;
            m_spotLightIds[m_spotLightCount++] = light.id;
        } else if (light.type == 3u && m_areaLightCount < MAX_AREA_LIGHTS) {
            auto &data = m_lightingUBO.areaLights[m_areaLightCount];
            data.positionRange = glm::vec4(light.position, light.range);
            data.direction = glm::vec4(direction, light.areaTwoSided ? 1.0f : 0.0f);
            data.rightWidth = glm::vec4(glm::normalize(light.right), light.areaSize.x);
            data.upHeight = glm::vec4(glm::normalize(light.up), light.areaSize.y);
            data.color = color;
            data.shadowParams = shadow;
            data.metadata = metadata;
            m_areaLightIds[m_areaLightCount++] = light.id;
        }
    }
    SortPointLightsByImportance(cameraPosition);
    m_lightingUBO.lightCounts =
        glm::ivec4(m_directionalLightCount, m_pointLightCount, m_spotLightCount, m_areaLightCount);
    PrepareSimpleLightingUBO();
}
void SceneLightCollector::Clear()
{
    m_canonicalLightSnapshot.Clear(m_canonicalGeneration);
    m_lightingUBO = LightingUBO{};
    m_simpleLightingUBO = SimpleLightingUBO{};
    m_directionalLightCount = 0;
    m_pointLightCount = 0;
    m_spotLightCount = 0;
    m_areaLightCount = 0;
    m_directionalLightIds.fill(0);
    m_pointLightIds.fill(0);
    m_spotLightIds.fill(0);
    m_areaLightIds.fill(0);
    m_pointLightSortBuffer.clear();
    m_shadowFrame = {};

    // Set default ambient (authored sRGB -> linear; alpha channels carry intensity/mode)
    m_lightingUBO.ambientSkyColor = glm::vec4(inx::color::SrgbToLinear(glm::vec3(0.2f, 0.2f, 0.3f)), 0.5f);
    m_lightingUBO.ambientGroundColor = glm::vec4(inx::color::SrgbToLinear(glm::vec3(0.1f, 0.1f, 0.1f)), 0.3f);
    m_lightingUBO.ambientEquatorColor =
        glm::vec4(inx::color::SrgbToLinear(glm::vec3(0.15f, 0.15f, 0.2f)), 0.0f); // mode = 0 (flat)
}

void SceneLightCollector::PublishCanonicalGeneration()
{
    const auto bytesEqual = [](const std::vector<CanonicalLightData> &left,
                               const std::vector<CanonicalLightData> &right) {
        return left.size() == right.size() &&
               (left.empty() || std::memcmp(left.data(), right.data(), left.size() * sizeof(CanonicalLightData)) == 0);
    };
    const bool changed = !m_hasCanonicalPublication ||
                         !bytesEqual(m_canonicalLightSnapshot.directionalLights, m_lastCanonicalDirectionalLights) ||
                         !bytesEqual(m_canonicalLightSnapshot.localLights, m_lastCanonicalLocalLights);
    if (changed) {
        ++m_canonicalGeneration;
        m_lastCanonicalDirectionalLights = m_canonicalLightSnapshot.directionalLights;
        m_lastCanonicalLocalLights = m_canonicalLightSnapshot.localLights;
        m_hasCanonicalPublication = true;
    }
    m_canonicalLightSnapshot.generation = m_canonicalGeneration;
}

void SceneLightCollector::SortPointLightsByImportance(const glm::vec3 &cameraPosition)
{
    // Calculate importance for each point light
    for (auto &sortData : m_pointLightSortBuffer) {
        glm::vec3 lightPos = glm::vec3(sortData.data.position);
        float range = sortData.data.position.w;
        float intensity = sortData.data.color.a;
        float distance = glm::length(lightPos - cameraPosition);

        // Importance = intensity / (distance + 1)^2, capped at range
        if (distance > range * 2.0f) {
            sortData.importance = 0.0f;
        } else {
            sortData.importance = intensity / ((distance + 1.0f) * (distance + 1.0f));
        }
    }

    // Sort by importance (highest first)
    std::sort(m_pointLightSortBuffer.begin(), m_pointLightSortBuffer.end(),
              [](const PointLightSortData &a, const PointLightSortData &b) { return a.importance > b.importance; });

    // Copy to UBO (up to MAX_POINT_LIGHTS)
    m_pointLightCount = std::min(static_cast<uint32_t>(m_pointLightSortBuffer.size()), MAX_POINT_LIGHTS);
    for (uint32_t i = 0; i < m_pointLightCount; ++i) {
        m_lightingUBO.pointLights[i] = m_pointLightSortBuffer[i].data;
        m_pointLightIds[i] = m_pointLightSortBuffer[i].lightId;
    }
}

void SceneLightCollector::PrepareSimpleLightingUBO()
{
    // Copy ambient
    m_simpleLightingUBO.ambientColor = m_lightingUBO.ambientSkyColor;

    // Main directional light (first directional light)
    if (m_directionalLightCount > 0) {
        m_simpleLightingUBO.mainLightDirection = m_lightingUBO.directionalLights[0].direction;
        m_simpleLightingUBO.mainLightColor = m_lightingUBO.directionalLights[0].color;
    } else {
        m_simpleLightingUBO.mainLightDirection = glm::vec4(0.0f, -1.0f, 0.0f, 0.0f);
        m_simpleLightingUBO.mainLightColor = glm::vec4(0.0f);
    }

    // Camera position
    m_simpleLightingUBO.cameraPosition = m_lightingUBO.worldSpaceCameraPos;

    // Point lights (up to 16 for simple mode)
    m_simpleLightingUBO.pointLightCount = static_cast<int>(std::min(m_pointLightCount, 16u));
    for (int i = 0; i < m_simpleLightingUBO.pointLightCount; ++i) {
        m_simpleLightingUBO.pointLights[i] = m_lightingUBO.pointLights[i];
    }
}

void SceneLightCollector::SetAmbientColor(const glm::vec3 &color, float intensity)
{
    m_lightingUBO.ambientSkyColor = glm::vec4(inx::color::SrgbToLinear(color), intensity);
    m_lightingUBO.ambientEquatorColor.a = 0.0f; // Flat mode
}

void SceneLightCollector::SetAmbientGradient(const glm::vec3 &skyColor, const glm::vec3 &equatorColor,
                                             const glm::vec3 &groundColor, float intensity)
{
    const float scale = std::max(intensity, 0.0f);
    m_lightingUBO.ambientSkyColor = glm::vec4(inx::color::SrgbToLinear(skyColor) * scale, 1.0f);
    m_lightingUBO.ambientEquatorColor =
        glm::vec4(inx::color::SrgbToLinear(equatorColor) * scale, 1.0f); // Gradient mode
    m_lightingUBO.ambientGroundColor = glm::vec4(inx::color::SrgbToLinear(groundColor) * scale, 1.0f);
}

void SceneLightCollector::SetAmbientLinear(const glm::vec4 &skyColor, const glm::vec4 &equatorColor,
                                           const glm::vec4 &groundColor)
{
    m_lightingUBO.ambientSkyColor = skyColor;
    m_lightingUBO.ambientEquatorColor = equatorColor;
    m_lightingUBO.ambientGroundColor = groundColor;
}

void SceneLightCollector::SetFog(bool enabled, const glm::vec3 &color, float density, float start, float end, int mode)
{
    m_lightingUBO.fogColor = glm::vec4(inx::color::SrgbToLinear(color), enabled ? 1.0f : 0.0f);
    m_lightingUBO.fogParams = glm::vec4(density, start, end, static_cast<float>(mode));
}

void SceneLightCollector::UpdateTime(float time, float deltaTime)
{
    m_lightingUBO.time = glm::vec4(time, std::sin(time), std::cos(time), deltaTime);
}

void SceneLightCollector::SetCameraPosition(const glm::vec3 &position)
{
    m_lightingUBO.worldSpaceCameraPos = glm::vec4(position, 1.0f);
    m_simpleLightingUBO.cameraPosition = glm::vec4(position, 1.0f);
}

void SceneLightCollector::ComputeShadowVP(Scene *scene, const glm::vec3 &cameraPos, float shadowMapResolution,
                                          const Camera *camera, const lighting::ShadowDepthRange &visibleDepthRange)
{
    if (!scene) {
        m_shadowFrame = {};
        return;
    }
    if (!camera)
        camera = SceneRenderBridge::Instance().GetEditorCamera();
    RenderLightingSource source;
    source.lights = CaptureRenderLights();
    RenderViewData view = camera ? SceneRenderExtractor::CaptureCameraView(*camera) : RenderViewData{};
    view.position = cameraPos;
    ComputeShadowVP(source, view, shadowMapResolution, visibleDepthRange);
}
void SceneLightCollector::ComputeShadowVP(const RenderLightingSource &source, const RenderViewData &view,
                                          float shadowMapResolution,
                                          const lighting::ShadowDepthRange &visibleDepthRange)
{
    m_shadowFrame = {};
    if (!std::isfinite(shadowMapResolution) || shadowMapResolution < 1.0f)
        return;
    m_shadowFrame.atlasSize = static_cast<uint32_t>(shadowMapResolution);

    lighting::ShadowCamera shadowCamera;
    shadowCamera.position = view.position;

    std::vector<const RenderLightSource *> shadowLights;
    for (const auto &light : source.lights)
        if (light.shadows != 0u)
            shadowLights.push_back(&light);
    std::stable_sort(shadowLights.begin(), shadowLights.end(),
                     [&](const RenderLightSource *left, const RenderLightSource *right) {
                         const auto typePriority = [](LightType type) {
                             return type == LightType::Directional ? 0u
                                    : type == LightType::Spot      ? 1u
                                    : type == LightType::Point     ? 2u
                                                                   : 3u;
                         };
                         const uint32_t leftType = typePriority(static_cast<LightType>(left->type));
                         const uint32_t rightType = typePriority(static_cast<LightType>(right->type));
                         if (leftType != rightType)
                             return leftType < rightType;
                         const float leftImportance = left->intensity * std::max(left->range, 1.0f);
                         const float rightImportance = right->intensity * std::max(right->range, 1.0f);
                         if (leftImportance != rightImportance)
                             return leftImportance > rightImportance;
                         return left->id < right->id;
                     });

    std::array<glm::dvec3, 4> rayOrigins{};
    std::array<glm::dvec3, 4> rayDepthSteps{};
    if (view.valid && !shadowLights.empty() &&
        static_cast<LightType>(shadowLights.front()->type) == LightType::Directional) {
        const auto inverseProjection = glm::inverse(glm::dmat4(view.projection));
        const glm::dmat4 cameraToWorld(glm::inverse(view.view));
        const std::array<glm::dvec2, 4> ndcCorners = {glm::dvec2(-1, -1), glm::dvec2(1, -1), glm::dvec2(1, 1),
                                                      glm::dvec2(-1, 1)};
        double nearDepth = std::numeric_limits<double>::infinity();
        double farDepth = 0.0;
        bool unboundedFar = false;
        for (size_t index = 0; index < ndcCorners.size(); ++index) {
            const auto a = inverseProjection * glm::dvec4(ndcCorners[index], 0, 1);
            const auto b = inverseProjection * glm::dvec4(ndcCorners[index], 1, 1);
            // Intersect the homogeneous pixel ray with view.z = depth.
            // Keeping its endpoints homogeneous handles infinite far planes
            // and oblique planes without dividing a point at infinity by w.
            const double denominator = a.w * b.z - b.w * a.z;
            if (denominator == 0.0)
                throw std::invalid_argument("Camera projection has an unbounded view-depth shadow slice");
            const auto origin = (glm::dvec3(a) * b.z - glm::dvec3(b) * a.z) / denominator;
            const auto depthStep = (glm::dvec3(b) * a.w - glm::dvec3(a) * b.w) / denominator;
            rayOrigins[index] = glm::dvec3(cameraToWorld * glm::dvec4(origin, 1));
            rayDepthSteps[index] = glm::dvec3(cameraToWorld * glm::dvec4(depthStep, 0));
            for (const auto &point : {a, b}) {
                if (point.w > 0.0) {
                    const double depth = point.z / point.w;
                    nearDepth = std::min(nearDepth, depth);
                    farDepth = std::max(farDepth, depth);
                } else {
                    // The positive-w part of this edge reaches infinity.
                    unboundedFar = true;
                }
            }
        }
        shadowCamera.nearClip = std::max(static_cast<float>(nearDepth), 0.001f);
        shadowCamera.farClip = unboundedFar ? std::numeric_limits<float>::infinity()
                                            : std::max(static_cast<float>(farDepth), shadowCamera.nearClip + 0.01f);
    }
    const auto frustumSliceAt = [&](float viewDepth) {
        std::array<glm::vec3, 4> corners{};
        for (size_t index = 0; index < corners.size(); ++index)
            corners[index] = glm::vec3(rayOrigins[index] + rayDepthSteps[index] * double(viewDepth));
        return corners;
    };

    lighting::ShadowAtlasAllocator atlas(m_shadowFrame.atlasSize);
    bool mainDirectional = true;
    for (const RenderLightSource *light : shadowLights) {
        if (m_shadowFrame.views.size() >= lighting::MaxShadowViews)
            break;
        const uint64_t lightId = light->id;
        const glm::vec3 position = light->position;
        const glm::vec3 direction = light->direction;
        const uint32_t firstView = static_cast<uint32_t>(m_shadowFrame.views.size());

        if (static_cast<LightType>(light->type) == LightType::Directional) {
            if (firstView + lighting::DirectionalCascadeCount > lighting::MaxShadowViews)
                continue;
            const uint32_t atlasSize = m_shadowFrame.atlasSize;
            // Unity-style equal cascade tiles: every cascade of a directional
            // light uses the same resolution, so shadow quality does not fall
            // off a cliff at a split boundary the way a 2:2:1:1 layout did.
            // The sole shadow light fills the whole atlas with a 2x2 grid;
            // with more shadow lights the main light keeps 3/8-atlas tiles
            // (56% of the area), leaving room for spot/point/secondary views.
            const bool soleShadowLight = shadowLights.size() == 1;
            const uint32_t mainTile = soleShadowLight ? atlasSize / 2u : atlasSize * 3u / 8u;
            const std::array<uint32_t, lighting::DirectionalCascadeCount> mainSizes{mainTile, mainTile, mainTile,
                                                                                    mainTile};
            const std::array<uint32_t, lighting::DirectionalCascadeCount> secondarySizes{
                atlasSize / 4u, atlasSize / 4u, atlasSize / 4u, atlasSize / 4u};
            const auto &sizes = mainDirectional ? mainSizes : secondarySizes;
            // Fit each camera to its visible depth range. Distant objects move
            // the logarithmic distribution outward without wasting close-up
            // resolution in ordinary scenes.
            const auto splits =
                lighting::AdaptiveCascadeSplits(shadowCamera.nearClip, visibleDepthRange, shadowCamera.farClip);
            const auto tiles = atlas.AllocateBatchWithFallback(sizes, std::max(atlasSize / 16u, 16u), 4);
            if (!tiles)
                continue;
            for (uint32_t cascade = 0; cascade < lighting::DirectionalCascadeCount; ++cascade) {
                if (view.valid) {
                    const auto nearSlice = frustumSliceAt(splits[cascade]);
                    const auto farSlice = frustumSliceAt(splits[cascade + 1]);
                    const std::array<glm::vec3, 8> sliceCorners = {nearSlice[0], nearSlice[1], nearSlice[2],
                                                                   nearSlice[3], farSlice[0],  farSlice[1],
                                                                   farSlice[2],  farSlice[3]};
                    m_shadowFrame.views.push_back(lighting::BuildStableDirectionalCascade(
                        lightId, cascade, sliceCorners, direction, splits[cascade], splits[cascade + 1],
                        (*tiles)[cascade]));
                } else {
                    m_shadowFrame.views.push_back(lighting::BuildStableDirectionalCascade(
                        lightId, cascade, shadowCamera, direction, splits[cascade], splits[cascade + 1],
                        (*tiles)[cascade]));
                }
            }
            mainDirectional = false;
        } else if (static_cast<LightType>(light->type) == LightType::Spot) {
            if (firstView + 1u > lighting::MaxShadowViews)
                continue;
            // Spend the available atlas budget on the active light first. A
            // lone spot should not be capped at quarter resolution; the
            // allocator still halves this request when higher-priority views
            // already occupy the atlas.
            const std::array<uint32_t, 1> preferredSize{m_shadowFrame.atlasSize / 2u};
            const auto tiles =
                atlas.AllocateBatchWithFallback(preferredSize, std::max(m_shadowFrame.atlasSize / 32u, 16u), 4);
            if (!tiles)
                continue;
            m_shadowFrame.views.push_back(lighting::BuildSpotShadowView(
                lightId, position, direction, light->outerSpotAngle, light->range, (*tiles)[0]));
        } else {
            constexpr uint32_t faceCount = 6u;
            if (firstView + faceCount > lighting::MaxShadowViews)
                continue;
            // Six quarter-atlas faces fit in an otherwise empty atlas. Under
            // pressure AllocateBatchWithFallback reduces the whole cubemap
            // uniformly, preserving equal resolution across every seam.
            const uint32_t faceSize = m_shadowFrame.atlasSize / 4u;
            const std::array<uint32_t, faceCount> faceSizes{faceSize, faceSize, faceSize, faceSize, faceSize, faceSize};
            const auto tiles =
                atlas.AllocateBatchWithFallback(faceSizes, std::max(m_shadowFrame.atlasSize / 32u, 16u), 4);
            if (!tiles)
                continue;
            const auto type = static_cast<LightType>(light->type) == LightType::Area
                                  ? lighting::ShadowViewType::AreaFace
                                  : lighting::ShadowViewType::PointFace;
            const auto views = lighting::BuildPointShadowViews(lightId, position, light->range, *tiles, type);
            m_shadowFrame.views.insert(m_shadowFrame.views.end(), views.begin(), views.end());
        }

        const uint32_t viewCount = static_cast<uint32_t>(m_shadowFrame.views.size()) - firstView;
        if (viewCount > 0) {
            for (uint32_t index = firstView; index < firstView + viewCount; ++index) {
                m_shadowFrame.views[index].cullingMask = light->cullingMask;
                m_shadowFrame.views[index].filterRadiusTexels = light->shadowSoftness;
                m_shadowFrame.views[index].depthBiasTexels = light->shadowBias;
                m_shadowFrame.views[index].normalBiasTexels = light->shadowNormalBias;
            }
            m_shadowFrame.assignments.push_back({lightId, firstView, viewCount});
        }
    }
}

void SceneLightCollector::BuildShaderLightingUBO()
{
    // Build shader-compatible UBO from full UBO data
    // This structure exactly matches lit.frag layout

    // Light counts
    m_shaderLightingUBO.lightCounts = m_lightingUBO.lightCounts;

    // Ambient data (flat + hemisphere/gradient probe)
    m_shaderLightingUBO.ambientColor = m_lightingUBO.ambientSkyColor;
    m_shaderLightingUBO.ambientSkyColor = m_lightingUBO.ambientSkyColor;
    m_shaderLightingUBO.ambientEquatorColor = m_lightingUBO.ambientEquatorColor;
    m_shaderLightingUBO.ambientGroundColor = m_lightingUBO.ambientGroundColor;

    // Camera position
    m_shaderLightingUBO.cameraPos = m_lightingUBO.worldSpaceCameraPos;

    // Copy directional lights
    for (uint32_t i = 0; i < MAX_DIRECTIONAL_LIGHTS; ++i) {
        m_shaderLightingUBO.directionalLights[i] = m_lightingUBO.directionalLights[i];
    }

    // Copy point lights
    for (uint32_t i = 0; i < MAX_POINT_LIGHTS; ++i) {
        m_shaderLightingUBO.pointLights[i] = m_lightingUBO.pointLights[i];
    }

    // Copy spot lights
    for (uint32_t i = 0; i < MAX_SPOT_LIGHTS; ++i) {
        m_shaderLightingUBO.spotLights[i] = m_lightingUBO.spotLights[i];
    }
    for (uint32_t i = 0; i < MAX_AREA_LIGHTS; ++i) {
        m_shaderLightingUBO.areaLights[i] = m_lightingUBO.areaLights[i];
    }

    const auto applyAssignment = [&](uint64_t lightId, glm::uvec4 &metadata) {
        if (const auto *assignment = m_shadowFrame.Find(lightId)) {
            metadata.z = assignment->firstView;
            metadata.w = assignment->viewCount;
        } else {
            metadata.z = 0;
            metadata.w = 0;
        }
    };
    for (uint32_t i = 0; i < m_directionalLightCount; ++i)
        applyAssignment(m_directionalLightIds[i], m_shaderLightingUBO.directionalLights[i].metadata);
    for (uint32_t i = 0; i < m_pointLightCount; ++i)
        applyAssignment(m_pointLightIds[i], m_shaderLightingUBO.pointLights[i].metadata);
    for (uint32_t i = 0; i < m_spotLightCount; ++i)
        applyAssignment(m_spotLightIds[i], m_shaderLightingUBO.spotLights[i].metadata);
    for (uint32_t i = 0; i < m_areaLightCount; ++i)
        applyAssignment(m_areaLightIds[i], m_shaderLightingUBO.areaLights[i].metadata);

    auto applyCanonicalAssignments = [&](std::vector<CanonicalLightData> &lights) {
        for (auto &light : lights) {
            const uint64_t lightId = static_cast<uint64_t>(light.identityAndShadow.x) |
                                     (static_cast<uint64_t>(light.identityAndShadow.y) << 32u);
            if (const auto *assignment = m_shadowFrame.Find(lightId)) {
                light.identityAndShadow.z = assignment->firstView;
                light.identityAndShadow.w = assignment->viewCount;
            } else {
                light.identityAndShadow.z = 0;
                light.identityAndShadow.w = 0;
            }
        }
    };
    applyCanonicalAssignments(m_canonicalLightSnapshot.directionalLights);
    applyCanonicalAssignments(m_canonicalLightSnapshot.localLights);
    PublishCanonicalGeneration();

    const uint32_t shadowViewCount =
        std::min<uint32_t>(static_cast<uint32_t>(m_shadowFrame.views.size()), lighting::MaxShadowViews);
    m_shaderLightingUBO.shadowViewHeader =
        glm::uvec4(shadowViewCount, m_shadowFrame.atlasSize, static_cast<uint32_t>(m_canonicalLightSnapshot.generation),
                   static_cast<uint32_t>(m_canonicalLightSnapshot.generation >> 32u));
    for (uint32_t index = 0; index < lighting::MaxShadowViews; ++index) {
        auto &gpuView = m_shaderLightingUBO.shadowViews[index];
        if (index >= shadowViewCount) {
            gpuView = {};
            continue;
        }
        const auto &view = m_shadowFrame.views[index];
        gpuView.viewProjection = view.viewProjection;
        gpuView.atlasScaleOffset = view.atlas.ScaleOffset(m_shadowFrame.atlasSize);
        gpuView.depthTexel = glm::vec4(view.nearPlane, view.farPlane, view.worldUnitsPerTexel, view.filterRadiusTexels);
        gpuView.splitData = glm::vec4(view.splitNear, view.splitFar, 0.0f, 0.0f);
        gpuView.metadata = glm::uvec4(static_cast<uint32_t>(view.type), view.subView, 0u, 0u);
    }
}

} // namespace infernux
