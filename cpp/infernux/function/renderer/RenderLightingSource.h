#pragma once

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <vector>

namespace infernux
{
class InxMaterial;

// All values read by scene lighting and shadow fitting. No Component or
// Transform identity is required after this source has been captured.
struct RenderLightSource
{
    uint64_t id = 0;
    uint32_t type = 0;
    glm::vec3 position{0.0f};
    glm::vec3 direction{0.0f, 0.0f, 1.0f};
    glm::vec3 right{1.0f, 0.0f, 0.0f};
    glm::vec3 up{0.0f, 1.0f, 0.0f};
    glm::vec3 linearColor{1.0f};
    float intensity = 1.0f;
    float range = 10.0f;
    float innerSpotAngle = 30.0f;
    float outerSpotAngle = 45.0f;
    glm::vec2 areaSize{1.0f};
    bool areaTwoSided = false;
    uint32_t shadows = 0;
    float shadowStrength = 1.0f;
    float shadowBias = 1.0f;
    float shadowNormalBias = 1.0f;
    float shadowSoftness = 1.0f;
    uint32_t cullingMask = 0xFFFFFFFFu;
    uint32_t influenceDomains = 3u;

    bool operator==(const RenderLightSource &other) const noexcept
    {
        return id == other.id && type == other.type && position == other.position && direction == other.direction &&
               right == other.right && up == other.up && linearColor == other.linearColor &&
               intensity == other.intensity && range == other.range && innerSpotAngle == other.innerSpotAngle &&
               outerSpotAngle == other.outerSpotAngle && areaSize == other.areaSize &&
               areaTwoSided == other.areaTwoSided && shadows == other.shadows &&
               shadowStrength == other.shadowStrength && shadowBias == other.shadowBias &&
               shadowNormalBias == other.shadowNormalBias && shadowSoftness == other.shadowSoftness &&
               cullingMask == other.cullingMask && influenceDomains == other.influenceDomains;
    }
};

struct alignas(16) LightSamplingStratum
{
    glm::vec3 lower{};
    uint32_t dominant = 0;
    glm::vec3 upper{};
    uint32_t flags = 0; // bit 0: contains directional; bit 1: equal power
};
static_assert(sizeof(LightSamplingStratum) == 32);

// Canonical directional-first order. Shared by every View of this immutable World publication.
struct LightSamplingDistribution
{
    std::vector<LightSamplingStratum> strata;
    std::vector<float> cumulative;
};
std::shared_ptr<const LightSamplingDistribution>
BuildLightSamplingDistribution(const std::vector<RenderLightSource> &lights);

struct RenderLightingSource
{
    // Stable generation advances only when actual source values change.
    uint64_t generation = 0;
    /// Last abrupt change in illumination shared by distant receivers. Persists across skipped views.
    uint64_t diffuseCutGeneration = 0;
    /// Last discontinuity in the light distribution sampled by reusable paths.
    uint64_t proposalCutGeneration = 0;
    std::array<glm::vec3, 6> distantDiffuse{};
    void UpdateDiffuseHistory(const RenderLightingSource *previous);
    std::vector<RenderLightSource> lights;
    std::shared_ptr<const LightSamplingDistribution> sampling;
    glm::vec4 ambientSky{0.0f};
    glm::vec4 ambientEquator{0.0f};
    glm::vec4 ambientGround{0.0f};
    // Render-owned material copy: source authoring cannot mutate its values.
    std::shared_ptr<InxMaterial> skyboxMaterial;

    bool SameValues(const RenderLightingSource &other) const noexcept
    {
        return lights == other.lights && ambientSky == other.ambientSky && ambientEquator == other.ambientEquator &&
               ambientGround == other.ambientGround && skyboxMaterial == other.skyboxMaterial;
    }
};
// Scene-side capture boundary. Call only in the intended World scope.
std::vector<RenderLightSource> CaptureRenderLights();
} // namespace infernux
