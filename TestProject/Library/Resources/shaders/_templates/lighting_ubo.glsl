// ============================================================================
// lighting_ubo.glsl — LightingUBO and shadow map sampler for lit shaders
//
// Matches ShaderLightingUBO layout in C++ (InfLight.h / InfRenderer).
// ============================================================================

#define MAX_DIRECTIONAL_LIGHTS 4
#define MAX_POINT_LIGHTS 64
#define MAX_SPOT_LIGHTS 32

struct DirectionalLightData {
    vec4 direction;      // xyz = direction, w = unused
    vec4 color;          // xyz = color, w = intensity
    vec4 shadowParams;   // x = shadow strength, y = shadow bias, zw = unused
};

struct PointLightData {
    vec4 position;       // xyz = position, w = range
    vec4 color;          // xyz = color, w = intensity
    vec4 attenuation;    // x = constant, y = linear, z = quadratic, w = unused
};

struct SpotLightData {
    vec4 position;       // xyz = position, w = range
    vec4 direction;      // xyz = direction, w = unused
    vec4 color;          // xyz = color, w = intensity
    vec4 spotParams;     // x = inner angle cos, y = outer angle cos, zw = unused
    vec4 attenuation;    // x = constant, y = linear, z = quadratic, w = unused
};

layout(std140, binding = 1) uniform LightingUBO {
    ivec4 lightCounts;   // x = directional, y = point, z = spot, w = unused
    vec4 ambientColor;   // xyz = flat/legacy ambient color, w = ambient intensity
    vec4 ambientSkyColor;     // xyz = sky ambient color, w = intensity
    vec4 ambientEquatorColor; // xyz = equator color, w = ambient mode
    vec4 ambientGroundColor;  // xyz = ground ambient color, w = intensity
    vec4 cameraPos;      // xyz = camera world position, w = unused
    DirectionalLightData directionalLights[MAX_DIRECTIONAL_LIGHTS];
    PointLightData pointLights[MAX_POINT_LIGHTS];
    SpotLightData spotLights[MAX_SPOT_LIGHTS];
    mat4 lightVP[4];     // Light view-projection matrices per cascade
    vec4 shadowCascadeSplits; // x,y,z,w = cascade split distances (view-space Z)
    vec4 shadowMapParams;     // x = resolution, y = enabled(1/0), z = numCascades, w = unused
} lighting;

// Shadow map sampler (per-view descriptor set 1)
layout(set = 1, binding = 0) uniform sampler2D shadowMap;
