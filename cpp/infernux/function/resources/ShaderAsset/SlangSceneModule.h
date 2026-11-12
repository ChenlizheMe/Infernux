#pragma once

#include "SlangCompiler.h"

namespace infernux
{

/// Native scene stages use the shared compiler on every graphics target.
/// The host owns resources and fixed-function state; this module owns only
/// the portable vertex, lighting, sky and shadow calculations.
inline const ShaderSourceModule &GetSlangSceneModule()
{
    static const ShaderSourceModule module{"inx_scene", "<infernux/renderer/scene>", R"slang(

struct PunctualLightData {
    float4 position_range;
    float4 color_intensity;
    float4 direction_outer_cos;
    float4 parameters;
};

struct CameraData {
    column_major float4x4 view_projection;
    column_major float4x4 inverse_view_projection;
    column_major float4x4 light_view_projection;
    float4 camera_position;
    float4 light_direction_strength;
    float4 light_color_intensity;
    float4 sky_top_exposure;
    float4 sky_horizon;
    float4 sky_ground;
    float4 ambient;
    float4 ambient_sky;
    float4 ambient_equator;
    float4 ambient_ground;
    uint4 light_counts;
    PunctualLightData punctual_lights[8];
};

[[vk::binding(0, 0)]] ConstantBuffer<CameraData> camera;
[[vk::binding(1, 0)]] DepthTexture2D shadow_map;
[[vk::binding(2, 0)]] SamplerComparisonState shadow_sampler;

struct DrawData {
    column_major float4x4 model;
    column_major float4x4 normal;
};
[[vk::binding(3, 0)]] StructuredBuffer<DrawData> draws;

struct VertexInput {
    [[vk::location(0)]] float3 position : TEXCOORD0;
    [[vk::location(1)]] float3 normal : TEXCOORD1;
    [[vk::location(2)]] float2 uv : TEXCOORD2;
    [[vk::location(3)]] float4 color : TEXCOORD3;
    [[vk::location(4)]] float4 emission : TEXCOORD4;
    [[vk::location(5)]] float4 material : TEXCOORD5;
    [[vk::location(6)]] float4 surface : TEXCOORD6;
    [[vk::location(7)]] float4 tangent : TEXCOORD7;
    [[vk::location(8)]] float2 uv1 : TEXCOORD8;
    [[vk::location(9)]] float4 texture_uv_sets0 : TEXCOORD9;
    [[vk::location(10)]] float2 texture_uv_sets1 : TEXCOORD10;
    [[vk::location(11)]] float4 metallic_channels : TEXCOORD11;
    [[vk::location(12)]] float4 smoothness_channels : TEXCOORD12;
    [[vk::location(13)]] float2 material_sampling : TEXCOORD13;
    uint draw_index : SV_InstanceID;
};

struct VertexOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 normal : TEXCOORD0;
    [[vk::location(1)]] float4 color : TEXCOORD1;
    [[vk::location(2)]] float3 world_position : TEXCOORD2;
    [[vk::location(3)]] float4 shadow_position : TEXCOORD3;
    [[vk::location(4)]] float4 emission : TEXCOORD4;
    [[vk::location(5)]] float4 material : TEXCOORD5;
    [[vk::location(6)]] float4 surface : TEXCOORD6;
    [[vk::location(7)]] float2 uv : TEXCOORD7;
    [[vk::location(8)]] float4 tangent : TEXCOORD8;
    [[vk::location(9)]] float2 uv1 : TEXCOORD9;
    [[vk::location(10)]] float4 texture_uv_sets0 : TEXCOORD10;
    [[vk::location(11)]] float2 texture_uv_sets1 : TEXCOORD11;
    [[vk::location(12)]] float4 metallic_channels : TEXCOORD12;
    [[vk::location(13)]] float4 smoothness_channels : TEXCOORD13;
    [[vk::location(14)]] float2 material_sampling : TEXCOORD14;
};

[[vk::binding(0, 1)]] SamplerState material_base_color_sampler;
[[vk::binding(1, 1)]] Texture2D<float4> material_base_color;
[[vk::binding(2, 1)]] SamplerState material_metallic_sampler;
[[vk::binding(3, 1)]] Texture2D<float4> material_metallic_map;
[[vk::binding(4, 1)]] SamplerState material_smoothness_sampler;
[[vk::binding(5, 1)]] Texture2D<float4> material_smoothness_map;
[[vk::binding(6, 1)]] SamplerState material_ao_sampler;
[[vk::binding(7, 1)]] Texture2D<float4> material_ao_map;
[[vk::binding(8, 1)]] SamplerState material_normal_sampler;
[[vk::binding(9, 1)]] Texture2D<float4> material_normal_map;
[[vk::binding(10, 1)]] SamplerState material_emission_sampler;
[[vk::binding(11, 1)]] Texture2D<float4> material_emission_map;

uint2 shadow_dimensions() { uint w,h; shadow_map.GetDimensions(w,h); return uint2(w,h); }

[shader("vertex")]
VertexOutput scene_vertex(VertexInput input) {
    VertexOutput output;
    var draw = draws[input.draw_index];
    var world_position = mul(draw.model, float4(input.position, 1.0));
    output.position = mul(camera.view_projection, world_position);
    output.normal = normalize((mul(draw.normal, float4(input.normal, 0.0))).xyz);
    output.color = input.color;
    output.world_position = world_position.xyz;
    output.shadow_position = mul(camera.light_view_projection, world_position);
    output.emission = input.emission;
    output.material = input.material;
    output.surface = input.surface;
    output.uv = input.uv;
    output.tangent = float4(normalize((mul(draw.model, float4(input.tangent.xyz, 0.0))).xyz), input.tangent.w);
    output.uv1 = input.uv1;
    output.texture_uv_sets0 = input.texture_uv_sets0;
    output.texture_uv_sets1 = input.texture_uv_sets1;
    output.metallic_channels = input.metallic_channels;
    output.smoothness_channels = input.smoothness_channels;
    output.material_sampling = input.material_sampling;
    return output;
}

static const float PI = 3.14159265358979323846;

float3 fresnel_schlick(float3 f0, float f90, float u) {
    var x = 1.0 - u;
    var x2 = x * x;
    var x5 = x * x2 * x2;
    return f0 * (1.0 - x5) + float3(f90 * x5);
}

float3 fresnel_schlick_roughness(float3 f0, float f90, float ndotv, float perceptual_roughness) {
    // Match PBR.glsl F_SchlickRoughness exactly. f90 is part of the shared
    // signature, but HDRP's roughness-aware indirect Fresnel uses a white
    // grazing target attenuated by perceptual roughness rather than scaling
    // that target by the material's direct-light f90.
    var grazing = max(float3(1.0 - perceptual_roughness), f0);
    var x = 1.0 - ndotv;
    var x2 = x * x;
    var x5 = x * x2 * x2;
    return f0 + (grazing - f0) * x5;
}

float2 env_brdf_approx(float perceptual_roughness, float ndotv) {
    var c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    var c1 = float4(1.0, 0.0425, 1.04, -0.04);
    var r = perceptual_roughness * c0 + c1;
    var a004 = min(r.x * r.x, exp2(-9.28 * ndotv)) * r.x + r.y;
    return float2(-1.04, 1.04) * a004 + r.zw;
}

float3 sky_gradient(float y, float3 sky, float3 equator, float3 ground) {
    var base = lerp(ground, sky, smoothstep(-0.10, 0.45, y));
    var horizon_mask = 0.0;
    if (y < 0.0) {
        var side = 1.0 - smoothstep(0.0, 0.10, -y);
        horizon_mask = side * side;
    } else {
        horizon_mask = 1.0 - smoothstep(0.0, 0.45, y);
    }
    return lerp(base, equator, horizon_mask * 0.35);
}

float3 sample_ambient_probe(float3 direction) {
    if (camera.ambient_equator.w < 0.5) {
        return max(camera.ambient.rgb * camera.ambient.w, float3(0.0));
    }
    return max(sky_gradient(direction.y, camera.ambient_sky.rgb,
                            camera.ambient_equator.rgb, camera.ambient_ground.rgb),
               float3(0.0));
}

float3 sample_ambient_irradiance(float3 direction) {
    if (camera.ambient_equator.w < 0.5) {
        return max(camera.ambient.rgb * camera.ambient.w, float3(0.0));
    }
    var y = clamp(direction.y, -1.0, 1.0);
    if (y >= 0.0) {
        return max(lerp(camera.ambient_equator.rgb, camera.ambient_sky.rgb, y), float3(0.0));
    }
    return max(lerp(camera.ambient_equator.rgb, camera.ambient_ground.rgb, -y), float3(0.0));
}

float3 specular_ambient_direction(float3 normal, float3 view_direction, float perceptual_roughness) {
    var reflection = reflect(-view_direction, normal);
    var roughness = perceptual_roughness * perceptual_roughness;
    var factor = (1.0 - roughness) * (sqrt(max(1.0 - roughness, 0.0)) + roughness);
    return normalize(lerp(normal, reflection, clamp(factor, 0.0, 1.0)));
}

float specular_occlusion(float ndotv, float occlusion, float perceptual_roughness) {
    return clamp(pow(ndotv + occlusion, exp2(-16.0 * perceptual_roughness - 1.0))
                 - 1.0 + occlusion, 0.0, 1.0);
}

float geometric_specular_aa(float3 world_normal, float roughness) {
    var du = ddx(world_normal);
    var dv = ddy(world_normal);
    var variance = 0.25 * (dot(du, du) + dot(dv, dv));
    var kernel_roughness = min(variance, 0.18);
    return max(roughness, sqrt(roughness * roughness + kernel_roughness));
}

float horizon_occlusion(float3 reflection_direction, float3 geometric_normal) {
    var horizon = clamp(1.0 + dot(reflection_direction, geometric_normal), 0.0, 1.0);
    return horizon * horizon;
}

float disney_diffuse(float ndotv, float ndotl, float ldotv, float perceptual_roughness) {
    var fd90 = 0.5 + perceptual_roughness * (1.0 + ldotv);
    var light_x = 1.0 - ndotl;
    var view_x = 1.0 - ndotv;
    var light_scatter = 1.0 + (fd90 - 1.0) * light_x * light_x * light_x * light_x * light_x;
    var view_scatter = 1.0 + (fd90 - 1.0) * view_x * view_x * view_x * view_x * view_x;
    return (1.0 / PI) * (1.0 / 1.03571) * light_scatter * view_scatter;
}

float dv_smith_joint_ggx(float ndoth, float ndotl, float ndotv, float roughness) {
    var a2 = roughness * roughness;
    var s = (ndoth * a2 - ndoth) * ndoth + 1.0;
    var lambda_v = ndotl * sqrt(max((-ndotv * a2 + ndotv) * ndotv + a2, 1.0e-7));
    var lambda_l = ndotv * sqrt(max((-ndotl * a2 + ndotl) * ndotl + a2, 1.0e-7));
    return (1.0 / PI) * 0.5 * a2 / max(s * s * (lambda_v + lambda_l), 1.0e-7);
}

float3 evaluate_pbr_light(float3 normal, float3 view_direction, float3 light_direction, float3 radiance, float3 albedo, float metallic, float perceptual_roughness, float roughness, float3 f0, float f90, float3 energy_compensation, float specular_highlights) {
    var half_vector_sum = view_direction + light_direction;
    var half_vector = normal;
    if (dot(half_vector_sum, half_vector_sum) > 1.0e-8) {
        half_vector = normalize(half_vector_sum);
    }
    var ndotl = max(dot(normal, light_direction), 0.0);
    if (ndotl <= 0.0) {
        return float3(0.0);
    }
    var ndotv = max(dot(normal, view_direction), 0.0);
    var ndoth = max(dot(normal, half_vector), 0.0);
    var ldoth = max(dot(light_direction, half_vector), 0.0);
    var ldotv = max(dot(light_direction, view_direction), 0.0);
    var fresnel = fresnel_schlick(f0, f90, ldoth);
    var specular = fresnel * dv_smith_joint_ggx(ndoth, ndotl, ndotv, roughness)
                   * energy_compensation * clamp(specular_highlights, 0.0, 1.0);
    var diffuse = albedo * (1.0 - metallic) *
                  disney_diffuse(ndotv, ndotl, ldotv, perceptual_roughness);
    return (diffuse + specular) * radiance * (ndotl * PI);
}

float punctual_attenuation(float distance_to_light, float range) {
    var safe_range = max(range, 1.0e-4);
    var distance_squared = distance_to_light * distance_to_light;
    var ratio_squared = distance_squared / (safe_range * safe_range);
    var factor = clamp(1.0 - ratio_squared * ratio_squared, 0.0, 1.0);
    return factor * factor / (distance_squared + 1.0);
}

float toon_step(float value, float threshold, float softness) {
    var width = max(softness, 0.0001);
    return smoothstep(threshold - width, threshold + width, value);
}

float3 evaluate_toon_light(float3 normal, float3 view_direction, float3 light_direction, float3 radiance, float3 albedo, float threshold, float softness, float smoothness, float specular_strength) {
    var ndotl = max(dot(normal, light_direction), 0.0);
    var diffuse_band = toon_step(ndotl, threshold, softness);
    var half_sum = view_direction + light_direction;
    var half_vector = normal;
    if (dot(half_sum, half_sum) > 1.0e-8) {
        half_vector = normalize(half_sum);
    }
    var exponent = lerp(12.0, 256.0, clamp(smoothness, 0.0, 1.0));
    var raw_specular = pow(max(dot(normal, half_vector), 0.0), exponent);
    var specular_band = toon_step(raw_specular, 0.5, softness * 0.5);
    var diffuse = albedo * lerp(0.18, 1.0, diffuse_band);
    var specular = float3(specular_band * clamp(specular_strength, 0.0, 1.0));
    return (diffuse + specular) * radiance * (ndotl >= 0.0001 ? 1.0 : 0.0);
}

float sample_shadow(float4 position, float3 normal) {
    if (camera.light_direction_strength.w <= 0.0 || position.w <= 0.0) {
        return 1.0;
    }
    var ndc = position.xyz / position.w;
    var uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (ndc.z <= 0.0 || ndc.z >= 1.0 || any(uv < float2(0.0)) || any(uv > float2(1.0))) {
        return 1.0;
    }
    var toward_light = normalize(camera.light_direction_strength.xyz);
    var normal_bias = 0.0015 * (1.0 - max(dot(normal, toward_light), 0.0));
    var texel = 1.0 / float2(shadow_dimensions());
    var visibility = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            visibility += shadow_map.SampleCmpLevelZero(shadow_sampler, uv + float2(float(x), float(y)) * texel, ndc.z + 0.0008 + normal_bias);
        }
    }
    var filtered = visibility / 9.0;
    return lerp(1.0 - camera.light_direction_strength.w, 1.0, filtered);
}

float2 material_uv(VertexOutput input, float selector) {
    return (selector >= 0.5 ? input.uv1 : input.uv);
}

float3 sample_material_normal(VertexOutput input, float3 geometric_normal) {
    var tangent = input.tangent.xyz - geometric_normal * dot(input.tangent.xyz, geometric_normal);
    if (dot(tangent, tangent) <= 1.0e-8) {
        var fallback = (abs(geometric_normal.y) > 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0));
        tangent = normalize(cross(fallback, geometric_normal));
    } else {
        tangent = normalize(tangent);
    }
    var handedness = (input.tangent.w >= 0.0 ? 1.0 : -1.0);
    var bitangent = normalize(cross(geometric_normal, tangent)) * handedness;
    var uv = material_uv(input, input.texture_uv_sets1.x);
    var tangent_normal = material_normal_map.Sample(material_normal_sampler, uv).rgb * 2.0 - 1.0;
    tangent_normal.x *= abs(input.tangent.w);
    tangent_normal.y *= abs(input.tangent.w);
    tangent_normal = normalize(tangent_normal);
    var authored_normal = normalize(tangent * tangent_normal.x + bitangent * tangent_normal.y +
                                    geometric_normal * tangent_normal.z);

    // Imported tangents are authored for UV0. Match the desktop Lit contract
    // by reconstructing the UV1 basis from fragment derivatives. Derivatives
    // stay outside control flow so WebGPU uniformity validation remains strict.
    var position_dx = ddx(input.world_position);
    var position_dy = ddy(input.world_position);
    var uv_dx = ddx(uv);
    var uv_dy = ddy(uv);
    var dy_perpendicular = cross(position_dy, geometric_normal);
    var dx_perpendicular = cross(geometric_normal, position_dx);
    var derivative_tangent = dy_perpendicular * uv_dx.x + dx_perpendicular * uv_dy.x;
    var derivative_bitangent = dy_perpendicular * uv_dx.y + dx_perpendicular * uv_dy.y;
    var basis_length_sq = max(dot(derivative_tangent, derivative_tangent),
                              dot(derivative_bitangent, derivative_bitangent));
    var inverse_basis_length = rsqrt(max(basis_length_sq, 1.0e-12));
    var derivative_normal = normalize(derivative_tangent * inverse_basis_length * tangent_normal.x +
                                      derivative_bitangent * inverse_basis_length * tangent_normal.y +
                                      geometric_normal * tangent_normal.z);
    var use_derivative_basis = input.texture_uv_sets1.x >= 0.5 && basis_length_sq > 1.0e-12;
    return (use_derivative_basis ? derivative_normal : authored_normal);
}

[shader("fragment")]
float4 scene_fragment(VertexOutput input) : SV_Target0 {
    var geometric_normal = normalize(input.normal);
    var normal = sample_material_normal(input, geometric_normal);
    var sampled_color = material_base_color.Sample(material_base_color_sampler, material_uv(input, input.texture_uv_sets0.x));
    var surface_color = input.color * sampled_color;
    if (input.surface.w >= 0.0 && surface_color.a < input.surface.w) {
        discard;
    }
    var sampled_metallic_texel = material_metallic_map.Sample(material_metallic_sampler, material_uv(input, input.texture_uv_sets0.y));
    var sampled_metallic = dot(sampled_metallic_texel, input.metallic_channels);
    var sampled_smoothness_texel = material_smoothness_map.Sample(material_smoothness_sampler, material_uv(input, input.texture_uv_sets0.z));
    var sampled_smoothness_channel = dot(sampled_smoothness_texel, input.smoothness_channels);
    var direct_smoothness = sampled_smoothness_channel * input.material.y;
    var roughness_smoothness = 1.0 - sampled_smoothness_channel * (1.0 - input.material.y);
    var sampled_smoothness = lerp(direct_smoothness, roughness_smoothness,
                                 clamp(input.material_sampling.x, 0.0, 1.0));
    var sampled_occlusion = material_ao_map.Sample(material_ao_sampler, material_uv(input, input.texture_uv_sets0.w)).r;
    var occlusion_strength = clamp(input.material_sampling.y, 0.0, 1.0);
    var material_occlusion = lerp(1.0, sampled_occlusion, occlusion_strength);
    var sampled_emission = material_emission_map.Sample(material_emission_sampler, material_uv(input, input.texture_uv_sets1.y)).rgb;
    var emission = sampled_emission * input.emission.rgb * input.emission.a;
    var perceptual_roughness = clamp(1.0 - sampled_smoothness, 0.045, 1.0);
    var roughness = perceptual_roughness * perceptual_roughness;
    // Derivatives must execute in uniform fragment control flow on WebGPU.
    // Evaluate geometric AA before the per-material unlit early return.
    roughness = geometric_specular_aa(normal, roughness);
    perceptual_roughness = max(perceptual_roughness, sqrt(roughness));
    if (input.surface.x > 0.5 && input.surface.x < 1.5) {
        return float4(surface_color.rgb + emission, surface_color.a);
    }
    var view_direction = normalize(camera.camera_position.xyz - input.world_position);
    if (input.surface.x >= 1.5) {
        var toward_light = normalize(camera.light_direction_strength.xyz);
        var shadow = sample_shadow(input.shadow_position, normal);
        var radiance = camera.light_color_intensity.rgb * camera.light_color_intensity.w;
        var direct = evaluate_toon_light(
            normal, view_direction, toward_light, radiance, surface_color.rgb,
            clamp(input.surface.y, 0.0, 1.0), clamp(input.surface.z, 0.0, 0.25),
            input.material.y, input.material.w) * shadow;
        for (var light_index = 0u; light_index < camera.light_counts.x; light_index = light_index + 1u) {
            var light = camera.punctual_lights[light_index];
            if (light.parameters.z > 0.5) {
                direct += evaluate_toon_light(
                    normal, view_direction, -normalize(light.direction_outer_cos.xyz),
                    light.color_intensity.rgb * light.color_intensity.w,
                    surface_color.rgb, clamp(input.surface.y, 0.0, 1.0),
                    clamp(input.surface.z, 0.0, 0.25), input.material.y, input.material.w);
                continue;
            }
            var to_light = light.position_range.xyz - input.world_position;
            var distance_to_light = length(to_light);
            if (distance_to_light <= 1.0e-5 || distance_to_light >= light.position_range.w) {
                continue;
            }
            var local_direction = to_light / distance_to_light;
            var cone = 1.0;
            if (light.parameters.y > 0.5) {
                var theta = dot(local_direction, -normalize(light.direction_outer_cos.xyz));
                cone = clamp((theta - light.direction_outer_cos.w) /
                             max(light.parameters.x - light.direction_outer_cos.w, 1.0e-4), 0.0, 1.0);
            }
            var attenuation = punctual_attenuation(distance_to_light, light.position_range.w) * cone;
            direct += evaluate_toon_light(
                normal, view_direction, local_direction,
                light.color_intensity.rgb * light.color_intensity.w * attenuation,
                surface_color.rgb, clamp(input.surface.y, 0.0, 1.0),
                clamp(input.surface.z, 0.0, 0.25), input.material.y, input.material.w);
        }
        var ambient = sample_ambient_irradiance(normal) * surface_color.rgb
                      * clamp(input.material.z * material_occlusion, 0.0, 1.0) * 0.65;
        return float4(ambient + direct + emission,
                         surface_color.a);
    }
    var ndotv = max(dot(normal, view_direction), 0.0);
    var metallic = clamp(input.material.x * sampled_metallic, 0.0, 1.0);
    var occlusion = clamp(input.material.z * material_occlusion, 0.0, 1.0);
    var specular_highlights = clamp(input.material.w, 0.0, 1.0);
    var f0 = lerp(float3(0.04), surface_color.rgb, float3(metallic));
    var f90 = clamp(50.0 * dot(f0, float3(0.33333)), 0.0, 1.0);
    var env_brdf = env_brdf_approx(perceptual_roughness, ndotv);
    var reflectivity = env_brdf.x + env_brdf.y;
    var energy_compensation = float3(1.0) + f0 * (1.0 / max(reflectivity, 0.001) - 1.0);
    var toward_light = normalize(camera.light_direction_strength.xyz);
    var shadow = sample_shadow(input.shadow_position, normal);
    var radiance = camera.light_color_intensity.rgb * camera.light_color_intensity.w;
    var direct = evaluate_pbr_light(normal, view_direction, toward_light, radiance, surface_color.rgb,
                                    metallic, perceptual_roughness, roughness, f0, f90,
                                    energy_compensation, specular_highlights) * shadow;

    for (var light_index = 0u; light_index < camera.light_counts.x; light_index = light_index + 1u) {
        var light = camera.punctual_lights[light_index];
        if (light.parameters.z > 0.5) {
            direct += evaluate_pbr_light(
                normal, view_direction, -normalize(light.direction_outer_cos.xyz),
                light.color_intensity.rgb * light.color_intensity.w, surface_color.rgb,
                metallic, perceptual_roughness, roughness, f0, f90,
                energy_compensation, specular_highlights);
            continue;
        }
        var to_light = light.position_range.xyz - input.world_position;
        var distance_to_light = length(to_light);
        if (distance_to_light <= 1.0e-5 || distance_to_light >= light.position_range.w) {
            continue;
        }
        var local_direction = to_light / distance_to_light;
        var cone = 1.0;
        if (light.parameters.y > 0.5) {
            var theta = dot(local_direction, -normalize(light.direction_outer_cos.xyz));
            cone = clamp((theta - light.direction_outer_cos.w) /
                         max(light.parameters.x - light.direction_outer_cos.w, 1.0e-4), 0.0, 1.0);
        }
        var attenuation = punctual_attenuation(distance_to_light, light.position_range.w) * cone;
        var local_radiance = light.color_intensity.rgb * light.color_intensity.w * attenuation;
        direct += evaluate_pbr_light(normal, view_direction, local_direction, local_radiance,
                                     surface_color.rgb, metallic, perceptual_roughness, roughness, f0, f90,
                                     energy_compensation, specular_highlights);
    }
    var diffuse_irradiance = sample_ambient_irradiance(normal);
    var reflection_direction = specular_ambient_direction(normal, view_direction, perceptual_roughness);
    var prefilter = clamp(perceptual_roughness * (1.7 - 0.7 * perceptual_roughness), 0.0, 1.0);
    var specular_irradiance = lerp(sample_ambient_probe(reflection_direction),
                                  sample_ambient_irradiance(reflection_direction), prefilter);
    var environment_fresnel = fresnel_schlick_roughness(f0, f90, ndotv, perceptual_roughness);
    var diffuse_weight = (float3(1.0) - environment_fresnel) * (1.0 - metallic);
    var ambient_diffuse = diffuse_weight * surface_color.rgb * diffuse_irradiance * occlusion;
    var ambient_specular = specular_irradiance * (f0 * env_brdf.x + float3(env_brdf.y))
                           * energy_compensation
                           * specular_occlusion(ndotv, occlusion, perceptual_roughness)
                            * horizon_occlusion(reflection_direction, geometric_normal)
                           * specular_highlights;
    return float4(ambient_diffuse + ambient_specular + direct + emission,
                     surface_color.a);
}

struct SkyOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 clip_position : TEXCOORD0;
};

[shader("vertex")]
SkyOutput sky_vertex(uint index : SV_VertexID) {
    float2 positions[3] = {
        float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)};
    SkyOutput output;
    output.position = float4(positions[index], 0.0, 1.0);
    output.clip_position = positions[index];
    return output;
}

[shader("fragment")]
float4 sky_fragment(SkyOutput input) : SV_Target0 {
    var far_world = mul(camera.inverse_view_projection, float4(input.clip_position, 0.0, 1.0));
    var direction = normalize(far_world.xyz / far_world.w - camera.camera_position.xyz);
    if (camera.sky_horizon.w > 0.5) {
        var uv = float2(frac(atan2(direction.z, direction.x) / 6.28318530718 + 0.5
                                 + camera.sky_ground.w / 360.0),
                           acos(clamp(direction.y, -1.0, 1.0)) / 3.14159265359);
        var color = material_base_color.SampleLevel(material_base_color_sampler, uv, 0.0).rgb;
        return float4(color * camera.sky_top_exposure.rgb * camera.sky_top_exposure.w, 1.0);
    }
    var y = direction.y;
    var base = lerp(camera.sky_ground.rgb, camera.sky_top_exposure.rgb,
                   smoothstep(-0.10, 0.45, y));
    var horizon_mask = 0.0;
    if (y < 0.0) {
        var side = 1.0 - smoothstep(0.0, 0.10, -y);
        horizon_mask = side * side;
    } else {
        horizon_mask = 1.0 - smoothstep(0.0, 0.45, y);
    }
    var color = lerp(base, camera.sky_horizon.rgb, horizon_mask * 0.35);
    var horizon_glow = pow(1.0 - abs(y), 8.0) * 0.15;
    color += float3(horizon_glow);
    return float4(color * camera.sky_top_exposure.w, 1.0);
}

[shader("vertex")]
float4 shadow_vertex([[vk::location(0)]] float3 position : TEXCOORD0, uint draw_index : SV_InstanceID) : SV_Position {
    return mul(camera.light_view_projection, mul(draws[draw_index].model, float4(position, 1.0)));
}
)slang"};
    return module;
}

} // namespace infernux
