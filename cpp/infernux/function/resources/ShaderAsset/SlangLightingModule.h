#pragma once
#include "SlangCompiler.h"
namespace infernux
{
inline const ShaderSourceModule &GetSlangLightingModule()
{
    static const ShaderSourceModule module{
        "inx_lighting", "<infernux/shader/lighting>",
        R"slang(import inx_geometry;
static const float PI = 3.14159265358979323846;
static const float INV_PI = 0.31830988618379067154;
static const float FLT_MIN = 1.175494351e-38;
static const int MAX_DIRECTIONAL_LIGHTS = 4;
static const int MAX_POINT_LIGHTS = 64;
static const int MAX_SPOT_LIGHTS = 32;
static const int MAX_AREA_LIGHTS = 16;
struct InxDirectionalLight {
    float4 direction;      
    float4 color;          
    float4 shadow_params;   
    uint4 metadata;      
};
struct InxPointLight {
    float4 position;       
    float4 color;          
    float4 attenuation;    
    float4 shadow_params;
    uint4 metadata;
};
struct InxSpotLight {
    float4 position;       
    float4 direction;      
    float4 color;          
    float4 spot_params;     
    float4 attenuation;    
    float4 shadow_params;
    uint4 metadata;
};
struct InxAreaLight {
    float4 position_range;
    float4 direction;
    float4 right_width;
    float4 up_height;
    float4 color;
    float4 shadow_params;
    uint4 metadata;
};
struct InxShadowView {
    column_major float4x4 view_projection;
    float4 atlas_scale_offset;
    float4 depth_texel;
    float4 split_data;
    uint4 metadata;
};
struct InxLightingData {
    int4 light_counts;
    float4 ambient_color, ambient_sky_color, ambient_equator_color, ambient_ground_color, camera_pos;
    InxDirectionalLight directional_lights[4];
    InxPointLight point_lights[64];
    InxSpotLight spot_lights[32];
    InxAreaLight area_lights[16];
    uint4 shadow_view_header;
    InxShadowView shadow_views[64];
};
struct InxCanonicalLight {
    float4 position_range, direction_outer_cos, color_intensity, shadow_and_inner_cos, area_right_width, area_up_height;
    uint4 metadata, identity_and_shadow;
};
struct InxLight {
    float3  direction;    
    float3  color;        
    float attenuation;  
    float shadow;       
};
[[vk::binding(0, 1)]] DepthTexture2D shadow_image;
[[vk::binding(6, 1)]] SamplerState shadow_sampler;
[[vk::binding(4, 1)]] ConstantBuffer<InxLightingData> lighting;
[[vk::binding(1, 1)]] ByteAddressBuffer canonical_lights;
[[vk::binding(2, 1)]] StructuredBuffer<uint4> forward_plus_tile_headers;
[[vk::binding(3, 1)]] StructuredBuffer<uint> forward_plus_tile_masks;
uint4 light_counts() { return canonical_lights.Load<uint4>(0); }
InxCanonicalLight load_light(uint index) { return canonical_lights.Load<InxCanonicalLight>(16 + index * sizeof(InxCanonicalLight)); }
float3 sky_gradient(float y, float3 sky, float3 equator, float3 ground) {
    const float SKY_EDGE    = 0.45;   
    const float GROUND_EDGE = 0.10;   
    const float EQUATOR_STRENGTH = 0.35;  
    float t = smoothstep(-GROUND_EDGE, SKY_EDGE, y);
    float3 base = lerp(ground, sky, t);
    float horizon_mask;
    if (y < 0.0) {
        float s = 1.0 - smoothstep(0.0, GROUND_EDGE, -y);
        horizon_mask = s * s;
    } else {
        horizon_mask = 1.0 - smoothstep(0.0, SKY_EDGE, y);
    }
    return lerp(base, equator, horizon_mask * EQUATOR_STRENGTH);
}
float f_schlick(float f0, float f90, float u) {
    float x  = 1.0 - u;
    float x2 = x * x;
    float x5 = x * x2 * x2;
    return (f90 - f0) * x5 + f0;
}
float3 f_schlick(float3 f0, float f90, float u) {
    float x  = 1.0 - u;
    float x2 = x * x;
    float x5 = x * x2 * x2;
    return f0 * (1.0 - x5) + (f90 * x5);
}
float3 f_schlick_roughness(float3 f0, float f90, float u, float perceptual_roughness) {
    float x  = 1.0 - u;
    float x2 = x * x;
    float x5 = x * x2 * x2;
    return f0 + (max(float3(1.0 - perceptual_roughness), f0) - f0) * x5;
}
float compute_f90(float3 F0) {
    return saturate(50.0 * dot(F0, float3(0.33333)));
}
float D_GGX(float ndot_h, float roughness) {
    float a2 = roughness * roughness;
    float s  = (ndot_h * a2 - ndot_h) * ndot_h + 1.0;
    return INV_PI * a2 / max(s * s, FLT_MIN);
}
float v_smith_joint_ggx(float ndot_l, float ndot_v, float roughness) {
    float a2 = roughness * roughness;
    float lambda_v = ndot_l * sqrt((-ndot_v * a2 + ndot_v) * ndot_v + a2);
    float lambda_l = ndot_v * sqrt((-ndot_l * a2 + ndot_l) * ndot_l + a2);
    return 0.5 / max(lambda_v + lambda_l, FLT_MIN);
}
float dv_smith_joint_ggx(float ndot_h, float ndot_l, float ndot_v, float roughness) {
    float a2 = roughness * roughness;
    float s  = (ndot_h * a2 - ndot_h) * ndot_h + 1.0;
    float lambda_v = ndot_l * sqrt((-ndot_v * a2 + ndot_v) * ndot_v + a2);
    float lambda_l = ndot_v * sqrt((-ndot_l * a2 + ndot_l) * ndot_l + a2);
    return INV_PI * 0.5 * a2 / max(s * s * (lambda_v + lambda_l), FLT_MIN);
}
float disney_diffuse(float ndot_v, float ndot_l, float ldot_v, float perceptual_roughness) {
    float fd90 = 0.5 + (perceptual_roughness + perceptual_roughness * ldot_v);
    float light_scatter = f_schlick(1.0, fd90, ndot_l);
    float view_scatter  = f_schlick(1.0, fd90, ndot_v);
    return INV_PI * (1.0 / 1.03571) * light_scatter * view_scatter;
}
float2 env_brdf_approx(float perceptual_roughness, float ndot_v) {
    float4 c0 = float4(-1.0, -0.0275, -0.572,  0.022);
    float4 c1 = float4( 1.0,  0.0425,  1.04,  -0.04);
    float4 r  = perceptual_roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * ndot_v)) * r.x + r.y;
    return float2(-1.04, 1.04) * a004 + r.zw;
}
float compute_specular_occlusion(float ndot_v, float ao, float perceptual_roughness) {
    return saturate(pow(ndot_v + ao, exp2(-16.0 * perceptual_roughness - 1.0)) - 1.0 + ao);
}
float horizon_occlusion(float3 R, float3 geom_normal) {
    float horizon = saturate(1.0 + dot(R, geom_normal));
    return horizon * horizon;
}
float calculate_attenuation(float3 atten_params, float distance) {
    float range = max(atten_params.x, 1e-4);
    float d2 = distance * distance;
    float r2 = range * range;
    float ratio2 = d2 / r2;
    float factor = saturate(1.0 - ratio2 * ratio2);
    return (factor * factor) / (d2 + 1.0);
}
float geometric_specular_aa(float3 world_normal, float roughness) {
    float3 du = ddx(world_normal);
    float3 dv = ddy(world_normal);
    float variance = 0.25 * (dot(du, du) + dot(dv, dv));
    float kernel_roughness = min(variance, 0.18);
    return max(roughness, sqrt(roughness * roughness + kernel_roughness));
}
float calculate_spot_falloff(float3 light_dir, float3 spot_dir, float inner_angle_cos, float outer_angle_cos) {
    float spot_len2 = max(dot(spot_dir, spot_dir), 1e-8);
    float theta   = dot(light_dir, -spot_dir * rsqrt(spot_len2));
    float epsilon = max(inner_angle_cos - outer_angle_cos, 1e-4);
    return clamp((theta - outer_angle_cos) / epsilon, 0.0, 1.0);
}
float3 evaluate_pbr_light(float3 N, float3 V, float3 L, float3 light_radiance,
                      float3 albedo, float metallic,
                      float roughness, float perceptual_roughness,
                      float3 F0, float f90, float3 energy_compensation,
                      float specular_highlights) {
    float3  half_vec = V + L;
    float half_len2 = dot(half_vec, half_vec);
    float3  H     = (half_len2 > 1e-8) ? half_vec * rsqrt(half_len2) : N;
    float ndot_l = max(dot(N, L), 0.0);
    float ndot_v = max(dot(N, V), 0.0);
    float ndot_h = max(dot(N, H), 0.0);
    float ldot_h = max(dot(L, H), 0.0);
    float ldot_v = max(dot(L, V), 0.0);
    if (ndot_l <= 0.0) return float3(0.0);
    float3  F  = f_schlick(F0, f90, ldot_h);
    float DV = dv_smith_joint_ggx(ndot_h, ndot_l, ndot_v, roughness);
    float3  specular = F * DV;
    specular *= energy_compensation * clamp(specular_highlights, 0.0, 1.0);
    float diffuse_term = disney_diffuse(ndot_v, ndot_l, ldot_v, perceptual_roughness);
    float3  diffuse = albedo * (1.0 - metallic) * diffuse_term;
    return (diffuse + specular) * light_radiance * (ndot_l * PI);
}
struct InxLighting<let tiled_lighting : bool>
{
    uint layer_mask;
    bool receives_shadows;
    float2 pixel_position;
    uint4 inx_forward_plus_tile_header()
    {
        uint4 grid = forward_plus_tile_headers[0];
)slang"
        R"slang(        uint2 tile = min(uint2(pixel_position) / grid.z, grid.xy - 1u);
        return forward_plus_tile_headers[1u + tile.y * grid.x + tile.x];
    }

float3 sample_ambient_probe(float3 direction) {
    float mode = lighting.ambient_equator_color.a;
    if (mode < 0.5) {
        return lighting.ambient_color.rgb * lighting.ambient_color.a;
    }
    float3 sky     = lighting.ambient_sky_color.rgb * lighting.ambient_sky_color.a;
    float3 equator = lighting.ambient_equator_color.rgb;
    float3 ground  = lighting.ambient_ground_color.rgb * lighting.ambient_ground_color.a;
    return max(sky_gradient(direction.y, sky, equator, ground), float3(0.0));
}
float3 sample_ambient_probe_average() {
    float mode = lighting.ambient_equator_color.a;
    if (mode < 0.5) {
        return lighting.ambient_color.rgb * lighting.ambient_color.a;
    }
    float3 sky     = lighting.ambient_sky_color.rgb * lighting.ambient_sky_color.a;
    float3 equator = lighting.ambient_equator_color.rgb;
    float3 ground  = lighting.ambient_ground_color.rgb * lighting.ambient_ground_color.a;
    return max((sky + equator + ground) * 0.33333333, float3(0.0));
}
float3 sample_ambient_irradiance(float3 direction) {
    float mode = lighting.ambient_equator_color.a;
    if (mode < 0.5) {
        return lighting.ambient_color.rgb * lighting.ambient_color.a;
    }
    float3 sky     = lighting.ambient_sky_color.rgb * lighting.ambient_sky_color.a;
    float3 equator = lighting.ambient_equator_color.rgb;
    float3 ground  = lighting.ambient_ground_color.rgb * lighting.ambient_ground_color.a;
    float y = clamp(direction.y, -1.0, 1.0);
    float3 irradiance = (y >= 0.0) ? lerp(equator, sky, y) : lerp(equator, ground, -y);
    return max(irradiance, float3(0.0));
}
float3 get_specular_ambient_direction(float3 N, float3 V, float perceptual_roughness) {
    float3 R = reflect(-V, N);
    float roughness = perceptual_roughness * perceptual_roughness;
    float lerp_factor = (1.0 - roughness) * (sqrt(1.0 - roughness) + roughness);
    return normalize(lerp(N, R, saturate(lerp_factor)));
}
float shadow_bilinear_pcf(float2 atlas_uv, float receiver_depth, float2 tile_min, float2 tile_max,
                        float atlas_size) {
    float2 inverse_atlas = float2(1.0 / atlas_size);
    float2 pixel = atlas_uv * atlas_size - float2(0.5);
    float2 base = floor(pixel);
    float2 fraction = frac(pixel);
    float2 gather_uv = (base + float2(1.0)) * inverse_atlas;
    gather_uv = clamp(gather_uv, tile_min + inverse_atlas * 0.5, tile_max - inverse_atlas * 0.5);
    float4 depths = shadow_image.Gather(shadow_sampler, gather_uv);
    float4 comparison = step(depths, float4(receiver_depth));
    float lower = lerp(comparison.w, comparison.z, fraction.x);
    float upper = lerp(comparison.x, comparison.y, fraction.x);
    return lerp(lower, upper, fraction.y);
}
float shadow_tent_weight(int index) {
    return (index == 0 || index == 3) ? 1.0 : 3.0;
}
float sample_shadow_view_visibility(uint view_index, float3 world_pos, float3 normal, float3 to_light,
                                 float4 shadow_params, bool soft_filter) {
    if (view_index >= lighting.shadow_view_header.x) return 1.0;
    InxShadowView shadow_view = lighting.shadow_views[view_index];
    float3 N = normalize(normal);
    float3 L = normalize(to_light);
    float n_dot_l = clamp(dot(N, L), 0.0, 1.0);
    float slope = clamp(sqrt(max(1.0 - n_dot_l * n_dot_l, 0.0)) / max(n_dot_l, 0.2), 0.0, 4.0);
    float4 unbiassed_clip = mul(shadow_view.view_projection, float4(world_pos, 1.0));
    float perspective_scale = shadow_view.metadata.x == 0u
        ? 1.0
        : clamp(abs(unbiassed_clip.w) / max(shadow_view.depth_texel.y, 0.000001), 0.001, 1.0);
    float world_texel = max(shadow_view.depth_texel.z * perspective_scale, 0.000001);
    float filter_texels = soft_filter ? clamp(shadow_view.depth_texel.w, 0.5, 4.0) : 1.0;
    float3 biased_position = world_pos;
    biased_position += N * (shadow_params.z * world_texel * (0.5 * filter_texels + slope));
    biased_position += L * (shadow_params.y * world_texel);
    float4 clip = mul(shadow_view.view_projection, float4(biased_position, 1.0));
    if (abs(clip.w) < 0.000001) return 1.0;
    float3 ndc = clip.xyz / clip.w;
    float2 local_uv = ndc.xy * 0.5 + 0.5;
    if (any((local_uv < float2(0.0))) || any((local_uv > float2(1.0))) ||
        ndc.z <= 0.0 || ndc.z >= 1.0) return 1.0;
    float2 atlas_scale = shadow_view.atlas_scale_offset.xy;
    float2 atlas_offset = shadow_view.atlas_scale_offset.zw;
    float2 atlas_uv = atlas_offset + local_uv * atlas_scale;
    float atlas_size = max(float(lighting.shadow_view_header.y), 1.0);
    float2 texel = float2(1.0 / atlas_size);
    float2 tile_min = atlas_offset + texel * 0.5;
    float2 tile_max = atlas_offset + atlas_scale - texel * 0.5;
    if (!soft_filter) {
        float visibility = 0.0;
        for (int y = 0; y < 2; ++y) {
            for (int x = 0; x < 2; ++x) {
                float2 offset = (float2(float(x), float(y)) - float2(0.5)) * texel;
                visibility += shadow_bilinear_pcf(atlas_uv + offset, ndc.z, tile_min, tile_max, atlas_size);
            }
        }
        return visibility * 0.25;
    }
    float step_size = filter_texels / 1.5;
    float visibility = 0.0;
    float total_weight = 0.0;
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            float weight = shadow_tent_weight(x) * shadow_tent_weight(y);
            float2 offset = (float2(float(x), float(y)) - float2(1.5)) * step_size * texel;
            visibility += shadow_bilinear_pcf(atlas_uv + offset, ndc.z, tile_min, tile_max, atlas_size) * weight;
            total_weight += weight;
        }
    }
    return visibility / max(total_weight, 1.0);
}
float sample_shadow_view(uint view_index, float3 world_pos, float3 normal, float3 to_light, float4 shadow_params) {
    if (view_index >= lighting.shadow_view_header.x || shadow_params.w < 0.5 || shadow_params.x <= 0.0) return 1.0;
    float visibility = sample_shadow_view_visibility(
        view_index, world_pos, normal, to_light, shadow_params, shadow_params.w > 1.5);
    return lerp(1.0, visibility, shadow_params.x);
}
float sample_directional_views(uint first_view, uint view_count, float4 shadow_params,
                             float3 world_pos, float3 normal, float3 to_light, float view_depth) {
    if (!receives_shadows) return 1.0;
    uint available_views = lighting.shadow_view_header.x;
    if (view_count == 0u || first_view >= available_views) return 1.0;
    view_count = min(view_count, available_views - first_view);
    uint selected = view_count - 1u;
    for (uint index = 0u; index < view_count; ++index) {
        if (view_depth < lighting.shadow_views[first_view + index].split_data.y) {
            selected = index;
            break;
        }
    }
    float shadow = sample_shadow_view(first_view + selected, world_pos, normal, to_light, shadow_params);
    if (selected + 1u < view_count) {
        InxShadowView current = lighting.shadow_views[first_view + selected];
        float overlap = max((current.split_data.y - current.split_data.x) * 0.1, 0.0001);
        float blend = smoothstep(0.0, overlap, current.split_data.y - view_depth);
        float next_shadow = sample_shadow_view(first_view + selected + 1u, world_pos, normal, to_light, shadow_params);
        shadow = lerp(next_shadow, shadow, blend);
    } else {
        InxShadowView last = lighting.shadow_views[first_view + selected];
        float fade_width = max((last.split_data.y - last.split_data.x) * 0.1, 0.0001);
        float fade = smoothstep(last.split_data.y - fade_width, last.split_data.y, view_depth);
        shadow = lerp(shadow, 1.0, fade);
    }
    return shadow;
}
uint point_shadow_face(float3 direction_from_light) {
    float3 a = abs(direction_from_light);
    if (a.x >= a.y && a.x >= a.z) return direction_from_light.x >= 0.0 ? 0u : 1u;
)slang"
        R"slang(    if (a.y >= a.z) return direction_from_light.y >= 0.0 ? 2u : 3u;
    return direction_from_light.z >= 0.0 ? 4u : 5u;
}
float sample_local_shadow(uint first_view, uint view_count, uint light_type, float4 shadow_params,
                        float3 light_position, float3 world_pos, float3 normal, float3 to_light) {
    if (!receives_shadows) return 1.0;
    uint available_views = lighting.shadow_view_header.x;
    if (view_count == 0u || first_view >= available_views) return 1.0;
    view_count = min(view_count, available_views - first_view);
    if (light_type == 2u || shadow_params.w < 1.5) {
        uint offset = light_type == 2u ? 0u : min(point_shadow_face(world_pos - light_position), view_count - 1u);
        return sample_shadow_view(first_view + offset, world_pos, normal, to_light, shadow_params);
    }
    float3 radial = world_pos - light_position;
    float radial_length = max(length(radial), 0.0001);
    float3 radial_direction = radial / radial_length;
    float3 helper = abs(radial_direction.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(helper, radial_direction));
    float3 bitangent = cross(radial_direction, tangent);
    uint center_face = min(point_shadow_face(radial), view_count - 1u);
    InxShadowView center_view = lighting.shadow_views[first_view + center_face];
    float inner_resolution = max(center_view.atlas_scale_offset.x * float(lighting.shadow_view_header.y), 1.0);
    float radius_world = shadow_params.w > 1.5
        ? center_view.depth_texel.w * (2.0 * radial_length / inner_resolution)
        : 0.0;
    float2 disk[8] = {
        float2(0.0, 0.0), float2(0.7071, 0.0), float2(-0.7071, 0.0),
        float2(0.0, 0.7071), float2(0.0, -0.7071), float2(0.5, 0.5),
        float2(-0.5, 0.5), float2(0.5, -0.5)};
    const int disk_sample_count = 8;
    float visibility = 0.0;
    for (int sample_index = 0; sample_index < disk_sample_count; ++sample_index) {
        float3 sample_position = world_pos +
            (tangent * disk[sample_index].x + bitangent * disk[sample_index].y) * radius_world;
        uint face = min(point_shadow_face(sample_position - light_position), view_count - 1u);
        visibility += sample_shadow_view_visibility(first_view + face, sample_position, normal, to_light,
                                                  shadow_params, false);
    }
    return lerp(1.0, visibility / float(disk_sample_count), shadow_params.x);
}
float calculate_shadow(float3 world_pos, float3 normal, float frag_view_depth_val) {
    if (lighting.shadow_view_header.x == 0u) return 1.0;
if (tiled_lighting) {
    if (light_counts().x == 0u) return 1.0;
    InxCanonicalLight light = load_light(0);
    return sample_directional_views(light.identity_and_shadow.z, light.identity_and_shadow.w,
                                  float4(light.shadow_and_inner_cos.xyz, float(light.metadata.z)), world_pos, normal,
                                  normalize(-light.direction_outer_cos.xyz), frag_view_depth_val);
} else {
    if (lighting.light_counts.x <= 0) return 1.0;
    InxDirectionalLight light = lighting.directional_lights[0];
    return sample_directional_views(light.metadata.z, light.metadata.w, light.shadow_params, world_pos, normal,
                                  normalize(-light.direction.xyz), frag_view_depth_val);
}
}
InxLight get_main_light(float3 world_pos, float3 normal, float frag_view_depth_val) {
    InxLight l;
if (tiled_lighting) {
    if (light_counts().x > 0u) {
        InxCanonicalLight dl = load_light(0);
        l.direction = normalize(-dl.direction_outer_cos.xyz);
        l.color = dl.color_intensity.rgb * dl.color_intensity.w;
        l.attenuation = 1.0;
        l.shadow = calculate_shadow(world_pos, normal, frag_view_depth_val);
    } else {
        l.direction = float3(0.0, 1.0, 0.0);
        l.color = float3(0.0);
        l.attenuation = 0.0;
        l.shadow = 1.0;
    }
} else {
    if (lighting.light_counts.x > 0) {
        InxDirectionalLight dl = lighting.directional_lights[0];
        l.direction = normalize(-dl.direction.xyz);
        l.color = dl.color.rgb * dl.color.w;
        l.attenuation = 1.0;
        l.shadow = calculate_shadow(world_pos, normal, frag_view_depth_val);
    } else {
        l.direction = float3(0.0, 1.0, 0.0);
        l.color = float3(0.0);
        l.attenuation = 0.0;
        l.shadow = 1.0;
    }
}
    return l;
}
float3 evaluate_rectangle_area_light(float3 light_position, float3 ray_direction, float4 right_width, float4 up_height,
                                float3 light_color, float intensity, float range, bool two_sided,
                                uint first_shadow_view, uint shadow_view_count, float4 shadow_params,
                                float3 world_pos, float3 N, float3 V, float3 albedo, float metallic,
                                float roughness, float perceptual_roughness, float3 F0, float f90,
                                float3 energy_compensation, float specular_highlights) {
    float3 from_light = world_pos - light_position;
    float emitter_facing = dot(normalize(ray_direction), normalize(from_light));
    emitter_facing = two_sided ? abs(emitter_facing) : max(emitter_facing, 0.0);
    if (emitter_facing <= 0.0001) return float3(0.0);
    float3 right = normalize(right_width.xyz) * right_width.w * 0.5;
    float3 up = normalize(up_height.xyz) * up_height.w * 0.5;
    float3 samples[4] = {light_position - right - up, light_position + right - up,
                              light_position + right + up, light_position - right + up};
    float3 result = float3(0.0);
    float3 center_to_light = normalize(light_position - world_pos);
    float shadow = sample_local_shadow(first_shadow_view, shadow_view_count, 3u, shadow_params,
                                     light_position, world_pos, N, center_to_light);
    for (int sample_index = 0; sample_index < 4; ++sample_index) {
        float3 light_vector = samples[sample_index] - world_pos;
        float distance_to_light = length(light_vector);
        if (distance_to_light <= 0.00001 || distance_to_light >= range) continue;
        float3 L = light_vector / distance_to_light;
        float attenuation = calculate_attenuation(float3(range, 0.0, 0.0), distance_to_light);
        float3 radiance = light_color * intensity * attenuation * emitter_facing * shadow * 0.25;
        result += evaluate_pbr_light(N, V, L, radiance, albedo, metallic, roughness,
                                   perceptual_roughness, F0, f90, energy_compensation,
                                   specular_highlights);
    }
    return result;
}
float3 calculate_all_lighting(float3 world_pos, float3 N, float3 V,
                          float3 albedo, float metallic,
                          float roughness, float perceptual_roughness,
                          float3 F0, float f90, float3 energy_compensation,
                          float specular_highlights, float view_depth, float shadow) {
    float3 lo = float3(0.0);
    const uint current_light_domain = 1u;
if (tiled_lighting) {
    uint directional_count = light_counts().x;
    for (uint i = 0u; i < directional_count; ++i) {
        InxCanonicalLight light = load_light(i);
        if ((light.metadata.w & current_light_domain) == 0u) continue;
        if ((light.metadata.y & layer_mask) == 0u) continue;
        float3 L = normalize(-light.direction_outer_cos.xyz);
        float3 radiance = light.color_intensity.rgb * light.color_intensity.w;
        float light_shadow =
            i == 0u ? shadow
                    : sample_directional_views(
                          light.identity_and_shadow.z, light.identity_and_shadow.w,
                          float4(light.shadow_and_inner_cos.xyz, float(light.metadata.z)), world_pos, N, L, view_depth);
        lo += evaluate_pbr_light(N, V, L, radiance, albedo, metallic,
                               roughness, perceptual_roughness,
)slang"
        R"slang(                               F0, f90, energy_compensation,
                               specular_highlights) * light_shadow;
    }
    uint4 tile_header = inx_forward_plus_tile_header();
    for (uint word_entry = 0u; word_entry < tile_header.y; ++word_entry) {
        uint light_mask = forward_plus_tile_masks[tile_header.x + word_entry];
        while (light_mask != 0u) {
            uint bit = uint(firstbitlow(light_mask));
            uint local_index = word_entry * 32u + bit;
            light_mask &= light_mask - 1u;
            if (local_index >= tile_header.z) continue;
            InxCanonicalLight light = load_light(directional_count + local_index);
            if ((light.metadata.w & current_light_domain) == 0u) continue;
            if ((light.metadata.y & layer_mask) == 0u) continue;
            if (light.metadata.x == 3u) {
                lo += evaluate_rectangle_area_light(
                    light.position_range.xyz, light.direction_outer_cos.xyz, light.area_right_width,
                    light.area_up_height, light.color_intensity.rgb, light.color_intensity.w,
                    light.position_range.w, light.direction_outer_cos.w > 0.5, light.identity_and_shadow.z,
                    light.identity_and_shadow.w, float4(light.shadow_and_inner_cos.xyz, float(light.metadata.z)),
                    world_pos, N, V, albedo, metallic, roughness, perceptual_roughness, F0, f90,
                    energy_compensation, specular_highlights);
                continue;
            }
            float3 light_vec = light.position_range.xyz - world_pos;
            float distance_to_light = length(light_vec);
            float range = max(light.position_range.w, 0.0001);
            if (distance_to_light <= 0.00001 || distance_to_light >= range) continue;
            float3 L = light_vec / distance_to_light;
            float attenuation = calculate_attenuation(float3(range, 0.0, 0.0), distance_to_light);
            if (light.metadata.x == 2u) {
                float cone = dot(L, -normalize(light.direction_outer_cos.xyz));
                attenuation *= smoothstep(light.direction_outer_cos.w, light.shadow_and_inner_cos.w, cone);
            }
            if (attenuation <= 0.0001) continue;
            float local_shadow = sample_local_shadow(
                light.identity_and_shadow.z, light.identity_and_shadow.w, light.metadata.x,
                float4(light.shadow_and_inner_cos.xyz, float(light.metadata.z)), light.position_range.xyz,
                world_pos, N, L);
            float3 radiance = light.color_intensity.rgb * light.color_intensity.w * attenuation * local_shadow;
            lo += evaluate_pbr_light(N, V, L, radiance, albedo, metallic,
                                   roughness, perceptual_roughness,
                                   F0, f90, energy_compensation,
                                   specular_highlights);
        }
    }
} else {
    for (int i = 0; i < lighting.light_counts.x && i < MAX_DIRECTIONAL_LIGHTS; ++i) {
        InxDirectionalLight light = lighting.directional_lights[i];
        if ((light.metadata.y & current_light_domain) == 0u) continue;
        if ((light.metadata.x & layer_mask) == 0u) continue;
        float3 L        = normalize(-light.direction.xyz);
        float3 radiance = light.color.rgb * light.color.w;
        float light_shadow =
            i == 0 ? shadow
                   : sample_directional_views(light.metadata.z, light.metadata.w, light.shadow_params, world_pos, N, L,
                                            view_depth);
        lo += evaluate_pbr_light(N, V, L, radiance, albedo, metallic,
                               roughness, perceptual_roughness,
                               F0, f90, energy_compensation,
                               specular_highlights) * light_shadow;
    }
    for (int i = 0; i < lighting.light_counts.y && i < MAX_POINT_LIGHTS; ++i) {
        InxPointLight light = lighting.point_lights[i];
        if ((light.metadata.y & current_light_domain) == 0u) continue;
        if ((light.metadata.x & layer_mask) == 0u) continue;
        float3  light_vec = light.position.xyz - world_pos;
        float distance = length(light_vec);
        if (distance > 1e-5) {
            float3  L = light_vec / distance;
            float attenuation = calculate_attenuation(light.attenuation.xyz, distance);
            if (attenuation > 0.001) {
                float local_shadow = sample_local_shadow(light.metadata.z, light.metadata.w, 1u, light.shadow_params,
                                                      light.position.xyz, world_pos, N, L);
                float3 radiance = light.color.rgb * light.color.w * attenuation * local_shadow;
                lo += evaluate_pbr_light(N, V, L, radiance, albedo, metallic,
                                       roughness, perceptual_roughness,
                                       F0, f90, energy_compensation,
                                       specular_highlights);
            }
        }
    }
    for (int i = 0; i < lighting.light_counts.z && i < MAX_SPOT_LIGHTS; ++i) {
        InxSpotLight light = lighting.spot_lights[i];
        if ((light.metadata.y & current_light_domain) == 0u) continue;
        if ((light.metadata.x & layer_mask) == 0u) continue;
        float3  light_vec = light.position.xyz - world_pos;
        float distance = length(light_vec);
        if (distance > 1e-5) {
            float3 L = light_vec / distance;
            float spot_falloff = calculate_spot_falloff(L, light.direction.xyz,
                                                      light.spot_params.x, light.spot_params.y);
            if (spot_falloff > 0.0) {
                float attenuation = calculate_attenuation(light.attenuation.xyz, distance);
                if (attenuation > 0.001) {
                    float local_shadow = sample_local_shadow(light.metadata.z, light.metadata.w, 2u, light.shadow_params,
                                                          light.position.xyz, world_pos, N, L);
                    float3 radiance = light.color.rgb * light.color.w * attenuation * spot_falloff * local_shadow;
                    lo += evaluate_pbr_light(N, V, L, radiance, albedo, metallic,
                                           roughness, perceptual_roughness,
                                           F0, f90, energy_compensation,
                                           specular_highlights);
                }
            }
        }
    }
    for (int i = 0; i < lighting.light_counts.w && i < MAX_AREA_LIGHTS; ++i) {
        InxAreaLight light = lighting.area_lights[i];
        if ((light.metadata.y & current_light_domain) == 0u) continue;
        if ((light.metadata.x & layer_mask) == 0u) continue;
        lo += evaluate_rectangle_area_light(
            light.position_range.xyz, light.direction.xyz, light.right_width, light.up_height,
            light.color.rgb, light.color.w, light.position_range.w, light.direction.w > 0.5,
            light.metadata.z, light.metadata.w, light.shadow_params, world_pos, N, V, albedo,
            metallic, roughness, perceptual_roughness, F0, f90, energy_compensation,
            specular_highlights);
    }
}
    return lo;
}
float4 shade(InxSurfaceData s, InxMeshInput ctx) {
    float3 albedo = clamp(s.albedo, 0.0, 1.0);
    float metallic = clamp(s.metallic, 0.0, 1.0);
    float perceptual_roughness = clamp(1.0 - s.smoothness, 0.045, 1.0);
    float roughness = perceptual_roughness * perceptual_roughness;
    float ao = clamp(s.occlusion, 0.0, 1.0);
    float3 N = normalize(s.normal_ws);
    roughness = geometric_specular_aa(N, roughness);
    perceptual_roughness = max(perceptual_roughness, sqrt(roughness));
    float3 V = normalize(lighting.camera_pos.xyz - ctx.position_ws);
    float3  F0    = lerp(float3(0.04), albedo, metallic);
    float f90   = compute_f90(F0);
    float ndot_v = max(dot(N, V), 0.0);
)slang"
        R"slang(    float2  env_brdf      = env_brdf_approx(perceptual_roughness, ndot_v);
    float reflectivity = env_brdf.x + env_brdf.y;
    float3  energy_compensation = 1.0 + F0 * (1.0 / max(reflectivity, 0.001) - 1.0);
    InxLight main_light = get_main_light(ctx.position_ws, N, ctx.view_depth);
    float3 lo = calculate_all_lighting(ctx.position_ws, N, V,
                                   albedo, metallic,
                                   roughness, perceptual_roughness,
                                   F0, f90, energy_compensation,
                                   s.specular_highlights,
                                   ctx.view_depth, main_light.shadow);
    float3 diffuse_irradiance = sample_ambient_irradiance(N);
    float3 refl_dir = get_specular_ambient_direction(N, V, perceptual_roughness);
    float prefilter = saturate(perceptual_roughness * (1.7 - 0.7 * perceptual_roughness));
    float3 specular_irradiance = lerp(sample_ambient_probe(refl_dir),
                                  sample_ambient_irradiance(refl_dir),
                                  prefilter);
    float3 k_s_env = f_schlick_roughness(F0, f90, ndot_v, perceptual_roughness);
    float3 k_d_env = (1.0 - k_s_env) * (1.0 - metallic);
    float3 diffuse_env = k_d_env * albedo * diffuse_irradiance * ao;
    float spec_occlusion = compute_specular_occlusion(ndot_v, ao, perceptual_roughness);
    float3 geom_n = normalize(ctx.normal_ws);
    geom_n = (dot(geom_n, N) < 0.0) ? -geom_n : geom_n;
    float horizon_fade = horizon_occlusion(refl_dir, geom_n);
    float3 spec_env = specular_irradiance
                 * (F0 * env_brdf.x + env_brdf.y)
                 * energy_compensation
                 * (spec_occlusion * horizon_fade)
                 * s.specular_highlights;
    float3 ambient = diffuse_env + spec_env;
    return float4(ambient + lo + s.emission, s.alpha);
}
};
public float4 inx_shade_pbr<let tiled_lighting : bool>(InxSurfaceData surface, InxMeshInput input)
{
    InxLighting<tiled_lighting> context;
    context.layer_mask = input.layer_mask;
    context.receives_shadows = (input.object_flags & 2u) == 0u;
    context.pixel_position = input.position.xy;
    return context.shade(surface, input);
}
)slang"};
    return module;
}
} // namespace infernux
