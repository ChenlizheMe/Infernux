#pragma once
#include "SlangCompiler.h"
namespace infernux
{
inline const ShaderSourceModule &GetSlangLightingModule()
{
    static const ShaderSourceModule module{"inx_lighting", "<infernux/shader/lighting>",
                                           R"slang(import inx_geometry;
import inx_lit_scattering;
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
    float4 environment_panorama;
};
struct InxCanonicalLight {
    float4 position_range, direction_outer_cos, color_intensity, shadow_and_inner_cos, area_right_width, area_up_height;
    uint4 metadata, identity_and_shadow;
};
[[vk::binding(0, 1)]] DepthTexture2D shadow_image;
[__AttributeUsage(_AttributeTargets.Var)]
struct InxNonFilteringSamplerAttribute {};
[__AttributeUsage(_AttributeTargets.Var)]
struct InxUnfilterableTextureAttribute {};
[InxUnfilterableTexture]
[[vk::binding(9, 1)]] Texture2D<float4> environment_panorama_image;
[InxNonFilteringSampler]
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
float dv_smith_joint_ggx(float ndot_h, float ndot_l, float ndot_v, float roughness) {
    float a2 = roughness * roughness;
    float s = inx_ggx_ndf_denominator(ndot_h, a2);
    float lambda_v = ndot_l * inx_ggx_projected_roughness(ndot_v, a2);
    float lambda_l = ndot_v * inx_ggx_projected_roughness(ndot_l, a2);
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
    float ndot_l = saturate(dot(N, L));
    float ndot_v = saturate(dot(N, V));
    float ndot_h = saturate(dot(N, H));
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
public interface IInxLightResponse
{
    float3 evaluate(InxLight light);
};
InxLight inx_light_sample(float3 direction, float3 color, float attenuation, float shadow)
{
    InxLight light;
    light.direction = direction;
    light.color = color;
    light.attenuation = attenuation;
    light.shadow = shadow;
    return light;
}
public struct InxPbrLightResponse : IInxLightResponse
{
    float3 normal;
    float3 view;
    float3 albedo;
    float metallic;
    float roughness;
    float perceptual_roughness;
    float3 f0;
    float f90;
    float3 energy_compensation;
    float specular_highlights;
    public float3 evaluate(InxLight light)
    {
        return evaluate_pbr_light(normal, view, light.direction,
                                  light.color * light.attenuation * light.shadow,
                                  albedo, metallic, roughness, perceptual_roughness,
                                  f0, f90, energy_compensation, specular_highlights);
    }
};
float3 inx_pbr_energy_compensation(float3 f0, float2 environment_brdf)
{
    return 1.0 + f0 * (1.0 / max(environment_brdf.x + environment_brdf.y, 0.001) - 1.0);
}
InxPbrLightResponse inx_pbr_light_response(InxLitScattering scattering, float3 N, float3 V,
                                          float3 energy_compensation, float specular_highlights)
{
    InxPbrLightResponse response;
    response.normal = N;
    response.view = V;
    response.albedo = scattering.albedo;
    response.metallic = scattering.metallic;
    response.roughness = scattering.ggx_alpha;
    response.perceptual_roughness = scattering.perceptual_roughness;
    response.f0 = scattering.f0;
    response.f90 = scattering.f90;
    response.energy_compensation = energy_compensation;
    response.specular_highlights = specular_highlights;
    return response;
}
public InxPbrLightResponse inx_secondary_light_response(InxSurfaceData surface, float3 N, float3 outgoing)
{
    // The ray has a geometric/UV footprint rather than raster normal derivatives.
    let scattering = inx_prepare_lit_scattering(surface.albedo, surface.metallic, surface.smoothness, 0.0);
    float2 brdf = env_brdf_approx(scattering.perceptual_roughness, max(dot(N, outgoing), 0.0));
    return inx_pbr_light_response(scattering, N, outgoing, inx_pbr_energy_compensation(scattering.f0, brdf),
                                  surface.specular_highlights);
}
// Path transport uses f * cos, independently of the analytic lights' pi-scaled convention.
public float3 inx_secondary_projected_bsdf(InxSurfaceData surface, float3 N, float3 outgoing, float3 incoming)
{
    InxPbrLightResponse response = inx_secondary_light_response(surface, N, outgoing);
    return response.evaluate(inx_light_sample(incoming, float3(1.0), 1.0, 1.0)) * INV_PI;
}
float3 rectangle_extent(float4 axis_width) { return normalize(axis_width.xyz) * axis_width.w * 0.5; }
float rectangle_facing(float3 direction, float3 from_light, bool two_sided)
{
    float facing = dot(normalize(direction), normalize(from_light));
    return two_sided ? abs(facing) : max(facing, 0.0);
}
public struct InxTransportLight
{
    public InxLight sample;
    public float3 endpoint;
    public uint layer_mask;
    public float shadow_strength;
    public bool directional;
};
public uint2 inx_transport_light_counts() { return light_counts().xy; }
public InxTransportLight inx_transport_light(uint index, float3 position, uint rectangle_corner)
{
    InxCanonicalLight light = load_light(index);
    InxTransportLight result = (InxTransportLight)0;
    result.layer_mask = (light.metadata.w & 1u) != 0u ? light.metadata.y : 0u;
    result.shadow_strength = light.metadata.z != 0u ? saturate(light.shadow_and_inner_cos.x) : 0.0;
    result.directional = light.metadata.x == 0u;
    float3 radiance = light.color_intensity.rgb * light.color_intensity.w;
    if (result.directional)
    {
        result.sample = inx_light_sample(normalize(-light.direction_outer_cos.xyz), radiance, 1.0, 1.0);
        return result;
    }

    result.endpoint = light.position_range.xyz;
    float emitter_facing = 1.0;
    if (light.metadata.x == 3u)
    {
        emitter_facing = rectangle_facing(light.direction_outer_cos.xyz, position - result.endpoint,
                                          light.direction_outer_cos.w > 0.5);
        // One uniform sample of the same four rectangle locations used by raster lighting.
        float side = (rectangle_corner & 1u) != 0u ? 1.0 : -1.0;
        float height = (rectangle_corner & 2u) != 0u ? 1.0 : -1.0;
        result.endpoint += rectangle_extent(light.area_right_width) * side +
                           rectangle_extent(light.area_up_height) * height;
    }
    float3 to_light = result.endpoint - position;
    float distance = length(to_light);
    if (distance <= 0.00001 || distance >= light.position_range.w || emitter_facing <= 0.0001)
        return result;
    float3 direction = to_light / distance;
    float attenuation = calculate_attenuation(float3(light.position_range.w, 0.0, 0.0), distance) * emitter_facing;
    if (light.metadata.x == 2u)
    {
        float cone = dot(direction, -normalize(light.direction_outer_cos.xyz));
        attenuation *= smoothstep(light.direction_outer_cos.w, light.shadow_and_inner_cos.w, cone);
    }
    result.sample = inx_light_sample(direction, radiance, attenuation, 1.0);
    return result;
}
// The immutable World publication appends four-stratum metadata and normalized CDFs
// after the canonical records. Actual stored differences define every sampled PDF.
public struct InxTransportLightSelection
{
    public uint index;
    public float inverse_pdf;
};
float transport_light_cdf(uint offset, uint index)
{
    return canonical_lights.Load<float>(offset + index * 4u);
}
public InxTransportLightSelection inx_sample_transport_light(uint count, uint strata, uint stratum, float u,
    float3 position, float3 geometric_normal, float3 shading_normal, uint layer_mask)
{
    uint begin = stratum * count / strata;
    uint end = (stratum + 1u) * count / strata;
    uint width = end - begin;
    if (width == 1u)
        return {begin, 1.0};
    uint proposal = 16u + count * 128u;
    uint4 upper = canonical_lights.Load<uint4>(proposal + stratum * 32u + 16u);
    float alpha = 0.0;
    if ((upper.w & 2u) == 0u)
    {
        uint4 lower = canonical_lights.Load<uint4>(proposal + stratum * 32u);
        InxTransportLight dominant = inx_transport_light(lower.w, position, 0u);
        if ((dominant.layer_mask & layer_mask) != 0u &&
            dot(geometric_normal, dominant.sample.direction) > 0.0 &&
            dot(shading_normal, dominant.sample.direction) > 0.0)
        {
            float3 distance = max(max(asfloat(lower.xyz) - position, position - asfloat(upper.xyz)), float3(0.0));
            float attenuation_bound = (upper.w & 1u) != 0u ? 1.0 : 1.0 / (1.0 + dot(distance, distance));
            alpha = 0.5 * saturate(dominant.sample.attenuation / attenuation_bound);
        }
    }
    if (alpha == 0.0)
        return {begin + min(uint(u * float(width)), width - 1u), float(width)};
    uint index;
    uint cdf_offset = proposal + strata * 32u;
    if (u < alpha)
    {
        float v = u / alpha;
        uint low = begin, high = end - 1u;
        while (low < high)
        {
            uint middle = low + (high - low) / 2u;
            if (transport_light_cdf(cdf_offset, middle) > v) high = middle;
            else low = middle + 1u;
        }
        index = low;
    }
    else
        index = begin + min(uint((u - alpha) / (1.0 - alpha) * float(width)), width - 1u);
    float cumulative = transport_light_cdf(cdf_offset, index);
    float previous = index == begin ? 0.0 : transport_light_cdf(cdf_offset, index - 1u);
    float density = (1.0 - alpha) / float(width) + alpha * (cumulative - previous);
    return {index, 1.0 / density};
}

)slang"
                                           R"slang(struct InxLighting<let tiled_lighting : bool>
{
    uint layer_mask;
    uint light_domain;
    bool receives_shadows;
    float2 pixel_position;
    uint4 inx_forward_plus_tile_header()
    {
        uint4 grid = forward_plus_tile_headers[0];
        uint2 tile = min(uint2(pixel_position) / grid.z, grid.xy - 1u);
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
// NDC depth is affine over a receiver plane, including perspective shadow views.
// Express its gradient per atlas texel so every gathered depth uses its own receiver.
float2 shadow_receiver_gradient(float4x4 projection, float3 ndc, float3 normal, float2 ndc_per_texel) {
    float3 dx = projection[0].xyz - ndc.x * projection[3].xyz;
    float3 dy = projection[1].xyz - ndc.y * projection[3].xyz;
    float3 dz = projection[2].xyz - ndc.z * projection[3].xyz;
    float3 tangent = cross(dy, normal);
    float determinant = dot(dx, tangent);
    if (abs(determinant) <= 0.0001 * length(cross(dx, dy))) return float2(0.0);
    return ndc_per_texel * float2(dot(dz, tangent), dot(dz, cross(normal, dx))) / determinant;
}
float shadow_bilinear_pcf(float2 atlas_uv, float2 receiver_texel, float receiver_depth, float2 gradient,
                        float2 tile_min, float2 tile_max, float atlas_size) {
    float2 inverse_atlas = float2(1.0 / atlas_size);
    float2 pixel = atlas_uv * atlas_size - float2(0.5);
    float2 base = floor(pixel);
    float2 fraction = frac(pixel);
    float2 gather_uv = (base + float2(1.0)) * inverse_atlas;
    gather_uv = clamp(gather_uv, tile_min + inverse_atlas * 0.5, tile_max - inverse_atlas * 0.5);
    float4 depths = shadow_image.Gather(shadow_sampler, gather_uv);
    float center = receiver_depth + dot(gather_uv * atlas_size - receiver_texel, gradient);
    float4 depth_offsets = 0.5 * float4(-gradient.x + gradient.y, gradient.x + gradient.y,
                                        gradient.x - gradient.y, -gradient.x - gradient.y);
    float4 comparison = step(depths, center + depth_offsets);
    float lower = lerp(comparison.w, comparison.z, fraction.x);
    float upper = lerp(comparison.x, comparison.y, fraction.x);
    return lerp(lower, upper, fraction.y);
}
float shadow_tent_weight(int index) {
    return (index == 0 || index == 3) ? 1.0 : 3.0;
}
float try_sample_shadow_view_visibility(uint view_index, float3 world_pos, float3 normal,
                                      float4 shadow_params, bool soft_filter) {
)slang"
                                           R"slang(    if (view_index >= lighting.shadow_view_header.x) return -1.0;
    InxShadowView shadow_view = lighting.shadow_views[view_index];
    float4 clip = mul(shadow_view.view_projection, float4(world_pos, 1.0));
    if (abs(clip.w) < 0.000001) return -1.0;
    float3 ndc = clip.xyz / clip.w;
    float2 local_uv = ndc.xy * 0.5 + 0.5;
    if (any((local_uv < float2(0.0))) || any((local_uv > float2(1.0))) ||
        ndc.z <= 0.0 || ndc.z >= 1.0) return -1.0;
    float2 atlas_scale = shadow_view.atlas_scale_offset.xy;
    float2 atlas_offset = shadow_view.atlas_scale_offset.zw;
    float2 atlas_uv = atlas_offset + local_uv * atlas_scale;
    float atlas_size = max(float(lighting.shadow_view_header.y), 1.0);
    float2 texel = float2(1.0 / atlas_size);
    float2 tile_min = atlas_offset + texel * 0.5;
    float2 tile_max = atlas_offset + atlas_scale - texel * 0.5;
    float2 receiver_texel = atlas_uv * atlas_size;
    float2 gradient = shadow_params.z * shadow_receiver_gradient(
        shadow_view.view_projection, ndc, normalize(normal), 2.0 * texel / atlas_scale);
    float receiver_depth = ndc.z + shadow_params.y * mad(ndc.z, shadow_view.split_data.z, shadow_view.split_data.w);
    if (!soft_filter) {
        return shadow_bilinear_pcf(atlas_uv, receiver_texel, receiver_depth, gradient, tile_min, tile_max, atlas_size);
    }
    float step_size = clamp(shadow_view.depth_texel.w, 0.5, 4.0) / 1.5;
    float visibility = 0.0;
    float total_weight = 0.0;
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            float weight = shadow_tent_weight(x) * shadow_tent_weight(y);
            float2 offset = (float2(float(x), float(y)) - float2(1.5)) * step_size * texel;
            visibility += shadow_bilinear_pcf(atlas_uv + offset, receiver_texel, receiver_depth, gradient,
                                              tile_min, tile_max, atlas_size) * weight;
            total_weight += weight;
        }
    }
    return visibility / max(total_weight, 1.0);
}
float sample_shadow_view_visibility(uint view_index, float3 world_pos, float3 normal,
                                    float4 shadow_params, bool soft_filter) {
    float visibility = try_sample_shadow_view_visibility(view_index, world_pos, normal, shadow_params, soft_filter);
    return visibility < 0.0 ? 1.0 : visibility;
}
float sample_shadow_view(uint view_index, float3 world_pos, float3 normal, float4 shadow_params) {
    if (view_index >= lighting.shadow_view_header.x || shadow_params.w < 0.5 || shadow_params.x <= 0.0) return 1.0;
    float visibility = sample_shadow_view_visibility(
        view_index, world_pos, normal, shadow_params, shadow_params.w > 1.5);
    return lerp(1.0, visibility, shadow_params.x);
}
uint select_directional_view(uint first_view, uint view_count, float view_depth) {
    for (uint index = 0u; index < view_count; ++index) {
        if (view_depth < lighting.shadow_views[first_view + index].split_data.y)
            return index;
    }
    return view_count - 1u;
}
float sample_directional_views(uint first_view, uint view_count, float4 shadow_params,
                             float3 world_pos, float3 normal, float view_depth) {
    if (!receives_shadows) return 1.0;
    uint available_views = lighting.shadow_view_header.x;
    if (view_count == 0u || first_view >= available_views) return 1.0;
    view_count = min(view_count, available_views - first_view);
    uint selected = select_directional_view(first_view, view_count, view_depth);
    float shadow = sample_shadow_view(first_view + selected, world_pos, normal, shadow_params);
    if (selected + 1u < view_count) {
        InxShadowView current = lighting.shadow_views[first_view + selected];
        float overlap = max((current.split_data.y - current.split_data.x) * 0.1, 0.0001);
        float remaining = current.split_data.y - view_depth;
        if (remaining < overlap) {
            float next_shadow = sample_shadow_view(first_view + selected + 1u, world_pos, normal, shadow_params);
            shadow = lerp(next_shadow, shadow, smoothstep(0.0, overlap, remaining));
        }
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
    if (a.y >= a.z) return direction_from_light.y >= 0.0 ? 2u : 3u;
    return direction_from_light.z >= 0.0 ? 4u : 5u;
}
float sample_local_shadow(uint first_view, uint view_count, uint light_type, float4 shadow_params,
                        float3 light_position, float3 world_pos, float3 normal) {
    if (!receives_shadows) return 1.0;
    uint available_views = lighting.shadow_view_header.x;
    if (view_count == 0u || first_view >= available_views) return 1.0;
    view_count = min(view_count, available_views - first_view);
    if (light_type == 2u || shadow_params.w < 1.5) {
        uint offset = light_type == 2u ? 0u : min(point_shadow_face(world_pos - light_position), view_count - 1u);
        return sample_shadow_view(first_view + offset, world_pos, normal, shadow_params);
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
    float radius_world = center_view.depth_texel.w * (2.0 * radial_length / inner_resolution);
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
        visibility += sample_shadow_view_visibility(first_view + face, sample_position, normal,
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
                                  frag_view_depth_val);
} else {
    if (lighting.light_counts.x <= 0) return 1.0;
    InxDirectionalLight light = lighting.directional_lights[0];
    return sample_directional_views(light.metadata.z, light.metadata.w, light.shadow_params, world_pos, normal,
                                  frag_view_depth_val);
}
}
InxLight get_main_light(float3 world_pos, float3 normal, float frag_view_depth_val) {
    InxLight l;
if (tiled_lighting) {
    if (light_counts().x > 0u) {
        InxCanonicalLight dl = load_light(0);
        l.direction = normalize(-dl.direction_outer_cos.xyz);
        bool affects_surface = (dl.metadata.w & light_domain) != 0u && (dl.metadata.y & layer_mask) != 0u;
        l.color = affects_surface ? dl.color_intensity.rgb * dl.color_intensity.w : float3(0.0);
        l.attenuation = affects_surface ? 1.0 : 0.0;
        l.shadow = affects_surface ? calculate_shadow(world_pos, normal, frag_view_depth_val) : 1.0;
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
        bool affects_surface = (dl.metadata.y & light_domain) != 0u && (dl.metadata.x & layer_mask) != 0u;
        l.color = affects_surface ? dl.color.rgb * dl.color.w : float3(0.0);
        l.attenuation = affects_surface ? 1.0 : 0.0;
        l.shadow = affects_surface ? calculate_shadow(world_pos, normal, frag_view_depth_val) : 1.0;
    } else {
        l.direction = float3(0.0, 1.0, 0.0);
        l.color = float3(0.0);
        l.attenuation = 0.0;
        l.shadow = 1.0;
    }
}
    return l;
}
float3 evaluate_rectangle_area_light<T : IInxLightResponse>(float3 light_position, float3 ray_direction, float4 right_width, float4 up_height,
                                float3 light_color, float intensity, float range, bool two_sided,
                                uint first_shadow_view, uint shadow_view_count, float4 shadow_params,
                                float3 world_pos, float3 N, T response) {
    float3 from_light = world_pos - light_position;
    float emitter_facing = rectangle_facing(ray_direction, from_light, two_sided);
    if (emitter_facing <= 0.0001) return float3(0.0);
    float3 right = rectangle_extent(right_width);
    float3 up = rectangle_extent(up_height);
    float3 samples[4] = {light_position - right - up, light_position + right - up,
                              light_position + right + up, light_position - right + up};
    float3 result = float3(0.0);
    float shadow = sample_local_shadow(first_shadow_view, shadow_view_count, 3u, shadow_params,
                                     light_position, world_pos, N);
    for (int sample_index = 0; sample_index < 4; ++sample_index) {
        float3 light_vector = samples[sample_index] - world_pos;
        float distance_to_light = length(light_vector);
        if (distance_to_light <= 0.00001 || distance_to_light >= range) continue;
        float3 L = light_vector / distance_to_light;
        float attenuation = calculate_attenuation(float3(range, 0.0, 0.0), distance_to_light);
        float3 radiance = light_color * intensity * 0.25;
        result += response.evaluate(inx_light_sample(L, radiance, attenuation * emitter_facing, shadow));
    }
    return result;
}
float3 calculate_all_lighting<T : IInxLightResponse>(float3 world_pos, float3 N, float view_depth, float shadow, T response) {
    float3 lo = float3(0.0);
if (tiled_lighting) {
    uint directional_count = light_counts().x;
    for (uint i = 0u; i < directional_count; ++i) {
        InxCanonicalLight light = load_light(i);
        if ((light.metadata.w & light_domain) == 0u) continue;
        if ((light.metadata.y & layer_mask) == 0u) continue;
)slang"
                                           R"slang(        float3 L = normalize(-light.direction_outer_cos.xyz);
        float3 radiance = light.color_intensity.rgb * light.color_intensity.w;
        float light_shadow =
            i == 0u ? shadow
                    : sample_directional_views(
                          light.identity_and_shadow.z, light.identity_and_shadow.w,
                          float4(light.shadow_and_inner_cos.xyz, float(light.metadata.z)), world_pos, N, view_depth);
        lo += response.evaluate(inx_light_sample(L, radiance, 1.0, light_shadow));
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
            if ((light.metadata.w & light_domain) == 0u) continue;
            if ((light.metadata.y & layer_mask) == 0u) continue;
            if (light.metadata.x == 3u) {
                lo += evaluate_rectangle_area_light(
                    light.position_range.xyz, light.direction_outer_cos.xyz, light.area_right_width,
                    light.area_up_height, light.color_intensity.rgb, light.color_intensity.w,
                    light.position_range.w, light.direction_outer_cos.w > 0.5, light.identity_and_shadow.z,
                    light.identity_and_shadow.w, float4(light.shadow_and_inner_cos.xyz, float(light.metadata.z)),
                    world_pos, N, response);
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
                world_pos, N);
            float3 radiance = light.color_intensity.rgb * light.color_intensity.w;
            lo += response.evaluate(inx_light_sample(L, radiance, attenuation, local_shadow));
        }
    }
} else {
    for (int i = 0; i < lighting.light_counts.x && i < MAX_DIRECTIONAL_LIGHTS; ++i) {
        InxDirectionalLight light = lighting.directional_lights[i];
        if ((light.metadata.y & light_domain) == 0u) continue;
        if ((light.metadata.x & layer_mask) == 0u) continue;
        float3 L        = normalize(-light.direction.xyz);
        float3 radiance = light.color.rgb * light.color.w;
        float light_shadow =
            i == 0 ? shadow
                   : sample_directional_views(light.metadata.z, light.metadata.w, light.shadow_params, world_pos, N,
                                            view_depth);
        lo += response.evaluate(inx_light_sample(L, radiance, 1.0, light_shadow));
    }
    for (int i = 0; i < lighting.light_counts.y && i < MAX_POINT_LIGHTS; ++i) {
        InxPointLight light = lighting.point_lights[i];
        if ((light.metadata.y & light_domain) == 0u) continue;
        if ((light.metadata.x & layer_mask) == 0u) continue;
        float3  light_vec = light.position.xyz - world_pos;
        float distance = length(light_vec);
        if (distance > 1e-5) {
            float3  L = light_vec / distance;
            float attenuation = calculate_attenuation(light.attenuation.xyz, distance);
            if (attenuation > 0.001) {
                float local_shadow = sample_local_shadow(light.metadata.z, light.metadata.w, 1u, light.shadow_params,
                                                      light.position.xyz, world_pos, N);
                float3 radiance = light.color.rgb * light.color.w;
                lo += response.evaluate(inx_light_sample(L, radiance, attenuation, local_shadow));
            }
        }
    }
    for (int i = 0; i < lighting.light_counts.z && i < MAX_SPOT_LIGHTS; ++i) {
        InxSpotLight light = lighting.spot_lights[i];
        if ((light.metadata.y & light_domain) == 0u) continue;
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
                                                          light.position.xyz, world_pos, N);
                    float3 radiance = light.color.rgb * light.color.w;
                    lo += response.evaluate(inx_light_sample(L, radiance, attenuation * spot_falloff, local_shadow));
                }
            }
        }
    }
    for (int i = 0; i < lighting.light_counts.w && i < MAX_AREA_LIGHTS; ++i) {
        InxAreaLight light = lighting.area_lights[i];
        if ((light.metadata.y & light_domain) == 0u) continue;
        if ((light.metadata.x & layer_mask) == 0u) continue;
        lo += evaluate_rectangle_area_light(
            light.position_range.xyz, light.direction.xyz, light.right_width, light.up_height,
            light.color.rgb, light.color.w, light.position_range.w, light.direction.w > 0.5,
            light.metadata.z, light.metadata.w, light.shadow_params, world_pos, N, response);
    }
}
    return lo;
}
float3 shade<let indirect_only : bool>(InxSurfaceData s, InxShadingContext ctx,
                                      float4 ray_diffuse = float4(0.0), float4 ray_specular = float4(0.0)) {
    let scattering = inx_prepare_lit_scattering(s.albedo, s.metallic, s.smoothness, ctx.normal_variance);
    float3 albedo = scattering.albedo;
    float metallic = scattering.metallic;
    float perceptual_roughness = scattering.perceptual_roughness;
    float ao = clamp(s.occlusion, 0.0, 1.0);
    float3 N = normalize(s.normal_ws);
    float3 V = normalize(lighting.camera_pos.xyz - ctx.position_ws);
    float3  F0    = scattering.f0;
    float f90   = scattering.f90;
    float ndot_v = max(dot(N, V), 0.0);
    float2  env_brdf      = env_brdf_approx(perceptual_roughness, ndot_v);
    float3 energy_compensation = inx_pbr_energy_compensation(F0, env_brdf);
    float3 lo = float3(0.0);
    if (!indirect_only) {
        InxLight main_light = get_main_light(ctx.position_ws, ctx.shadow_plane_ws, ctx.view_depth);
        InxPbrLightResponse response = inx_pbr_light_response(scattering, N, V, energy_compensation,
                                                             s.specular_highlights);
        lo = calculate_all_lighting(ctx.position_ws, ctx.shadow_plane_ws, ctx.view_depth, main_light.shadow, response);
    }
    float3 k_s_env = f_schlick_roughness(F0, f90, ndot_v, perceptual_roughness);
    float3 k_d_env = (1.0 - k_s_env) * (1.0 - metallic);
    // Diffuse alpha is the contribution of traced irradiance in the material blend.
    float3 diffuse_irradiance = ray_diffuse.rgb * INV_PI;
    if (ray_diffuse.a < 1.0)
        diffuse_irradiance = lerp(sample_ambient_irradiance(N), diffuse_irradiance, ray_diffuse.a);
    float3 diffuse_env = k_d_env * albedo * diffuse_irradiance * ao;
    float3 spec_env = ray_specular.rgb;
    if (!(ray_specular.a > 0.0)) {
        float3 refl_dir = get_specular_ambient_direction(N, V, perceptual_roughness);
        float prefilter = saturate(perceptual_roughness * (1.7 - 0.7 * perceptual_roughness));
        float3 specular_irradiance = lerp(sample_ambient_probe(refl_dir),
                                        sample_ambient_irradiance(refl_dir), prefilter);
        float spec_occlusion = compute_specular_occlusion(ndot_v, ao, perceptual_roughness);
        float3 geom_n = normalize(ctx.geometric_normal_ws);
        geom_n = (dot(geom_n, N) < 0.0) ? -geom_n : geom_n;
        float horizon_fade = horizon_occlusion(refl_dir, geom_n);
        spec_env = specular_irradiance * (F0 * env_brdf.x + env_brdf.y) * energy_compensation
                 * (spec_occlusion * horizon_fade) * s.specular_highlights;
    }
    float3 ambient = diffuse_env + spec_env;
    return indirect_only ? ambient : ambient + lo + s.emission;
}
};
// Return raw light visibility and how much of it is covered by this camera's atlas.
// Missing cascade coverage is distinct from an unoccluded receiver.
public float2 inx_secondary_directional_shadow(uint index, float3 world_pos, float3 geometric_normal, float depth)
{
    InxCanonicalLight light = load_light(index);
    uint first_view = light.identity_and_shadow.z;
    uint view_count = light.identity_and_shadow.w;
    uint available_views = lighting.shadow_view_header.x;
    if (view_count == 0u || first_view >= available_views)
        return float2(0.0);
    view_count = min(view_count, available_views - first_view);

    InxLighting<false> context = (InxLighting<false>)0;
    uint selected = context.select_directional_view(first_view, view_count, depth);
    float2 split = lighting.shadow_views[first_view + selected].split_data.xy;
    if (depth < lighting.shadow_views[first_view].split_data.x || depth >= split.y)
        return float2(0.0);

    float4 parameters = float4(light.shadow_and_inner_cos.xyz, float(light.metadata.z));
    bool soft = light.metadata.z > 1u;
    float visibility = context.try_sample_shadow_view_visibility(
        first_view + selected, world_pos, geometric_normal, parameters, soft);
    if (visibility < 0.0)
        return float2(0.0);
    float overlap = max((split.y - split.x) * 0.1, 0.0001);
    float blend = smoothstep(0.0, overlap, split.y - depth);
    if (blend >= 1.0)
        return float2(visibility, 1.0);
    if (selected + 1u < view_count) {
        float next = context.try_sample_shadow_view_visibility(
            first_view + selected + 1u, world_pos, geometric_normal, parameters, soft);
        if (next >= 0.0)
            return float2(lerp(next, visibility, blend), 1.0);
    }
    return float2(visibility, blend);
}

InxLighting<tiled_lighting> inx_lighting_context<let tiled_lighting : bool>(InxShadingContext input)
{
    InxLighting<tiled_lighting> context;
    context.layer_mask = input.layer_mask;
    context.light_domain = input.light_domain;
    context.receives_shadows = input.receives_shadows;
    context.pixel_position = input.pixel_position;
    return context;
}
public float3 inx_shade_pbr<let tiled_lighting : bool>(InxSurfaceData surface, InxShadingContext input)
{
    return inx_lighting_context<tiled_lighting>(input).shade<false>(surface, input);
}
public float3 inx_shade_pbr_ray<let tiled_lighting : bool>(InxSurfaceData surface, InxShadingContext input,
                                                        float4 diffuse, float4 specular)
{
    return inx_lighting_context<tiled_lighting>(input).shade<false>(surface, input, diffuse, specular);
}
public float3 inx_environment_radiance(float3 direction)
{
    float4 panorama = lighting.environment_panorama;
    if (panorama.w >= 0.0)
    {
        float2 uv = float2(frac(atan2(direction.z, direction.x) * (0.5 * INV_PI) + 0.5 + panorama.w),
                           acos(clamp(direction.y, -1.0, 1.0)) * INV_PI);
        uint width, height;
        environment_panorama_image.GetDimensions(width, height);
        int2 size = int2(width, height);
        // Normalize the panorama's seam and pole addressing across texture formats.
        // Texel loads are also valid on WebGPU devices without float32-filterable.
        float2 position = uv * float2(size) - 0.5;
        int2 base = int2(floor(position));
        float2 blend = frac(position);
        int2 x = (int2(base.x, base.x + 1) + size.x) % size.x;
        int2 y = clamp(int2(base.y, base.y + 1), 0, size.y - 1);
        float3 north = lerp(environment_panorama_image.Load(int3(x.x, y.x, 0)).rgb,
                            environment_panorama_image.Load(int3(x.y, y.x, 0)).rgb, blend.x);
        float3 south = lerp(environment_panorama_image.Load(int3(x.x, y.y, 0)).rgb,
                            environment_panorama_image.Load(int3(x.y, y.y, 0)).rgb, blend.x);
        return lerp(north, south, blend.y) * panorama.rgb;
    }
    return InxLighting<false>().sample_ambient_probe(direction);
}
public float3 inx_pbr_indirect(InxSurfaceData surface, InxShadingContext input)
{
    return inx_lighting_context<false>(input).shade<true>(surface, input);
}
public InxLight inx_main_light<let tiled_lighting : bool>(InxShadingContext input, float3 normal_ws)
{
    return inx_lighting_context<tiled_lighting>(input).get_main_light(input.position_ws, normal_ws, input.view_depth);
}
public float3 inx_view_direction(InxShadingContext input)
{
    return normalize(lighting.camera_pos.xyz - input.position_ws);
}

public float3 inx_direct_lighting<let tiled_lighting : bool, T : IInxLightResponse>(
    InxShadingContext input, float3 normal_ws, T response)
{
    InxLighting<tiled_lighting> context = inx_lighting_context<tiled_lighting>(input);
    InxLight main_light = context.get_main_light(input.position_ws, normal_ws, input.view_depth);
    return context.calculate_all_lighting(input.position_ws, normal_ws, input.view_depth, main_light.shadow, response);
}
public float3 inx_ambient_irradiance(InxShadingContext input, float3 normal_ws)
{
    return inx_lighting_context<false>(input).sample_ambient_irradiance(normal_ws);
}
)slang"};
    return module;
}
} // namespace infernux
