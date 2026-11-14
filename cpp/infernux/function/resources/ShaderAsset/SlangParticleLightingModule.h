#pragma once

#include "SlangCompiler.h"

namespace infernux
{

inline const ShaderSourceModule &GetSlangParticleLightingModule()
{
    static const ShaderSourceModule module{"inx_particle_lighting", "<infernux/shader/particle_lighting>",
                                           R"slang(import inx_geometry;
import inx_particle;
import inx_lighting;

InxShadingContext particle_shading_context(InxParticleInput input)
{
    InxShadingContext result = inx_shading_context(
        input.position_ws, input.normal_ws, input.view_depth, input.position.xy,
        input.layer_mask, inx_particle_draw().rendering_control.x > 0.5);
    result.light_domain = 2u;
    return result;
}

[__NoSideEffect]
public float3 inx_particle_pbr(InxSurfaceData surface, InxParticleInput input)
{
    if (inx_particle_draw().lighting_control.w < 0.5)
        return surface.albedo + surface.emission;
    InxShadingContext context = particle_shading_context(input);
    if (inx_particle_draw().lighting_control.x > 0.5)
        return inx_shade_pbr<true>(surface, context);
    return inx_shade_pbr<false>(surface, context);
}

float3 six_way_transmission(float transmission, float range, float3 density_scale)
{
    return pow(saturate(float3(transmission / max(range, 1e-5))), density_scale) * range;
}

public struct SixWayResponse : IInxLightResponse
{
    float3 right, up, back;
    float3 positive, negative;
    float3 density_scale;
    float range;
    public float3 evaluate(InxLight light)
    {
        float3 direction = float3(dot(right, light.direction), dot(up, light.direction), dot(back, light.direction));
        float3 coefficients = float3(direction.x >= 0.0 ? positive.x : negative.x,
                                      direction.y >= 0.0 ? positive.y : negative.y,
                                      direction.z >= 0.0 ? positive.z : negative.z);
        float transmission = max(dot(coefficients, direction * direction), 0.0);
        return six_way_transmission(transmission, range, density_scale) *
            light.color * (light.attenuation * light.shadow * 3.14159265359);
    }
};

[__NoSideEffect]
public float3 inx_particle_six_way_smoke(InxParticleInput input, float3 positive, float3 negative,
    float3 tint, float alpha, float absorption, float lighting_intensity, float ambient_intensity, float ambient_saturation)
{
    if (inx_particle_draw().lighting_control.w < 0.5)
        return float3(0.0);
    SixWayResponse response;
    response.right = normalize(input.tangent_ws.xyz);
    response.back = normalize(input.normal_ws);
    response.up = normalize(cross(response.back, response.right)) * input.tangent_ws.w;
    response.positive = max(positive, 0.0);
    response.negative = max(negative, 0.0);
    response.density_scale = 1.0 + log2(max(tint, 1e-5)) / log2(0.2);
    response.range = lerp(0.31830988618, 1.0, saturate(absorption)) * max(alpha, 0.0);
    InxShadingContext context = particle_shading_context(input);
    float3 direct;
    if (inx_particle_draw().lighting_control.x > 0.5)
        direct = inx_direct_lighting<true, SixWayResponse>(context, response.back, response);
    else
        direct = inx_direct_lighting<false, SixWayResponse>(context, response.back, response);
    float3 ambient = float3(0.0);
    float ambient_weight = clamp(ambient_intensity, 0.0, 0.002);
    if (ambient_weight > 0.0001) {
        float3 directions[6] = {response.right, -response.right, response.up, -response.up, response.back, -response.back};
        float coefficients[6] = {positive.x, negative.x, positive.y, negative.y, positive.z, negative.z};
        for (uint index = 0; index < 6; ++index)
            ambient += six_way_transmission(coefficients[index], response.range, response.density_scale) *
                inx_ambient_irradiance(context, directions[index]);
        ambient /= 6.0;
        float energy = max(dot(ambient, float3(0.2126, 0.7152, 0.0722)), 0.0);
        float neutral = min(energy, 1.0);
        float3 capped = saturate(ambient * (neutral / max(energy, 1e-5)));
        ambient = lerp(float3(neutral), capped, saturate(ambient_saturation)) *
            lerp(1.0, 0.18, smoothstep(0.15, 0.85, alpha)) * ambient_weight;
    }
    return max(direct * lighting_intensity + ambient, 0.0);
}
)slang"};
    return module;
}

} // namespace infernux
