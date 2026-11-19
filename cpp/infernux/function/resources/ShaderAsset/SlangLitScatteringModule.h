#pragma once

#include "SlangCompiler.h"

namespace infernux
{

inline const ShaderSourceModule &GetSlangLitScatteringModule()
{
    static const ShaderSourceModule module{"inx_lit_scattering", "<infernux/shader/lit_scattering>", R"slang(
public struct InxLitScattering
{
    public float3 albedo;
    public float metallic;
    public float perceptual_roughness;
    public float ggx_alpha;
    public float3 f0;
    public float f90;
};

public float inx_footprint_ggx_alpha(float perceptual, float normal_variance)
{
    float alpha = perceptual * perceptual;
    return sqrt(alpha * alpha + min(normal_variance, 0.18));
}

// At the specular peak the denominator is alpha squared, even when NoH rounds to one.
public float inx_ggx_ndf_denominator(float noh, float alpha_squared)
{
    return (1.0 - noh) * (1.0 + noh) + alpha_squared * noh * noh;
}

public float inx_ggx_projected_roughness(float cosine, float alpha_squared)
{
    float cosine_squared = cosine * cosine;
    return sqrt(cosine_squared + alpha_squared * (1.0 - cosine_squared));
}

public InxLitScattering inx_prepare_lit_scattering(float3 authored_albedo, float authored_metallic,
                                                    float smoothness, float normal_variance)
{
    float3 albedo = clamp(authored_albedo, 0.0, 1.0);
    float metallic = clamp(authored_metallic, 0.0, 1.0);
    float perceptual = clamp(1.0 - smoothness, 0.045, 1.0);
    float alpha = inx_footprint_ggx_alpha(perceptual, normal_variance);
    perceptual = max(perceptual, sqrt(alpha));
    float3 f0 = lerp(float3(0.04), albedo, metallic);
    float f90 = saturate(50.0 * dot(f0, float3(0.33333)));
    return {albedo, metallic, perceptual, alpha, f0, f90};
}
)slang"};
    return module;
}

} // namespace infernux
