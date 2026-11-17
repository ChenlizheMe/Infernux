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

public InxLitScattering inx_prepare_lit_scattering(float3 authored_albedo, float authored_metallic,
                                                    float smoothness, float normal_variance)
{
    float3 albedo = clamp(authored_albedo, 0.0, 1.0);
    float metallic = clamp(authored_metallic, 0.0, 1.0);
    float perceptual = clamp(1.0 - smoothness, 0.045, 1.0);
    float alpha = perceptual * perceptual;
    float kernel_roughness = min(normal_variance, 0.18);
    alpha = max(alpha, sqrt(alpha * alpha + kernel_roughness));
    perceptual = max(perceptual, sqrt(alpha));
    float3 f0 = lerp(float3(0.04), albedo, metallic);
    float f90 = saturate(50.0 * dot(f0, float3(0.33333)));
    return {albedo, metallic, perceptual, alpha, f0, f90};
}
)slang"};
    return module;
}

} // namespace infernux
