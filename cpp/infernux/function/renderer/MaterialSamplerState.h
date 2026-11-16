#pragma once

#include "rhi/RhiCapabilities.h"
#include "rhi/RhiDescriptors.h"

#include <function/resources/InxMaterial/MaterialProperty.h>

#include <algorithm>
#include <stdexcept>
#include <string_view>

namespace infernux
{

inline void ApplyMaterialSamplerOverrides(rhi::SamplerDesc &sampler, const MaterialTextureSampler &overrides)
{
    const auto filter = [](MaterialSamplerFilter value, rhi::FilterMode inherited) {
        switch (value) {
        case MaterialSamplerFilter::Nearest:
            return rhi::FilterMode::Nearest;
        case MaterialSamplerFilter::Linear:
            return rhi::FilterMode::Linear;
        default:
            return inherited;
        }
    };
    const auto address = [](MaterialSamplerAddress value, rhi::AddressMode inherited) {
        switch (value) {
        case MaterialSamplerAddress::Repeat:
            return rhi::AddressMode::Repeat;
        case MaterialSamplerAddress::Clamp:
            return rhi::AddressMode::ClampToEdge;
        case MaterialSamplerAddress::Mirror:
            return rhi::AddressMode::MirroredRepeat;
        default:
            return inherited;
        }
    };
    sampler.minFilter = filter(overrides.minFilter, sampler.minFilter);
    sampler.magFilter = filter(overrides.magFilter, sampler.magFilter);
    sampler.mipFilter = filter(overrides.mipFilter, sampler.mipFilter);
    sampler.addressU = address(overrides.addressU, sampler.addressU);
    sampler.addressV = address(overrides.addressV, sampler.addressV);
    sampler.addressW = address(overrides.addressW, sampler.addressW);
    // An explicit point-filter override wins over asset anisotropy on every backend.
    if (sampler.minFilter != rhi::FilterMode::Linear || sampler.magFilter != rhi::FilterMode::Linear ||
        sampler.mipFilter != rhi::FilterMode::Linear)
        sampler.maxAnisotropy = 1.0f;
}

/// Resolve imported texture settings once, when publishing its GPU resource.
/// Renderers use the same result for scene, UI and material-override bindings.
[[nodiscard]] inline rhi::SamplerDesc MaterialSamplerDescription(std::string_view filterMode, std::string_view wrapMode,
                                                                 int anisoLevel, uint32_t mipCount,
                                                                 const rhi::DeviceCaps &caps,
                                                                 const MaterialTextureSampler &overrides = {})
{
    rhi::SamplerDesc sampler;
    sampler.maxLod = static_cast<float>(mipCount - 1);
    const float maximum = caps.features.samplerAnisotropy ? (std::max)(1.0f, caps.limits.maxSamplerAnisotropy) : 1.0f;
    sampler.maxAnisotropy = std::clamp(anisoLevel < 0 ? maximum : static_cast<float>(anisoLevel), 1.0f, maximum);
    if (filterMode == "point") {
        sampler.minFilter = sampler.magFilter = sampler.mipFilter = rhi::FilterMode::Nearest;
        sampler.maxAnisotropy = 1.0f;
    } else if (filterMode == "linear" || filterMode == "bilinear") {
        sampler.mipFilter = sampler.maxAnisotropy > 1.0f ? rhi::FilterMode::Linear : rhi::FilterMode::Nearest;
    } else if (filterMode != "trilinear") {
        throw std::invalid_argument("texture filter mode must be point, linear, bilinear or trilinear");
    }
    if (wrapMode == "clamp")
        sampler.addressU = sampler.addressV = sampler.addressW = rhi::AddressMode::ClampToEdge;
    else if (wrapMode == "mirror")
        sampler.addressU = sampler.addressV = sampler.addressW = rhi::AddressMode::MirroredRepeat;
    ApplyMaterialSamplerOverrides(sampler, overrides);
    return sampler;
}

} // namespace infernux
