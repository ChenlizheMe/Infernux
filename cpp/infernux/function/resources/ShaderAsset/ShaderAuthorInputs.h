#pragma once

#include <function/renderer/shader/ShaderProgramCode.h>

#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace infernux
{

inline rhi::ComputeBufferDesc ShaderBufferElementLayout(std::string_view element)
{
    static const std::unordered_map<std::string_view, rhi::ComputeScalarType> scalars = {
        {"float", rhi::ComputeScalarType::Float32},
        {"int", rhi::ComputeScalarType::Int32},
        {"uint", rhi::ComputeScalarType::UInt32},
    };
    const uint8_t lanes = element.back() >= '2' && element.back() <= '4' ? element.back() - '0' : 1;
    return {1, scalars.at(element.substr(0, element.size() - (lanes > 1))), lanes};
}

// Logical image contracts complement target reflection. Raw depth is sampled
// without filtering so the same program accepts depth planes and R32 resolves.
inline void ApplyShaderAuthorInputs(ShaderCompilation &compilation,
                                    const std::vector<std::pair<std::string, std::string>> &inputs)
{
    for (const auto &[name, kind] : inputs) {
        if (kind != "Depth2D" && kind != "Depth2DMS" && kind != "Color2DMS")
            continue;
        const auto image = "inx_" + name + ".texture";
        const auto sampler = "inx_" + name + ".sampler";
        for (auto &target : compilation.targets)
            for (auto &resource : target.resources) {
                if (resource.name == image) {
                    resource.binding.depthRead = false;
                    resource.binding.sampleType = rhi::TextureSampleType::UnfilterableFloat;
                } else if (resource.name == sampler)
                    resource.binding.samplerType = rhi::SamplerBindingType::NonFiltering;
            }
    }
}

} // namespace infernux
