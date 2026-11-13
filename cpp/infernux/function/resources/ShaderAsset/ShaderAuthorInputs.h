#pragma once

#include <function/renderer/shader/ShaderProgramCode.h>

#include <string>
#include <utility>
#include <vector>

namespace infernux
{

// Logical image contracts complement target reflection. Raw depth is sampled
// without filtering so the same program accepts depth planes and R32 resolves.
inline void ApplyShaderAuthorInputs(ShaderCompilation &compilation,
                                    const std::vector<std::pair<std::string, std::string>> &inputs)
{
    for (const auto &[name, kind] : inputs) {
        if (kind != "Depth2D")
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
