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

// Resolve generated names once at publication. Every backend binds an author
// field by its logical name and uses reflection for its physical descriptor.
// Image and sampler descriptors share a name, but retain distinct types/slots.
inline void ApplyShaderAuthorResources(ShaderCompilation &compilation,
                                       const std::vector<std::pair<std::string, std::string>> &fields)
{
    std::unordered_map<std::string, const std::pair<std::string, std::string> *> roots;
    for (const auto &field : fields)
        roots.emplace("inx_" + field.first, &field);
    for (auto &target : compilation.targets)
        for (auto &resource : target.resources) {
            const auto field = roots.find(resource.name.substr(0, resource.name.find('.')));
            if (field == roots.end())
                continue; // Engine-owned resources retain their native names.
            const auto &[name, kind] = *field->second;
            resource.name = name;
            // Raw depth uses non-filtering reads on both depth planes and R32 resolves.
            if (kind == "Depth2D" || kind == "Depth2DMS" || kind == "Color2DMS") {
                if (resource.binding.type == rhi::BindingType::SampledTexture) {
                    resource.binding.depthRead = false;
                    resource.binding.sampleType = rhi::TextureSampleType::UnfilterableFloat;
                } else if (resource.binding.type == rhi::BindingType::Sampler)
                    resource.binding.samplerType = rhi::SamplerBindingType::NonFiltering;
            }
        }
}

} // namespace infernux
