#pragma once

#include <function/renderer/shader/ShaderProgramCode.h>

#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
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
    for (auto &target : compilation.targets) {
        std::unordered_set<std::string_view> sampled;
        std::vector<ShaderResourceLayout *> colorImages;
        for (auto &resource : target.resources) {
            const auto field = roots.find(resource.name.substr(0, resource.name.find('.')));
            if (field == roots.end())
                continue; // Engine-owned resources retain their native names.
            const auto &[name, kind] = *field->second;
            resource.name = name;
            if (kind == "Color2D" || kind == "Color3D" || kind == "History2D" || kind == "texture_2d") {
                if (resource.binding.type == rhi::BindingType::SampledTexture)
                    colorImages.push_back(&resource);
                else if (resource.binding.type == rhi::BindingType::Sampler &&
                         resource.binding.visibility != rhi::ShaderStage::None)
                    sampled.insert(name);
            }
            // Raw depth uses non-filtering reads on both depth planes and R32 resolves.
            if (kind == "Depth2D" || kind == "Depth2DMS" || kind == "Color2DMS") {
                if (resource.binding.type == rhi::BindingType::SampledTexture) {
                    resource.binding.depthRead = false;
                    resource.binding.sampleType = rhi::TextureSampleType::UnfilterableFloat;
                } else if (resource.binding.type == rhi::BindingType::Sampler)
                    resource.binding.samplerType = rhi::SamplerBindingType::NonFiltering;
            }
        }
        // Author images carry their own sampler. A load-only image needs no filtering
        // capability; WebGPU can then bind regular float images and full-precision HDR.
        for (auto *image : colorImages)
            if (sampled.find(image->name) == sampled.end())
                image->binding.sampleType = rhi::TextureSampleType::UnfilterableFloat;
    }
}

} // namespace infernux
