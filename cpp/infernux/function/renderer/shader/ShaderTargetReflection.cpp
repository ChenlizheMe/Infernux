#include "ShaderProgramCode.h"
#include "ShaderReflection.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace infernux
{
namespace
{
VkFormat StageFormat(const ShaderStageVariable &value)
{
    using Scalar = ShaderScalarKind;
    const std::array<VkFormat, 4> *formats;
    static constexpr std::array f32{VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32B32_SFLOAT,
                                    VK_FORMAT_R32G32B32A32_SFLOAT};
    static constexpr std::array i32{VK_FORMAT_R32_SINT, VK_FORMAT_R32G32_SINT, VK_FORMAT_R32G32B32_SINT,
                                    VK_FORMAT_R32G32B32A32_SINT};
    static constexpr std::array u32{VK_FORMAT_R32_UINT, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32B32_UINT,
                                    VK_FORMAT_R32G32B32A32_UINT};
    static constexpr std::array f16{VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R16G16B16_SFLOAT,
                                    VK_FORMAT_R16G16B16A16_SFLOAT};
    static constexpr std::array i16{VK_FORMAT_R16_SINT, VK_FORMAT_R16G16_SINT, VK_FORMAT_R16G16B16_SINT,
                                    VK_FORMAT_R16G16B16A16_SINT};
    static constexpr std::array u16{VK_FORMAT_R16_UINT, VK_FORMAT_R16G16_UINT, VK_FORMAT_R16G16B16_UINT,
                                    VK_FORMAT_R16G16B16A16_UINT};
    static constexpr std::array f64{VK_FORMAT_R64_SFLOAT, VK_FORMAT_R64G64_SFLOAT, VK_FORMAT_R64G64B64_SFLOAT,
                                    VK_FORMAT_R64G64B64A64_SFLOAT};
    static constexpr std::array i64{VK_FORMAT_R64_SINT, VK_FORMAT_R64G64_SINT, VK_FORMAT_R64G64B64_SINT,
                                    VK_FORMAT_R64G64B64A64_SINT};
    static constexpr std::array u64{VK_FORMAT_R64_UINT, VK_FORMAT_R64G64_UINT, VK_FORMAT_R64G64B64_UINT,
                                    VK_FORMAT_R64G64B64A64_UINT};
    static constexpr std::array i8{VK_FORMAT_R8_SINT, VK_FORMAT_R8G8_SINT, VK_FORMAT_R8G8B8_SINT,
                                   VK_FORMAT_R8G8B8A8_SINT};
    static constexpr std::array u8{VK_FORMAT_R8_UINT, VK_FORMAT_R8G8_UINT, VK_FORMAT_R8G8B8_UINT,
                                   VK_FORMAT_R8G8B8A8_UINT};
    switch (value.scalar) {
    case Scalar::Float32:
        formats = &f32;
        break;
    case Scalar::Int32:
        formats = &i32;
        break;
    case Scalar::UInt32:
        formats = &u32;
        break;
    case Scalar::Float16:
        formats = &f16;
        break;
    case Scalar::Int16:
        formats = &i16;
        break;
    case Scalar::UInt16:
        formats = &u16;
        break;
    case Scalar::Float64:
        formats = &f64;
        break;
    case Scalar::Int64:
        formats = &i64;
        break;
    case Scalar::UInt64:
        formats = &u64;
        break;
    case Scalar::Int8:
        formats = &i8;
        break;
    case Scalar::UInt8:
        formats = &u8;
        break;
    default:
        throw std::runtime_error("Graphics stage value has no device format: " + value.name);
    }
    return formats->at(value.components - 1);
}

ReflectedImageDimension ImageDimension(rhi::TextureViewDimension dimension)
{
    using Dimension = rhi::TextureViewDimension;
    switch (dimension) {
    case Dimension::Texture1D:
    case Dimension::Texture1DArray:
        return ReflectedImageDimension::D1;
    case Dimension::Texture2D:
    case Dimension::Texture2DArray:
        return ReflectedImageDimension::D2;
    case Dimension::Texture3D:
        return ReflectedImageDimension::D3;
    case Dimension::Cube:
    case Dimension::CubeArray:
        return ReflectedImageDimension::Cube;
    default:
        return ReflectedImageDimension::Unknown;
    }
}
} // namespace

void ShaderReflection::LoadTargetLayout(const ShaderTargetCode &target, std::string_view entryPoint)
{
    Clear();
    const auto found = std::find_if(target.entries.begin(), target.entries.end(),
                                    [&](const auto &entry) { return entry.name == entryPoint; });
    if (found == target.entries.end())
        throw std::logic_error("Graphics program entry is absent from its compiler layout: " + std::string(entryPoint));
    const auto &entry = *found;
    m_stage = entry.stage == rhi::ShaderStage::Vertex ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
    m_sampleRateShading = entry.sampleRateShading;
    for (const auto &value : entry.inputs)
        m_inputs.push_back({value.name, value.location, StageFormat(value)});
    for (const auto &value : entry.outputs)
        m_outputs.push_back({value.name, value.location, StageFormat(value)});
    for (const auto &resource : target.resources) {
        const auto &binding = resource.binding;
        const std::array<uint32_t, 2> key{resource.set, binding.binding};
        if (std::find(entry.bindings.begin(), entry.bindings.end(), key) == entry.bindings.end())
            continue;
        switch (binding.type) {
        case rhi::BindingType::UniformBuffer:
            m_uniformBuffers.push_back({resource.name, binding.binding, resource.set,
                                        static_cast<uint32_t>(resource.value.byteSize),
                                        static_cast<VkShaderStageFlags>(m_stage), resource.value.fields});
            break;
        case rhi::BindingType::StorageBuffer: {
            StorageBufferInfo buffer;
            buffer.name = resource.name;
            buffer.binding = binding.binding;
            buffer.set = resource.set;
            buffer.arraySize = binding.count;
            buffer.stageFlags = m_stage;
            buffer.readOnly = binding.readOnlyStorage;
            m_storageBuffers.push_back(std::move(buffer));
            break;
        }
        case rhi::BindingType::StorageTexture:
            m_storageImages.push_back({resource.name, binding.binding, resource.set, binding.count,
                                       static_cast<VkShaderStageFlags>(m_stage),
                                       binding.storageAccess == rhi::StorageTextureAccess::ReadOnly,
                                       binding.storageAccess == rhi::StorageTextureAccess::WriteOnly});
            break;
        case rhi::BindingType::SampledTexture:
        case rhi::BindingType::Sampler:
        case rhi::BindingType::CombinedTextureSampler: {
            SampledImageInfo image;
            image.name = resource.name;
            image.binding = binding.binding;
            image.set = resource.set;
            image.arraySize = binding.count;
            image.stageFlags = m_stage;
            image.descriptorType = binding.type == rhi::BindingType::Sampler ? VK_DESCRIPTOR_TYPE_SAMPLER
                                   : binding.type == rhi::BindingType::SampledTexture
                                       ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
                                       : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            image.multisampled = binding.multisampled;
            image.dimension = ImageDimension(binding.viewDimension);
            image.arrayed = binding.viewDimension == rhi::TextureViewDimension::Texture1DArray ||
                            binding.viewDimension == rhi::TextureViewDimension::Texture2DArray ||
                            binding.viewDimension == rhi::TextureViewDimension::CubeArray;
            image.hasArrayDimension = binding.count != 1;
            m_sampledImages.push_back(std::move(image));
            break;
        }
        }
    }
}

} // namespace infernux
