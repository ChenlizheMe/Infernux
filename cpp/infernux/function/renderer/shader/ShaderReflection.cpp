#include "ShaderReflection.h"
#include <core/log/InxLog.h>

// SPIRV-Cross headers
#if defined(INFERNUX_SPIRV_CROSS_FLAT_INCLUDE)
#include <spirv_cross.hpp>
#include <spirv_glsl.hpp>
#else
#include <spirv_cross/spirv_cross.hpp>
#include <spirv_cross/spirv_glsl.hpp>
#endif

#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace infernux
{
namespace
{
template <typename T, typename = void> struct HasGlPlainUniforms : std::false_type
{
};

template <typename T>
struct HasGlPlainUniforms<T, std::void_t<decltype(std::declval<const T &>().gl_plain_uniforms)>> : std::true_type
{
};

template <typename T, typename = void> struct HasTensors : std::false_type
{
};

template <typename T> struct HasTensors<T, std::void_t<decltype(std::declval<const T &>().tensors)>> : std::true_type
{
};

template <typename T, typename = void> struct HasShaderRecordBuffers : std::false_type
{
};

template <typename T>
struct HasShaderRecordBuffers<T, std::void_t<decltype(std::declval<const T &>().shader_record_buffers)>>
    : std::true_type
{
};

ReflectedImageDimension ImageDimension(spv::Dim dimension)
{
    switch (dimension) {
    case spv::Dim1D:
        return ReflectedImageDimension::D1;
    case spv::Dim2D:
        return ReflectedImageDimension::D2;
    case spv::Dim3D:
        return ReflectedImageDimension::D3;
    case spv::DimCube:
        return ReflectedImageDimension::Cube;
    case spv::DimRect:
        return ReflectedImageDimension::Rect;
    case spv::DimBuffer:
        return ReflectedImageDimension::Buffer;
    case spv::DimSubpassData:
        return ReflectedImageDimension::SubpassData;
    default:
        return ReflectedImageDimension::Unknown;
    }
}

VkFormat StageIoFormat(const spirv_cross::SPIRType &type)
{
    if (type.width != 32)
        return VK_FORMAT_UNDEFINED;

    if (type.basetype == spirv_cross::SPIRType::Float) {
        switch (type.vecsize) {
        case 1:
            return VK_FORMAT_R32_SFLOAT;
        case 2:
            return VK_FORMAT_R32G32_SFLOAT;
        case 3:
            return VK_FORMAT_R32G32B32_SFLOAT;
        case 4:
            return VK_FORMAT_R32G32B32A32_SFLOAT;
        default:
            return VK_FORMAT_UNDEFINED;
        }
    }
    if (type.basetype == spirv_cross::SPIRType::Int) {
        switch (type.vecsize) {
        case 1:
            return VK_FORMAT_R32_SINT;
        case 2:
            return VK_FORMAT_R32G32_SINT;
        case 3:
            return VK_FORMAT_R32G32B32_SINT;
        case 4:
            return VK_FORMAT_R32G32B32A32_SINT;
        default:
            return VK_FORMAT_UNDEFINED;
        }
    }
    if (type.basetype == spirv_cross::SPIRType::UInt) {
        switch (type.vecsize) {
        case 1:
            return VK_FORMAT_R32_UINT;
        case 2:
            return VK_FORMAT_R32G32_UINT;
        case 3:
            return VK_FORMAT_R32G32B32_UINT;
        case 4:
            return VK_FORMAT_R32G32B32A32_UINT;
        default:
            return VK_FORMAT_UNDEFINED;
        }
    }
    return VK_FORMAT_UNDEFINED;
}

ShaderScalarKind ScalarKind(spirv_cross::SPIRType::BaseType type)
{
    using T = spirv_cross::SPIRType;
    switch (type) {
    case T::Boolean:
        return ShaderScalarKind::Bool;
    case T::SByte:
        return ShaderScalarKind::Int8;
    case T::UByte:
        return ShaderScalarKind::UInt8;
    case T::Short:
        return ShaderScalarKind::Int16;
    case T::UShort:
        return ShaderScalarKind::UInt16;
    case T::Int:
        return ShaderScalarKind::Int32;
    case T::UInt:
        return ShaderScalarKind::UInt32;
    case T::Int64:
        return ShaderScalarKind::Int64;
    case T::UInt64:
        return ShaderScalarKind::UInt64;
    case T::Half:
        return ShaderScalarKind::Float16;
    case T::Float:
        return ShaderScalarKind::Float32;
    case T::Double:
        return ShaderScalarKind::Float64;
    default:
        return ShaderScalarKind::None;
    }
}

// Temporary SPIR-V producer for the remaining legacy programs. Both compiler
// paths publish the same physical layout; material upload never guesses std140.
ShaderValueLayout ReflectValue(spirv_cross::Compiler &compiler, spirv_cross::TypeID id, uint32_t matrixStride = 0,
                               bool rowMajor = false)
{
    const auto &type = compiler.get_type(id);
    ShaderValueLayout value;
    if (!type.array.empty()) {
        value.kind = ShaderValueKind::Array;
        value.elementCount = type.array.back();
        value.stride = compiler.get_decoration(id, spv::DecorationArrayStride);
        value.fields.push_back(ReflectValue(compiler, type.parent_type, matrixStride, rowMajor));
        value.byteSize = value.elementCount * value.stride;
    } else if (type.basetype == spirv_cross::SPIRType::Struct) {
        value.kind = ShaderValueKind::Struct;
        value.byteSize = compiler.get_declared_struct_size(type);
        for (uint32_t index = 0; index < type.member_types.size(); ++index) {
            const auto &memberType = compiler.get_type(type.member_types[index]);
            const auto matrixStride =
                memberType.columns > 1 ? compiler.type_struct_member_matrix_stride(type, index) : 0;
            auto member = ReflectValue(compiler, type.member_types[index], matrixStride,
                                       compiler.has_member_decoration(type.self, index, spv::DecorationRowMajor));
            member.name = compiler.get_member_name(type.self, index);
            member.offset = compiler.type_struct_member_offset(type, index);
            member.byteSize = compiler.get_declared_struct_member_size(type, index);
            value.fields.push_back(std::move(member));
        }
    } else {
        value.scalar = ScalarKind(type.basetype);
        value.rowMajor = rowMajor;
        value.rows = type.columns > 1 ? type.vecsize : 1;
        value.columns = type.columns > 1 ? type.columns : type.vecsize;
        value.kind = type.columns > 1 ? ShaderValueKind::Matrix
                                      : (type.vecsize > 1 ? ShaderValueKind::Vector : ShaderValueKind::Scalar);
        value.stride = matrixStride;
        value.byteSize = type.columns > 1 ? matrixStride * (rowMajor ? value.rows : value.columns)
                                          : (type.width / 8u) * type.vecsize;
    }
    return value;
}
} // namespace

bool ShaderReflection::Reflect(const std::vector<char> &spirvCode, VkShaderStageFlagBits stage)
{
    if (spirvCode.empty() || spirvCode.size() % 4 != 0) {
        INXLOG_ERROR("Invalid SPIR-V code size: ", spirvCode.size());
        return false;
    }

    // Convert char vector to uint32_t vector
    std::vector<uint32_t> spirv(spirvCode.size() / 4);
    std::memcpy(spirv.data(), spirvCode.data(), spirvCode.size());

    return Reflect(spirv, stage);
}

bool ShaderReflection::Reflect(const std::vector<uint32_t> &spirvCode, VkShaderStageFlagBits stage)
{
    Clear();
    m_stage = stage;

    if (spirvCode.empty()) {
        INXLOG_ERROR("Empty SPIR-V code");
        return false;
    }

    try {
        spirv_cross::Compiler compiler(spirvCode);
        const auto &capabilities = compiler.get_declared_capabilities();
        m_sampleRateShading =
            std::find(capabilities.begin(), capabilities.end(), spv::CapabilitySampleRateShading) != capabilities.end();

        // Get all shader resources
        spirv_cross::ShaderResources resources = compiler.get_shader_resources();

        // The general renderer has specialized descriptor paths for these
        // classes; UI has only one combined 2D sampler. Preserve every other
        // SPIR-V resource class so UI validation cannot silently miss one.
        const auto recordUnsupported = [this](const auto &entries, const char *kind) {
            for (const auto &entry : entries)
                m_unsupportedDescriptorResources.emplace_back(std::string(kind) + " '" + entry.name + "'");
        };
        recordUnsupported(resources.subpass_inputs, "input attachment");
        recordUnsupported(resources.atomic_counters, "atomic counter");
        recordUnsupported(resources.acceleration_structures, "acceleration structure");
        const auto recordExtendedUnsupported = [&recordUnsupported](const auto &resourceTable) {
            using ResourceTable = std::decay_t<decltype(resourceTable)>;
            if constexpr (HasGlPlainUniforms<ResourceTable>::value)
                recordUnsupported(resourceTable.gl_plain_uniforms, "plain uniform");
            if constexpr (HasTensors<ResourceTable>::value)
                recordUnsupported(resourceTable.tensors, "tensor");
            if constexpr (HasShaderRecordBuffers<ResourceTable>::value)
                recordUnsupported(resourceTable.shader_record_buffers, "shader record buffer");
        };
        recordExtendedUnsupported(resources);

        // Process uniform buffers
        for (const auto &ubo : resources.uniform_buffers) {
            UniformBufferInfo info;
            info.name = ubo.name;
            info.binding = compiler.get_decoration(ubo.id, spv::DecorationBinding);
            info.set = compiler.get_decoration(ubo.id, spv::DecorationDescriptorSet);
            info.stageFlags = stage;

            // Get buffer size
            const auto &type = compiler.get_type(ubo.base_type_id);
            info.size = static_cast<uint32_t>(compiler.get_declared_struct_size(type));

            info.members = ReflectValue(compiler, ubo.base_type_id).fields;

            m_uniformBuffers.push_back(info);
        }

        // Process sampled images (combined image samplers and separate textures)
        for (const auto &image : resources.sampled_images) {
            SampledImageInfo info;
            info.name = image.name;
            info.binding = compiler.get_decoration(image.id, spv::DecorationBinding);
            info.set = compiler.get_decoration(image.id, spv::DecorationDescriptorSet);
            info.stageFlags = stage;
            info.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

            const auto &type = compiler.get_type(image.type_id);
            info.arraySize = type.array.empty() ? 1 : type.array[0];
            info.multisampled = type.image.ms;
            info.dimension = ImageDimension(type.image.dim);
            info.arrayed = type.image.arrayed;
            info.hasArrayDimension = !type.array.empty();

            m_sampledImages.push_back(info);
        }

        // Process separate samplers
        for (const auto &sampler : resources.separate_samplers) {
            SampledImageInfo info;
            info.name = sampler.name;
            info.binding = compiler.get_decoration(sampler.id, spv::DecorationBinding);
            info.set = compiler.get_decoration(sampler.id, spv::DecorationDescriptorSet);
            info.stageFlags = stage;
            info.arraySize = 1;
            info.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;

            m_sampledImages.push_back(info);
        }

        // Process separate images
        for (const auto &image : resources.separate_images) {
            SampledImageInfo info;
            info.name = image.name;
            info.binding = compiler.get_decoration(image.id, spv::DecorationBinding);
            info.set = compiler.get_decoration(image.id, spv::DecorationDescriptorSet);
            info.stageFlags = stage;
            info.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;

            const auto &type = compiler.get_type(image.type_id);
            info.arraySize = type.array.empty() ? 1 : type.array[0];
            info.multisampled = type.image.ms;
            info.dimension = ImageDimension(type.image.dim);
            info.arrayed = type.image.arrayed;
            info.hasArrayDimension = !type.array.empty();

            m_sampledImages.push_back(info);
        }

        for (const auto &buffer : resources.storage_buffers) {
            StorageBufferInfo info;
            info.binding = compiler.get_decoration(buffer.id, spv::DecorationBinding);
            info.set = compiler.get_decoration(buffer.id, spv::DecorationDescriptorSet);
            // An instantiated block is bound by its shader variable, including
            // ShaderInfo Resources generated blocks. A block without an instance
            // exposes only its declared block name.
            info.name = compiler.get_name(buffer.id);
            if (info.name.empty())
                info.name = compiler.get_name(buffer.base_type_id);
            if (info.name.empty())
                throw std::runtime_error("Storage buffer interface block at set " + std::to_string(info.set) +
                                         ", binding " + std::to_string(info.binding) + " has no declared name");
            info.stageFlags = stage;
            const auto &type = compiler.get_type(buffer.type_id);
            info.arraySize = type.array.empty() ? 1 : type.array[0];
            const auto accessFlags = compiler.get_buffer_block_flags(buffer.id);
            info.readOnly = accessFlags.get(spv::DecorationNonWritable);
            info.writeOnly = accessFlags.get(spv::DecorationNonReadable);
            const auto &block = compiler.get_type(buffer.base_type_id);
            if (block.member_types.size() == 1) {
                const auto &member = compiler.get_type(block.member_types.front());
                if (member.width == 32 && member.vecsize == 1 && member.columns == 1 && member.array.size() == 1 &&
                    member.array_size_literal.front()) {
                    info.scalarArrayFormat = StageIoFormat(member);
                    info.arrayMemberOffset = compiler.type_struct_member_offset(block, 0);
                    info.arrayStride = compiler.type_struct_member_array_stride(block, 0);
                    info.arrayElementCount = member.array.front();
                }
            }
            m_storageBuffers.push_back(info);
        }

        for (const auto &image : resources.storage_images) {
            StorageImageInfo info;
            info.name = image.name;
            info.binding = compiler.get_decoration(image.id, spv::DecorationBinding);
            info.set = compiler.get_decoration(image.id, spv::DecorationDescriptorSet);
            info.stageFlags = stage;
            const auto &type = compiler.get_type(image.type_id);
            info.arraySize = type.array.empty() ? 1 : type.array[0];
            info.readOnly = type.image.access == spv::AccessQualifierReadOnly ||
                            compiler.has_decoration(image.id, spv::DecorationNonWritable);
            info.writeOnly = type.image.access == spv::AccessQualifierWriteOnly ||
                             compiler.has_decoration(image.id, spv::DecorationNonReadable);
            m_storageImages.push_back(info);
        }

        // Process push constants
        for (const auto &pc : resources.push_constant_buffers) {
            PushConstantInfo info;
            info.name = pc.name;
            info.stageFlags = stage;

            const auto &type = compiler.get_type(pc.base_type_id);
            info.size = static_cast<uint32_t>(compiler.get_declared_struct_size(type));
            info.offset = 0; // Push constants start at offset 0

            info.members = ReflectValue(compiler, pc.base_type_id).fields;

            m_pushConstants.push_back(info);
        }

        // Process stage inputs
        for (const auto &input : resources.stage_inputs) {
            ShaderIOVariable var;
            var.name = input.name;
            var.location = compiler.get_decoration(input.id, spv::DecorationLocation);

            const auto &type = compiler.get_type(input.base_type_id);
            var.format = StageIoFormat(type);

            m_inputs.push_back(var);
        }

        // Process stage outputs
        for (const auto &output : resources.stage_outputs) {
            ShaderIOVariable var;
            var.name = output.name;
            var.location = compiler.get_decoration(output.id, spv::DecorationLocation);

            const auto &type = compiler.get_type(output.base_type_id);
            var.format = StageIoFormat(type);

            m_outputs.push_back(var);
        }

        return true;
    } catch (const std::exception &e) {
        INXLOG_ERROR("SPIRV-Cross reflection failed: ", e.what());
        return false;
    }
}

std::vector<VkDescriptorSetLayoutBinding> ShaderReflection::GetDescriptorSetLayoutBindings(uint32_t set) const
{
    std::vector<VkDescriptorSetLayoutBinding> bindings;

    // Add uniform buffer bindings
    for (const auto &ubo : m_uniformBuffers) {
        if (ubo.set == set) {
            VkDescriptorSetLayoutBinding binding{};
            binding.binding = ubo.binding;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            binding.descriptorCount = 1;
            binding.stageFlags = ubo.stageFlags;
            binding.pImmutableSamplers = nullptr;
            bindings.push_back(binding);
        }
    }

    // Add sampled image bindings
    for (const auto &image : m_sampledImages) {
        if (image.set == set) {
            VkDescriptorSetLayoutBinding binding{};
            binding.binding = image.binding;
            binding.descriptorType = image.descriptorType;
            binding.descriptorCount = image.arraySize;
            binding.stageFlags = image.stageFlags;
            binding.pImmutableSamplers = nullptr;
            bindings.push_back(binding);
        }
    }

    for (const auto &buffer : m_storageBuffers) {
        if (buffer.set == set) {
            VkDescriptorSetLayoutBinding binding{};
            binding.binding = buffer.binding;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binding.descriptorCount = buffer.arraySize;
            binding.stageFlags = buffer.stageFlags;
            bindings.push_back(binding);
        }
    }

    for (const auto &image : m_storageImages) {
        if (image.set == set) {
            VkDescriptorSetLayoutBinding binding{};
            binding.binding = image.binding;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            binding.descriptorCount = image.arraySize;
            binding.stageFlags = image.stageFlags;
            bindings.push_back(binding);
        }
    }

    std::sort(bindings.begin(), bindings.end(),
              [](const auto &left, const auto &right) { return left.binding < right.binding; });

    return bindings;
}

std::vector<uint32_t> ShaderReflection::GetUsedDescriptorSets() const
{
    std::vector<uint32_t> sets;

    for (const auto &ubo : m_uniformBuffers) {
        if (std::find(sets.begin(), sets.end(), ubo.set) == sets.end()) {
            sets.push_back(ubo.set);
        }
    }

    for (const auto &image : m_sampledImages) {
        if (std::find(sets.begin(), sets.end(), image.set) == sets.end()) {
            sets.push_back(image.set);
        }
    }

    for (const auto &buffer : m_storageBuffers) {
        if (std::find(sets.begin(), sets.end(), buffer.set) == sets.end())
            sets.push_back(buffer.set);
    }

    for (const auto &image : m_storageImages) {
        if (std::find(sets.begin(), sets.end(), image.set) == sets.end())
            sets.push_back(image.set);
    }

    std::sort(sets.begin(), sets.end());
    return sets;
}

} // namespace infernux
