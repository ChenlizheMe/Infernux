#include "SlangLayout.h"
#include <function/renderer/shader/ShaderProgramCode.h>

#include <slang.h>

#include <algorithm>
#include <stdexcept>
#include <string_view>
#include <tuple>

namespace infernux
{
namespace
{
using Kind = slang::TypeReflection::Kind;

ShaderScalarKind ScalarKind(slang::TypeReflection::ScalarType scalar)
{
    using Scalar = slang::TypeReflection;
    switch (scalar) {
    case Scalar::Bool:
        return ShaderScalarKind::Bool;
    case Scalar::Int32:
        return ShaderScalarKind::Int32;
    case Scalar::UInt32:
        return ShaderScalarKind::UInt32;
    case Scalar::Int64:
        return ShaderScalarKind::Int64;
    case Scalar::UInt64:
        return ShaderScalarKind::UInt64;
    case Scalar::Float16:
        return ShaderScalarKind::Float16;
    case Scalar::Float32:
        return ShaderScalarKind::Float32;
    case Scalar::Float64:
        return ShaderScalarKind::Float64;
    case Scalar::Int8:
        return ShaderScalarKind::Int8;
    case Scalar::UInt8:
        return ShaderScalarKind::UInt8;
    case Scalar::Int16:
        return ShaderScalarKind::Int16;
    case Scalar::UInt16:
        return ShaderScalarKind::UInt16;
    default:
        throw std::runtime_error("Shader value has no engine scalar representation");
    }
}

ShaderValueLayout ValueLayout(slang::TypeLayoutReflection *type, std::string name = {})
{
    ShaderValueLayout result;
    result.name = std::move(name);
    result.byteSize = type->getSize();
    switch (type->getKind()) {
    case Kind::Struct:
        result.kind = ShaderValueKind::Struct;
        for (unsigned index = 0; index < type->getFieldCount(); ++index) {
            auto *field = type->getFieldByIndex(index);
            auto *fieldType = field->getTypeLayout();
            if (fieldType->getSize() == 0)
                continue; // Opaque resources are separate descriptor bindings.
            auto value = ValueLayout(fieldType, field->getName());
            value.offset = field->getOffset();
            result.fields.push_back(std::move(value));
        }
        break;
    case Kind::Array:
        result.kind = ShaderValueKind::Array;
        result.elementCount = type->getElementCount();
        result.stride = type->getElementStride(SLANG_PARAMETER_CATEGORY_UNIFORM);
        result.fields.push_back(ValueLayout(type->getElementTypeLayout()));
        break;
    case Kind::Matrix:
        result.kind = ShaderValueKind::Matrix;
        result.rows = type->getRowCount();
        result.columns = type->getColumnCount();
        result.scalar = ScalarKind(type->getScalarType());
        result.rowMajor = type->getMatrixLayoutMode() == SLANG_MATRIX_LAYOUT_ROW_MAJOR;
        // Matrix getElementStride() is not defined by Slang. Its full storage
        // stride includes the padding of every row/column in the major axis.
        result.stride =
            type->getStride(SLANG_PARAMETER_CATEGORY_UNIFORM) / (result.rowMajor ? result.rows : result.columns);
        break;
    case Kind::Vector:
        result.kind = ShaderValueKind::Vector;
        result.columns = type->getColumnCount();
        result.scalar = ScalarKind(type->getScalarType());
        break;
    case Kind::Scalar:
        result.scalar = ScalarKind(type->getScalarType());
        break;
    default:
        throw std::runtime_error("Shader value layout is not a concrete scalar, vector, matrix, array or struct");
    }
    return result;
}

// Binding ranges preserve logical fields while descriptor ranges describe
// physical slots. Use Slang's range offsets rather than reconstructing a
// backend's register allocation from source text or declaration order.
void ResourceNames(slang::TypeLayoutReflection *type, SlangInt base, const std::string &path,
                   std::vector<std::string> &names)
{
    if (type->getBindingRangeCount() == 0)
        return;
    if (type->getKind() == Kind::Struct) {
        for (unsigned index = 0; index < type->getFieldCount(); ++index) {
            auto *field = type->getFieldByIndex(index);
            ResourceNames(field->getTypeLayout(), base + type->getFieldBindingRangeOffset(index),
                          path.empty() ? field->getName() : path + "." + field->getName(), names);
        }
    } else if (type->getKind() == Kind::Array) {
        ResourceNames(type->getElementTypeLayout(), base, path, names);
    } else
        names.at(static_cast<size_t>(base)) = path;
}

rhi::TextureViewDimension TextureDimension(SlangResourceShape shape)
{
    switch (shape & (SLANG_RESOURCE_BASE_SHAPE_MASK | SLANG_TEXTURE_ARRAY_FLAG)) {
    case SLANG_TEXTURE_1D:
        return rhi::TextureViewDimension::Texture1D;
    case SLANG_TEXTURE_1D_ARRAY:
        return rhi::TextureViewDimension::Texture1DArray;
    case SLANG_TEXTURE_2D:
        return rhi::TextureViewDimension::Texture2D;
    case SLANG_TEXTURE_2D_ARRAY:
        return rhi::TextureViewDimension::Texture2DArray;
    case SLANG_TEXTURE_3D:
        return rhi::TextureViewDimension::Texture3D;
    case SLANG_TEXTURE_CUBE:
        return rhi::TextureViewDimension::Cube;
    case SLANG_TEXTURE_CUBE_ARRAY:
        return rhi::TextureViewDimension::CubeArray;
    default:
        throw std::runtime_error("Shader image shape has no engine texture view representation");
    }
}

rhi::PixelFormat ImageFormat(SlangImageFormat format)
{
    using Format = rhi::PixelFormat;
    switch (format) {
    case SLANG_IMAGE_FORMAT_rgba32f:
        return Format::RGBA32SFloat;
    case SLANG_IMAGE_FORMAT_rgba16f:
        return Format::RGBA16SFloat;
    case SLANG_IMAGE_FORMAT_rg16f:
        return Format::RG16SFloat;
    case SLANG_IMAGE_FORMAT_r32f:
        return Format::R32SFloat;
    case SLANG_IMAGE_FORMAT_r16f:
        return Format::R16SFloat;
    case SLANG_IMAGE_FORMAT_rgba16:
        return Format::RGBA16UNorm;
    case SLANG_IMAGE_FORMAT_rgb10_a2:
        return Format::RGB10A2UNorm;
    case SLANG_IMAGE_FORMAT_rgba8:
        return Format::RGBA8UNorm;
    case SLANG_IMAGE_FORMAT_rg8:
        return Format::RG8UNorm;
    case SLANG_IMAGE_FORMAT_r8:
        return Format::R8UNorm;
    case SLANG_IMAGE_FORMAT_rgba32ui:
        return Format::RGBA32UInt;
    case SLANG_IMAGE_FORMAT_rg32ui:
        return Format::RG32UInt;
    case SLANG_IMAGE_FORMAT_bgra8:
        return Format::BGRA8UNorm;
    case SLANG_IMAGE_FORMAT_unknown:
        return Format::Undefined;
    default:
        throw std::runtime_error("Shader storage image format has no engine pixel format representation");
    }
}
struct BindingOffset
{
    uint32_t set = 0;
    uint32_t binding = 0;

    [[nodiscard]] BindingOffset At(slang::VariableLayoutReflection *variable) const
    {
        return {set + static_cast<uint32_t>(variable->getOffset(SLANG_PARAMETER_CATEGORY_SUB_ELEMENT_REGISTER_SPACE) +
                                            variable->getBindingSpace(SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT)),
                binding + static_cast<uint32_t>(variable->getOffset(SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT))};
    }
};

void ReflectValue(slang::TypeLayoutReflection *layout, BindingOffset offset, const std::string &path,
                  std::vector<ShaderResourceLayout> &result);

void ReflectContainer(slang::TypeLayoutReflection *container, BindingOffset offset, const std::string &path,
                      std::vector<ShaderResourceLayout> &result)
{
    if (container->getKind() == Kind::ParameterBlock)
        offset.binding = 0; // A parameter block owns its descriptor set.
    auto *element = container->getElementVarLayout();
    auto *value = element->getTypeLayout();
    if (value->getSize(SLANG_PARAMETER_CATEGORY_UNIFORM) != 0) {
        const auto location = offset.At(container->getContainerVarLayout());
        ShaderResourceLayout resource;
        resource.name = path.empty() ? "$global" : path;
        resource.set = location.set;
        resource.binding.binding = location.binding;
        resource.binding.type = rhi::BindingType::UniformBuffer;
        resource.value = ValueLayout(value);
        resource.binding.minBufferBindingSize = resource.value.byteSize;
        result.push_back(std::move(resource));
    }
    // The element variable holds offsets relative to the container. Its type
    // is unadjusted; getElementTypeLayout() already folds these offsets in.
    ReflectValue(value, offset.At(element), path, result);
}

void ReflectValue(slang::TypeLayoutReflection *layout, BindingOffset offset, const std::string &path,
                  std::vector<ShaderResourceLayout> &result)
{
    if (layout->getKind() == Kind::ConstantBuffer || layout->getKind() == Kind::ParameterBlock) {
        ReflectContainer(layout, offset, path, result);
        return;
    }
    const auto rangeCount = layout->getBindingRangeCount();
    std::vector<std::string> names(static_cast<size_t>(rangeCount));
    ResourceNames(layout, 0, path, names);
    for (SlangInt index = 0; index < rangeCount; ++index) {
        const auto bindingType = layout->getBindingRangeType(index);
        if (bindingType == slang::BindingType::ConstantBuffer || bindingType == slang::BindingType::ParameterBlock)
            continue; // Container ranges are traversed once below, including their implicit uniform data.
        const auto set = layout->getBindingRangeDescriptorSetIndex(index);
        const auto first = layout->getBindingRangeFirstDescriptorRangeIndex(index);
        const auto count = layout->getBindingRangeDescriptorRangeCount(index);
        auto *type = layout->getBindingRangeLeafTypeLayout(index);
        for (SlangInt range = first; range < first + count; ++range) {
            ShaderResourceLayout resource;
            resource.name = names[index];
            resource.set = offset.set + static_cast<uint32_t>(layout->getDescriptorSetSpaceOffset(set));
            resource.binding.binding =
                offset.binding + static_cast<uint32_t>(layout->getDescriptorSetDescriptorRangeIndexOffset(set, range));
            resource.binding.count =
                static_cast<uint32_t>(layout->getDescriptorSetDescriptorRangeDescriptorCount(set, range));
            switch (layout->getDescriptorSetDescriptorRangeType(set, range)) {
            case slang::BindingType::RawBuffer:
            case slang::BindingType::MutableRawBuffer:
                resource.binding.type = rhi::BindingType::StorageBuffer;
                resource.binding.readOnlyStorage = type->getResourceAccess() == SLANG_RESOURCE_ACCESS_READ;
                if ((type->getResourceShape() & SLANG_RESOURCE_BASE_SHAPE_MASK) == SLANG_STRUCTURED_BUFFER) {
                    resource.value = ValueLayout(type->getElementTypeLayout());
                    resource.elementStride = type->getElementTypeLayout()->getStride(SLANG_PARAMETER_CATEGORY_UNIFORM);
                }
                break;
            case slang::BindingType::Texture:
            case slang::BindingType::MutableTexture:
                resource.binding.type = type->getResourceAccess() == SLANG_RESOURCE_ACCESS_READ
                                            ? rhi::BindingType::SampledTexture
                                            : rhi::BindingType::StorageTexture;
                resource.binding.viewDimension = TextureDimension(type->getResourceShape());
                resource.binding.multisampled = (type->getResourceShape() & SLANG_TEXTURE_MULTISAMPLE_FLAG) != 0;
                resource.binding.depthRead = (type->getResourceShape() & SLANG_TEXTURE_SHADOW_FLAG) != 0;
                if (auto *value = type->getResourceResultType()) {
                    if (value->getScalarType() == slang::TypeReflection::Int32)
                        resource.binding.sampleType = rhi::TextureSampleType::Sint;
                    else if (value->getScalarType() == slang::TypeReflection::UInt32)
                        resource.binding.sampleType = rhi::TextureSampleType::Uint;
                }
                if (resource.binding.type == rhi::BindingType::StorageTexture) {
                    resource.binding.storageFormat = ImageFormat(layout->getBindingRangeImageFormat(index));
                    resource.binding.storageAccess = type->getResourceAccess() == SLANG_RESOURCE_ACCESS_WRITE
                                                         ? rhi::StorageTextureAccess::WriteOnly
                                                         : rhi::StorageTextureAccess::ReadWrite;
                }
                break;
            case slang::BindingType::Sampler:
                resource.binding.type = rhi::BindingType::Sampler;
                if (std::string_view(type->getName()) == "SamplerComparisonState")
                    resource.binding.samplerType = rhi::SamplerBindingType::Comparison;
                break;
            default:
                throw std::runtime_error("Shader descriptor has no engine binding representation: " + resource.name);
            }
            result.push_back(std::move(resource));
        }
    }
    for (SlangInt index = 0; index < layout->getSubObjectRangeCount(); ++index) {
        const auto range = layout->getSubObjectRangeBindingRangeIndex(index);
        const auto type = layout->getBindingRangeType(range);
        if (type == slang::BindingType::ConstantBuffer || type == slang::BindingType::ParameterBlock)
            ReflectContainer(layout->getBindingRangeLeafTypeLayout(range),
                             offset.At(layout->getSubObjectRangeOffset(index)), names[range], result);
    }
}
void StageType(slang::TypeLayoutReflection *type, SlangParameterCategory category, uint32_t location,
               const std::string &name, std::vector<ShaderStageVariable> &result);

void StageVariable(slang::VariableLayoutReflection *variable, SlangParameterCategory category, uint32_t base,
                   const std::string &parent, std::vector<ShaderStageVariable> &result)
{
    auto *type = variable->getTypeLayout();
    if (type->getSize(category) == 0)
        return; // System values have no vertex-buffer/varying location.
    const auto *field = variable->getName();
    const auto name = field ? (parent.empty() ? field : parent + "." + field) : parent;
    StageType(type, category, base + static_cast<uint32_t>(variable->getOffset(category)), name, result);
}

void StageType(slang::TypeLayoutReflection *type, SlangParameterCategory category, uint32_t location,
               const std::string &name, std::vector<ShaderStageVariable> &result)
{
    switch (type->getKind()) {
    case Kind::Struct:
        for (unsigned index = 0; index < type->getFieldCount(); ++index)
            StageVariable(type->getFieldByIndex(index), category, location, name, result);
        break;
    case Kind::Array: {
        const auto stride = static_cast<uint32_t>(type->getElementStride(category));
        for (size_t index = 0; index < type->getElementCount(); ++index)
            StageType(type->getElementTypeLayout(), category, location + static_cast<uint32_t>(index) * stride,
                      name + "[" + std::to_string(index) + "]", result);
        break;
    }
    case Kind::Matrix: {
        const auto count = static_cast<uint32_t>(type->getSize(category));
        const auto width = type->getRowCount() * type->getColumnCount() / count;
        for (uint32_t index = 0; index < count; ++index)
            result.push_back(
                {name + "[" + std::to_string(index) + "]", location + index, ScalarKind(type->getScalarType()), width});
        break;
    }
    case Kind::Scalar:
    case Kind::Vector:
        result.push_back({name, location, ScalarKind(type->getScalarType()),
                          type->getKind() == Kind::Scalar ? 1u : type->getColumnCount()});
        break;
    default:
        throw std::runtime_error("Shader stage IO has no concrete value layout: " + name);
    }
}
} // namespace

std::vector<ShaderResourceLayout> ReflectSlangResources(slang::TypeLayoutReflection *layout)
{
    std::vector<ShaderResourceLayout> result;
    ReflectValue(layout, {}, {}, result);
    std::sort(result.begin(), result.end(), [](const auto &left, const auto &right) {
        return std::tie(left.set, left.binding.binding) < std::tie(right.set, right.binding.binding);
    });
    return result;
}

void ReflectSlangStageIO(slang::EntryPointReflection *layout, ShaderEntryPointCode &entry)
{
    for (unsigned index = 0; index < layout->getParameterCount(); ++index) {
        auto *parameter = layout->getParameterByIndex(index);
        StageVariable(parameter, SLANG_PARAMETER_CATEGORY_VARYING_INPUT, 0, {}, entry.inputs);
        StageVariable(parameter, SLANG_PARAMETER_CATEGORY_VARYING_OUTPUT, 0, {}, entry.outputs);
    }
    StageVariable(layout->getResultVarLayout(), SLANG_PARAMETER_CATEGORY_VARYING_OUTPUT, 0, {}, entry.outputs);
    entry.sampleRateShading = layout->getStage() == SLANG_STAGE_FRAGMENT && layout->usesAnySampleRateInput();
}

} // namespace infernux
