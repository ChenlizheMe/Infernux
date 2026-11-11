#include "SlangLayout.h"

#include <slang.h>

#include <stdexcept>

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
        result.stride = type->getElementStride(SLANG_PARAMETER_CATEGORY_UNIFORM);
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
    } else {
        names.at(static_cast<size_t>(base)) = path;
        if (type->getKind() == Kind::ConstantBuffer && type->getBindingRangeCount() > 1)
            ResourceNames(type->getElementTypeLayout(), base + 1, path, names);
    }
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
} // namespace

std::vector<ShaderResourceLayout> ReflectSlangResources(slang::TypeLayoutReflection *layout)
{
    const auto rangeCount = layout->getBindingRangeCount();
    std::vector<std::string> names(static_cast<size_t>(rangeCount));
    ResourceNames(layout, 0, {}, names);
    std::vector<ShaderResourceLayout> result;
    for (SlangInt index = 0; index < rangeCount; ++index) {
        const auto set = layout->getBindingRangeDescriptorSetIndex(index);
        const auto first = layout->getBindingRangeFirstDescriptorRangeIndex(index);
        const auto count = layout->getBindingRangeDescriptorRangeCount(index);
        auto *type = layout->getBindingRangeLeafTypeLayout(index);
        for (SlangInt range = first; range < first + count; ++range) {
            ShaderResourceLayout resource;
            resource.name = names[index];
            resource.set = static_cast<uint32_t>(layout->getDescriptorSetSpaceOffset(set));
            resource.binding.binding =
                static_cast<uint32_t>(layout->getDescriptorSetDescriptorRangeIndexOffset(set, range));
            resource.binding.count =
                static_cast<uint32_t>(layout->getDescriptorSetDescriptorRangeDescriptorCount(set, range));
            switch (layout->getDescriptorSetDescriptorRangeType(set, range)) {
            case slang::BindingType::ConstantBuffer:
                resource.binding.type = rhi::BindingType::UniformBuffer;
                resource.value = ValueLayout(type->getElementTypeLayout());
                break;
            case slang::BindingType::RawBuffer:
            case slang::BindingType::MutableRawBuffer:
                resource.binding.type = rhi::BindingType::StorageBuffer;
                resource.binding.readOnlyStorage = type->getResourceAccess() == SLANG_RESOURCE_ACCESS_READ;
                if ((type->getResourceShape() & SLANG_RESOURCE_BASE_SHAPE_MASK) == SLANG_STRUCTURED_BUFFER)
                    resource.value = ValueLayout(type->getElementTypeLayout());
                break;
            case slang::BindingType::Texture:
            case slang::BindingType::MutableTexture:
                resource.binding.type = type->getResourceAccess() == SLANG_RESOURCE_ACCESS_READ
                                            ? rhi::BindingType::SampledTexture
                                            : rhi::BindingType::StorageTexture;
                resource.dimension = TextureDimension(type->getResourceShape());
                resource.multisampled = (type->getResourceShape() & SLANG_TEXTURE_MULTISAMPLE_FLAG) != 0;
                resource.binding.depthRead = (type->getResourceShape() & SLANG_TEXTURE_SHADOW_FLAG) != 0;
                break;
            case slang::BindingType::Sampler:
                resource.binding.type = rhi::BindingType::Sampler;
                break;
            default:
                throw std::runtime_error("Shader descriptor has no engine binding representation: " + resource.name);
            }
            result.push_back(std::move(resource));
        }
    }
    return result;
}

} // namespace infernux
