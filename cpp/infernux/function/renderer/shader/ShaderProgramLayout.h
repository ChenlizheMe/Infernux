#pragma once

#include <function/renderer/rhi/RhiDescriptors.h>

#include <string>
#include <vector>

namespace infernux
{

enum class ShaderValueKind : uint8_t
{
    Scalar,
    Vector,
    Matrix,
    Array,
    Struct,
};

enum class ShaderScalarKind : uint8_t
{
    None,
    Bool,
    Int32,
    UInt32,
    Int64,
    UInt64,
    Float16,
    Float32,
    Float64,
    Int8,
    UInt8,
    Int16,
    UInt16,
};

/// A target's physical value layout. Offsets are relative to the containing
/// struct; an array has one element layout and its actual target stride.
/// This is compiler output, independent of Inspector presentation metadata.
struct ShaderValueLayout
{
    std::string name;
    ShaderValueKind kind = ShaderValueKind::Scalar;
    ShaderScalarKind scalar = ShaderScalarKind::None;
    uint64_t offset = 0;
    uint64_t byteSize = 0;
    uint64_t stride = 0;
    uint32_t rows = 1;
    uint32_t columns = 1;
    uint64_t elementCount = 1;
    bool rowMajor = true;
    std::vector<ShaderValueLayout> fields;

    [[nodiscard]] size_t GetRuntimeMemoryBytes() const noexcept
    {
        size_t bytes = sizeof(*this) + name.capacity() + fields.capacity() * sizeof(ShaderValueLayout);
        for (const auto &field : fields)
            bytes += field.GetRuntimeMemoryBytes() - sizeof(field);
        return bytes;
    }
};

struct ShaderResourceLayout
{
    /// Qualified logical resource path, e.g. inx_scene.texture.
    std::string name;
    uint32_t set = 0;
    rhi::BindingLayoutEntry binding;
    /// Uniform contents, or a structured buffer's element layout. Raw byte
    /// buffers and images carry no value layout.
    ShaderValueLayout value;

    [[nodiscard]] size_t GetRuntimeMemoryBytes() const noexcept
    {
        return sizeof(*this) + name.capacity() + value.GetRuntimeMemoryBytes() - sizeof(value);
    }
};

} // namespace infernux
