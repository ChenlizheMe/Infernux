#pragma once

#include "ShaderProgramCode.h"

#include <core/serialization/BinaryCodec.h>

#include <limits>

namespace infernux
{
namespace shader_target_archive
{

inline constexpr std::string_view Magic = "INXSTGT1";
inline constexpr uint32_t MaxCount = (std::numeric_limits<uint32_t>::max)();
inline constexpr unsigned ValueDepthLimit = 128;
using Reader = binary_codec::Reader<MaxCount>;
using binary_codec::AppendU32;
using binary_codec::AppendU64;
using binary_codec::AppendUnsigned;

inline void AppendCount(std::string &out, size_t count)
{
    if (count > MaxCount)
        throw std::invalid_argument("shader target archive length exceeds its format limit");
    AppendU32(out, static_cast<uint32_t>(count));
}

inline void AppendBytes(std::string &out, const char *bytes, size_t size)
{
    AppendCount(out, size);
    if (size != 0)
        out.append(bytes, size);
}

inline void AppendString(std::string &out, std::string_view value)
{
    AppendBytes(out, value.data(), value.size());
}

template <typename Enum> void AppendEnum(std::string &out, Enum value)
{
    AppendUnsigned(out, static_cast<uint8_t>(value));
}

template <typename Enum> Enum ReadEnum(Reader &in, Enum maximum)
{
    const auto value = in.ReadUnsigned<uint8_t>();
    if (value > static_cast<uint8_t>(maximum))
        throw std::invalid_argument("shader target archive contains an invalid enum value");
    return static_cast<Enum>(value);
}

inline bool ReadBool(Reader &in)
{
    const auto value = in.ReadUnsigned<uint8_t>();
    if (value > 1)
        throw std::invalid_argument("shader target archive contains an invalid boolean");
    return value != 0;
}

template <typename T, typename Append> void AppendVector(std::string &out, const std::vector<T> &values, Append append)
{
    AppendCount(out, values.size());
    for (const auto &value : values)
        append(out, value);
}

template <typename T, typename Read> std::vector<T> ReadVector(Reader &in, Read read)
{
    const auto count = in.ReadCount(MaxCount);
    std::vector<T> values;
    for (uint32_t i = 0; i < count; ++i)
        values.push_back(read(in));
    return values;
}

inline void CheckValueDepth(unsigned depth)
{
    if (depth >= ValueDepthLimit)
        throw std::invalid_argument("shader target archive value layout nesting is too deep");
}

inline void AppendValue(std::string &out, const ShaderValueLayout &value, unsigned depth = 0)
{
    CheckValueDepth(depth);
    AppendString(out, value.name);
    AppendEnum(out, value.kind);
    AppendEnum(out, value.scalar);
    AppendU64(out, value.offset);
    AppendU64(out, value.byteSize);
    AppendU64(out, value.stride);
    AppendU32(out, value.rows);
    AppendU32(out, value.columns);
    AppendU64(out, value.elementCount);
    AppendUnsigned(out, static_cast<uint8_t>(value.rowMajor));
    AppendVector(out, value.fields,
                 [depth](std::string &bytes, const ShaderValueLayout &field) { AppendValue(bytes, field, depth + 1); });
}

inline ShaderValueLayout ReadValue(Reader &in, unsigned depth = 0)
{
    CheckValueDepth(depth);
    ShaderValueLayout value;
    value.name = in.ReadString();
    value.kind = ReadEnum(in, ShaderValueKind::Struct);
    value.scalar = ReadEnum(in, ShaderScalarKind::UInt16);
    value.offset = in.ReadU64();
    value.byteSize = in.ReadU64();
    value.stride = in.ReadU64();
    value.rows = in.ReadU32();
    value.columns = in.ReadU32();
    value.elementCount = in.ReadU64();
    value.rowMajor = ReadBool(in);
    value.fields = ReadVector<ShaderValueLayout>(in, [depth](Reader &bytes) { return ReadValue(bytes, depth + 1); });
    return value;
}

inline constexpr auto StageMask = rhi::ShaderStage::Vertex | rhi::ShaderStage::Fragment | rhi::ShaderStage::Compute;

inline void AppendBinding(std::string &out, const rhi::BindingLayoutEntry &binding)
{
    AppendU32(out, binding.binding);
    AppendEnum(out, binding.type);
    AppendEnum(out, binding.visibility);
    AppendU32(out, binding.count);
    AppendUnsigned(out, static_cast<uint8_t>(binding.depthRead));
    AppendUnsigned(out, static_cast<uint8_t>(binding.readOnlyStorage));
    AppendEnum(out, binding.viewDimension);
    AppendEnum(out, binding.sampleType);
    AppendUnsigned(out, static_cast<uint8_t>(binding.multisampled));
    AppendEnum(out, binding.storageFormat);
    AppendEnum(out, binding.storageAccess);
    AppendEnum(out, binding.samplerType);
    AppendU64(out, binding.minBufferBindingSize);
}

inline rhi::BindingLayoutEntry ReadBinding(Reader &in)
{
    rhi::BindingLayoutEntry binding;
    binding.binding = in.ReadU32();
    binding.type = ReadEnum(in, rhi::BindingType::CombinedTextureSampler);
    binding.visibility = ReadEnum(in, StageMask);
    binding.count = in.ReadU32();
    binding.depthRead = ReadBool(in);
    binding.readOnlyStorage = ReadBool(in);
    binding.viewDimension = ReadEnum(in, rhi::TextureViewDimension::CubeArray);
    binding.sampleType = ReadEnum(in, rhi::TextureSampleType::Uint);
    binding.multisampled = ReadBool(in);
    binding.storageFormat = ReadEnum(in, static_cast<rhi::PixelFormat>(rhi::kPixelFormatCount - 1));
    binding.storageAccess = ReadEnum(in, rhi::StorageTextureAccess::ReadWrite);
    binding.samplerType = ReadEnum(in, rhi::SamplerBindingType::Comparison);
    binding.minBufferBindingSize = in.ReadU64();
    return binding;
}

inline void AppendResource(std::string &out, const ShaderResourceLayout &resource)
{
    AppendString(out, resource.name);
    AppendU32(out, resource.set);
    AppendBinding(out, resource.binding);
    AppendU64(out, resource.elementStride);
    AppendValue(out, resource.value);
}

inline ShaderResourceLayout ReadResource(Reader &in)
{
    ShaderResourceLayout resource;
    resource.name = in.ReadString();
    resource.set = in.ReadU32();
    resource.binding = ReadBinding(in);
    resource.elementStride = in.ReadU64();
    resource.value = ReadValue(in);
    return resource;
}

inline void AppendVariable(std::string &out, const ShaderStageVariable &variable)
{
    AppendString(out, variable.name);
    AppendU32(out, variable.location);
    AppendEnum(out, variable.scalar);
    AppendU32(out, variable.components);
}

inline ShaderStageVariable ReadVariable(Reader &in)
{
    return {in.ReadString(), in.ReadU32(), ReadEnum(in, ShaderScalarKind::UInt16), in.ReadU32()};
}

inline void AppendEntry(std::string &out, const ShaderEntryPointCode &entry)
{
    AppendString(out, entry.name);
    AppendEnum(out, entry.stage);
    for (auto size : entry.workgroupSize)
        AppendU32(out, size);
    AppendVector(out, entry.bindings, [](std::string &bytes, const std::array<uint32_t, 2> &address) {
        AppendU32(bytes, address[0]);
        AppendU32(bytes, address[1]);
    });
    AppendBytes(out, entry.code.data(), entry.code.size());
    AppendVector(out, entry.inputs, AppendVariable);
    AppendVector(out, entry.outputs, AppendVariable);
    AppendUnsigned(out, static_cast<uint8_t>(entry.sampleRateShading));
    AppendUnsigned(out, static_cast<uint8_t>(entry.rootConstants.has_value()));
    if (entry.rootConstants)
        AppendValue(out, *entry.rootConstants);
}

inline ShaderEntryPointCode ReadEntry(Reader &in)
{
    ShaderEntryPointCode entry;
    entry.name = in.ReadString();
    entry.stage = ReadEnum(in, StageMask);
    for (auto &size : entry.workgroupSize)
        size = in.ReadU32();
    entry.bindings = ReadVector<std::array<uint32_t, 2>>(in, [](Reader &bytes) {
        return std::array<uint32_t, 2>{bytes.ReadU32(), bytes.ReadU32()};
    });
    const auto code = in.ReadString();
    entry.code.assign(code.begin(), code.end());
    entry.inputs = ReadVector<ShaderStageVariable>(in, ReadVariable);
    entry.outputs = ReadVector<ShaderStageVariable>(in, ReadVariable);
    entry.sampleRateShading = ReadBool(in);
    if (ReadBool(in))
        entry.rootConstants = ReadValue(in);
    return entry;
}

} // namespace shader_target_archive

/// Store one compiled target with its concrete layouts and opaque code bytes.
[[nodiscard]] inline std::string EncodeShaderTarget(const ShaderTargetCode &target)
{
    using namespace shader_target_archive;
    std::string out(Magic);
    AppendEnum(out, target.target);
    AppendVector(out, target.entries, AppendEntry);
    AppendVector(out, target.resources, AppendResource);
    AppendString(out, target.reflection);
    return out;
}

/// Read the current archive format. Shader and device validation use the target as usual.
[[nodiscard]] inline ShaderTargetCode DecodeShaderTarget(std::string_view bytes)
{
    using namespace shader_target_archive;
    Reader in(bytes);
    if (!in.ReadPrefix(Magic))
        throw std::invalid_argument("unsupported shader target archive format");
    ShaderTargetCode target;
    target.target = ReadEnum(in, ShaderCodeTarget::Wgsl);
    target.entries = ReadVector<ShaderEntryPointCode>(in, ReadEntry);
    target.resources = ReadVector<ShaderResourceLayout>(in, ReadResource);
    target.reflection = in.ReadString();
    if (!in.AtEnd())
        throw std::invalid_argument("shader target archive contains trailing data");
    return target;
}

} // namespace infernux
