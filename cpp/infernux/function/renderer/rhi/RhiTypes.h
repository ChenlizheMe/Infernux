#pragma once

#include <core/types/PixelFormat.h>
#include <cstdint>

namespace infernux::rhi
{

enum class SampleCount : uint8_t
{
    One = 1,
    Two = 2,
    Four = 4,
    Eight = 8,
};

enum class PipelineStage : uint32_t
{
    None = 0,
    Top = 1u << 0,
    DrawIndirect = 1u << 1,
    VertexInput = 1u << 2,
    VertexShader = 1u << 3,
    FragmentShader = 1u << 4,
    EarlyDepth = 1u << 5,
    LateDepth = 1u << 6,
    ColorOutput = 1u << 7,
    ComputeShader = 1u << 8,
    Transfer = 1u << 9,
    Bottom = 1u << 10,
    Host = 1u << 11,
    AllGraphics = 1u << 12,
    AllCommands = 1u << 13,
};

enum class Access : uint32_t
{
    None = 0,
    IndirectRead = 1u << 0,
    IndexRead = 1u << 1,
    VertexRead = 1u << 2,
    UniformRead = 1u << 3,
    ShaderRead = 1u << 4,
    ShaderWrite = 1u << 5,
    ColorRead = 1u << 6,
    ColorWrite = 1u << 7,
    DepthRead = 1u << 8,
    DepthWrite = 1u << 9,
    TransferRead = 1u << 10,
    TransferWrite = 1u << 11,
    HostRead = 1u << 12,
    HostWrite = 1u << 13,
    MemoryRead = 1u << 14,
    MemoryWrite = 1u << 15,
};

enum class TextureLayout : uint8_t
{
    Automatic = 0,
    Undefined,
    General,
    ColorAttachment,
    DepthStencilAttachment,
    DepthStencilReadOnly,
    ShaderReadOnly,
    TransferSource,
    TransferDestination,
    Present,
};

[[nodiscard]] constexpr PipelineStage operator|(PipelineStage lhs, PipelineStage rhs) noexcept
{
    return static_cast<PipelineStage>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

[[nodiscard]] constexpr PipelineStage operator&(PipelineStage lhs, PipelineStage rhs) noexcept
{
    return static_cast<PipelineStage>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}

[[nodiscard]] constexpr Access operator|(Access lhs, Access rhs) noexcept
{
    return static_cast<Access>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

[[nodiscard]] constexpr Access operator&(Access lhs, Access rhs) noexcept
{
    return static_cast<Access>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}

[[nodiscard]] constexpr bool HasAny(PipelineStage value, PipelineStage flags) noexcept
{
    return (value & flags) != PipelineStage::None;
}

[[nodiscard]] constexpr bool HasAny(Access value, Access flags) noexcept
{
    return (value & flags) != Access::None;
}

} // namespace infernux::rhi
