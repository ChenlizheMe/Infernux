#pragma once

#include "ShaderProgramLayout.h"

#include <array>

namespace infernux
{

enum class ShaderCodeTarget : uint8_t
{
    SpirV,
    Wgsl,
};

struct ShaderEntryPointCode
{
    std::string name;
    rhi::ShaderStage stage = rhi::ShaderStage::Compute;
    std::array<uint32_t, 3> workgroupSize{1, 1, 1};
    /// Actual descriptor set/binding pairs retained by this target entry.
    std::vector<std::array<uint32_t, 2>> bindings;
    std::vector<char> code;
};

struct ShaderTargetCode
{
    ShaderCodeTarget target = ShaderCodeTarget::SpirV;
    std::vector<ShaderEntryPointCode> entries;
    std::vector<ShaderResourceLayout> resources;
    /// Full target reflection retained for tools and compiler diagnostics.
    std::string reflection;
};

struct ShaderCompilation
{
    std::vector<ShaderTargetCode> targets;
    std::string diagnostics;
};

} // namespace infernux
