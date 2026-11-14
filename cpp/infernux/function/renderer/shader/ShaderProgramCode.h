#pragma once

#include "ShaderDiagnostic.h"
#include "ShaderProgramLayout.h"

#include <array>
#include <optional>

namespace infernux
{

enum class ShaderCodeTarget : uint8_t
{
    SpirV,
    Wgsl,
};

/// One concrete scalar/vector location after the compiler assigns stage IO.
/// Arrays and matrices are expanded by the compiler layout, without parsing code.
struct ShaderStageVariable
{
    std::string name;
    uint32_t location = 0;
    ShaderScalarKind scalar = ShaderScalarKind::None;
    uint32_t components = 1;
};

struct ShaderEntryPointCode
{
    std::string name;
    rhi::ShaderStage stage = rhi::ShaderStage::Compute;
    std::array<uint32_t, 3> workgroupSize{1, 1, 1};
    /// Actual descriptor set/binding pairs retained by this target entry.
    std::vector<std::array<uint32_t, 2>> bindings;
    std::vector<char> code;
    std::vector<ShaderStageVariable> inputs;
    std::vector<ShaderStageVariable> outputs;
    bool sampleRateShading = false;
    /// Small per-execution values. Vulkan uses a push-constant range; the
    /// portable target uses the RHI's reserved uniform binding instead.
    std::optional<ShaderValueLayout> rootConstants;
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
    std::vector<ShaderDiagnostic> diagnostics;
};

} // namespace infernux
