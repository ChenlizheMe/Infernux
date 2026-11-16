#pragma once

#include <core/FoundationApi.h>
#include <cstdint>

namespace infernux
{

/// Process totals of real build attempts, including failed attempts. Pipeline
/// counts cover the engine's Vulkan creation sites, not third-party GUI code.
struct ShaderBuildStatistics
{
    uint64_t compilationAttempts = 0;
    uint64_t graphicsPipelineCreationAttempts = 0;
    uint64_t computePipelineCreationAttempts = 0;
};

INFERNUX_FOUNDATION_API void RecordShaderCompilationAttempt() noexcept;
INFERNUX_FOUNDATION_API void RecordGraphicsPipelineCreationAttempt() noexcept;
INFERNUX_FOUNDATION_API void RecordComputePipelineCreationAttempt() noexcept;
[[nodiscard]] INFERNUX_FOUNDATION_API ShaderBuildStatistics GetShaderBuildStatistics() noexcept;

} // namespace infernux
