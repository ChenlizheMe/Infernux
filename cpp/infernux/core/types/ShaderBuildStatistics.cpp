#include "ShaderBuildStatistics.h"

#include <atomic>

namespace infernux
{
namespace
{
std::atomic<uint64_t> compilationAttempts{0};
std::atomic<uint64_t> graphicsPipelineCreationAttempts{0};
std::atomic<uint64_t> computePipelineCreationAttempts{0};
} // namespace

void RecordShaderCompilationAttempt() noexcept
{
    compilationAttempts.fetch_add(1, std::memory_order_relaxed);
}

void RecordGraphicsPipelineCreationAttempt() noexcept
{
    graphicsPipelineCreationAttempts.fetch_add(1, std::memory_order_relaxed);
}

void RecordComputePipelineCreationAttempt() noexcept
{
    computePipelineCreationAttempts.fetch_add(1, std::memory_order_relaxed);
}

ShaderBuildStatistics GetShaderBuildStatistics() noexcept
{
    return {compilationAttempts.load(std::memory_order_relaxed),
            graphicsPipelineCreationAttempts.load(std::memory_order_relaxed),
            computePipelineCreationAttempts.load(std::memory_order_relaxed)};
}

} // namespace infernux
