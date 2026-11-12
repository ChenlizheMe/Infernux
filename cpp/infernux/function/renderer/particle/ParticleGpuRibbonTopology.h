#pragma once

#include "ParticleRuntimeApi.h"

#include "ParticleGpuRuntime.h"

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

namespace infernux::particle
{

struct GpuParticleRibbonShaderSources
{
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API static std::string_view Reset() noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API static std::string_view Initialize() noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API static std::string_view Histogram() noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API static std::string_view Scan() noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API static std::string_view Scatter() noexcept;
};

struct GpuParticleRibbonProgram
{
    CompiledParticleShader reset;
    CompiledParticleShader initialize;
    CompiledParticleShader histogram;
    CompiledParticleShader scan;
    CompiledParticleShader scatter;

    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool IsValid() const noexcept;
};

struct GpuParticleRibbonProgramStorage
{
    std::array<CompiledParticleShader, 5> shaders;

    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool Assign(const GpuParticleRibbonProgram &program);
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool IsValid() const noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API GpuParticleRibbonProgram View() const noexcept;
};

struct GpuParticleRibbonDesc
{
    uint32_t capacity = 0;
    rhi::BufferHandle instances;
    rhi::BufferHandle sourceIndices;
    rhi::BufferHandle sourceIndirectArguments;
    rhi::BufferHandle simulationControl;
    GpuParticleRibbonProgram program;
};

struct GpuParticleRibbonConstants
{
    uint32_t capacity = 0;
    uint32_t blockCount = 0;
    uint32_t keyField = 0;
    uint32_t digitShift = 0;
};

/// Builds a camera-independent, GPU-resident ribbon topology. The source
/// particle stream is stably ordered by the complete 96-bit tuple
/// (strip_id, order, source_id) using three full-width 32-bit radix keys.
class ParticleGpuRibbonTopology
{
  public:
    static constexpr uint32_t WorkgroupSize = 256;
    static constexpr uint32_t Radix = 16;
    static constexpr uint32_t PassesPerKey = 8;
    static constexpr uint32_t KeyCount = 3;
    static constexpr uint32_t PassCount = PassesPerKey * KeyCount;

    INFERNUX_PARTICLE_RUNTIME_API ParticleGpuRibbonTopology();
    INFERNUX_PARTICLE_RUNTIME_API ~ParticleGpuRibbonTopology();

    ParticleGpuRibbonTopology(const ParticleGpuRibbonTopology &) = delete;
    ParticleGpuRibbonTopology &operator=(const ParticleGpuRibbonTopology &) = delete;

    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool Create(rhi::Device &device, const GpuParticleRibbonDesc &desc);
    INFERNUX_PARTICLE_RUNTIME_API void Destroy() noexcept;

    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool IsValid() const noexcept;
    [[nodiscard]] uint32_t Capacity() const noexcept
    {
        return m_capacity;
    }
    [[nodiscard]] uint32_t BlockCount() const noexcept
    {
        return m_blockCount;
    }
    [[nodiscard]] rhi::BufferHandle InstanceBuffer() const noexcept
    {
        return m_instances;
    }
    [[nodiscard]] rhi::BufferHandle SourceIndexBuffer() const noexcept
    {
        return m_sourceIndices;
    }
    [[nodiscard]] rhi::BufferHandle SourceIndirectBuffer() const noexcept
    {
        return m_sourceIndirectArguments;
    }
    [[nodiscard]] rhi::BufferHandle SimulationControlBuffer() const noexcept
    {
        return m_simulationControl;
    }
    [[nodiscard]] rhi::BufferHandle SortedIndexBuffer() const noexcept
    {
        return m_indices[0];
    }
    [[nodiscard]] rhi::BufferHandle IndexBuffer(uint32_t pingPong) const noexcept
    {
        return pingPong < m_indices.size() ? m_indices[pingPong] : rhi::BufferHandle{};
    }
    [[nodiscard]] rhi::BufferHandle DrawIndirectBuffer() const noexcept
    {
        return m_drawIndirectArguments;
    }
    [[nodiscard]] rhi::BufferHandle DispatchBuffer() const noexcept
    {
        return m_dispatchArguments;
    }
    [[nodiscard]] rhi::BufferHandle HistogramBuffer() const noexcept
    {
        return m_histograms;
    }
    [[nodiscard]] rhi::BufferHandle BlockOffsetBuffer() const noexcept
    {
        return m_blockOffsets;
    }
    [[nodiscard]] rhi::BufferHandle GlobalOffsetBuffer() const noexcept
    {
        return m_globalOffsets;
    }

    INFERNUX_PARTICLE_RUNTIME_API void RecordReset(const rhi::ComputeCommandEncoder &encoder) const;
    INFERNUX_PARTICLE_RUNTIME_API void RecordInitialize(const rhi::ComputeCommandEncoder &encoder) const;
    INFERNUX_PARTICLE_RUNTIME_API void RecordHistogram(const rhi::ComputeCommandEncoder &encoder,
                                                       uint32_t passIndex) const;
    INFERNUX_PARTICLE_RUNTIME_API void RecordScan(const rhi::ComputeCommandEncoder &encoder, uint32_t passIndex) const;
    INFERNUX_PARTICLE_RUNTIME_API void RecordScatter(const rhi::ComputeCommandEncoder &encoder,
                                                     uint32_t passIndex) const;

  private:
    [[nodiscard]] GpuParticleRibbonConstants Constants(uint32_t passIndex = 0) const noexcept;
    void RecordDirect(const rhi::ComputeCommandEncoder &encoder, const ParticleComputePipeline &pipeline,
                      uint32_t variant, const GpuParticleRibbonConstants &constants, uint32_t groups) const;
    void RecordIndirect(const rhi::ComputeCommandEncoder &encoder, const ParticleComputePipeline &pipeline,
                        uint32_t variant, const GpuParticleRibbonConstants &constants) const;

    rhi::Device *m_device = nullptr;
    uint32_t m_capacity = 0;
    uint32_t m_blockCount = 0;
    rhi::BufferHandle m_instances;
    rhi::BufferHandle m_sourceIndices;
    rhi::BufferHandle m_sourceIndirectArguments;
    rhi::BufferHandle m_simulationControl;
    std::array<rhi::BufferHandle, 2> m_indices{};
    rhi::BufferHandle m_drawIndirectArguments;
    rhi::BufferHandle m_dispatchArguments;
    rhi::BufferHandle m_histograms;
    rhi::BufferHandle m_blockOffsets;
    rhi::BufferHandle m_globalOffsets;
    std::unique_ptr<ParticleComputePipeline> m_resetPipeline;
    std::unique_ptr<ParticleComputePipeline> m_initializePipeline;
    std::unique_ptr<ParticleComputePipeline> m_histogramPipeline;
    std::unique_ptr<ParticleComputePipeline> m_scanPipeline;
    std::unique_ptr<ParticleComputePipeline> m_scatterPipeline;
};

static_assert(sizeof(GpuParticleRibbonConstants) == 16);

} // namespace infernux::particle
