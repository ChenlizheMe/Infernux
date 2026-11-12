#pragma once

#include "ParticleRuntimeApi.h"

#include "ParticleGpuRuntime.h"
#include "ParticleOutputSemantics.h"

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

namespace infernux::particle
{

struct GpuParticleSortProgram
{
    CompiledParticleShader small;
    CompiledParticleShader generate;
    CompiledParticleShader histogram;
    CompiledParticleShader scan;
    CompiledParticleShader scatter;

    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool IsValid() const noexcept;
};

/// Shared, owning representation of the four sort kernels. View render graphs
/// keep only this immutable program while allocating independent workspaces.
struct GpuParticleSortProgramStorage
{
    std::array<CompiledParticleShader, 5> shaders;

    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool Assign(const GpuParticleSortProgram &program);
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool IsValid() const noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API GpuParticleSortProgram View() const noexcept;
};

struct GpuParticleSortShaderSources
{
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API static std::string_view Small() noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API static std::string_view Generate() noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API static std::string_view Histogram() noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API static std::string_view Scan() noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API static std::string_view Scatter() noexcept;
};

struct GpuParticleSorterDesc
{
    uint32_t capacity = 0;
    rhi::BufferHandle visibility;
    rhi::BufferHandle indirectArguments;
    rhi::BufferHandle sourceIndices;
    rhi::BufferHandle dispatchArguments;
    GpuParticleSortProgram program;
};

struct alignas(16) GpuParticleSortConstants
{
    std::array<float, 16> view{};
    uint32_t capacity = 0;
    uint32_t blockCount = 0;
    uint32_t digitShift = 0;
    uint32_t descending = 0;
};

/// Per-view radix-sort workspace. Simulation-owned instance data remains
/// immutable; only a compact uint index stream is reordered for drawing.
class ParticleGpuSorter
{
  public:
    static constexpr uint32_t WorkgroupSize = 256;
    static constexpr uint32_t Radix = 16;
    // Large particle sets use a deterministic 32-bit key: 24 ordered depth
    // bits plus 8 stable particle-id bits. This preserves depth ordering far
    // below visible precision while halving radix passes and key bandwidth.
    // Small sets retain the full depth + particle-id tuple below.
    static constexpr uint32_t PassCount = 8;
    static constexpr uint32_t PackedKeyStride = sizeof(uint32_t);
    // One 256-lane workgroup loads four entries per lane.  Keep this a
    // power-of-two so the in-workgroup bitonic network has no partial stage.
    static constexpr uint32_t SmallSortCapacity = WorkgroupSize * 4u;

    INFERNUX_PARTICLE_RUNTIME_API ParticleGpuSorter();
    INFERNUX_PARTICLE_RUNTIME_API ~ParticleGpuSorter();

    ParticleGpuSorter(const ParticleGpuSorter &) = delete;
    ParticleGpuSorter &operator=(const ParticleGpuSorter &) = delete;
    ParticleGpuSorter(ParticleGpuSorter &&) = delete;
    ParticleGpuSorter &operator=(ParticleGpuSorter &&) = delete;

    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool Create(rhi::Device &device, const GpuParticleSorterDesc &desc);
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
    [[nodiscard]] rhi::BufferHandle VisibilityBuffer() const noexcept
    {
        return m_visibility;
    }
    [[nodiscard]] rhi::BufferHandle IndirectBuffer() const noexcept
    {
        return m_indirectArguments;
    }
    [[nodiscard]] rhi::BufferHandle SourceIndexBuffer() const noexcept
    {
        return m_sourceIndices;
    }
    [[nodiscard]] rhi::BufferHandle DispatchBuffer() const noexcept
    {
        return m_dispatchArguments;
    }
    [[nodiscard]] rhi::BufferHandle SortedIndices() const noexcept
    {
        return m_indices[0];
    }

    [[nodiscard]] bool UsesSmallSort() const noexcept
    {
        return m_capacity <= SmallSortCapacity;
    }
    [[nodiscard]] rhi::BufferHandle KeyBuffer(uint32_t pingPong) const noexcept
    {
        return pingPong < m_keys.size() ? m_keys[pingPong] : rhi::BufferHandle{};
    }
    [[nodiscard]] rhi::BufferHandle IndexBuffer(uint32_t pingPong) const noexcept
    {
        return pingPong < m_indices.size() ? m_indices[pingPong] : rhi::BufferHandle{};
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

    INFERNUX_PARTICLE_RUNTIME_API void RecordSmall(const rhi::ComputeCommandEncoder &encoder,
                                                   const std::array<float, 16> &view, ParticleSortMode mode) const;
    INFERNUX_PARTICLE_RUNTIME_API void RecordGenerate(const rhi::ComputeCommandEncoder &encoder,
                                                      const std::array<float, 16> &view, ParticleSortMode mode) const;
    INFERNUX_PARTICLE_RUNTIME_API void RecordHistogram(const rhi::ComputeCommandEncoder &encoder,
                                                       uint32_t passIndex) const;
    INFERNUX_PARTICLE_RUNTIME_API void RecordScan(const rhi::ComputeCommandEncoder &encoder, uint32_t passIndex) const;
    INFERNUX_PARTICLE_RUNTIME_API void RecordScatter(const rhi::ComputeCommandEncoder &encoder,
                                                     uint32_t passIndex) const;

  private:
    [[nodiscard]] GpuParticleSortConstants Constants(uint32_t passIndex = 0) const noexcept;
    void RecordDirect(const rhi::ComputeCommandEncoder &encoder, const ParticleComputePipeline &pipeline,
                      uint32_t variant, const GpuParticleSortConstants &constants, uint32_t groups) const;
    void RecordIndirect(const rhi::ComputeCommandEncoder &encoder, const ParticleComputePipeline &pipeline,
                        uint32_t variant, const GpuParticleSortConstants &constants) const;

    rhi::Device *m_device = nullptr;
    uint32_t m_capacity = 0;
    uint32_t m_blockCount = 0;
    rhi::BufferHandle m_visibility;
    rhi::BufferHandle m_indirectArguments;
    rhi::BufferHandle m_sourceIndices;
    rhi::BufferHandle m_dispatchArguments;
    std::array<rhi::BufferHandle, 2> m_keys{};
    std::array<rhi::BufferHandle, 2> m_indices{};
    rhi::BufferHandle m_histograms;
    rhi::BufferHandle m_blockOffsets;
    rhi::BufferHandle m_globalOffsets;
    std::unique_ptr<ParticleComputePipeline> m_smallPipeline;
    std::unique_ptr<ParticleComputePipeline> m_generatePipeline;
    std::unique_ptr<ParticleComputePipeline> m_histogramPipeline;
    std::unique_ptr<ParticleComputePipeline> m_scanPipeline;
    std::unique_ptr<ParticleComputePipeline> m_scatterPipeline;
};

static_assert(sizeof(GpuParticleSortConstants) == 80);

} // namespace infernux::particle
