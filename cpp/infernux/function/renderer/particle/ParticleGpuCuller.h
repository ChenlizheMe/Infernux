#pragma once

#include "ParticleRuntimeApi.h"

#include "ParticleGpuRuntime.h"

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

namespace infernux::particle
{

struct GpuParticleCullProgram
{
    CompiledParticleShader reset;
    CompiledParticleShader cull;
    CompiledParticleShader finalize;

    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool IsValid() const noexcept;
};

struct GpuParticleCullProgramStorage
{
    std::array<CompiledParticleShader, 3> shaders;

    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool Assign(const GpuParticleCullProgram &program);
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool IsValid() const noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API GpuParticleCullProgram View() const noexcept;
};

enum class GpuParticleCullMode : uint32_t
{
    Instances = 0,
    RibbonSegments = 1,
};

struct GpuParticleCullShaderSources
{
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API static std::string_view Reset() noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API static std::string_view Cull() noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API static std::string_view Finalize() noexcept;
};

struct GpuParticleCullerDesc
{
    uint32_t capacity = 0;
    uint32_t vertexCount = 0;
    rhi::BufferHandle visibility;
    // Ribbon connectivity still needs authored ribbon flags. Ordinary
    // instance culling never dereferences this full render payload.
    rhi::BufferHandle ribbonInstances;
    rhi::BufferHandle sourceIndirectArguments;
    rhi::BufferHandle sourceIndices;
    rhi::BufferHandle bounds;
    rhi::BufferHandle simulationControl;
    GpuParticleCullMode mode = GpuParticleCullMode::Instances;
    GpuParticleCullProgram program;
};

struct alignas(16) GpuParticleCullConstants
{
    std::array<float, 24> frustumPlanes{};
    uint32_t capacity = 0;
    uint32_t vertexCount = 0;
    GpuParticleCullMode mode = GpuParticleCullMode::Instances;
    uint32_t viewEnabled = 1;
};

struct GpuParticleCullDispatchState
{
    uint32_t groupCountX = 0;
    uint32_t groupCountY = 1;
    uint32_t groupCountZ = 1;
    uint32_t sourceCount = 0;
    uint32_t flags = 0;
};

/// Per-view visibility workspace. It never mutates simulation-owned instance
/// data and produces compact draw indices plus draw/sort indirect arguments.
class ParticleGpuCuller
{
  public:
    static constexpr uint32_t WorkgroupSize = 256;
    static constexpr uint32_t PlaneCount = 6;

    INFERNUX_PARTICLE_RUNTIME_API ParticleGpuCuller();
    INFERNUX_PARTICLE_RUNTIME_API ~ParticleGpuCuller();

    ParticleGpuCuller(const ParticleGpuCuller &) = delete;
    ParticleGpuCuller &operator=(const ParticleGpuCuller &) = delete;
    ParticleGpuCuller(ParticleGpuCuller &&) = delete;
    ParticleGpuCuller &operator=(ParticleGpuCuller &&) = delete;

    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool Create(rhi::Device &device, const GpuParticleCullerDesc &desc);
    INFERNUX_PARTICLE_RUNTIME_API void Destroy() noexcept;

    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool IsValid() const noexcept;
    [[nodiscard]] uint32_t Capacity() const noexcept
    {
        return m_capacity;
    }
    [[nodiscard]] uint32_t VertexCount() const noexcept
    {
        return m_vertexCount;
    }
    [[nodiscard]] rhi::BufferHandle VisibilityBuffer() const noexcept
    {
        return m_visibility;
    }
    [[nodiscard]] rhi::BufferHandle SourceIndirectBuffer() const noexcept
    {
        return m_sourceIndirectArguments;
    }
    [[nodiscard]] rhi::BufferHandle SourceIndexBuffer() const noexcept
    {
        return m_sourceIndices;
    }
    [[nodiscard]] GpuParticleCullMode Mode() const noexcept
    {
        return m_mode;
    }
    [[nodiscard]] rhi::BufferHandle BoundsBuffer() const noexcept
    {
        return m_bounds;
    }
    [[nodiscard]] rhi::BufferHandle SimulationControlBuffer() const noexcept
    {
        return m_simulationControl;
    }
    [[nodiscard]] rhi::BufferHandle VisibleIndexBuffer() const noexcept
    {
        return m_visibleIndices;
    }
    [[nodiscard]] rhi::BufferHandle DrawIndirectBuffer() const noexcept
    {
        return m_drawIndirectArguments;
    }
    [[nodiscard]] rhi::BufferHandle SortDispatchBuffer() const noexcept
    {
        return m_sortDispatchArguments;
    }
    INFERNUX_PARTICLE_RUNTIME_API void RecordReset(const rhi::ComputeCommandEncoder &encoder,
                                                   const std::array<float, PlaneCount * 4> &frustumPlanes,
                                                   bool viewEnabled = true) const;
    INFERNUX_PARTICLE_RUNTIME_API void RecordCull(const rhi::ComputeCommandEncoder &encoder,
                                                  const std::array<float, PlaneCount * 4> &frustumPlanes) const;
    INFERNUX_PARTICLE_RUNTIME_API void RecordFinalize(const rhi::ComputeCommandEncoder &encoder) const;

  private:
    void Record(const rhi::ComputeCommandEncoder &encoder, const ParticleComputePipeline &pipeline,
                const GpuParticleCullConstants &constants, uint32_t groups) const;

    rhi::Device *m_device = nullptr;
    uint32_t m_capacity = 0;
    uint32_t m_vertexCount = 0;
    rhi::BufferHandle m_visibility;
    rhi::BufferHandle m_ribbonInstances;
    rhi::BufferHandle m_sourceIndirectArguments;
    rhi::BufferHandle m_sourceIndices;
    rhi::BufferHandle m_bounds;
    rhi::BufferHandle m_simulationControl;
    GpuParticleCullMode m_mode = GpuParticleCullMode::Instances;
    rhi::BufferHandle m_visibleIndices;
    rhi::BufferHandle m_drawIndirectArguments;
    rhi::BufferHandle m_sortDispatchArguments;
    std::unique_ptr<ParticleComputePipeline> m_resetPipeline;
    std::unique_ptr<ParticleComputePipeline> m_cullPipeline;
    std::unique_ptr<ParticleComputePipeline> m_finalizePipeline;
};

static_assert(sizeof(GpuParticleCullConstants) == 112);
static_assert(sizeof(GpuParticleCullDispatchState) == 20);

} // namespace infernux::particle
