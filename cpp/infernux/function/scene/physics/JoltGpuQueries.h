#pragma once

#include "JoltGpuGeometryStore.h"
#include "JoltGpuScene.h"
#include <array>
#include <cstdint>
#include <exception>
#include <optional>
#include <stdexcept>
#include <vector>

namespace infernux
{
namespace particle
{
struct GpuParticleCollisionDevicePayload;
}
/// Keep a rejected read local to its request throughout the selected GPU work.
[[nodiscard]] JoltGpuControlWork PrepareJoltGpuQueryWork(std::function<JoltGpuControlWork()> prepare,
                                                         std::exception_ptr &error);

struct JoltGpuParticleExportRow
{
    JPH::GPU::UInt4 selection;       // Shape row, previous row or UINT_MAX, generation low/high.
    JPH::GPU::UInt4 identity;        // Collider low/high, GameObject low/high.
    JPH::GPU::Float4 localBoundsMin; // Final native COM-local bounds, w0.
    JPH::GPU::Float4 localBoundsMax;
};
static_assert(sizeof(JoltGpuParticleExportRow) == 64);

struct JoltGpuParticleExportResult
{
    std::shared_ptr<particle::GpuParticleCollisionDevicePayload> payload;
    std::shared_ptr<rhi::ComputeBuffer> status;
    std::shared_ptr<rhi::PreparedComputeBatch> batch;
    /// Validate the completed status and publish frozen outputs for graphics.
    void Validate();

  private:
    friend class JoltGpuQueries;
    void ReadStatus();
    std::shared_ptr<rhi::ComputeReadback> statusReadback;
    JPH::GPU::UInt4 statusValue{};
    bool statusComplete = false;
    bool validated = false;
};

struct JoltGpuPreparedParticleExport
{
    std::shared_ptr<JoltGpuParticleExportResult> result;
    JoltGpuControlWork work;
};

/// Particle collision projection and grid publication from accepted GPU poses.
class JoltGpuQueries final
{
  public:
    JoltGpuQueries(rhi::ComputeHost &host, const ShaderTargetCode &target);
    [[nodiscard]] JoltGpuPreparedParticleExport
    PrepareParticleExport(const std::shared_ptr<const JoltGpuScene::Publication> &publication,
                          const std::vector<JoltGpuParticleExportRow> &rows,
                          std::shared_ptr<rhi::ComputeBuffer> previousRecords, uint32_t staticCount, uint64_t revision,
                          uint64_t topologyRevision = 1) const;

  private:
    struct Program
    {
        std::shared_ptr<rhi::ComputeKernel> kernel;
        std::vector<uint8_t> ports;
    };
    rhi::ComputeHost m_host;
    std::array<Program, 3> m_programs;
};
} // namespace infernux
