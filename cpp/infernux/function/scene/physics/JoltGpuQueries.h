#pragma once

#include "JoltGpuGeometryStore.h"
#include "JoltGpuScene.h"
#include "PhysicsDeviceQueryOutput.h"
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

enum class JoltGpuQueryReadback
{
    Host,
    Device,
    DeviceWithSpans, // A host consumer needs the per-input offsets of a variable-size device result.
};
/// Native explicit operands use the same immutable geometry arena as the publication.
/// CPU shapes/handles carry author intent until the existing writer accepts capture.
struct JoltGpuQueryGeometry
{
    std::shared_ptr<JoltGpuGeometryStore> store;
    JoltGpuGeometryHandles operands;
    std::vector<JPH::RefConst<JPH::Shape>> shapes;
};

/// Exact publication identity and typed records share their real packet/receipt owners.
struct JoltGpuQueryResult
{
    std::shared_ptr<const JoltGpuScene::Publication> publication;
    uint64_t numericalGeneration = 0;
    // Buffer generation AND exact immutable ranges survive all query/export consumers.
    std::shared_ptr<const JoltGpuGeometryLease> geometry;
    std::vector<JPH::RefConst<JPH::Shape>> explicitShapes;
    std::shared_ptr<rhi::ComputeBuffer> hits;
    // QuerySpan[queryCount] is the byte-zero prefix; the retained backing
    // control also owns private CA and generic-job state. Metadata reads/export use the prefix.
    std::shared_ptr<rhi::ComputeBuffer> spans;
    std::shared_ptr<rhi::ComputeBuffer> summary;
    uint32_t queryCount = 0;
    uint32_t hitCapacity = 0;
    uint32_t outputFormat = JPH::GPU::QueryOutputHit;
    uint32_t recordBytes = sizeof(JPH::GPU::QueryHit);
    std::shared_ptr<rhi::ComputeReadback> summaryReadback;
    std::shared_ptr<rhi::PreparedComputeBatch> tailBatch;
    JPH::GPU::QuerySummary summaryValue{};

    /// ReadSpans materializes metadata only. Device requests never transfer
    /// numerical records; their typed accessors reject an implicit roundtrip.
    [[nodiscard]] const JPH::GPU::QuerySummary &GetSummary();
    [[nodiscard]] uint32_t GetRecordCount();
    [[nodiscard]] const std::vector<JPH::GPU::QueryHit> &ReadHits();
    [[nodiscard]] const std::vector<JPH::GPU::QueryShapeState> &ReadShapes();
    [[nodiscard]] const std::vector<JPH::GPU::QuerySpan> &ReadSpans();

  private:
    friend class JoltGpuQueries;
    void ReadSummary();
    [[nodiscard]] bool NeedsHostSpans() const;
    void Materialize();
    template <class T> const std::vector<T> &ReadTyped(uint32_t format, std::vector<T> &records);
    uint32_t hitReadbackIndex = 0;
    uint32_t spanReadbackIndex = 0;
    JoltGpuQueryReadback readback = JoltGpuQueryReadback::Host;
    bool knownCount = false;
    bool headerComplete = false;
    bool materialized = false;
    bool recordsMaterialized = false;
    bool shapeBoxesOnly = false;
    std::vector<JPH::GPU::QueryHit> hostHits;
    std::vector<JPH::GPU::QueryShapeState> hostShapes;
    std::vector<JPH::GPU::QuerySpan> hostSpans;
};

struct JoltGpuPreparedQuery
{
    std::shared_ptr<JoltGpuQueryResult> result;
    JoltGpuControlWork work;
};

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
    std::shared_ptr<JoltGpuQueryResult> source;
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

/// One compiler artifact, phase registry and source packet builder for all
/// operations and typed families; no private query queue or CPU geometry path.
class JoltGpuQueries final
{
  public:
    JoltGpuQueries(rhi::ComputeHost &host, const ShaderTargetCode &target);
    [[nodiscard]] JoltGpuPreparedQuery
    Prepare(std::shared_ptr<const JoltGpuScene::Publication> publication,
            const std::vector<JPH::GPU::QueryInput> &queries, const std::vector<JPH::GPU::QueryPrimitive> &primitives,
            uint64_t numericalGeneration, JoltGpuQueryReadback readback = JoltGpuQueryReadback::Host,
            uint32_t castIterationQuantum = 32, JoltGpuQueryGeometry geometry = {}) const;

    /// Prepare after the gather metadata receipt, inside the same source's
    /// continuation. Writes only the validated prefix; no CPU numerical upload.
    [[nodiscard]] JoltGpuControlWork PrepareShapeExport(const std::shared_ptr<JoltGpuQueryResult> &source,
                                                        const std::vector<PhysicsDeviceQueryOutput> &bindings) const;

    [[nodiscard]] JoltGpuPreparedParticleExport
    PrepareParticleExport(const std::shared_ptr<JoltGpuQueryResult> &shapes,
                          const std::vector<JoltGpuParticleExportRow> &rows,
                          std::shared_ptr<rhi::ComputeBuffer> previousRecords, uint32_t staticCount, uint64_t revision,
                          uint64_t topologyRevision = 1) const;

  private:
    [[nodiscard]] JoltGpuPreparedQuery PrepareCaptured(std::shared_ptr<const JoltGpuScene::Publication> publication,
                                                       const std::vector<JPH::GPU::QueryInput> &queries,
                                                       const std::vector<JPH::GPU::QueryPrimitive> &primitives,
                                                       uint64_t numericalGeneration, JoltGpuQueryReadback readback,
                                                       uint32_t castIterationQuantum,
                                                       const std::shared_ptr<JoltGpuQueryResult> &result) const;
    struct Execution;
    static JoltGpuControlWork Continue(const std::shared_ptr<Execution> &execution);
    struct Program
    {
        std::shared_ptr<rhi::ComputeKernel> kernel;
        std::vector<uint8_t> ports;
    };
    rhi::ComputeHost m_host;
    std::array<Program, 12> m_programs;
};
} // namespace infernux
