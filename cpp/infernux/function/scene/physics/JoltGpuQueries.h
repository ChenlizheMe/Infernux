#pragma once

#include "JoltGpuScene.h"
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace infernux
{
namespace particle
{
struct GpuParticleCollisionDevicePayload;
}
class JoltGpuQueryRequestError : public std::invalid_argument
{
  public:
    using std::invalid_argument::invalid_argument;
};

enum class JoltGpuQueryReadback
{
    Host,
    Device
};
enum class JoltGpuQueryExportField : uint32_t
{
    MotionPosition = JPH::GPU::QueryExportMotionPosition,
    MotionRotation = JPH::GPU::QueryExportMotionRotation,
    MotionCenterOfMass = JPH::GPU::QueryExportMotionCenterOfMass,
    MotionLinearVelocity = JPH::GPU::QueryExportMotionLinearVelocity,
    MotionAngularVelocity = JPH::GPU::QueryExportMotionAngularVelocity,
    MotionInverseMass = JPH::GPU::QueryExportMotionInverseMass,
    MotionInverseInertia = JPH::GPU::QueryExportMotionInverseInertia,
    ShapeBodyIndex = JPH::GPU::QueryExportShapeBodyIndex,
    ShapeCenter = JPH::GPU::QueryExportShapeCenter,
    ShapeHalfExtents = JPH::GPU::QueryExportShapeHalfExtents,
    ShapeRotation = JPH::GPU::QueryExportShapeRotation,
    ShapeFriction = JPH::GPU::QueryExportShapeFriction,
    ShapeBounciness = JPH::GPU::QueryExportShapeBounciness,
};

struct JoltGpuQueryExportBinding
{
    JoltGpuQueryExportField field;
    std::shared_ptr<rhi::ComputeBuffer> buffer;
    uint64_t byteOffset = 0;
};

/// Exact publication identity and one typed GPU record stream. Metadata and
/// numerical materialization share the same completed packet/receipt owners.
struct JoltGpuQueryResult
{
    std::shared_ptr<const JoltGpuScene::Publication> publication;
    uint64_t numericalGeneration = 0;
    std::shared_ptr<rhi::ComputeBuffer> hits;
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
    [[nodiscard]] const std::vector<JPH::GPU::QueryHit> &ReadHits();
    [[nodiscard]] const std::vector<JPH::GPU::QueryShapeState> &ReadShapes();
    [[nodiscard]] const std::vector<JPH::GPU::QueryMotionState> &ReadMotion();
    [[nodiscard]] const std::vector<JPH::GPU::QuerySpan> &ReadSpans();

  private:
    friend class JoltGpuQueries;
    void ReadSummary();
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
    std::vector<JPH::GPU::QueryMotionState> hostMotion;
    std::vector<JPH::GPU::QuerySpan> hostSpans;
};

struct JoltGpuPreparedQuery
{
    std::shared_ptr<JoltGpuQueryResult> result;
    JoltGpuControlWork work;
};

struct JoltGpuParticleExportRow
{
    JPH::GPU::UInt4 selection; // Shape row, previous row or UINT_MAX, then zero.
    JPH::GPU::UInt4 identity;  // Collider low/high, GameObject low/high.
};
static_assert(sizeof(JoltGpuParticleExportRow) == 32);

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
    [[nodiscard]] JoltGpuPreparedQuery Prepare(std::shared_ptr<const JoltGpuScene::Publication> publication,
                                               const std::vector<JPH::GPU::QueryInput> &queries,
                                               const std::vector<JPH::GPU::QueryPrimitive> &primitives,
                                               uint64_t numericalGeneration,
                                               JoltGpuQueryReadback readback = JoltGpuQueryReadback::Host,
                                               uint32_t castIterationQuantum = 32) const;

    /// Prepare after the gather metadata receipt, inside the same source's
    /// continuation. Writes only the validated prefix; no CPU numerical upload.
    [[nodiscard]] JoltGpuControlWork PrepareExport(const std::vector<std::shared_ptr<JoltGpuQueryResult>> &results,
                                                   const std::vector<JoltGpuQueryExportBinding> &bindings) const;

    [[nodiscard]] JoltGpuPreparedParticleExport
    PrepareParticleExport(const std::shared_ptr<JoltGpuQueryResult> &shapes,
                          const std::vector<JoltGpuParticleExportRow> &rows,
                          std::shared_ptr<rhi::ComputeBuffer> previousRecords, uint32_t staticCount, uint64_t revision,
                          uint64_t topologyRevision = 1) const;

  private:
    struct Execution;
    static JoltGpuControlWork Continue(const std::shared_ptr<Execution> &execution);
    struct Program
    {
        std::shared_ptr<rhi::ComputeKernel> kernel;
        std::vector<uint8_t> ports;
    };
    rhi::ComputeHost m_host;
    std::array<Program, 8> m_programs;
};
} // namespace infernux
