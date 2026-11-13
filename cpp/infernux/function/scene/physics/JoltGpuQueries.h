#pragma once

#include "JoltGpuScene.h"
#include <stdexcept>

#include <array>
#include <cstdint>
#include <vector>

namespace infernux
{

/// Invalid request data/capacity; no query packet has been submitted. This
/// does not invalidate a completed world or its accepted authoring prefix.
class JoltGpuQueryRequestError : public std::invalid_argument
{
  public:
    using std::invalid_argument::invalid_argument;
};

/// One logical request's immutable scene identity and device outputs. Every
/// continuation retains this exact publication; decoding must never substitute
/// the scene's current publication after slots or members have changed.
struct JoltGpuQueryResult
{
    std::shared_ptr<const JoltGpuScene::Publication> publication;
    uint64_t numericalGeneration = 0;
    std::shared_ptr<rhi::ComputeBuffer> hits;
    std::shared_ptr<rhi::ComputeBuffer> spans;
    std::shared_ptr<rhi::ComputeBuffer> summary;
    uint32_t queryCount = 0;
    uint32_t hitCapacity = 0;

    /// Header and output transfers retain their actual source packets.
    /// Known-count collectors fuse summary and outputs into the initial packet.
    /// Variable-count collectors read the 16-byte summary first and prepare a
    /// valid-prefix tail only after its receipt; no CPU candidate download.
    std::shared_ptr<rhi::ComputeReadback> summaryReadback;
    std::shared_ptr<rhi::PreparedComputeBatch> tailBatch;
    JPH::GPU::QuerySummary summaryValue{};

    /// Call only after the whole ControlWork completes. Materializes final
    /// output records once and keeps them available for container adapters.
    [[nodiscard]] const std::vector<JPH::GPU::QueryHit> &ReadHits();
    [[nodiscard]] const std::vector<JPH::GPU::QuerySpan> &ReadSpans();

  private:
    friend class JoltGpuQueries;
    void ReadSummary();
    void Materialize();
    uint32_t hitReadbackIndex = 0;
    uint32_t spanReadbackIndex = 0;
    bool knownCount = false;
    bool headerComplete = false;
    bool materialized = false;
    std::vector<JPH::GPU::QueryHit> hostHits;
    std::vector<JPH::GPU::QuerySpan> hostSpans;
};

struct JoltGpuPreparedQuery
{
    std::shared_ptr<JoltGpuQueryResult> result;
    JoltGpuControlWork work;
};

/// Shared operation/collector pipeline compiled from the common query artifact.
/// Geometry, filtering, counting and compaction run on the device. Host work
/// prepares packets, reads status and wraps only the compact result prefix.
class JoltGpuQueries final
{
  public:
    JoltGpuQueries(rhi::ComputeHost &host, const ShaderTargetCode &target);

    /// Prepare only; the scene's existing source admits every returned packet
    /// and continuation, retaining its budgets, cancellation and real receipts.
    /// A null publication is valid for explicit immutable geometry requests.
    [[nodiscard]] JoltGpuPreparedQuery Prepare(std::shared_ptr<const JoltGpuScene::Publication> publication,
                                               const std::vector<JPH::GPU::QueryInput> &queries,
                                               const std::vector<JPH::GPU::QueryPrimitive> &primitives,
                                               uint64_t numericalGeneration) const;

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
