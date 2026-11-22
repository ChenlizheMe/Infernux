#include "JoltGpuQueries.h"
#include "JoltGpuProgramSources.h"
#include "PhysicsBodyState.h"
#include <function/renderer/particle/ParticleGpuCollisionScene.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace infernux
{
namespace
{
using namespace JPH::GPU;
enum Port : uint8_t
{
    Parameters,
    Inputs,
    BodyIdentities,
    Models,
    Shapes,
    States,
    Candidates,
    Control,
    Summary,
    Hits,
    ParticleParameters,
    ParticleRows,
    ParticlePrevious,
    ParticleRecords,
    ParticleHeader,
    ParticleOffsets,
    ParticleIndices,
    ParticleStatus,
    CastParameters,
    Geometry,
    PortCount
};
enum Stage : uint8_t
{
    Evaluate,
    Scan,
    Compact,
    ParticleProject,
    ParticleCount,
    ParticleScatter,
    ResumeCasts,
    InitJobs,
    AdvanceJobs,
    JobProgress,
    StageCount
};
struct ParticleExportParameters
{
    UInt4 counts;   // Output rows, static rows, first body slot, previous rows.
    UInt4 revision; // Scene revision low/high, topology revision low/high.
};
static_assert(sizeof(ParticleExportParameters) == 32);

struct PortLayout
{
    std::string_view name;
    uint64_t stride;
    std::array<std::string_view, 8> fields;
    uint32_t integerFields;
};
constexpr PortLayout Ports[] = {
    {"query_params", sizeof(QueryParams), {"counts", "sizes", "selection", "selectedCounts"}, 15},
    {"query_inputs", sizeof(QueryRecordWord), {"value"}, 1},
    {"query_body_ids", sizeof(QueryBodyIdentity), {"ids"}, 1},
    {"body_models", sizeof(BodyModel), {"meta", "mass", "invInertiaDiagonal", "inertiaRotation", "damping"}, 1},
    {"shape_models",
     sizeof(ShapeModel),
     {"meta", "geometry", "localPositionCOM", "localRotation", "material", "surface", "source", "geometryReference"},
     225},
    {"body_states", sizeof(BodyState), {"positionCOM", "rotation", "linearVelocity", "angularVelocity"}, 0},
    {"query_candidates", sizeof(QueryRecordWord), {"value"}, 1},
    {"query_control", sizeof(UInt4), {}, 0},
    {"query_summary", sizeof(QuerySummary), {"counts"}, 1},
    {"query_hits", sizeof(QueryRecordWord), {"value"}, 1},
    {"particle_export_params", sizeof(ParticleExportParameters), {"counts", "revision"}, 3},
    {"particle_export_rows",
     sizeof(JoltGpuParticleExportRow),
     {"selection", "identity", "localBoundsMin", "localBoundsMax"},
     3},
    {"particle_export_previous", sizeof(QueryRecordWord), {"value"}, 1},
    {"particle_export_records", sizeof(QueryRecordWord), {"value"}, 1},
    {"particle_export_header", sizeof(QueryRecordWord), {"value"}, 1},
    {"particle_export_offsets", sizeof(uint32_t), {}, 0},
    {"particle_export_indices", sizeof(uint32_t), {}, 0},
    {"particle_export_status", sizeof(uint32_t), {}, 0},
    {"query_cast_params", sizeof(QueryCastParams), {"controls"}, 1},
    {"query_geometry", sizeof(UInt4), {}, 0},
};
constexpr std::array<std::string_view, StageCount> Entries{"query_evaluate",
                                                           "query_scan",
                                                           "query_compact",
                                                           "particle_collision_project",
                                                           "particle_collision_count",
                                                           "particle_collision_scatter",
                                                           "query_resume_casts",
                                                           "query_init_jobs",
                                                           "query_advance_jobs",
                                                           "query_job_progress"};
constexpr uint32_t Bit(Port port)
{
    return 1u << port;
}
constexpr std::array<uint32_t, StageCount> StagePorts{
    Bit(Parameters) | Bit(Inputs) | Bit(BodyIdentities) | Bit(Models) | Bit(Shapes) | Bit(States) | Bit(Candidates) |
        Bit(Control) | Bit(CastParameters),
    Bit(Parameters) | Bit(Control) | Bit(Summary),
    Bit(Parameters) | Bit(Candidates) | Bit(Control) | Bit(Summary) | Bit(Hits),
    Bit(ParticleParameters) | Bit(ParticleRows) | Bit(ParticlePrevious) | Bit(Shapes) | Bit(States) |
        Bit(ParticleRecords) | Bit(ParticleStatus),
    Bit(ParticleParameters) | Bit(ParticleRecords) | Bit(ParticleHeader) | Bit(ParticleOffsets) | Bit(ParticleStatus),
    Bit(ParticleParameters) | Bit(ParticleRecords) | Bit(ParticleHeader) | Bit(ParticleOffsets) | Bit(ParticleIndices) |
        Bit(ParticleStatus),
    Bit(Parameters) | Bit(Inputs) | Bit(BodyIdentities) | Bit(Models) | Bit(Shapes) | Bit(States) | Bit(Candidates) |
        Bit(Control) | Bit(CastParameters),
    Bit(Parameters) | Bit(Control),
    Bit(Parameters) | Bit(Inputs) | Bit(BodyIdentities) | Bit(Models) | Bit(Shapes) | Bit(States) | Bit(Candidates) |
        Bit(Control) | Bit(CastParameters) | Bit(Geometry),
    Bit(Parameters) | Bit(Control) | Bit(Summary)};

uint64_t CheckedBytes(uint64_t count, uint64_t stride)
{
    if (count > std::numeric_limits<uint64_t>::max() / stride || count > std::numeric_limits<size_t>::max() / stride)
        throw JoltGpuQueryRequestError("Jolt GPU query byte count exceeds the host/device address range");
    return count * stride;
}

template <class T> std::vector<uint8_t> Bytes(const T *records, size_t count)
{
    std::vector<uint8_t> bytes(static_cast<size_t>(CheckedBytes(count, sizeof(T))));
    if (!bytes.empty())
        std::memcpy(bytes.data(), records, bytes.size());
    return bytes;
}

void ValidateLayout(uint8_t port, const ShaderResourceLayout &resource)
{
    const auto &expected = Ports[port];
    const bool uniform = port == Parameters || port == ParticleParameters || port == CastParameters;
    if (port == Control || port == Geometry) {
        if (resource.binding.type != rhi::BindingType::StorageBuffer ||
            resource.binding.readOnlyStorage != (port == Geometry) || resource.elementStride != sizeof(UInt4) ||
            resource.value.kind != ShaderValueKind::Vector || resource.value.scalar != ShaderScalarKind::UInt32 ||
            resource.value.byteSize != sizeof(UInt4) || resource.value.rows != 1 || resource.value.columns != 4)
            throw std::invalid_argument("Jolt GPU query uint4 stream differs from its shared layout");
        return;
    }
    if (port == ParticleOffsets || port == ParticleIndices || port == ParticleStatus) {
        if (resource.binding.type != rhi::BindingType::StorageBuffer || resource.binding.readOnlyStorage ||
            resource.elementStride != sizeof(uint32_t) || resource.value.kind != ShaderValueKind::Scalar ||
            resource.value.scalar != ShaderScalarKind::UInt32 || resource.value.byteSize != sizeof(uint32_t))
            throw std::invalid_argument("Jolt GPU query export scalar stream differs from its shared layout");
        return;
    }
    if (resource.binding.type != (uniform ? rhi::BindingType::UniformBuffer : rhi::BindingType::StorageBuffer) ||
        (uniform ? resource.value.byteSize : resource.elementStride) != expected.stride ||
        resource.value.kind != ShaderValueKind::Struct || resource.value.fields.size() != expected.stride / 16 ||
        (!uniform &&
         resource.binding.readOnlyStorage != (port < Candidates || port == ParticleRows || port == ParticlePrevious)))
        throw std::invalid_argument("Jolt GPU query target differs from its shared record layout");
    for (size_t field = 0; field < resource.value.fields.size(); ++field) {
        const auto &actual = resource.value.fields[field];
        const auto scalar =
            expected.integerFields & (1u << field) ? ShaderScalarKind::UInt32 : ShaderScalarKind::Float32;
        if (actual.name != expected.fields[field] || actual.offset != field * 16 || actual.byteSize != 16 ||
            actual.kind != ShaderValueKind::Vector || actual.scalar != scalar || actual.rows != 1 ||
            actual.columns != 4)
            throw std::invalid_argument("Jolt GPU query target field differs from its shared device record");
    }
}

bool Finite(const Float4 &value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z) && std::isfinite(value.w);
}

void ValidateGeometry(uint32_t type, const Float4 &geometry, const Float4 &rotation, const UInt4 &reference,
                      const std::shared_ptr<const JoltGpuGeometryLease> &lease)
{
    if (!Finite(geometry) || !Finite(rotation))
        throw JoltGpuQueryRequestError("Jolt GPU query geometry and rotation must be finite");
    const double length = double(rotation.x) * rotation.x + double(rotation.y) * rotation.y +
                          double(rotation.z) * rotation.z + double(rotation.w) * rotation.w;
    if (std::abs(length - 1.0) > 1.0e-3)
        throw JoltGpuQueryRequestError("Jolt GPU query rotation must be a unit quaternion");
    if (type == ShapeConvexHull) {
        if (geometry.x < 0 || geometry.y || geometry.z || geometry.w || !reference.x || reference.y < 5 || !lease ||
            !lease->Contains(reference))
            throw JoltGpuQueryRequestError("Jolt GPU Hull requires its exact retained geometry range");
        return;
    }
    if (reference.x || reference.y || reference.z || reference.w)
        throw JoltGpuQueryRequestError("Jolt GPU analytic primitive must not reference a geometry range");
    if (type == ShapeCylinder) {
        if (geometry.x <= 0 || geometry.y <= 0 || geometry.z < 0 || geometry.z > std::min(geometry.x, geometry.y) ||
            geometry.w)
            throw JoltGpuQueryRequestError("Jolt GPU Cylinder requires positive dimensions and a valid cooked radius");
    } else if (type == ShapeSphere) {
        if (geometry.x != 0 || geometry.y != 0 || geometry.z != 0 || geometry.w <= 0)
            throw JoltGpuQueryRequestError("Jolt GPU query Sphere requires a positive radius");
    } else if (type == ShapeBox) {
        if (geometry.x <= 0 || geometry.y <= 0 || geometry.z <= 0 || geometry.w < 0 ||
            geometry.w > std::min({geometry.x, geometry.y, geometry.z}))
            throw JoltGpuQueryRequestError("Jolt GPU query Box requires positive extents and a valid cooked radius");
    } else if (type == ShapeCapsule) {
        if (geometry.x != 0 || geometry.y < 0 || geometry.z != 0 || geometry.w <= 0)
            throw JoltGpuQueryRequestError("Jolt GPU query Capsule requires a nonnegative segment and positive radius");
    } else {
        throw JoltGpuQueryRequestError("Jolt GPU query primitive type is not supported");
    }
}

void ValidateInput(const QueryInput &query, uint32_t bodyBase, uint32_t bodyCount,
                   const std::shared_ptr<const JoltGpuGeometryLease> &lease)
{
    const bool hasLifetime = query.lifetime.x || query.lifetime.y;
    if (query.lifetime.z || query.lifetime.w)
        throw JoltGpuQueryRequestError("GPU query source lifetime reserved words must be zero");
    const bool sweep = (query.meta.w & QuerySweepPolicy) != 0;
    if (query.meta.x != QueryShapeCast || query.meta.y != QueryClosest || query.source.y ||
        (query.meta.w & ~(QueryExcludeBody | QuerySweepPolicy | QuerySweepPairPolicy)) ||
        ((query.meta.w & QuerySweepPairPolicy) && !sweep) || query.filter.y > 1 ||
        (!sweep && (query.source.z || query.source.w || hasLifetime)) ||
        ((query.meta.w & QuerySweepPairPolicy) && !hasLifetime) || !Finite(query.positionMaximum) ||
        !Finite(query.direction) || !Finite(query.rotation) || !Finite(query.geometry) || query.direction.w != 0)
        throw JoltGpuQueryRequestError("Jolt GPU query contains an invalid operation, filter or finite input record");
    if ((query.meta.w & QueryExcludeBody) ? (query.source.x < bodyBase || query.source.x - bodyBase >= bodyCount)
                                          : query.source.x != InvalidQueryBodySlot)
        throw JoltGpuQueryRequestError("Jolt GPU query excluded body differs from its retained publication");
    if (query.positionMaximum.w <= 0 || (query.direction.x == 0 && query.direction.y == 0 && query.direction.z == 0))
        throw JoltGpuQueryRequestError("Jolt GPU shape cast requires nonzero direction and positive maximum distance");
    ValidateGeometry(query.meta.z, query.geometry, query.rotation, query.geometryReference, lease);
}

rhi::ComputeBufferAccess Access(uint8_t stage, uint8_t port)
{
    if (stage >= ParticleProject && stage <= ParticleScatter) {
        if (port == ParticleStatus)
            return rhi::ComputeBufferAccess::ReadWrite;
        if ((stage == ParticleProject && port == ParticleRecords) ||
            (stage == ParticleCount && (port == ParticleHeader || port == ParticleOffsets)) ||
            (stage == ParticleScatter && port == ParticleIndices))
            return rhi::ComputeBufferAccess::Write;
        return rhi::ComputeBufferAccess::Read;
    }
    if (stage == InitJobs && port == Control)
        return rhi::ComputeBufferAccess::Write;
    if (stage == AdvanceJobs && (port == Candidates || port == Control))
        return rhi::ComputeBufferAccess::ReadWrite;
    if (stage == JobProgress && port == Summary)
        return rhi::ComputeBufferAccess::Write;
    if (stage == Evaluate && (port == Candidates || port == Control))
        return rhi::ComputeBufferAccess::Write;
    if (stage == ResumeCasts && (port == Candidates || port == Control))
        return rhi::ComputeBufferAccess::ReadWrite;
    if (stage == Scan && port == Control)
        return rhi::ComputeBufferAccess::ReadWrite;
    if ((stage == Scan && port == Summary) || (stage == Compact && port == Hits))
        return rhi::ComputeBufferAccess::Write;
    return rhi::ComputeBufferAccess::Read;
}

rhi::ComputeDispatchDesc ParticleDispatch(const std::shared_ptr<rhi::ComputeKernel> &kernel,
                                          const std::vector<uint8_t> &ports, uint8_t stage,
                                          const std::array<std::shared_ptr<rhi::ComputeBuffer>, PortCount> &buffers,
                                          uint32_t groups)
{
    rhi::ComputeDispatchDesc dispatch;
    dispatch.kernel = kernel;
    dispatch.groupCountX = groups;
    rhi::ComputeDispatchProof proof;
    for (const auto port : ports) {
        const auto local = static_cast<uint32_t>(dispatch.buffers.size());
        const auto access = Access(stage, port);
        dispatch.buffers.push_back(buffers[port]);
        dispatch.bufferAccesses.push_back(access);
        const bool status = port == ParticleStatus;
        const uint64_t readOffset = status ? sizeof(uint32_t) : 0;
        const uint64_t readBytes = status ? sizeof(uint32_t) : buffers[port]->GetByteSize();
        if (static_cast<uint8_t>(access) & static_cast<uint8_t>(rhi::ComputeBufferAccess::Read))
            proof.reads.push_back({local, readOffset, readBytes});
        if (static_cast<uint8_t>(access) & static_cast<uint8_t>(rhi::ComputeBufferAccess::Write)) {
            const bool fullStatus = status && stage == ParticleCount;
            proof.writes.push_back({local, status && !fullStatus ? sizeof(uint32_t) : 0,
                                    status && !fullStatus ? sizeof(uint32_t) : buffers[port]->GetByteSize()});
        }
    }
    dispatch.proof = std::move(proof);
    return dispatch;
}

// The existing query identity stream also carries the immutable policy suffix.
// Copy in ordinary bounded control work; prepare the reading dispatch only after
// every destination interval has an accepted writer. No additional SSBO is used.
struct QueryPolicyCopy : std::enable_shared_from_this<QueryPolicyCopy>
{
    explicit QueryPolicyCopy(const rhi::ComputeHost &value) : host(value)
    {
    }
    rhi::ComputeHost host;
    std::shared_ptr<rhi::ComputeBuffer> source, target;
    std::vector<rhi::ComputeBufferUpdate> uploads;
    std::function<JoltGpuControlWork()> ready;
    uint32_t first = 0, count = 0, targetBase = 0; // Pair indices; targetBase is a 16-byte word offset.

    JoltGpuControlWork Next()
    {
        if (first == count)
            return ready();
        const uint32_t pairs = std::min(1024u, count - first);
        const uint64_t stride = sizeof(ColliderPairKey);
        std::vector<rhi::ComputeBufferCopy> copies{
            {source,
             target,
             {uint64_t(first) * stride, uint64_t(targetBase) * sizeof(QueryRecordWord) + uint64_t(first) * stride,
              uint64_t(pairs) * stride}}};
        auto packet = rhi::PrepareComputeBatch(host, std::move(uploads), {}, {}, std::move(copies));
        return {packet,
                [self = shared_from_this(), pairs] {
                    self->first += pairs;
                    return self->Next();
                },
                JoltGpuControlWork::ResumeAfter::Acceptance};
    }
};

void RequireSuccess(uint32_t status)
{
    if (status == QueryUnsupported)
        throw std::runtime_error("Jolt GPU query operation or geometry is unsupported");
    if (status == QueryNumericalFailure)
        throw std::runtime_error("Jolt GPU query could not resolve its numerical result");
    if (status != QuerySuccess)
        throw std::runtime_error("Jolt GPU query returned an invalid device status");
}

void ValidateSummary(const QuerySummary &summary, uint32_t queryCount, uint32_t capacity, bool knownCount,
                     bool allowPending = false)
{
    const auto &counts = summary.counts;
    if (counts.y != QuerySuccess && counts.y != QueryUnsupported && counts.y != QueryNumericalFailure &&
        !(allowPending && counts.y == QueryPending))
        throw std::runtime_error("Jolt GPU query returned an invalid device status");
    if (counts.x > capacity || counts.z != queryCount || counts.w || (knownCount && counts.x != queryCount))
        throw std::runtime_error("Jolt GPU query compact summary differs from its prepared output range");
}
} // namespace

JoltGpuQueries::JoltGpuQueries(rhi::ComputeHost &host, const ShaderTargetCode &target) : m_host(host)
{
    for (uint8_t stage = 0; stage < StageCount; ++stage) {
        const auto code = std::find_if(target.entries.begin(), target.entries.end(),
                                       [&](const auto &entry) { return entry.name == Entries[stage]; });
        if (code == target.entries.end() || code->stage != rhi::ShaderStage::Compute ||
            code->workgroupSize != std::array<uint32_t, 3>{64, 1, 1})
            throw std::invalid_argument("Jolt GPU query target is missing its shared compute entry");
        auto &program = m_programs[stage];
        uint32_t actual = 0;
        std::vector<rhi::ComputeBufferBinding> bindings;
        for (const auto &resource : target.resources) {
            const std::array<uint32_t, 2> key{resource.set, resource.binding.binding};
            if (std::find(code->bindings.begin(), code->bindings.end(), key) == code->bindings.end())
                continue;
            const auto port = std::find_if(std::begin(Ports), std::end(Ports),
                                           [&](const auto &value) { return value.name == resource.name; });
            if (port == std::end(Ports) || resource.set != 0)
                throw std::invalid_argument("Jolt GPU query entry contains an unknown resource");
            const auto index = static_cast<uint8_t>(port - std::begin(Ports));
            if (actual & (1u << index))
                throw std::invalid_argument("Jolt GPU query entry repeats a resource");
            ValidateLayout(index, resource);
            bindings.push_back({resource.binding.binding, resource.binding.type, resource.binding.readOnlyStorage});
            program.ports.push_back(index);
            actual |= 1u << index;
        }
        if (actual != StagePorts[stage] || code->bindings.size() != program.ports.size())
            throw std::invalid_argument("Jolt GPU query entry differs from its shared phase resource set");
        program.kernel = PrepareJoltGpuKernel(m_host, target.target, *code, std::move(bindings));
    }
}

struct JoltGpuQueries::Execution
{
    rhi::ComputeHost host;
    std::shared_ptr<JoltGpuQueryResult> result;
    std::vector<rhi::ComputeDispatchDesc> resume;
    std::shared_ptr<rhi::PreparedComputeBatch> packet;
    bool payloadInPacket;
    bool jobs = false;
    std::vector<rhi::ComputeDispatchDesc> finalize;
};

JoltGpuControlWork JoltGpuQueries::Continue(const std::shared_ptr<Execution> &execution)
{
    const auto &result = execution->result;
    if (!result->summaryReadback || !result->summaryReadback->IsComplete())
        throw std::logic_error("Jolt GPU query continuation requires its completed receipt");
    QuerySummary summary{};
    result->summaryReadback->ReadInto(&summary, sizeof(summary));
    if (execution->jobs) {
        ValidateSummary(summary, result->queryCount, result->hitCapacity, false, true);
        if (summary.counts.x)
            throw std::runtime_error("Jolt GPU private geometry progress must not publish compact counts");
        if (summary.counts.y == QuerySuccess) {
            execution->jobs = false;
            execution->resume.clear();
            execution->packet = rhi::PrepareComputeBatch(execution->host, {}, std::move(execution->finalize),
                                                         {{result->summary, 0, sizeof(QuerySummary)}});
            result->summaryReadback = execution->packet->Readback(0);
            JoltGpuControlWork work{execution->packet, [execution]() { return Continue(execution); }};
            work.geometry = result->geometry;
            return work;
        }
    } else {
        ValidateSummary(summary, result->queryCount, result->hitCapacity, result->knownCount,
                        !execution->resume.empty());
    }
    if (summary.counts.y == QueryPending) {
        // The device retains the current leaf, advancement and best hit. Only
        // metadata crosses the bus while another bounded slice is required.
        execution->packet = rhi::PrepareComputeBatch(execution->host, {}, execution->resume,
                                                     {{result->summary, 0, sizeof(QuerySummary)}});
        execution->payloadInPacket = false;
        result->summaryReadback = execution->packet->Readback(0);
        JoltGpuControlWork work{execution->packet, [execution]() { return Continue(execution); }};
        work.geometry = result->geometry;
        return work;
    }
    result->summaryValue = summary;
    result->headerComplete = true;
    // Recognized numerical failures belong to the read-only result consumer,
    // not to the world's solver source. No incomplete hit is published.
    if (result->summaryValue.counts.y != QuerySuccess)
        return {};
    if (execution->payloadInPacket) {
        result->tailBatch = execution->packet;
        return {};
    }
    std::vector<rhi::ComputeBufferRead> reads;
    const auto count = result->summaryValue.counts.x;
    if (count) {
        result->hitReadbackIndex = static_cast<uint32_t>(reads.size());
        reads.push_back({result->hits, 0, CheckedBytes(count, sizeof(QueryHit))});
    }
    result->spanReadbackIndex = static_cast<uint32_t>(reads.size());
    reads.push_back({result->spans, 0, CheckedBytes(result->queryCount, sizeof(QuerySpan))});
    result->tailBatch = rhi::PrepareComputeBatch(execution->host, {}, {}, std::move(reads));
    JoltGpuControlWork work{result->tailBatch};
    work.geometry = result->geometry;
    return work;
}

JoltGpuControlWork PrepareJoltGpuQueryWork(std::function<JoltGpuControlWork()> prepare, std::exception_ptr &error)
{
    try {
        auto work = prepare();
        const auto observe = [&error](std::function<JoltGpuControlWork()> &resume) {
            if (resume)
                resume = [next = std::move(resume), &error]() mutable {
                    return PrepareJoltGpuQueryWork(std::move(next), error);
                };
        };
        observe(work.continueWork);
        observe(work.prepareMinimum);
        return work;
    } catch (const JoltGpuQueryRequestError &) {
        error = std::current_exception();
        return {};
    }
}

JoltGpuPreparedQuery JoltGpuQueries::Prepare(std::shared_ptr<const JoltGpuScene::Publication> publication,
                                             const std::vector<QueryInput> &queries, uint64_t numericalGeneration,
                                             uint32_t castIterationQuantum) const
{
    auto result = std::make_shared<JoltGpuQueryResult>();
    result->geometry = publication->geometry;
    const uint64_t queryCount = queries.size();
    if (queryCount > UINT32_MAX)
        throw JoltGpuQueryRequestError("Jolt GPU query input count exceeds device indexing");
    std::array<std::shared_ptr<rhi::ComputeBuffer>, PortCount> buffers;
    uint32_t bodyBase = 0, bodyCount = 0, leafBase = 0, leafCount = 0;
    uint32_t totalBodies = 0, totalLeaves = 0;
    if (publication) {
        bodyBase = publication->bodyBase;
        bodyCount = publication->bodyCount;
        leafBase = publication->leafBase;
        leafCount = publication->leafCount;
        buffers[Models] = publication->bodyModels;
        buffers[Shapes] = publication->shapeModels;
        buffers[States] = publication->bodyStates;
        if (publication->directory->actors.size() > UINT32_MAX || publication->directory->leaves.size() > UINT32_MAX)
            throw JoltGpuQueryRequestError("GPU query author directory exceeds global indexing");
        totalBodies = static_cast<uint32_t>(publication->directory->actors.size());
        totalLeaves = static_cast<uint32_t>(publication->directory->leaves.size());
        if (bodyBase > totalBodies || bodyCount > totalBodies - bodyBase || leafBase > totalLeaves ||
            leafCount > totalLeaves - leafBase)
            throw std::invalid_argument("GPU query ranges differ from their exact author directory");
        const auto frozen = [&](uint8_t port, uint64_t bytes) {
            const auto &buffer = buffers[port];
            if (!buffer || !buffer->IsFrozen() || !m_host.SharesServicesWith(buffer->GetHost()) ||
                buffer->GetByteSize() != bytes)
                throw std::invalid_argument("GPU query requires its exact frozen author/output generation");
        };
        if (totalBodies)
            frozen(Models, CheckedBytes(totalBodies, sizeof(BodyModel)));
        if (bodyCount)
            frozen(States, CheckedBytes(bodyCount, sizeof(BodyState)));
        else if (buffers[States])
            throw std::invalid_argument("Empty domain output must not carry body numerical rows");
        if (totalLeaves)
            frozen(Shapes, CheckedBytes(totalLeaves, sizeof(ShapeModel)));
    }
    for (const auto &query : queries)
        ValidateInput(query, bodyBase, bodyCount, result->geometry);
    const bool generic =
        publication->hasGenericGeometry || std::any_of(queries.begin(), queries.end(), [](const auto &query) {
            return query.meta.z == ShapeCylinder || query.meta.z == ShapeConvexHull;
        });
    if (!queries.empty() && (!result->geometry || !result->geometry->buffer ||
                             !m_host.SharesServicesWith(result->geometry->buffer->GetHost())))
        throw JoltGpuQueryRequestError("GPU query requires its compatible initialized geometry lease");
    const bool hasCasts =
        std::any_of(queries.begin(), queries.end(), [](const auto &query) { return query.meta.x == QueryShapeCast; });
    if (!castIterationQuantum)
        throw JoltGpuQueryRequestError("Jolt GPU cast iteration quantum must be positive");
    constexpr bool knownCount = true;
    constexpr uint32_t stride = 1;
    const uint64_t records = queryCount * stride;
    if (records > std::numeric_limits<uint32_t>::max())
        throw JoltGpuQueryRequestError("Jolt GPU query scratch capacity exceeds device indexing");
    const auto capacity = static_cast<uint32_t>(records);
    constexpr auto words = sizeof(QueryHit) / sizeof(QueryRecordWord);
    if (capacity > std::numeric_limits<uint32_t>::max() / words)
        throw JoltGpuQueryRequestError("Jolt GPU query record words exceed device indexing");
    // Span/CA control fits within the already required per-request input.
    // Keep this dominance proof independent of the input's current ABI size.
    static_assert(sizeof(QuerySpan) == sizeof(QueryRecordWord));
    static_assert(sizeof(QueryCastState) == 4 * sizeof(QueryRecordWord));
    static_assert(sizeof(QueryInput) >= sizeof(QuerySpan) + sizeof(QueryCastState));
    const uint64_t spanBytes = CheckedBytes(queryCount, sizeof(QuerySpan));
    const uint64_t prefixBytes = CheckedBytes(queryCount, sizeof(QuerySpan) + sizeof(QueryCastState));
    const uint64_t prefixWords = prefixBytes / sizeof(UInt4);
    // Tail header4 + up to32 fixed2048-word workers coexist with all spans/CA states.
    // Fewer workers change scheduling only, never request/candidate capacity.
    const auto bindingLimit = m_host.device.GetCapabilities().portable.maxStorageBufferBinding;
    const uint64_t maxControlWords = std::min<uint64_t>(UINT32_MAX, bindingLimit / sizeof(UInt4));
    uint32_t workers = 0;
    if (generic) {
        if (prefixWords > maxControlWords || maxControlWords - prefixWords < 4u + 2048u)
            throw JoltGpuQueryRequestError("Jolt GPU generic query control exceeds device indexing/binding range");
        workers =
            static_cast<uint32_t>(std::min<uint64_t>({32u, queryCount, (maxControlWords - prefixWords - 4u) / 2048u}));
    }
    const uint64_t controlWords = prefixWords + (generic ? 4u + uint64_t(workers) * 2048u : 0u);
    if (controlWords > UINT32_MAX)
        throw JoltGpuQueryRequestError("Jolt GPU query control exceeds device indexing");
    result->publication = std::move(publication);
    result->numericalGeneration = numericalGeneration;
    result->queryCount = static_cast<uint32_t>(queryCount);
    result->hitCapacity = capacity;
    result->knownCount = knownCount;
    if (!queryCount) {
        result->headerComplete = result->materialized = result->recordsMaterialized = true;
        return {result, {}};
    }
    // Reject impossible dispatch dimensions before allocating request storage.
    const auto groups = [](uint32_t count) { return count / 64 + static_cast<uint32_t>(count % 64 != 0); };
    if (groups(capacity) > m_host.device.GetCapabilities().limits.maxComputeWorkgroupCount[0])
        throw JoltGpuQueryRequestError("Jolt GPU query capacity exceeds the device dispatch range");
    rhi::ComputeHost host = m_host;
    const auto allocate = [&](uint8_t port, uint64_t count) {
        const auto bytes = CheckedBytes(count, Ports[port].stride);
        if (port != Parameters && port != CastParameters &&
            !host.device.GetCapabilities().IsStorageBufferRangeValid(0, bytes))
            throw JoltGpuQueryRequestError("Jolt GPU query storage exceeds the device binding range");
        buffers[port] =
            rhi::ComputeBuffer::TryCreatePrivate(host, {bytes / sizeof(uint32_t), rhi::ComputeScalarType::UInt32, 1});
        if (!buffers[port])
            throw JoltGpuQueryRequestError("Jolt GPU query preparation exceeds available managed device capacity");
    };
    std::vector<rhi::ComputeBufferUpdate> updates;
    const auto upload = [&](uint8_t port, std::vector<uint8_t> bytes) {
        allocate(port, bytes.size() / Ports[port].stride);
        updates.push_back({buffers[port], 0, std::move(bytes)});
    };
    if (!totalBodies) {
        const BodyModel model{};
        upload(Models, Bytes(&model, 1));
    }
    if (!bodyCount) {
        const BodyState state{};
        upload(States, Bytes(&state, 1));
    }
    if (!totalLeaves) {
        const ShapeModel shape{};
        upload(Shapes, Bytes(&shape, 1));
    }
    const QueryParams parameters{
        {result->queryCount, bodyBase, bodyCount, leafCount},
        {0, stride, capacity, 0},
        {result->publication ? result->publication->worldRow : 0u, bodyBase, leafBase,
         result->publication ? result->publication->jointBase : 0u},
        {bodyCount, leafCount, result->publication ? result->publication->jointCount : 0u, 0u}};
    upload(Parameters, Bytes(&parameters, 1));
    upload(Inputs, Bytes(queries.data(), queries.size()));
    std::shared_ptr<QueryPolicyCopy> copy;
    const bool hasSweep = std::any_of(queries.begin(), queries.end(),
                                      [](const auto &query) { return (query.meta.w & QuerySweepPolicy) != 0; });
    // Identity records are compact for this retained publication's world.
    // Author model/shape addresses and query result addresses remain global.
    const bool hasPairPolicy = std::any_of(
        queries.begin(), queries.end(), [](const auto &query) { return (query.meta.w & QuerySweepPairPolicy) != 0; });
    const auto pairs = result->publication ? result->publication->colliderPairs : nullptr;
    if (hasPairPolicy && !pairs)
        throw JoltGpuQueryRequestError("GPU source sweep requires its retained collider policy");
    const uint64_t pairCount64 = hasPairPolicy ? pairs->keys.size() : 0;
    const uint64_t identityPrefix = std::max<uint64_t>(1, uint64_t(bodyCount) + (hasSweep ? leafCount : 0u));
    if (identityPrefix > UINT32_MAX || pairCount64 > (UINT32_MAX - identityPrefix) / 2u)
        throw JoltGpuQueryRequestError("Jolt GPU query identity words exceed device indexing");
    const uint32_t pairCount = static_cast<uint32_t>(pairCount64);
    if (pairCount &&
        (!pairs->buffer || !pairs->buffer->IsFrozen() || !host.SharesServicesWith(pairs->buffer->GetHost()) ||
         pairs->buffer->GetByteSize() != CheckedBytes(pairCount, sizeof(ColliderPairKey))))
        throw JoltGpuQueryRequestError("GPU source sweep policy differs from its exact frozen generation");
    std::vector<QueryBodyIdentity> identities(static_cast<size_t>(identityPrefix));
    identities.front().ids.x = InvalidQueryBodySlot;
    for (uint32_t local = 0; local < bodyCount; ++local) {
        const auto &actor = result->publication->directory->actors[bodyBase + local];
        identities[local].ids = {actor ? actor->bodyId : InvalidQueryBodySlot, 0, 0, 0};
    }
    if (hasSweep) {
        for (uint32_t local = 0; local < leafCount; ++local) {
            const auto &address = result->publication->directory->leaves[leafBase + local];
            const auto &actor = result->publication->directory->actors[address.bodySlot];
            const auto id = actor->leaves[address.actorLeaf].surfaceGameObjectId;
            identities[bodyCount + local].ids = {static_cast<uint32_t>(id), static_cast<uint32_t>(id >> 32), 0, 0};
        }
    }
    allocate(BodyIdentities, identityPrefix + pairCount64 * 2u);
    updates.push_back({buffers[BodyIdentities], 0, Bytes(identities.data(), identities.size())});
    if (pairCount) {
        copy = std::make_shared<QueryPolicyCopy>(host);
        copy->source = pairs->buffer;
        copy->target = buffers[BodyIdentities];
        copy->count = pairCount;
        copy->targetBase = static_cast<uint32_t>(identityPrefix);
    }
    const QueryCastParams castParameters{{generic ? std::min(32u, castIterationQuantum) : castIterationQuantum,
                                          bodyCount, static_cast<uint32_t>(identityPrefix), pairCount}};
    upload(CastParameters, Bytes(&castParameters, 1));
    allocate(Candidates, uint64_t(capacity) * words);
    buffers[Geometry] = result->geometry->buffer;
    allocate(Control, controlWords);
    allocate(Summary, 1);
    allocate(Hits, uint64_t(capacity) * words);
    const auto dispatch = [&](uint8_t stage) {
        const auto &program = m_programs[stage];
        rhi::ComputeDispatchDesc dispatch;
        dispatch.kernel = program.kernel;
        dispatch.groupCountX = stage == Scan || stage == JobProgress ? 1
                               : stage == InitJobs ? groups(static_cast<uint32_t>(controlWords - prefixWords))
                               : stage == AdvanceJobs
                                   ? groups(workers)
                                   : groups(stage == Evaluate || stage == ResumeCasts ? result->queryCount : capacity);
        if (dispatch.groupCountX > host.device.GetCapabilities().limits.maxComputeWorkgroupCount[0])
            throw JoltGpuQueryRequestError("Jolt GPU query phase exceeds the device dispatch range");
        rhi::ComputeDispatchProof proof;
        for (const auto port : program.ports) {
            const auto binding = static_cast<uint32_t>(dispatch.buffers.size());
            const auto access = Access(stage, port);
            dispatch.buffers.push_back(buffers[port]);
            dispatch.bufferAccesses.push_back(access);
            // Scan and compact only access spans. Cast work also keeps a per-request cursor.
            if (port == Geometry) {
                for (const auto &range : result->geometry->wordRanges)
                    proof.reads.push_back(
                        {binding, range.first * sizeof(UInt4), (range.second - range.first) * sizeof(UInt4)});
                continue;
            }
            uint64_t offset = 0;
            uint64_t bytes = buffers[port]->GetByteSize();
            if (port == Control) {
                if (stage == Scan || stage == Compact)
                    bytes = spanBytes;
                else if (stage == Evaluate || stage == ResumeCasts)
                    bytes = prefixBytes;
                else if (stage == InitJobs) {
                    offset = prefixBytes;
                    bytes -= prefixBytes;
                }
            }
            const rhi::ComputeBindingRange range{binding, offset, bytes};
            if (static_cast<uint8_t>(access) & static_cast<uint8_t>(rhi::ComputeBufferAccess::Read))
                proof.reads.push_back(range);
            if (static_cast<uint8_t>(access) & static_cast<uint8_t>(rhi::ComputeBufferAccess::Write))
                proof.writes.push_back(range);
        }
        dispatch.proof = std::move(proof);
        return dispatch;
    };
    std::vector<rhi::ComputeDispatchDesc> dispatches;
    if (generic)
        dispatches = {dispatch(Evaluate), dispatch(InitJobs), dispatch(AdvanceJobs), dispatch(JobProgress)};
    else
        dispatches = {dispatch(Evaluate), dispatch(Scan), dispatch(Compact)};
    std::vector<rhi::ComputeDispatchDesc> finalize;
    if (generic)
        finalize = {dispatch(Scan), dispatch(Compact)};
    result->hits = buffers[Hits];
    result->spans = buffers[Control];
    result->summary = buffers[Summary];
    std::vector<rhi::ComputeBufferRead> reads{{result->summary, 0, sizeof(QuerySummary)}};
    if (!generic && knownCount) {
        result->hitReadbackIndex = static_cast<uint32_t>(reads.size());
        reads.push_back({result->hits, 0, CheckedBytes(result->queryCount, sizeof(QueryHit))});
        result->spanReadbackIndex = static_cast<uint32_t>(reads.size());
        reads.push_back({result->spans, 0, CheckedBytes(result->queryCount, sizeof(QuerySpan))});
    }
    std::vector<rhi::ComputeDispatchDesc> resume;
    if (generic)
        resume = {dispatch(AdvanceJobs), dispatch(JobProgress)};
    else if (hasCasts)
        resume = {dispatch(ResumeCasts), dispatch(Scan), dispatch(Compact)};
    if (copy)
        copy->uploads = std::move(updates);
    auto ready = [host, result, updates = std::move(updates), dispatches = std::move(dispatches),
                  reads = std::move(reads), resume = std::move(resume), finalize = std::move(finalize), generic,
                  payload = !generic && knownCount]() mutable -> JoltGpuControlWork {
        auto batch = rhi::PrepareComputeBatch(host, std::move(updates), std::move(dispatches), std::move(reads));
        result->summaryReadback = batch->Readback(0);
        auto execution = std::make_shared<Execution>(
            Execution{host, result, std::move(resume), batch, payload, generic, std::move(finalize)});
        JoltGpuControlWork work{batch, [execution]() { return Continue(execution); }};
        work.geometry = result->geometry;
        return work;
    };
    if (copy) {
        copy->ready = std::move(ready);
        return {result, copy->Next()};
    }
    return {result, ready()};
}

JoltGpuPreparedParticleExport
JoltGpuQueries::PrepareParticleExport(const std::shared_ptr<const JoltGpuScene::Publication> &publication,
                                      const std::vector<JoltGpuParticleExportRow> &rows,
                                      std::shared_ptr<rhi::ComputeBuffer> previousRecords, uint32_t staticCount,
                                      uint64_t revision, uint64_t topologyRevision) const
{
    m_host.RequireAlive();
    if (!publication || !publication->geometry || !publication->geometry->buffer)
        throw JoltGpuQueryRequestError("GPU particle export requires the accepted geometry lease");
    const auto sourceMatches = [&](const auto &buffer) {
        return buffer && buffer->IsFrozen() && m_host.SharesServicesWith(buffer->GetHost());
    };
    if ((!rows.empty() && (!sourceMatches(publication->shapeModels) || !sourceMatches(publication->bodyStates))) ||
        rows.size() > std::numeric_limits<uint32_t>::max() / 512u || staticCount > rows.size() || !topologyRevision)
        throw JoltGpuQueryRequestError("GPU particle export source, row domain or topology is invalid");
    auto result = std::make_shared<JoltGpuParticleExportResult>();
    result->payload = std::make_shared<particle::GpuParticleCollisionDevicePayload>();
    const uint32_t count = static_cast<uint32_t>(rows.size());
    if (count >
        std::numeric_limits<uint32_t>::max() / (sizeof(particle::GpuParticleColliderRecord) / sizeof(QueryRecordWord)))
        throw JoltGpuQueryRequestError("GPU particle collider words exceed device indexing");
    uint32_t previousCount = 0;
    if (previousRecords) {
        if (!previousRecords->IsFrozen() || !m_host.SharesServicesWith(previousRecords->GetHost()) ||
            !previousRecords->GetByteSize() ||
            previousRecords->GetByteSize() % sizeof(particle::GpuParticleColliderRecord))
            throw JoltGpuQueryRequestError("GPU particle history requires frozen records from the same compute owner");
        const auto capacity = previousRecords->GetByteSize() / sizeof(particle::GpuParticleColliderRecord);
        if (capacity > std::numeric_limits<uint32_t>::max() /
                           (sizeof(particle::GpuParticleColliderRecord) / sizeof(QueryRecordWord)) ||
            !previousRecords->HasInitializedRange(0, previousRecords->GetByteSize()))
            throw JoltGpuQueryRequestError("GPU particle history exceeds its initialized record range");
        previousCount = static_cast<uint32_t>(capacity);
    }
    for (const auto &row : rows)
        if ((row.selection.x < publication->leafBase ||
             row.selection.x - publication->leafBase >= publication->leafCount) ||
            (!row.selection.z && !row.selection.w) || !Finite(row.localBoundsMin) || !Finite(row.localBoundsMax) ||
            row.localBoundsMin.w || row.localBoundsMax.w || row.localBoundsMin.x > row.localBoundsMax.x ||
            row.localBoundsMin.y > row.localBoundsMax.y || row.localBoundsMin.z > row.localBoundsMax.z ||
            (row.selection.y != InvalidQueryBodySlot && row.selection.y >= previousCount))
            throw JoltGpuQueryRequestError("GPU particle row or previous identity map exceeds its retained source");
    auto &payload = *result->payload;
    payload.rowCount = count;
    payload.staticCount = staticCount;
    payload.revision = revision;
    payload.topologyRevision = topologyRevision;
    payload.retained = publication;
    rhi::ComputeHost host = m_host;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, PortCount> buffers;
    std::vector<rhi::ComputeBufferUpdate> updates;
    const auto allocate = [&](uint8_t port, uint64_t bytes) {
        buffers[port] =
            rhi::ComputeBuffer::TryCreatePrivate(host, {bytes / sizeof(uint32_t), rhi::ComputeScalarType::UInt32, 1});
        if (!buffers[port])
            throw JoltGpuQueryRequestError("GPU particle export exceeds available managed device capacity");
    };
    const auto upload = [&](uint8_t port, std::vector<uint8_t> bytes) {
        allocate(port, bytes.size());
        updates.push_back({buffers[port], 0, std::move(bytes)});
    };
    const ParticleExportParameters parameters{{count, staticCount, publication->bodyBase, previousCount},
                                              {static_cast<uint32_t>(revision), static_cast<uint32_t>(revision >> 32),
                                               static_cast<uint32_t>(topologyRevision),
                                               static_cast<uint32_t>(topologyRevision >> 32)}};
    upload(ParticleParameters, Bytes(&parameters, 1));
    if (rows.empty()) {
        const JoltGpuParticleExportRow empty{};
        upload(ParticleRows, Bytes(&empty, 1));
    } else
        upload(ParticleRows, Bytes(rows.data(), rows.size()));
    buffers[States] = publication->bodyStates;
    buffers[Shapes] = publication->shapeModels;
    buffers[ParticlePrevious] = std::move(previousRecords);
    if (!buffers[ParticlePrevious])
        upload(ParticlePrevious, std::vector<uint8_t>(sizeof(particle::GpuParticleColliderRecord)));
    allocate(ParticleRecords, CheckedBytes(std::max(1u, count), sizeof(particle::GpuParticleColliderRecord)));
    if (!count)
        updates.push_back(
            {buffers[ParticleRecords], 0, std::vector<uint8_t>(sizeof(particle::GpuParticleColliderRecord))});
    allocate(ParticleHeader, sizeof(particle::GpuParticleCollisionSceneHeader));
    allocate(ParticleOffsets, CheckedBytes(513u, sizeof(uint32_t)));
    const UInt4 status{0, QuerySuccess, count, 1};
    upload(ParticleStatus, Bytes(&status, 1));
    payload.records = buffers[ParticleRecords];
    payload.header = buffers[ParticleHeader];
    payload.gridOffsets = buffers[ParticleOffsets];
    result->status = buffers[ParticleStatus];
    payload.worldId = publication->worldId;
    payload.simulationEpoch = publication->simulationEpoch;
    payload.geometry = publication->geometry->buffer;
    payload.geometryLease = publication->geometry;
    payload.geometryWordRanges = publication->geometry->wordRanges;
    const uint32_t groups = count / 64 + static_cast<uint32_t>(count % 64 != 0);
    if (groups > m_host.device.GetCapabilities().limits.maxComputeWorkgroupCount[0])
        throw JoltGpuQueryRequestError("GPU particle collider count exceeds the device dispatch range");
    std::vector<rhi::ComputeDispatchDesc> dispatches;
    if (count)
        dispatches.push_back(ParticleDispatch(m_programs[ParticleProject].kernel, m_programs[ParticleProject].ports,
                                              ParticleProject, buffers, groups));
    dispatches.push_back(
        ParticleDispatch(m_programs[ParticleCount].kernel, m_programs[ParticleCount].ports, ParticleCount, buffers, 1));
    auto batch =
        rhi::PrepareComputeBatch(host, std::move(updates), std::move(dispatches), {{result->status, 0, sizeof(UInt4)}});
    result->batch = batch;
    result->statusReadback = batch->Readback(0);
    const auto scatter = m_programs[ParticleScatter];
    JoltGpuControlWork work{
        batch, [host, result, buffers, scatter, geometry = publication->geometry]() mutable -> JoltGpuControlWork {
            result->ReadStatus();
            if (result->statusValue.y != QuerySuccess)
                return {};
            const auto references = result->statusValue.x;
            result->payload->gridRefCount = references;
            buffers[ParticleIndices] = rhi::ComputeBuffer::TryCreatePrivate(
                host, {std::max(1u, references), rhi::ComputeScalarType::UInt32, 1});
            if (!buffers[ParticleIndices])
                throw JoltGpuQueryRequestError(
                    "GPU particle reference output exceeds available managed device capacity");
            result->payload->gridRefs = buffers[ParticleIndices];
            std::vector<rhi::ComputeBufferUpdate> updates;
            std::vector<rhi::ComputeDispatchDesc> dispatches;
            if (references) {
                const auto cells = result->statusValue.w;
                const auto groups = cells / 64 + static_cast<uint32_t>(cells % 64 != 0);
                dispatches.push_back(ParticleDispatch(scatter.kernel, scatter.ports, ParticleScatter, buffers, groups));
            } else {
                const uint32_t invalid = InvalidQueryBodySlot;
                updates.push_back({buffers[ParticleIndices], 0, Bytes(&invalid, 1)});
            }
            result->batch = rhi::PrepareComputeBatch(host, std::move(updates), std::move(dispatches),
                                                     {{result->status, 0, sizeof(UInt4)}});
            result->statusReadback = result->batch->Readback(0);
            result->statusComplete = false;
            JoltGpuControlWork tail{result->batch};
            tail.geometry = geometry;
            return tail;
        }};
    work.geometry = publication->geometry;
    return {std::move(result), std::move(work)};
}

void JoltGpuParticleExportResult::ReadStatus()
{
    if (statusComplete)
        return;
    if (!statusReadback || !statusReadback->IsComplete())
        throw std::logic_error("GPU particle export status requires its completed receipt");
    statusReadback->ReadInto(&statusValue, sizeof(statusValue));
    if ((statusValue.y != QuerySuccess && statusValue.y != QueryUnsupported &&
         statusValue.y != QueryNumericalFailure) ||
        statusValue.x > uint64_t(payload->rowCount) * 512u || statusValue.z != payload->rowCount || !statusValue.w ||
        statusValue.w > 512u)
        throw std::runtime_error("GPU particle export returned corrupt status metadata");
    statusComplete = true;
}

void JoltGpuParticleExportResult::Validate()
{
    if (validated)
        return;
    ReadStatus();
    RequireSuccess(statusValue.y);
    if (statusValue.x != payload->gridRefCount || !payload->records || !payload->header || !payload->gridOffsets ||
        !payload->gridRefs)
        throw std::runtime_error("GPU particle export has no completed reference prefix");
    for (auto *buffer : {&payload->records, &payload->header, &payload->gridOffsets, &payload->gridRefs}) {
        *buffer = (*buffer)->Freeze();
        (*buffer)->PublishStorage();
    }
    validated = true;
}

void JoltGpuQueryResult::ReadSummary()
{
    if (headerComplete)
        return;
    if (!summaryReadback || !summaryReadback->IsComplete())
        throw std::logic_error("Jolt GPU query summary requires its completed receipt");
    summaryReadback->ReadInto(&summaryValue, sizeof(QuerySummary));
    ValidateSummary(summaryValue, queryCount, hitCapacity, knownCount);
    headerComplete = true;
}

void JoltGpuQueryResult::Materialize()
{
    if (materialized)
        return;
    ReadSummary();
    RequireSuccess(summaryValue.counts.y);
    if (!tailBatch || !tailBatch->GetSubmission().IsValid())
        throw std::logic_error("Jolt GPU query metadata requires its completed receipt");
    const auto count = summaryValue.counts.x;
    std::vector<QuerySpan> hostSpans(queryCount);
    if (!tailBatch->Readback(spanReadbackIndex)->IsComplete())
        throw std::logic_error("Jolt GPU query metadata requires its completed receipt");
    tailBatch->Readback(spanReadbackIndex)->ReadInto(hostSpans.data(), CheckedBytes(queryCount, sizeof(QuerySpan)));
    uint32_t end = 0;
    for (const auto &span : hostSpans) {
        RequireSuccess(span.range.z);
        if (span.range.x != end || span.range.y > count - end || span.range.w != 0u ||
            (knownCount && span.range.y != 1))
            throw std::runtime_error("Jolt GPU query span differs from its compact result prefix");
        end += span.range.y;
    }
    if (end != count)
        throw std::runtime_error("Jolt GPU query spans do not cover its compact result prefix");
    materialized = true;
}

namespace
{
void ValidateRecord(const QueryHit &hit)
{
    if (hit.meta.y > 1 || !Finite(hit.positionDistance) || !Finite(hit.normalFraction) || !Finite(hit.pointADepth) ||
        hit.positionDistance.w < 0 || hit.pointADepth.w < 0)
        throw std::runtime_error("Jolt GPU query returned an invalid or nonfinite hit record");
}
} // namespace

const std::vector<QueryHit> &JoltGpuQueryResult::ReadHits()
{
    if (!queryCount)
        return hostHits;
    Materialize();
    if (!recordsMaterialized) {
        const auto count = summaryValue.counts.x;
        if (count && !tailBatch->Readback(hitReadbackIndex)->IsComplete())
            throw std::logic_error("Jolt GPU query records require their completed receipt");
        hostHits.resize(count);
        if (count)
            tailBatch->Readback(hitReadbackIndex)->ReadInto(hostHits.data(), CheckedBytes(count, sizeof(QueryHit)));
        for (const auto &record : hostHits)
            ValidateRecord(record);
        recordsMaterialized = true;
    }
    return hostHits;
}

} // namespace infernux
