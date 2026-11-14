#include "JoltGpuQueries.h"
#include <function/renderer/particle/ParticleGpuCollisionScene.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
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
    Spans,
    Summary,
    Hits,
    ExportParameters,
    ExportValues,
    ParticleParameters,
    ParticleRows,
    ParticlePrevious,
    ParticleRecords,
    ParticleHeader,
    ParticleOffsets,
    ParticleIndices,
    ParticleStatus,
    CastParameters,
    CastStates,
    PortCount
};
enum Stage : uint8_t
{
    Evaluate,
    Scan,
    Compact,
    Export,
    ParticleProject,
    ParticleCount,
    ParticleScatter,
    ResumeCasts,
    StageCount
};
struct ParticleExportParameters
{
    UInt4 counts;   // Output rows, static rows, source shape rows, previous rows.
    UInt4 revision; // Scene revision low/high, topology revision low/high.
};
static_assert(sizeof(ParticleExportParameters) == 32);

struct PortLayout
{
    std::string_view name;
    uint64_t stride;
    std::array<std::string_view, 7> fields;
    uint32_t integerFields;
};
constexpr PortLayout Ports[] = {
    {"query_params", sizeof(QueryParams), {"counts", "sizes", "selection", "selectedCounts"}, 15},
    {"query_inputs", sizeof(QueryRecordWord), {"value"}, 1},
    {"query_body_ids", sizeof(QueryBodyIdentity), {"ids"}, 1},
    {"body_models", sizeof(BodyModel), {"meta", "mass", "invInertiaDiagonal", "inertiaRotation", "damping"}, 1},
    {"shape_models",
     sizeof(ShapeModel),
     {"meta", "geometry", "localPositionCOM", "localRotation", "material", "surface", "source"},
     97},
    {"body_states", sizeof(BodyState), {"positionCOM", "rotation", "linearVelocity", "angularVelocity"}, 0},
    {"query_candidates", sizeof(QueryRecordWord), {"value"}, 1},
    {"query_spans", sizeof(QuerySpan), {"range"}, 1},
    {"query_summary", sizeof(QuerySummary), {"counts"}, 1},
    {"query_hits", sizeof(QueryRecordWord), {"value"}, 1},
    {"query_export_params", sizeof(QueryExportParams), {"output", "sizes", "selection"}, 7},
    {"query_export_values", sizeof(uint32_t), {}, 0},
    {"particle_export_params", sizeof(ParticleExportParameters), {"counts", "revision"}, 3},
    {"particle_export_rows", sizeof(JoltGpuParticleExportRow), {"selection", "identity"}, 3},
    {"particle_export_previous", sizeof(QueryRecordWord), {"value"}, 1},
    {"particle_export_records", sizeof(QueryRecordWord), {"value"}, 1},
    {"particle_export_header", sizeof(QueryRecordWord), {"value"}, 1},
    {"particle_export_offsets", sizeof(uint32_t), {}, 0},
    {"particle_export_indices", sizeof(uint32_t), {}, 0},
    {"particle_export_status", sizeof(uint32_t), {}, 0},
    {"query_cast_params", sizeof(QueryCastParams), {"controls"}, 1},
    {"query_cast_states", sizeof(QueryCastState), {"cursor", "progress", "normal", "iteration"}, 9},
};
constexpr std::array<std::string_view, StageCount> Entries{"query_evaluate",
                                                           "query_scan",
                                                           "query_compact",
                                                           "query_export",
                                                           "particle_collision_project",
                                                           "particle_collision_count",
                                                           "particle_collision_scatter",
                                                           "query_resume_casts"};
constexpr uint32_t Bit(Port port)
{
    return 1u << port;
}
constexpr std::array<uint32_t, StageCount> StagePorts{
    Bit(Parameters) | Bit(Inputs) | Bit(BodyIdentities) | Bit(Models) | Bit(Shapes) | Bit(States) | Bit(Candidates) |
        Bit(Spans) | Bit(CastParameters) | Bit(CastStates),
    Bit(Parameters) | Bit(Spans) | Bit(Summary),
    Bit(Parameters) | Bit(Candidates) | Bit(Spans) | Bit(Summary) | Bit(Hits),
    Bit(ExportParameters) | Bit(Hits) | Bit(Spans) | Bit(Summary) | Bit(ExportValues),
    Bit(ParticleParameters) | Bit(ParticleRows) | Bit(ParticlePrevious) | Bit(Hits) | Bit(ParticleRecords) |
        Bit(ParticleStatus),
    Bit(ParticleParameters) | Bit(ParticleRecords) | Bit(ParticleHeader) | Bit(ParticleOffsets) | Bit(ParticleStatus),
    Bit(ParticleParameters) | Bit(ParticleRecords) | Bit(ParticleHeader) | Bit(ParticleOffsets) | Bit(ParticleIndices) |
        Bit(ParticleStatus),
    Bit(Parameters) | Bit(Inputs) | Bit(BodyIdentities) | Bit(Models) | Bit(Shapes) | Bit(States) | Bit(Candidates) |
        Bit(Spans) | Bit(CastParameters) | Bit(CastStates)};

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
    const bool uniform =
        port == Parameters || port == ExportParameters || port == ParticleParameters || port == CastParameters;
    if (port == ExportValues || port == ParticleOffsets || port == ParticleIndices || port == ParticleStatus) {
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

void ValidateGeometry(uint32_t type, const Float4 &geometry, const Float4 &rotation)
{
    if (!Finite(geometry) || !Finite(rotation))
        throw JoltGpuQueryRequestError("Jolt GPU query geometry and rotation must be finite");
    const double length = double(rotation.x) * rotation.x + double(rotation.y) * rotation.y +
                          double(rotation.z) * rotation.z + double(rotation.w) * rotation.w;
    if (std::abs(length - 1.0) > 1.0e-3)
        throw JoltGpuQueryRequestError("Jolt GPU query rotation must be a unit quaternion");
    if (type == ShapeSphere) {
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

uint32_t OutputFormat(const QueryInput &query)
{
    return query.meta.x == QueryGatherShapes   ? QueryOutputShape
           : query.meta.x == QueryGatherMotion ? QueryOutputMotion
                                               : QueryOutputHit;
}

uint32_t RecordBytes(uint32_t format)
{
    return format == QueryOutputShape    ? sizeof(QueryShapeState)
           : format == QueryOutputMotion ? sizeof(QueryMotionState)
                                         : sizeof(QueryHit);
}

bool Zero(const Float4 &value)
{
    return value.x == 0 && value.y == 0 && value.z == 0 && value.w == 0;
}

void ValidateInput(const QueryInput &query, size_t primitiveCount, uint32_t bodyBase, uint32_t bodyCount,
                   bool hasPublication)
{
    if (query.meta.x == QueryGatherShapes || query.meta.x == QueryGatherMotion) {
        const bool motion = query.meta.x == QueryGatherMotion;
        if (!hasPublication || query.source.x != InvalidQueryBodySlot || query.source.w ||
            !Zero(query.positionMaximum) || !Zero(query.direction) || query.rotation.x || query.rotation.y ||
            query.rotation.z || query.rotation.w != 1 || !Finite(query.geometry))
            throw JoltGpuQueryRequestError(
                "Jolt GPU gather requires a retained publication and a valid reserved-field layout");
        if (motion) {
            if (query.meta.y != QueryClosest || query.meta.z || query.meta.w || query.filter.x || query.filter.y ||
                query.filter.z || query.filter.w ||
                (query.source.y < bodyBase || query.source.y - bodyBase >= bodyCount) || (query.source.z & ~0x3ffu) ||
                query.geometry.w)
                throw JoltGpuQueryRequestError(
                    "Jolt GPU motion gather requires one published body slot and requested field mask");
        } else {
            if (query.meta.y != QueryAll || (query.meta.z != QueryAllShapeTypes && query.meta.z > ShapeCapsule) ||
                (query.meta.w & ~QueryTargetCollider) || query.filter.y > 1 || query.source.z ||
                !Zero(query.geometry) ||
                (query.source.y != InvalidQueryBodySlot &&
                 (query.source.y < bodyBase || query.source.y - bodyBase >= bodyCount)))
                throw JoltGpuQueryRequestError("Jolt GPU shape gather requires an admitted body selection and filter");
        }
        return;
    }
    const bool cast = query.meta.x == QueryShapeCast;
    const bool sweep = cast && (query.meta.w & QuerySweepPolicy);
    if ((!cast && query.meta.x > QueryPenetration) || query.meta.y > QueryAny ||
        (query.meta.w & ~(QueryTargetCollider | QueryExcludeBody | (cast ? QuerySweepPolicy : 0u))) ||
        (sweep && (query.meta.w & QueryTargetCollider)) || query.filter.y > 1 ||
        (!sweep && (query.source.z || query.source.w)) || !Finite(query.positionMaximum) || !Finite(query.direction) ||
        !Finite(query.rotation) || !Finite(query.geometry) || query.direction.w != 0)
        throw JoltGpuQueryRequestError("Jolt GPU query contains an invalid operation, filter or finite input record");
    if ((query.meta.w & QueryExcludeBody) ? (query.source.x < bodyBase || query.source.x - bodyBase >= bodyCount)
                                          : query.source.x != InvalidQueryBodySlot)
        throw JoltGpuQueryRequestError("Jolt GPU query excluded body differs from its retained publication");
    if (query.meta.x == QueryPenetration) {
        if (query.source.y >= primitiveCount || query.meta.w || query.positionMaximum.w != 0)
            throw JoltGpuQueryRequestError(
                "Jolt GPU pair query requires an explicit second primitive and no world policy");
        ValidateGeometry(query.meta.z, query.geometry, query.rotation);
    } else {
        if (!hasPublication || query.source.y)
            throw JoltGpuQueryRequestError("Jolt GPU world query requires one retained publication");
        if (query.meta.x == QueryRay || cast) {
            if (query.positionMaximum.w <= 0 ||
                (query.direction.x == 0 && query.direction.y == 0 && query.direction.z == 0))
                throw JoltGpuQueryRequestError("Jolt GPU ray requires nonzero direction and positive maximum distance");
            if (cast)
                ValidateGeometry(query.meta.z, query.geometry, query.rotation);
        } else {
            if (query.positionMaximum.w != 0)
                throw JoltGpuQueryRequestError("Jolt GPU non-ray query maximum must be zero");
            if (query.meta.x == QueryOverlap)
                ValidateGeometry(query.meta.z, query.geometry, query.rotation);
            if (query.meta.x == QueryBodyBounds &&
                (query.direction.x < 0 || query.direction.y < 0 || query.direction.z < 0))
                throw JoltGpuQueryRequestError("Jolt GPU bounds query requires nonnegative half extents");
        }
    }
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
    if (stage == Evaluate && (port == Candidates || port == Spans || port == CastStates))
        return rhi::ComputeBufferAccess::Write;
    if (stage == ResumeCasts && (port == Candidates || port == Spans || port == CastStates))
        return rhi::ComputeBufferAccess::ReadWrite;
    if (stage == Scan && port == Spans)
        return rhi::ComputeBufferAccess::ReadWrite;
    if ((stage == Scan && port == Summary) || (stage == Compact && port == Hits) ||
        (stage == Export && port == ExportValues))
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
        if (target.target == ShaderCodeTarget::SpirV) {
            if (code->code.empty() || code->code.size() % sizeof(uint32_t))
                throw std::invalid_argument("Jolt GPU query target contains invalid SPIR-V");
            std::vector<uint32_t> words(code->code.size() / sizeof(uint32_t));
            std::memcpy(words.data(), code->code.data(), code->code.size());
            program.kernel = std::make_shared<rhi::ComputeKernel>(
                m_host, rhi::ShaderModuleDesc::FromSpirV(words.data(), words.size()), code->name, std::move(bindings),
                0);
        } else {
            program.kernel = std::make_shared<rhi::ComputeKernel>(
                m_host, rhi::ShaderModuleDesc::FromWgsl(code->code.data(), code->code.size()), code->name,
                std::move(bindings), 0);
        }
    }
}

struct JoltGpuQueries::Execution
{
    rhi::ComputeHost host;
    std::shared_ptr<JoltGpuQueryResult> result;
    std::vector<rhi::ComputeDispatchDesc> resume;
    std::shared_ptr<rhi::PreparedComputeBatch> packet;
    bool payloadInPacket;
};

JoltGpuControlWork JoltGpuQueries::Continue(const std::shared_ptr<Execution> &execution)
{
    const auto &result = execution->result;
    if (!result->summaryReadback || !result->summaryReadback->IsComplete())
        throw std::logic_error("Jolt GPU query continuation requires its completed receipt");
    QuerySummary summary{};
    result->summaryReadback->ReadInto(&summary, sizeof(summary));
    ValidateSummary(summary, result->queryCount, result->hitCapacity, result->knownCount, !execution->resume.empty());
    if (summary.counts.y == QueryPending) {
        // The device retains the current leaf, advancement and best hit. Only
        // metadata crosses the bus while another bounded slice is required.
        execution->packet = rhi::PrepareComputeBatch(execution->host, {}, execution->resume,
                                                     {{result->summary, 0, sizeof(QuerySummary)}});
        execution->payloadInPacket = false;
        result->summaryReadback = execution->packet->Readback(0);
        return {execution->packet, [execution]() { return Continue(execution); }};
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
    if (count && result->readback == JoltGpuQueryReadback::Host) {
        result->hitReadbackIndex = static_cast<uint32_t>(reads.size());
        reads.push_back({result->hits, 0, CheckedBytes(count, result->recordBytes)});
    }
    result->spanReadbackIndex = static_cast<uint32_t>(reads.size());
    reads.push_back({result->spans, 0, CheckedBytes(result->queryCount, sizeof(QuerySpan))});
    result->tailBatch = rhi::PrepareComputeBatch(execution->host, {}, {}, std::move(reads));
    return result->tailBatch;
}

JoltGpuPreparedQuery JoltGpuQueries::Prepare(std::shared_ptr<const JoltGpuScene::Publication> publication,
                                             const std::vector<QueryInput> &queries,
                                             const std::vector<QueryPrimitive> &primitives,
                                             uint64_t numericalGeneration, JoltGpuQueryReadback readback,
                                             uint32_t castIterationQuantum) const
{
    if (queries.size() > std::numeric_limits<uint32_t>::max() ||
        primitives.size() > std::numeric_limits<uint32_t>::max())
        throw JoltGpuQueryRequestError("Jolt GPU query input count exceeds device indexing");
    const auto inputBytes = CheckedBytes(queries.size(), sizeof(QueryInput));
    const auto primitiveBytes = CheckedBytes(primitives.size(), sizeof(QueryPrimitive));
    const auto requestWords = inputBytes / sizeof(QueryRecordWord) + primitiveBytes / sizeof(QueryRecordWord);
    if (requestWords > std::numeric_limits<uint32_t>::max())
        throw JoltGpuQueryRequestError("Jolt GPU query request words exceed device indexing");
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
        if (publication->actors.size() > UINT32_MAX || publication->leaves.size() > UINT32_MAX)
            throw JoltGpuQueryRequestError("GPU query author directory exceeds global indexing");
        totalBodies = static_cast<uint32_t>(publication->actors.size());
        totalLeaves = static_cast<uint32_t>(publication->leaves.size());
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
    for (const auto &primitive : primitives) {
        if (primitive.meta.y || primitive.meta.z || primitive.meta.w || !Finite(primitive.position) ||
            primitive.position.w)
            throw std::invalid_argument("Jolt GPU explicit query primitive has invalid pose or padding");
        ValidateGeometry(primitive.meta.x, primitive.geometry, primitive.rotation);
    }
    const uint32_t format = queries.empty() ? QueryOutputHit : OutputFormat(queries.front());
    const uint32_t recordBytes = RecordBytes(format);
    for (const auto &query : queries) {
        ValidateInput(query, primitives.size(), bodyBase, bodyCount, bool(publication));
        if (OutputFormat(query) != format)
            throw JoltGpuQueryRequestError("Jolt GPU query batch must use one typed output family");
        if ((query.meta.x == QueryGatherMotion || query.meta.x == QueryGatherShapes) &&
            query.source.y != InvalidQueryBodySlot && !publication->actors[query.source.y])
            throw JoltGpuQueryRequestError("Jolt GPU gather body slot has no retained actor identity");
    }
    const bool hasCasts =
        std::any_of(queries.begin(), queries.end(), [](const auto &query) { return query.meta.x == QueryShapeCast; });
    if (!castIterationQuantum)
        throw JoltGpuQueryRequestError("Jolt GPU cast iteration quantum must be positive");
    const bool knownCount = std::all_of(queries.begin(), queries.end(), [](const auto &query) {
        return query.meta.x == QueryGatherMotion ||
               ((query.meta.x <= QueryPenetration || query.meta.x == QueryShapeCast) && query.meta.y == QueryClosest);
    });
    const auto stride = knownCount ? 1u : std::max({1u, bodyCount, leafCount});
    if (queries.size() > std::numeric_limits<uint32_t>::max() / stride)
        throw JoltGpuQueryRequestError("Jolt GPU query scratch capacity exceeds device indexing");
    const auto capacity = static_cast<uint32_t>(queries.size()) * stride;
    const auto words = recordBytes / sizeof(QueryRecordWord);
    if (capacity > std::numeric_limits<uint32_t>::max() / words)
        throw JoltGpuQueryRequestError("Jolt GPU query record words exceed device indexing");
    auto result = std::make_shared<JoltGpuQueryResult>();
    result->publication = std::move(publication);
    result->numericalGeneration = numericalGeneration;
    result->queryCount = static_cast<uint32_t>(queries.size());
    result->hitCapacity = capacity;
    result->knownCount = knownCount;
    result->outputFormat = format;
    result->recordBytes = recordBytes;
    result->readback = readback;
    result->shapeBoxesOnly =
        format == QueryOutputShape &&
        std::all_of(queries.begin(), queries.end(), [](const auto &query) { return query.meta.z == ShapeBox; });
    if (queries.empty()) {
        result->headerComplete = result->materialized = result->recordsMaterialized = true;
        return {std::move(result), {}};
    }
    // Reject impossible dispatch dimensions before allocating request storage.
    const auto groups = [](uint32_t count) { return count / 64 + static_cast<uint32_t>(count % 64 != 0); };
    if (groups(capacity) > m_host.device.GetCapabilities().limits.maxComputeWorkgroupCount[0])
        throw JoltGpuQueryRequestError("Jolt GPU query capacity exceeds the device dispatch range");
    rhi::ComputeHost host = m_host;
    const auto allocate = [&](uint8_t port, uint64_t count) {
        const auto bytes = CheckedBytes(count, Ports[port].stride);
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
    const bool hasSweep = std::any_of(queries.begin(), queries.end(),
                                      [](const auto &query) { return (query.meta.w & QuerySweepPolicy) != 0; });
    const uint64_t identityCount = uint64_t(totalBodies) + (hasSweep ? totalLeaves : 0u);
    if (identityCount > std::numeric_limits<uint32_t>::max())
        throw JoltGpuQueryRequestError("Jolt GPU query identity words exceed device indexing");
    std::vector<QueryBodyIdentity> identities(static_cast<size_t>(std::max(uint64_t(1), identityCount)));
    identities.front().ids.x = InvalidQueryBodySlot;
    for (uint32_t slot = 0; slot < totalBodies; ++slot) {
        const auto actor = result->publication->actors[slot];
        const uint64_t rigidbodyId = actor ? actor->rigidbodyComponentId : 0;
        identities[slot].ids = {actor ? actor->bodyId : InvalidQueryBodySlot, rigidbodyId ? 1u : 0u,
                                static_cast<uint32_t>(rigidbodyId), static_cast<uint32_t>(rigidbodyId >> 32)};
    }
    if (hasSweep) {
        for (uint32_t leaf = 0; leaf < totalLeaves; ++leaf) {
            const auto &address = result->publication->leaves[leaf];
            const auto &actor = result->publication->actors[address.bodySlot];
            const auto id = actor->leaves[address.actorLeaf].surfaceGameObjectId;
            identities[totalBodies + leaf].ids = {static_cast<uint32_t>(id), static_cast<uint32_t>(id >> 32), 0, 0};
        }
    }
    upload(BodyIdentities, Bytes(identities.data(), identities.size()));
    const QueryCastParams castParameters{{castIterationQuantum, totalBodies, 0, 0}};
    upload(CastParameters, Bytes(&castParameters, 1));
    const QueryParams parameters{
        {result->queryCount, bodyBase, bodyCount, leafCount},
        {static_cast<uint32_t>(primitives.size()), stride, capacity, format},
        {result->publication ? result->publication->worldRow : 0u, bodyBase, leafBase,
         result->publication ? result->publication->jointBase : 0u},
        {bodyCount, leafCount, result->publication ? result->publication->jointCount : 0u, 0u}};
    upload(Parameters, Bytes(&parameters, 1));
    // Requests and explicit primitives share one immutable word stream. This
    // removes an upload/binding without changing either public input layout.
    std::vector<uint8_t> requestData(static_cast<size_t>(CheckedBytes(requestWords, sizeof(QueryRecordWord))));
    std::memcpy(requestData.data(), queries.data(), static_cast<size_t>(inputBytes));
    if (primitiveBytes)
        std::memcpy(requestData.data() + inputBytes, primitives.data(), static_cast<size_t>(primitiveBytes));
    upload(Inputs, std::move(requestData));
    allocate(Candidates, uint64_t(capacity) * words);
    allocate(Spans, result->queryCount);
    allocate(Summary, 1);
    allocate(Hits, uint64_t(capacity) * words);
    allocate(CastStates, result->queryCount);
    const auto dispatch = [&](uint8_t stage) {
        const auto &program = m_programs[stage];
        rhi::ComputeDispatchDesc dispatch;
        dispatch.kernel = program.kernel;
        dispatch.groupCountX =
            stage == Scan ? 1 : groups(stage == Evaluate || stage == ResumeCasts ? result->queryCount : capacity);
        rhi::ComputeDispatchProof proof;
        for (const auto port : program.ports) {
            const auto binding = static_cast<uint32_t>(dispatch.buffers.size());
            const auto access = Access(stage, port);
            dispatch.buffers.push_back(buffers[port]);
            dispatch.bufferAccesses.push_back(access);
            const rhi::ComputeBindingRange range{binding, 0, buffers[port]->GetByteSize()};
            if (static_cast<uint8_t>(access) & static_cast<uint8_t>(rhi::ComputeBufferAccess::Read))
                proof.reads.push_back(range);
            if (static_cast<uint8_t>(access) & static_cast<uint8_t>(rhi::ComputeBufferAccess::Write))
                proof.writes.push_back(range);
        }
        dispatch.proof = std::move(proof);
        return dispatch;
    };
    std::vector<rhi::ComputeDispatchDesc> dispatches{dispatch(Evaluate), dispatch(Scan), dispatch(Compact)};
    result->hits = buffers[Hits];
    result->spans = buffers[Spans];
    result->summary = buffers[Summary];
    std::vector<rhi::ComputeBufferRead> reads{{result->summary, 0, sizeof(QuerySummary)}};
    if (knownCount || readback == JoltGpuQueryReadback::Device) {
        if (readback == JoltGpuQueryReadback::Host) {
            result->hitReadbackIndex = static_cast<uint32_t>(reads.size());
            reads.push_back({result->hits, 0, CheckedBytes(result->queryCount, recordBytes)});
        }
        result->spanReadbackIndex = static_cast<uint32_t>(reads.size());
        reads.push_back({result->spans, 0, CheckedBytes(result->queryCount, sizeof(QuerySpan))});
    }
    auto batch = rhi::PrepareComputeBatch(host, std::move(updates), std::move(dispatches), std::move(reads));
    result->summaryReadback = batch->Readback(0);
    std::vector<rhi::ComputeDispatchDesc> resume;
    if (hasCasts)
        resume = {dispatch(ResumeCasts), dispatch(Scan), dispatch(Compact)};
    auto execution = std::make_shared<Execution>(
        Execution{host, result, std::move(resume), batch, knownCount || readback == JoltGpuQueryReadback::Device});
    JoltGpuControlWork work{batch, [execution]() { return Continue(execution); }};
    return {std::move(result), std::move(work)};
}

JoltGpuControlWork JoltGpuQueries::PrepareExport(const std::vector<std::shared_ptr<JoltGpuQueryResult>> &results,
                                                 const std::vector<JoltGpuQueryExportBinding> &bindings) const
{
    m_host.RequireAlive();
    for (const auto &binding : bindings)
        if (binding.field > JoltGpuQueryExportField::ShapeBounciness)
            throw JoltGpuQueryRequestError("Jolt GPU export has an unknown field");
    std::array<std::shared_ptr<JoltGpuQueryResult>, 3> sources;
    const JoltGpuScene::Publication *publication = nullptr;
    uint64_t generation = 0;
    bool captured = false;
    for (const auto &result : results) {
        if (!result)
            continue;
        if (captured && (publication != result->publication.get() || generation != result->numericalGeneration))
            throw JoltGpuQueryRequestError(
                "Jolt GPU typed exports must retain one publication and numerical generation");
        publication = result->publication.get();
        generation = result->numericalGeneration;
        captured = true;
        if (result->queryCount && (!result->hits || !m_host.SharesServicesWith(result->hits->GetHost())))
            throw JoltGpuQueryRequestError("Jolt GPU export records belong to another compute owner");
        if (result->GetSummary().counts.y != QuerySuccess)
            return {}; // The completed result consumer reports the read-only failure.
        result->Materialize();
        if (!result->queryCount)
            continue;
        if (result->outputFormat != QueryOutputMotion && result->outputFormat != QueryOutputShape)
            throw JoltGpuQueryRequestError("Jolt GPU exports require motion or shape record sources");
        if (sources[result->outputFormat])
            throw JoltGpuQueryRequestError("Jolt GPU export batch repeats a typed source family");
        sources[result->outputFormat] = result;
    }
    if ((!sources[QueryOutputMotion] && !sources[QueryOutputShape]) || bindings.empty())
        return {};
    constexpr std::array<uint32_t, 13> Widths{3, 4, 3, 3, 3, 3, 9, 1, 3, 3, 4, 1, 1};
    struct Target
    {
        std::shared_ptr<JoltGpuQueryResult> source;
        std::shared_ptr<rhi::ComputeBuffer> buffer;
        uint64_t offset, bytes;
        uint32_t field, count, first, bodyIndex;
    };
    std::vector<Target> targets;
    for (const auto &binding : bindings) {
        const auto field = static_cast<uint32_t>(binding.field);
        if (field >= Widths.size() || !binding.buffer || binding.buffer->IsFrozen() ||
            !m_host.SharesServicesWith(binding.buffer->GetHost()) || binding.byteOffset % sizeof(uint32_t))
            throw JoltGpuQueryRequestError("Jolt GPU export field or destination differs from its compute owner");
        const bool motion = field <= static_cast<uint32_t>(JoltGpuQueryExportField::MotionInverseInertia);
        const auto &source = sources[motion ? QueryOutputMotion : QueryOutputShape];
        if (!source)
            throw JoltGpuQueryRequestError("Jolt GPU export has no matching typed query source");
        if (binding.field == JoltGpuQueryExportField::ShapeHalfExtents && !source->shapeBoxesOnly)
            throw JoltGpuQueryRequestError("Jolt GPU box export requires an exclusively Box selection");
        const uint32_t total = source->summaryValue.counts.x;
        const uint32_t first = binding.recordFirst;
        const uint32_t count = binding.recordCount == UINT32_MAX ? total - std::min(first, total) : binding.recordCount;
        if (first > total || count > total - first)
            throw JoltGpuQueryRequestError("GPU export selection exceeds its completed typed source");
        const auto &desc = binding.buffer->GetDesc();
        const auto scalar = binding.field == JoltGpuQueryExportField::ShapeBodyIndex ? rhi::ComputeScalarType::Int32
                                                                                     : rhi::ComputeScalarType::Float32;
        const auto lanes = Widths[field] == 9 ? 1u : Widths[field];
        const auto bytes = CheckedBytes(uint64_t(count) * Widths[field], sizeof(uint32_t));
        if (desc.scalarType != scalar || desc.lanes != lanes || binding.byteOffset > binding.buffer->GetByteSize() ||
            bytes > binding.buffer->GetByteSize() - binding.byteOffset ||
            binding.byteOffset / sizeof(uint32_t) > std::numeric_limits<uint32_t>::max() ||
            uint64_t(count) * Widths[field] >
                std::numeric_limits<uint32_t>::max() - binding.byteOffset / sizeof(uint32_t))
            throw JoltGpuQueryRequestError(
                "Jolt GPU export destination has an invalid type, prefix capacity or index range");
        for (const auto &input : sources) {
            if (!input)
                continue;
            for (const auto &buffer : {input->hits, input->spans, input->summary})
                if (binding.buffer->GetAllocation() == buffer->GetAllocation())
                    throw JoltGpuQueryRequestError("Jolt GPU export destination must not alias query input records");
        }
        for (const auto &previous : targets)
            if ((binding.buffer->GetWriterStateIdentity() == previous.buffer->GetWriterStateIdentity() ||
                 binding.buffer->GetAllocation() == previous.buffer->GetAllocation()) &&
                bytes && previous.bytes && binding.byteOffset < previous.offset + previous.bytes &&
                previous.offset < binding.byteOffset + bytes)
                throw JoltGpuQueryRequestError(
                    "Jolt GPU export output prefixes must not overlap across typed families");
        const uint32_t groups = count / 64 + static_cast<uint32_t>(count % 64 != 0);
        if (groups > m_host.device.GetCapabilities().limits.maxComputeWorkgroupCount[0])
            throw JoltGpuQueryRequestError("Jolt GPU export count exceeds the device dispatch range");
        targets.push_back({source, binding.buffer, binding.byteOffset, bytes, field, count, first, binding.bodyIndex});
    }
    rhi::ComputeHost host = m_host;
    std::vector<rhi::ComputeBufferUpdate> updates;
    std::vector<rhi::ComputeDispatchDesc> dispatches;
    const auto &program = m_programs[Export];
    for (const auto &target : targets) {
        if (!target.count)
            continue;
        const auto &source = target.source;
        const QueryExportParams parameters{
            {source->outputFormat, target.field, target.count, static_cast<uint32_t>(target.offset / sizeof(uint32_t))},
            {source->queryCount, source->recordBytes / 16, Widths[target.field], 0},
            {target.first, target.bodyIndex, target.bodyIndex != UINT32_MAX ? 1u : 0u, 0u}};
        auto uniform = rhi::ComputeBuffer::TryCreatePrivate(
            host, {sizeof(parameters) / sizeof(uint32_t), rhi::ComputeScalarType::UInt32, 1});
        if (!uniform)
            throw JoltGpuQueryRequestError("Jolt GPU export preparation exceeds available managed device capacity");
        updates.push_back({uniform, 0, Bytes(&parameters, 1)});
        std::array<std::shared_ptr<rhi::ComputeBuffer>, PortCount> buffers;
        buffers[ExportParameters] = uniform;
        buffers[Hits] = source->hits;
        buffers[Spans] = source->spans;
        buffers[Summary] = source->summary;
        buffers[ExportValues] = target.buffer;
        rhi::ComputeDispatchDesc dispatch;
        dispatch.kernel = program.kernel;
        dispatch.groupCountX = target.count / 64 + static_cast<uint32_t>(target.count % 64 != 0);
        rhi::ComputeDispatchProof proof;
        for (const auto port : program.ports) {
            const auto local = static_cast<uint32_t>(dispatch.buffers.size());
            dispatch.buffers.push_back(buffers[port]);
            dispatch.bufferAccesses.push_back(Access(Export, port));
            if (port == ExportValues)
                proof.writes.push_back({local, target.offset, target.bytes});
            else
                proof.reads.push_back({local, 0, buffers[port]->GetByteSize()});
        }
        dispatch.proof = std::move(proof);
        dispatches.push_back(std::move(dispatch));
    }
    if (dispatches.empty())
        return {};
    return rhi::PrepareComputeBatch(host, std::move(updates), std::move(dispatches));
}

JoltGpuPreparedParticleExport JoltGpuQueries::PrepareParticleExport(const std::shared_ptr<JoltGpuQueryResult> &shapes,
                                                                    const std::vector<JoltGpuParticleExportRow> &rows,
                                                                    std::shared_ptr<rhi::ComputeBuffer> previousRecords,
                                                                    uint32_t staticCount, uint64_t revision,
                                                                    uint64_t topologyRevision) const
{
    m_host.RequireAlive();
    if (!shapes || (shapes->queryCount && (!shapes->hits || !m_host.SharesServicesWith(shapes->hits->GetHost()))))
        throw JoltGpuQueryRequestError("GPU particle export requires shape records from its compute owner");
    auto result = std::make_shared<JoltGpuParticleExportResult>();
    result->source = shapes;
    result->payload = std::make_shared<particle::GpuParticleCollisionDevicePayload>();
    const auto &summary = shapes->GetSummary();
    if (summary.counts.y != QuerySuccess)
        return {std::move(result), {}};
    (void)shapes->ReadSpans();
    if ((shapes->queryCount &&
         (shapes->outputFormat != QueryOutputShape || shapes->recordBytes != sizeof(QueryShapeState))) ||
        rows.size() > std::numeric_limits<uint32_t>::max() / 512u || staticCount > rows.size() || !topologyRevision)
        throw JoltGpuQueryRequestError("GPU particle export shape format, row domain or topology is invalid");
    const uint32_t count = static_cast<uint32_t>(rows.size());
    if (count > std::numeric_limits<uint32_t>::max() / 21u)
        throw JoltGpuQueryRequestError("GPU particle collider words exceed device indexing");
    uint32_t previousCount = 0;
    if (previousRecords) {
        if (!previousRecords->IsFrozen() || !m_host.SharesServicesWith(previousRecords->GetHost()) ||
            !previousRecords->GetByteSize() ||
            previousRecords->GetByteSize() % sizeof(particle::GpuParticleColliderRecord))
            throw JoltGpuQueryRequestError("GPU particle history requires frozen records from the same compute owner");
        const auto capacity = previousRecords->GetByteSize() / sizeof(particle::GpuParticleColliderRecord);
        if (capacity > std::numeric_limits<uint32_t>::max() / 21u ||
            !previousRecords->HasInitializedRange(0, previousRecords->GetByteSize()))
            throw JoltGpuQueryRequestError("GPU particle history exceeds its initialized record range");
        previousCount = static_cast<uint32_t>(capacity);
    }
    for (const auto &row : rows)
        if (row.selection.x >= summary.counts.x || row.selection.z || row.selection.w ||
            (row.selection.y != InvalidQueryBodySlot && row.selection.y >= previousCount))
            throw JoltGpuQueryRequestError("GPU particle row or previous identity map exceeds its retained source");
    auto &payload = *result->payload;
    payload.rowCount = count;
    payload.staticCount = staticCount;
    payload.revision = revision;
    payload.topologyRevision = topologyRevision;
    payload.retained = shapes;
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
    const ParticleExportParameters parameters{{count, staticCount, summary.counts.x, previousCount},
                                              {static_cast<uint32_t>(revision), static_cast<uint32_t>(revision >> 32),
                                               static_cast<uint32_t>(topologyRevision),
                                               static_cast<uint32_t>(topologyRevision >> 32)}};
    upload(ParticleParameters, Bytes(&parameters, 1));
    if (rows.empty()) {
        const JoltGpuParticleExportRow empty{};
        upload(ParticleRows, Bytes(&empty, 1));
    } else
        upload(ParticleRows, Bytes(rows.data(), rows.size()));
    buffers[Hits] = shapes->hits;
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
        batch, [host, result, buffers, scatter]() mutable -> JoltGpuControlWork {
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
            return result->batch;
        }};
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
    RequireSuccess(source->GetSummary().counts.y);
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
    if (!tailBatch || !tailBatch->GetSubmission().IsValid() || !tailBatch->Readback(spanReadbackIndex)->IsComplete())
        throw std::logic_error("Jolt GPU query metadata requires its completed receipt");
    const auto count = summaryValue.counts.x;
    hostSpans.resize(queryCount);
    tailBatch->Readback(spanReadbackIndex)->ReadInto(hostSpans.data(), CheckedBytes(queryCount, sizeof(QuerySpan)));
    uint32_t end = 0;
    for (const auto &span : hostSpans) {
        RequireSuccess(span.range.z);
        if (span.range.x != end || span.range.y > count - end || span.range.w || (knownCount && span.range.y != 1))
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
void ValidateRecord(const QueryShapeState &shape)
{
    if (shape.meta.y != 1 || (shape.meta.w & ~(QueryShapeSensor | QueryShapeMoving)) || !Finite(shape.geometry) ||
        !Finite(shape.center) || !Finite(shape.rotation) || !Finite(shape.bodyCenterOfMass) ||
        !Finite(shape.linearVelocity) || !Finite(shape.angularVelocity) || !Finite(shape.material) || shape.center.w ||
        shape.bodyCenterOfMass.w || shape.linearVelocity.w || shape.angularVelocity.w || shape.material.z ||
        shape.material.w)
        throw std::runtime_error("Jolt GPU query returned an invalid or nonfinite shape record");
}
void ValidateRecord(const QueryMotionState &motion)
{
    if (motion.meta.y != 1 || !Finite(motion.positionInverseMass) || !Finite(motion.rotation) ||
        !Finite(motion.centerOfMass) || !Finite(motion.linearVelocity) || !Finite(motion.angularVelocity) ||
        !Finite(motion.inverseInertia0) || !Finite(motion.inverseInertia1) || !Finite(motion.inverseInertia2) ||
        !Finite(motion.inertia0) || !Finite(motion.inertia1) || !Finite(motion.inertia2) ||
        motion.positionInverseMass.w < 0 || motion.centerOfMass.w || motion.linearVelocity.w ||
        motion.angularVelocity.w || motion.inverseInertia0.w || motion.inverseInertia1.w || motion.inverseInertia2.w ||
        motion.inertia0.w || motion.inertia1.w || motion.inertia2.w)
        throw std::runtime_error("Jolt GPU query returned an invalid or nonfinite motion record");
}
} // namespace

template <class T> const std::vector<T> &JoltGpuQueryResult::ReadTyped(uint32_t format, std::vector<T> &records)
{
    if (!queryCount)
        return records;
    if (outputFormat != format || recordBytes != sizeof(T))
        throw std::logic_error("Jolt GPU query typed accessor differs from its record format");
    if (readback != JoltGpuQueryReadback::Host)
        throw std::logic_error("Jolt GPU device query does not materialize numerical records");
    Materialize();
    if (!recordsMaterialized) {
        const auto count = summaryValue.counts.x;
        if (count && !tailBatch->Readback(hitReadbackIndex)->IsComplete())
            throw std::logic_error("Jolt GPU query records require their completed receipt");
        records.resize(count);
        if (count)
            tailBatch->Readback(hitReadbackIndex)->ReadInto(records.data(), CheckedBytes(count, sizeof(T)));
        for (const auto &record : records)
            ValidateRecord(record);
        recordsMaterialized = true;
    }
    return records;
}

const QuerySummary &JoltGpuQueryResult::GetSummary()
{
    ReadSummary();
    return summaryValue;
}

const std::vector<QueryHit> &JoltGpuQueryResult::ReadHits()
{
    return ReadTyped(QueryOutputHit, hostHits);
}
const std::vector<QueryShapeState> &JoltGpuQueryResult::ReadShapes()
{
    return ReadTyped(QueryOutputShape, hostShapes);
}
const std::vector<QueryMotionState> &JoltGpuQueryResult::ReadMotion()
{
    return ReadTyped(QueryOutputMotion, hostMotion);
}
const std::vector<QuerySpan> &JoltGpuQueryResult::ReadSpans()
{
    Materialize();
    return hostSpans;
}

} // namespace infernux
