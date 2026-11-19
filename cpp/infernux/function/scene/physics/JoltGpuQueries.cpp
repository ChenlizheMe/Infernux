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
bool IsDeviceResult(JoltGpuQueryReadback destination)
{
    return destination == JoltGpuQueryReadback::Device || destination == JoltGpuQueryReadback::DeviceWithSpans;
}
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
    Geometry,
    RayParameters,
    RayOrigins,
    RayDirections,
    RayInputs,
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
    InitJobs,
    AdvanceJobs,
    JobProgress,
    ReadMotion,
    InitRays,
    StageCount
};
struct RayInputParameters
{
    UInt4 source; // Count, origin scalar offset, direction scalar offset, zero.
    UInt4 filter;
    Float4 maximum;
};
static_assert(sizeof(RayInputParameters) == 48);
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
    {"query_export_params", sizeof(QueryExportParams), {"output", "sizes", "selection"}, 7},
    {"query_export_values", sizeof(uint32_t), {}, 0},
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
    {"query_ray_params", sizeof(RayInputParameters), {"source", "filter", "maximum"}, 3},
    {"query_ray_origins", sizeof(float), {}, 0},
    {"query_ray_directions", sizeof(float), {}, 0},
    {"query_ray_inputs", sizeof(QueryRecordWord), {"value"}, 1},
};
constexpr std::array<std::string_view, StageCount> Entries{"query_evaluate",
                                                           "query_scan",
                                                           "query_compact",
                                                           "query_export",
                                                           "particle_collision_project",
                                                           "particle_collision_count",
                                                           "particle_collision_scatter",
                                                           "query_resume_casts",
                                                           "query_init_jobs",
                                                           "query_advance_jobs",
                                                           "query_job_progress",
                                                           "query_read_motion",
                                                           "query_init_rays"};
constexpr uint32_t Bit(Port port)
{
    return 1u << port;
}
constexpr std::array<uint32_t, StageCount> StagePorts{
    Bit(Parameters) | Bit(Inputs) | Bit(BodyIdentities) | Bit(Models) | Bit(Shapes) | Bit(States) | Bit(Candidates) |
        Bit(Control) | Bit(CastParameters) | Bit(Geometry),
    Bit(Parameters) | Bit(Control) | Bit(Summary),
    Bit(Parameters) | Bit(Candidates) | Bit(Control) | Bit(Summary) | Bit(Hits),
    Bit(ExportParameters) | Bit(Hits) | Bit(Control) | Bit(Summary) | Bit(ExportValues),
    Bit(ParticleParameters) | Bit(ParticleRows) | Bit(ParticlePrevious) | Bit(Shapes) | Bit(Hits) |
        Bit(ParticleRecords) | Bit(ParticleStatus),
    Bit(ParticleParameters) | Bit(ParticleRecords) | Bit(ParticleHeader) | Bit(ParticleOffsets) | Bit(ParticleStatus),
    Bit(ParticleParameters) | Bit(ParticleRecords) | Bit(ParticleHeader) | Bit(ParticleOffsets) | Bit(ParticleIndices) |
        Bit(ParticleStatus),
    Bit(Parameters) | Bit(Inputs) | Bit(BodyIdentities) | Bit(Models) | Bit(Shapes) | Bit(States) | Bit(Candidates) |
        Bit(Control) | Bit(CastParameters),
    Bit(Parameters) | Bit(Control),
    Bit(Parameters) | Bit(Inputs) | Bit(BodyIdentities) | Bit(Models) | Bit(Shapes) | Bit(States) | Bit(Candidates) |
        Bit(Control) | Bit(CastParameters) | Bit(Geometry),
    Bit(Parameters) | Bit(Control) | Bit(Summary),
    Bit(Parameters) | Bit(Inputs) | Bit(Models) | Bit(States) | Bit(Control) | Bit(Hits),
    Bit(RayParameters) | Bit(RayOrigins) | Bit(RayDirections) | Bit(RayInputs)};

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
    const bool uniform = port == Parameters || port == ExportParameters || port == ParticleParameters ||
                         port == CastParameters || port == RayParameters;
    if (port == RayOrigins || port == RayDirections) {
        if (resource.binding.type != rhi::BindingType::StorageBuffer || !resource.binding.readOnlyStorage ||
            resource.elementStride != sizeof(float) || resource.value.kind != ShaderValueKind::Scalar ||
            resource.value.scalar != ShaderScalarKind::Float32 || resource.value.byteSize != sizeof(float))
            throw std::invalid_argument("Jolt GPU ray input differs from its scalar float storage layout");
        return;
    }
    if (port == Control || port == Geometry) {
        if (resource.binding.type != rhi::BindingType::StorageBuffer ||
            resource.binding.readOnlyStorage != (port == Geometry) || resource.elementStride != sizeof(UInt4) ||
            resource.value.kind != ShaderValueKind::Vector || resource.value.scalar != ShaderScalarKind::UInt32 ||
            resource.value.byteSize != sizeof(UInt4) || resource.value.rows != 1 || resource.value.columns != 4)
            throw std::invalid_argument("Jolt GPU query uint4 stream differs from its shared layout");
        return;
    }
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

void ValidateGeometry(uint32_t type, const Float4 &geometry, const Float4 &rotation, const UInt4 &reference,
                      const std::shared_ptr<const JoltGpuGeometryLease> &lease, uint32_t flags = 0)
{
    const uint32_t triangleFlags = ShapeOneSidedTriangle | (7u << ShapeTriangleActiveEdgeShift);
    const bool triangle = (flags & ShapeOneSidedTriangle) != 0u;
    if ((flags & ~(ShapeSensor | triangleFlags)) || (!triangle && (flags & triangleFlags)) ||
        (triangle && type != ShapeConvexHull))
        throw JoltGpuQueryRequestError("Jolt GPU query primitive has invalid shape flags");
    if (!Finite(geometry) || !Finite(rotation))
        throw JoltGpuQueryRequestError("Jolt GPU query geometry and rotation must be finite");
    const double length = double(rotation.x) * rotation.x + double(rotation.y) * rotation.y +
                          double(rotation.z) * rotation.z + double(rotation.w) * rotation.w;
    if (std::abs(length - 1.0) > 1.0e-3)
        throw JoltGpuQueryRequestError("Jolt GPU query rotation must be a unit quaternion");
    if (type == ShapeConvexHull) {
        if (geometry.x < 0 || (!triangle && (geometry.y || geometry.z || geometry.w)) || !reference.x ||
            reference.y < 5 || !lease || !lease->Contains(reference))
            throw JoltGpuQueryRequestError("Jolt GPU Hull requires its exact retained geometry range");
        if (triangle) {
            const auto entry = std::find_if(lease->records.begin(), lease->records.end(), [&](const auto &value) {
                return value && value->reference.x == reference.x && value->reference.y == reference.y;
            });
            const auto *record = entry == lease->records.end() ? nullptr : (*entry)->record.get();
            if (geometry.x != 0 || !record || record->convexRadius != 0 || record->words.size() < 5 ||
                record->words[0].z != 3u || record->words[0].w != 2u ||
                std::abs(geometry.y * geometry.y + geometry.z * geometry.z + geometry.w * geometry.w - 1.0f) > 1e-4f)
                throw JoltGpuQueryRequestError(
                    "Jolt GPU triangle requires its exact planar three-point hull and CCW unit normal");
        }
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
                   bool hasPublication, const std::shared_ptr<const JoltGpuGeometryLease> &lease)
{
    const bool hasReference = query.geometryReference.x || query.geometryReference.y || query.geometryReference.z ||
                              query.geometryReference.w;
    if (hasReference && query.meta.x != QueryShapeCast && query.meta.x != QueryPenetration &&
        query.meta.x != QueryOverlap)
        throw JoltGpuQueryRequestError("Jolt GPU query operation has no explicit geometry operand");
    const bool hasLifetime = query.lifetime.x || query.lifetime.y;
    if (query.lifetime.z || query.lifetime.w)
        throw JoltGpuQueryRequestError("GPU query source lifetime reserved words must be zero");
    if (query.meta.x == QueryGatherShapes || query.meta.x == QueryGatherMotion) {
        const bool motion = query.meta.x == QueryGatherMotion;
        if (!hasPublication || hasLifetime || query.source.x != InvalidQueryBodySlot || query.source.w ||
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
            if (query.meta.y != QueryAll || (query.meta.z != QueryAllShapeTypes && query.meta.z > ShapeConvexHull) ||
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
        (query.meta.w &
         ~(QueryTargetCollider | QueryExcludeBody | (cast ? (QuerySweepPolicy | QuerySweepPairPolicy) : 0u))) ||
        (sweep && (query.meta.w & QueryTargetCollider)) || ((query.meta.w & QuerySweepPairPolicy) && !sweep) ||
        query.filter.y > 1 || (!sweep && (query.source.z || query.source.w || hasLifetime)) ||
        ((query.meta.w & QuerySweepPairPolicy) && !hasLifetime) || !Finite(query.positionMaximum) ||
        !Finite(query.direction) || !Finite(query.rotation) || !Finite(query.geometry) || query.direction.w != 0)
        throw JoltGpuQueryRequestError("Jolt GPU query contains an invalid operation, filter or finite input record");
    if ((query.meta.w & QueryExcludeBody) ? (query.source.x < bodyBase || query.source.x - bodyBase >= bodyCount)
                                          : query.source.x != InvalidQueryBodySlot)
        throw JoltGpuQueryRequestError("Jolt GPU query excluded body differs from its retained publication");
    if (query.meta.x == QueryPenetration) {
        if (query.source.y >= primitiveCount || query.meta.w || query.positionMaximum.w != 0)
            throw JoltGpuQueryRequestError(
                "Jolt GPU pair query requires an explicit second primitive and no world policy");
        ValidateGeometry(query.meta.z, query.geometry, query.rotation, query.geometryReference, lease);
    } else {
        if (!hasPublication || query.source.y)
            throw JoltGpuQueryRequestError("Jolt GPU world query requires one retained publication");
        if (query.meta.x == QueryRay || cast) {
            if (query.positionMaximum.w <= 0 ||
                (query.direction.x == 0 && query.direction.y == 0 && query.direction.z == 0))
                throw JoltGpuQueryRequestError("Jolt GPU ray requires nonzero direction and positive maximum distance");
            if (cast)
                ValidateGeometry(query.meta.z, query.geometry, query.rotation, query.geometryReference, lease);
        } else {
            if (query.positionMaximum.w != 0)
                throw JoltGpuQueryRequestError("Jolt GPU non-ray query maximum must be zero");
            if (query.meta.x == QueryOverlap)
                ValidateGeometry(query.meta.z, query.geometry, query.rotation, query.geometryReference, lease);
            if (query.meta.x == QueryBodyBounds &&
                (query.direction.x < 0 || query.direction.y < 0 || query.direction.z < 0))
                throw JoltGpuQueryRequestError("Jolt GPU bounds query requires nonnegative half extents");
        }
    }
}

rhi::ComputeBufferAccess Access(uint8_t stage, uint8_t port)
{
    if (stage == InitRays && port == RayInputs)
        return rhi::ComputeBufferAccess::Write;
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
    if (stage == ReadMotion && (port == Hits || port == Control))
        return rhi::ComputeBufferAccess::Write;
    if (stage == Evaluate && (port == Candidates || port == Control))
        return rhi::ComputeBufferAccess::Write;
    if (stage == ResumeCasts && (port == Candidates || port == Control))
        return rhi::ComputeBufferAccess::ReadWrite;
    if (stage == Scan && port == Control)
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
    if (count && !IsDeviceResult(result->readback)) {
        result->hitReadbackIndex = static_cast<uint32_t>(reads.size());
        reads.push_back({result->hits, 0, CheckedBytes(count, result->recordBytes)});
    }
    if (result->NeedsHostSpans()) {
        result->spanReadbackIndex = static_cast<uint32_t>(reads.size());
        reads.push_back({result->spans, 0, CheckedBytes(result->queryCount, sizeof(QuerySpan))});
    }
    if (reads.empty()) {
        result->tailBatch = execution->packet;
        return {};
    }
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
                                             const std::vector<QueryInput> &queries,
                                             const std::vector<QueryPrimitive> &primitives,
                                             uint64_t numericalGeneration, JoltGpuQueryReadback readback,
                                             uint32_t castIterationQuantum, JoltGpuQueryGeometry geometry,
                                             std::optional<JoltGpuQueryRayInputs> rays) const
{
    auto result = std::make_shared<JoltGpuQueryResult>();
    result->explicitShapes = std::move(geometry.shapes);
    if (publication) {
        if (geometry.store && geometry.store != publication->geometryStore)
            throw JoltGpuQueryRequestError("Explicit and published geometry must share one compatible arena");
        geometry.store = publication->geometryStore;
    }
    if (geometry.operands.empty() && publication && publication->geometry) {
        result->geometry = publication->geometry;
        return PrepareCaptured(std::move(publication), queries, primitives, numericalGeneration, readback,
                               castIterationQuantum, result, rays);
    }
    if (!geometry.store)
        return PrepareCaptured(std::move(publication), queries, primitives, numericalGeneration, readback,
                               castIterationQuantum, result, rays);
    if (publication && publication->geometry)
        geometry.operands.insert(geometry.operands.end(), publication->geometry->records.begin(),
                                 publication->geometry->records.end());
    // The immutable handles pin both explicit operands before capture begins.
    // Store work uses the same writer budget/receipts; no synchronous upload.
    auto service = std::make_shared<JoltGpuQueries>(*this);
    auto capture = geometry.store->PrepareCapture(
        std::move(geometry.operands),
        [result](std::shared_ptr<const JoltGpuGeometryLease> lease) { result->geometry = std::move(lease); });
    auto next = [service, result, publication = std::move(publication), queries, primitives, numericalGeneration,
                 readback, castIterationQuantum, rays]() {
        return service
            ->PrepareCaptured(publication, queries, primitives, numericalGeneration, readback, castIterationQuantum,
                              result, rays)
            .work;
    };
    return {result, ContinueJoltGpuControlWork(std::move(capture), std::move(next))};
}

JoltGpuPreparedQuery JoltGpuQueries::PrepareCaptured(std::shared_ptr<const JoltGpuScene::Publication> publication,
                                                     const std::vector<QueryInput> &queries,
                                                     const std::vector<QueryPrimitive> &primitives,
                                                     uint64_t numericalGeneration, JoltGpuQueryReadback readback,
                                                     uint32_t castIterationQuantum,
                                                     const std::shared_ptr<JoltGpuQueryResult> &result,
                                                     const std::optional<JoltGpuQueryRayInputs> &rays) const
{
    const uint64_t queryCount = rays ? rays->count : queries.size();
    if (queryCount > std::numeric_limits<uint32_t>::max() || primitives.size() > std::numeric_limits<uint32_t>::max())
        throw JoltGpuQueryRequestError("Jolt GPU query input count exceeds device indexing");
    if (rays && (queries.size() != 1 || queries.front().meta.x != QueryRay || queries.front().meta.y != QueryClosest ||
                 !primitives.empty()))
        throw JoltGpuQueryRequestError("GPU ray inputs require one closest-ray template without explicit primitives");
    const uint32_t format = queries.empty() ? QueryOutputHit : OutputFormat(queries.front());
    const bool motion = format == QueryOutputMotion;
    const auto inputBytes = CheckedBytes(queryCount, motion ? sizeof(QueryMotionInput) : sizeof(QueryInput));
    const auto primitiveBytes = CheckedBytes(primitives.size(), sizeof(QueryPrimitive));
    const auto requestWords = inputBytes / sizeof(QueryRecordWord) + primitiveBytes / sizeof(QueryRecordWord);
    if (requestWords > std::numeric_limits<uint32_t>::max())
        throw JoltGpuQueryRequestError("Jolt GPU query request words exceed device indexing");
    if (rays && queryCount) {
        for (const auto *view : {&rays->origins, &rays->directions}) {
            const auto &buffer = view->buffer;
            const auto bytes = CheckedBytes(queryCount, 3u * sizeof(float));
            if (!buffer || !m_host.SharesServicesWith(buffer->GetHost()) ||
                buffer->GetDesc().scalarType != rhi::ComputeScalarType::Float32 || buffer->GetDesc().lanes != 3 ||
                view->byteOffset % sizeof(float))
                throw JoltGpuQueryRequestError("GPU ray inputs require float3 storage from the query compute owner");
            const auto end = std::min(view->byteEnd, buffer->GetByteSize());
            if (view->byteOffset > end || bytes > end - view->byteOffset ||
                view->byteOffset / sizeof(float) > UINT32_MAX ||
                bytes / sizeof(float) > UINT32_MAX - view->byteOffset / sizeof(float))
                throw JoltGpuQueryRequestError("GPU ray input view exceeds its storage or shader indexing range");
        }
    }
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
    for (const auto &primitive : primitives) {
        if (primitive.meta.z || primitive.meta.w || !Finite(primitive.position) || primitive.position.w)
            throw std::invalid_argument("Jolt GPU explicit query primitive has invalid pose or padding");
        ValidateGeometry(primitive.meta.x, primitive.geometry, primitive.rotation, primitive.geometryReference,
                         result->geometry, primitive.meta.y);
    }
    const uint32_t recordBytes = RecordBytes(format);
    for (const auto &query : queries) {
        ValidateInput(query, primitives.size(), bodyBase, bodyCount, bool(publication), result->geometry);
        if (OutputFormat(query) != format)
            throw JoltGpuQueryRequestError("Jolt GPU query batch must use one typed output family");
        if ((query.meta.x == QueryGatherMotion || query.meta.x == QueryGatherShapes) &&
            query.source.y != InvalidQueryBodySlot && !publication->directory->actors[query.source.y])
            throw JoltGpuQueryRequestError("Jolt GPU gather body slot has no retained actor identity");
    }
    const bool geometric = std::any_of(queries.begin(), queries.end(), [](const auto &query) {
        return query.meta.x == QueryRay || query.meta.x == QueryOverlap || query.meta.x == QueryClosestPoint ||
               query.meta.x == QueryPenetration || query.meta.x == QueryShapeCast;
    });
    bool generic = geometric && std::any_of(primitives.begin(), primitives.end(), [](const auto &primitive) {
                       return primitive.meta.x == ShapeCylinder || primitive.meta.x == ShapeConvexHull;
                   });
    generic |= geometric && std::any_of(queries.begin(), queries.end(), [](const auto &query) {
                   return query.meta.z == ShapeCylinder || query.meta.z == ShapeConvexHull;
               });
    // Producer captures this bit in its existing accepted-directory loop.
    generic |= geometric && publication && publication->hasGenericGeometry;
    if (!queries.empty() && (!result->geometry || !result->geometry->buffer ||
                             !m_host.SharesServicesWith(result->geometry->buffer->GetHost())))
        throw JoltGpuQueryRequestError("GPU query requires its compatible initialized geometry lease");
    const bool hasCasts =
        std::any_of(queries.begin(), queries.end(), [](const auto &query) { return query.meta.x == QueryShapeCast; });
    if (!castIterationQuantum)
        throw JoltGpuQueryRequestError("Jolt GPU cast iteration quantum must be positive");
    const bool knownCount = std::all_of(queries.begin(), queries.end(), [](const auto &query) {
        return query.meta.x == QueryGatherMotion ||
               ((query.meta.x <= QueryPenetration || query.meta.x == QueryShapeCast) && query.meta.y == QueryClosest);
    });
    const bool shapeGather = format == QueryOutputShape;
    uint32_t stride = knownCount ? 1u : std::max({1u, bodyCount, leafCount});
    uint64_t records = queryCount * stride;
    std::vector<uint8_t> shapeRequests;
    if (shapeGather) {
        // A selected body can emit at most its own published leaves. Each request,
        // including a repeated body, owns a disjoint range in the candidate buffer.
        stride = 0;
        records = 0;
        shapeRequests.resize(static_cast<size_t>(inputBytes));
        for (size_t row = 0; row < queries.size(); ++row) {
            auto request = queries[row];
            const uint32_t slot = request.source.y;
            const uint32_t leaves =
                slot == InvalidQueryBodySlot ? leafCount : publication->directory->actors[slot]->physicalModel.meta.y;
            request.source.z = static_cast<uint32_t>(records);
            request.source.w = std::max(1u, leaves);
            records += request.source.w;
            std::memcpy(shapeRequests.data() + row * sizeof(request), &request, sizeof(request));
        }
    }
    if (records > std::numeric_limits<uint32_t>::max())
        throw JoltGpuQueryRequestError("Jolt GPU query scratch capacity exceeds device indexing");
    const auto capacity = static_cast<uint32_t>(records);
    // Only typed Host batches opt in. The same motion invocation writes one
    // selected Float4 stream per input row; device exports keep their full ABI.
    uint32_t projectedMotionBlocks = 0;
    if (readback == JoltGpuQueryReadback::HostProjected && format == QueryOutputMotion) {
        uint32_t requested = 0;
        for (const auto &query : queries)
            requested |= query.source.z & static_cast<uint32_t>(PhysicsStateFields::All);
        if (requested) {
            const auto fields = static_cast<PhysicsStateFields>(requested);
            const auto select = [&](uint32_t block) { projectedMotionBlocks |= 1u << block; };
            select(QueryMotionBlockMeta);
            if (HasPhysicsStateField(fields, PhysicsStateFields::Position) ||
                HasPhysicsStateField(fields, PhysicsStateFields::InverseMass))
                select(QueryMotionBlockPositionInverseMass);
            if (HasPhysicsStateField(fields, PhysicsStateFields::Rotation))
                select(QueryMotionBlockRotation);
            if (HasPhysicsStateField(fields, PhysicsStateFields::CenterOfMass))
                select(QueryMotionBlockCenterOfMass);
            if (HasPhysicsStateField(fields, PhysicsStateFields::LinearVelocity))
                select(QueryMotionBlockLinearVelocity);
            if (HasPhysicsStateField(fields, PhysicsStateFields::AngularVelocity))
                select(QueryMotionBlockAngularVelocity);
            if (HasPhysicsStateField(fields, PhysicsStateFields::InverseInertia))
                projectedMotionBlocks |= QueryMotionInertiaBlocks;
            result->projectedMotionBlocks.reserve(QueryMotionBlockCount);
            for (uint32_t block = 0; block < QueryMotionBlockCount; ++block)
                if (projectedMotionBlocks & (1u << block))
                    result->projectedMotionBlocks.push_back(block);
        }
    }
    const uint32_t storageRecordBytes =
        result->projectedMotionBlocks.empty()
            ? recordBytes
            : static_cast<uint32_t>(result->projectedMotionBlocks.size()) * sizeof(QueryRecordWord);
    const auto words = storageRecordBytes / sizeof(QueryRecordWord);
    if (capacity > std::numeric_limits<uint32_t>::max() / words)
        throw JoltGpuQueryRequestError("Jolt GPU query record words exceed device indexing");
    // Span/CA control fits within the already required per-request input.
    // Keep this dominance proof independent of the input's current ABI size.
    static_assert(sizeof(QuerySpan) == sizeof(QueryRecordWord));
    static_assert(sizeof(QueryCastState) == 4 * sizeof(QueryRecordWord));
    static_assert(sizeof(QueryInput) >= sizeof(QuerySpan) + sizeof(QueryCastState));
    const uint64_t spanBytes = CheckedBytes(queryCount, sizeof(QuerySpan));
    const uint64_t prefixBytes =
        motion || shapeGather ? spanBytes : CheckedBytes(queryCount, sizeof(QuerySpan) + sizeof(QueryCastState));
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
    result->outputFormat = format;
    result->recordBytes = recordBytes;
    result->readback = readback;
    result->shapeBoxesOnly =
        format == QueryOutputShape &&
        std::all_of(queries.begin(), queries.end(), [](const auto &query) { return query.meta.z == ShapeBox; });
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
    if (!totalLeaves && !motion) {
        const ShapeModel shape{};
        upload(Shapes, Bytes(&shape, 1));
    }
    const QueryParams parameters{
        {result->queryCount, bodyBase, bodyCount, leafCount},
        {static_cast<uint32_t>(primitives.size()), stride, capacity, format},
        {result->publication ? result->publication->worldRow : 0u, bodyBase, leafBase,
         result->publication ? result->publication->jointBase : 0u},
        {bodyCount, leafCount, result->publication ? result->publication->jointCount : 0u,
         projectedMotionBlocks | (knownCount && IsDeviceResult(readback) ? QueryFixedRowMapping : 0u)}};
    upload(Parameters, Bytes(&parameters, 1));
    if (motion) {
        // Motion requests name one retained actor each, including duplicate rows.
        // Carry its identity beside the slot so other actors never enter this upload.
        std::vector<uint8_t> data(static_cast<size_t>(inputBytes));
        for (size_t row = 0; row < queries.size(); ++row) {
            const auto &request = queries[row];
            const auto slot = request.source.y;
            const QueryMotionInput input{{slot, result->publication->directory->actors[slot]->bodyId, 0u, 0u},
                                         request.geometry};
            std::memcpy(data.data() + row * sizeof(input), &input, sizeof(input));
        }
        upload(Inputs, std::move(data));
    } else if (shapeGather) {
        upload(Inputs, std::move(shapeRequests));
    } else if (rays) {
        allocate(Inputs, requestWords);
        buffers[RayInputs] = buffers[Inputs];
        buffers[RayOrigins] = rays->origins.buffer;
        buffers[RayDirections] = rays->directions.buffer;
        const RayInputParameters rayParameters{{result->queryCount,
                                                static_cast<uint32_t>(rays->origins.byteOffset / sizeof(float)),
                                                static_cast<uint32_t>(rays->directions.byteOffset / sizeof(float)), 0},
                                               queries.front().filter,
                                               {queries.front().positionMaximum.w, 0, 0, 0}};
        upload(RayParameters, Bytes(&rayParameters, 1));
    } else {
        // Geometric requests and their explicit primitives share one immutable word stream.
        std::vector<uint8_t> requestData(static_cast<size_t>(CheckedBytes(requestWords, sizeof(QueryRecordWord))));
        std::memcpy(requestData.data(), queries.data(), static_cast<size_t>(inputBytes));
        if (primitiveBytes)
            std::memcpy(requestData.data() + inputBytes, primitives.data(), static_cast<size_t>(primitiveBytes));
        upload(Inputs, std::move(requestData));
    }
    std::shared_ptr<QueryPolicyCopy> copy;
    if (!motion) {
        const bool hasSweep = std::any_of(queries.begin(), queries.end(),
                                          [](const auto &query) { return (query.meta.w & QuerySweepPolicy) != 0; });
        // Identity records are compact for this retained publication's world.
        // Author model/shape addresses and query result addresses remain global.
        const bool hasPairPolicy = std::any_of(queries.begin(), queries.end(), [](const auto &query) {
            return (query.meta.w & QuerySweepPairPolicy) != 0;
        });
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
            const uint64_t rigidbodyId = actor ? actor->rigidbodyComponentId : 0;
            identities[local].ids = {actor ? actor->bodyId : InvalidQueryBodySlot, rigidbodyId ? 1u : 0u,
                                     static_cast<uint32_t>(rigidbodyId), static_cast<uint32_t>(rigidbodyId >> 32)};
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
    }
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
            if (port == RayOrigins || port == RayDirections) {
                offset = port == RayOrigins ? rays->origins.byteOffset : rays->directions.byteOffset;
                bytes = CheckedBytes(queryCount, 3u * sizeof(float));
            } else if (port == Control) {
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
    if (motion)
        dispatches = {dispatch(ReadMotion), dispatch(Scan)};
    else if (generic)
        dispatches = {dispatch(Evaluate), dispatch(InitJobs), dispatch(AdvanceJobs), dispatch(JobProgress)};
    else
        dispatches = {dispatch(Evaluate), dispatch(Scan), dispatch(Compact)};
    if (rays)
        dispatches.insert(dispatches.begin(), dispatch(InitRays));
    std::vector<rhi::ComputeDispatchDesc> finalize;
    if (generic)
        finalize = {dispatch(Scan), dispatch(Compact)};
    result->hits = buffers[Hits];
    result->spans = buffers[Control];
    result->summary = buffers[Summary];
    std::vector<rhi::ComputeBufferRead> reads{{result->summary, 0, sizeof(QuerySummary)}};
    if (!generic && (knownCount || IsDeviceResult(readback))) {
        if (!IsDeviceResult(readback)) {
            result->hitReadbackIndex = static_cast<uint32_t>(reads.size());
            reads.push_back({result->hits, 0, CheckedBytes(result->queryCount, storageRecordBytes)});
        }
        if (result->NeedsHostSpans()) {
            result->spanReadbackIndex = static_cast<uint32_t>(reads.size());
            reads.push_back({result->spans, 0, CheckedBytes(result->queryCount, sizeof(QuerySpan))});
        }
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
                  payload = !generic && (knownCount || IsDeviceResult(readback))]() mutable -> JoltGpuControlWork {
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

JoltGpuControlWork JoltGpuQueries::PrepareExport(const std::vector<std::shared_ptr<JoltGpuQueryResult>> &results,
                                                 const std::vector<JoltGpuQueryExportBinding> &bindings) const
{
    m_host.RequireAlive();
    for (const auto &binding : bindings)
        if (binding.field > JoltGpuQueryExportField::HitValid)
            throw JoltGpuQueryRequestError("Jolt GPU export has an unknown field");
    std::array<std::shared_ptr<JoltGpuQueryResult>, 3> sources;
    const JoltGpuScene::Publication *publication = nullptr;
    uint64_t generation = 0;
    bool captured = false;
    for (const auto &result : results) {
        if (!result)
            continue;
        if (!result->projectedMotionBlocks.empty() && result->projectedMotionBlocks.size() != QueryMotionBlockCount)
            throw JoltGpuQueryRequestError("GPU exports require the full device motion record layout");
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
        if (result->outputFormat != QueryOutputMotion && result->outputFormat != QueryOutputShape &&
            (result->outputFormat != QueryOutputHit || !result->knownCount))
            throw JoltGpuQueryRequestError("Jolt GPU exports require typed state or fixed closest-hit records");
        if (sources[result->outputFormat])
            throw JoltGpuQueryRequestError("Jolt GPU export batch repeats a typed source family");
        sources[result->outputFormat] = result;
    }
    if ((!sources[QueryOutputMotion] && !sources[QueryOutputShape] && !sources[QueryOutputHit]) || bindings.empty())
        return {};
    constexpr std::array<uint32_t, 17> Widths{3, 4, 3, 3, 3, 3, 9, 1, 3, 3, 4, 1, 1, 3, 3, 1, 1};
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
        const bool hit = field >= static_cast<uint32_t>(JoltGpuQueryExportField::HitPoint);
        const auto &source = sources[motion ? QueryOutputMotion : hit ? QueryOutputHit : QueryOutputShape];
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
                            : binding.field == JoltGpuQueryExportField::HitValid     ? rhi::ComputeScalarType::UInt32
                                                                                     : rhi::ComputeScalarType::Float32;
        const auto lanes = Widths[field] == 9 ? 1u : Widths[field];
        const auto bytes = CheckedBytes(uint64_t(count) * Widths[field], sizeof(uint32_t));
        const auto end = std::min(binding.byteEnd, binding.buffer->GetByteSize());
        if (desc.scalarType != scalar || desc.lanes != lanes || binding.byteOffset > end ||
            bytes > end - binding.byteOffset ||
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
        buffers[Control] = source->spans;
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
                proof.reads.push_back({local, 0,
                                       port == Control ? CheckedBytes(source->queryCount, sizeof(QuerySpan))
                                                       : buffers[port]->GetByteSize()});
        }
        dispatch.proof = std::move(proof);
        dispatches.push_back(std::move(dispatch));
    }
    if (dispatches.empty())
        return {};
    JoltGpuControlWork work{rhi::PrepareComputeBatch(host, std::move(updates), std::move(dispatches))};
    work.geometry = publication ? (sources[QueryOutputMotion]  ? sources[QueryOutputMotion]->geometry
                                   : sources[QueryOutputShape] ? sources[QueryOutputShape]->geometry
                                                               : sources[QueryOutputHit]->geometry)
                                : nullptr;
    return work;
}

JoltGpuPreparedParticleExport JoltGpuQueries::PrepareParticleExport(const std::shared_ptr<JoltGpuQueryResult> &shapes,
                                                                    const std::vector<JoltGpuParticleExportRow> &rows,
                                                                    std::shared_ptr<rhi::ComputeBuffer> previousRecords,
                                                                    uint32_t staticCount, uint64_t revision,
                                                                    uint64_t topologyRevision) const
{
    m_host.RequireAlive();
    if (!shapes || !shapes->publication || !shapes->geometry || !shapes->geometry->buffer ||
        (shapes->queryCount && (!shapes->hits || !m_host.SharesServicesWith(shapes->hits->GetHost()))))
        throw JoltGpuQueryRequestError("GPU particle export requires shape records from its compute owner");
    auto result = std::make_shared<JoltGpuParticleExportResult>();
    result->source = shapes;
    result->payload = std::make_shared<particle::GpuParticleCollisionDevicePayload>();
    const auto &summary = shapes->GetSummary();
    if (summary.counts.y != QuerySuccess)
        return {std::move(result), {}};
    shapes->Materialize();
    if ((shapes->queryCount &&
         (shapes->outputFormat != QueryOutputShape || shapes->recordBytes != sizeof(QueryShapeState))) ||
        rows.size() > std::numeric_limits<uint32_t>::max() / 512u || staticCount > rows.size() || !topologyRevision)
        throw JoltGpuQueryRequestError("GPU particle export shape format, row domain or topology is invalid");
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
        if (row.selection.x >= summary.counts.x || (!row.selection.z && !row.selection.w) ||
            !Finite(row.localBoundsMin) || !Finite(row.localBoundsMax) || row.localBoundsMin.w ||
            row.localBoundsMax.w || row.localBoundsMin.x > row.localBoundsMax.x ||
            row.localBoundsMin.y > row.localBoundsMax.y || row.localBoundsMin.z > row.localBoundsMax.z ||
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
    buffers[Shapes] = shapes->publication->shapeModels;
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
    payload.worldId = shapes->publication->worldId;
    payload.geometry = shapes->geometry->buffer;
    payload.geometryLease = shapes->geometry;
    payload.geometryWordRanges = shapes->geometry->wordRanges;
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
            JoltGpuControlWork tail{result->batch};
            tail.geometry = result->source->geometry;
            return tail;
        }};
    work.geometry = shapes->geometry;
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

bool JoltGpuQueryResult::NeedsHostSpans() const
{
    if (IsDeviceResult(readback))
        return readback == JoltGpuQueryReadback::DeviceWithSpans && !knownCount;
    return projectedMotionBlocks.empty();
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
    if (!NeedsHostSpans()) {
        // The scan validates row ranges on device. A device consumer can use
        // those ranges directly even when the number of shapes varies by body.
        materialized = true;
        return;
    }
    hostSpans.resize(queryCount);
    if (!tailBatch->Readback(spanReadbackIndex)->IsComplete())
        throw std::logic_error("Jolt GPU query metadata requires its completed receipt");
    tailBatch->Readback(spanReadbackIndex)->ReadInto(hostSpans.data(), CheckedBytes(queryCount, sizeof(QuerySpan)));
    uint32_t end = 0;
    for (const auto &span : hostSpans) {
        RequireSuccess(span.range.z);
        if (span.range.x != end || span.range.y > count - end || span.range.w > hitCapacity - span.range.y ||
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
    if (IsDeviceResult(readback))
        throw std::logic_error("Jolt GPU device query does not materialize numerical records");
    Materialize();
    if (!recordsMaterialized) {
        const auto count = summaryValue.counts.x;
        if (count && !tailBatch->Readback(hitReadbackIndex)->IsComplete())
            throw std::logic_error("Jolt GPU query records require their completed receipt");
        records.resize(count);
        const bool expand =
            !projectedMotionBlocks.empty() && projectedMotionBlocks.size() * sizeof(QueryRecordWord) != sizeof(T);
        if (count && expand) {
            std::fill(records.begin(), records.end(), T{});
            const auto stride = CheckedBytes(projectedMotionBlocks.size(), sizeof(QueryRecordWord));
            std::vector<uint8_t> packed(static_cast<size_t>(CheckedBytes(count, stride)));
            tailBatch->Readback(hitReadbackIndex)->ReadInto(packed.data(), packed.size());
            for (uint32_t row = 0; row < count; ++row)
                for (size_t block = 0; block < projectedMotionBlocks.size(); ++block)
                    std::memcpy(reinterpret_cast<uint8_t *>(&records[row]) +
                                    projectedMotionBlocks[block] * sizeof(QueryRecordWord),
                                packed.data() + row * stride + block * sizeof(QueryRecordWord),
                                sizeof(QueryRecordWord));
        } else if (count) {
            tailBatch->Readback(hitReadbackIndex)->ReadInto(records.data(), CheckedBytes(count, sizeof(T)));
        }
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

uint32_t JoltGpuQueryResult::GetRecordCount()
{
    const auto &counts = GetSummary().counts;
    RequireSuccess(counts.y);
    return counts.x;
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
    if (!NeedsHostSpans() && hostSpans.size() != queryCount) {
        if (!knownCount)
            throw std::logic_error("Variable query spans were retained on the device");
        hostSpans.resize(queryCount);
        for (uint32_t i = 0; i < queryCount; ++i)
            hostSpans[i].range = {i, 1u, QuerySuccess, 0u};
    }
    return hostSpans;
}

} // namespace infernux
