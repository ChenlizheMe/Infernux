#include "JoltGpuQueries.h"

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
    Primitives,
    BodyIdentities,
    Models,
    Shapes,
    States,
    Candidates,
    Spans,
    Summary,
    Hits,
    PortCount
};
enum Stage : uint8_t
{
    Evaluate,
    Scan,
    Compact,
    StageCount
};
struct PortLayout
{
    std::string_view name;
    uint64_t stride;
    std::array<std::string_view, 7> fields;
    uint32_t integerFields;
};
constexpr PortLayout Ports[] = {
    {"query_params", sizeof(QueryParams), {"counts", "sizes"}, 3},
    {"query_inputs",
     sizeof(QueryInput),
     {"meta", "filter", "source", "positionMaximum", "direction", "rotation", "geometry"},
     7},
    {"query_primitives", sizeof(QueryPrimitive), {"meta", "geometry", "position", "rotation"}, 1},
    {"query_body_ids", sizeof(QueryBodyIdentity), {"ids"}, 1},
    {"body_models", sizeof(BodyModel), {"meta", "mass", "invInertiaDiagonal", "inertiaRotation", "damping"}, 1},
    {"shape_models",
     sizeof(ShapeModel),
     {"meta", "geometry", "localPositionCOM", "localRotation", "material", "surface", "source"},
     97},
    {"body_states", sizeof(BodyState), {"positionCOM", "rotation", "linearVelocity", "angularVelocity"}, 0},
    {"query_candidates", sizeof(QueryHit), {"meta", "positionDistance", "normalFraction", "pointADepth"}, 1},
    {"query_spans", sizeof(QuerySpan), {"range"}, 1},
    {"query_summary", sizeof(QuerySummary), {"counts"}, 1},
    {"query_hits", sizeof(QueryHit), {"meta", "positionDistance", "normalFraction", "pointADepth"}, 1},
};
constexpr std::array<std::string_view, StageCount> Entries{"query_evaluate", "query_scan", "query_compact"};
constexpr uint32_t Bit(Port port)
{
    return 1u << port;
}
constexpr std::array<uint32_t, StageCount> StagePorts{
    Bit(Parameters) | Bit(Inputs) | Bit(Primitives) | Bit(BodyIdentities) | Bit(Models) | Bit(Shapes) | Bit(States) |
        Bit(Candidates) | Bit(Spans),
    Bit(Parameters) | Bit(Spans) | Bit(Summary),
    Bit(Parameters) | Bit(Candidates) | Bit(Spans) | Bit(Summary) | Bit(Hits)};

uint64_t CheckedBytes(uint64_t count, uint64_t stride)
{
    if (count > std::numeric_limits<uint64_t>::max() / stride || count > std::numeric_limits<size_t>::max() / stride)
        throw std::length_error("Jolt GPU query byte count exceeds the host/device address range");
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
    const bool uniform = port == Parameters;
    if (resource.binding.type != (uniform ? rhi::BindingType::UniformBuffer : rhi::BindingType::StorageBuffer) ||
        (uniform ? resource.value.byteSize : resource.elementStride) != expected.stride ||
        resource.value.kind != ShaderValueKind::Struct || resource.value.fields.size() != expected.stride / 16 ||
        (!uniform && resource.binding.readOnlyStorage != (port < Candidates)))
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
        throw std::invalid_argument("Jolt GPU query geometry and rotation must be finite");
    const double length = double(rotation.x) * rotation.x + double(rotation.y) * rotation.y +
                          double(rotation.z) * rotation.z + double(rotation.w) * rotation.w;
    if (std::abs(length - 1.0) > 1.0e-3)
        throw std::invalid_argument("Jolt GPU query rotation must be a unit quaternion");
    if (type == ShapeSphere) {
        if (geometry.x != 0 || geometry.y != 0 || geometry.z != 0 || geometry.w <= 0)
            throw std::invalid_argument("Jolt GPU query Sphere requires a positive radius");
    } else if (type == ShapeBox) {
        if (geometry.x <= 0 || geometry.y <= 0 || geometry.z <= 0 || geometry.w < 0 ||
            geometry.w > std::min({geometry.x, geometry.y, geometry.z}))
            throw std::invalid_argument("Jolt GPU query Box requires positive extents and a valid cooked radius");
    } else if (type == ShapeCapsule) {
        if (geometry.x != 0 || geometry.y < 0 || geometry.z != 0 || geometry.w <= 0)
            throw std::invalid_argument("Jolt GPU query Capsule requires a nonnegative segment and positive radius");
    } else {
        throw std::invalid_argument("Jolt GPU query primitive type is not supported");
    }
}

void ValidateInput(const QueryInput &query, size_t primitiveCount, uint32_t bodyCount, bool hasPublication)
{
    if (query.meta.x > QueryPenetration || query.meta.y > QueryAny ||
        (query.meta.w & ~(QueryTargetCollider | QueryExcludeBody)) || query.filter.y > 1 || query.source.z ||
        query.source.w || !Finite(query.positionMaximum) || !Finite(query.direction) || !Finite(query.rotation) ||
        !Finite(query.geometry) || query.direction.w != 0)
        throw std::invalid_argument("Jolt GPU query contains an invalid operation, filter or finite input record");
    if ((query.meta.w & QueryExcludeBody) ? query.source.x >= bodyCount : query.source.x != InvalidQueryBodySlot)
        throw std::invalid_argument("Jolt GPU query excluded body differs from its retained publication");
    if (query.meta.x == QueryPenetration) {
        if (query.source.y >= primitiveCount || query.meta.w || query.positionMaximum.w != 0)
            throw std::invalid_argument(
                "Jolt GPU pair query requires an explicit second primitive and no world policy");
        ValidateGeometry(query.meta.z, query.geometry, query.rotation);
    } else {
        if (!hasPublication || query.source.y)
            throw std::invalid_argument("Jolt GPU world query requires one retained publication");
        if (query.meta.x == QueryRay) {
            if (query.positionMaximum.w <= 0 ||
                (query.direction.x == 0 && query.direction.y == 0 && query.direction.z == 0))
                throw std::invalid_argument("Jolt GPU ray requires nonzero direction and positive maximum distance");
        } else {
            if (query.positionMaximum.w != 0)
                throw std::invalid_argument("Jolt GPU non-ray query maximum must be zero");
            if (query.meta.x == QueryOverlap)
                ValidateGeometry(query.meta.z, query.geometry, query.rotation);
            if (query.meta.x == QueryBodyBounds &&
                (query.direction.x < 0 || query.direction.y < 0 || query.direction.z < 0))
                throw std::invalid_argument("Jolt GPU bounds query requires nonnegative half extents");
        }
    }
}

rhi::ComputeBufferAccess Access(uint8_t stage, uint8_t port)
{
    if (stage == Evaluate && (port == Candidates || port == Spans))
        return rhi::ComputeBufferAccess::Write;
    if (stage == Scan && port == Spans)
        return rhi::ComputeBufferAccess::ReadWrite;
    if ((stage == Scan && port == Summary) || (stage == Compact && port == Hits))
        return rhi::ComputeBufferAccess::Write;
    return rhi::ComputeBufferAccess::Read;
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

JoltGpuPreparedQuery JoltGpuQueries::Prepare(std::shared_ptr<const JoltGpuScene::Publication> publication,
                                             const std::vector<QueryInput> &queries,
                                             const std::vector<QueryPrimitive> &primitives,
                                             uint64_t numericalGeneration) const
{
    if (queries.size() > std::numeric_limits<uint32_t>::max() ||
        primitives.size() > std::numeric_limits<uint32_t>::max())
        throw std::length_error("Jolt GPU query input count exceeds device indexing");
    std::array<std::shared_ptr<rhi::ComputeBuffer>, PortCount> buffers;
    uint32_t bodyCount = 0, leafCount = 0;
    if (publication) {
        buffers[Models] = publication->bodyModels;
        buffers[Shapes] = publication->shapeModels;
        buffers[States] = publication->bodyStates;
        if (publication->leaves.size() > std::numeric_limits<uint32_t>::max())
            throw std::length_error("Jolt GPU query publication exceeds leaf indexing");
        leafCount = static_cast<uint32_t>(publication->leaves.size());
        const bool empty = !buffers[Models] && !buffers[States];
        if (empty && (buffers[Shapes] || leafCount))
            throw std::invalid_argument("Jolt GPU query publication has leaves without body models/states");
        if (!empty) {
            for (const auto port : {Models, States}) {
                const auto &buffer = buffers[port];
                if (!buffer || !buffer->IsFrozen() || !m_host.SharesServicesWith(buffer->GetHost()) ||
                    !buffer->GetByteSize() || buffer->GetByteSize() % Ports[port].stride)
                    throw std::invalid_argument("Jolt GPU query requires matching frozen publication records");
            }
            const auto count = buffers[Models]->GetByteSize() / sizeof(BodyModel);
            if (count > MaxBodiesPerWorld || publication->actors.size() != count ||
                buffers[States]->GetByteSize() / sizeof(BodyState) != count)
                throw std::invalid_argument("Jolt GPU query body capacity differs from its publication");
            bodyCount = static_cast<uint32_t>(count);
            const auto &shapes = buffers[Shapes];
            if (leafCount ? (!shapes || !shapes->IsFrozen() || !m_host.SharesServicesWith(shapes->GetHost()) ||
                             shapes->GetByteSize() != CheckedBytes(leafCount, sizeof(ShapeModel)))
                          : bool(shapes))
                throw std::invalid_argument("Jolt GPU query leaf count differs from its frozen publication");
        }
    }
    for (const auto &primitive : primitives) {
        if (primitive.meta.y || primitive.meta.z || primitive.meta.w || !Finite(primitive.position) ||
            primitive.position.w)
            throw std::invalid_argument("Jolt GPU explicit query primitive has invalid pose or padding");
        ValidateGeometry(primitive.meta.x, primitive.geometry, primitive.rotation);
    }
    for (const auto &query : queries)
        ValidateInput(query, primitives.size(), bodyCount, bool(publication));
    const bool knownCount =
        std::all_of(queries.begin(), queries.end(), [](const auto &query) { return query.meta.y == QueryClosest; });
    const auto stride = knownCount ? 1u : std::max({1u, bodyCount, leafCount});
    if (queries.size() > std::numeric_limits<uint32_t>::max() / stride)
        throw std::length_error("Jolt GPU query scratch capacity exceeds device indexing");
    const auto capacity = static_cast<uint32_t>(queries.size()) * stride;
    auto result = std::make_shared<JoltGpuQueryResult>();
    result->publication = std::move(publication);
    result->numericalGeneration = numericalGeneration;
    result->queryCount = static_cast<uint32_t>(queries.size());
    result->hitCapacity = capacity;
    result->knownCount = knownCount;
    if (queries.empty()) {
        result->headerComplete = result->materialized = true;
        return {std::move(result), {}};
    }
    // Reject impossible dispatch dimensions before allocating request storage.
    const auto groups = [](uint32_t count) { return count / 64 + static_cast<uint32_t>(count % 64 != 0); };
    if (groups(capacity) > m_host.device.GetCapabilities().limits.maxComputeWorkgroupCount[0])
        throw std::length_error("Jolt GPU query capacity exceeds the device dispatch range");
    rhi::ComputeHost host = m_host;
    const auto allocate = [&](uint8_t port, uint64_t count) {
        const auto bytes = CheckedBytes(count, Ports[port].stride);
        buffers[port] =
            rhi::ComputeBuffer::TryCreatePrivate(host, {bytes / sizeof(uint32_t), rhi::ComputeScalarType::UInt32, 1});
        if (!buffers[port])
            throw std::runtime_error("Jolt GPU query preparation exceeds available managed device capacity");
    };
    std::vector<rhi::ComputeBufferUpdate> updates;
    const auto upload = [&](uint8_t port, std::vector<uint8_t> bytes) {
        allocate(port, bytes.size() / Ports[port].stride);
        updates.push_back({buffers[port], 0, std::move(bytes)});
    };
    if (!bodyCount) {
        const BodyModel model{};
        const BodyState state{};
        upload(Models, Bytes(&model, 1));
        upload(States, Bytes(&state, 1));
    }
    if (!leafCount) {
        const ShapeModel shape{};
        upload(Shapes, Bytes(&shape, 1));
    }
    std::vector<QueryBodyIdentity> identities(std::max(1u, bodyCount));
    for (uint32_t slot = 0; slot < identities.size(); ++slot) {
        const auto actor = bodyCount ? result->publication->actors[slot] : nullptr;
        const uint64_t rigidbodyId = actor ? actor->rigidbodyComponentId : 0;
        identities[slot].ids = {actor ? actor->bodyId : InvalidQueryBodySlot, rigidbodyId ? 1u : 0u,
                                static_cast<uint32_t>(rigidbodyId), static_cast<uint32_t>(rigidbodyId >> 32)};
    }
    upload(BodyIdentities, Bytes(identities.data(), identities.size()));
    const QueryParams parameters{{result->queryCount, 0, bodyCount, leafCount},
                                 {static_cast<uint32_t>(primitives.size()), stride, capacity, 0}};
    upload(Parameters, Bytes(&parameters, 1));
    upload(Inputs, Bytes(queries.data(), queries.size()));
    if (primitives.empty()) {
        const QueryPrimitive primitive{};
        upload(Primitives, Bytes(&primitive, 1));
    } else {
        upload(Primitives, Bytes(primitives.data(), primitives.size()));
    }
    allocate(Candidates, capacity);
    allocate(Spans, result->queryCount);
    allocate(Summary, 1);
    allocate(Hits, capacity);
    std::vector<rhi::ComputeDispatchDesc> dispatches;
    for (uint8_t stage = 0; stage < StageCount; ++stage) {
        const auto &program = m_programs[stage];
        rhi::ComputeDispatchDesc dispatch;
        dispatch.kernel = program.kernel;
        dispatch.groupCountX = stage == Scan ? 1 : groups(stage == Evaluate ? result->queryCount : capacity);
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
        dispatches.push_back(std::move(dispatch));
    }
    result->hits = buffers[Hits];
    result->spans = buffers[Spans];
    result->summary = buffers[Summary];
    std::vector<rhi::ComputeBufferRead> reads{{result->summary, 0, sizeof(QuerySummary)}};
    if (knownCount) {
        result->hitReadbackIndex = static_cast<uint32_t>(reads.size());
        reads.push_back({result->hits, 0, CheckedBytes(result->queryCount, sizeof(QueryHit))});
        result->spanReadbackIndex = static_cast<uint32_t>(reads.size());
        reads.push_back({result->spans, 0, CheckedBytes(result->queryCount, sizeof(QuerySpan))});
    }
    auto batch = rhi::PrepareComputeBatch(host, std::move(updates), std::move(dispatches), std::move(reads));
    result->summaryReadback = batch->Readback(0);
    if (knownCount) {
        result->tailBatch = batch;
        return {std::move(result), std::move(batch)};
    }
    JoltGpuControlWork work{batch, [result, host]() mutable -> JoltGpuControlWork {
                                result->ReadSummary();
                                std::vector<rhi::ComputeBufferRead> prefixReads;
                                const auto count = result->summaryValue.counts.x;
                                if (count) {
                                    result->hitReadbackIndex = static_cast<uint32_t>(prefixReads.size());
                                    prefixReads.push_back({result->hits, 0, CheckedBytes(count, sizeof(QueryHit))});
                                }
                                result->spanReadbackIndex = static_cast<uint32_t>(prefixReads.size());
                                prefixReads.push_back(
                                    {result->spans, 0, CheckedBytes(result->queryCount, sizeof(QuerySpan))});
                                result->tailBatch = rhi::PrepareComputeBatch(host, {}, {}, std::move(prefixReads));
                                return result->tailBatch;
                            }};
    return {std::move(result), std::move(work)};
}

void JoltGpuQueryResult::ReadSummary()
{
    if (headerComplete)
        return;
    if (!summaryReadback || !summaryReadback->IsComplete())
        throw std::logic_error("Jolt GPU query summary requires its completed receipt");
    summaryReadback->ReadInto(&summaryValue, sizeof(QuerySummary));
    const auto &counts = summaryValue.counts;
    RequireSuccess(counts.y);
    if (counts.x > hitCapacity || counts.z != queryCount || counts.w || (knownCount && counts.x != queryCount))
        throw std::runtime_error("Jolt GPU query compact summary differs from its prepared output range");
    headerComplete = true;
}

void JoltGpuQueryResult::Materialize()
{
    if (materialized)
        return;
    ReadSummary();
    if (!tailBatch || !tailBatch->GetSubmission().IsValid())
        throw std::logic_error("Jolt GPU query output has not completed its prefix transfer");
    const auto count = summaryValue.counts.x;
    if ((count && !tailBatch->Readback(hitReadbackIndex)->IsComplete()) ||
        !tailBatch->Readback(spanReadbackIndex)->IsComplete())
        throw std::logic_error("Jolt GPU query output requires its completed prefix receipt");
    RequireSuccess(summaryValue.counts.y);
    hostHits.resize(count);
    hostSpans.resize(queryCount);
    if (count)
        tailBatch->Readback(hitReadbackIndex)->ReadInto(hostHits.data(), CheckedBytes(count, sizeof(QueryHit)));
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
    for (const auto &hit : hostHits) {
        if (hit.meta.y > 1 || !Finite(hit.positionDistance) || !Finite(hit.normalFraction) ||
            !Finite(hit.pointADepth) || hit.positionDistance.w < 0 || hit.pointADepth.w < 0)
            throw std::runtime_error("Jolt GPU query returned an invalid or nonfinite output record");
    }
    materialized = true;
}

const std::vector<JPH::GPU::QueryHit> &JoltGpuQueryResult::ReadHits()
{
    Materialize();
    return hostHits;
}

const std::vector<JPH::GPU::QuerySpan> &JoltGpuQueryResult::ReadSpans()
{
    Materialize();
    return hostSpans;
}

} // namespace infernux
