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
    Shapes,
    States,
    ParticleParameters,
    ParticleRows,
    ParticlePrevious,
    ParticleRecords,
    ParticleHeader,
    ParticleOffsets,
    ParticleIndices,
    ParticleStatus,
    PortCount
};
enum Stage : uint8_t
{
    ParticleProject,
    ParticleCount,
    ParticleScatter,
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
    {"shape_models",
     sizeof(ShapeModel),
     {"meta", "geometry", "localPositionCOM", "localRotation", "material", "surface", "source", "geometryReference"},
     225},
    {"body_states", sizeof(BodyState), {"positionCOM", "rotation", "linearVelocity", "angularVelocity"}, 0},
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
};
constexpr std::array<std::string_view, StageCount> Entries{"particle_collision_project", "particle_collision_count",
                                                           "particle_collision_scatter"};
constexpr uint32_t Bit(Port port)
{
    return 1u << port;
}
constexpr std::array<uint32_t, StageCount> StagePorts{
    Bit(ParticleParameters) | Bit(ParticleRows) | Bit(ParticlePrevious) | Bit(Shapes) | Bit(States) |
        Bit(ParticleRecords) | Bit(ParticleStatus),
    Bit(ParticleParameters) | Bit(ParticleRecords) | Bit(ParticleHeader) | Bit(ParticleOffsets) | Bit(ParticleStatus),
    Bit(ParticleParameters) | Bit(ParticleRecords) | Bit(ParticleHeader) | Bit(ParticleOffsets) | Bit(ParticleIndices) |
        Bit(ParticleStatus)};

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
    const bool uniform = port == ParticleParameters;
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
        (!uniform && resource.binding.readOnlyStorage !=
                         (port == Shapes || port == States || port == ParticleRows || port == ParticlePrevious)))
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

rhi::ComputeBufferAccess Access(uint8_t stage, uint8_t port)
{
    if (port == ParticleStatus)
        return rhi::ComputeBufferAccess::ReadWrite;
    if ((stage == ParticleProject && port == ParticleRecords) ||
        (stage == ParticleCount && (port == ParticleHeader || port == ParticleOffsets)) ||
        (stage == ParticleScatter && port == ParticleIndices))
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

} // namespace infernux
