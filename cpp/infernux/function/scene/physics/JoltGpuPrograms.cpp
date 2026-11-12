#include "JoltGpuPrograms.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace infernux
{
namespace
{
using namespace JPH::GPU;

struct Port
{
    Resource resource;
    std::string_view name;
    uint64_t stride;
    std::array<std::string_view, 6> fields;
    uint32_t integerFields;
};

constexpr Port Ports[] = {
    {Resource::StepParams,
     "step_params",
     sizeof(StepParams),
     {"counts", "solve", "gravityDt", "contact", "restitution"},
     3},
    {Resource::BodyModels,
     "body_models",
     sizeof(BodyModel),
     {"meta", "mass", "invInertiaDiagonal", "inertiaRotation", "damping", "surface"},
     (1u << 0) | (1u << 5)},
    {Resource::ShapeModels, "shape_models", sizeof(ShapeModel), {"geometry", "localPositionCOM", "localRotation"}, 0},
    {Resource::Worlds, "worlds", sizeof(WorldRange), {"range"}, 1},
    {Resource::BodyForces, "body_forces", sizeof(BodyForces), {"force", "torque"}, 0},
    {Resource::BodyStates,
     "body_states",
     sizeof(BodyState),
     {"positionCOM", "rotation", "linearVelocity", "angularVelocity"},
     0},
    {Resource::Bounds, "bounds", sizeof(Aabb), {"minimum", "maximum"}, 0},
    {Resource::Contacts,
     "contacts",
     sizeof(Contact),
     {"meta", "normalDepth", "localAnchorA", "localAnchorB", "impulse"},
     1},
};

constexpr size_t PortCount = sizeof(Ports) / sizeof(Ports[0]);
constexpr size_t StageCount = sizeof(Stages) / sizeof(Stages[0]);

constexpr size_t PortIndex(Resource resource)
{
    switch (resource) {
    case Resource::StepParams:
        return 0;
    case Resource::BodyModels:
        return 1;
    case Resource::ShapeModels:
        return 2;
    case Resource::Worlds:
        return 3;
    case Resource::BodyForces:
        return 4;
    case Resource::BodyStates:
        return 5;
    case Resource::Bounds:
        return 6;
    case Resource::Contacts:
        return 7;
    }
    return PortCount;
}

// These two stage bodies initialize every slot. Access qualifiers alone do not
// establish complete stores: all read/write solvers retain their input proof.
constexpr uint32_t CompleteWrites(Stage stage)
{
    switch (stage) {
    case Stage::BuildBounds:
        return Mask(Resource::Bounds);
    case Stage::CollidePairs:
        return Mask(Resource::Contacts);
    default:
        return 0;
    }
}

constexpr bool IndexedDefinitions()
{
    for (size_t slot = 0; slot < PortCount; ++slot)
        if (Mask(Ports[slot].resource) != (1u << slot) || PortIndex(Ports[slot].resource) != slot)
            return false;
    for (size_t slot = 0; slot < StageCount; ++slot) {
        const auto &stage = Stages[slot];
        if (static_cast<size_t>(stage.stage) != slot || stage.groupSizeX == 0 || stage.groupSizeY == 0 ||
            stage.groupSizeZ == 0 || (CompleteWrites(stage.stage) & ~stage.writes) != 0 ||
            (CompleteWrites(stage.stage) & stage.reads) != 0)
            return false;
    }
    return true;
}

static_assert(PortCount == 8, "Jolt GPU storage slots must match the resource table");
static_assert(IndexedDefinitions(), "Jolt GPU resources/stages must retain their compact index contract");

bool Finite3(const Float4 &value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

bool Finite4(const Float4 &value)
{
    return Finite3(value) && std::isfinite(value.w);
}

bool Normalized(const Float4 &value)
{
    return Finite4(value) &&
           std::abs(value.x * value.x + value.y * value.y + value.z * value.z + value.w * value.w - 1.0f) <= 1e-4f;
}

bool Zero3(const Float4 &value)
{
    return value.x == 0.0f && value.y == 0.0f && value.z == 0.0f;
}

struct ModelCounts
{
    uint32_t bodies, worlds, contacts, maxBodiesPerWorld;
};

template <class Change, class Record>
const Record &ModelRecord(const std::vector<Record> &records, const std::vector<Change> &changes, uint32_t index)
{
    const auto change = std::lower_bound(changes.begin(), changes.end(), index,
                                         [](const auto &value, uint32_t key) { return value.index < key; });
    return change != changes.end() && change->index == index ? change->model : records[index];
}

ModelCounts ValidateModel(const JoltGpuModel &model, const std::vector<JoltGpuBodyModelUpdate> &bodyChanges = {},
                          const std::vector<JoltGpuShapeModelUpdate> &shapeChanges = {})
{
    if (model.bodies.size() > std::numeric_limits<uint32_t>::max() ||
        model.worlds.size() > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("Jolt GPU model exceeds the device index range");
    ModelCounts counts{static_cast<uint32_t>(model.bodies.size()), static_cast<uint32_t>(model.worlds.size()), 0, 0};
    uint64_t bodyEnd = 0, contactEnd = 0;
    for (size_t world = 0; world < model.worlds.size(); ++world) {
        const auto range = model.worlds[world].range;
        if (range.x != bodyEnd || range.z != contactEnd || range.y > MaxBodiesPerWorld ||
            range.w != range.y * range.y || bodyEnd + range.y > counts.bodies)
            throw std::invalid_argument("Jolt GPU bootstrap requires contiguous bounded world ranges");
        for (uint32_t body = range.x; body < range.x + range.y; ++body) {
            const auto &record = ModelRecord(model.bodies, bodyChanges, body);
            if (record.meta.w != world || record.meta.x >= model.shapes.size() ||
                (record.meta.y != ShapeSphere && record.meta.y != ShapeBox && record.meta.y != ShapeEmpty) ||
                (record.meta.z != MotionStatic && record.meta.z != MotionDynamic) ||
                (record.meta.y == ShapeBox && record.meta.z != MotionStatic))
                throw std::invalid_argument("Unsupported body in the Jolt GPU bootstrap model");
            const auto &shape = ModelRecord(model.shapes, shapeChanges, record.meta.x);
            if (record.surface.x >= 32 || record.surface.z > MaterialCombineMaximum ||
                record.surface.w > MaterialCombineMaximum)
                throw std::invalid_argument("Jolt GPU body layer or material combine mode is invalid");
            const auto geometry = shape.geometry;
            if (!Finite4(shape.localPositionCOM) || shape.localPositionCOM.w != 0 || !Normalized(shape.localRotation))
                throw std::invalid_argument("Jolt GPU shape local position or rotation is invalid");
            const bool dynamic = record.meta.z == MotionDynamic;
            if (!Finite4(record.mass) || record.mass.y < 0 || record.mass.z < 0 || record.mass.z > 1 ||
                (dynamic ? record.mass.x <= 0 : record.mass.x != 0) || !Finite4(record.damping) ||
                record.damping.x < 0 || record.damping.y < 0 || record.damping.z < 0 || record.damping.w < 0 ||
                (dynamic && (!Finite3(record.invInertiaDiagonal) || record.invInertiaDiagonal.x <= 0 ||
                             record.invInertiaDiagonal.y <= 0 || record.invInertiaDiagonal.z <= 0 ||
                             !Normalized(record.inertiaRotation))))
                throw std::invalid_argument("Jolt GPU body has invalid mass, inertia, material or damping");
            if (record.meta.y == ShapeSphere) {
                if (!std::isfinite(geometry.w) || geometry.w <= 0)
                    throw std::invalid_argument("Jolt GPU sphere radius must be positive and finite");
            } else if (record.meta.y == ShapeBox &&
                       (!Finite4(geometry) || geometry.x <= 0 || geometry.y <= 0 || geometry.z <= 0 || geometry.w < 0 ||
                        geometry.w > std::min({geometry.x, geometry.y, geometry.z}))) {
                throw std::invalid_argument("Jolt GPU box dimensions or convex radius are invalid");
            }
        }
        bodyEnd += range.y;
        contactEnd += range.w;
        counts.maxBodiesPerWorld = std::max(counts.maxBodiesPerWorld, range.y);
    }
    if (bodyEnd != counts.bodies || contactEnd > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("Jolt GPU ranges do not cover the model");
    counts.contacts = static_cast<uint32_t>(contactEnd);
    return counts;
}

template <class Change> void ValidateModelChangeIndices(std::vector<Change> &changes, size_t capacity)
{
    std::sort(changes.begin(), changes.end(), [](const auto &a, const auto &b) { return a.index < b.index; });
    for (size_t index = 0; index < changes.size(); ++index) {
        const auto &change = changes[index];
        if (change.index >= capacity || (index && changes[index - 1].index == change.index))
            throw std::invalid_argument("Jolt GPU model update requires unique indices within existing capacity");
    }
}

void ValidateForces(const std::vector<BodyForces> &forces, uint32_t bodyCount)
{
    if (forces.size() != bodyCount)
        throw std::invalid_argument("Jolt GPU force input does not match its resident model");
    for (const auto &force : forces)
        if (!Finite3(force.force) || !Finite3(force.torque))
            throw std::invalid_argument("Jolt GPU force and torque must be finite");
}

void ValidateLayout(const Port &port, const ShaderResourceLayout &resource)
{
    const bool uniform = port.resource == Resource::StepParams;
    const auto &value = resource.value;
    if (resource.binding.type != (uniform ? rhi::BindingType::UniformBuffer : rhi::BindingType::StorageBuffer) ||
        (uniform ? value.byteSize : resource.elementStride) != port.stride || value.kind != ShaderValueKind::Struct ||
        value.fields.size() != port.stride / 16)
        throw std::invalid_argument("Jolt GPU target layout does not match its shared device records");
    for (size_t index = 0; index < value.fields.size(); ++index) {
        const auto &field = value.fields[index];
        const auto scalar = (port.integerFields & (1u << index)) ? ShaderScalarKind::UInt32 : ShaderScalarKind::Float32;
        if (field.name != port.fields[index] || field.offset != index * 16 || field.byteSize != 16 ||
            field.kind != ShaderValueKind::Vector || field.scalar != scalar || field.rows != 1 || field.columns != 4)
            throw std::invalid_argument("Jolt GPU target field differs from the shared device record");
    }
}

template <class T> std::vector<uint8_t> Bytes(const T *values, size_t count)
{
    std::vector<uint8_t> bytes(count * sizeof(T));
    if (!bytes.empty())
        std::memcpy(bytes.data(), values, bytes.size());
    return bytes;
}

template <class Change, class Record>
void AppendModelUpload(std::vector<rhi::ComputeBufferUpdate> &updates,
                       const std::shared_ptr<rhi::ComputeBuffer> &buffer, const std::vector<Change> &changes,
                       const std::vector<Record> &records)
{
    if (changes.empty())
        return;
    const auto first = changes.front().index;
    const size_t count = size_t(changes.back().index) - first + 1;
    auto bytes = Bytes(records.data() + first, count);
    // Allocate only the upload envelope; its holes retain accepted author data.
    for (const auto &change : changes)
        std::memcpy(bytes.data() + size_t(change.index - first) * sizeof(Record), &change.model, sizeof(Record));
    updates.push_back({buffer, uint64_t(first) * sizeof(Record), std::move(bytes)});
}

uint32_t Groups(uint32_t count, uint32_t width)
{
    return count / width + static_cast<uint32_t>(count % width != 0);
}
} // namespace

JoltGpuStepParameters::JoltGpuStepParameters(StepParams parameters) : m_parameters(parameters)
{
    if (parameters.solve.y == 0 || !Finite4(parameters.contact) || parameters.contact.x < 0 ||
        parameters.contact.y < 0 || parameters.contact.z < 0 || parameters.contact.w < 0 || parameters.contact.w > 1 ||
        !std::isfinite(parameters.restitution.x) || parameters.restitution.x < 0)
        throw std::invalid_argument("Jolt GPU substep count and contact parameters are invalid");
    SetGravityAndDeltaTime(parameters.gravityDt);
}

void JoltGpuStepParameters::SetGravityAndDeltaTime(Float4 gravityDt)
{
    if (!Finite4(gravityDt) || gravityDt.w <= 0.0f)
        throw std::invalid_argument("Jolt GPU gravity and substep duration must be finite with positive duration");
    m_parameters.gravityDt = gravityDt;
}

JoltGpuStorage::JoltGpuStorage(rhi::ComputeHost &host, JoltGpuModel modelData)
    : m_host(host), m_model(std::move(modelData))
{
    const auto &model = m_model;
    const auto counts = ValidateModel(model);
    m_bodyCount = counts.bodies;
    m_worldCount = counts.worlds;
    m_contactCount = counts.contacts;
    m_maxBodiesPerWorld = counts.maxBodiesPerWorld;
    if (m_bodyCount == 0)
        return;

    const uint64_t sizes[] = {sizeof(StepParams),
                              model.bodies.size() * sizeof(BodyModel),
                              model.shapes.size() * sizeof(ShapeModel),
                              model.worlds.size() * sizeof(WorldRange),
                              uint64_t(m_bodyCount) * sizeof(BodyForces),
                              uint64_t(m_bodyCount) * sizeof(BodyState),
                              uint64_t(m_bodyCount) * sizeof(Aabb),
                              uint64_t(m_contactCount) * sizeof(Contact)};
    for (size_t port = 0; port < m_buffers.size(); ++port) {
        const bool mixed = Ports[port].integerFields != 0;
        // Preserve the scalar type for homogeneous float ports so numerical
        // consumers need no hidden host cast. Mixed records remain raw words;
        // their structured element interpretation comes from target reflection.
        m_buffers[port] = std::make_shared<rhi::ComputeBuffer>(
            m_host, rhi::ComputeBufferDesc{sizes[port] / (mixed ? sizeof(uint32_t) : sizeof(Float4)),
                                           mixed ? rhi::ComputeScalarType::UInt32 : rhi::ComputeScalarType::Float32,
                                           static_cast<uint8_t>(mixed ? 1 : 4)});
    }
}

const std::shared_ptr<rhi::ComputeBuffer> &JoltGpuStorage::Get(Resource resource) const
{
    const auto slot = PortIndex(resource);
    if (slot == PortCount)
        throw std::invalid_argument("Unknown Jolt GPU resource");
    return m_buffers[slot];
}

std::shared_ptr<rhi::PreparedComputeBatch> JoltGpuStorage::PrepareInitialization(const std::vector<BodyState> &states,
                                                                                 const std::vector<BodyForces> &forces)
{
    if (states.size() != m_bodyCount || forces.size() != m_bodyCount)
        throw std::invalid_argument("Jolt GPU initialization does not match its resident model");
    for (size_t body = 0; body < states.size(); ++body) {
        const auto &state = states[body];
        if (!Finite3(state.positionCOM) || !Normalized(state.rotation) || !Finite3(state.linearVelocity) ||
            !Finite3(state.angularVelocity) || !Finite3(forces[body].force) || !Finite3(forces[body].torque) ||
            (m_model.bodies[body].meta.z == MotionStatic &&
             (!Zero3(state.linearVelocity) || !Zero3(state.angularVelocity))))
            throw std::invalid_argument("Jolt GPU initial pose, velocity or force is invalid");
    }
    if (m_bodyCount == 0)
        return {};
    std::vector<rhi::ComputeBufferUpdate> updates{
        {Get(Resource::BodyModels), 0, Bytes(m_model.bodies.data(), m_model.bodies.size())},
        {Get(Resource::ShapeModels), 0, Bytes(m_model.shapes.data(), m_model.shapes.size())},
        {Get(Resource::Worlds), 0, Bytes(m_model.worlds.data(), m_model.worlds.size())},
        {Get(Resource::BodyForces), 0, Bytes(forces.data(), forces.size())},
        {Get(Resource::BodyStates), 0, Bytes(states.data(), states.size())},
    };
    return rhi::PrepareComputeBatch(m_host, std::move(updates), {});
}

std::shared_ptr<rhi::PreparedComputeBatch> JoltGpuStorage::PrepareParameters(const JoltGpuStepParameters &prepared)
{
    return PrepareInputs(prepared);
}

std::shared_ptr<rhi::PreparedComputeBatch> JoltGpuStorage::PrepareInputs(const JoltGpuStepParameters &prepared,
                                                                         const std::vector<BodyForces> *cpuForces)
{
    if (cpuForces)
        ValidateForces(*cpuForces, m_bodyCount);
    if (m_bodyCount == 0)
        return {};
    auto parameters = prepared.Get();
    parameters.counts.x = m_bodyCount;
    parameters.counts.y = m_worldCount;
    parameters.counts.z = m_maxBodiesPerWorld;
    parameters.solve.z = m_contactCount;
    std::vector<rhi::ComputeBufferUpdate> updates{{Get(Resource::StepParams), 0, Bytes(&parameters, 1)}};
    if (cpuForces)
        updates.push_back({Get(Resource::BodyForces), 0, Bytes(cpuForces->data(), cpuForces->size())});
    return rhi::PrepareComputeBatch(m_host, std::move(updates), {});
}

std::shared_ptr<rhi::PreparedComputeBatch> JoltGpuStorage::PrepareForces(const std::vector<BodyForces> &forces)
{
    ValidateForces(forces, m_bodyCount);
    if (m_bodyCount == 0)
        return {};
    return rhi::PrepareComputeBatch(m_host, {{Get(Resource::BodyForces), 0, Bytes(forces.data(), forces.size())}}, {});
}

std::unique_ptr<JoltGpuModelUpdate> JoltGpuStorage::PrepareModelUpdate(
    std::vector<JoltGpuBodyModelUpdate> bodyChanges, std::vector<JoltGpuShapeModelUpdate> shapeChanges,
    std::vector<rhi::ComputeBufferUpdate> additionalUpdates, std::vector<rhi::ComputeDispatchDesc> mutations)
{
    if (bodyChanges.empty() && shapeChanges.empty() && additionalUpdates.empty() && mutations.empty())
        return {};
    if (m_modelUpdatePending)
        throw std::logic_error("Jolt GPU storage already has an unaccepted model transaction");

    const auto &bodyBuffer = Get(Resource::BodyModels);
    const auto &shapeBuffer = Get(Resource::ShapeModels);
    for (const auto &update : additionalUpdates) {
        if (update.buffer &&
            ((bodyBuffer && update.buffer->GetWriterStateIdentity() == bodyBuffer->GetWriterStateIdentity()) ||
             (shapeBuffer && update.buffer->GetWriterStateIdentity() == shapeBuffer->GetWriterStateIdentity())))
            throw std::invalid_argument("Jolt GPU model ports must be updated through their author-model delta");
    }

    ValidateModelChangeIndices(bodyChanges, m_model.bodies.size());
    ValidateModelChangeIndices(shapeChanges, m_model.shapes.size());
    if (!bodyChanges.empty() || !shapeChanges.empty())
        (void)ValidateModel(m_model, bodyChanges, shapeChanges);
    // Neither existing BodyStates nor BodyForces is read or uploaded here.
    // Explicit caller mutations, when present, execute in this same RHI packet.
    AppendModelUpload(additionalUpdates, bodyBuffer, bodyChanges, m_model.bodies);
    AppendModelUpload(additionalUpdates, shapeBuffer, shapeChanges, m_model.shapes);
    auto batch = rhi::PrepareComputeBatch(m_host, std::move(additionalUpdates), std::move(mutations));
    auto transaction = std::unique_ptr<JoltGpuModelUpdate>(
        new JoltGpuModelUpdate(*this, std::move(batch), std::move(bodyChanges), std::move(shapeChanges)));
    m_modelUpdatePending = true;
    return transaction;
}

JoltGpuModelUpdate::JoltGpuModelUpdate(JoltGpuStorage &storage, std::shared_ptr<rhi::PreparedComputeBatch> batch,
                                       std::vector<JoltGpuBodyModelUpdate> bodyChanges,
                                       std::vector<JoltGpuShapeModelUpdate> shapeChanges)
    : m_storage(&storage), m_batch(std::move(batch)), m_bodyChanges(std::move(bodyChanges)),
      m_shapeChanges(std::move(shapeChanges))
{
}

JoltGpuModelUpdate::~JoltGpuModelUpdate()
{
    Cancel();
}

void JoltGpuModelUpdate::Commit() noexcept
{
    for (const auto &change : m_bodyChanges)
        m_storage->m_model.bodies[change.index] = change.model;
    for (const auto &change : m_shapeChanges)
        m_storage->m_model.shapes[change.index] = change.model;
    m_storage->m_modelUpdatePending = false;
    m_storage = nullptr;
}

std::optional<rhi::SubmissionTicket> JoltGpuModelUpdate::TrySubmit()
{
    if (!m_storage)
        throw std::logic_error("Jolt GPU model transaction is no longer pending");
    std::optional<rhi::SubmissionTicket> ticket;
    try {
        ticket = m_batch->TrySubmit();
    } catch (...) {
        // Acceptance remains final if later RHI bookkeeping throws. No new
        // allocations or validation are needed to install the prepared delta.
        if (m_batch->GetSubmission().IsValid())
            Commit();
        throw;
    }
    if (ticket)
        Commit();
    return ticket;
}

void JoltGpuModelUpdate::Cancel() noexcept
{
    if (!m_storage)
        return;
    (void)m_batch->Cancel();
    m_storage->m_modelUpdatePending = false;
    m_storage = nullptr;
}

void JoltGpuStepInputs::Capture(const JoltGpuStorage &storage, Resource resource)
{
    const auto slot = PortIndex(resource);
    if (slot >= m_buffers.size())
        throw std::invalid_argument("Jolt GPU tick inputs accept only parameters, models, worlds and forces");
    if (storage.GetBodyCount() != 0)
        m_buffers[slot] = storage.Get(resource)->Freeze();
}

JoltGpuPrograms::JoltGpuPrograms(rhi::ComputeHost &host, const ShaderTargetCode &target) : m_host(host)
{
    for (const auto &definition : Stages) {
        const auto code = std::find_if(target.entries.begin(), target.entries.end(),
                                       [&](const auto &entry) { return entry.name == definition.entry; });
        if (code == target.entries.end() || code->stage != rhi::ShaderStage::Compute ||
            code->workgroupSize !=
                std::array<uint32_t, 3>{definition.groupSizeX, definition.groupSizeY, definition.groupSizeZ})
            throw std::invalid_argument("Jolt GPU target is missing a matching stage entry");

        Entry entry{definition, {}, {}};
        std::vector<rhi::ComputeBufferBinding> bindings;
        uint32_t actualResources = 0;
        for (const auto &resource : target.resources) {
            const std::array<uint32_t, 2> key{resource.set, resource.binding.binding};
            if (std::find(code->bindings.begin(), code->bindings.end(), key) == code->bindings.end())
                continue;
            const auto port = std::find_if(std::begin(Ports), std::end(Ports),
                                           [&](const Port &value) { return value.name == resource.name; });
            if (port == std::end(Ports) || resource.set != 0)
                throw std::invalid_argument("Jolt GPU stage contains an unknown resource port");
            ValidateLayout(*port, resource);
            if ((definition.writes & Mask(port->resource)) && resource.binding.readOnlyStorage)
                throw std::invalid_argument("Jolt GPU writer cannot bind a read-only target port");
            bindings.push_back({resource.binding.binding, resource.binding.type, resource.binding.readOnlyStorage});
            const bool read = (definition.reads & Mask(port->resource)) != 0;
            const bool write = (definition.writes & Mask(port->resource)) != 0;
            const auto access = read && write ? rhi::ComputeBufferAccess::ReadWrite
                                : write       ? rhi::ComputeBufferAccess::Write
                                              : rhi::ComputeBufferAccess::Read;
            entry.bindings.push_back({PortIndex(port->resource), access, read,
                                      (CompleteWrites(definition.stage) & Mask(port->resource)) != 0});
            actualResources |= Mask(port->resource);
        }
        if (actualResources != (definition.reads | definition.writes))
            throw std::invalid_argument("Jolt GPU entry resources differ from the stage recipe");

        if (target.target == ShaderCodeTarget::SpirV) {
            if (code->code.empty() || code->code.size() % sizeof(uint32_t))
                throw std::invalid_argument("Jolt GPU target contains invalid SPIR-V bytes");
            std::vector<uint32_t> words(code->code.size() / sizeof(uint32_t));
            std::memcpy(words.data(), code->code.data(), code->code.size());
            entry.kernel = std::make_shared<rhi::ComputeKernel>(
                m_host, rhi::ShaderModuleDesc::FromSpirV(words.data(), words.size()), code->name, std::move(bindings),
                0);
        } else {
            entry.kernel = std::make_shared<rhi::ComputeKernel>(
                m_host, rhi::ShaderModuleDesc::FromWgsl(code->code.data(), code->code.size()), code->name,
                std::move(bindings), 0);
        }
        m_entries[static_cast<size_t>(definition.stage)] = std::move(entry);
    }
}

rhi::ComputeDispatchDesc JoltGpuPrograms::BuildDispatch(const Entry &entry, const JoltGpuStorage &storage,
                                                        const JoltGpuStepInputs &inputs) const
{
    rhi::ComputeDispatchDesc dispatch;
    dispatch.kernel = entry.kernel;
    dispatch.buffers.reserve(entry.bindings.size());
    dispatch.bufferAccesses.reserve(entry.bindings.size());
    rhi::ComputeDispatchProof proof;
    proof.reads.reserve(entry.bindings.size());
    proof.writes.reserve(entry.bindings.size());
    for (size_t index = 0; index < entry.bindings.size(); ++index) {
        const auto &binding = entry.bindings[index];
        const auto &buffer =
            binding.slot < inputs.m_buffers.size() ? inputs.m_buffers[binding.slot] : storage.m_buffers[binding.slot];
        if (!buffer)
            throw std::invalid_argument("Jolt GPU stage requires captured input: " +
                                        std::string(Ports[binding.slot].name));
        dispatch.buffers.push_back(buffer);
        dispatch.bufferAccesses.push_back(binding.access);
        if (binding.read)
            proof.reads.push_back({static_cast<uint32_t>(index), 0, buffer->GetByteSize()});
        if (binding.completeWrite)
            proof.writes.push_back({static_cast<uint32_t>(index), 0, buffer->GetByteSize()});
    }
    dispatch.proof = std::move(proof);
    const auto &definition = entry.definition;
    switch (definition.domain) {
    case Domain::Bodies:
        dispatch.groupCountX = Groups(storage.GetBodyCount(), definition.groupSizeX);
        break;
    case Domain::WorldPairs:
        dispatch.groupCountX = Groups(storage.GetMaxBodiesPerWorld(), definition.groupSizeX);
        dispatch.groupCountY = Groups(storage.GetMaxBodiesPerWorld(), definition.groupSizeY);
        dispatch.groupCountZ = storage.GetWorldCount();
        break;
    case Domain::Contacts:
        dispatch.groupCountX = Groups(storage.GetContactCount(), definition.groupSizeX);
        break;
    case Domain::Worlds:
        dispatch.groupCountX = Groups(storage.GetWorldCount(), definition.groupSizeX);
        break;
    }
    return dispatch;
}

std::shared_ptr<rhi::PreparedComputeBatch> JoltGpuPrograms::PrepareStages(const JoltGpuStorage &storage,
                                                                          const JoltGpuStepInputs &inputs,
                                                                          const std::vector<Stage> &stages)
{
    if (!m_host.SharesServicesWith(storage.GetHost()))
        throw std::invalid_argument("Jolt GPU programs and state belong to different compute owners");
    if (storage.GetBodyCount() == 0 || stages.empty())
        return {};
    std::vector<rhi::ComputeDispatchDesc> dispatches;
    dispatches.reserve(stages.size());
    for (const auto stage : stages) {
        const auto index = static_cast<size_t>(stage);
        if (index >= m_entries.size())
            throw std::invalid_argument("Unknown Jolt GPU stage");
        dispatches.push_back(BuildDispatch(m_entries[index], storage, inputs));
    }
    return rhi::PrepareComputeBatch(m_host, {}, std::move(dispatches));
}

} // namespace infernux
