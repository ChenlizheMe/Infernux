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
    std::array<std::string_view, 7> fields;
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
     {"meta", "mass", "invInertiaDiagonal", "inertiaRotation", "damping"},
     1},
    {Resource::ShapeModels,
     "shape_models",
     sizeof(ShapeModel),
     {"meta", "geometry", "localPositionCOM", "localRotation", "material", "surface", "source"},
     (1u << 0) | (1u << 5) | (1u << 6)},
    {Resource::Worlds, "worlds", sizeof(WorldRange), {"bodies", "leaves"}, 3},
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
    {Resource::KinematicTargets,
     "kinematic_targets",
     sizeof(KinematicTarget),
     {"positionSpeed", "rotation", "rootCOMDt", "state"},
     1u << 3},
    {Resource::ContactPatches, "contact_patches", sizeof(ContactPatch), {"leaves"}, 1},
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
    case Resource::KinematicTargets:
        return 8;
    case Resource::ContactPatches:
        return 9;
    }
    return PortCount;
}

constexpr size_t InputIndex(Resource resource)
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
    default:
        return 5;
    }
}

std::shared_ptr<rhi::ComputeBuffer> Allocate(rhi::ComputeHost &host, size_t port, uint64_t bytes)
{
    if (!bytes)
        return {};
    const bool mixed = Ports[port].integerFields != 0;
    return std::make_shared<rhi::ComputeBuffer>(
        host, rhi::ComputeBufferDesc{bytes / (mixed ? sizeof(uint32_t) : sizeof(Float4)),
                                     mixed ? rhi::ComputeScalarType::UInt32 : rhi::ComputeScalarType::Float32,
                                     static_cast<uint8_t>(mixed ? 1 : 4)});
}

// These stage bodies initialize every slot. Access qualifiers alone do not
// establish complete stores: all read/write solvers retain their input proof.
constexpr uint32_t CompleteWrites(Stage stage)
{
    switch (stage) {
    case Stage::BuildBounds:
        return Mask(Resource::Bounds);
    case Stage::CollidePairs:
        return Mask(Resource::Contacts) | Mask(Resource::ContactPatches);
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

static_assert(PortCount == 10, "Jolt GPU storage slots must match the resource table");
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
    uint32_t bodies, worlds, shapes, contacts = 0, patches = 0, maxBodiesPerWorld = 0, maxLeavesPerWorld = 0;
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
    constexpr auto limit = std::numeric_limits<uint32_t>::max();
    if (model.bodies.size() > limit || model.worlds.size() > limit || model.shapes.size() > limit)
        throw std::invalid_argument("Jolt GPU model exceeds the device index range");
    ModelCounts counts{static_cast<uint32_t>(model.bodies.size()), static_cast<uint32_t>(model.worlds.size()),
                       static_cast<uint32_t>(model.shapes.size())};
    uint64_t bodyEnd = 0, leafEnd = 0, patchEnd = 0;
    for (size_t world = 0; world < model.worlds.size(); ++world) {
        const auto range = model.worlds[world].bodies;
        const auto leaves = model.worlds[world].leaves;
        const uint64_t patches = uint64_t(leaves.y) * leaves.y;
        if (range.x != bodyEnd || leaves.x != leafEnd || range.z != patchEnd || range.y > MaxBodiesPerWorld ||
            patches > limit || range.w != patches || bodyEnd + range.y > counts.bodies ||
            leafEnd + leaves.y > counts.shapes || leaves.z || leaves.w)
            throw std::invalid_argument("Jolt GPU model requires contiguous body, leaf and patch ranges");
        uint64_t bodyLeafEnd = leafEnd;
        for (uint32_t body = range.x; body < bodyEnd + range.y; ++body) {
            const auto &record = ModelRecord(model.bodies, bodyChanges, body);
            const auto motionType = record.meta.z & ~MotionNotResident;
            if (record.meta.w != world || record.meta.x != bodyLeafEnd ||
                bodyLeafEnd + record.meta.y > leafEnd + leaves.y ||
                (motionType != MotionStatic && motionType != MotionKinematic && motionType != MotionDynamic))
                throw std::invalid_argument("Invalid body or leaf range in the Jolt GPU model");
            const bool nonstatic = motionType != MotionStatic;
            if (!Finite4(record.mass) || record.mass.y != 0 || record.mass.z != 0 ||
                (nonstatic ? record.mass.x <= 0 : record.mass.x != 0) || !Finite4(record.damping) ||
                record.damping.x < 0 || record.damping.y < 0 || record.damping.z < 0 || record.damping.w < 0 ||
                (nonstatic && (!Finite3(record.invInertiaDiagonal) || record.invInertiaDiagonal.x <= 0 ||
                               record.invInertiaDiagonal.y <= 0 || record.invInertiaDiagonal.z <= 0 ||
                               !Normalized(record.inertiaRotation))))
                throw std::invalid_argument("Jolt GPU body has invalid mass, inertia or damping");
            for (uint32_t leaf = record.meta.x; leaf < bodyLeafEnd + record.meta.y; ++leaf) {
                const auto &shape = ModelRecord(model.shapes, shapeChanges, leaf);
                if (shape.meta.y != body || (shape.meta.x != ShapeSphere && shape.meta.x != ShapeBox) ||
                    (shape.meta.w & ~ShapeSensor) || shape.surface.x >= 32 ||
                    shape.surface.z > MaterialCombineMaximum || shape.surface.w > MaterialCombineMaximum ||
                    shape.source.w > 32)
                    throw std::invalid_argument("Jolt GPU leaf identity, flags or material mode is invalid");
                if (!Finite4(shape.localPositionCOM) || shape.localPositionCOM.w != 0 ||
                    !Normalized(shape.localRotation) || !Finite4(shape.material) || shape.material.x < 0 ||
                    shape.material.y < 0 || shape.material.y > 1 || shape.material.z != 0 || shape.material.w != 0)
                    throw std::invalid_argument("Jolt GPU leaf pose or material is invalid");
                const auto geometry = shape.geometry;
                if (shape.meta.x == ShapeSphere) {
                    if (!std::isfinite(geometry.w) || geometry.w <= 0)
                        throw std::invalid_argument("Jolt GPU sphere radius must be positive and finite");
                } else if (!Finite4(geometry) || geometry.x <= 0 || geometry.y <= 0 || geometry.z <= 0 ||
                           geometry.w < 0 || geometry.w > std::min({geometry.x, geometry.y, geometry.z})) {
                    throw std::invalid_argument("Jolt GPU box dimensions or convex radius are invalid");
                }
            }
            bodyLeafEnd += record.meta.y;
        }
        if (bodyLeafEnd != leafEnd + leaves.y)
            throw std::invalid_argument("Jolt GPU body leaf ranges do not cover their world");
        bodyEnd += range.y;
        leafEnd += leaves.y;
        patchEnd += patches;
        if (patchEnd > limit / MaxContactPointsPerPair)
            throw std::invalid_argument("Jolt GPU contact capacity exceeds the device index range");
        counts.maxBodiesPerWorld = std::max(counts.maxBodiesPerWorld, range.y);
        counts.maxLeavesPerWorld = std::max(counts.maxLeavesPerWorld, leaves.y);
    }
    if (bodyEnd != counts.bodies || leafEnd != counts.shapes)
        throw std::invalid_argument("Jolt GPU ranges do not cover the model");
    counts.patches = static_cast<uint32_t>(patchEnd);
    counts.contacts = counts.patches * MaxContactPointsPerPair;
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
    m_shapeCount = counts.shapes;
    m_contactCount = counts.contacts;
    m_patchCount = counts.patches;
    m_maxBodiesPerWorld = counts.maxBodiesPerWorld;
    m_maxLeavesPerWorld = counts.maxLeavesPerWorld;
    if (m_bodyCount == 0)
        return;

    const uint64_t sizes[] = {sizeof(StepParams),
                              model.bodies.size() * sizeof(BodyModel),
                              model.shapes.size() * sizeof(ShapeModel),
                              model.worlds.size() * sizeof(WorldRange),
                              uint64_t(m_bodyCount) * sizeof(BodyForces),
                              uint64_t(m_bodyCount) * sizeof(BodyState),
                              uint64_t(m_shapeCount) * sizeof(Aabb),
                              uint64_t(m_contactCount) * sizeof(Contact),
                              uint64_t(m_bodyCount) * sizeof(KinematicTarget),
                              uint64_t(m_patchCount) * sizeof(ContactPatch)};
    for (size_t port = 0; port < m_buffers.size(); ++port) {
        // Preserve the scalar type for homogeneous float ports so numerical
        // consumers need no hidden host cast. Mixed records remain raw words;
        // their structured element interpretation comes from target reflection.
        m_buffers[port] = Allocate(m_host, port, sizes[port]);
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
            ((m_model.bodies[body].meta.z & ~MotionNotResident) == MotionStatic &&
             (!Zero3(state.linearVelocity) || !Zero3(state.angularVelocity))))
            throw std::invalid_argument("Jolt GPU initial pose, velocity or force is invalid");
    }
    if (m_bodyCount == 0)
        return {};
    const std::vector<KinematicTarget> targets(m_bodyCount);
    std::vector<rhi::ComputeBufferUpdate> updates{
        {Get(Resource::BodyModels), 0, Bytes(m_model.bodies.data(), m_model.bodies.size())},
        {Get(Resource::Worlds), 0, Bytes(m_model.worlds.data(), m_model.worlds.size())},
        {Get(Resource::BodyForces), 0, Bytes(forces.data(), forces.size())},
        {Get(Resource::BodyStates), 0, Bytes(states.data(), states.size())},
        {Get(Resource::KinematicTargets), 0, Bytes(targets.data(), targets.size())},
    };
    if (m_shapeCount)
        updates.push_back({Get(Resource::ShapeModels), 0, Bytes(m_model.shapes.data(), m_model.shapes.size())});
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
    parameters.counts.z = m_shapeCount;
    parameters.solve.z = m_contactCount;
    parameters.solve.w = m_patchCount;
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
    const auto &worldBuffer = Get(Resource::Worlds);
    for (const auto &update : additionalUpdates) {
        if (update.buffer &&
            ((bodyBuffer && update.buffer->GetWriterStateIdentity() == bodyBuffer->GetWriterStateIdentity()) ||
             (shapeBuffer && update.buffer->GetWriterStateIdentity() == shapeBuffer->GetWriterStateIdentity()) ||
             (worldBuffer && update.buffer->GetWriterStateIdentity() == worldBuffer->GetWriterStateIdentity())))
            throw std::invalid_argument("Jolt GPU model ports must be updated through their author-model delta");
    }

    ValidateModelChangeIndices(bodyChanges, m_model.bodies.size());
    ValidateModelChangeIndices(shapeChanges, m_model.shapes.size());
    if (!bodyChanges.empty() || !shapeChanges.empty())
        (void)ValidateModel(m_model, bodyChanges, shapeChanges);
    // Existing runtime state, force and target ports are retained unless an
    // explicit caller mutation updates them in this same RHI packet.
    AppendModelUpload(additionalUpdates, bodyBuffer, bodyChanges, m_model.bodies);
    AppendModelUpload(additionalUpdates, shapeBuffer, shapeChanges, m_model.shapes);
    auto batch = rhi::PrepareComputeBatch(m_host, std::move(additionalUpdates), std::move(mutations));
    auto transaction = std::unique_ptr<JoltGpuModelUpdate>(
        new JoltGpuModelUpdate(*this, std::move(batch), std::move(bodyChanges), std::move(shapeChanges)));
    m_modelUpdatePending = true;
    return transaction;
}

std::unique_ptr<JoltGpuModelUpdate>
JoltGpuStorage::PrepareModelReplacement(JoltGpuModel replacement,
                                        std::vector<rhi::ComputeBufferUpdate> additionalUpdates,
                                        std::vector<rhi::ComputeDispatchDesc> mutations)
{
    if (m_modelUpdatePending)
        throw std::logic_error("Jolt GPU storage already has an unaccepted model transaction");
    const auto counts = ValidateModel(replacement);
    if (counts.bodies != m_bodyCount || counts.worlds != m_worldCount)
        throw std::invalid_argument("Jolt GPU replacement must retain body capacity and world count");
    if (!m_bodyCount)
        return PrepareModelUpdate({}, {}, std::move(additionalUpdates), std::move(mutations));
    for (const auto &update : additionalUpdates)
        for (auto resource : {Resource::BodyModels, Resource::ShapeModels, Resource::Worlds}) {
            const auto &author = Get(resource);
            if (update.buffer && author && update.buffer->GetWriterStateIdentity() == author->GetWriterStateIdentity())
                throw std::invalid_argument("Jolt GPU replacement owns all author-model uploads");
        }

    auto buffers = m_buffers;
    for (auto resource : {Resource::BodyModels, Resource::ShapeModels, Resource::Worlds, Resource::Bounds,
                          Resource::Contacts, Resource::ContactPatches}) {
        const auto port = PortIndex(resource);
        const uint64_t size = resource == Resource::BodyModels    ? uint64_t(counts.bodies) * sizeof(BodyModel)
                              : resource == Resource::ShapeModels ? uint64_t(counts.shapes) * sizeof(ShapeModel)
                              : resource == Resource::Worlds      ? uint64_t(counts.worlds) * sizeof(WorldRange)
                              : resource == Resource::Bounds      ? uint64_t(counts.shapes) * sizeof(Aabb)
                              : resource == Resource::Contacts    ? uint64_t(counts.contacts) * sizeof(Contact)
                                                                  : uint64_t(counts.patches) * sizeof(ContactPatch);
        buffers[port] = Allocate(m_host, port, size);
    }
    additionalUpdates.push_back(
        {buffers[PortIndex(Resource::BodyModels)], 0, Bytes(replacement.bodies.data(), replacement.bodies.size())});
    if (counts.shapes)
        additionalUpdates.push_back({buffers[PortIndex(Resource::ShapeModels)], 0,
                                     Bytes(replacement.shapes.data(), replacement.shapes.size())});
    additionalUpdates.push_back(
        {buffers[PortIndex(Resource::Worlds)], 0, Bytes(replacement.worlds.data(), replacement.worlds.size())});
    auto batch = rhi::PrepareComputeBatch(m_host, std::move(additionalUpdates), std::move(mutations));
    auto transaction = std::unique_ptr<JoltGpuModelUpdate>(new JoltGpuModelUpdate(
        *this, std::move(batch), std::move(replacement), std::move(buffers),
        {counts.shapes, counts.contacts, counts.patches, counts.maxBodiesPerWorld, counts.maxLeavesPerWorld}));
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

JoltGpuModelUpdate::JoltGpuModelUpdate(JoltGpuStorage &storage, std::shared_ptr<rhi::PreparedComputeBatch> batch,
                                       JoltGpuModel replacement,
                                       std::array<std::shared_ptr<rhi::ComputeBuffer>, 10> replacementBuffers,
                                       std::array<uint32_t, 5> replacementCounts)
    : m_storage(&storage), m_batch(std::move(batch)), m_replacement(std::move(replacement)),
      m_replacementBuffers(std::move(replacementBuffers)), m_replacementCounts(replacementCounts)
{
}

void JoltGpuModelUpdate::Commit() noexcept
{
    if (m_replacement) {
        m_storage->m_model.bodies.swap(m_replacement->bodies);
        m_storage->m_model.shapes.swap(m_replacement->shapes);
        m_storage->m_model.worlds.swap(m_replacement->worlds);
        m_storage->m_buffers.swap(m_replacementBuffers);
        m_storage->m_shapeCount = m_replacementCounts[0];
        m_storage->m_contactCount = m_replacementCounts[1];
        m_storage->m_patchCount = m_replacementCounts[2];
        m_storage->m_maxBodiesPerWorld = m_replacementCounts[3];
        m_storage->m_maxLeavesPerWorld = m_replacementCounts[4];
    }
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
    const auto slot = InputIndex(resource);
    if (slot >= m_buffers.size())
        throw std::invalid_argument("Jolt GPU tick inputs accept only parameters, models, worlds and forces");
    const auto &buffer = storage.Get(resource);
    m_buffers[slot] = buffer ? buffer->Freeze() : nullptr;
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
        const bool writer = binding.access != rhi::ComputeBufferAccess::Read;
        const auto input = InputIndex(Ports[binding.slot].resource);
        const auto &buffer =
            !writer && input < inputs.m_buffers.size() ? inputs.m_buffers[input] : storage.m_buffers[binding.slot];
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
    case Domain::Shapes:
        dispatch.groupCountX = Groups(storage.GetShapeCount(), definition.groupSizeX);
        break;
    case Domain::WorldPairs:
        dispatch.groupCountX = Groups(storage.GetMaxLeavesPerWorld(), definition.groupSizeX);
        dispatch.groupCountY = Groups(storage.GetMaxLeavesPerWorld(), definition.groupSizeY);
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
        const auto &entry = m_entries[index];
        if ((entry.definition.domain == Domain::Shapes && !storage.GetShapeCount()) ||
            (entry.definition.domain == Domain::WorldPairs && !storage.GetMaxLeavesPerWorld()) ||
            (entry.definition.domain == Domain::Contacts && !storage.GetContactCount()) ||
            (entry.definition.domain == Domain::Worlds &&
             (!storage.GetWorldCount() ||
              ((stage == Stage::SolveVelocity || stage == Stage::SolvePosition) && !storage.GetPatchCount()))))
            continue;
        dispatches.push_back(BuildDispatch(entry, storage, inputs));
    }
    if (dispatches.empty())
        return {};
    return rhi::PrepareComputeBatch(m_host, {}, std::move(dispatches));
}

} // namespace infernux
