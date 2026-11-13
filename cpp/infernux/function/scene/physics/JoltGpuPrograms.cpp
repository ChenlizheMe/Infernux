#include "JoltGpuPrograms.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
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
    {Resource::Joints, "joint_stream", sizeof(QueryRecordWord), {"value"}, 1},
    {Resource::StageWork, "stage_work", sizeof(StageWork), {"range"}, 1},
    {Resource::ConstraintSchedule, "constraint_schedule", sizeof(ConstraintScheduleWord), {"value"}, 1},
};

constexpr size_t PortCount = sizeof(Ports) / sizeof(Ports[0]);
constexpr size_t StoragePortCount = 11;
constexpr size_t WorkUniformCacheCapacity = 64;
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
    case Resource::Joints:
        return 10;
    case Resource::StageWork:
        return 11;
    case Resource::ConstraintSchedule:
        return 12;
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
            (CompleteWrites(stage.stage) & stage.reads) != 0 || !(stage.reads & Mask(Resource::StageWork)) ||
            (stage.writes & Mask(Resource::StageWork)))
            return false;
    }
    return true;
}

static_assert(PortCount == StoragePortCount + 2, "Jolt GPU has eleven persistent ports, work uniform and schedule");
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
    uint32_t bodies, worlds, shapes, maxBodiesPerWorld = 0, maxLeavesPerWorld = 0;
};

template <class Change, class Record>
const Record &ModelRecord(const std::vector<Record> &records, const std::vector<Change> &changes, uint32_t index)
{
    const auto change = std::lower_bound(changes.begin(), changes.end(), index,
                                         [](const auto &value, uint32_t key) { return value.index < key; });
    return change != changes.end() && change->index == index ? change->model : records[index];
}

ModelCounts ValidateModel(const JoltGpuModel &model, const std::vector<JoltGpuBodyModelUpdate> &bodyChanges = {},
                          const std::vector<JoltGpuShapeModelUpdate> &shapeChanges = {},
                          const std::vector<JoltGpuJointModelUpdate> &jointChanges = {})
{
    constexpr auto limit = std::numeric_limits<uint32_t>::max();
    if (model.bodies.size() > limit || model.worlds.size() > limit || model.shapes.size() > limit ||
        model.joints.size() > (limit - 1u) / 24u)
        throw std::invalid_argument("Jolt GPU model exceeds the device index range");
    ModelCounts counts{static_cast<uint32_t>(model.bodies.size()), static_cast<uint32_t>(model.worlds.size()),
                       static_cast<uint32_t>(model.shapes.size())};
    uint64_t bodyEnd = 0, leafEnd = 0, jointEnd = 0;
    for (size_t world = 0; world < model.worlds.size(); ++world) {
        const auto range = model.worlds[world].bodies;
        const auto leaves = model.worlds[world].leaves;
        if (range.x != bodyEnd || leaves.x != leafEnd || range.z || range.w || bodyEnd + range.y > counts.bodies ||
            leafEnd + leaves.y > counts.shapes || leaves.z != jointEnd || jointEnd + leaves.w > model.joints.size())
            throw std::invalid_argument("Jolt GPU model requires contiguous author ranges and zero contact fields");
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
                if (shape.meta.y != body || (shape.meta.w & ~ShapeSensor) || shape.surface.x >= 32 ||
                    shape.surface.z > MaterialCombineMaximum || shape.surface.w > MaterialCombineMaximum ||
                    shape.source.w > 32)
                    throw std::invalid_argument("Jolt GPU leaf identity, flags or material mode is invalid");
                if (!Finite4(shape.localPositionCOM) || shape.localPositionCOM.w != 0 ||
                    !Normalized(shape.localRotation) || !Finite4(shape.material) || shape.material.x < 0 ||
                    shape.material.y < 0 || shape.material.y > 1 || shape.material.z != 0 || shape.material.w != 0)
                    throw std::invalid_argument("Jolt GPU leaf pose or material is invalid");
                const auto geometry = shape.geometry;
                switch (shape.meta.x) {
                case ShapeSphere:
                    if (!std::isfinite(geometry.w) || geometry.w <= 0)
                        throw std::invalid_argument("Jolt GPU sphere radius must be positive and finite");
                    break;
                case ShapeBox:
                    if (!Finite4(geometry) || geometry.x <= 0 || geometry.y <= 0 || geometry.z <= 0 || geometry.w < 0 ||
                        geometry.w > std::min({geometry.x, geometry.y, geometry.z}))
                        throw std::invalid_argument("Jolt GPU box dimensions or convex radius are invalid");
                    break;
                case ShapeCapsule:
                    if (!Finite4(geometry) || geometry.x != 0 || geometry.z != 0 || geometry.y <= 0 || geometry.w <= 0)
                        throw std::invalid_argument("Jolt GPU capsule half cylinder and radius must be positive and "
                                                    "finite with zero unused dimensions");
                    break;
                default:
                    throw std::invalid_argument("Jolt GPU leaf has an unsupported primitive type");
                }
            }
            bodyLeafEnd += record.meta.y;
        }
        if (bodyLeafEnd != leafEnd + leaves.y)
            throw std::invalid_argument("Jolt GPU body leaf ranges do not cover their world");
        for (uint32_t index = leaves.z; index < jointEnd + leaves.w; ++index) {
            const auto change = std::lower_bound(jointChanges.begin(), jointChanges.end(), index,
                                                 [](const auto &value, uint32_t key) { return value.index < key; });
            const auto &joint =
                change != jointChanges.end() && change->index == index ? change->author : model.joints[index];
            const auto inWorld = [&](uint32_t slot) { return slot >= range.x && uint64_t(slot) < bodyEnd + range.y; };
            if (joint.identity.z != world || joint.identity.w || (!joint.identity.x && !joint.identity.y) ||
                !inWorld(joint.bodies.y) ||
                (joint.bodies.x != UINT32_MAX && (!inWorld(joint.bodies.x) || joint.bodies.x == joint.bodies.y)) ||
                joint.bodies.z > 1 || (joint.bodies.w & ~15u))
                throw std::invalid_argument("Jolt GPU joint identity, endpoints or world range is invalid");
            for (const auto &value : {joint.ownerAnchor, joint.ownerAxis, joint.ownerLocalAnchor, joint.rootCOM1,
                                      joint.rootCOM2, joint.scale1, joint.scale2})
                if (!Finite4(value) || value.w != 0)
                    throw std::invalid_argument("Jolt GPU joint anchor, axis, COM and scale must be finite");
            const auto &axis = joint.ownerAxis;
            if (joint.bodies.z == 1 && std::abs(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z - 1.0f) > 1e-4f)
                throw std::invalid_argument("Jolt GPU hinge axis must be normalized");
            if (!Finite4(joint.limits) || joint.limits.x > 0 || joint.limits.y < 0 ||
                joint.limits.x < -3.14159265358979323846f || joint.limits.y > 3.14159265358979323846f ||
                joint.limits.z || joint.limits.w || !Finite4(joint.motor) || joint.motor.y < 0 || joint.motor.z < 0 ||
                joint.motor.w < 0 || !Finite4(joint.passive) || joint.passive.x < 0 || joint.passive.y < 0 ||
                joint.passive.z < 0 || joint.passive.w)
                throw std::invalid_argument("Jolt GPU joint limits, motor or passive settings are invalid");
        }
        bodyEnd += range.y;
        leafEnd += leaves.y;
        jointEnd += leaves.w;
        counts.maxBodiesPerWorld = std::max(counts.maxBodiesPerWorld, range.y);
        counts.maxLeavesPerWorld = std::max(counts.maxLeavesPerWorld, leaves.y);
    }
    if (bodyEnd != counts.bodies || leafEnd != counts.shapes || jointEnd != model.joints.size())
        throw std::invalid_argument("Jolt GPU ranges do not cover the model");
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
    const bool uniform = port.resource == Resource::StepParams || port.resource == Resource::StageWork;
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
                       const std::vector<Record> &records, Record Change::*member, uint64_t baseOffset = 0)
{
    if (changes.empty())
        return;
    const auto first = changes.front().index;
    const size_t count = size_t(changes.back().index) - first + 1;
    auto bytes = Bytes(records.data() + first, count);
    // Allocate only the upload envelope; its holes retain accepted author data.
    for (const auto &change : changes)
        std::memcpy(bytes.data() + size_t(change.index - first) * sizeof(Record), &(change.*member), sizeof(Record));
    updates.push_back({buffer, baseOffset + uint64_t(first) * sizeof(Record), std::move(bytes)});
}

uint32_t Groups(uint32_t count, uint32_t width)
{
    return count / width + static_cast<uint32_t>(count % width != 0);
}

bool IslandDomain(Domain domain)
{
    return domain == Domain::IslandPatches || domain == Domain::IslandContacts || domain == Domain::IslandJoints;
}

const JoltGpuContactWorkset &RequireContacts(const JoltGpuStepInputs &inputs)
{
    if (!inputs.GetContacts())
        throw std::invalid_argument("Jolt GPU contact work requires its substep workset");
    return *inputs.GetContacts();
}

bool ValidateWork(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs, const StageDefinition &definition,
                  const StageWork &work)
{
    const auto range = work.range;
    const uint32_t quantum = WorkItemLimit(definition.domain);
    if (range.y > quantum)
        throw std::invalid_argument("Jolt GPU work exceeds its bounded item quantum");
    if (IslandDomain(definition.domain)) {
        const auto &constraints = inputs.GetConstraints();
        if (!constraints)
            throw std::invalid_argument("Jolt GPU island work requires its validated substep plan");
        const bool joints = definition.domain == Domain::IslandJoints;
        const uint32_t tableCount = joints ? constraints->jointIslandCount : constraints->contactIslandCount;
        uint64_t maximum = joints ? constraints->maxJointItems : constraints->maxPatchItems;
        if (definition.domain == Domain::IslandContacts)
            maximum *= MaxContactPointsPerPair;
        if (range.z > tableCount || range.w > tableCount - range.z)
            throw std::invalid_argument("Jolt GPU island work addresses outside its installed island table");
        if (uint64_t(range.w) * range.y > quantum)
            throw std::invalid_argument("Jolt GPU island work exceeds its total rectangular item quantum");
        if (range.x > maximum || uint64_t(range.y) > maximum - range.x)
            throw std::invalid_argument("Jolt GPU island work addresses outside its accepted summary maximum");
        return range.w != 0 && range.y != 0;
    }
    if (range.z || range.w)
        throw std::invalid_argument("Jolt GPU global work must leave its unused island range zero");
    uint64_t count = 0;
    switch (definition.domain) {
    case Domain::Bodies:
        count = storage.GetBodyCount();
        break;
    case Domain::Shapes:
        count = storage.GetShapeCount();
        break;
    case Domain::Contacts:
        count = uint64_t(RequireContacts(inputs).patchCount) * MaxContactPointsPerPair;
        break;
    case Domain::Patches:
        count = RequireContacts(inputs).patchCount;
        break;
    case Domain::Joints:
        count = storage.GetJointCount();
        break;
    default:
        throw std::invalid_argument("Jolt GPU work has an unknown global domain");
    }
    if (range.x > count || range.y > count - range.x)
        throw std::invalid_argument("Jolt GPU work addresses global items outside its accepted model");
    return range.y != 0;
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
    if (!model.joints.empty())
        throw std::invalid_argument("Jolt GPU initial joint topology requires an ordered GPU initialization");
    m_bodyCount = counts.bodies;
    m_worldCount = counts.worlds;
    m_shapeCount = counts.shapes;
    m_maxBodiesPerWorld = counts.maxBodiesPerWorld;
    m_maxLeavesPerWorld = counts.maxLeavesPerWorld;
    if (m_bodyCount == 0)
        return;

    const uint64_t sizes[] = {sizeof(StepParams),
                              model.bodies.size() * sizeof(BodyModel),
                              std::max<size_t>(1, model.shapes.size()) * sizeof(ShapeModel),
                              model.worlds.size() * sizeof(WorldRange),
                              uint64_t(m_bodyCount) * sizeof(BodyForces),
                              uint64_t(m_bodyCount) * sizeof(BodyState),
                              uint64_t(m_shapeCount) * sizeof(Aabb),
                              sizeof(Contact),
                              uint64_t(m_bodyCount) * sizeof(KinematicTarget),
                              sizeof(ContactPatch),
                              sizeof(QueryRecordWord)};
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
    if (slot >= StoragePortCount)
        throw std::invalid_argument("Jolt GPU storage requires a persistent resource, not a per-work uniform");
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
    const UInt4 jointHeader{0, 1, 0, 0};
    updates.push_back({Get(Resource::Joints), 0, Bytes(&jointHeader, 1)});
    if (m_shapeCount)
        updates.push_back({Get(Resource::ShapeModels), 0, Bytes(m_model.shapes.data(), m_model.shapes.size())});
    else
        updates.push_back({Get(Resource::ShapeModels), 0, std::vector<uint8_t>(sizeof(ShapeModel))});
    updates.push_back({Get(Resource::Contacts), 0, std::vector<uint8_t>(sizeof(Contact))});
    updates.push_back({Get(Resource::ContactPatches), 0, std::vector<uint8_t>(sizeof(ContactPatch))});
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
    parameters.solve.z = parameters.solve.w = 0;
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
    std::vector<rhi::ComputeBufferUpdate> additionalUpdates, std::vector<rhi::ComputeDispatchDesc> mutations,
    std::vector<JoltGpuJointModelUpdate> jointChanges)
{
    if (bodyChanges.empty() && shapeChanges.empty() && jointChanges.empty() && additionalUpdates.empty() &&
        mutations.empty())
        return {};
    if (m_modelUpdatePending)
        throw std::logic_error("Jolt GPU storage already has an unaccepted model transaction");

    const auto &bodyBuffer = Get(Resource::BodyModels);
    const auto &shapeBuffer = Get(Resource::ShapeModels);
    const auto &worldBuffer = Get(Resource::Worlds);
    const auto &jointBuffer = Get(Resource::Joints);
    for (const auto &update : additionalUpdates) {
        if (update.buffer &&
            ((bodyBuffer && update.buffer->GetWriterStateIdentity() == bodyBuffer->GetWriterStateIdentity()) ||
             (shapeBuffer && update.buffer->GetWriterStateIdentity() == shapeBuffer->GetWriterStateIdentity()) ||
             (worldBuffer && update.buffer->GetWriterStateIdentity() == worldBuffer->GetWriterStateIdentity()) ||
             (jointBuffer && update.buffer->GetWriterStateIdentity() == jointBuffer->GetWriterStateIdentity() &&
              update.offset < sizeof(UInt4) + m_model.joints.size() * sizeof(JointAuthor))))
            throw std::invalid_argument("Jolt GPU model ports must be updated through their author-model delta");
    }

    ValidateModelChangeIndices(bodyChanges, m_model.bodies.size());
    ValidateModelChangeIndices(shapeChanges, m_model.shapes.size());
    ValidateModelChangeIndices(jointChanges, m_model.joints.size());
    for (const auto &change : jointChanges) {
        const auto &old = m_model.joints[change.index];
        if (std::memcmp(&old.bodies, &change.author.bodies, 3 * sizeof(uint32_t)) ||
            std::memcmp(&old.identity, &change.author.identity, sizeof(UInt4)))
            throw std::invalid_argument("Jolt GPU joint endpoint or identity changes require topology replacement");
    }
    if (!bodyChanges.empty() || !shapeChanges.empty() || !jointChanges.empty())
        (void)ValidateModel(m_model, bodyChanges, shapeChanges, jointChanges);
    // Existing runtime state, force and target ports are retained unless an
    // explicit caller mutation updates them in this same RHI packet.
    AppendModelUpload(additionalUpdates, bodyBuffer, bodyChanges, m_model.bodies, &JoltGpuBodyModelUpdate::model);
    AppendModelUpload(additionalUpdates, shapeBuffer, shapeChanges, m_model.shapes, &JoltGpuShapeModelUpdate::model);
    AppendModelUpload(additionalUpdates, jointBuffer, jointChanges, m_model.joints, &JoltGpuJointModelUpdate::author,
                      sizeof(UInt4));
    auto batch = rhi::PrepareComputeBatch(m_host, std::move(additionalUpdates), std::move(mutations));
    auto transaction = std::unique_ptr<JoltGpuModelUpdate>(new JoltGpuModelUpdate(
        *this, std::move(batch), std::move(bodyChanges), std::move(shapeChanges), std::move(jointChanges)));
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
    // Leaf topology replacement cannot recreate numerical joint state or
    // overwrite accepted joint membership with the caller's geometry model.
    replacement.joints = m_model.joints;
    if (replacement.worlds.size() == m_model.worlds.size())
        for (size_t index = 0; index < replacement.worlds.size(); ++index) {
            replacement.worlds[index].leaves.z = m_model.worlds[index].leaves.z;
            replacement.worlds[index].leaves.w = m_model.worlds[index].leaves.w;
        }
    const auto counts = ValidateModel(replacement);
    if (counts.bodies != m_bodyCount || counts.worlds != m_worldCount)
        throw std::invalid_argument("Jolt GPU replacement must retain body capacity and world count");
    if (!m_bodyCount)
        return PrepareModelUpdate({}, {}, std::move(additionalUpdates), std::move(mutations));
    for (const auto &update : additionalUpdates)
        for (auto resource : {Resource::BodyModels, Resource::ShapeModels, Resource::Worlds, Resource::Joints}) {
            const auto &author = Get(resource);
            if (update.buffer && author && update.buffer->GetWriterStateIdentity() == author->GetWriterStateIdentity())
                throw std::invalid_argument("Jolt GPU replacement owns all author-model uploads");
        }

    auto buffers = m_buffers;
    for (auto resource : {Resource::BodyModels, Resource::ShapeModels, Resource::Worlds, Resource::Bounds}) {
        const auto port = PortIndex(resource);
        const uint64_t size = resource == Resource::BodyModels ? uint64_t(counts.bodies) * sizeof(BodyModel)
                              : resource == Resource::ShapeModels
                                  ? uint64_t(std::max(1u, counts.shapes)) * sizeof(ShapeModel)
                              : resource == Resource::Worlds ? uint64_t(counts.worlds) * sizeof(WorldRange)
                                                             : uint64_t(counts.shapes) * sizeof(Aabb);
        buffers[port] = Allocate(m_host, port, size);
    }
    additionalUpdates.push_back(
        {buffers[PortIndex(Resource::BodyModels)], 0, Bytes(replacement.bodies.data(), replacement.bodies.size())});
    if (counts.shapes)
        additionalUpdates.push_back({buffers[PortIndex(Resource::ShapeModels)], 0,
                                     Bytes(replacement.shapes.data(), replacement.shapes.size())});
    else
        additionalUpdates.push_back(
            {buffers[PortIndex(Resource::ShapeModels)], 0, std::vector<uint8_t>(sizeof(ShapeModel))});
    additionalUpdates.push_back(
        {buffers[PortIndex(Resource::Worlds)], 0, Bytes(replacement.worlds.data(), replacement.worlds.size())});
    auto batch = rhi::PrepareComputeBatch(m_host, std::move(additionalUpdates), std::move(mutations));
    auto transaction = std::unique_ptr<JoltGpuModelUpdate>(
        new JoltGpuModelUpdate(*this, std::move(batch), std::move(replacement), std::move(buffers),
                               {counts.shapes, counts.maxBodiesPerWorld, counts.maxLeavesPerWorld}));
    m_modelUpdatePending = true;
    return transaction;
}

std::unique_ptr<JoltGpuModelUpdate> JoltGpuStorage::PrepareBodyGrowth(uint32_t capacity,
                                                                      JoltGpuBodyGrowthBuffers candidate)
{
    if (m_modelUpdatePending)
        throw std::logic_error("Jolt GPU storage already has an unaccepted model transaction");
    if (!m_bodyCount || capacity <= m_bodyCount || !m_worldCount || candidate.bodies.count != capacity)
        throw std::invalid_argument("Jolt GPU growth requires a larger body capacity in its accepted world domain");

    auto replacement = m_model;
    BodyModel unused{};
    unused.meta = {m_shapeCount, 0, MotionStatic | MotionNotResident, m_worldCount - 1};
    unused.inertiaRotation.w = 1;
    replacement.bodies.resize(capacity, unused);
    replacement.worlds.back().bodies.y += capacity - m_bodyCount;
    const auto counts = ValidateModel(replacement);
    auto buffers = m_buffers;
    for (const auto &[resource, buffer] :
         {std::pair{Resource::BodyModels, candidate.models}, std::pair{Resource::Worlds, candidate.worlds},
          std::pair{Resource::BodyStates, candidate.bodies.states},
          std::pair{Resource::BodyForces, candidate.bodies.forces},
          std::pair{Resource::KinematicTargets, candidate.bodies.targets}}) {
        const auto port = PortIndex(resource);
        const auto count = resource == Resource::Worlds ? m_worldCount : capacity;
        const auto &previous = m_buffers[port];
        if (!buffer || buffer->IsFrozen() || !m_host.SharesServicesWith(buffer->GetHost()) ||
            buffer->GetByteSize() != uint64_t(count) * Ports[port].stride ||
            buffer->GetDesc().scalarType != previous->GetDesc().scalarType ||
            buffer->GetDesc().lanes != previous->GetDesc().lanes)
            throw std::invalid_argument("Jolt GPU growth endpoints differ from their body domain layout");
        if (resource != Resource::BodyModels && resource != Resource::Worlds &&
            !buffer->HasInitializedRange(0, buffer->GetByteSize()))
            throw std::logic_error("Jolt GPU growth requires its completed numerical prefix and tail");
        buffers[port] = buffer;
    }
    auto batch =
        rhi::PrepareComputeBatch(m_host,
                                 {{candidate.models, 0, Bytes(replacement.bodies.data(), replacement.bodies.size())},
                                  {candidate.worlds, 0, Bytes(replacement.worlds.data(), replacement.worlds.size())}},
                                 {});
    auto transaction = std::unique_ptr<JoltGpuModelUpdate>(
        new JoltGpuModelUpdate(*this, std::move(batch), std::move(replacement), std::move(buffers),
                               {counts.shapes, counts.maxBodiesPerWorld, counts.maxLeavesPerWorld}));
    m_modelUpdatePending = true;
    return transaction;
}

std::unique_ptr<JoltGpuModelUpdate>
JoltGpuStorage::PrepareJointReplacement(std::vector<JointAuthor> joints, std::shared_ptr<rhi::ComputeBuffer> newStream,
                                        std::vector<rhi::ComputeBufferUpdate> uploads,
                                        std::vector<rhi::ComputeDispatchDesc> remapInit)
{
    if (m_modelUpdatePending)
        throw std::logic_error("Jolt GPU storage already has an unaccepted model transaction");
    JoltGpuModel replacement = m_model;
    replacement.joints = std::move(joints);
    size_t joint = 0;
    for (size_t world = 0; world < replacement.worlds.size(); ++world) {
        auto &range = replacement.worlds[world].leaves;
        range.z = static_cast<uint32_t>(joint);
        while (joint < replacement.joints.size() && replacement.joints[joint].identity.z == world)
            ++joint;
        range.w = static_cast<uint32_t>(joint - range.z);
    }
    const auto counts = ValidateModel(replacement);
    const uint64_t authorEnd = sizeof(UInt4) + uint64_t(replacement.joints.size()) * sizeof(JointAuthor);
    const uint64_t streamBytes = authorEnd + uint64_t(replacement.joints.size()) * sizeof(JointState);
    if (!newStream || newStream->IsFrozen() || !m_host.SharesServicesWith(newStream->GetHost()) ||
        newStream->GetByteSize() != streamBytes || newStream->GetDesc().scalarType != rhi::ComputeScalarType::UInt32)
        throw std::invalid_argument("Jolt GPU replacement stream has invalid ownership, mutability, size or type");
    for (const auto &buffer : m_buffers)
        if (buffer && buffer->GetWriterStateIdentity() == newStream->GetWriterStateIdentity())
            throw std::invalid_argument("Jolt GPU joint topology replacement requires a new stream");

    bool authorUploaded = false;
    const UInt4 expectedHeader{static_cast<uint32_t>(replacement.joints.size()),
                               static_cast<uint32_t>(authorEnd / sizeof(UInt4)), 0, 0};
    for (const auto &upload : uploads) {
        if (!upload.buffer)
            throw std::invalid_argument("Jolt GPU replacement upload requires a buffer");
        for (const auto &buffer : m_buffers)
            if (buffer && buffer->GetWriterStateIdentity() == upload.buffer->GetWriterStateIdentity())
                throw std::invalid_argument("Jolt GPU joint replacement cannot overwrite accepted scene resources");
        if (upload.buffer->GetWriterStateIdentity() != newStream->GetWriterStateIdentity())
            continue;
        if (authorUploaded || upload.offset || upload.data.size() != authorEnd ||
            std::memcmp(upload.data.data(), &expectedHeader, sizeof(expectedHeader)) ||
            (!replacement.joints.empty() &&
             std::memcmp(upload.data.data() + sizeof(UInt4), replacement.joints.data(), authorEnd - sizeof(UInt4))))
            throw std::invalid_argument("Jolt GPU replacement must upload exactly its header and author prefix once");
        authorUploaded = true;
    }
    if (!authorUploaded)
        throw std::invalid_argument("Jolt GPU replacement is missing its accepted author prefix");
    std::vector<std::pair<uint64_t, uint64_t>> initialized;
    for (const auto &dispatch : remapInit) {
        for (size_t binding = 0; binding < dispatch.buffers.size(); ++binding) {
            const auto &buffer = dispatch.buffers[binding];
            if (!buffer || buffer->GetWriterStateIdentity() != newStream->GetWriterStateIdentity())
                continue;
            if (binding >= dispatch.bufferAccesses.size() || !dispatch.proof)
                throw std::invalid_argument("Jolt GPU replacement runtime requires explicit range proofs");
            for (const auto &write : dispatch.proof->writes) {
                if (write.binding != binding)
                    continue;
                if (write.offset < authorEnd || write.offset > streamBytes ||
                    write.byteSize > streamBytes - write.offset)
                    throw std::invalid_argument("Jolt GPU initialization cannot overwrite its author prefix");
                initialized.emplace_back(write.offset, write.offset + write.byteSize);
            }
        }
    }
    std::sort(initialized.begin(), initialized.end());
    uint64_t initializedEnd = authorEnd;
    for (const auto &[begin, end] : initialized) {
        if (begin > initializedEnd)
            break;
        initializedEnd = std::max(initializedEnd, end);
    }
    if (initializedEnd != streamBytes)
        throw std::invalid_argument("Jolt GPU replacement must initialize every runtime record on the device");
    auto buffers = m_buffers;
    buffers[PortIndex(Resource::Joints)] = std::move(newStream);
    buffers[PortIndex(Resource::Worlds)] =
        Allocate(m_host, PortIndex(Resource::Worlds), uint64_t(counts.worlds) * sizeof(WorldRange));
    if (counts.worlds)
        uploads.push_back(
            {buffers[PortIndex(Resource::Worlds)], 0, Bytes(replacement.worlds.data(), replacement.worlds.size())});
    auto batch = rhi::PrepareComputeBatch(m_host, std::move(uploads), std::move(remapInit));
    auto transaction = std::unique_ptr<JoltGpuModelUpdate>(
        new JoltGpuModelUpdate(*this, std::move(batch), std::move(replacement), std::move(buffers),
                               {counts.shapes, counts.maxBodiesPerWorld, counts.maxLeavesPerWorld}));
    m_modelUpdatePending = true;
    return transaction;
}

JoltGpuModelUpdate::JoltGpuModelUpdate(JoltGpuStorage &storage, std::shared_ptr<rhi::PreparedComputeBatch> batch,
                                       std::vector<JoltGpuBodyModelUpdate> bodyChanges,
                                       std::vector<JoltGpuShapeModelUpdate> shapeChanges,
                                       std::vector<JoltGpuJointModelUpdate> jointChanges)
    : m_storage(&storage), m_batch(std::move(batch)), m_bodyChanges(std::move(bodyChanges)),
      m_shapeChanges(std::move(shapeChanges)), m_jointChanges(std::move(jointChanges))
{
}

JoltGpuModelUpdate::~JoltGpuModelUpdate()
{
    Cancel();
}

JoltGpuModelUpdate::JoltGpuModelUpdate(JoltGpuStorage &storage, std::shared_ptr<rhi::PreparedComputeBatch> batch,
                                       JoltGpuModel replacement,
                                       std::array<std::shared_ptr<rhi::ComputeBuffer>, 11> replacementBuffers,
                                       std::array<uint32_t, 3> replacementCounts)
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
        m_storage->m_model.joints.swap(m_replacement->joints);
        m_storage->m_buffers.swap(m_replacementBuffers);
        m_storage->m_bodyCount = static_cast<uint32_t>(m_storage->m_model.bodies.size());
        m_storage->m_worldCount = static_cast<uint32_t>(m_storage->m_model.worlds.size());
        m_storage->m_shapeCount = m_replacementCounts[0];
        m_storage->m_maxBodiesPerWorld = m_replacementCounts[1];
        m_storage->m_maxLeavesPerWorld = m_replacementCounts[2];
    }
    for (const auto &change : m_bodyChanges)
        m_storage->m_model.bodies[change.index] = change.model;
    for (const auto &change : m_shapeChanges)
        m_storage->m_model.shapes[change.index] = change.model;
    for (const auto &change : m_jointChanges)
        m_storage->m_model.joints[change.index] = change.author;
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
    m_workUniforms.clear();
    m_nextWorkUniform = 0;
}

const std::shared_ptr<rhi::ComputeBuffer> &JoltGpuStepInputs::GetCaptured(Resource resource) const
{
    const auto slot = InputIndex(resource);
    if (slot >= m_buffers.size())
        throw std::invalid_argument("Jolt GPU captured inputs accept only parameters, models, worlds and forces");
    return m_buffers[slot];
}

void JoltGpuStepInputs::SetContacts(std::shared_ptr<const JoltGpuContactWorkset> contacts)
{
    if (!contacts || contacts->patchCount > std::numeric_limits<uint32_t>::max() / MaxContactPointsPerPair ||
        contacts->maxPatchesPerWorld > contacts->patchCount ||
        ((contacts->patchCount == 0) != (contacts->maxPatchesPerWorld == 0)))
        throw std::invalid_argument("Jolt GPU contact workset has invalid logical candidate counts");
    const auto &authorWorlds = GetCaptured(Resource::Worlds);
    if (!authorWorlds)
        throw std::invalid_argument("Jolt GPU contact workset requires its captured author world prefix");
    for (const auto &buffer : {contacts->parameters, contacts->worlds, contacts->contacts, contacts->patches})
        if (!buffer || !authorWorlds->GetHost().SharesServicesWith(buffer->GetHost()))
            throw std::invalid_argument("Jolt GPU contact workset resources must share their captured compute owner");
    const auto contactCount = uint64_t(contacts->patchCount) * MaxContactPointsPerPair;
    if (contacts->parameters->GetByteSize() != sizeof(StepParams) ||
        contacts->worlds->GetByteSize() != authorWorlds->GetByteSize() ||
        contacts->contacts->GetByteSize() != std::max(uint64_t{1}, contactCount) * sizeof(Contact) ||
        contacts->patches->GetByteSize() < sizeof(ContactPatch) ||
        contacts->patches->GetByteSize() % sizeof(ContactPatch))
        throw std::invalid_argument("Jolt GPU contact workset differs from its logical resource layout");
    ClearConstraints();
    m_contacts = std::move(contacts);
}

void JoltGpuStepInputs::ClearContacts() noexcept
{
    m_contacts.reset();
    ClearConstraints();
}

void JoltGpuStepInputs::SetConstraints(std::shared_ptr<const JoltGpuConstraintWorkset> constraints) noexcept
{
    m_constraints = std::move(constraints);
}

void JoltGpuStepInputs::ClearConstraints() noexcept
{
    m_constraints.reset();
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
        uint32_t storageBindings = 0, uniformBindings = 0;
        for (const auto &resource : target.resources) {
            const std::array<uint32_t, 2> key{resource.set, resource.binding.binding};
            if (std::find(code->bindings.begin(), code->bindings.end(), key) == code->bindings.end())
                continue;
            const auto port = std::find_if(std::begin(Ports), std::end(Ports),
                                           [&](const Port &value) { return value.name == resource.name; });
            if (port == std::end(Ports) || resource.set != 0)
                throw std::invalid_argument("Jolt GPU stage contains an unknown resource port");
            ValidateLayout(*port, resource);
            if (resource.binding.type == rhi::BindingType::UniformBuffer)
                ++uniformBindings;
            else
                ++storageBindings;
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
        if (actualResources != (definition.reads | definition.writes) || storageBindings > 8 || uniformBindings > 2)
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
                                                        const JoltGpuStepInputs &inputs, const JoltGpuStageWork &work,
                                                        const std::shared_ptr<rhi::ComputeBuffer> &workUniform) const
{
    rhi::ComputeDispatchDesc dispatch;
    dispatch.kernel = entry.kernel;
    dispatch.buffers.reserve(entry.bindings.size());
    dispatch.bufferAccesses.reserve(entry.bindings.size());
    rhi::ComputeDispatchProof proof;
    proof.reads.reserve(entry.bindings.size());
    proof.writes.reserve(entry.bindings.size());
    const auto range = work.parameters.range;
    for (size_t index = 0; index < entry.bindings.size(); ++index) {
        const auto &binding = entry.bindings[index];
        const bool writer = binding.access != rhi::ComputeBufferAccess::Read;
        const auto resource = Ports[binding.slot].resource;
        const auto input = InputIndex(resource);
        const auto &contacts = inputs.GetContacts();
        const auto &constraints = inputs.GetConstraints();
        const auto &buffer = resource == Resource::StageWork ? workUniform
                             : resource == Resource::ConstraintSchedule
                                 ? (constraints ? constraints->schedule : nullptr)
                             : contacts && resource == Resource::StepParams     ? contacts->parameters
                             : contacts && resource == Resource::Worlds         ? contacts->worlds
                             : contacts && resource == Resource::Contacts       ? contacts->contacts
                             : contacts && resource == Resource::ContactPatches ? contacts->patches
                             : !writer && input < inputs.m_buffers.size()       ? inputs.m_buffers[input]
                                                                                : storage.m_buffers[binding.slot];
        if (!buffer || !m_host.SharesServicesWith(buffer->GetHost()))
            throw std::invalid_argument("Jolt GPU stage requires captured input: " +
                                        std::string(Ports[binding.slot].name));
        dispatch.buffers.push_back(buffer);
        dispatch.bufferAccesses.push_back(binding.access);
        if (binding.read)
            proof.reads.push_back({static_cast<uint32_t>(index), 0, buffer->GetByteSize()});
        if (binding.completeWrite) {
            const auto complete = [&](uint64_t first, uint64_t count, uint64_t stride) {
                const uint64_t offset = first * stride, bytes = count * stride;
                if (offset > buffer->GetByteSize() || bytes > buffer->GetByteSize() - offset)
                    throw std::invalid_argument("Jolt GPU work complete-write range exceeds its typed resource");
                if (bytes)
                    proof.writes.push_back({static_cast<uint32_t>(index), offset, bytes});
            };
            if (resource == Resource::Bounds) {
                complete(range.x, range.y, sizeof(Aabb));
            } else if (resource == Resource::Contacts) {
                complete(uint64_t(range.x) * MaxContactPointsPerPair, uint64_t(range.y) * MaxContactPointsPerPair,
                         sizeof(Contact));
            } else {
                throw std::invalid_argument("Jolt GPU stage lacks a typed bounded complete-write domain");
            }
        }
    }
    dispatch.proof = std::move(proof);
    const auto &definition = entry.definition;
    dispatch.groupCountY = dispatch.groupCountZ = 1;
    if (IslandDomain(definition.domain)) {
        dispatch.groupCountX = Groups(range.w, definition.groupSizeX);
    } else {
        dispatch.groupCountX = Groups(range.y, definition.groupSizeX);
    }
    const auto &maximum = m_host.device.GetCapabilities().limits.maxComputeWorkgroupCount;
    if (!dispatch.groupCountX || !dispatch.groupCountY || !dispatch.groupCountZ || dispatch.groupCountX > maximum[0] ||
        dispatch.groupCountY > maximum[1] || dispatch.groupCountZ > maximum[2])
        throw std::invalid_argument("Jolt GPU bounded work exceeds the device dispatch range");
    return dispatch;
}

std::shared_ptr<rhi::PreparedComputeBatch> JoltGpuPrograms::PrepareStages(const JoltGpuStorage &storage,
                                                                          const JoltGpuStepInputs &inputs,
                                                                          const std::vector<JoltGpuStageWork> &work)
{
    if (!m_host.SharesServicesWith(storage.GetHost()))
        throw std::invalid_argument("Jolt GPU programs and state belong to different compute owners");
    if (storage.GetBodyCount() == 0 || work.empty())
        return {};
    std::vector<rhi::ComputeDispatchDesc> dispatches;
    dispatches.reserve(work.size());
    std::vector<rhi::ComputeBufferUpdate> uploads;
    std::map<std::array<uint32_t, 4>, std::shared_ptr<rhi::ComputeBuffer>> uniforms;
    for (const auto &unit : work) {
        const auto index = static_cast<size_t>(unit.stage);
        if (index >= m_entries.size())
            throw std::invalid_argument("Unknown Jolt GPU stage");
        const auto &entry = m_entries[index];
        if (!ValidateWork(storage, inputs, entry.definition, unit.parameters))
            continue;
        const auto range = unit.parameters.range;
        const std::array<uint32_t, 4> key{range.x, range.y, range.z, range.w};
        auto uniform = uniforms.find(key);
        if (uniform == uniforms.end()) {
            auto cached =
                std::find_if(inputs.m_workUniforms.begin(), inputs.m_workUniforms.end(), [&](const auto &value) {
                    const auto a = value.parameters.range;
                    return a.x == range.x && a.y == range.y && a.z == range.z && a.w == range.w;
                });
            std::shared_ptr<rhi::ComputeBuffer> buffer;
            if (cached != inputs.m_workUniforms.end() && cached->buffer &&
                m_host.SharesServicesWith(cached->buffer->GetHost()) &&
                cached->buffer->HasInitializedRange(0, sizeof(StageWork))) {
                buffer = cached->buffer;
            } else {
                buffer = Allocate(m_host, PortIndex(Resource::StageWork), sizeof(StageWork));
                uploads.push_back({buffer, 0, Bytes(&unit.parameters, 1)});
                const JoltGpuStepInputs::WorkUniform value{unit.parameters, buffer};
                if (cached != inputs.m_workUniforms.end())
                    *cached = value;
                else if (inputs.m_workUniforms.size() < WorkUniformCacheCapacity)
                    inputs.m_workUniforms.push_back(value);
                else {
                    inputs.m_workUniforms[inputs.m_nextWorkUniform] = value;
                    inputs.m_nextWorkUniform = (inputs.m_nextWorkUniform + 1) % WorkUniformCacheCapacity;
                }
            }
            uniform = uniforms.emplace(key, std::move(buffer)).first;
        }
        dispatches.push_back(BuildDispatch(entry, storage, inputs, unit, uniform->second));
    }
    if (dispatches.empty())
        return {};
    return rhi::PrepareComputeBatch(m_host, std::move(uploads), std::move(dispatches));
}

} // namespace infernux
