#include "JoltGpuPrograms.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>

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
    std::array<std::string_view, 5> fields;
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
    {Resource::ShapeModels, "shape_models", sizeof(ShapeModel), {"geometry"}, 0},
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

size_t PortIndex(Resource resource)
{
    const auto port = std::find_if(std::begin(Ports), std::end(Ports),
                                   [resource](const Port &value) { return value.resource == resource; });
    if (port == std::end(Ports))
        throw std::invalid_argument("Unknown Jolt GPU resource");
    return static_cast<size_t>(port - std::begin(Ports));
}

template <class T> std::vector<uint8_t> Bytes(const T *values, size_t count)
{
    std::vector<uint8_t> bytes(count * sizeof(T));
    if (!bytes.empty())
        std::memcpy(bytes.data(), values, bytes.size());
    return bytes;
}

uint32_t Groups(uint32_t count, uint32_t width)
{
    return count / width + static_cast<uint32_t>(count % width != 0);
}
} // namespace

JoltGpuStorage::JoltGpuStorage(rhi::ComputeHost &host, JoltGpuModel modelData)
    : m_host(host), m_model(std::move(modelData))
{
    const auto &model = m_model;
    if (model.bodies.size() > std::numeric_limits<uint32_t>::max() ||
        model.worlds.size() > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("Jolt GPU model exceeds the device index range");
    m_bodyCount = static_cast<uint32_t>(model.bodies.size());
    m_worldCount = static_cast<uint32_t>(model.worlds.size());
    uint64_t bodyEnd = 0, contactEnd = 0;
    for (size_t world = 0; world < model.worlds.size(); ++world) {
        const auto range = model.worlds[world].range;
        if (range.x != bodyEnd || range.z != contactEnd || range.y > MaxBodiesPerWorld ||
            range.w != range.y * range.y || bodyEnd + range.y > m_bodyCount)
            throw std::invalid_argument("Jolt GPU bootstrap requires contiguous bounded world ranges");
        for (uint32_t body = range.x; body < range.x + range.y; ++body) {
            const auto &record = model.bodies[body];
            if (record.meta.w != world || record.meta.x >= model.shapes.size() ||
                (record.meta.y != ShapeSphere && record.meta.y != ShapeBox) ||
                (record.meta.z != MotionStatic && record.meta.z != MotionDynamic) ||
                (record.meta.y == ShapeBox && record.meta.z != MotionStatic))
                throw std::invalid_argument("Unsupported body in the Jolt GPU bootstrap model");
            const auto geometry = model.shapes[record.meta.x].geometry;
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
            } else if (!Finite4(geometry) || geometry.x <= 0 || geometry.y <= 0 || geometry.z <= 0 || geometry.w < 0 ||
                       geometry.w > std::min({geometry.x, geometry.y, geometry.z})) {
                throw std::invalid_argument("Jolt GPU box dimensions or convex radius are invalid");
            }
        }
        bodyEnd += range.y;
        contactEnd += range.w;
        m_maxBodiesPerWorld = std::max(m_maxBodiesPerWorld, range.y);
    }
    if (bodyEnd != m_bodyCount || contactEnd > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("Jolt GPU ranges do not cover the model");
    m_contactCount = static_cast<uint32_t>(contactEnd);
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
        const bool mixed = port == PortIndex(Resource::StepParams) || port == PortIndex(Resource::BodyModels) ||
                           port == PortIndex(Resource::Worlds) || port == PortIndex(Resource::Contacts);
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
    return m_buffers[PortIndex(resource)];
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
        return rhi::PrepareComputeBatch(m_host, {}, {});
    std::vector<rhi::ComputeBufferUpdate> updates{
        {Get(Resource::BodyModels), 0, Bytes(m_model.bodies.data(), m_model.bodies.size())},
        {Get(Resource::ShapeModels), 0, Bytes(m_model.shapes.data(), m_model.shapes.size())},
        {Get(Resource::Worlds), 0, Bytes(m_model.worlds.data(), m_model.worlds.size())},
        {Get(Resource::BodyForces), 0, Bytes(forces.data(), forces.size())},
        {Get(Resource::BodyStates), 0, Bytes(states.data(), states.size())},
    };
    return rhi::PrepareComputeBatch(m_host, std::move(updates), {});
}

std::shared_ptr<rhi::PreparedComputeBatch> JoltGpuStorage::PrepareParameters(StepParams parameters)
{
    if (!Finite4(parameters.gravityDt) || parameters.gravityDt.w <= 0.0f || parameters.solve.y == 0 ||
        !Finite4(parameters.contact) || parameters.contact.x < 0 || parameters.contact.y < 0 ||
        parameters.contact.z < 0 || parameters.contact.w < 0 || parameters.contact.w > 1 ||
        !std::isfinite(parameters.restitution.x) || parameters.restitution.x < 0)
        throw std::invalid_argument("Jolt GPU step duration and contact parameters are invalid");
    if (m_bodyCount == 0)
        return rhi::PrepareComputeBatch(m_host, {}, {});
    parameters.counts.x = m_bodyCount;
    parameters.counts.y = m_worldCount;
    parameters.counts.z = m_maxBodiesPerWorld;
    parameters.solve.z = m_contactCount;
    return rhi::PrepareComputeBatch(m_host, {{Get(Resource::StepParams), 0, Bytes(&parameters, 1)}}, {});
}

std::shared_ptr<rhi::PreparedComputeBatch> JoltGpuStorage::PrepareForces(const std::vector<BodyForces> &forces)
{
    if (forces.size() != m_bodyCount)
        throw std::invalid_argument("Jolt GPU force input does not match its resident model");
    for (const auto &force : forces)
        if (!Finite3(force.force) || !Finite3(force.torque))
            throw std::invalid_argument("Jolt GPU force and torque must be finite");
    if (m_bodyCount == 0)
        return rhi::PrepareComputeBatch(m_host, {}, {});
    return rhi::PrepareComputeBatch(m_host, {{Get(Resource::BodyForces), 0, Bytes(forces.data(), forces.size())}}, {});
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
            entry.resources.push_back(port->resource);
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
        m_entries.push_back(std::move(entry));
    }
}

std::shared_ptr<rhi::PreparedComputeBatch> JoltGpuPrograms::PrepareStage(Stage stage, const JoltGpuStorage &storage)
{
    if (!m_host.SharesServicesWith(storage.GetHost()))
        throw std::invalid_argument("Jolt GPU programs and state belong to different compute owners");
    const auto entry = std::find_if(m_entries.begin(), m_entries.end(),
                                    [stage](const Entry &value) { return value.definition.stage == stage; });
    if (entry == m_entries.end())
        throw std::invalid_argument("Unknown Jolt GPU stage");
    if (storage.GetBodyCount() == 0)
        return rhi::PrepareComputeBatch(m_host, {}, {});

    rhi::ComputeDispatchDesc dispatch;
    dispatch.kernel = entry->kernel;
    rhi::ComputeDispatchProof proof;
    for (size_t index = 0; index < entry->resources.size(); ++index) {
        const auto resource = entry->resources[index];
        const auto &buffer = storage.Get(resource);
        dispatch.buffers.push_back(buffer);
        const bool read = (entry->definition.reads & Mask(resource)) != 0;
        const bool write = (entry->definition.writes & Mask(resource)) != 0;
        dispatch.bufferAccesses.push_back(read && write ? rhi::ComputeBufferAccess::ReadWrite
                                          : write       ? rhi::ComputeBufferAccess::Write
                                                        : rhi::ComputeBufferAccess::Read);
        if (read)
            proof.reads.push_back({static_cast<uint32_t>(index), 0, buffer->GetByteSize()});
        // Bounds and pair construction overwrite every slot. Read/write
        // solvers preserve already initialized values and may skip inactive
        // bodies/contacts; they must not claim stores they never execute.
        if (write && !read)
            proof.writes.push_back({static_cast<uint32_t>(index), 0, buffer->GetByteSize()});
    }
    dispatch.proof = std::move(proof);
    const auto &definition = entry->definition;
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
    return rhi::PrepareComputeBatch(m_host, {}, {std::move(dispatch)});
}

} // namespace infernux
