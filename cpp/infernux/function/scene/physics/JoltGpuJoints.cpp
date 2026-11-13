#include "JoltGpuJoints.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <glm/gtc/constants.hpp>
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

enum Port : uint8_t
{
    Parameters,
    Remap,
    Previous,
    Stream,
    States,
    Rows,
    Motion,
    PortCount
};
enum Stage : uint8_t
{
    Initialize,
    ReadMotion,
    ReadStatus,
    UpdateState,
    StageCount
};
struct PortLayout
{
    std::string_view name;
    uint64_t stride;
    std::array<std::string_view, 4> fields;
    bool scalar;
    bool integer;
    bool readOnly;
};
constexpr PortLayout Ports[] = {
    {"joint_params", sizeof(JointParams), {"counts"}, false, true, true},
    {"joint_remap", sizeof(uint32_t), {}, true, true, true},
    {"old_joint_stream", sizeof(QueryRecordWord), {"value"}, false, true, true},
    {"joint_stream", sizeof(QueryRecordWord), {"value"}, false, true, false},
    {"body_states",
     sizeof(BodyState),
     {"positionCOM", "rotation", "linearVelocity", "angularVelocity"},
     false,
     false,
     true},
    {"joint_query_rows", sizeof(uint32_t), {}, true, true, true},
    {"joint_motion", sizeof(QueryRecordWord), {"value"}, false, true, false},
};
constexpr std::array<std::string_view, StageCount> Entries{"initialize_joints", "read_joint_motion",
                                                           "read_joint_status", "update_joint_state"};
constexpr uint32_t Bit(Port port)
{
    return 1u << port;
}
constexpr std::array<uint32_t, StageCount> StagePorts{
    Bit(Parameters) | Bit(Remap) | Bit(Previous) | Bit(Stream) | Bit(States),
    Bit(Parameters) | Bit(Stream) | Bit(States) | Bit(Rows) | Bit(Motion), Bit(Parameters) | Bit(Stream) | Bit(Motion),
    Bit(Parameters) | Bit(Stream) | Bit(States) | Bit(Rows)};
static_assert(sizeof(JointAuthor) == 12 * sizeof(QueryRecordWord));
static_assert(sizeof(JointState) == 12 * sizeof(QueryRecordWord));

uint64_t StreamBytes(size_t count)
{
    if (count > (std::numeric_limits<uint32_t>::max() - 1u) / (2u * JointRecordWords))
        throw JoltGpuQueryRequestError("Jolt GPU joint stream exceeds device word indexing");
    const uint64_t bytes = sizeof(UInt4) + uint64_t(count) * (sizeof(JointAuthor) + sizeof(JointState));
    if (bytes > std::numeric_limits<size_t>::max())
        throw JoltGpuQueryRequestError("Jolt GPU joint stream exceeds the host address range");
    return bytes;
}

template <class T> std::vector<uint8_t> Bytes(const T *records, size_t count)
{
    if (count > std::numeric_limits<size_t>::max() / sizeof(T))
        throw JoltGpuQueryRequestError("Jolt GPU joint upload exceeds the host address range");
    std::vector<uint8_t> result(count * sizeof(T));
    if (!result.empty())
        std::memcpy(result.data(), records, result.size());
    return result;
}

std::shared_ptr<rhi::ComputeBuffer> Allocate(rhi::ComputeHost &host, uint64_t bytes)
{
    auto buffer =
        rhi::ComputeBuffer::TryCreatePrivate(host, {bytes / sizeof(uint32_t), rhi::ComputeScalarType::UInt32, 1});
    if (!buffer)
        throw JoltGpuQueryRequestError("Jolt GPU joint preparation exceeds available managed device capacity");
    return buffer;
}

uint32_t Groups(const rhi::ComputeHost &host, size_t count)
{
    if (count > std::numeric_limits<uint32_t>::max())
        throw JoltGpuQueryRequestError("Jolt GPU joint dispatch exceeds device indexing");
    const auto groups = static_cast<uint32_t>(count / 64 + (count % 64 != 0));
    if (groups > host.device.GetCapabilities().limits.maxComputeWorkgroupCount[0])
        throw JoltGpuQueryRequestError("Jolt GPU joint dispatch exceeds the device workgroup range");
    return groups;
}

void ValidateLayout(uint8_t port, const ShaderResourceLayout &resource)
{
    const auto &expected = Ports[port];
    const bool uniform = port == Parameters;
    const auto &value = resource.value;
    if (resource.binding.type != (uniform ? rhi::BindingType::UniformBuffer : rhi::BindingType::StorageBuffer) ||
        (uniform ? value.byteSize : resource.elementStride) != expected.stride ||
        (!uniform && resource.binding.readOnlyStorage != expected.readOnly))
        throw std::invalid_argument("Jolt GPU joint target differs from its shared resource layout");
    if (expected.scalar) {
        if (value.kind != ShaderValueKind::Scalar || value.scalar != ShaderScalarKind::UInt32 ||
            value.byteSize != sizeof(uint32_t))
            throw std::invalid_argument("Jolt GPU joint index stream must contain uint scalars");
        return;
    }
    if (value.kind != ShaderValueKind::Struct || value.fields.size() != expected.stride / sizeof(UInt4))
        throw std::invalid_argument("Jolt GPU joint target differs from its shared record layout");
    for (size_t index = 0; index < value.fields.size(); ++index) {
        const auto &field = value.fields[index];
        if (field.name != expected.fields[index] || field.offset != index * sizeof(UInt4) ||
            field.byteSize != sizeof(UInt4) || field.kind != ShaderValueKind::Vector ||
            field.scalar != (expected.integer ? ShaderScalarKind::UInt32 : ShaderScalarKind::Float32) ||
            field.rows != 1 || field.columns != 4)
            throw std::invalid_argument("Jolt GPU joint target field differs from its shared device record");
    }
}

void ValidateFrozen(const rhi::ComputeHost &host, const std::shared_ptr<rhi::ComputeBuffer> &buffer, uint64_t bytes,
                    const char *message)
{
    if (!buffer || !buffer->IsFrozen() || !host.SharesServicesWith(buffer->GetHost()) || buffer->GetByteSize() != bytes)
        throw JoltGpuQueryRequestError(message);
}

float DecodeFloat(uint32_t bits)
{
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
} // namespace

JoltGpuJoints::JoltGpuJoints(rhi::ComputeHost &host, const ShaderTargetCode &target) : m_host(host)
{
    for (uint8_t stage = 0; stage < StageCount; ++stage) {
        const auto code = std::find_if(target.entries.begin(), target.entries.end(),
                                       [&](const auto &entry) { return entry.name == Entries[stage]; });
        if (code == target.entries.end() || code->stage != rhi::ShaderStage::Compute ||
            code->workgroupSize != std::array<uint32_t, 3>{64, 1, 1})
            throw std::invalid_argument("Jolt GPU joint target is missing its shared compute entry");
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
                throw std::invalid_argument("Jolt GPU joint entry contains an unknown resource");
            const auto index = static_cast<uint8_t>(port - std::begin(Ports));
            if (actual & (1u << index))
                throw std::invalid_argument("Jolt GPU joint entry repeats a resource");
            ValidateLayout(index, resource);
            bindings.push_back({resource.binding.binding, resource.binding.type, resource.binding.readOnlyStorage});
            program.ports.push_back(index);
            actual |= 1u << index;
        }
        if (actual != StagePorts[stage] || code->bindings.size() != program.ports.size())
            throw std::invalid_argument("Jolt GPU joint entry differs from its shared resource set");
        if (target.target == ShaderCodeTarget::SpirV) {
            if (code->code.empty() || code->code.size() % sizeof(uint32_t))
                throw std::invalid_argument("Jolt GPU joint target contains invalid SPIR-V");
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

JoltGpuJointReplacement JoltGpuJoints::PrepareReplacement(const std::vector<JointAuthor> &authors,
                                                          const std::vector<uint32_t> &previousRows,
                                                          const std::shared_ptr<rhi::ComputeBuffer> &oldStream,
                                                          uint32_t oldCount,
                                                          const std::shared_ptr<rhi::ComputeBuffer> &bodyStates) const
{
    const auto streamBytes = StreamBytes(authors.size());
    if (previousRows.size() != authors.size())
        throw JoltGpuQueryRequestError("Jolt GPU replacement requires one previous-row mapping per joint");
    for (const auto row : previousRows)
        if (row != InvalidQueryBodySlot && row >= oldCount)
            throw JoltGpuQueryRequestError("Jolt GPU replacement references a previous row outside its stream");
    if (oldStream)
        ValidateFrozen(m_host, oldStream, StreamBytes(oldCount),
                       "Jolt GPU previous joint stream differs from its frozen topology");
    else if (oldCount)
        throw JoltGpuQueryRequestError("Jolt GPU replacement is missing its previous joint stream");
    const auto groups = Groups(m_host, authors.size());
    if (!authors.empty()) {
        if (!bodyStates || !bodyStates->IsFrozen() || !m_host.SharesServicesWith(bodyStates->GetHost()) ||
            !bodyStates->GetByteSize() || bodyStates->GetByteSize() % sizeof(BodyState))
            throw JoltGpuQueryRequestError("Jolt GPU joint initialization requires frozen ordered body states");
        const uint64_t bodyCount = bodyStates->GetByteSize() / sizeof(BodyState);
        for (const auto &author : authors)
            if (author.bodies.y >= bodyCount ||
                (author.bodies.x != InvalidQueryBodySlot && author.bodies.x >= bodyCount))
                throw JoltGpuQueryRequestError("Jolt GPU joint initialization endpoint exceeds its body states");
    }
    rhi::ComputeHost host = m_host;
    JoltGpuJointReplacement replacement;
    replacement.stream = Allocate(host, streamBytes);
    const uint64_t authorEnd = sizeof(UInt4) + uint64_t(authors.size()) * sizeof(JointAuthor);
    const UInt4 header{static_cast<uint32_t>(authors.size()), static_cast<uint32_t>(authorEnd / sizeof(UInt4)), 0, 0};
    std::vector<uint8_t> prefix(static_cast<size_t>(authorEnd));
    std::memcpy(prefix.data(), &header, sizeof(header));
    if (!authors.empty())
        std::memcpy(prefix.data() + sizeof(header), authors.data(), authors.size() * sizeof(JointAuthor));
    replacement.updates.push_back({replacement.stream, 0, std::move(prefix)});
    if (authors.empty())
        return replacement;

    std::array<std::shared_ptr<rhi::ComputeBuffer>, PortCount> buffers{};
    buffers[Stream] = replacement.stream;
    buffers[Previous] = oldStream;
    buffers[States] = bodyStates;
    const auto upload = [&](Port port, std::vector<uint8_t> data) {
        buffers[port] = Allocate(host, data.size());
        replacement.updates.push_back({buffers[port], 0, std::move(data)});
    };
    const JointParams parameters{{static_cast<uint32_t>(authors.size()), oldCount, 0, 0}};
    upload(Parameters, Bytes(&parameters, 1));
    upload(Remap, Bytes(previousRows.data(), previousRows.size()));
    if (!oldStream) {
        const UInt4 emptyHeader{0, 1, 0, 0};
        upload(Previous, Bytes(&emptyHeader, 1));
    }
    const auto &program = m_programs[Initialize];
    rhi::ComputeDispatchDesc dispatch;
    dispatch.kernel = program.kernel;
    dispatch.groupCountX = groups;
    rhi::ComputeDispatchProof proof;
    for (const auto port : program.ports) {
        const auto binding = static_cast<uint32_t>(dispatch.buffers.size());
        dispatch.buffers.push_back(buffers[port]);
        dispatch.bufferAccesses.push_back(port == Stream ? rhi::ComputeBufferAccess::ReadWrite
                                                         : rhi::ComputeBufferAccess::Read);
        proof.reads.push_back({binding, 0, port == Stream ? authorEnd : buffers[port]->GetByteSize()});
        if (port == Stream)
            proof.writes.push_back({binding, authorEnd, streamBytes - authorEnd});
    }
    dispatch.proof = std::move(proof);
    replacement.dispatches.push_back(std::move(dispatch));
    return replacement;
}

std::unique_ptr<JoltGpuModelUpdate> JoltGpuJoints::PrepareUpdate(JoltGpuStorage &storage,
                                                                 std::vector<JoltGpuJointModelUpdate> changes) const
{
    if (!m_host.SharesServicesWith(storage.GetHost()))
        throw std::invalid_argument("Jolt GPU joint updates belong to a different compute owner");
    const auto &accepted = storage.GetModel().joints;
    if (changes.size() > accepted.size())
        throw std::invalid_argument("Jolt GPU joint update count exceeds accepted topology");
    std::sort(changes.begin(), changes.end(), [](const auto &a, const auto &b) { return a.index < b.index; });
    (void)StreamBytes(accepted.size());
    std::vector<uint32_t> rows;
    rows.reserve(changes.size() * 2);
    for (size_t index = 0; index < changes.size(); ++index) {
        const auto &change = changes[index];
        if (change.index >= accepted.size() || (index && changes[index - 1].index == change.index))
            throw std::invalid_argument("Jolt GPU joint updates require unique rows within accepted topology");
        const auto &old = accepted[change.index];
        const auto &author = change.author;
        uint32_t flags = 0;
        if (old.motor.y != author.motor.y || old.motor.z != author.motor.z || old.motor.w != author.motor.w ||
            old.passive.x != author.passive.x || old.passive.y != author.passive.y ||
            ((old.bodies.w ^ author.bodies.w) & JointMotorEnabled))
            flags |= JointResetWarmStart;
        const auto limits = [](const JointAuthor &value) {
            return value.bodies.w & JointLimitsEnabled ? std::array<float, 2>{value.limits.x, value.limits.y}
                                                       : std::array<float, 2>{-glm::pi<float>(), glm::pi<float>()};
        };
        if (limits(old) != limits(author))
            flags |= JointResetWarmStart;
        if (old.passive.z != author.passive.z)
            flags |= JointResetRotor;
        if (flags) {
            rows.push_back(change.index);
            rows.push_back(flags);
        }
    }
    if (rows.empty())
        return storage.PrepareModelUpdate({}, {}, {}, {}, std::move(changes));

    const auto requestCount = rows.size() / 2;
    const auto groups = Groups(m_host, requestCount);
    const auto &stream = storage.Get(Resource::Joints);
    if (!stream || stream->IsFrozen() || !m_host.SharesServicesWith(stream->GetHost()) ||
        stream->GetByteSize() != StreamBytes(accepted.size()))
        throw std::invalid_argument("Jolt GPU joint updates require their accepted writable stream");
    const auto &states = storage.Get(Resource::BodyStates);
    if (!states || !m_host.SharesServicesWith(states->GetHost()) ||
        states->GetByteSize() != uint64_t(storage.GetBodyCount()) * sizeof(BodyState))
        throw std::invalid_argument("Jolt GPU joint updates require their ordered body states");

    rhi::ComputeHost host = m_host;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, PortCount> buffers{};
    buffers[Stream] = stream;
    buffers[States] = states->Freeze();
    std::vector<rhi::ComputeBufferUpdate> uploads;
    const auto upload = [&](Port port, std::vector<uint8_t> data) {
        buffers[port] = Allocate(host, data.size());
        uploads.push_back({buffers[port], 0, std::move(data)});
    };
    const JointParams parameters{{static_cast<uint32_t>(accepted.size()), 0, static_cast<uint32_t>(requestCount), 0}};
    upload(Parameters, Bytes(&parameters, 1));
    upload(Rows, Bytes(rows.data(), rows.size()));
    const auto &program = m_programs[UpdateState];
    rhi::ComputeDispatchDesc dispatch;
    dispatch.kernel = program.kernel;
    dispatch.groupCountX = groups;
    rhi::ComputeDispatchProof proof;
    const uint64_t runtimeBase = sizeof(UInt4) + uint64_t(accepted.size()) * sizeof(JointAuthor);
    for (const auto port : program.ports) {
        const auto binding = static_cast<uint32_t>(dispatch.buffers.size());
        dispatch.buffers.push_back(buffers[port]);
        dispatch.bufferAccesses.push_back(port == Stream ? rhi::ComputeBufferAccess::ReadWrite
                                                         : rhi::ComputeBufferAccess::Read);
        if (port != Stream) {
            proof.reads.push_back({binding, 0, buffers[port]->GetByteSize()});
            continue;
        }
        proof.reads.push_back({binding, 0, sizeof(UInt4)});
        for (size_t request = 0; request < rows.size(); request += 2) {
            const uint64_t row = rows[request];
            proof.reads.push_back({binding, sizeof(UInt4) + row * sizeof(JointAuthor), sizeof(JointAuthor)});
            const rhi::ComputeBindingRange runtime{binding, runtimeBase + row * sizeof(JointState), sizeof(JointState)};
            proof.reads.push_back(runtime);
            proof.writes.push_back(runtime);
        }
    }
    dispatch.proof = std::move(proof);
    std::vector<rhi::ComputeDispatchDesc> dispatches;
    dispatches.push_back(std::move(dispatch));
    return storage.PrepareModelUpdate({}, {}, std::move(uploads), std::move(dispatches), std::move(changes));
}

JoltGpuPreparedJointMotion
JoltGpuJoints::PrepareReadMotion(std::shared_ptr<const JoltGpuScene::Publication> publication,
                                 const std::vector<uint32_t> &rows) const
{
    if (!publication)
        throw JoltGpuQueryRequestError("Jolt GPU joint motion requires one retained publication");
    const auto groups = Groups(m_host, rows.size());
    auto result = std::make_shared<JoltGpuJointMotionResult>();
    result->publication = std::move(publication);
    result->count = static_cast<uint32_t>(rows.size());
    if (rows.empty())
        return {std::move(result), {}};
    const auto &source = *result->publication;
    ValidateFrozen(m_host, source.jointStream, StreamBytes(source.joints.size()),
                   "Jolt GPU joint motion stream differs from its retained catalog");
    ValidateFrozen(m_host, source.bodyStates, uint64_t(source.actors.size()) * sizeof(BodyState),
                   "Jolt GPU joint motion body states differ from their retained catalog");
    if (!source.worlds || !source.worlds->IsFrozen() || !m_host.SharesServicesWith(source.worlds->GetHost()) ||
        !source.worlds->GetByteSize() || source.worlds->GetByteSize() % sizeof(WorldRange))
        throw JoltGpuQueryRequestError("Jolt GPU joint motion requires its retained world ranges");
    const uint64_t worldCount = source.worlds->GetByteSize() / sizeof(WorldRange);
    for (const auto row : rows) {
        if (row >= source.joints.size())
            throw JoltGpuQueryRequestError("Jolt GPU joint motion row is absent from its retained catalog");
        const auto &joint = source.joints[row];
        if (joint.row != row || joint.kind != JointHinge || joint.author.bodies.z != JointHinge || !joint.id ||
            !(joint.author.bodies.w & JointEnabled) || joint.author.identity.x != static_cast<uint32_t>(joint.id) ||
            joint.author.identity.y != static_cast<uint32_t>(joint.id >> 32) || joint.author.identity.z >= worldCount ||
            joint.author.identity.w)
            throw JoltGpuQueryRequestError("Jolt GPU joint motion identity differs from its retained catalog");
        const auto endpoint = [&](uint32_t slot, uint32_t id) {
            return slot < source.actors.size() && source.actors[slot] && source.actors[slot]->slot == slot &&
                   source.actors[slot]->bodyId == id && slot < source.residentBodies.size() &&
                   source.residentBodies[slot];
        };
        if (!endpoint(joint.author.bodies.y, joint.bodyIdA) ||
            (joint.author.bodies.x == InvalidQueryBodySlot ? joint.bodyIdB != JoltGpuScene::InvalidBodyId
                                                           : !endpoint(joint.author.bodies.x, joint.bodyIdB)))
            throw JoltGpuQueryRequestError("Jolt GPU joint motion endpoints differ from their retained catalog");
    }

    rhi::ComputeHost host = m_host;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, PortCount> buffers{};
    buffers[Stream] = source.jointStream;
    buffers[States] = source.bodyStates;
    std::vector<rhi::ComputeBufferUpdate> updates;
    const auto upload = [&](Port port, std::vector<uint8_t> data) {
        buffers[port] = Allocate(host, data.size());
        updates.push_back({buffers[port], 0, std::move(data)});
    };
    const JointParams parameters{{static_cast<uint32_t>(source.joints.size()), 0, result->count, 0}};
    upload(Parameters, Bytes(&parameters, 1));
    upload(Rows, Bytes(rows.data(), rows.size()));
    const uint64_t outputBytes = uint64_t(result->count) * sizeof(QueryRecordWord);
    buffers[Motion] = Allocate(host, outputBytes);
    const auto &program = m_programs[ReadMotion];
    rhi::ComputeDispatchDesc dispatch;
    dispatch.kernel = program.kernel;
    dispatch.groupCountX = groups;
    rhi::ComputeDispatchProof proof;
    for (const auto port : program.ports) {
        const auto binding = static_cast<uint32_t>(dispatch.buffers.size());
        dispatch.buffers.push_back(buffers[port]);
        dispatch.bufferAccesses.push_back(port == Motion ? rhi::ComputeBufferAccess::Write
                                                         : rhi::ComputeBufferAccess::Read);
        const rhi::ComputeBindingRange range{binding, 0, buffers[port]->GetByteSize()};
        if (port == Motion)
            proof.writes.push_back(range);
        else
            proof.reads.push_back(range);
    }
    dispatch.proof = std::move(proof);
    result->packet =
        rhi::PrepareComputeBatch(host, std::move(updates), {std::move(dispatch)}, {{buffers[Motion], 0, outputBytes}});
    return {result, JoltGpuControlWork{result->packet}};
}

JoltGpuControlWork JoltGpuJoints::PrepareStatus(std::shared_ptr<const JoltGpuScene::Publication> publication) const
{
    if (!publication)
        throw std::runtime_error("Jolt GPU joint status requires its final publication identity");
    if (publication->joints.empty())
        return {};
    const auto streamBytes = StreamBytes(publication->joints.size());
    const auto &stream = publication->jointStream;
    if (!stream || !stream->IsFrozen() || !m_host.SharesServicesWith(stream->GetHost()) ||
        stream->GetByteSize() != streamBytes)
        throw std::runtime_error("Jolt GPU joint status requires its frozen final numerical stream");
    const auto count = static_cast<uint32_t>(publication->joints.size());
    rhi::ComputeHost host = m_host;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, PortCount> buffers{};
    buffers[Stream] = stream;
    buffers[Parameters] = Allocate(host, sizeof(JointParams));
    buffers[Motion] = Allocate(host, sizeof(QueryRecordWord));
    const JointParams parameters{{count, 0, 0, 0}};
    const auto &program = m_programs[ReadStatus];
    rhi::ComputeDispatchDesc dispatch;
    dispatch.kernel = program.kernel;
    dispatch.groupCountX = 1;
    rhi::ComputeDispatchProof proof;
    for (const auto port : program.ports) {
        const auto binding = static_cast<uint32_t>(dispatch.buffers.size());
        dispatch.buffers.push_back(buffers[port]);
        dispatch.bufferAccesses.push_back(port == Motion ? rhi::ComputeBufferAccess::Write
                                                         : rhi::ComputeBufferAccess::Read);
        const rhi::ComputeBindingRange range{binding, 0, buffers[port]->GetByteSize()};
        if (port == Motion)
            proof.writes.push_back(range);
        else
            proof.reads.push_back(range);
    }
    dispatch.proof = std::move(proof);
    auto packet = rhi::PrepareComputeBatch(host, {{buffers[Parameters], 0, Bytes(&parameters, 1)}},
                                           {std::move(dispatch)}, {{buffers[Motion], 0, sizeof(QueryRecordWord)}});
    struct StatusResult
    {
        UInt4 value{};
        bool materialized = false;
    };
    auto result = std::make_shared<StatusResult>();
    return {packet, [packet, publication = std::move(publication), result, count]() -> JoltGpuControlWork {
                if (!result->materialized) {
                    const auto readback = packet->Readback(0);
                    if (!readback || !readback->IsComplete())
                        throw std::logic_error("Jolt GPU joint status requires its completed source receipt");
                    readback->ReadInto(&result->value, sizeof(result->value));
                    result->materialized = true;
                }
                const auto &value = result->value;
                if (value.z || value.w || value.x > JointStatusNumericalFailure ||
                    (value.x == JointStatusSuccess ? value.y != InvalidQueryBodySlot : value.y >= count))
                    throw std::runtime_error("Jolt GPU joint publication returned a malformed numerical status");
                if (value.x != JointStatusSuccess)
                    throw std::runtime_error("Jolt GPU joint numerical failure blocks publication at joint " +
                                             std::to_string(publication->joints[value.y].id) + " (status " +
                                             std::to_string(value.x) + ")");
                return {};
            }};
}

void JoltGpuJointMotionResult::CopyMotion(float *output, size_t valueCount)
{
    if (valueCount != size_t(count) * 2 || (valueCount && !output))
        throw JoltGpuQueryRequestError("Jolt GPU joint motion output must contain two floats per requested row");
    if (!materialized) {
        if (count) {
            if (!packet)
                throw std::logic_error("Jolt GPU joint motion has no readback packet");
            const auto readback = packet->Readback(0);
            if (!readback || !readback->IsComplete())
                throw std::logic_error("Jolt GPU joint motion requires its completed source receipt");
            values.resize(count);
            readback->ReadInto(values.data(), uint64_t(count) * sizeof(QueryRecordWord));
        }
        materialized = true;
    }
    // A failed row must not leave a partly updated caller-owned ndarray. The
    // cached values also make a repeated consume independent of staging lifetime.
    for (const auto &record : values) {
        if (record.value.z != JointStatusSuccess || record.value.w || !std::isfinite(DecodeFloat(record.value.x)) ||
            !std::isfinite(DecodeFloat(record.value.y)))
            throw JoltGpuQueryRequestError("Jolt GPU joint motion contains an invalid or failed numerical row");
    }
    for (size_t row = 0; row < values.size(); ++row) {
        output[row * 2] = DecodeFloat(values[row].value.x);
        output[row * 2 + 1] = DecodeFloat(values[row].value.y);
    }
}

} // namespace infernux
