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
    PoseDrive,
    PortCount
};
enum Stage : uint8_t
{
    Initialize,
    ReadMotion,
    ReadStatus,
    UpdateState,
    DrivePose,
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
    {"joint_motion", sizeof(JointReadback), {"values", "orientation", "status"}, false, false, false},
    {"joint_pose_drives", sizeof(JointPoseDrive), {"meta", "relative"}, false, false, true},
};
constexpr std::array<std::string_view, StageCount> Entries{
    "initialize_joints", "read_joint_motion", "read_joint_status", "update_joint_state", "drive_joint_pose"};
constexpr uint32_t Bit(Port port)
{
    return 1u << port;
}
constexpr std::array<uint32_t, StageCount> StagePorts{
    Bit(Parameters) | Bit(Remap) | Bit(Previous) | Bit(Stream) | Bit(States),
    Bit(Parameters) | Bit(Stream) | Bit(States) | Bit(Rows) | Bit(Motion), Bit(Parameters) | Bit(Stream) | Bit(Motion),
    Bit(Parameters) | Bit(Stream) | Bit(States) | Bit(Rows), Bit(Parameters) | Bit(Stream) | Bit(PoseDrive)};
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
            field.scalar != ((expected.integer || (port == Motion && index == 2) || (port == PoseDrive && index == 0))
                                 ? ShaderScalarKind::UInt32
                                 : ShaderScalarKind::Float32) ||
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

// The final publication gate scans finite packets through the same source as
// the solve. Only the final typed 48-byte reduction reaches the host.
struct JointStatusExecution : std::enable_shared_from_this<JointStatusExecution>
{
    static uint32_t OutputCount(uint32_t input)
    {
        return input / 64 + static_cast<uint32_t>(input % 64 != 0);
    }

    static uint64_t ScratchBytes(uint32_t input)
    {
        uint64_t words = 0;
        do {
            input = OutputCount(input);
            words += input;
        } while (input > 1);
        return words * sizeof(JointReadback);
    }

    JointStatusExecution(rhi::ComputeHost source, std::shared_ptr<rhi::ComputeKernel> program,
                         std::vector<uint8_t> bindings, std::shared_ptr<const JoltGpuScene::Publication> value)
        : host(std::move(source)), kernel(std::move(program)), ports(std::move(bindings)),
          publication(std::move(value)), count(static_cast<uint32_t>(publication->GetJoints().size())),
          inputCount(count), motion(Allocate(host, ScratchBytes(count)))
    {
    }

    JoltGpuControlWork Next()
    {
        constexpr uint32_t ItemsPerDispatch = 2048, DispatchesPerPacket = 8;
        const uint32_t end = cursor + std::min(inputCount - cursor, ItemsPerDispatch * DispatchesPerPacket);
        std::vector<rhi::ComputeBufferUpdate> uploads;
        std::vector<rhi::ComputeDispatchDesc> dispatches;
        for (uint32_t first = cursor; first < end;) {
            const auto size = std::min(end - first, ItemsPerDispatch);
            const JointParams parameters{{inputFirst + first, size, outputFirst + first / 64, level}};
            auto uniform = Allocate(host, sizeof(parameters));
            uploads.push_back({uniform, 0, Bytes(&parameters, 1)});
            rhi::ComputeDispatchDesc dispatch;
            dispatch.kernel = kernel;
            dispatch.groupCountX = Groups(host, size);
            rhi::ComputeDispatchProof proof;
            for (const auto port : ports) {
                const auto binding = static_cast<uint32_t>(dispatch.buffers.size());
                const auto &buffer = port == Parameters ? uniform : port == Stream ? publication->jointStream : motion;
                dispatch.buffers.push_back(buffer);
                dispatch.bufferAccesses.push_back(port == Motion ? rhi::ComputeBufferAccess::ReadWrite
                                                                 : rhi::ComputeBufferAccess::Read);
                if (port == Motion) {
                    if (level)
                        proof.reads.push_back({binding, uint64_t(inputFirst + first) * sizeof(JointReadback),
                                               uint64_t(size) * sizeof(JointReadback)});
                    proof.writes.push_back({binding, uint64_t(outputFirst + first / 64) * sizeof(JointReadback),
                                            uint64_t(dispatch.groupCountX) * sizeof(JointReadback)});
                } else
                    proof.reads.push_back({binding, 0, buffer->GetByteSize()});
            }
            dispatch.proof = std::move(proof);
            dispatches.push_back(std::move(dispatch));
            first += size;
        }
        const bool final = end == inputCount && OutputCount(inputCount) == 1;
        std::vector<rhi::ComputeBufferRead> reads;
        if (final)
            reads.push_back({motion, uint64_t(outputFirst) * sizeof(JointReadback), sizeof(JointReadback)});
        auto packet = rhi::PrepareComputeBatch(host, std::move(uploads), std::move(dispatches), std::move(reads));
        if (end == inputCount && !final) {
            inputFirst = outputFirst;
            inputCount = OutputCount(inputCount);
            outputFirst += inputCount;
            ++level;
            cursor = 0;
        } else
            cursor = end;
        return {packet, [self = shared_from_this(), packet, final]() -> JoltGpuControlWork {
                    if (!final)
                        return self->Next();
                    const auto readback = packet->Readback(0);
                    if (!readback || !readback->IsComplete())
                        throw std::logic_error("Jolt GPU joint status requires its completed source receipt");
                    JointReadback record{};
                    readback->ReadInto(&record, sizeof(record));
                    if (record.values.x || record.values.y || record.values.z || record.values.w)
                        throw std::runtime_error("Jolt GPU joint status returned malformed payload");
                    const auto &value = record.status;
                    if (value.z || value.w || value.x > JointStatusNumericalFailure ||
                        (value.x == JointStatusSuccess ? value.y != InvalidQueryBodySlot : value.y >= self->count))
                        throw std::runtime_error("Jolt GPU joint publication returned a malformed numerical status");
                    if (value.x != JointStatusSuccess)
                        throw std::runtime_error("Jolt GPU joint numerical failure blocks publication at joint " +
                                                 std::to_string(self->publication->GetJoints()[value.y].id) +
                                                 " (status " + std::to_string(value.x) + ")");
                    return {};
                }};
    }

    rhi::ComputeHost host;
    std::shared_ptr<rhi::ComputeKernel> kernel;
    std::vector<uint8_t> ports;
    std::shared_ptr<const JoltGpuScene::Publication> publication;
    uint32_t count, inputCount, inputFirst = 0, outputFirst = 0, level = 0, cursor = 0;
    std::shared_ptr<rhi::ComputeBuffer> motion;
};
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
                                                          std::shared_ptr<const JoltGpuJointLayout> oldLayout,
                                                          const std::shared_ptr<rhi::ComputeBuffer> &bodyStates) const
{
    const auto layout = std::make_shared<JoltGpuJointLayout>(authors);
    const auto streamBytes = layout->ByteSize();
    const auto oldCount = oldLayout ? oldLayout->Rows().size() : 0u;
    if (oldStream && !oldLayout)
        throw JoltGpuQueryRequestError("Joint remap source requires its retained compact layout");
    if (previousRows.size() != authors.size())
        throw JoltGpuQueryRequestError("Jolt GPU replacement requires one previous-row mapping per joint");
    for (uint32_t target = 0; target < previousRows.size(); ++target) {
        const auto row = previousRows[target];
        if (row != InvalidQueryBodySlot && row >= oldCount)
            throw JoltGpuQueryRequestError("Jolt GPU replacement references a previous row outside its stream");
        if (row != InvalidQueryBodySlot) {
            const auto &previous = oldLayout->Row(row);
            const auto &next = authors[target];
            if (previous.bodies.z != next.bodies.z || previous.identity.x != next.identity.x ||
                previous.identity.y != next.identity.y)
                throw JoltGpuQueryRequestError("Joint state remap requires unchanged stable identity and kind");
        }
    }
    if (oldStream)
        ValidateFrozen(m_host, oldStream, oldLayout->ByteSize(),
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
    const uint64_t authorEnd = layout->AuthorEndBytes();
    auto prefix = layout->PrefixBytes(authors);
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
    const JointParams parameters{{static_cast<uint32_t>(authors.size()), static_cast<uint32_t>(oldCount), 0, 0}};
    upload(Parameters, Bytes(&parameters, 1));
    upload(Remap, Bytes(previousRows.data(), previousRows.size()));
    if (!oldStream) {
        const UInt4 emptyHeader{0, 1, 1, 0};
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
    std::vector<uint32_t> rows;
    rows.reserve(changes.size() * 2);
    for (size_t index = 0; index < changes.size(); ++index) {
        const auto &change = changes[index];
        if (change.index >= accepted.size() || (index && changes[index - 1].index == change.index))
            throw std::invalid_argument("Jolt GPU joint updates require unique rows within accepted topology");
        const auto &old = accepted[change.index];
        const auto &author = change.author;
        uint32_t flags = change.runtimeIntent;
        if (author.bodies.z == JointSixDOF) {
            const auto changed = [](const Float4 &a, const Float4 &b) {
                return std::memcmp(&a, &b, sizeof(Float4)) != 0;
            };
            const auto pointBlock = [](const JointAuthor &a) {
                const auto k = a.sixLimitStiffness();
                return (a.sixModes().x & 63u) == 0u && k.x == 0 && k.y == 0 && k.z == 0;
            };
            const auto lane = [](const Float4 &v, uint32_t i) { return i == 0 ? v.x : i == 1 ? v.y : v.z; };
            if (changed(old.angularFrame(), author.angularFrame()))
                throw std::invalid_argument("SixDOF frame update requires rebuild");
            if (old.sixModes().x != author.sixModes().x || pointBlock(old) != pointBlock(author))
                flags |= JointResetSixClassification;
            if (changed(old.sixAngularMin(), author.sixAngularMin()) ||
                changed(old.sixAngularMax(), author.sixAngularMax()))
                flags |= JointResetSwingLimits;
            for (uint32_t axis = 0; axis < 3; ++axis) {
                if (lane(old.sixLinearMin(), axis) != lane(author.sixLinearMin(), axis) ||
                    lane(old.sixLinearMax(), axis) != lane(author.sixLinearMax(), axis) ||
                    lane(old.sixLimitStiffness(), axis) != lane(author.sixLimitStiffness(), axis) ||
                    lane(old.sixLimitDamping(), axis) != lane(author.sixLimitDamping(), axis))
                    flags |= 1u << (JointResetSixLimitShift + axis);
                if (((old.sixModes().y ^ author.sixModes().y) & (3u << (2u * axis))) ||
                    lane(old.sixLinearMotorK(), axis) != lane(author.sixLinearMotorK(), axis) ||
                    lane(old.sixLinearMotorC(), axis) != lane(author.sixLinearMotorC(), axis) ||
                    lane(old.sixLinearMotorForce(), axis) != lane(author.sixLinearMotorForce(), axis) ||
                    lane(old.sixLinearFriction(), axis) != lane(author.sixLinearFriction(), axis))
                    flags |= 1u << (JointResetSixLinearMotorShift + axis);
                if (((old.sixModes().y ^ author.sixModes().y) & (3u << (2u * (axis + 3u)))) ||
                    lane(old.sixAngularMotorK(), axis) != lane(author.sixAngularMotorK(), axis) ||
                    lane(old.sixAngularMotorC(), axis) != lane(author.sixAngularMotorC(), axis) ||
                    lane(old.sixAngularMotorTorque(), axis) != lane(author.sixAngularMotorTorque(), axis) ||
                    lane(old.sixAngularFriction(), axis) != lane(author.sixAngularFriction(), axis))
                    flags |= 1u << (JointResetSixAngularMotorShift + axis);
            }
        } else if (author.bodies.z == JointSwingTwist) {
            const auto changed = [](const Float4 &a, const Float4 &b) {
                return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
            };
            if (changed(old.angularFrame(), author.angularFrame()))
                throw std::invalid_argument("SwingTwist frame changes require topology rebuild");
            if (old.angularModes().x != author.angularModes().x || changed(old.angularLimits(), author.angularLimits()))
                flags |= JointResetSwingLimits;
            if (old.angularModes().y != author.angularModes().y)
                flags |= JointResetSwingMotor | JointResetTwistMotor;
            if (old.angularModes().z != author.angularModes().z)
                flags |= JointResetTwistMotor;
            if (old.angularSpring().x != author.angularSpring().x ||
                old.angularSpring().y != author.angularSpring().y || old.angularTorque().x != author.angularTorque().x)
                flags |= JointResetSwingMotor;
            if (old.angularSpring().z != author.angularSpring().z ||
                old.angularSpring().w != author.angularSpring().w || old.angularTorque().y != author.angularTorque().y)
                flags |= JointResetTwistMotor;
            if (old.angularTorque().z != author.angularTorque().z) {
                if (author.angularModes().y == 0)
                    flags |= JointResetSwingMotor;
                if (author.angularModes().z == 0)
                    flags |= JointResetTwistMotor;
            }
        } else {
            if (old.motor().y != author.motor().y || old.motor().z != author.motor().z ||
                old.motor().w != author.motor().w || old.passive().x != author.passive().x ||
                old.passive().y != author.passive().y || ((old.bodies.w ^ author.bodies.w) & JointMotorEnabled))
                flags |= JointResetWarmStart;
            const auto limits = [](const JointAuthor &value) {
                return value.bodies.w & JointLimitsEnabled ? std::array<float, 2>{value.limits().x, value.limits().y}
                                                           : std::array<float, 2>{-glm::pi<float>(), glm::pi<float>()};
            };
            const bool sliderLimitsChanged =
                author.bodies.z == JointSlider &&
                (((old.bodies.w ^ author.bodies.w) & JointLimitsEnabled) ||
                 ((author.bodies.w & JointLimitsEnabled) &&
                  (old.limits().x != author.limits().x || old.limits().y != author.limits().y)));
            const bool distanceSettingsChanged =
                author.bodies.z == JointDistance &&
                (((old.bodies.w ^ author.bodies.w) & JointAutoDistance) ||
                 (!(author.bodies.w & JointAutoDistance) &&
                  (old.limits().x != author.limits().x || old.limits().y != author.limits().y)) ||
                 old.linearSpring().x != author.linearSpring().x || old.linearSpring().y != author.linearSpring().y);
            if (author.bodies.z == JointDistance ? distanceSettingsChanged
                : author.bodies.z == JointSlider ? sliderLimitsChanged
                                                 : limits(old) != limits(author))
                flags |= JointResetWarmStart;
            if (old.passive().z != author.passive().z)
                flags |= JointResetRotor;
        }
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
        stream->GetByteSize() != storage.GetJointLayout()->ByteSize())
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
            const auto &entry = storage.GetJointLayout()->Row(static_cast<uint32_t>(row));
            proof.reads.push_back({binding, (1u + row * JointDirectoryWords) * 16u, sizeof(JointDirectory)});
            proof.reads.push_back({binding, uint64_t(entry.layout.x) * 16u,
                                   uint64_t(JoltGpuJointLayout::AuthorWords(entry.bodies.z)) * 16u});
            const rhi::ComputeBindingRange runtime{binding, uint64_t(entry.layout.y) * 16u,
                                                   uint64_t(JoltGpuJointLayout::StateWords(entry.bodies.z)) * 16u};
            proof.reads.push_back(runtime);
            proof.writes.push_back(runtime);
        }
    }
    dispatch.proof = std::move(proof);
    std::vector<rhi::ComputeDispatchDesc> dispatches;
    dispatches.push_back(std::move(dispatch));
    return storage.PrepareModelUpdate({}, {}, std::move(uploads), std::move(dispatches), std::move(changes));
}
JoltGpuJointReplacement JoltGpuJoints::PrepareMobilityReset(JoltGpuStorage &storage, uint32_t bodySlot) const
{
    if (!m_host.SharesServicesWith(storage.GetHost()))
        throw std::invalid_argument("Mobility reset belongs to a different compute owner");
    const auto &accepted = storage.GetModel().joints;
    std::vector<uint32_t> rows;
    for (uint32_t row = 0; row < accepted.size(); ++row)
        if (accepted[row].bodies.x == bodySlot || accepted[row].bodies.y == bodySlot) {
            rows.push_back(row);
            rows.push_back(JointResetMobility);
        }
    if (rows.empty())
        return {};
    const auto requestCount = rows.size() / 2;
    const auto groups = Groups(m_host, requestCount);
    const auto &stream = storage.Get(Resource::Joints);
    if (!stream || stream->IsFrozen() || !m_host.SharesServicesWith(stream->GetHost()) ||
        stream->GetByteSize() != storage.GetJointLayout()->ByteSize())
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
            const auto &entry = storage.GetJointLayout()->Row(static_cast<uint32_t>(row));
            proof.reads.push_back({binding, (1u + row * JointDirectoryWords) * 16u, sizeof(JointDirectory)});
            proof.reads.push_back({binding, uint64_t(entry.layout.x) * 16u,
                                   uint64_t(JoltGpuJointLayout::AuthorWords(entry.bodies.z)) * 16u});
            const rhi::ComputeBindingRange runtime{binding, uint64_t(entry.layout.y) * 16u,
                                                   uint64_t(JoltGpuJointLayout::StateWords(entry.bodies.z)) * 16u};
            proof.reads.push_back(runtime);
            proof.writes.push_back(runtime);
        }
    }
    dispatch.proof = std::move(proof);
    std::vector<rhi::ComputeDispatchDesc> dispatches;
    dispatches.push_back(std::move(dispatch));
    return {nullptr, std::move(uploads), std::move(dispatches)};
}

JoltGpuPreparedJointMotion
JoltGpuJoints::PrepareReadMotion(std::shared_ptr<const JoltGpuScene::Publication> publication,
                                 const std::vector<uint32_t> &rows, uint32_t kind) const
{
    if (!publication)
        throw JoltGpuQueryRequestError("Jolt GPU joint motion requires one retained publication");
    const auto groups = Groups(m_host, rows.size());
    auto result = std::make_shared<JoltGpuJointMotionResult>();
    result->publication = std::move(publication);
    result->count = static_cast<uint32_t>(rows.size());
    if (kind < JointHinge || kind > JointSixDOF)
        throw JoltGpuQueryRequestError("Jolt GPU motion kind has no readable coordinate");
    result->kind = kind;
    if (rows.empty())
        return {std::move(result), {}};
    const auto &source = *result->publication;
    if (!source.jointLayout || source.jointLayout->Rows().size() != source.GetJoints().size())
        throw JoltGpuQueryRequestError("Joint query lost its retained compact layout");
    ValidateFrozen(m_host, source.jointStream, source.jointLayout->ByteSize(),
                   "Jolt GPU joint motion stream differs from its retained catalog");
    ValidateFrozen(m_host, source.bodyStates, uint64_t(source.bodyCount) * sizeof(BodyState),
                   "Jolt GPU joint motion body states differ from their retained catalog");
    if (!source.worlds || !source.worlds->IsFrozen() || !m_host.SharesServicesWith(source.worlds->GetHost()) ||
        !source.worlds->GetByteSize() || source.worlds->GetByteSize() % sizeof(WorldRange))
        throw JoltGpuQueryRequestError("Jolt GPU joint motion requires its retained world ranges");
    const uint64_t worldCount = source.worlds->GetByteSize() / sizeof(WorldRange);
    for (const auto row : rows) {
        if (row >= source.GetJoints().size())
            throw JoltGpuQueryRequestError("Jolt GPU joint motion row is absent from its retained catalog");
        const auto &joint = source.GetJoints()[row];
        if (joint.row != row || joint.kind != kind || joint.author.bodies.z != joint.kind || !joint.id ||
            !(joint.author.bodies.w & JointEnabled) || joint.author.identity.x != static_cast<uint32_t>(joint.id) ||
            joint.author.identity.y != static_cast<uint32_t>(joint.id >> 32) ||
            joint.author.identity.z != source.worldRow || joint.author.identity.z >= worldCount ||
            joint.author.identity.w)
            throw JoltGpuQueryRequestError("Jolt GPU joint motion identity differs from its retained catalog");
        const auto endpoint = [&](uint32_t slot, uint32_t id) {
            return slot >= source.bodyBase && slot - source.bodyBase < source.bodyCount &&
                   slot < source.directory->actors.size() && source.directory->actors[slot] &&
                   source.directory->actors[slot]->slot == slot && source.directory->actors[slot]->bodyId == id &&
                   slot < source.directory->residentBodies.size() && source.directory->residentBodies[slot];
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
    const JointParams parameters{{source.jointCount, kind, result->count, source.bodyBase}};
    upload(Parameters, Bytes(&parameters, 1));
    upload(Rows, Bytes(rows.data(), rows.size()));
    const uint64_t outputBytes = uint64_t(result->count) * sizeof(JointReadback);
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

JoltGpuControlWork JoltGpuJoints::PreparePoseDrive(JoltGpuStorage &storage, std::vector<JointPoseDrive> requests) const
{
    if (!m_host.SharesServicesWith(storage.GetHost()))
        throw std::invalid_argument("Ragdoll drive belongs to a different compute owner");
    const auto layout = storage.GetJointLayout();
    for (const auto &request : requests) {
        if (!layout || request.meta.x >= layout->Rows().size())
            throw std::logic_error("Ragdoll drive precedes its accepted joint topology");
        const auto &row = layout->Row(request.meta.x);
        if (row.identity.x != request.meta.y || row.identity.y != request.meta.z ||
            (row.bodies.z != JointHinge && row.bodies.z != JointSwingTwist && row.bodies.z != JointSixDOF) ||
            request.meta.w > JointPoseDriveSet)
            throw std::logic_error("Ragdoll drive differs from its captured joint lifetime");
        if (request.meta.w == JointPoseDriveSet &&
            !(std::isfinite(request.relative.x) && std::isfinite(request.relative.y) &&
              std::isfinite(request.relative.z) && std::isfinite(request.relative.w)))
            throw std::invalid_argument("Ragdoll drive requires finite relative targets");
    }
    return PreparePoseDriveChunk(storage, std::make_shared<const std::vector<JointPoseDrive>>(std::move(requests)), 0);
}

JoltGpuControlWork JoltGpuJoints::PreparePoseDriveChunk(JoltGpuStorage &storage,
                                                        std::shared_ptr<const std::vector<JointPoseDrive>> requests,
                                                        size_t first) const
{
    if (first == requests->size())
        return {};
    constexpr size_t RequestsPerPacket = 1024;
    const auto count = std::min(RequestsPerPacket, requests->size() - first);
    const auto &stream = storage.Get(Resource::Joints);
    const auto layout = storage.GetJointLayout();
    if (!stream || stream->IsFrozen() || !layout || stream->GetByteSize() != layout->ByteSize())
        throw std::logic_error("Ragdoll drive requires the accepted writable joint stream");
    rhi::ComputeHost host = m_host;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, PortCount> buffers{};
    buffers[Stream] = stream;
    std::vector<rhi::ComputeBufferUpdate> updates;
    const auto upload = [&](Port port, std::vector<uint8_t> bytes) {
        buffers[port] = Allocate(host, bytes.size());
        updates.push_back({buffers[port], 0, std::move(bytes)});
    };
    const JointParams parameters{{static_cast<uint32_t>(layout->Rows().size()), 0, static_cast<uint32_t>(count), 0}};
    upload(Parameters, Bytes(&parameters, 1));
    upload(PoseDrive, Bytes(requests->data() + first, count));
    const auto &program = m_programs[DrivePose];
    rhi::ComputeDispatchDesc dispatch;
    dispatch.kernel = program.kernel;
    dispatch.groupCountX = Groups(host, count);
    rhi::ComputeDispatchProof proof;
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
        for (size_t index = first; index < first + count; ++index) {
            const auto row = requests->at(index).meta.x;
            const auto &entry = layout->Row(row);
            proof.reads.push_back({binding, (1ull + uint64_t(row) * JointDirectoryWords) * 16, sizeof(JointDirectory)});
            const rhi::ComputeBindingRange state{binding, uint64_t(entry.layout.y) * 16,
                                                 uint64_t(JoltGpuJointLayout::StateWords(entry.bodies.z)) * 16};
            proof.reads.push_back(state);
            proof.writes.push_back(state);
        }
    }
    dispatch.proof = std::move(proof);
    auto packet = rhi::PrepareComputeBatch(host, std::move(updates), {std::move(dispatch)});
    return {std::move(packet),
            [this, &storage, requests, next = first + count] { return PreparePoseDriveChunk(storage, requests, next); },
            JoltGpuControlWork::ResumeAfter::Acceptance};
}

JoltGpuControlWork JoltGpuJoints::PrepareStatus(std::shared_ptr<const JoltGpuScene::Publication> publication) const
{
    if (!publication)
        throw std::runtime_error("Jolt GPU joint status requires its final publication identity");
    if (publication->GetJoints().empty())
        return {};
    if (!publication->jointLayout)
        throw JoltGpuQueryRequestError("Joint publication has no retained layout");
    const auto streamBytes = publication->jointLayout->ByteSize();
    const auto &stream = publication->jointStream;
    if (!stream || !stream->IsFrozen() || !m_host.SharesServicesWith(stream->GetHost()) ||
        stream->GetByteSize() != streamBytes)
        throw std::runtime_error("Jolt GPU joint status requires its frozen final numerical stream");
    const auto &program = m_programs[ReadStatus];
    return std::make_shared<JointStatusExecution>(m_host, program.kernel, program.ports, std::move(publication))
        ->Next();
}

void JoltGpuJointMotionResult::CopyMotion(float *output, size_t valueCount)
{
    const size_t width = kind == JointSixDOF ? 7 : kind == JointSwingTwist ? 4 : 2;
    if (valueCount != size_t(count) * width || (valueCount && !output))
        throw JoltGpuQueryRequestError("Joint motion output width differs from requested kind");
    if (!materialized) {
        if (count) {
            if (!packet)
                throw std::logic_error("Joint motion has no packet");
            auto read = packet->Readback(0);
            if (!read || !read->IsComplete())
                throw std::logic_error("Joint motion needs its completed source receipt");
            values.resize(count);
            read->ReadInto(values.data(), uint64_t(count) * sizeof(JointReadback));
        }
        materialized = true;
    }
    const auto finite = [](const Float4 &v) {
        return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z) && std::isfinite(v.w);
    };
    for (const auto &r : values) {
        const auto &v = r.values, &q = r.orientation;
        const auto &status = r.status;
        const bool angular = kind == JointSwingTwist || kind == JointSixDOF;
        const float norm = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
        if (status.x != JointStatusSuccess || status.y || status.z || status.w || !finite(v) || !finite(q) ||
            (angular ? std::abs(norm - 1.0f) > 1e-4f : (q.x || q.y || q.z || q.w)) ||
            (kind == JointSwingTwist ? (v.x || v.y || v.z || v.w)
             : kind == JointSixDOF   ? v.w != 0.0f
                                     : (v.z || v.w)))
            throw JoltGpuQueryRequestError("Joint motion has an invalid/failed numerical row");
    }
    for (size_t row = 0; row < values.size(); ++row) {
        const auto &v = values[row].values, &q = values[row].orientation;
        if (kind == JointSixDOF) {
            output[row * width] = v.x;
            output[row * width + 1] = v.y;
            output[row * width + 2] = v.z;
        }
        if (kind == JointSixDOF || kind == JointSwingTwist) {
            const size_t first = row * width + (kind == JointSixDOF ? 3 : 0);
            output[first] = q.x;
            output[first + 1] = q.y;
            output[first + 2] = q.z;
            output[first + 3] = q.w;
        } else {
            output[row * width] = v.x;
            output[row * width + 1] = v.y;
        }
    }
}

} // namespace infernux
