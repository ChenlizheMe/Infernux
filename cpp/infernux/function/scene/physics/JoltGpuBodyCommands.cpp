#include "JoltGpuBodyCommands.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string_view>

namespace infernux
{
namespace
{
using namespace JPH::GPU;
enum Port : uint8_t
{
    Parameters,
    Commands,
    Spans,
    Models,
    States,
    Forces,
    Targets,
    PortCount
};
struct PortLayout
{
    std::string_view name;
    uint64_t stride;
};
constexpr PortLayout Ports[] = {{"command_params", sizeof(BodyCommandParams)},
                                {"body_commands", sizeof(BodyCommand)},
                                {"command_spans", sizeof(BodyCommandSpan)},
                                {"body_models", sizeof(BodyModel)},
                                {"body_states", sizeof(BodyState)},
                                {"body_forces", sizeof(BodyForces)},
                                {"kinematic_targets", sizeof(KinematicTarget)}};

template <class T> std::vector<uint8_t> Bytes(const T *records, size_t count)
{
    std::vector<uint8_t> bytes(count * sizeof(T));
    if (!bytes.empty())
        std::memcpy(bytes.data(), records, bytes.size());
    return bytes;
}

bool Finite3(const Float4 &value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

bool FiniteVector(const Float4 &value)
{
    return Finite3(value) && value.w == 0;
}

bool NormalizedRotation(const Float4 &value)
{
    return Finite3(value) && std::isfinite(value.w) &&
           std::abs(value.x * value.x + value.y * value.y + value.z * value.z + value.w * value.w - 1.0f) <= 1e-4f;
}

void ValidateTargetLayout(const ShaderValueLayout &value)
{
    constexpr std::array<std::string_view, 4> fields{"positionSpeed", "rotation", "rootCOMDt", "state"};
    if (value.fields.size() != fields.size())
        throw std::invalid_argument("Jolt GPU command target fields differ from KinematicTarget");
    for (size_t index = 0; index < fields.size(); ++index) {
        const auto &field = value.fields[index];
        if (field.name != fields[index] || field.offset != index * 16 || field.byteSize != 16 ||
            field.kind != ShaderValueKind::Vector || field.rows != 1 || field.columns != 4 ||
            field.scalar != (index == 3 ? ShaderScalarKind::UInt32 : ShaderScalarKind::Float32))
            throw std::invalid_argument("Jolt GPU command target fields differ from KinematicTarget");
    }
}

void Validate(const BodyCommand &command, uint32_t bodyCount)
{
    const auto kind = command.meta.y;
    if (command.meta.x >= bodyCount || kind > CommandMotionBoundary || command.meta.w)
        throw std::invalid_argument("Jolt GPU command has an invalid body slot, kind or padding");
    const auto mode = command.meta.z;
    if (kind <= CommandSetPose) {
        if ((mode & ~CommandPreserveKinematicMove) != 0 || !FiniteVector(command.rootCOM))
            throw std::invalid_argument("Jolt GPU pose command has invalid control flags or root COM");
        if (kind != CommandSetRotation && !FiniteVector(command.value))
            throw std::invalid_argument("Jolt GPU pose position must be finite with zero padding");
        if (kind != CommandSetPosition && !NormalizedRotation(command.auxiliary))
            throw std::invalid_argument("Jolt GPU command rotation must be normalized and finite");
        return;
    }
    if (kind >= CommandMovePosition && kind <= CommandMovePose) {
        if ((mode & ~CommandTemporaryStatic) != 0 || !std::isfinite(command.value.w) || command.value.w <= 0 ||
            !Finite3(command.rootCOM) || !std::isfinite(command.rootCOM.w) || command.rootCOM.w < 0)
            throw std::invalid_argument("Jolt GPU move command has invalid flags, duration, root COM or speed");
        if (kind != CommandMoveRotation && !Finite3(command.value))
            throw std::invalid_argument("Jolt GPU move target position must be finite");
        if (kind != CommandMovePosition && !NormalizedRotation(command.auxiliary))
            throw std::invalid_argument("Jolt GPU move target rotation must be normalized and finite");
        return;
    }
    if (kind == CommandMotionBoundary) {
        constexpr uint32_t flags = MotionBoundaryActionMask | MotionBoundaryCarryAuthoredPosition |
                                   MotionBoundaryCarryAuthoredRotation;
        if ((mode & ~flags) != 0 || (mode & MotionBoundaryActionMask) > MotionBoundarySuspendReset)
            throw std::invalid_argument("Jolt GPU motion boundary has invalid action or carry flags");
        // Boundary pose channels come from the live device target. No command
        // payload lane is consumed, and none is a CPU copy of that device pose.
        return;
    }
    const bool force = kind >= CommandAddForce && kind <= CommandAddForceAtPosition;
    if (force ? mode > CommandVelocityChange : mode != 0)
        throw std::invalid_argument("Jolt GPU command has an invalid ForceMode");
    if (kind != CommandResetMotion && !FiniteVector(command.value))
        throw std::invalid_argument("Jolt GPU command value must be finite with zero padding");
    if (kind == CommandAddForceAtPosition && !FiniteVector(command.auxiliary))
        throw std::invalid_argument("Jolt GPU command point must be finite with zero padding");
}
} // namespace

JoltGpuBodyCommands::JoltGpuBodyCommands(rhi::ComputeHost &host, const ShaderTargetCode &target) : m_host(host)
{
    const auto code = std::find_if(target.entries.begin(), target.entries.end(),
                                   [](const auto &entry) { return entry.name == "apply_body_commands"; });
    if (code == target.entries.end() || code->stage != rhi::ShaderStage::Compute ||
        code->workgroupSize != std::array<uint32_t, 3>{64, 1, 1})
        throw std::invalid_argument("Jolt GPU command target is missing its compute entry");
    uint32_t actual = 0;
    std::vector<rhi::ComputeBufferBinding> bindings;
    for (const auto &resource : target.resources) {
        const std::array<uint32_t, 2> key{resource.set, resource.binding.binding};
        if (std::find(code->bindings.begin(), code->bindings.end(), key) == code->bindings.end())
            continue;
        const auto port = std::find_if(std::begin(Ports), std::end(Ports),
                                       [&](const auto &value) { return value.name == resource.name; });
        if (port == std::end(Ports) || resource.set != 0)
            throw std::invalid_argument("Jolt GPU command target has an unknown resource");
        const auto index = static_cast<uint8_t>(port - std::begin(Ports));
        const bool uniform = index == Parameters;
        const bool writer = index == States || index == Forces || index == Targets;
        if (resource.binding.type != (uniform ? rhi::BindingType::UniformBuffer : rhi::BindingType::StorageBuffer) ||
            resource.value.kind != ShaderValueKind::Struct ||
            (uniform ? resource.value.byteSize : resource.elementStride) != port->stride ||
            (!uniform && resource.binding.readOnlyStorage == writer))
            throw std::invalid_argument("Jolt GPU command target differs from its shared records");
        if (index == Targets)
            ValidateTargetLayout(resource.value);
        bindings.push_back({resource.binding.binding, resource.binding.type, resource.binding.readOnlyStorage});
        m_ports.push_back(index);
        actual |= 1u << index;
    }
    if (actual != (1u << PortCount) - 1)
        throw std::invalid_argument("Jolt GPU command entry has an unexpected resource set");
    if (target.target == ShaderCodeTarget::SpirV) {
        if (code->code.empty() || code->code.size() % sizeof(uint32_t))
            throw std::invalid_argument("Jolt GPU command target contains invalid SPIR-V");
        std::vector<uint32_t> words(code->code.size() / sizeof(uint32_t));
        std::memcpy(words.data(), code->code.data(), code->code.size());
        m_kernel = std::make_shared<rhi::ComputeKernel>(
            m_host, rhi::ShaderModuleDesc::FromSpirV(words.data(), words.size()), code->name, std::move(bindings), 0);
    } else {
        m_kernel = std::make_shared<rhi::ComputeKernel>(
            m_host, rhi::ShaderModuleDesc::FromWgsl(code->code.data(), code->code.size()), code->name,
            std::move(bindings), 0);
    }
}

std::unique_ptr<JoltGpuCommandCursor> JoltGpuBodyCommands::Prepare(const JoltGpuStorage &storage,
                                                                   std::vector<BodyCommand> commands) const
{
    if (!m_host.SharesServicesWith(storage.GetHost()))
        throw std::invalid_argument("Jolt GPU commands and storage belong to different compute owners");
    if (commands.size() > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("Jolt GPU command count exceeds the device index range");
    for (const auto &command : commands)
        Validate(command, storage.GetBodyCount());
    auto cursor = std::unique_ptr<JoltGpuCommandCursor>(new JoltGpuCommandCursor(m_host, m_kernel, m_ports, storage));
    if (commands.empty())
        return cursor;
    // Sort indices, retaining FIFO with the original ordinal and copying each
    // payload only once when building the immutable dispatch plan.
    std::vector<uint32_t> order(commands.size());
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return commands[a].meta.x == commands[b].meta.x ? a < b : commands[a].meta.x < commands[b].meta.x;
    });
    struct Group
    {
        uint32_t first, end;
    };
    std::vector<Group> active;
    for (uint32_t first = 0; first < commands.size();) {
        uint32_t end = first + 1;
        while (end < commands.size() && commands[order[end]].meta.x == commands[order[first]].meta.x)
            ++end;
        active.push_back({first, end});
        first = end;
    }
    cursor->m_commands.reserve(commands.size());
    // Each round contains only unfinished bodies. Work is proportional to the
    // number of commands, including for highly unequal per-body FIFO lengths.
    while (!active.empty()) {
        std::vector<Group> next;
        next.reserve(active.size());
        uint32_t previousBody = 0;
        bool firstInRound = true;
        for (auto group : active) {
            const uint32_t body = commands[order[group.first]].meta.x;
            const uint32_t count = std::min(MaxCommandsPerBodyDispatch, group.end - group.first);
            if (firstInRound || body != previousBody + 1 ||
                cursor->m_dispatches.back().spanCount == MaxCommandBodiesPerDispatch) {
                cursor->m_dispatches.push_back({static_cast<uint32_t>(cursor->m_commands.size()), 0,
                                                static_cast<uint32_t>(cursor->m_spans.size()), 0, body});
            }
            auto &dispatch = cursor->m_dispatches.back();
            cursor->m_spans.push_back({{body, static_cast<uint32_t>(cursor->m_commands.size()), count, 0}});
            for (uint32_t offset = 0; offset < count; ++offset)
                cursor->m_commands.push_back(commands[order[group.first + offset]]);
            dispatch.commandCount += count;
            ++dispatch.spanCount;
            group.first += count;
            if (group.first < group.end)
                next.push_back(group);
            previousBody = body;
            firstInRound = false;
        }
        active = std::move(next);
    }
    return cursor;
}

JoltGpuCommandCursor::JoltGpuCommandCursor(const rhi::ComputeHost &host, std::shared_ptr<rhi::ComputeKernel> kernel,
                                           std::vector<uint8_t> ports, const JoltGpuStorage &storage)
    : m_host(host), m_kernel(std::move(kernel)), m_ports(std::move(ports)),
      m_models(storage.GetBodyCount() ? storage.Get(Resource::BodyModels)->Freeze() : nullptr),
      m_states(storage.Get(Resource::BodyStates)), m_forces(storage.Get(Resource::BodyForces)),
      m_targets(storage.Get(Resource::KinematicTargets))
{
}

JoltGpuCommandCursor::~JoltGpuCommandCursor()
{
    if (m_batch)
        (void)m_batch->Cancel();
}

bool JoltGpuCommandCursor::IsComplete() const noexcept
{
    return m_next == m_dispatches.size();
}

void JoltGpuCommandCursor::DiscardPreparedPrefix() noexcept
{
    if (m_batch && !m_batch->GetSubmission().IsValid()) {
        (void)m_batch->Cancel();
        m_batch.reset();
    }
}

std::shared_ptr<rhi::PreparedComputeBatch> JoltGpuCommandCursor::PrepareNext(rhi::ComputeHost &host,
                                                                             uint32_t maxDispatches)
{
    if (!m_host.SharesServicesWith(host))
        throw std::invalid_argument("Jolt GPU command continuation requires the same compute services");
    if (m_batch)
        return m_batch;
    if (maxDispatches == 0)
        throw std::invalid_argument("Jolt GPU command prefix requires a positive dispatch budget");
    if (IsComplete())
        return {};
    const size_t end = m_next + std::min<size_t>(maxDispatches, m_dispatches.size() - m_next);
    const auto &first = m_dispatches[m_next];
    const auto &last = m_dispatches[end - 1];
    const uint32_t commandCount = last.firstCommand + last.commandCount - first.firstCommand;
    const uint32_t spanCount = last.firstSpan + last.spanCount - first.firstSpan;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, PortCount> buffers;
    std::vector<rhi::ComputeBufferUpdate> updates;
    const auto input = [&](Port port, std::vector<uint8_t> bytes) {
        buffers[port] = std::make_shared<rhi::ComputeBuffer>(
            host, rhi::ComputeBufferDesc{bytes.size() / sizeof(uint32_t), rhi::ComputeScalarType::UInt32, 1});
        updates.push_back({buffers[port], 0, std::move(bytes)});
    };
    input(Commands, Bytes(m_commands.data() + first.firstCommand, commandCount));
    std::vector<BodyCommandSpan> spans(m_spans.begin() + first.firstSpan,
                                       m_spans.begin() + first.firstSpan + spanCount);
    for (auto &span : spans)
        span.range.y -= first.firstCommand;
    input(Spans, Bytes(spans.data(), spans.size()));
    buffers[Models] = m_models;
    buffers[States] = m_states;
    buffers[Forces] = m_forces;
    buffers[Targets] = m_targets;
    std::vector<rhi::ComputeDispatchDesc> dispatches;
    dispatches.reserve(end - m_next);
    for (size_t index = m_next; index < end; ++index) {
        const auto &slice = m_dispatches[index];
        const BodyCommandParams params{{slice.firstSpan - first.firstSpan, slice.spanCount, 0, 0}};
        input(Parameters, Bytes(&params, 1));
        rhi::ComputeDispatchDesc dispatch;
        dispatch.kernel = m_kernel;
        dispatch.groupCountX = 1;
        rhi::ComputeDispatchProof proof;
        for (const auto port : m_ports) {
            const uint32_t binding = static_cast<uint32_t>(dispatch.buffers.size());
            dispatch.buffers.push_back(buffers[port]);
            const bool writer = port == States || port == Forces || port == Targets;
            dispatch.bufferAccesses.push_back(writer ? rhi::ComputeBufferAccess::ReadWrite
                                                     : rhi::ComputeBufferAccess::Read);
            uint64_t offset = 0, byteSize = buffers[port]->GetByteSize();
            if (port == Commands) {
                offset = uint64_t(slice.firstCommand - first.firstCommand) * sizeof(BodyCommand);
                byteSize = uint64_t(slice.commandCount) * sizeof(BodyCommand);
            } else if (port == Spans) {
                offset = uint64_t(slice.firstSpan - first.firstSpan) * sizeof(BodyCommandSpan);
                byteSize = uint64_t(slice.spanCount) * sizeof(BodyCommandSpan);
            } else if (port >= Models) {
                offset = uint64_t(slice.firstBody) * Ports[port].stride;
                byteSize = uint64_t(slice.spanCount) * Ports[port].stride;
            }
            proof.reads.push_back({binding, offset, byteSize});
            if (writer)
                proof.writes.push_back({binding, offset, byteSize});
        }
        dispatch.proof = std::move(proof);
        dispatches.push_back(std::move(dispatch));
    }
    m_batch = rhi::PrepareComputeBatch(host, std::move(updates), std::move(dispatches));
    m_pendingEnd = end;
    return m_batch;
}

void JoltGpuCommandCursor::Accepted() noexcept
{
    m_next = m_pendingEnd;
    m_batch.reset();
}

std::optional<rhi::SubmissionTicket> JoltGpuCommandCursor::TrySubmit()
{
    if (!m_batch)
        throw std::logic_error("Jolt GPU commands require a prepared prefix before submission");
    std::optional<rhi::SubmissionTicket> ticket;
    try {
        ticket = m_batch->TrySubmit();
    } catch (...) {
        if (m_batch->GetSubmission().IsValid())
            Accepted();
        throw;
    }
    if (ticket)
        Accepted();
    return ticket;
}

} // namespace infernux
