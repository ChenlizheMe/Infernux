#include "PhysicsGpuRenderPoses.h"
#include "PhysicsGpuRenderBindings.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace infernux
{
namespace
{
using namespace physics::gpu_render;

enum Port : uint8_t
{
    Parameters,
    Nodes,
    States,
    NodeModels,
    NodeModelsRead,
    Bindings,
    Models,
    Bounds,
    MotionParameters,
    MotionRows,
    MotionValues,
    BoxValues,
    PortCount
};

struct PortLayout
{
    std::string_view name;
    uint64_t stride;
    std::array<std::string_view, 12> fields;
    uint32_t integerFields;
    bool vector;
};

constexpr PortLayout Ports[] = {
    {"pose_params", sizeof(PoseParams), {"counts", "stateLayout"}, 3, false},
    {"pose_nodes",
     sizeof(PoseNode),
     {"bodyParent", "rootCOM", "localScale", "parentRotation", "parentLocalToAncestor0", "parentLocalToAncestor1",
      "parentLocalToAncestor2", "parentLocalToAncestor3", "parentAncestorToLocal0", "parentAncestorToLocal1",
      "parentAncestorToLocal2", "parentAncestorToLocal3"},
     1,
     false},
    {"body_states", sizeof(Float4), {}, 0, true},
    {"pose_node_models", sizeof(Float4), {}, 0, true},
    {"pose_node_models_read", sizeof(Float4), {}, 0, true},
    {"pose_bindings",
     sizeof(PoseBinding),
     {"nodeIndex", "objectLocal0", "objectLocal1", "objectLocal2", "objectLocal3", "drawPivot", "boundsCenter",
      "boundsExtent"},
     1,
     false},
    {"pose_models", sizeof(Float4), {}, 0, true},
    {"pose_bounds", sizeof(Float4), {}, 0, true},
    {"motion_params", 48, {"source", "target", "output"}, 7, false},
    {"motion_rows", sizeof(uint32_t), {}, 0, false},
    {"motion_values", sizeof(float), {}, 0, false},
    {"box_values", sizeof(uint32_t), {}, 0, false},
};

bool Writer(uint8_t port)
{
    return port == NodeModels || port == Models || port == Bounds || port == MotionValues || port == BoxValues;
}

void ValidateVector(const ShaderValueLayout &value, ShaderScalarKind scalar)
{
    if (value.byteSize != 16 || value.kind != ShaderValueKind::Vector || value.scalar != scalar || value.rows != 1 ||
        value.columns != 4)
        throw std::invalid_argument("GPU render-pose target differs from its shared vector record");
}

void ValidateLayout(uint8_t port, const ShaderResourceLayout &resource)
{
    const auto &expected = Ports[port];
    const bool uniform = port == Parameters || port == MotionParameters;
    if (resource.binding.type != (uniform ? rhi::BindingType::UniformBuffer : rhi::BindingType::StorageBuffer) ||
        (uniform ? resource.value.byteSize : resource.elementStride) != expected.stride ||
        (!uniform && resource.binding.readOnlyStorage == Writer(port)))
        throw std::invalid_argument("GPU render-pose target differs from its shared port layout");
    if (port == MotionRows || port == MotionValues || port == BoxValues) {
        const auto scalar = port == MotionValues ? ShaderScalarKind::Float32 : ShaderScalarKind::UInt32;
        if (resource.value.kind != ShaderValueKind::Scalar || resource.value.scalar != scalar)
            throw std::invalid_argument("GPU body motion export differs from its scalar buffer layout");
        return;
    }
    if (expected.vector) {
        ValidateVector(resource.value, ShaderScalarKind::Float32);
        return;
    }
    if (resource.value.kind != ShaderValueKind::Struct ||
        resource.value.fields.size() != expected.stride / sizeof(Float4))
        throw std::invalid_argument("GPU render-pose target differs from its shared structured record");
    for (size_t field = 0; field < resource.value.fields.size(); ++field) {
        const auto &value = resource.value.fields[field];
        if (value.name != expected.fields[field] || value.offset != field * sizeof(Float4))
            throw std::invalid_argument("GPU render-pose target has a different field name or offset");
        ValidateVector(value,
                       expected.integerFields & (1u << field) ? ShaderScalarKind::UInt32 : ShaderScalarKind::Float32);
    }
}

} // namespace

PhysicsGpuRenderPoses::PhysicsGpuRenderPoses(rhi::ComputeHost &host, const ShaderTargetCode &target) : m_host(host)
{
    const auto build = [&](const char *name, uint32_t required) {
        const auto code = std::find_if(target.entries.begin(), target.entries.end(),
                                       [&](const auto &entry) { return entry.name == name; });
        if (code == target.entries.end() || code->stage != rhi::ShaderStage::Compute ||
            code->workgroupSize != std::array<uint32_t, 3>{64, 1, 1} ||
            std::count_if(target.entries.begin(), target.entries.end(),
                          [&](const auto &entry) { return entry.name == name; }) != 1)
            throw std::invalid_argument("GPU render-pose target is missing its unique compute entry");
        Entry entry;
        uint32_t actual = 0;
        std::vector<rhi::ComputeBufferBinding> bindings;
        for (const auto &resource : target.resources) {
            const std::array<uint32_t, 2> key{resource.set, resource.binding.binding};
            if (std::find(code->bindings.begin(), code->bindings.end(), key) == code->bindings.end())
                continue;
            const auto port = std::find_if(std::begin(Ports), std::end(Ports),
                                           [&](const auto &value) { return value.name == resource.name; });
            if (port == std::end(Ports) || resource.set != 0)
                throw std::invalid_argument("GPU render-pose entry contains an unknown resource");
            const auto index = static_cast<uint8_t>(port - std::begin(Ports));
            if (actual & (1u << index))
                throw std::invalid_argument("GPU render-pose entry repeats a resource");
            ValidateLayout(index, resource);
            bindings.push_back({resource.binding.binding, resource.binding.type, resource.binding.readOnlyStorage});
            entry.ports.push_back(index);
            actual |= 1u << index;
        }
        if (actual != required || code->bindings.size() != entry.ports.size())
            throw std::invalid_argument("GPU render-pose entry has an unexpected resource set");
        const rhi::ShaderModuleDesc module{target.target == ShaderCodeTarget::SpirV ? rhi::ShaderSourceLanguage::SpirV
                                                                                    : rhi::ShaderSourceLanguage::Wgsl,
                                           code->code.data(), code->code.size()};
        entry.kernel = std::make_shared<rhi::ComputeKernel>(m_host, module, code->name, std::move(bindings), 0,
                                                            rhi::ComputeKernel::PipelineCreation::FirstDispatch);
        return entry;
    };
    m_nodes = build("export_body_pose_nodes", (1u << Parameters) | (1u << Nodes) | (1u << States) | (1u << NodeModels));
    m_meshes = build("export_render_poses",
                     (1u << Parameters) | (1u << Bindings) | (1u << NodeModelsRead) | (1u << Models) | (1u << Bounds));
    m_motion = build("export_body_motion",
                     (1u << MotionParameters) | (1u << States) | (1u << MotionRows) | (1u << MotionValues));
    m_box =
        build("export_body_boxes", (1u << MotionParameters) | (1u << MotionRows) | (1u << States) | (1u << BoxValues));
}

PhysicsGpuRenderPoseBatch PhysicsGpuRenderPoses::Prepare(const PhysicsGpuPoseSource &output,
                                                         const std::vector<PoseNode> &nodes,
                                                         const std::vector<PoseBinding> &bindings) const
{
    const auto &frozenBodyStates = output.states;
    if (bindings.empty())
        return {};
    if (nodes.empty() || nodes.size() > std::numeric_limits<uint32_t>::max() / 8 ||
        bindings.size() > std::numeric_limits<uint32_t>::max() / 4)
        throw std::invalid_argument("GPU render-pose inputs exceed their node or mesh device index range");
    if (!frozenBodyStates || !frozenBodyStates->IsFrozen() || !m_host.SharesServicesWith(frozenBodyStates->GetHost()) ||
        frozenBodyStates->GetByteSize() % sizeof(Float4))
        throw std::invalid_argument("GPU render poses require a frozen final state from the same compute owner");
    const auto bodyCount = output.bodyCount;
    const bool origin = output.layout == PhysicsGpuPoseSource::Layout::OriginWXYZ;
    const auto bodyStride = output.rowBytes;
    if (bodyStride < 2 * sizeof(Float4) || bodyStride % sizeof(Float4) != 0)
        throw std::invalid_argument("GPU render pose rows must contain aligned position and quaternion records");
    const auto rows = frozenBodyStates->GetByteSize() / bodyStride;
    if (output.firstRow > rows || bodyCount > rows - output.firstRow ||
        output.firstRow + bodyCount > UINT32_MAX / (bodyStride / sizeof(Float4)))
        throw std::invalid_argument("GPU render poses address outside the completed body state extent");
    const uint64_t stateOffset = output.firstRow * bodyStride;
    const uint64_t stateBytes = uint64_t{bodyCount} * bodyStride;
    const UInt4 stateLayout{static_cast<uint32_t>(stateOffset / sizeof(Float4)),
                            static_cast<uint32_t>(bodyStride / sizeof(Float4)), origin, 0};
    const auto &caps = m_host.device.GetCapabilities();
    const auto checkedBytes = [&](uint64_t count, uint64_t stride) {
        if (count > std::numeric_limits<size_t>::max() / stride ||
            count > std::numeric_limits<uint64_t>::max() / stride)
            throw std::invalid_argument("GPU render-pose storage exceeds the host/device byte range");
        const auto bytes = count * stride;
        if (caps.portable.maxStorageBufferBinding && bytes > caps.portable.maxStorageBufferBinding)
            throw std::invalid_argument("GPU render-pose storage exceeds the device binding range");
        return static_cast<size_t>(bytes);
    };
    const auto nodeInputBytes = checkedBytes(nodes.size(), sizeof(PoseNode));
    const auto bindingInputBytes = checkedBytes(bindings.size(), sizeof(PoseBinding));
    (void)checkedBytes(nodes.size(), 8 * sizeof(Float4));
    (void)checkedBytes(bindings.size(), 4 * sizeof(Float4));
    (void)checkedBytes(bindings.size(), 2 * sizeof(Float4));
    (void)checkedBytes(frozenBodyStates->GetByteSize() / sizeof(Float4), sizeof(Float4));

    // Every node has at most one physical parent. Stable breadth-first Kahn
    // ordering visits each edge once and makes each level a contiguous range.
    std::vector<uint32_t> children(nodes.size(), PoseNoParent), next(nodes.size(), PoseNoParent), order;
    order.reserve(nodes.size());
    for (const auto &node : nodes)
        if (node.bodyParent.x >= bodyCount || (node.bodyParent.y != PoseNoParent && node.bodyParent.y >= nodes.size()))
            throw std::invalid_argument("GPU render-pose node addresses an absent body or parent");
    for (uint32_t index = 0; index < nodes.size(); ++index)
        if (nodes[index].bodyParent.y == PoseNoParent)
            order.push_back(index);
    for (size_t index = nodes.size(); index-- > 0;) {
        const auto parent = nodes[index].bodyParent.y;
        if (parent != PoseNoParent) {
            next[index] = children[parent];
            children[parent] = static_cast<uint32_t>(index);
        }
    }
    std::vector<std::array<uint32_t, 2>> levels;
    uint32_t maxLevelCount = 0;
    for (size_t first = 0; first < order.size();) {
        const auto end = order.size();
        const auto count = static_cast<uint32_t>(end - first);
        levels.push_back({static_cast<uint32_t>(first), count});
        maxLevelCount = std::max(maxLevelCount, count);
        for (size_t index = first; index < end; ++index)
            for (auto child = children[order[index]]; child != PoseNoParent; child = next[child])
                order.push_back(child);
        first = end;
    }
    if (order.size() != nodes.size())
        throw std::invalid_argument("GPU render-pose parent graph contains a cycle");
    const auto groups = [](uint32_t count) { return count / 64 + static_cast<uint32_t>(count % 64 != 0); };
    if (groups(maxLevelCount) > caps.limits.maxComputeWorkgroupCount[0] ||
        groups(static_cast<uint32_t>(bindings.size())) > caps.limits.maxComputeWorkgroupCount[0])
        throw std::invalid_argument("GPU render-pose dispatch exceeds the device workgroup range");
    for (const auto &binding : bindings)
        if (binding.nodeIndex.x >= nodes.size())
            throw std::invalid_argument("GPU render-pose binding addresses an absent node");

    std::vector<uint32_t> remap(nodes.size());
    for (uint32_t row = 0; row < order.size(); ++row)
        remap[order[row]] = row;
    std::vector<PoseNode> orderedNodes;
    orderedNodes.reserve(nodes.size());
    for (const auto old : order) {
        auto node = nodes[old];
        if (node.bodyParent.y != PoseNoParent)
            node.bodyParent.y = remap[node.bodyParent.y];
        orderedNodes.push_back(node);
    }
    auto orderedBindings = bindings;
    for (auto &binding : orderedBindings)
        binding.nodeIndex.x = remap[binding.nodeIndex.x];

    rhi::ComputeHost host = m_host;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, PortCount> buffers;
    std::vector<rhi::ComputeBufferUpdate> uploads;
    const auto input = [&](Port port, InputCache &cache, const void *records, size_t byteSize) {
        if (cache.buffer && cache.bytes.size() == byteSize && std::memcmp(cache.bytes.data(), records, byteSize) == 0 &&
            cache.buffer->HasInitializedRange(0, byteSize)) {
            buffers[port] = cache.buffer;
            return;
        }
        std::vector<uint8_t> bytes(byteSize);
        std::memcpy(bytes.data(), records, byteSize);
        buffers[port] = std::make_shared<rhi::ComputeBuffer>(
            host, rhi::ComputeBufferDesc{bytes.size() / sizeof(uint32_t), rhi::ComputeScalarType::UInt32, 1});
        cache.bytes = bytes;
        cache.buffer = buffers[port];
        uploads.push_back({buffers[port], 0, std::move(bytes)});
    };
    const auto nodeCount = static_cast<uint32_t>(nodes.size());
    const auto count = static_cast<uint32_t>(bindings.size());
    input(Nodes, m_nodeInputs, orderedNodes.data(), nodeInputBytes);
    input(Bindings, m_bindingInputs, orderedBindings.data(), bindingInputBytes);
    buffers[States] = frozenBodyStates;
    const auto previous = m_outputProducer.lock();
    const bool reusableOutputs = !previous || previous->GetSubmission().IsValid();
    const auto outputBuffer = [&](std::shared_ptr<rhi::ComputeBuffer> &cache, uint64_t records) {
        if (!reusableOutputs || !cache || cache->GetByteSize() != records * sizeof(Float4))
            cache = std::make_shared<rhi::ComputeBuffer>(
                host, rhi::ComputeBufferDesc{records, rhi::ComputeScalarType::Float32, 4});
        return cache;
    };
    buffers[NodeModels] = outputBuffer(m_nodeModels, uint64_t(nodeCount) * 8);
    buffers[NodeModelsRead] = buffers[NodeModels];
    buffers[Models] = outputBuffer(m_models, uint64_t(count) * 4);
    buffers[Bounds] = outputBuffer(m_bounds, uint64_t(count) * 2);

    const auto dispatch = [&](const Entry &entry, uint32_t workCount, uint32_t first, bool nodeLevel) {
        rhi::ComputeDispatchDesc result;
        result.kernel = entry.kernel;
        result.groupCountX = groups(workCount);
        rhi::ComputeDispatchProof proof;
        for (const auto port : entry.ports) {
            const auto binding = static_cast<uint32_t>(result.buffers.size());
            result.buffers.push_back(buffers[port]);
            if (nodeLevel && port == NodeModels) {
                result.bufferAccesses.push_back(rhi::ComputeBufferAccess::ReadWrite);
                const uint64_t prefix = uint64_t(first) * 8 * sizeof(Float4);
                if (prefix)
                    proof.reads.push_back({binding, 0, prefix});
                proof.writes.push_back({binding, prefix, uint64_t(workCount) * 8 * sizeof(Float4)});
            } else if (port == States) {
                result.bufferAccesses.push_back(rhi::ComputeBufferAccess::Read);
                proof.reads.push_back({binding, stateOffset, stateBytes});
            } else {
                result.bufferAccesses.push_back(Writer(port) ? rhi::ComputeBufferAccess::Write
                                                             : rhi::ComputeBufferAccess::Read);
                auto &ranges = Writer(port) ? proof.writes : proof.reads;
                ranges.push_back({binding, 0, buffers[port]->GetByteSize()});
            }
        }
        result.proof = std::move(proof);
        return result;
    };
    std::vector<rhi::ComputeDispatchDesc> dispatches;
    dispatches.reserve(levels.size() + 1);
    m_levelParameters.resize(levels.size());
    for (size_t level = 0; level < levels.size(); ++level) {
        const auto [first, levelCount] = levels[level];
        const PoseParams parameters{{nodeCount, count, first, levelCount}, stateLayout};
        input(Parameters, m_levelParameters[level], &parameters, sizeof(parameters));
        dispatches.push_back(dispatch(m_nodes, levelCount, first, true));
    }
    const PoseParams parameters{{nodeCount, count, 0, 0}, {}};
    input(Parameters, m_parameters, &parameters, sizeof(parameters));
    dispatches.push_back(dispatch(m_meshes, count, 0, false));
    auto batch = rhi::PrepareComputeBatch(host, std::move(uploads), std::move(dispatches));
    m_outputProducer = batch;
    return {std::move(batch), buffers[Models], buffers[Bounds], count};
}

std::shared_ptr<rhi::PreparedComputeBatch>
PhysicsGpuRenderPoses::PrepareMotion(const PhysicsGpuMotionExport &input,
                                     const std::vector<PhysicsDeviceQueryOutput> &outputs,
                                     const std::vector<size_t> &scatter) const
{
    if (outputs.empty() || input.bodies.empty())
        return {};
    const bool origin = input.poses.layout == PhysicsGpuPoseSource::Layout::OriginWXYZ;
    const bool mapped = !scatter.empty();
    const bool needsAuthor = std::any_of(outputs.begin(), outputs.end(), [origin](const auto &output) {
        return (!origin && output.field == PhysicsDeviceQueryField::MotionPosition) ||
               output.field == PhysicsDeviceQueryField::MotionInverseMass ||
               output.field == PhysicsDeviceQueryField::MotionInverseInertia;
    });
    const uint32_t requestStride = needsAuthor ? 16 : (mapped ? 2 : 1);
    if (input.bodies.size() > UINT32_MAX / requestStride)
        throw std::length_error("GPU motion export exceeds its device index range");

    std::vector<uint32_t> packed(input.bodies.size() * requestStride);
    const auto word = [](float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(value));
        return bits;
    };
    for (size_t index = 0; index < input.bodies.size(); ++index) {
        const auto &entry = input.bodies[index];
        auto *row = packed.data() + index * requestStride;
        row[0] = entry.pose.row;
        if (needsAuthor) {
            row[2] = entry.frozenTranslation;
            for (size_t axis = 0; axis < 3; ++axis) {
                row[4 + axis] = word(entry.pose.centerOffset[axis]);
                row[8 + axis] = word(entry.inverseInertia[static_cast<glm::length_t>(axis)]);
            }
            row[7] = word(entry.inverseMass);
            row[11] = entry.frozenRotation;
            row[12] = word(entry.inertiaFrame.x);
            row[13] = word(entry.inertiaFrame.y);
            row[14] = word(entry.inertiaFrame.z);
            row[15] = word(entry.inertiaFrame.w);
        }
    }

    return PrepareFields(m_motion, input.poses, std::move(packed), requestStride, outputs, scatter);
}

std::shared_ptr<rhi::PreparedComputeBatch>
PhysicsGpuRenderPoses::PrepareBoxes(const PhysicsGpuBoxExport &input,
                                    const std::vector<PhysicsDeviceQueryOutput> &outputs,
                                    const std::vector<size_t> &scatter) const
{
    if (outputs.empty())
        return {};
    constexpr uint32_t stride = 16;
    if (input.boxes.size() > UINT32_MAX / stride)
        throw std::length_error("GPU box export exceeds its device index range");
    std::vector<uint32_t> packed(input.boxes.size() * stride);
    for (size_t index = 0; index < input.boxes.size(); ++index) {
        const auto &entry = input.boxes[index];
        const auto &box = entry.author;
        auto *row = packed.data() + index * stride;
        row[0] = entry.bodyRow;
        row[2] = entry.bodyIndex;
        const float values[]{
            box.centerOfMassOffset.x, box.centerOfMassOffset.y, box.centerOfMassOffset.z, box.friction,
            box.localRotation.x,      box.localRotation.y,      box.localRotation.z,      box.localRotation.w,
            box.halfExtents.x,        box.halfExtents.y,        box.halfExtents.z,        box.bounciness};
        std::memcpy(row + 4, values, sizeof(values));
    }
    return PrepareFields(m_box, input.poses, std::move(packed), stride, outputs, scatter);
}

std::shared_ptr<rhi::PreparedComputeBatch> PhysicsGpuRenderPoses::PrepareFields(
    const Entry &entry, const PhysicsGpuPoseSource &source, std::vector<uint32_t> packed, uint32_t requestStride,
    const std::vector<PhysicsDeviceQueryOutput> &outputs, const std::vector<size_t> &scatter) const
{
    const auto &states = source.states;
    const bool origin = source.layout == PhysicsGpuPoseSource::Layout::OriginWXYZ;
    const bool mapped = !scatter.empty();
    const auto total = static_cast<uint32_t>(packed.size() / requestStride);
    if (!states || !states->IsFrozen() || !m_host.SharesServicesWith(states->GetHost()) ||
        source.rowBytes < (origin ? 5 : 4) * sizeof(Float4) || source.rowBytes % sizeof(Float4) != 0 ||
        (source.firstRow + source.bodyCount) > UINT32_MAX / (source.rowBytes / sizeof(Float4)) ||
        (source.firstRow + source.bodyCount) > states->GetByteSize() / source.rowBytes ||
        (mapped && scatter.size() != total))
        throw std::invalid_argument("GPU state export requires frozen body rows from its compute owner");
    for (uint32_t index = 0; index < total; ++index) {
        auto *row = packed.data() + index * requestStride;
        if (row[0] >= source.bodyCount ||
            (mapped && (scatter[index] > UINT32_MAX || (index != 0 && scatter[index] <= scatter[index - 1]))))
            throw std::invalid_argument("GPU state export has an invalid source or scatter row");
        if (mapped)
            row[1] = static_cast<uint32_t>(scatter[index]);
    }

    const bool boxes = &entry == &m_box;
    auto host = m_host;
    std::vector<rhi::ComputeBufferUpdate> updates;
    const auto upload = [&](const auto &values) {
        const auto count = values.size();
        auto buffer = rhi::ComputeBuffer::TryCreatePrivate(host, {count, rhi::ComputeScalarType::UInt32, 1});
        if (!buffer)
            throw std::runtime_error("GPU state export exceeded its managed device budget");
        std::vector<uint8_t> bytes(count * sizeof(uint32_t));
        std::memcpy(bytes.data(), values.data(), bytes.size());
        updates.push_back({buffer, 0, std::move(bytes)});
        return buffer;
    };
    std::shared_ptr<rhi::ComputeBuffer> rows;
    std::vector<rhi::ComputeDispatchDesc> dispatches;
    for (const auto &output : outputs) {
        const auto field = static_cast<uint32_t>(output.field);
        const auto &buffer = output.buffer;
        const auto first = output.recordFirst;
        const auto count = output.recordCount == UINT32_MAX ? total - std::min(first, total) : output.recordCount;
        if (field >= PhysicsDeviceQueryWidths.size() ||
            boxes != (output.field >= PhysicsDeviceQueryField::ShapeBodyIndex) || !buffer || buffer->IsFrozen() ||
            !host.SharesServicesWith(buffer->GetHost()) || buffer->GetAllocation() == states->GetAllocation() ||
            first > total || count > total - first || output.byteOffset % sizeof(float) != 0)
            throw std::invalid_argument("GPU state export has an invalid field, selection or destination");
        const auto width = PhysicsDeviceQueryWidths[field];
        const auto end = std::min(output.byteEnd, buffer->GetByteSize());
        const auto byteOffset = output.byteOffset;
        const uint64_t slots = mapped && count != 0 ? scatter[first + count - 1] + 1 : count;
        const auto scalar = output.field == PhysicsDeviceQueryField::ShapeBodyIndex ? rhi::ComputeScalarType::Int32
                                                                                    : rhi::ComputeScalarType::Float32;
        if (buffer->GetDesc().scalarType != scalar || buffer->GetDesc().lanes != (width == 9 ? 1u : width) ||
            byteOffset > end || slots * width * sizeof(float) > end - byteOffset ||
            byteOffset / sizeof(float) > UINT32_MAX || slots * width > UINT32_MAX - byteOffset / sizeof(float))
            throw std::invalid_argument("GPU state export destination type or prefix capacity is invalid");
        if (count == 0)
            continue;
        if (!rows)
            rows = upload(packed);
        const std::array<uint32_t, 12> parameters{static_cast<uint32_t>(source.firstRow),
                                                  static_cast<uint32_t>(source.rowBytes / sizeof(Float4)),
                                                  static_cast<uint32_t>(origin),
                                                  requestStride,
                                                  first,
                                                  count,
                                                  static_cast<uint32_t>(byteOffset / sizeof(float)),
                                                  field,
                                                  width,
                                                  static_cast<uint32_t>(mapped),
                                                  0,
                                                  0};
        const auto uniform = upload(parameters);
        const std::array<std::shared_ptr<rhi::ComputeBuffer>, 4> inputs{uniform, rows, states, buffer};
        rhi::ComputeDispatchDesc dispatch;
        dispatch.kernel = entry.kernel;
        dispatch.groupCountX = (count - 1) / 64 + 1;
        if (dispatch.groupCountX > host.device.GetCapabilities().limits.maxComputeWorkgroupCount[0])
            throw std::length_error("GPU state export exceeds the device dispatch range");
        auto &proof = dispatch.proof.emplace();
        for (auto port : entry.ports) {
            const auto slot = static_cast<uint32_t>(dispatch.buffers.size());
            const auto inputIndex = port == MotionParameters ? 0 : port == MotionRows ? 1 : port == States ? 2 : 3;
            dispatch.buffers.push_back(inputs[inputIndex]);
            const bool writing = port == MotionValues || port == BoxValues;
            dispatch.bufferAccesses.push_back(writing ? rhi::ComputeBufferAccess::Write
                                                      : rhi::ComputeBufferAccess::Read);
            if (writing) {
                const auto stride = width * sizeof(float);
                for (uint32_t row = 0; row < count; ++row) {
                    const auto offset = byteOffset + (mapped ? scatter[first + row] : row) * stride;
                    if (proof.writes.empty() || proof.writes.back().offset + proof.writes.back().byteSize != offset)
                        proof.writes.push_back({slot, offset, stride});
                    else
                        proof.writes.back().byteSize += stride;
                }
            } else if (port == States) {
                proof.reads.push_back(
                    {slot, source.firstRow * source.rowBytes, uint64_t(source.bodyCount) * source.rowBytes});
            } else {
                proof.reads.push_back({slot, 0, inputs[inputIndex]->GetByteSize()});
            }
        }
        dispatches.push_back(std::move(dispatch));
    }
    return dispatches.empty() ? nullptr : rhi::PrepareComputeBatch(host, std::move(updates), std::move(dispatches));
}

} // namespace infernux
