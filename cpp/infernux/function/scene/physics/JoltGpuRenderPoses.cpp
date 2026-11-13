#include "JoltGpuRenderPoses.h"

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
using namespace JPH::GPU;

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
    {"pose_params", sizeof(PoseParams), {"counts"}, 1, false},
    {"pose_nodes",
     sizeof(PoseNode),
     {"bodyParent", "rootCOM", "localScale", "parentRotation", "parentLocalToAncestor0", "parentLocalToAncestor1",
      "parentLocalToAncestor2", "parentLocalToAncestor3", "parentAncestorToLocal0", "parentAncestorToLocal1",
      "parentAncestorToLocal2", "parentAncestorToLocal3"},
     1,
     false},
    {"body_states", sizeof(BodyState), {"positionCOM", "rotation", "linearVelocity", "angularVelocity"}, 0, false},
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
};

bool Writer(uint8_t port)
{
    return port == NodeModels || port == Models || port == Bounds;
}

void ValidateVector(const ShaderValueLayout &value, ShaderScalarKind scalar)
{
    if (value.byteSize != 16 || value.kind != ShaderValueKind::Vector || value.scalar != scalar || value.rows != 1 ||
        value.columns != 4)
        throw std::invalid_argument("Jolt GPU render-pose target differs from its shared vector record");
}

void ValidateLayout(uint8_t port, const ShaderResourceLayout &resource)
{
    const auto &expected = Ports[port];
    const bool uniform = port == Parameters;
    if (resource.binding.type != (uniform ? rhi::BindingType::UniformBuffer : rhi::BindingType::StorageBuffer) ||
        (uniform ? resource.value.byteSize : resource.elementStride) != expected.stride ||
        (!uniform && resource.binding.readOnlyStorage == Writer(port)))
        throw std::invalid_argument("Jolt GPU render-pose target differs from its shared port layout");
    if (expected.vector) {
        ValidateVector(resource.value, ShaderScalarKind::Float32);
        return;
    }
    if (resource.value.kind != ShaderValueKind::Struct ||
        resource.value.fields.size() != expected.stride / sizeof(Float4))
        throw std::invalid_argument("Jolt GPU render-pose target differs from its shared structured record");
    for (size_t field = 0; field < resource.value.fields.size(); ++field) {
        const auto &value = resource.value.fields[field];
        if (value.name != expected.fields[field] || value.offset != field * sizeof(Float4))
            throw std::invalid_argument("Jolt GPU render-pose target has a different field name or offset");
        ValidateVector(value,
                       expected.integerFields & (1u << field) ? ShaderScalarKind::UInt32 : ShaderScalarKind::Float32);
    }
}

} // namespace

JoltGpuRenderPoses::JoltGpuRenderPoses(rhi::ComputeHost &host, const ShaderTargetCode &target) : m_host(host)
{
    const auto build = [&](const char *name, uint32_t required) {
        const auto code = std::find_if(target.entries.begin(), target.entries.end(),
                                       [&](const auto &entry) { return entry.name == name; });
        if (code == target.entries.end() || code->stage != rhi::ShaderStage::Compute ||
            code->workgroupSize != std::array<uint32_t, 3>{64, 1, 1} ||
            std::count_if(target.entries.begin(), target.entries.end(),
                          [&](const auto &entry) { return entry.name == name; }) != 1)
            throw std::invalid_argument("Jolt GPU render-pose target is missing its unique compute entry");
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
                throw std::invalid_argument("Jolt GPU render-pose entry contains an unknown resource");
            const auto index = static_cast<uint8_t>(port - std::begin(Ports));
            if (actual & (1u << index))
                throw std::invalid_argument("Jolt GPU render-pose entry repeats a resource");
            ValidateLayout(index, resource);
            bindings.push_back({resource.binding.binding, resource.binding.type, resource.binding.readOnlyStorage});
            entry.ports.push_back(index);
            actual |= 1u << index;
        }
        if (actual != required || code->bindings.size() != entry.ports.size())
            throw std::invalid_argument("Jolt GPU render-pose entry has an unexpected resource set");
        if (target.target == ShaderCodeTarget::SpirV) {
            if (code->code.empty() || code->code.size() % sizeof(uint32_t))
                throw std::invalid_argument("Jolt GPU render-pose target contains invalid SPIR-V");
            std::vector<uint32_t> words(code->code.size() / sizeof(uint32_t));
            std::memcpy(words.data(), code->code.data(), code->code.size());
            entry.kernel = std::make_shared<rhi::ComputeKernel>(
                m_host, rhi::ShaderModuleDesc::FromSpirV(words.data(), words.size()), code->name, std::move(bindings),
                0);
        } else if (target.target == ShaderCodeTarget::Wgsl) {
            entry.kernel = std::make_shared<rhi::ComputeKernel>(
                m_host, rhi::ShaderModuleDesc::FromWgsl(code->code.data(), code->code.size()), code->name,
                std::move(bindings), 0);
        } else {
            throw std::invalid_argument("Unsupported GPU render-pose shader target");
        }
        return entry;
    };
    m_nodes = build("export_body_pose_nodes", (1u << Parameters) | (1u << Nodes) | (1u << States) | (1u << NodeModels));
    m_meshes = build("export_render_poses",
                     (1u << Parameters) | (1u << Bindings) | (1u << NodeModelsRead) | (1u << Models) | (1u << Bounds));
}

JoltGpuRenderPoseBatch JoltGpuRenderPoses::Prepare(const std::shared_ptr<rhi::ComputeBuffer> &frozenBodyStates,
                                                   const std::vector<PoseNode> &nodes,
                                                   const std::vector<PoseBinding> &bindings) const
{
    if (bindings.empty())
        return {};
    if (nodes.empty() || nodes.size() > std::numeric_limits<uint32_t>::max() / 8 ||
        bindings.size() > std::numeric_limits<uint32_t>::max() / 4)
        throw std::invalid_argument("GPU render-pose inputs exceed their node or mesh device index range");
    if (!frozenBodyStates || !frozenBodyStates->IsFrozen() || !m_host.SharesServicesWith(frozenBodyStates->GetHost()) ||
        frozenBodyStates->GetByteSize() % sizeof(BodyState))
        throw std::invalid_argument("GPU render poses require a frozen final state from the same compute owner");
    const auto bodyCount = frozenBodyStates->GetByteSize() / sizeof(BodyState);
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
    (void)checkedBytes(bodyCount, sizeof(BodyState));

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
    const auto output = [&](std::shared_ptr<rhi::ComputeBuffer> &cache, uint64_t records) {
        if (!reusableOutputs || !cache || cache->GetByteSize() != records * sizeof(Float4))
            cache = std::make_shared<rhi::ComputeBuffer>(
                host, rhi::ComputeBufferDesc{records, rhi::ComputeScalarType::Float32, 4});
        return cache;
    };
    buffers[NodeModels] = output(m_nodeModels, uint64_t(nodeCount) * 8);
    buffers[NodeModelsRead] = buffers[NodeModels];
    buffers[Models] = output(m_models, uint64_t(count) * 4);
    buffers[Bounds] = output(m_bounds, uint64_t(count) * 2);

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
        const PoseParams parameters{{nodeCount, count, first, levelCount}};
        input(Parameters, m_levelParameters[level], &parameters, sizeof(parameters));
        dispatches.push_back(dispatch(m_nodes, levelCount, first, true));
    }
    const PoseParams parameters{{nodeCount, count, 0, 0}};
    input(Parameters, m_parameters, &parameters, sizeof(parameters));
    dispatches.push_back(dispatch(m_meshes, count, 0, false));
    auto batch = rhi::PrepareComputeBatch(host, std::move(uploads), std::move(dispatches));
    m_outputProducer = batch;
    return {std::move(batch), buffers[Models], buffers[Bounds], count};
}

} // namespace infernux
