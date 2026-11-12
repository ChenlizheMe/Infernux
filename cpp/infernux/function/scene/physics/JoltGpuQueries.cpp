#include "JoltGpuQueries.h"

#include <algorithm>
#include <array>
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
enum Port : uint8_t
{
    Parameters,
    Rays,
    Models,
    Shapes,
    States,
    Hits,
    PortCount
};
struct PortLayout
{
    std::string_view name;
    uint64_t stride;
    std::array<std::string_view, 6> fields;
    uint32_t integerFields;
};
constexpr PortLayout Ports[] = {
    {"raycast_params", sizeof(RaycastParams), {"counts"}, 1},
    {"raycast_inputs", sizeof(RaycastInput), {"originMaxDistance", "direction", "filter"}, 4},
    {"body_models",
     sizeof(BodyModel),
     {"meta", "mass", "invInertiaDiagonal", "inertiaRotation", "damping", "surface"},
     33},
    {"shape_models", sizeof(ShapeModel), {"geometry", "localPositionCOM", "localRotation"}, 0},
    {"body_states", sizeof(BodyState), {"positionCOM", "rotation", "linearVelocity", "angularVelocity"}, 0},
    {"raycast_hits", sizeof(JPH::GPU::RaycastHit), {"meta", "positionDistance", "normalFraction"}, 1},
};

template <class T> std::vector<uint8_t> Bytes(const T *records, size_t count)
{
    std::vector<uint8_t> bytes(count * sizeof(T));
    if (!bytes.empty())
        std::memcpy(bytes.data(), records, bytes.size());
    return bytes;
}

void ValidateLayout(uint8_t port, const ShaderResourceLayout &resource)
{
    const auto &expected = Ports[port];
    const bool uniform = port == Parameters;
    if (resource.binding.type != (uniform ? rhi::BindingType::UniformBuffer : rhi::BindingType::StorageBuffer) ||
        (uniform ? resource.value.byteSize : resource.elementStride) != expected.stride ||
        resource.value.kind != ShaderValueKind::Struct || resource.value.fields.size() != expected.stride / 16 ||
        (!uniform && resource.binding.readOnlyStorage != (port != Hits)))
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

void ValidateRay(const RaycastInput &ray)
{
    const auto &o = ray.originMaxDistance;
    const auto &d = ray.direction;
    if (!std::isfinite(o.x) || !std::isfinite(o.y) || !std::isfinite(o.z) || !std::isfinite(o.w) || o.w <= 0 ||
        !std::isfinite(d.x) || !std::isfinite(d.y) || !std::isfinite(d.z) || d.w != 0 ||
        (d.x == 0 && d.y == 0 && d.z == 0) || ray.filter.y || ray.filter.z || ray.filter.w)
        throw std::invalid_argument(
            "Jolt GPU ray requires finite origin/direction, nonzero direction and positive distance");
}
} // namespace

JoltGpuQueries::JoltGpuQueries(rhi::ComputeHost &host, const ShaderTargetCode &target) : m_host(host)
{
    const auto code = std::find_if(target.entries.begin(), target.entries.end(),
                                   [](const auto &entry) { return entry.name == "raycast_closest"; });
    if (code == target.entries.end() || code->stage != rhi::ShaderStage::Compute ||
        code->workgroupSize != std::array<uint32_t, 3>{64, 1, 1})
        throw std::invalid_argument("Jolt GPU query target is missing raycast_closest");
    uint32_t actual = 0;
    std::vector<rhi::ComputeBufferBinding> bindings;
    for (const auto &resource : target.resources) {
        const std::array<uint32_t, 2> key{resource.set, resource.binding.binding};
        if (std::find(code->bindings.begin(), code->bindings.end(), key) == code->bindings.end())
            continue;
        const auto port = std::find_if(std::begin(Ports), std::end(Ports),
                                       [&](const auto &value) { return value.name == resource.name; });
        if (port == std::end(Ports) || resource.set != 0)
            throw std::invalid_argument("Jolt GPU query target contains an unknown resource");
        const auto index = static_cast<uint8_t>(port - std::begin(Ports));
        if (actual & (1u << index))
            throw std::invalid_argument("Jolt GPU query target repeats a resource");
        ValidateLayout(index, resource);
        bindings.push_back({resource.binding.binding, resource.binding.type, resource.binding.readOnlyStorage});
        m_ports.push_back(index);
        actual |= 1u << index;
    }
    if (actual != (1u << PortCount) - 1 || code->bindings.size() != PortCount)
        throw std::invalid_argument("Jolt GPU query target has an unexpected resource set");
    if (target.target == ShaderCodeTarget::SpirV) {
        if (code->code.empty() || code->code.size() % sizeof(uint32_t))
            throw std::invalid_argument("Jolt GPU query target contains invalid SPIR-V");
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

JoltGpuRaycastBatch JoltGpuQueries::Prepare(const JoltGpuQuerySnapshot &snapshot, const std::vector<RaycastInput> &rays,
                                            bool readback) const
{
    if (rays.size() > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("Jolt GPU ray count exceeds device indexing");
    for (const auto &ray : rays)
        ValidateRay(ray);
    if (snapshot.bodyCount > MaxBodiesPerWorld)
        throw std::invalid_argument("Jolt GPU query world exceeds the admitted body bound");
    std::array<std::shared_ptr<rhi::ComputeBuffer>, PortCount> buffers;
    buffers[Models] = snapshot.bodyModels;
    buffers[Shapes] = snapshot.shapeModels;
    buffers[States] = snapshot.bodyStates;
    const bool emptySnapshot = !buffers[Models] && !buffers[Shapes] && !buffers[States];
    if (emptySnapshot && (snapshot.firstBody || snapshot.bodyCount))
        throw std::invalid_argument("Jolt GPU query nonempty range has no publication");
    if (!emptySnapshot) {
        for (const auto port : {Models, Shapes, States}) {
            const auto &buffer = buffers[port];
            if (!buffer || !buffer->IsFrozen() || !m_host.SharesServicesWith(buffer->GetHost()) ||
                buffer->GetByteSize() == 0 || buffer->GetByteSize() % Ports[port].stride)
                throw std::invalid_argument("Jolt GPU query inputs require matching frozen publication records");
        }
        const auto capacity = buffers[Models]->GetByteSize() / sizeof(BodyModel);
        if (buffers[States]->GetByteSize() / sizeof(BodyState) != capacity || snapshot.firstBody > capacity ||
            snapshot.bodyCount > capacity - snapshot.firstBody)
            throw std::invalid_argument("Jolt GPU query range differs from its published model/state capacity");
    }
    if (rays.empty())
        return {};

    // The empty-world case binds initialized dummy records without CPU query
    // evaluation. The kernel's zero-body loop still writes every miss on GPU.
    rhi::ComputeHost host = m_host;
    std::vector<rhi::ComputeBufferUpdate> updates;
    const auto upload = [&](uint8_t port, std::vector<uint8_t> bytes) {
        buffers[port] = std::make_shared<rhi::ComputeBuffer>(
            host, rhi::ComputeBufferDesc{bytes.size() / sizeof(uint32_t), rhi::ComputeScalarType::UInt32, 1});
        updates.push_back({buffers[port], 0, std::move(bytes)});
    };
    if (emptySnapshot) {
        const BodyModel model{};
        const ShapeModel shape{};
        const BodyState state{};
        upload(Models, Bytes(&model, 1));
        upload(Shapes, Bytes(&shape, 1));
        upload(States, Bytes(&state, 1));
    }
    const RaycastParams parameters{{static_cast<uint32_t>(rays.size()), snapshot.firstBody, snapshot.bodyCount, 0}};
    upload(Parameters, Bytes(&parameters, 1));
    upload(Rays, Bytes(rays.data(), rays.size()));
    buffers[Hits] = std::make_shared<rhi::ComputeBuffer>(
        host, rhi::ComputeBufferDesc{uint64_t(rays.size()) * sizeof(JPH::GPU::RaycastHit) / sizeof(uint32_t),
                                     rhi::ComputeScalarType::UInt32, 1});
    rhi::ComputeDispatchDesc dispatch;
    dispatch.kernel = m_kernel;
    dispatch.groupCountX = parameters.counts.x / 64 + static_cast<uint32_t>(parameters.counts.x % 64 != 0);
    rhi::ComputeDispatchProof proof;
    for (const auto port : m_ports) {
        const auto binding = static_cast<uint32_t>(dispatch.buffers.size());
        dispatch.buffers.push_back(buffers[port]);
        dispatch.bufferAccesses.push_back(port == Hits ? rhi::ComputeBufferAccess::Write
                                                       : rhi::ComputeBufferAccess::Read);
        auto &ranges = port == Hits ? proof.writes : proof.reads;
        ranges.push_back({binding, 0, buffers[port]->GetByteSize()});
    }
    dispatch.proof = std::move(proof);
    std::vector<rhi::ComputeBufferRead> reads;
    if (readback)
        reads.push_back({buffers[Hits], 0, buffers[Hits]->GetByteSize()});
    auto batch = rhi::PrepareComputeBatch(host, std::move(updates), {std::move(dispatch)}, std::move(reads));
    return {std::move(batch), buffers[Hits], parameters.counts.x};
}

} // namespace infernux
