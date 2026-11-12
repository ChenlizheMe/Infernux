#include "JoltGpuBodyMutations.h"

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
    Initializations,
    CentersOfMass,
    States,
    Forces,
    PortCount
};
struct PortLayout
{
    std::string_view name;
    uint64_t stride;
};
constexpr PortLayout Ports[] = {{"mutation_params", sizeof(BodyMutationParams)},
                                {"body_initializations", sizeof(BodyInitialization)},
                                {"center_of_mass_updates", sizeof(CenterOfMassUpdate)},
                                {"body_states", sizeof(BodyState)},
                                {"body_forces", sizeof(BodyForces)}};

template <class T> std::vector<uint8_t> Bytes(const T *records, size_t count)
{
    std::vector<uint8_t> bytes(count * sizeof(T));
    std::memcpy(bytes.data(), records, bytes.size());
    return bytes;
}

bool Finite3(const Float4 &value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

void ValidateInitial(const BodyInitialization &initial, const BodyModel &model)
{
    const auto &state = initial.state;
    const auto &q = state.rotation;
    if (!Finite3(state.positionCOM) || !Finite3(q) || !std::isfinite(q.w) ||
        std::abs(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w - 1.0f) > 1e-4f || !Finite3(state.linearVelocity) ||
        !Finite3(state.angularVelocity) || !Finite3(initial.forces.force) || !Finite3(initial.forces.torque))
        throw std::invalid_argument("Jolt GPU body initialization has invalid pose, velocity or force");
    const auto zero = [](const Float4 &v) { return v.x == 0 && v.y == 0 && v.z == 0; };
    if (model.meta.z == MotionStatic && (!zero(state.linearVelocity) || !zero(state.angularVelocity)))
        throw std::invalid_argument("Jolt GPU static body initialization must have zero velocity");
}
} // namespace

JoltGpuBodyMutations::JoltGpuBodyMutations(rhi::ComputeHost &host, const ShaderTargetCode &target) : m_host(host)
{
    const auto build = [&](const char *name, uint32_t required) {
        const auto code = std::find_if(target.entries.begin(), target.entries.end(),
                                       [&](const auto &entry) { return entry.name == name; });
        if (code == target.entries.end() || code->stage != rhi::ShaderStage::Compute ||
            code->workgroupSize != std::array<uint32_t, 3>{64, 1, 1})
            throw std::invalid_argument("Jolt GPU mutation target is missing its compute entry");
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
                throw std::invalid_argument("Jolt GPU mutation target has an unknown resource");
            const auto index = static_cast<uint8_t>(port - std::begin(Ports));
            const bool uniform = index == Parameters;
            if (resource.binding.type !=
                    (uniform ? rhi::BindingType::UniformBuffer : rhi::BindingType::StorageBuffer) ||
                resource.value.kind != ShaderValueKind::Struct ||
                (uniform ? resource.value.byteSize : resource.elementStride) != port->stride ||
                ((index == States || index == Forces) && resource.binding.readOnlyStorage))
                throw std::invalid_argument("Jolt GPU mutation target differs from its shared records");
            bindings.push_back({resource.binding.binding, resource.binding.type, resource.binding.readOnlyStorage});
            entry.ports.push_back(index);
            actual |= 1u << index;
        }
        if (actual != required)
            throw std::invalid_argument("Jolt GPU mutation entry has an unexpected resource set");
        if (target.target == ShaderCodeTarget::SpirV) {
            if (code->code.empty() || code->code.size() % sizeof(uint32_t))
                throw std::invalid_argument("Jolt GPU mutation target contains invalid SPIR-V");
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
        return entry;
    };
    m_initialize =
        build("initialize_bodies", (1u << Parameters) | (1u << Initializations) | (1u << States) | (1u << Forces));
    m_centerOfMass = build("update_centers_of_mass", (1u << Parameters) | (1u << CentersOfMass) | (1u << States));
}

std::unique_ptr<JoltGpuModelUpdate> JoltGpuBodyMutations::Prepare(
    JoltGpuStorage &storage, std::vector<JoltGpuBodyModelUpdate> bodies, std::vector<JoltGpuShapeModelUpdate> shapes,
    const std::vector<BodyInitialization> &initializations, const std::vector<CenterOfMassUpdate> &centersOfMass)
{
    if (!m_host.SharesServicesWith(storage.GetHost()))
        throw std::invalid_argument("Jolt GPU mutations and storage belong to different compute owners");
    if (initializations.size() > std::numeric_limits<uint32_t>::max() ||
        centersOfMass.size() > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("Jolt GPU body mutation count exceeds the device index range");
    std::vector<uint32_t> indices;
    indices.reserve(initializations.size() + centersOfMass.size());
    const auto checkIndex = [&](const UInt4 &index) {
        if (index.x >= storage.GetBodyCount() || index.y || index.z || index.w)
            throw std::invalid_argument("Jolt GPU body mutation slot is outside its model");
        indices.push_back(index.x);
    };
    std::sort(bodies.begin(), bodies.end(), [](const auto &a, const auto &b) { return a.index < b.index; });
    for (const auto &initial : initializations) {
        checkIndex(initial.bodyIndex);
        const auto changed = std::lower_bound(bodies.begin(), bodies.end(), initial.bodyIndex.x,
                                              [](const auto &value, uint32_t index) { return value.index < index; });
        const auto &model = changed != bodies.end() && changed->index == initial.bodyIndex.x
                                ? changed->model
                                : storage.GetModel().bodies[initial.bodyIndex.x];
        ValidateInitial(initial, model);
    }
    std::vector<CenterOfMassUpdate> rebases;
    rebases.reserve(centersOfMass.size());
    for (const auto &change : centersOfMass) {
        checkIndex(change.bodyIndex);
        const auto &delta = change.deltaLocalCOM;
        if (!Finite3(delta) || delta.w != 0)
            throw std::invalid_argument("Jolt GPU COM change must have a finite local delta and zero padding");
        if (delta.x || delta.y || delta.z)
            rebases.push_back(change);
    }
    std::sort(indices.begin(), indices.end());
    if (std::adjacent_find(indices.begin(), indices.end()) != indices.end())
        throw std::invalid_argument("Jolt GPU body mutations require unique slots");
    if (initializations.empty() && rebases.empty())
        return storage.PrepareModelUpdate(std::move(bodies), std::move(shapes));

    std::array<std::shared_ptr<rhi::ComputeBuffer>, PortCount> buffers;
    std::vector<rhi::ComputeBufferUpdate> uploads;
    const auto input = [&](Port port, std::vector<uint8_t> bytes) {
        auto buffer = std::make_shared<rhi::ComputeBuffer>(
            m_host, rhi::ComputeBufferDesc{bytes.size() / sizeof(uint32_t), rhi::ComputeScalarType::UInt32, 1});
        buffers[port] = buffer;
        uploads.push_back({std::move(buffer), 0, std::move(bytes)});
    };
    auto ordered = initializations;
    std::sort(ordered.begin(), ordered.end(),
              [](const auto &a, const auto &b) { return a.bodyIndex.x < b.bodyIndex.x; });
    if (!ordered.empty())
        input(Initializations, Bytes(ordered.data(), ordered.size()));
    if (!rebases.empty())
        input(CentersOfMass, Bytes(rebases.data(), rebases.size()));
    buffers[States] = storage.Get(Resource::BodyStates);
    buffers[Forces] = storage.Get(Resource::BodyForces);

    std::vector<rhi::ComputeDispatchDesc> dispatches;
    const auto dispatch = [&](const Entry &entry, uint32_t count, bool rebase, uint32_t firstSlot) {
        rhi::ComputeDispatchDesc desc;
        desc.kernel = entry.kernel;
        desc.groupCountX = count / 64 + static_cast<uint32_t>(count % 64 != 0);
        rhi::ComputeDispatchProof proof;
        for (const auto port : entry.ports) {
            const auto slot = static_cast<uint32_t>(desc.buffers.size());
            desc.buffers.push_back(buffers[port]);
            if (port < States) {
                desc.bufferAccesses.push_back(rhi::ComputeBufferAccess::Read);
                proof.reads.push_back({slot, 0, buffers[port]->GetByteSize()});
            } else if (port == States && rebase) {
                desc.bufferAccesses.push_back(rhi::ComputeBufferAccess::ReadWrite);
                const auto range =
                    std::minmax_element(rebases.begin(), rebases.end(),
                                        [](const auto &a, const auto &b) { return a.bodyIndex.x < b.bodyIndex.x; });
                proof.reads.push_back(
                    {slot, uint64_t(range.first->bodyIndex.x) * sizeof(BodyState),
                     uint64_t(range.second->bodyIndex.x - range.first->bodyIndex.x + 1) * sizeof(BodyState)});
            } else {
                desc.bufferAccesses.push_back(rhi::ComputeBufferAccess::Write);
                const auto stride = port == States ? sizeof(BodyState) : sizeof(BodyForces);
                proof.writes.push_back({slot, uint64_t(firstSlot) * stride, uint64_t(count) * stride});
            }
        }
        // Each initialization dispatch addresses one truly contiguous slot
        // run. Its complete-write proof never spans an unwritten hole.
        desc.proof = std::move(proof);
        dispatches.push_back(std::move(desc));
    };
    for (size_t begin = 0; begin < ordered.size();) {
        size_t end = begin + 1;
        while (end < ordered.size() && ordered[end].bodyIndex.x == ordered[end - 1].bodyIndex.x + 1)
            ++end;
        const auto count = static_cast<uint32_t>(end - begin);
        const BodyMutationParams params{{count, 0, static_cast<uint32_t>(begin), 0}};
        input(Parameters, Bytes(&params, 1));
        dispatch(m_initialize, count, false, ordered[begin].bodyIndex.x);
        begin = end;
    }
    if (!rebases.empty()) {
        const BodyMutationParams params{{0, static_cast<uint32_t>(rebases.size()), 0, 0}};
        input(Parameters, Bytes(&params, 1));
        dispatch(m_centerOfMass, params.counts.y, true, 0);
    }
    return storage.PrepareModelUpdate(std::move(bodies), std::move(shapes), std::move(uploads), std::move(dispatches));
}

} // namespace infernux
