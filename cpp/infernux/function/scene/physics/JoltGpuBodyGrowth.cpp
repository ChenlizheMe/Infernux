#include "JoltGpuBodyGrowth.h"

#include "JoltGpuBodyMutations.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace infernux
{
namespace
{
using namespace JPH::GPU;

constexpr uint32_t ItemsPerDispatch = 2048;
constexpr uint32_t OperationsPerPacket = 16;
constexpr uint64_t BytesPerCopy = 64u * 1024u;

enum Endpoint : uint8_t
{
    Models,
    Worlds,
    States,
    Forces,
    Targets,
    EndpointCount
};
constexpr std::array<Resource, EndpointCount> Resources{Resource::BodyModels, Resource::Worlds, Resource::BodyStates,
                                                        Resource::BodyForces, Resource::KinematicTargets};
constexpr std::array<uint64_t, EndpointCount> Strides{sizeof(BodyModel), sizeof(WorldRange), sizeof(BodyState),
                                                      sizeof(BodyForces), sizeof(KinematicTarget)};

std::vector<uint8_t> Bytes(const StageWork &work)
{
    std::vector<uint8_t> bytes(sizeof(work));
    std::memcpy(bytes.data(), &work, sizeof(work));
    return bytes;
}
} // namespace

class JoltGpuBodyGrowth::Impl : public std::enable_shared_from_this<JoltGpuBodyGrowth::Impl>
{
  public:
    Impl(JoltGpuStorage &resident, JoltGpuBodyMutations &mutationProgram, uint32_t expected, uint32_t target)
        : storage(resident), mutations(mutationProgram), host(resident.GetHost()), from(expected), to(target)
    {
        if (!from || to <= from)
            throw std::invalid_argument("Jolt GPU body growth requires an existing domain and a larger capacity");
    }

    void CheckBytes(uint64_t bytes) const
    {
        if (!bytes || bytes > std::numeric_limits<size_t>::max() ||
            !host.device.GetCapabilities().IsStorageBufferRangeValid(0, bytes))
            throw std::length_error("Jolt GPU body growth exceeds the host or device storage range: " +
                                    std::to_string(bytes));
    }

    void Capture()
    {
        // This happens at the actual control boundary, not when Scene queued
        // the request; preceding author/model groups have already been accepted.
        if (storage.GetBodyCount() != from)
            throw std::logic_error("Jolt GPU body growth source capacity differs from its queued boundary");
        const uint32_t worldCount = storage.GetWorldCount();
        if (!worldCount)
            throw std::invalid_argument("Jolt GPU body growth requires an accepted world range");

        // Validate every target size before acquiring any managed endpoint.
        for (uint8_t index = 0; index < EndpointCount; ++index) {
            const uint64_t count = index == Worlds ? worldCount : to;
            const uint64_t bytes = count * Strides[index];
            CheckBytes(bytes);
            const auto &previous = storage.Get(Resources[index]);
            const uint64_t oldCount = index == Worlds ? worldCount : from;
            if (!previous || !host.SharesServicesWith(previous->GetHost()) ||
                previous->GetByteSize() != oldCount * Strides[index])
                throw std::invalid_argument("Jolt GPU growth endpoint differs from its accepted body capacity");
            descriptions[index] = previous->GetDesc();
            const uint64_t elementStride = descriptions[index].GetElementStride();
            if (!elementStride || bytes % elementStride)
                throw std::invalid_argument("Jolt GPU growth cannot preserve its typed endpoint layout");
            descriptions[index].elementCount = bytes / elementStride;
            if (index >= States)
                source[index - States] = previous->Freeze();
        }

        firstTail = from;
    }

    bool Capacity()
    {
        for (uint8_t index = 0; index < EndpointCount; ++index) {
            if (!candidates[index])
                candidates[index] = rhi::ComputeBuffer::TryCreatePrivate(host, descriptions[index]);
            if (!candidates[index])
                return false;
        }
        return true;
    }

    JoltGpuBodyStatePorts NumericalPorts() const
    {
        return {to, candidates[States], candidates[Forces], candidates[Targets]};
    }

    JoltGpuControlWork Next()
    {
        if (!Capacity())
            return {nullptr, [self = shared_from_this()] { return self->Next(); }};
        std::vector<rhi::ComputeBufferUpdate> updates;
        std::vector<rhi::ComputeDispatchDesc> dispatches;
        std::vector<rhi::ComputeBufferCopy> copies;
        bool allocationBlocked = false;
        for (uint32_t operation = 0; operation < OperationsPerPacket; ++operation) {
            while (copyPort < source.size() && copyByte == uint64_t(from) * Strides[States + copyPort]) {
                copyByte = 0;
                ++copyPort;
            }
            if (copyPort < source.size()) {
                const uint64_t bytes = uint64_t(from) * Strides[States + copyPort];
                const uint64_t size = std::min(BytesPerCopy, bytes - copyByte);
                copies.push_back({source[copyPort], candidates[States + copyPort], {copyByte, copyByte, size}});
                copyByte += size;
                continue;
            }
            if (firstTail == to)
                break;
            if (!pendingUniform) {
                const rhi::ComputeBufferDesc layout{sizeof(StageWork) / sizeof(uint32_t),
                                                    rhi::ComputeScalarType::UInt32, 1};
                pendingUniform = rhi::ComputeBuffer::TryCreatePrivate(host, layout);
            }
            if (!pendingUniform) {
                allocationBlocked = true;
                break;
            }
            const uint32_t count = std::min(ItemsPerDispatch, to - firstTail);
            const StageWork work{{firstTail, count, 0, 0}};
            auto dispatch = mutations.PrepareCapacityInitialization(NumericalPorts(), firstTail, count, pendingUniform);
            updates.push_back({pendingUniform, 0, Bytes(work)});
            dispatches.push_back(std::move(dispatch));
            pendingUniform.reset();
            firstTail += count;
        }
        // The final copy can end exactly at the packet operation limit.
        while (copyPort < source.size() && copyByte == uint64_t(from) * Strides[States + copyPort]) {
            copyByte = 0;
            ++copyPort;
        }
        if (dispatches.empty() && copies.empty()) {
            if (allocationBlocked)
                return {nullptr, [self = shared_from_this()] { return self->Next(); }};
            throw std::logic_error("Jolt GPU body growth produced an empty preparation packet");
        }
        const bool final = copyPort == source.size() && firstTail == to;
        // Tail initialization and the copied old prefix are disjoint. Native
        // copies record after dispatches, so mixing these operations is safe;
        // no dispatch in this packet reads the copied prefix.
        auto packet = rhi::PrepareComputeBatch(host, std::move(updates), std::move(dispatches), {}, std::move(copies));
        return {packet, [self = shared_from_this(), packet, final]() -> JoltGpuControlWork {
                    if (!packet->GetSubmission().IsValid() || !packet->GetCompletion().IsComplete())
                        throw std::logic_error("Jolt GPU body growth requires its completed preparation receipt");
                    if (final) {
                        // Only a real completed receipt grants Commit. Prepared
                        // packets and partial capacity grants never grant it.
                        self->source = {};
                        self->ready = true;
                        return {};
                    }
                    return self->Next();
                }};
    }

    JoltGpuControlWork Prepare()
    {
        if (started)
            throw std::logic_error("Jolt GPU body growth preparation may start only once");
        started = true;
        Capture();
        return Next();
    }

    std::unique_ptr<JoltGpuModelUpdate> Commit()
    {
        if (!ready || committed || storage.GetBodyCount() != from)
            throw std::logic_error("Jolt GPU body growth commit requires its completed, uncommitted source boundary");
        committed = true;
        JoltGpuBodyGrowthBuffers buffers{
            std::move(candidates[Models]),
            std::move(candidates[Worlds]),
            {to, std::move(candidates[States]), std::move(candidates[Forces]), std::move(candidates[Targets])}};
        return storage.PrepareBodyGrowth(to, std::move(buffers));
    }

    JoltGpuStorage &storage;
    JoltGpuBodyMutations &mutations;
    rhi::ComputeHost host;
    const uint32_t from, to;
    std::array<rhi::ComputeBufferDesc, EndpointCount> descriptions{};
    std::array<std::shared_ptr<rhi::ComputeBuffer>, EndpointCount> candidates;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, 3> source;
    std::shared_ptr<rhi::ComputeBuffer> pendingUniform;
    uint32_t copyPort = 0, firstTail = 0;
    uint64_t copyByte = 0;
    bool started = false, ready = false, committed = false;
};

JoltGpuBodyGrowth::JoltGpuBodyGrowth(JoltGpuStorage &storage, JoltGpuBodyMutations &mutations,
                                     uint32_t expectedFromCapacity, uint32_t targetCapacity)
    : m_impl(std::make_shared<Impl>(storage, mutations, expectedFromCapacity, targetCapacity))
{
}

JoltGpuControlWork JoltGpuBodyGrowth::Prepare()
{
    return m_impl->Prepare();
}

std::unique_ptr<JoltGpuModelUpdate> JoltGpuBodyGrowth::Commit()
{
    return m_impl->Commit();
}

} // namespace infernux
