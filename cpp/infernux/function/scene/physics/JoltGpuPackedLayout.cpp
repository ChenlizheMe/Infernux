#include "JoltGpuPackedLayout.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>

namespace infernux
{
class JoltGpuPackedLayout::Impl : public std::enable_shared_from_this<Impl>
{
  public:
    using Resource = JPH::GPU::Resource;
    struct CopyRun
    {
        uint64_t from, to, count;
    };

    Impl(JoltGpuStorage &owner, JoltGpuBodyMutations &mutationProgram, FrameWorkOwner &budget, JoltGpuModel target,
         std::vector<uint32_t> bodyRemap, std::vector<uint32_t> jointRemap)
        : storage(owner), mutations(mutationProgram), work(budget), model(std::move(target)),
          previousJoints(std::move(jointRemap)), host(owner.GetHost()),
          layout(std::make_shared<JoltGpuJointLayout>(model.joints)), oldLayout(owner.GetJointLayout())
    {
        if (bodyRemap.size() != model.bodies.size() || previousJoints.size() != model.joints.size())
            throw std::invalid_argument("Packed layout remaps must cover the complete target model");
        std::vector<uint8_t> seen(storage.GetBodyCount());
        for (uint32_t targetSlot = 0; targetSlot < bodyRemap.size(); ++targetSlot) {
            const auto source = bodyRemap[targetSlot];
            if (source == UINT32_MAX)
                continue;
            if (source >= seen.size() || seen[source]++)
                throw std::invalid_argument("Packed layout requires unique valid source body slots");
            if (!runs.empty() && runs.back().from + runs.back().count == source &&
                runs.back().to + runs.back().count == targetSlot)
                ++runs.back().count;
            else
                runs.push_back({source, targetSlot, 1u});
        }
        std::vector<uint8_t> jointSeen(storage.GetJointCount());
        for (uint32_t target = 0; target < previousJoints.size(); ++target) {
            const auto row = previousJoints[target];
            if (row == UINT32_MAX || row >= jointSeen.size() || jointSeen[row]++)
                throw std::invalid_argument("Packed layout requires unique valid source joint rows");
            const auto &old = oldLayout->Row(row);
            const auto &next = layout->Row(target);
            if (old.bodies.z != next.bodies.z || old.identity.x != next.identity.x || old.identity.y != next.identity.y)
                throw std::invalid_argument("Packed joint state remap changed stable identity or kind");
            const uint64_t from = uint64_t(old.layout.y) * 16u, to = uint64_t(next.layout.y) * 16u;
            const uint64_t bytes = uint64_t(JoltGpuJointLayout::StateWords(next.bodies.z)) * 16u;
            if (!jointRuns.empty() && jointRuns.back().from + jointRuns.back().count == from &&
                jointRuns.back().to + jointRuns.back().count == to)
                jointRuns.back().count += bytes;
            else
                jointRuns.push_back({from, to, bytes});
        }
        // Cold transaction only. Retain exactly this old numerical generation
        // until all remap reads retire; ordinary ticks never take these readers.
        for (size_t i = 0; i < Resources.size(); ++i) {
            const auto &port = storage.Get(Resources[i]);
            source[i] = port ? port->Freeze() : nullptr;
        }
        const auto &stream = storage.Get(Resource::Joints);
        oldJoints = stream ? stream->Freeze() : nullptr;
    }

    bool Capacity()
    {
        const uint64_t count = std::max<size_t>(1u, model.bodies.size());
        for (size_t i = 0; i < candidates.size(); ++i) {
            const uint64_t bytes = count * Strides[i];
            if (!host.device.GetCapabilities().IsStorageBufferRangeValid(0u, bytes))
                throw std::length_error("Packed layout numerical target exceeds device range");
            if (!candidates[i]) {
                const bool vectors = i == 0u || i == 1u;
                candidates[i] = rhi::ComputeBuffer::TryCreatePrivate(
                    host, {bytes / (vectors ? 16u : 4u),
                           vectors ? rhi::ComputeScalarType::Float32 : rhi::ComputeScalarType::UInt32,
                           static_cast<uint8_t>(vectors ? 4u : 1u)});
            }
            if (!candidates[i])
                return false;
        }
        const uint64_t streamBytes = layout->ByteSize();
        if (!host.device.GetCapabilities().IsStorageBufferRangeValid(0u, streamBytes))
            throw std::length_error("Packed layout joint stream exceeds device range");
        if (!jointStream)
            jointStream =
                rhi::ComputeBuffer::TryCreatePrivate(host, {streamBytes / 4u, rhi::ComputeScalarType::UInt32, 1});
        if (!jointStream)
            return false;
        return true;
    }

    JoltGpuControlWork Next()
    {
        using namespace JPH::GPU;
        if (!Capacity())
            return Retry();
        if (!jointPrepared) {
            auto prefix = layout->PrefixBytes(model.joints);
            jointPrepared = true;
            return Follow(rhi::PrepareComputeBatch(host, {{jointStream, 0u, std::move(prefix)}}, {}));
        }
        if (initialized < model.bodies.size()) {
            const auto count = std::min(uint32_t{2048}, static_cast<uint32_t>(model.bodies.size()) - initialized);
            auto uniform = rhi::ComputeBuffer::TryCreatePrivate(
                host, {sizeof(StageWork) / sizeof(uint32_t), rhi::ComputeScalarType::UInt32, 1});
            if (!uniform)
                return Retry();
            const StageWork range{{initialized, count, 0u, 0u}};
            std::vector<uint8_t> bytes(sizeof(range));
            std::memcpy(bytes.data(), &range, sizeof(range));
            auto dispatch =
                mutations.PrepareCapacityInitialization({static_cast<uint32_t>(model.bodies.size()), candidates[0],
                                                         candidates[1], candidates[2], candidates[3]},
                                                        initialized, count, uniform);
            auto packet = rhi::PrepareComputeBatch(host, {{uniform, 0u, std::move(bytes)}}, {std::move(dispatch)});
            initialized += count;
            return Follow(packet);
        }
        std::vector<rhi::ComputeBufferCopy> copies;
        const auto available = work.ResourceRemaining(FrameWorkResource::Transfer);
        const auto window = available ? available : work.ResourceState(FrameWorkResource::Transfer).limit;
        uint64_t remaining = std::max(uint64_t{16}, window - window % 16u);
        while (copyPort < candidates.size() + 1u && remaining) {
            const bool joint = copyPort == candidates.size();
            const auto &currentRuns = joint ? jointRuns : runs;
            if (copyRun == currentRuns.size()) {
                ++copyPort;
                copyRun = 0;
                continue;
            }
            const auto &run = currentRuns[copyRun];
            const uint64_t stride = joint ? 1u : Strides[copyPort];
            const uint64_t sourceBase = 0u, targetBase = 0u;
            const uint64_t total = uint64_t(run.count) * stride;
            const uint64_t bytes = std::min(remaining, total - copyOffset);
            copies.push_back({joint ? oldJoints : source[copyPort],
                              joint ? jointStream : candidates[copyPort],
                              {sourceBase + uint64_t(run.from) * stride + copyOffset,
                               targetBase + uint64_t(run.to) * stride + copyOffset, bytes}});
            copyOffset += bytes;
            remaining -= bytes;
            if (copyOffset == total) {
                copyOffset = 0;
                ++copyRun;
            }
        }
        if (!copies.empty())
            return Follow(rhi::PrepareComputeBatch(host, {}, {}, {}, std::move(copies)));
        source = {};
        oldJoints.reset();
        ready = true;
        return {};
    }

    JoltGpuControlWork Retry()
    {
        return {nullptr, [self = shared_from_this()] { return self->Next(); }};
    }
    JoltGpuControlWork Follow(std::shared_ptr<rhi::PreparedComputeBatch> packet)
    {
        return {packet, [self = shared_from_this()] { return self->Next(); }};
    }
    JoltGpuControlWork Prepare()
    {
        if (started)
            throw std::logic_error("Packed layout preparation is single-use");
        started = true;
        return Next();
    }
    std::unique_ptr<JoltGpuModelUpdate> Commit()
    {
        if (!ready || committed)
            throw std::logic_error("Packed layout commit requires completed remap receipts");
        committed = true;
        return storage.PreparePackedReplacement(std::move(model),
                                                {static_cast<uint32_t>(initialized), std::move(candidates[0]),
                                                 std::move(candidates[1]), std::move(candidates[2]),
                                                 std::move(candidates[3])},
                                                std::move(jointStream));
    }

    static constexpr std::array<Resource, 4> Resources{Resource::BodyStates, Resource::BodyForces,
                                                       Resource::KinematicTargets, Resource::SleepStates};
    static constexpr std::array<uint64_t, 4> Strides{sizeof(JPH::GPU::BodyState), sizeof(JPH::GPU::BodyForces),
                                                     sizeof(JPH::GPU::KinematicTarget),
                                                     sizeof(JPH::GPU::BodySleepRecord)};
    JoltGpuStorage &storage;
    JoltGpuBodyMutations &mutations;
    FrameWorkOwner &work;
    JoltGpuModel model;
    std::vector<uint32_t> previousJoints;
    rhi::ComputeHost host;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, 4> source, candidates;
    std::shared_ptr<rhi::ComputeBuffer> oldJoints, jointStream;
    std::shared_ptr<const JoltGpuJointLayout> layout, oldLayout;
    std::vector<CopyRun> runs, jointRuns;
    uint32_t initialized = 0;
    size_t copyPort = 0, copyRun = 0;
    uint64_t copyOffset = 0;
    bool started = false, ready = false, jointPrepared = false, committed = false;
};

JoltGpuPackedLayout::JoltGpuPackedLayout(JoltGpuStorage &storage, JoltGpuBodyMutations &mutations, FrameWorkOwner &work,
                                         JoltGpuModel model, std::vector<uint32_t> bodies,
                                         std::vector<uint32_t> jointRows)
    : m_impl(
          std::make_shared<Impl>(storage, mutations, work, std::move(model), std::move(bodies), std::move(jointRows)))
{
}
JoltGpuControlWork JoltGpuPackedLayout::Prepare()
{
    return m_impl->Prepare();
}
std::unique_ptr<JoltGpuModelUpdate> JoltGpuPackedLayout::Commit()
{
    return m_impl->Commit();
}
} // namespace infernux
