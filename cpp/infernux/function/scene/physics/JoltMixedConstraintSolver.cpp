#include "JoltMixedConstraintSolver.h"
#include "JoltCpuContactPublication.h"
#include "JoltCpuScene.h"
#include "JoltGpuProgramSources.h"
#include "JoltMixedContinuousContacts.h"

#include <Jolt/Physics/Body/MotionProperties.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace infernux
{
namespace
{
using namespace JPH::GPU;
using Contact = PhysicsContactPublication;
using Patch = JoltMixedContactStep::ResponsePatch;
using PatchKey = std::pair<Contact::EndpointKey, Contact::EndpointKey>;
constexpr uint32_t SolverLanes = 64;

struct Parameters
{
    UInt4 range;               // Serial groups, cooperative groups, rounds, clear accumulated CPU impulses.
    Float4 gravityDt, contact; // Warm-start scale, penetration slop, maximum correction, Baumgarte.
};
struct CpuBoundary
{
    BodyModel model;
    Float4 center, rotation;
};
struct Velocity
{
    Float4 linear, angular;
};
struct Pose
{
    Float4 center, rotation;
};
struct PositionAnchors
{
    std::array<Float4, MaxContactPointsPerPair> localA, localB;
};
struct DevicePatch
{
    // Zero points identifies a bilateral fixed joint. Its payload stores two
    // local COM arms and two local frames in place of contact points.
    UInt4 endpoints;
    Float4 normalFriction;
    std::array<Float4, MaxContactPointsPerPair> points;
    Float4 normalLambda, frictionLambda;
};
static_assert(sizeof(CpuBoundary) == 112);
static_assert(sizeof(Velocity) == 32);
static_assert(sizeof(Pose) == sizeof(Velocity));
static_assert(sizeof(PositionAnchors) == 128);
static_assert(sizeof(DevicePatch) == 128);

glm::vec3 XYZ(const Float4 &value)
{
    return {value.x, value.y, value.z};
}

template <typename V> Float4 Pack(V value)
{
    return {float(value.GetX()), float(value.GetY()), float(value.GetZ()), 0};
}

Float4 Pack(JPH::QuatArg rotation)
{
    return {rotation.GetX(), rotation.GetY(), rotation.GetZ(), rotation.GetW()};
}

JPH::Vec3 Vector(const Float4 &value)
{
    return {value.x, value.y, value.z};
}

JPH::Quat Rotation(const Float4 &value)
{
    return {value.x, value.y, value.z, value.w};
}

template <typename T> std::vector<uint8_t> Bytes(const T *source, size_t count)
{
    std::vector<uint8_t> bytes(count * sizeof(T));
    std::memcpy(bytes.data(), source, bytes.size());
    return bytes;
}

template <typename T> std::vector<uint8_t> Bytes(const std::vector<T> &source)
{
    return Bytes(source.data(), source.size());
}

PatchKey Key(const Patch &patch)
{
    return {patch.cpu.Key(), patch.gpu.Key()};
}

size_t GroupRoot(std::vector<size_t> &parents, size_t index)
{
    while (parents[index] != index) {
        parents[index] = parents[parents[index]];
        index = parents[index];
    }
    return index;
}

void LinkGroup(std::vector<size_t> &parents, std::unordered_map<uint32_t, size_t> &links, uint32_t endpoint,
               size_t index)
{
    const auto [first, added] = links.try_emplace(endpoint, index);
    if (!added)
        parents[GroupRoot(parents, index)] = GroupRoot(parents, first->second);
}
} // namespace

struct JoltMixedContactStep::ImpulseHistory
{
    struct Anchors
    {
        glm::vec3 cpu, gpu;
    };
    struct Manifold
    {
        PatchKey key;
        std::array<uint64_t, 2> shapeRevisions;
        glm::vec3 normal;
        uint32_t deviceIndex, pointCount;
        std::array<Anchors, MaxContactPointsPerPair> anchors;
    };
    // Sorted by the canonical manifold key, independently of device group order.
    std::vector<Manifold> manifolds;
    std::shared_ptr<rhi::ComputeBuffer> previousPatches, nextPatches;
    std::unordered_set<uint32_t> teleportedBodies;
    std::tuple<uint64_t, uint64_t, uint32_t> nextSubstep;
    float deltaTime = 0;
};

void JoltMixedContactStep::InvalidateImpulseHistory(uint32_t bodyId)
{
    if (m_impulseHistory)
        m_impulseHistory->teleportedBodies.insert(bodyId);
}

void JoltMixedContactStep::DiscardImpulseHistory() noexcept
{
    m_impulseHistory.reset();
}

JoltMixedConstraintSolver::JoltMixedConstraintSolver(rhi::ComputeHost &host, const ShaderTargetCode &target)
    : m_host(host)
{
    const auto load = [&](Program &program, std::string_view name) {
        const auto entry = std::find_if(target.entries.begin(), target.entries.end(),
                                        [&](const auto &code) { return code.name == name; });
        if (entry == target.entries.end() || entry->workgroupSize != std::array<uint32_t, 3>{SolverLanes, 1, 1})
            throw std::invalid_argument("Cross-domain solver shader has a different entry layout");
        std::vector<rhi::ComputeBufferBinding> bindings;
        for (const auto &resource : target.resources) {
            const auto &binding = resource.binding;
            const std::array<uint32_t, 2> key{resource.set, binding.binding};
            if (std::find(entry->bindings.begin(), entry->bindings.end(), key) == entry->bindings.end())
                continue;
            program.ports.push_back(binding.binding);
            bindings.push_back({binding.binding, binding.type, binding.readOnlyStorage});
        }
        program.kernel = PrepareJoltGpuKernel(host, target.target, *entry, std::move(bindings));
    };
    load(m_velocity, "exchange_cross_domain_velocity");
    load(m_position, "exchange_cross_domain_position");
    load(m_wake, "wake_cross_domain_bodies");
    m_continuous = std::make_shared<JoltMixedContinuousContacts>(host, target);
}

bool JoltMixedConstraintSolver::TryPreparePrograms(bool velocity, bool position, bool wake)
{
    // Start all needed driver jobs before checking whether preparation finished.
    const bool velocityReady = !velocity || m_velocity.kernel->TryPreparePipeline();
    const bool positionReady = !position || m_position.kernel->TryPreparePipeline();
    const bool wakeReady = !wake || m_wake.kernel->TryPreparePipeline();
    return velocityReady && positionReady && wakeReady;
}

struct JoltMixedConstraintSolver::Job : std::enable_shared_from_this<Job>
{
    enum Port : uint32_t
    {
        Params,
        GpuModels,
        GpuStates,
        CpuModels,
        CpuVelocities,
        Groups,
        Patches,
        Impulses,
        Sleep,
        Forces,
        WakeSlots,
        CpuPoses,
        Anchors,
        PreviousPatches,
        PortCount
    };

    JoltMixedConstraintSolver &owner;
    JoltGpuMixedContacts &discovery;
    JoltMixedContactStep &step;
    JoltCpuContactPublication &cpu;
    const JoltMixedContactStep::Policy &policy;
    std::shared_ptr<const JoltGpuScene::Publication> publication;
    std::shared_ptr<JoltMixedContactStep::ImpulseHistory> history;
    std::unordered_set<uint32_t> teleportedBodies;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, PortCount> buffers;
    std::vector<CpuBoundary> models;
    std::vector<Velocity> velocities, receipts;
    std::vector<Pose> poses, projected;
    std::vector<PositionAnchors> anchors;
    std::vector<DevicePatch> patches;
    std::vector<UInt4> groups;
    // Dynamic slots form the compact readback prefix. Prescribed boundaries are inputs only.
    std::vector<JPH::BodyID> native, prescribed;
    std::vector<uint32_t> wakeSlots;
    std::vector<size_t> order;
    struct EndpointPair
    {
        uint32_t cpuBody, gpuBody, gpuSlot;
        JPH::BodyID native;
        bool cpuDynamic, gpuDynamic, wakeGpu;
    };
    struct FixedRow
    {
        EndpointPair pair;
        std::array<Float4, 4> geometry;
        bool active;
    };
    struct FixedSource
    {
        uint32_t slot;
        bool dynamic, stateNeeded;
        UInt4 sleep{};
        BodyState state{};
    };
    struct FixedInput
    {
        std::shared_ptr<JoltMixedFixedJoint> joint;
        JPH::BodyID cpuId;
        size_t source;
    };
    std::vector<FixedRow> fixedRows;
    std::vector<JPH::PhysicsSystem::CoupledBodyImpulse> impulses;
    std::vector<JPH::PhysicsSystem::CoupledBodyCorrection> corrections;
    uint32_t substep = 0, substepCount = 0;
    uint32_t serialGroupCount = 0, cooperativeGroupCount = 0;
    uint32_t largestGroupPatchCount = 0;
    bool solveVelocity = false;
    bool solvePosition = false;
    bool interleaveCpu = false;
    bool velocityInitialized = false;
    bool solveFixed = false;
    float warmStartScale = 0.0f;
    Parameters params{};

    Job(JoltMixedConstraintSolver &owner, JoltGpuMixedContacts &discovery, JoltMixedContactStep &step,
        JoltCpuContactPublication &cpu, const JoltMixedContactStep::Policy &policy,
        std::shared_ptr<const JoltGpuScene::Publication> publication)
        : owner(owner), discovery(discovery), step(step), cpu(cpu), policy(policy), publication(std::move(publication))
    {
        if (!step.m_impulseHistory)
            step.m_impulseHistory = std::make_shared<JoltMixedContactStep::ImpulseHistory>();
        history = step.m_impulseHistory;
        teleportedBodies.swap(history->teleportedBodies);
    }

    const JPH::Body &Body(JPH::BodyID id) const
    {
        return *step.Scene().System().GetBodyLockInterfaceNoLock().TryGetBody(id);
    }

    void CaptureFixedRows(const std::vector<FixedInput> &joints, const std::vector<FixedSource> &sources)
    {
        fixedRows.reserve(joints.size());
        std::unordered_map<uint32_t, bool> cpuWake;
        for (const auto &input : joints) {
            auto &joint = *input.joint;
            const auto &source = sources[input.source];
            const auto &cpuBody = Body(input.cpuId);
            const auto &gpuActor = *publication->directory->actors[source.slot];
            const auto a = step.CpuMassFrame(joint.cpuBody);
            const JoltMixedContactStep::MassFrame b{Vector(gpuActor.rootScale), Vector(gpuActor.rootCenterOfMass)};
            const bool newJoint = !joint.reference;
            if (newJoint) {
                const auto qa = cpuBody.GetRotation(), qb = Rotation(source.state.rotation);
                const auto ca = cpuBody.GetCenterOfMassPosition();
                const JPH::RVec3 cb(Vector(source.state.positionCOM));
                const auto point = joint.cpuOwner ? ca + JPH::RVec3(qa * (joint.ownerBakedAnchor - a.localCOM))
                                                  : cb + JPH::RVec3(qb * (joint.ownerBakedAnchor - b.localCOM));
                const auto capture = [&](const auto &mass, JPH::RVec3Arg center, JPH::QuatArg q, bool owner) {
                    const auto baked = q.InverseRotate(JPH::Vec3(point - center)) + mass.localCOM;
                    JoltJointTransfer::Endpoint endpoint;
                    endpoint.frame = q.Conjugated();
                    endpoint.unscaled = owner ? joint.ownerLocalAnchor : JPH::Vec3::sZero();
                    for (uint32_t axis = 0; axis < 3; ++axis)
                        if (mass.scale[axis] != 0.0f)
                            endpoint.unscaled.SetComponent(axis, baked[axis] / mass.scale[axis]);
                        else
                            endpoint.collapsed.SetComponent(axis, baked[axis]);
                    return endpoint;
                };
                auto &reference = joint.reference.emplace();
                (joint.cpuOwner ? reference.owner : reference.connected) = capture(a, ca, qa, joint.cpuOwner);
                (joint.cpuOwner ? reference.connected : reference.owner) = capture(b, cb, qb, !joint.cpuOwner);
            }
            const bool cpuDynamic = cpuBody.IsDynamic();
            if (!cpuDynamic && !source.dynamic)
                continue;
            const bool cpuActive = cpuDynamic ? cpuBody.IsActive()
                                              : cpuBody.GetLinearVelocity() != JPH::Vec3::sZero() ||
                                                    cpuBody.GetAngularVelocity() != JPH::Vec3::sZero();
            const bool gpuActive = source.dynamic ? source.sleep.x != 0u
                                                  : Vector(source.state.linearVelocity) != JPH::Vec3::sZero() ||
                                                        Vector(source.state.angularVelocity) != JPH::Vec3::sZero();
            bool active = newJoint || cpuActive || gpuActive;
            if (!active && cpuDynamic)
                active = step.CpuJointMayWake(cpuBody, cpuWake);
            if (!active && cpuDynamic && source.dynamic)
                step.ObserveSleepingPair({joint.cpuBody, joint.gpuBody});
            const auto &ref = *joint.reference;
            const auto &cpuFrame = joint.cpuOwner ? ref.owner : ref.connected;
            const auto &gpuFrame = joint.cpuOwner ? ref.connected : ref.owner;
            fixedRows.push_back({{joint.cpuBody, joint.gpuBody, source.slot, input.cpuId, cpuDynamic, source.dynamic,
                                  source.dynamic && !gpuActive},
                                 {Pack(cpuFrame.unscaled * a.scale + cpuFrame.collapsed - a.localCOM),
                                  Pack(gpuFrame.unscaled * b.scale + gpuFrame.collapsed - b.localCOM),
                                  Pack(cpuFrame.frame), Pack(gpuFrame.frame)},
                                 active});
        }
    }

    JoltGpuControlWork ReadFixedSources(const JoltGpuStorage &storage, std::function<JoltGpuControlWork()> next)
    {
        fixedRows.clear();
        if (step.m_fixedJoints.empty() || (!solveVelocity && !solvePosition))
            return next();
        const auto &directory = *publication->directory;
        std::vector<FixedInput> joints;
        std::vector<FixedSource> sources;
        std::unordered_map<uint32_t, size_t> bySlot;
        for (const auto &joint : step.m_fixedJoints) {
            const auto slot = directory.FindBodySlot(joint->gpuBody);
            if (slot == UINT32_MAX || !directory.residentBodies[slot] || !step.Scene().HasBody(joint->cpuBody))
                continue;
            const auto cpuId = step.Scene().NativeBody(joint->cpuBody);
            if (!Body(cpuId).IsInBroadPhase())
                continue;
            const bool dynamic = directory.actors[slot]->acceptedMotionType == MotionDynamic;
            const bool stateNeeded = !dynamic || !joint->reference;
            const auto [entry, inserted] = bySlot.try_emplace(slot, sources.size());
            if (inserted)
                sources.push_back({slot, dynamic, stateNeeded});
            else
                sources[entry->second].stateNeeded |= stateNeeded;
            joints.push_back({joint, cpuId, entry->second});
        }
        if (joints.empty())
            return next();
        const auto &sleep = storage.Get(Resource::SleepStates);
        const auto &states = storage.Get(Resource::BodyStates);
        std::vector<rhi::ComputeBufferRead> reads;
        for (const auto &source : sources) {
            if (source.dynamic)
                reads.push_back({sleep,
                                 uint64_t(source.slot) * sizeof(BodySleepRecord) + offsetof(BodySleepRecord, runtime) +
                                     offsetof(BodySleepRuntime, state),
                                 sizeof(UInt4)});
            if (source.stateNeeded)
                reads.push_back({states, uint64_t(source.slot) * sizeof(BodyState), sizeof(BodyState)});
        }
        auto packet = rhi::PrepareComputeBatch(owner.m_host, {}, {}, std::move(reads));
        return {packet, [self = shared_from_this(), packet, joints = std::move(joints), sources = std::move(sources),
                         next = std::move(next)]() mutable {
                    size_t read = 0;
                    for (auto &source : sources) {
                        if (source.dynamic)
                            packet->Readback(read++)->ReadInto(&source.sleep, sizeof(source.sleep));
                        if (source.stateNeeded)
                            packet->Readback(read++)->ReadInto(&source.state, sizeof(source.state));
                    }
                    self->CaptureFixedRows(joints, sources);
                    return next();
                }};
    }

    EndpointPair Endpoints(size_t index) const
    {
        const auto &responses = step.Responses();
        if (index >= responses.size())
            return fixedRows[index - responses.size()].pair;
        const auto &response = responses[index];
        return {response.cpu.body,   response.gpu.body,   response.gpuSlot, response.native,
                response.cpuDynamic, response.gpuDynamic, response.wakeGpu};
    }

    void BuildGroupSchedule(const std::vector<UInt4> &connectedGroups)
    {
        // Headers precede the level ranges and packed patch ordinals. The stable
        // patch buffer order is still used by warm start and contact publication.
        groups.resize(connectedGroups.size());
        serialGroupCount = cooperativeGroupCount = 0;
        for (const auto &group : connectedGroups) {
            if (group.y < SolverLanes / 2) {
                groups[serialGroupCount++] = group;
                continue;
            }

            std::vector<uint32_t> nextCpuLevel(group.w);
            std::unordered_map<uint32_t, uint32_t> nextGpuLevel;
            std::vector<uint32_t> rowLevels(group.y), levelCounts;
            for (uint32_t row = 0; row < group.y; ++row) {
                const uint32_t patch = group.x + row;
                const auto pair = Endpoints(order[patch]);
                auto *cpuLevel = pair.cpuDynamic ? &nextCpuLevel[patches[patch].endpoints.x - group.z] : nullptr;
                auto *gpuLevel = pair.gpuDynamic ? &nextGpuLevel[pair.gpuSlot] : nullptr;
                const uint32_t level = std::max(cpuLevel ? *cpuLevel : 0u, gpuLevel ? *gpuLevel : 0u);
                if (cpuLevel)
                    *cpuLevel = level + 1u;
                if (gpuLevel)
                    *gpuLevel = level + 1u;
                rowLevels[row] = level;
                if (level == levelCounts.size())
                    levelCounts.push_back(0u);
                ++levelCounts[level];
            }
            // Preserve the dependency order at each writer. A chain with little
            // independent work uses one lane, like the small connected groups.
            if (levelCounts.size() > group.y / 2u) {
                groups[serialGroupCount++] = group;
                continue;
            }
            const auto firstLevel = static_cast<uint32_t>(groups.size());
            const auto levelCount = static_cast<uint32_t>(levelCounts.size());
            const auto patchWords = firstLevel + levelCount;
            groups[connectedGroups.size() - ++cooperativeGroupCount] = {firstLevel, levelCount, group.z, group.w};
            groups.resize(patchWords + (group.y + 3u) / 4u);
            uint32_t offset = 0;
            for (uint32_t level = 0; level < levelCount; ++level) {
                const uint32_t count = levelCounts[level];
                groups[firstLevel + level] = {patchWords * 4u + offset, count, 0, 0};
                levelCounts[level] = offset;
                offset += count;
            }
            std::vector<uint32_t> orderedPatches(group.y);
            for (uint32_t row = 0; row < group.y; ++row)
                orderedPatches[levelCounts[rowLevels[row]]++] = group.x + row;
            std::memcpy(groups.data() + patchWords, orderedPatches.data(), group.y * sizeof(uint32_t));
        }
    }

    void BuildWorkset(JoltGpuStepInputs &inputs)
    {
        auto &responses = step.Responses();
        std::sort(responses.begin(), responses.end(), [](const Patch &a, const Patch &b) { return Key(a) < Key(b); });
        order.resize(responses.size() + fixedRows.size());
        std::iota(order.begin(), order.end(), size_t(0));
        auto parents = order;
        std::unordered_map<uint32_t, size_t> cpuLinks, gpuLinks;
        std::vector<uint32_t> gpuContactBodies;
        gpuContactBodies.reserve(order.size());
        for (size_t i = 0; i < order.size(); ++i) {
            const auto pair = Endpoints(i);
            if (pair.cpuDynamic)
                LinkGroup(parents, cpuLinks, pair.cpuBody, i);
            if (pair.gpuDynamic)
                LinkGroup(parents, gpuLinks, pair.gpuSlot, i);
        }
        for (size_t i = 0; i < parents.size(); ++i)
            parents[i] = GroupRoot(parents, i);
        if (!fixedRows.empty()) {
            std::vector<uint8_t> active(order.size());
            for (size_t i = 0; i < order.size(); ++i)
                active[parents[i]] |= i < responses.size() || fixedRows[i - responses.size()].active;
            order.erase(std::remove_if(order.begin(), order.end(), [&](size_t i) { return !active[parents[i]]; }),
                        order.end());
            cpuLinks.clear();
            for (const size_t i : order) {
                const auto pair = Endpoints(i);
                if (pair.cpuDynamic)
                    cpuLinks.try_emplace(pair.cpuBody, i);
            }
        }
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return parents[a] < parents[b]; });

        const auto dynamicCount = cpuLinks.size();
        native.clear();
        prescribed.clear();
        std::vector<UInt4> connectedGroups;
        patches.clear();
        wakeSlots.clear();
        largestGroupPatchCount = 0;
        solveFixed = false;
        patches.reserve(order.size());
        cpuLinks.clear();
        size_t previous = SIZE_MAX;
        for (const size_t index : order) {
            if (parents[index] != previous) {
                previous = parents[index];
                connectedGroups.push_back(
                    {static_cast<uint32_t>(patches.size()), 0, static_cast<uint32_t>(native.size()), 0});
            }
            const auto pair = Endpoints(index);
            gpuContactBodies.push_back(pair.gpuSlot);
            if (pair.wakeGpu)
                wakeSlots.push_back(pair.gpuSlot);
            auto &group = connectedGroups.back();
            const size_t slot = pair.cpuDynamic ? native.size() : dynamicCount + prescribed.size();
            const auto [mapped, added] = cpuLinks.try_emplace(pair.cpuBody, slot);
            if (added) {
                if (pair.cpuDynamic) {
                    native.push_back(pair.native);
                    ++group.w;
                } else {
                    prescribed.push_back(pair.native);
                }
            }
            DevicePatch value{};
            value.endpoints = {static_cast<uint32_t>(mapped->second), pair.gpuSlot, 0, 0};
            if (index < responses.size()) {
                value.endpoints.z = responses[index].pointCount;
                value.normalFriction = responses[index].normalFriction;
            } else {
                value.points = fixedRows[index - responses.size()].geometry;
                solveFixed = true;
            }
            patches.push_back(value);
            ++group.y;
            largestGroupPatchCount = std::max(largestGroupPatchCount, group.y);
        }
        BuildGroupSchedule(connectedGroups);
        std::sort(gpuContactBodies.begin(), gpuContactBodies.end());
        gpuContactBodies.erase(std::unique(gpuContactBodies.begin(), gpuContactBodies.end()), gpuContactBodies.end());
        inputs.SetExternalContactBodies(std::move(gpuContactBodies));
    }

    struct SleepEdge
    {
        uint32_t cpuIsland, gpuEndpoint;
        bool cpuReady;
        std::pair<uint32_t, uint32_t> bodies;
    };

    std::vector<uint32_t> ResolveSleep(const std::vector<SleepEdge> &edges, const std::vector<UInt4> &votes)
    {
        std::vector<size_t> parents(edges.size());
        std::iota(parents.begin(), parents.end(), size_t{0});
        std::unordered_map<uint32_t, size_t> cpuIslands, gpuIslands;
        for (size_t i = 0; i < edges.size(); ++i) {
            const auto &edge = edges[i];
            LinkGroup(parents, cpuIslands, edge.cpuIsland, i);
            LinkGroup(parents, gpuIslands, votes[edge.gpuEndpoint].x, i);
        }
        std::vector<uint8_t> ready(edges.size(), 1);
        for (size_t i = 0; i < edges.size(); ++i) {
            const auto &edge = edges[i];
            ready[GroupRoot(parents, i)] &= uint8_t(edge.cpuReady && votes[edge.gpuEndpoint].y != 0);
        }
        std::vector<uint32_t> sleeping, awake;
        for (const auto &[island, edge] : cpuIslands)
            if (ready[GroupRoot(parents, edge)])
                sleeping.push_back(island);
        for (const auto &[island, edge] : gpuIslands)
            if (!ready[GroupRoot(parents, edge)])
                awake.push_back(votes[edges[edge].gpuEndpoint].z);
        for (size_t i = 0; i < edges.size(); ++i) {
            const auto island = GroupRoot(parents, i);
            if (ready[island])
                step.m_pendingSleepLinks.push_back({edges[i].bodies, island});
        }
        std::sort(sleeping.begin(), sleeping.end());
        std::sort(awake.begin(), awake.end());
        cpu.FinishCoupledSleep(sleeping);
        return awake;
    }

    JoltGpuSleepCoupling PrepareSleep()
    {
        JoltGpuSleepCoupling coupling;
        std::vector<SleepEdge> edges;
        bool anyCpuReady = false;
        for (const auto index : order) {
            const auto pair = Endpoints(index);
            if (!pair.cpuDynamic || !pair.gpuDynamic)
                continue;
            const auto island = cpu.CoupledSleepIsland(pair.native);
            edges.push_back({island.mIndex, pair.gpuSlot, island.mReady, {pair.cpuBody, pair.gpuBody}});
            coupling.endpoints.push_back(pair.gpuSlot);
            anyCpuReady |= island.mReady;
        }
        auto &endpoints = coupling.endpoints;
        std::sort(endpoints.begin(), endpoints.end());
        endpoints.erase(std::unique(endpoints.begin(), endpoints.end()), endpoints.end());
        if (!anyCpuReady) {
            // Every mixed component must stay awake. No GPU receipt can change that decision.
            cpu.FinishCoupledSleep({});
        } else {
            for (auto &edge : edges)
                edge.gpuEndpoint = static_cast<uint32_t>(
                    std::lower_bound(endpoints.begin(), endpoints.end(), edge.gpuEndpoint) - endpoints.begin());
            coupling.resolve = [self = shared_from_this(), edges = std::move(edges)](const auto &votes) {
                return self->ResolveSleep(edges, votes);
            };
        }
        return coupling;
    }

    void BeginCpuSubstep(JoltGpuStepInputs &inputs)
    {
        std::vector<JPH::BodyID> wakeEndpoints;
        const auto &directory = *publication->directory;
        for (const uint32_t id : step.TakeWakingIslands()) {
            const uint32_t gpu = directory.FindBodySlot(id);
            if (gpu != UINT32_MAX && directory.residentBodies[gpu]) {
                wakeSlots.push_back(gpu);
            } else if (step.Scene().HasBody(id)) {
                const auto nativeId = step.Scene().NativeBody(id);
                const auto &body = Body(nativeId);
                if (body.IsInBroadPhase() && body.IsDynamic() && !body.IsActive())
                    wakeEndpoints.push_back(nativeId);
            }
        }
        std::sort(wakeSlots.begin(), wakeSlots.end());
        wakeSlots.erase(std::unique(wakeSlots.begin(), wakeSlots.end()), wakeSlots.end());
        std::vector<JPH::BodyID> sleepEndpoints;
        if (substep + 1 == substepCount) {
            for (const auto index : order) {
                const auto pair = Endpoints(index);
                if (pair.cpuDynamic && pair.gpuDynamic)
                    sleepEndpoints.push_back(pair.native);
            }
            std::sort(sleepEndpoints.begin(), sleepEndpoints.end());
            sleepEndpoints.erase(std::unique(sleepEndpoints.begin(), sleepEndpoints.end()), sleepEndpoints.end());
        }
        cpu.BeginCoupledSubstep(substep, native, sleepEndpoints, wakeEndpoints);
        if (!sleepEndpoints.empty())
            inputs.SetSleepCoupling([self = shared_from_this()] { return self->PrepareSleep(); });
    }

    float MatchWarmStarts()
    {
        const auto &settings = step.Scene().System().GetPhysicsSettings();
        if (!settings.mConstraintWarmStart || history->manifolds.empty() ||
            history->nextSubstep != std::tuple(publication->simulationEpoch, publication->fixedStepIndex, substep))
            return 0.0f;

        bool matched = false;
        const auto &responses = step.Responses();
        for (size_t i = 0; i < order.size(); ++i) {
            if (order[i] >= responses.size())
                continue;
            const auto &response = responses[order[i]];
            if (substep == 0 && (teleportedBodies.find(response.gpu.body) != teleportedBodies.end() ||
                                 Body(response.native).IsCollisionCacheInvalid()))
                continue;
            const auto key = Key(response);
            const auto found = std::lower_bound(history->manifolds.begin(), history->manifolds.end(), key,
                                                [](const auto &cached, const auto &key) { return cached.key < key; });
            constexpr float alignedNormals = 0.95f;
            if (found == history->manifolds.end() || found->key != key ||
                found->shapeRevisions != response.shapeRevisions ||
                glm::dot(found->normal, XYZ(response.normalFriction)) < alignedNormals)
                continue;

            std::array<float, MaxContactPointsPerPair> pointIndices{};
            uint32_t used = 0;
            for (uint32_t point = 0; point < response.pointCount; ++point) {
                const auto &current = response.points[point];
                uint32_t nearest = found->pointCount;
                float distance = std::numeric_limits<float>::max();
                for (uint32_t previous = 0; previous < found->pointCount; ++previous) {
                    if (used & (1u << previous))
                        continue;
                    const auto da = current.localAnchorA - found->anchors[previous].cpu;
                    const auto db = current.localAnchorB - found->anchors[previous].gpu;
                    const float sa = glm::dot(da, da), sb = glm::dot(db, db);
                    const float limit = settings.mContactPointPreserveLambdaMaxDistSq;
                    if (sa <= limit && sb <= limit && sa + sb < distance) {
                        distance = sa + sb;
                        nearest = previous;
                    }
                }
                if (nearest != found->pointCount) {
                    used |= 1u << nearest;
                    pointIndices[point] = static_cast<float>(nearest + 1);
                }
            }
            if (used) {
                matched = true;
                auto &patch = patches[i];
                patch.endpoints.w = found->deviceIndex + 1;
                patch.normalLambda = {pointIndices[0], pointIndices[1], pointIndices[2], pointIndices[3]};
            }
        }
        return matched ? params.gravityDt.w / history->deltaTime : 0.0f;
    }

    void RetainImpulses()
    {
        const auto &responses = step.Responses();
        history->manifolds.resize(responses.size());
        for (uint32_t i = 0; i < order.size(); ++i) {
            if (order[i] >= responses.size())
                continue;
            const auto &response = responses[order[i]];
            auto &cached = history->manifolds[order[i]];
            cached.key = Key(response);
            cached.shapeRevisions = response.shapeRevisions;
            cached.normal = XYZ(response.normalFriction);
            cached.deviceIndex = i;
            cached.pointCount = response.pointCount;
            for (uint32_t point = 0; point < response.pointCount; ++point) {
                const auto &current = response.points[point];
                cached.anchors[point] = {current.localAnchorA, current.localAnchorB};
            }
        }
        history->previousPatches = buffers[Patches];
        history->nextPatches = buffers[PreviousPatches];
        const bool lastSubstep = substep + 1 == substepCount;
        history->nextSubstep = {publication->simulationEpoch, publication->fixedStepIndex + lastSubstep,
                                lastSubstep ? 0u : substep + 1};
        history->deltaTime = params.gravityDt.w;
    }

    void CaptureModels()
    {
        models.resize(native.size() + prescribed.size());
        velocities.resize(models.size());
        for (size_t i = 0; i < models.size(); ++i) {
            const bool dynamic = i < native.size();
            const auto id = dynamic ? native[i] : prescribed[i - native.size()];
            const auto &body = Body(id);
            auto &boundary = models[i];
            boundary.model = {};
            boundary.model.meta.z = dynamic ? MotionDynamic : MotionKinematic;
            boundary.center = Pack(body.GetCenterOfMassPosition());
            boundary.rotation = Pack(body.GetRotation());
            if (dynamic) {
                const auto *motion = body.GetMotionProperties();
                const auto locks = ~static_cast<uint32_t>(motion->GetAllowedDOFs());
                boundary.model.mass = {motion->GetInverseMass(), float(locks & 7u), float((locks >> 3) & 7u), 0};
                boundary.model.invInertiaDiagonal = Pack(motion->GetInverseInertiaDiagonal());
                boundary.model.inertiaRotation = Pack(motion->GetInertiaRotation());
                const auto free = cpu.CoupledFreeVelocity(id);
                velocities[i] = {Pack(free.mLinear), Pack(free.mAngular)};
            } else {
                // Capture velocity before Jolt advances the prescribed body to the end of the substep.
                velocities[i] = {Pack(body.GetLinearVelocity()), Pack(body.GetAngularVelocity())};
            }
        }
        anchors.resize(solvePosition ? patches.size() : 0);
        const float dt = params.gravityDt.w;
        const float minImpact = step.Scene().System().GetPhysicsSettings().mMinVelocityForRestitution;
        auto &responses = step.Responses();
        for (size_t i = 0; i < order.size(); ++i) {
            if (order[i] >= responses.size())
                continue;
            auto &response = responses[order[i]];
            auto &patch = patches[i];
            const auto &free = velocities[patch.endpoints.x];
            const auto v = XYZ(free.linear);
            const auto w = XYZ(free.angular);
            const auto center = XYZ(models[patch.endpoints.x].center);
            for (uint32_t j = 0; j < response.pointCount; ++j) {
                auto &point = response.points[j];
                const auto position = XYZ(point.positionDepth);
                point.gpuVelocity -= v + glm::cross(w, position - center);
                const float speed = glm::dot(point.gpuVelocity, XYZ(patch.normalFriction));
                const float speculative = std::max(0.0f, -point.positionDepth.w / dt);
                const float bias = speed < -minImpact && speed < -speculative && response.restitution > 0.0f
                                       ? response.restitution * speed
                                       : speculative;
                patch.points[j] = {position.x, position.y, position.z, bias};
                if (solvePosition) {
                    const auto &a = point.localAnchorA;
                    const auto &b = point.localAnchorB;
                    anchors[i].localA[j] = {a.x, a.y, a.z, 0};
                    anchors[i].localB[j] = {b.x, b.y, b.z, 0};
                }
            }
        }
    }

    bool Allocate(Port port, uint64_t bytes)
    {
        auto &buffer = buffers[port];
        if (buffer && buffer->GetByteSize() >= bytes)
            return true;
        buffer = rhi::ComputeBuffer::TryCreatePrivate(owner.m_host, {bytes / 4, rhi::ComputeScalarType::UInt32, 1, {}});
        return bool(buffer);
    }

    uint32_t SolverWorkgroups() const
    {
        return (serialGroupCount + SolverLanes - 1u) / SolverLanes + cooperativeGroupCount;
    }

    rhi::ComputeDispatchDesc Dispatch(const Program &program, uint32_t workgroups) const
    {
        rhi::ComputeDispatchDesc dispatch;
        dispatch.kernel = program.kernel;
        const auto width = owner.m_host.device.GetCapabilities().limits.maxComputeWorkgroupCount[0];
        dispatch.groupCountX = std::min(workgroups, width);
        dispatch.groupCountY = static_cast<uint32_t>((uint64_t(workgroups) + width - 1u) / width);
        for (const auto port : program.ports) {
            dispatch.buffers.push_back(buffers[port]);
            const bool modifies =
                port == GpuStates || port == CpuVelocities || port == CpuPoses || port == Patches || port == Sleep;
            using Access = rhi::ComputeBufferAccess;
            if (port == Impulses)
                dispatch.bufferAccesses.push_back(params.range.w ? Access::Write : Access::ReadWrite);
            else
                dispatch.bufferAccesses.push_back(modifies ? Access::ReadWrite : Access::Read);
        }
        return dispatch;
    }

    bool AllocateWorkset()
    {
        const auto velocityBytes = models.size() * sizeof(Velocity);
        return Allocate(CpuModels, models.size() * sizeof(CpuBoundary)) && Allocate(CpuVelocities, velocityBytes) &&
               Allocate(Groups, groups.size() * sizeof(UInt4)) &&
               Allocate(Patches, patches.size() * sizeof(DevicePatch)) &&
               (!solveVelocity || (Allocate(PreviousPatches, sizeof(DevicePatch)) &&
                                   Allocate(Impulses, std::max(size_t{1}, native.size()) * sizeof(Velocity)))) &&
               (!solvePosition || Allocate(Anchors, anchors.size() * sizeof(PositionAnchors)));
    }

    JoltGpuControlWork Install(std::function<void()> done)
    {
        if (patches.empty() && wakeSlots.empty()) {
            done();
            return {};
        }
        const bool pipelinesReady = !owner.m_host.SupportsPipelinePreparation() ||
                                    owner.TryPreparePrograms(solveVelocity && !patches.empty(),
                                                             solvePosition && !patches.empty(), !wakeSlots.empty());
        if (!pipelinesReady || !Allocate(Params, sizeof(params)) || (!patches.empty() && !AllocateWorkset()) ||
            (!wakeSlots.empty() && !Allocate(WakeSlots, wakeSlots.size() * sizeof(uint32_t)))) {
            return {nullptr, [self = shared_from_this(), done = std::move(done)]() mutable {
                        return self->Install(std::move(done));
                    }};
        }
        if (solvePosition)
            buffers[CpuPoses] = buffers[CpuVelocities];
        if (wakeSlots.empty()) {
            done();
            return {};
        }
        params.range = {0, static_cast<uint32_t>(wakeSlots.size()), 0, 0};
        auto packet = rhi::PrepareComputeBatch(
            owner.m_host, {{buffers[Params], 0, Bytes(&params, 1)}, {buffers[WakeSlots], 0, Bytes(wakeSlots)}},
            {Dispatch(owner.m_wake, (params.range.y + SolverLanes - 1u) / SolverLanes)});
        return {packet,
                [done = std::move(done)] {
                    done();
                    return JoltGpuControlWork{};
                },
                JoltGpuControlWork::ResumeAfter::Acceptance};
    }

    JoltGpuControlWork Prepare(const JoltGpuStorage &storage, JoltGpuStepInputs &inputs,
                               const JPH::GPU::StepParams &stepParams, uint32_t index, std::function<void()> done)
    {
        substep = index;
        substepCount = stepParams.solve.y;
        solveVelocity = stepParams.counts.w != 0;
        solvePosition = stepParams.solve.x != 0;
        velocityInitialized = false;
        params.gravityDt = stepParams.gravityDt;
        params.contact = stepParams.contact;
        inputs.SetNeedsCoupledSweep(step.NeedsContinuous(storage, *publication, stepParams.selection.x));
        buffers[GpuModels] = storage.Get(Resource::BodyModels);
        buffers[GpuStates] = storage.Get(Resource::BodyStates);
        buffers[Sleep] = storage.Get(Resource::SleepStates);
        buffers[Forces] = storage.Get(Resource::BodyForces);
        if (solveVelocity) {
            buffers[PreviousPatches] = history->previousPatches;
            buffers[Patches] = history->nextPatches;
        }
        return discovery.Discover(storage, stepParams, substep, publication, step, policy,
                                  [self = shared_from_this(), &storage, &inputs, done = std::move(done)]() mutable {
                                      return self->ReadFixedSources(
                                          storage, [self, &inputs, done = std::move(done)]() mutable {
                                              return self->PrepareWorkset(inputs, std::move(done));
                                          });
                                  });
    }

    JoltGpuControlWork PrepareWorkset(JoltGpuStepInputs &inputs, std::function<void()> done)
    {
        if (!solveVelocity && !solvePosition)
            step.Responses().clear();
        BuildWorkset(inputs);
        BeginCpuSubstep(inputs);
        inputs.SetExternalWakePending(!wakeSlots.empty());
        interleaveCpu = !native.empty() && cpu.CoupledHasLocalVelocityRows();
        if (patches.empty()) {
            history->manifolds.clear();
            cpu.AdvanceCoupledVelocity();
        } else {
            CaptureModels();
            warmStartScale = solveVelocity ? MatchWarmStarts() : 0.0f;
            if (!solveVelocity || !interleaveCpu)
                cpu.AdvanceCoupledVelocity();
            if (policy.Requested() && (!solveVelocity || !policy.impulses))
                PublishContacts();
        }
        return Install(std::move(done));
    }

    void ReadSolvedPatches(rhi::ComputeReadback &receipt)
    {
        receipt.ReadInto(patches.data(), patches.size() * sizeof(DevicePatch));
        if (solveFixed)
            for (const auto &patch : patches)
                if (patch.endpoints.z == 0 && patch.endpoints.w != JPH::GPU::JointStatusSuccess)
                    throw std::runtime_error("Cross-backend FixedJoint produced an invalid impulse or correction");
    }

    void PublishContacts()
    {
        auto &frame = step.ContactFrame(substep);
        auto &responses = step.Responses();
        for (size_t i = 0; i < order.size(); ++i) {
            if (order[i] >= responses.size())
                continue;
            const auto &patch = patches[i];
            const auto &response = responses[order[i]];
            if (!policy.Observes(response.cpu.SurfaceKey(), response.gpu.SurfaceKey(), false))
                continue;
            const auto n = XYZ(patch.normalFriction);
            const auto t =
                glm::normalize(std::abs(n.x) > std::abs(n.y) ? glm::vec3(n.z, 0, -n.x) : glm::vec3(0, n.z, -n.y));
            const auto friction =
                (t * patch.frictionLambda.x + glm::cross(n, t) * patch.frictionLambda.y) / float(response.pointCount);
            const float normal[] = {patch.normalLambda.x, patch.normalLambda.y, patch.normalLambda.z,
                                    patch.normalLambda.w};
            frame.manifolds.push_back({response.cpu, response.gpu, false, frame.points.size(), response.pointCount});
            for (uint32_t j = 0; j < response.pointCount; ++j) {
                const auto &point = response.points[j];
                frame.points.push_back(
                    {XYZ(point.positionDepth), n, point.gpuVelocity, n * normal[j] + friction, j, true});
            }
        }
    }

    uint32_t Rounds(uint32_t maximum) const
    {
        // Groups run in parallel. Bound the serial work within each group.
        return largestGroupPatchCount <= 32 && !interleaveCpu ? maximum : 1u;
    }

    void UploadWorkset(std::vector<rhi::ComputeBufferUpdate> &updates) const
    {
        updates.push_back({buffers[CpuModels], 0, Bytes(models)});
        updates.push_back({buffers[Groups], 0, Bytes(groups)});
        updates.push_back({buffers[Patches], 0, Bytes(patches)});
    }

    JoltGpuControlWork WarmVelocity(const JoltGpuStepInputs &inputs, std::function<void(uint32_t)> done)
    {
        const auto &plan = inputs.GetConstraints();
        const bool gpuHasLocalRows = plan && (plan->jointIslandCount || plan->contactIslandCount);
        // Both backends have prepared their local rows. Apply the cached cross-device response
        // before either backend solves its first local round against the warmed velocities.
        if (!patches.empty() && warmStartScale > 0.0f && (interleaveCpu || gpuHasLocalRows))
            return SubmitVelocity(0, 1, false, std::move(done));
        if (!patches.empty() && interleaveCpu)
            cpu.AdvanceCoupledVelocity();
        done(0);
        return {};
    }

    JoltGpuControlWork ExchangeVelocity(uint32_t iteration, uint32_t iterations, uint32_t maxRounds,
                                        std::function<void(uint32_t)> done)
    {
        if (patches.empty()) {
            done(maxRounds);
            return {};
        }
        const uint32_t rounds = Rounds(maxRounds);
        return SubmitVelocity(rounds, interleaveCpu ? rounds : iterations, iteration + rounds == iterations,
                              std::move(done));
    }

    JoltGpuControlWork SubmitVelocity(uint32_t rounds, uint32_t cpuRounds, bool last,
                                      std::function<void(uint32_t)> done)
    {
        const bool first = !velocityInitialized;
        const bool newVelocities = first || interleaveCpu;
        const size_t velocityCount = first ? velocities.size() : interleaveCpu ? native.size() : 0;
        if (newVelocities) {
            for (size_t i = 0; i < native.size(); ++i) {
                const auto &body = Body(native[i]);
                velocities[i] = {Pack(body.GetLinearVelocity()), Pack(body.GetAngularVelocity())};
            }
        }
        // An island with no native rows can keep the updated shadow and impulse sum on the GPU.
        params.range = {serialGroupCount, cooperativeGroupCount, rounds, newVelocities ? 1u : 0u};
        params.contact.x = first ? warmStartScale : 0.0f;
        const bool readFeedback = !native.empty() && (interleaveCpu || last);
        const bool readImpulses = last && (policy.impulses || solveFixed);
        std::vector<rhi::ComputeBufferRead> reads;
        if (readFeedback)
            reads.push_back({buffers[Impulses], 0, native.size() * sizeof(Velocity)});
        const auto contactRead = reads.size();
        if (readImpulses)
            reads.push_back({buffers[Patches], 0, patches.size() * sizeof(DevicePatch)});
        const auto resume =
            reads.empty() ? JoltGpuControlWork::ResumeAfter::Acceptance : JoltGpuControlWork::ResumeAfter::Completion;
        std::vector<rhi::ComputeBufferUpdate> updates{{buffers[Params], 0, Bytes(&params, 1)}};
        if (velocityCount)
            updates.push_back({buffers[CpuVelocities], 0, Bytes(velocities.data(), velocityCount)});
        if (first)
            UploadWorkset(updates);
        auto packet = rhi::PrepareComputeBatch(owner.m_host, std::move(updates),
                                               {Dispatch(owner.m_velocity, SolverWorkgroups())}, std::move(reads));
        velocityInitialized = true;
        return {packet,
                [self = shared_from_this(), packet, last, readFeedback, readImpulses, contactRead, rounds, cpuRounds,
                 done = std::move(done)] {
                    if (readFeedback) {
                        self->receipts.resize(self->native.size());
                        packet->Readback(0)->ReadInto(self->receipts.data(), self->receipts.size() * sizeof(Velocity));
                        self->impulses.clear();
                        for (size_t i = 0; i < self->native.size(); ++i) {
                            const auto &value = self->receipts[i];
                            self->impulses.push_back({self->native[i],
                                                      JPH::Vec3(value.linear.x, value.linear.y, value.linear.z),
                                                      JPH::Vec3(value.angular.x, value.angular.y, value.angular.z)});
                        }
                        // With no local rows the final response advances all the empty sweeps together.
                        bool velocityComplete = self->cpu.AdvanceCoupledVelocity(self->impulses);
                        for (uint32_t i = 1; i < cpuRounds; ++i)
                            velocityComplete = self->cpu.AdvanceCoupledVelocity();
                        if (velocityComplete != last)
                            throw std::logic_error(
                                "CPU and GPU coupling rounds differ from their shared velocity settings");
                    }
                    if (readImpulses) {
                        self->ReadSolvedPatches(*packet->Readback(contactRead));
                        if (self->policy.impulses)
                            self->PublishContacts();
                    }
                    if (last)
                        self->RetainImpulses();
                    done(rounds);
                    return JoltGpuControlWork{};
                },
                resume};
    }

    JoltGpuControlWork ExchangePosition(uint32_t iteration, uint32_t iterations, uint32_t maxRounds,
                                        std::function<void(uint32_t)> done)
    {
        if (iteration == 0)
            poses.resize(models.size());
        const size_t poseCount = iteration == 0 ? poses.size() : interleaveCpu ? native.size() : 0;
        for (size_t i = 0; i < poseCount; ++i) {
            const auto id = i < native.size() ? native[i] : prescribed[i - native.size()];
            const auto &body = Body(id);
            poses[i] = {Pack(body.GetCenterOfMassPosition()), Pack(body.GetRotation())};
        }
        const uint32_t rounds = Rounds(maxRounds);
        const bool last = iteration + rounds == iterations;
        const bool readFeedback = !native.empty() && (interleaveCpu || last);
        const uint32_t cpuRounds = interleaveCpu ? rounds : iterations;
        params.range = {serialGroupCount, cooperativeGroupCount, rounds, 0};
        std::vector<rhi::ComputeBufferUpdate> updates{{buffers[Params], 0, Bytes(&params, 1)}};
        if (poseCount)
            updates.push_back({buffers[CpuPoses], 0, Bytes(poses.data(), poseCount)});
        if (iteration == 0) {
            if (!solveVelocity)
                UploadWorkset(updates);
            updates.push_back({buffers[Anchors], 0, Bytes(anchors)});
        }
        std::vector<rhi::ComputeBufferRead> reads;
        if (readFeedback)
            reads.push_back({buffers[CpuPoses], 0, native.size() * sizeof(Pose)});
        const bool readJoints = last && solveFixed;
        if (readJoints)
            reads.push_back({buffers[Patches], 0, patches.size() * sizeof(DevicePatch)});
        const auto resume =
            reads.empty() ? JoltGpuControlWork::ResumeAfter::Acceptance : JoltGpuControlWork::ResumeAfter::Completion;
        auto packet = rhi::PrepareComputeBatch(owner.m_host, std::move(updates),
                                               {Dispatch(owner.m_position, SolverWorkgroups())}, std::move(reads));
        return {
            packet,
            [self = shared_from_this(), packet, last, readFeedback, readJoints, rounds, cpuRounds,
             done = std::move(done)] {
                if (readJoints)
                    self->ReadSolvedPatches(*packet->Readback(readFeedback ? 1 : 0));
                if (readFeedback) {
                    self->projected.resize(self->native.size());
                    packet->Readback(0)->ReadInto(self->projected.data(), self->projected.size() * sizeof(Pose));
                    self->corrections.clear();
                    self->corrections.reserve(self->native.size());
                    for (size_t i = 0; i < self->native.size(); ++i) {
                        const auto &origin = self->poses[i].center;
                        const auto &pose = self->projected[i];
                        self->corrections.push_back(
                            {self->native[i],
                             JPH::Vec3(pose.center.x - origin.x, pose.center.y - origin.y, pose.center.z - origin.z),
                             JPH::Quat(pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w)});
                    }
                    if (self->cpu.AdvanceCoupledPosition(self->corrections, cpuRounds) != last)
                        throw std::logic_error(
                            "CPU and GPU coupling rounds differ from their shared position settings");
                }
                done(rounds);
                return JoltGpuControlWork{};
            },
            resume};
    }
};

decltype(JoltGpuTickControl::prepareCoupledContacts)
JoltMixedConstraintSolver::Prepare(const std::shared_ptr<const JoltGpuScene::Publication> &publication,
                                   JoltGpuMixedContacts &discovery, JoltMixedContactStep &step,
                                   JoltCpuContactPublication &cpu, const JoltMixedContactStep::Policy &policy)
{
    auto job = std::make_shared<Job>(*this, discovery, step, cpu, policy, publication);
    return [job](const JoltGpuStorage &storage, JoltGpuStepInputs &inputs, const JPH::GPU::StepParams &params,
                 uint32_t substep, uint32_t iteration, JoltGpuCoupledContactStage stage, uint32_t maxRounds,
                 std::function<void(uint32_t)> done) {
        if (stage == JoltGpuCoupledContactStage::Prepare)
            return job->Prepare(storage, inputs, params, substep, [done = std::move(done)] { done(1); });
        if (stage == JoltGpuCoupledContactStage::WarmVelocity)
            return job->WarmVelocity(inputs, std::move(done));
        if (stage == JoltGpuCoupledContactStage::Integrate) {
            job->cpu.PrepareCoupledIntegration(false);
            job->cpu.AdvanceCoupledIntegration();
            done(1u);
            return JoltGpuControlWork{};
        }
        if (stage == JoltGpuCoupledContactStage::PrepareSweep) {
            job->cpu.PrepareCoupledIntegration(true);
            done(1u);
            return JoltGpuControlWork{};
        }
        if (stage == JoltGpuCoupledContactStage::Sweep)
            return job->owner.m_continuous->Prepare(storage, inputs, params, substep, job->publication, job->discovery,
                                                    job->step, job->cpu, job->policy,
                                                    [done = std::move(done)] { done(1u); });
        if (stage == JoltGpuCoupledContactStage::Position)
            return job->ExchangePosition(iteration, params.solve.x, maxRounds, std::move(done));
        return job->ExchangeVelocity(iteration, params.counts.w, maxRounds, std::move(done));
    };
}
} // namespace infernux
