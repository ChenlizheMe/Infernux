#include "JoltMixedConstraintSolver.h"
#include "JoltCpuContactPublication.h"
#include "JoltCpuScene.h"
#include "JoltGpuProgramSources.h"

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

struct Parameters
{
    UInt4 range;
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
        if (entry == target.entries.end() || entry->workgroupSize != std::array<uint32_t, 3>{64, 1, 1})
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
    std::vector<JPH::PhysicsSystem::CoupledBodyImpulse> impulses;
    std::vector<JPH::PhysicsSystem::CoupledBodyCorrection> corrections;
    uint32_t substep = 0, substepCount = 0;
    bool solveVelocity = false;
    bool solvePosition = false;
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

    void BuildWorkset(JoltGpuStepInputs &inputs)
    {
        auto &responses = step.Responses();
        std::sort(responses.begin(), responses.end(), [](const Patch &a, const Patch &b) { return Key(a) < Key(b); });
        order.resize(responses.size());
        std::iota(order.begin(), order.end(), size_t(0));
        auto parents = order;
        const auto root = [&parents](size_t i) {
            while (parents[i] != i) {
                parents[i] = parents[parents[i]];
                i = parents[i];
            }
            return i;
        };
        std::unordered_map<uint32_t, size_t> cpuLinks, gpuLinks;
        const auto link = [&](auto &links, uint32_t body, size_t i) {
            const auto [first, added] = links.try_emplace(body, i);
            if (!added)
                parents[root(i)] = root(first->second);
        };
        std::vector<uint32_t> gpuContactBodies;
        gpuContactBodies.reserve(responses.size());
        for (size_t i = 0; i < responses.size(); ++i) {
            const auto &response = responses[i];
            if (response.cpuDynamic)
                link(cpuLinks, response.cpu.body, i);
            if (response.gpuDynamic)
                link(gpuLinks, response.gpuSlot, i);
            gpuContactBodies.push_back(response.gpuSlot);
        }
        for (size_t i = 0; i < parents.size(); ++i)
            parents[i] = root(i);
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return parents[a] < parents[b]; });
        std::sort(gpuContactBodies.begin(), gpuContactBodies.end());
        gpuContactBodies.erase(std::unique(gpuContactBodies.begin(), gpuContactBodies.end()), gpuContactBodies.end());
        inputs.SetExternalContactBodies(std::move(gpuContactBodies));

        const auto dynamicCount = cpuLinks.size();
        native.clear();
        prescribed.clear();
        groups.clear();
        patches.clear();
        wakeSlots.clear();
        patches.reserve(order.size());
        cpuLinks.clear();
        size_t previous = order.size();
        for (const size_t index : order) {
            if (parents[index] != previous) {
                previous = parents[index];
                groups.push_back({static_cast<uint32_t>(patches.size()), 0, static_cast<uint32_t>(native.size()), 0});
            }
            const auto &response = responses[index];
            if (response.wakeGpu)
                wakeSlots.push_back(response.gpuSlot);
            auto &group = groups.back();
            const size_t slot = response.cpuDynamic ? native.size() : dynamicCount + prescribed.size();
            const auto [mapped, added] = cpuLinks.try_emplace(response.cpu.body, slot);
            if (added) {
                if (response.cpuDynamic) {
                    native.push_back(response.native);
                    ++group.w;
                } else {
                    prescribed.push_back(response.native);
                }
            }
            DevicePatch value{};
            value.endpoints = {static_cast<uint32_t>(mapped->second), response.gpuSlot, response.pointCount, 0};
            value.normalFriction = response.normalFriction;
            patches.push_back(value);
            ++group.y;
        }
        std::sort(wakeSlots.begin(), wakeSlots.end());
        wakeSlots.erase(std::unique(wakeSlots.begin(), wakeSlots.end()), wakeSlots.end());
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
        history->manifolds.resize(order.size());
        for (uint32_t i = 0; i < order.size(); ++i) {
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

    rhi::ComputeDispatchDesc Dispatch(const Program &program) const
    {
        rhi::ComputeDispatchDesc dispatch;
        dispatch.kernel = program.kernel;
        dispatch.groupCountX = (params.range.y + 63u) / 64u;
        for (const auto port : program.ports) {
            dispatch.buffers.push_back(buffers[port]);
            const bool modifies =
                port == GpuStates || port == CpuVelocities || port == CpuPoses || port == Patches || port == Sleep;
            dispatch.bufferAccesses.push_back(modifies           ? rhi::ComputeBufferAccess::ReadWrite
                                              : port == Impulses ? rhi::ComputeBufferAccess::Write
                                                                 : rhi::ComputeBufferAccess::Read);
        }
        return dispatch;
    }

    JoltGpuControlWork Install(std::function<void()> done)
    {
        const auto velocityBytes = models.size() * sizeof(Velocity);
        if (!Allocate(Params, sizeof(params)) || !Allocate(CpuModels, models.size() * sizeof(CpuBoundary)) ||
            !Allocate(CpuVelocities, velocityBytes) || !Allocate(Groups, groups.size() * sizeof(UInt4)) ||
            !Allocate(Patches, patches.size() * sizeof(DevicePatch)) ||
            (solveVelocity && !Allocate(PreviousPatches, sizeof(DevicePatch))) ||
            (solveVelocity && !Allocate(Impulses, std::max(size_t{1}, native.size()) * sizeof(Velocity))) ||
            (solvePosition && !Allocate(Anchors, anchors.size() * sizeof(PositionAnchors))) ||
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
            {Dispatch(owner.m_wake)});
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
        params.gravityDt = stepParams.gravityDt;
        params.contact = stepParams.contact;
        buffers[GpuModels] = storage.Get(Resource::BodyModels);
        buffers[GpuStates] = storage.Get(Resource::BodyStates);
        buffers[Sleep] = storage.Get(Resource::SleepStates);
        buffers[Forces] = storage.Get(Resource::BodyForces);
        if (solveVelocity) {
            buffers[PreviousPatches] = history->previousPatches;
            buffers[Patches] = history->nextPatches;
        }
        return discovery.Discover(
            storage, stepParams, substep, publication, step, policy,
            [self = shared_from_this(), &inputs, done = std::move(done)]() mutable {
                if (!self->solveVelocity && !self->solvePosition)
                    self->step.Responses().clear();
                self->BuildWorkset(inputs);
                self->cpu.BeginCoupledSubstep(self->substep, self->native);
                if (self->patches.empty()) {
                    self->history->manifolds.clear();
                    self->cpu.AdvanceCoupledVelocity();
                    done();
                    return JoltGpuControlWork{};
                }
                self->CaptureModels();
                self->warmStartScale = self->solveVelocity ? self->MatchWarmStarts() : 0.0f;
                self->cpu.AdvanceCoupledVelocity(); // First CPU local round, before the first cross-domain response.
                if (self->policy.Requested() && (!self->solveVelocity || !self->policy.impulses))
                    self->PublishContacts();
                return self->Install(std::move(done));
            });
    }

    void PublishContacts()
    {
        auto &frame = step.ContactFrame(substep);
        auto &responses = step.Responses();
        for (size_t i = 0; i < order.size(); ++i) {
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
        // Keep the packet bounded and preserve the native order wherever it can influence a cross-domain row.
        return patches.size() <= 32 && (native.empty() || !cpu.CoupledHasLocalVelocityRows()) ? maximum : 1u;
    }

    void UploadWorkset(std::vector<rhi::ComputeBufferUpdate> &updates) const
    {
        updates.push_back({buffers[CpuModels], 0, Bytes(models)});
        updates.push_back({buffers[Groups], 0, Bytes(groups)});
        updates.push_back({buffers[Patches], 0, Bytes(patches)});
    }

    JoltGpuControlWork ExchangeVelocity(uint32_t iteration, uint32_t iterations, uint32_t maxRounds,
                                        std::function<void(uint32_t)> done)
    {
        if (patches.empty()) {
            done(maxRounds);
            return {};
        }
        for (size_t i = 0; i < native.size(); ++i) {
            const auto &body = Body(native[i]);
            velocities[i] = {Pack(body.GetLinearVelocity()), Pack(body.GetAngularVelocity())};
        }
        const uint32_t rounds = Rounds(maxRounds);
        params.range = {0, static_cast<uint32_t>(groups.size()), rounds, 0};
        params.contact.x = iteration == 0 ? warmStartScale : 0.0f;
        const bool last = iteration + rounds == iterations;
        const bool readImpulses = last && policy.impulses;
        std::vector<rhi::ComputeBufferRead> reads;
        if (!native.empty())
            reads.push_back({buffers[Impulses], 0, native.size() * sizeof(Velocity)});
        const auto contactRead = reads.size();
        if (readImpulses)
            reads.push_back({buffers[Patches], 0, patches.size() * sizeof(DevicePatch)});
        const auto resume =
            reads.empty() ? JoltGpuControlWork::ResumeAfter::Acceptance : JoltGpuControlWork::ResumeAfter::Completion;
        std::vector<rhi::ComputeBufferUpdate> updates{{buffers[Params], 0, Bytes(&params, 1)}};
        const size_t velocityCount = iteration == 0 ? velocities.size() : native.size();
        if (velocityCount)
            updates.push_back({buffers[CpuVelocities], 0, Bytes(velocities.data(), velocityCount)});
        if (iteration == 0)
            UploadWorkset(updates);
        auto packet =
            rhi::PrepareComputeBatch(owner.m_host, std::move(updates), {Dispatch(owner.m_velocity)}, std::move(reads));
        return {packet,
                [self = shared_from_this(), packet, last, readImpulses, contactRead, rounds, done = std::move(done)] {
                    if (!self->native.empty()) {
                        self->receipts.resize(self->native.size());
                        packet->Readback(0)->ReadInto(self->receipts.data(), self->receipts.size() * sizeof(Velocity));
                        self->impulses.clear();
                        for (size_t i = 0; i < self->native.size(); ++i) {
                            const auto &value = self->receipts[i];
                            self->impulses.push_back({self->native[i],
                                                      JPH::Vec3(value.linear.x, value.linear.y, value.linear.z),
                                                      JPH::Vec3(value.angular.x, value.angular.y, value.angular.z)});
                        }
                        // Multiple rounds have no intervening CPU local rows.
                        bool integrated = self->cpu.AdvanceCoupledVelocity(self->impulses);
                        for (uint32_t i = 1; i < rounds; ++i)
                            integrated = self->cpu.AdvanceCoupledVelocity();
                        if (integrated != last)
                            throw std::logic_error(
                                "CPU and GPU coupling rounds differ from their shared velocity settings");
                    }
                    if (readImpulses) {
                        packet->Readback(contactRead)
                            ->ReadInto(self->patches.data(), self->patches.size() * sizeof(DevicePatch));
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
        const size_t poseCount = iteration == 0 ? poses.size() : native.size();
        for (size_t i = 0; i < poseCount; ++i) {
            const auto id = i < native.size() ? native[i] : prescribed[i - native.size()];
            const auto &body = Body(id);
            poses[i] = {Pack(body.GetCenterOfMassPosition()), Pack(body.GetRotation())};
        }
        const uint32_t rounds = Rounds(maxRounds);
        const bool last = iteration + rounds == iterations;
        params.range = {0, static_cast<uint32_t>(groups.size()), rounds, 0};
        std::vector<rhi::ComputeBufferUpdate> updates{{buffers[Params], 0, Bytes(&params, 1)}};
        if (poseCount)
            updates.push_back({buffers[CpuPoses], 0, Bytes(poses.data(), poseCount)});
        if (iteration == 0) {
            if (!solveVelocity)
                UploadWorkset(updates);
            updates.push_back({buffers[Anchors], 0, Bytes(anchors)});
        }
        std::vector<rhi::ComputeBufferRead> reads;
        if (!native.empty())
            reads.push_back({buffers[CpuPoses], 0, native.size() * sizeof(Pose)});
        const auto resume =
            reads.empty() ? JoltGpuControlWork::ResumeAfter::Acceptance : JoltGpuControlWork::ResumeAfter::Completion;
        auto packet =
            rhi::PrepareComputeBatch(owner.m_host, std::move(updates), {Dispatch(owner.m_position)}, std::move(reads));
        return {
            packet,
            [self = shared_from_this(), packet, last, rounds, done = std::move(done)] {
                if (!self->native.empty()) {
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
                    if (self->cpu.AdvanceCoupledPosition(self->corrections, rounds) != last)
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
        if (stage == JoltGpuCoupledContactStage::Position)
            return job->ExchangePosition(iteration, params.solve.x, maxRounds, std::move(done));
        return job->ExchangeVelocity(iteration, params.counts.w, maxRounds, std::move(done));
    };
}
} // namespace infernux
