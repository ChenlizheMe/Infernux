#include "JoltMixedConstraintSolver.h"
#include "JoltCpuContactPublication.h"
#include "JoltCpuScene.h"
#include "JoltGpuProgramSources.h"

#include <Jolt/Physics/Body/MotionProperties.h>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace infernux
{
namespace
{
using namespace JPH::GPU;
using Contact = PhysicsContactPublication;
using Patch = JoltMixedContactStep::ResponsePatch;

struct Parameters
{
    UInt4 range;
    Float4 gravityDt;
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
struct DevicePatch
{
    UInt4 endpoints;
    Float4 normalFriction;
    std::array<Float4, MaxContactPointsPerPair> points;
    Float4 normalLambda, frictionLambda;
};
static_assert(sizeof(CpuBoundary) == 112);
static_assert(sizeof(Velocity) == 32);
static_assert(sizeof(DevicePatch) == 128);

glm::vec3 XYZ(const Float4 &value)
{
    return {value.x, value.y, value.z};
}

template <typename V> Float4 Pack(V value)
{
    return {float(value.GetX()), float(value.GetY()), float(value.GetZ()), 0};
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

bool PatchOrder(const Patch &a, const Patch &b)
{
    return std::tuple(a.cpu.Key(), a.gpu.Key()) < std::tuple(b.cpu.Key(), b.gpu.Key());
}
} // namespace

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
    load(m_exchange, "exchange_cross_domain_velocity");
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
        PortCount
    };

    JoltMixedConstraintSolver &owner;
    JoltGpuMixedContacts &discovery;
    JoltMixedContactStep &step;
    JoltCpuContactPublication &cpu;
    const JoltMixedContactStep::Policy &policy;
    std::shared_ptr<const JoltGpuScene::Publication> publication;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, PortCount> buffers;
    std::vector<CpuBoundary> models;
    std::vector<Velocity> velocities, receipts;
    std::vector<DevicePatch> patches;
    std::vector<UInt4> groups;
    std::vector<JPH::BodyID> native;
    std::vector<uint32_t> wakeSlots;
    std::vector<size_t> order;
    std::vector<JPH::PhysicsSystem::CoupledBodyImpulse> impulses;
    uint32_t substep = 0;
    Parameters params{};

    Job(JoltMixedConstraintSolver &owner, JoltGpuMixedContacts &discovery, JoltMixedContactStep &step,
        JoltCpuContactPublication &cpu, const JoltMixedContactStep::Policy &policy,
        std::shared_ptr<const JoltGpuScene::Publication> publication)
        : owner(owner), discovery(discovery), step(step), cpu(cpu), policy(policy), publication(std::move(publication))
    {
    }

    const JPH::Body &Body(JPH::BodyID id) const
    {
        return *step.Scene().System().GetBodyLockInterfaceNoLock().TryGetBody(id);
    }

    void BuildWorkset()
    {
        auto &responses = step.Responses();
        std::sort(responses.begin(), responses.end(), PatchOrder);
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
        for (size_t i = 0; i < responses.size(); ++i) {
            const auto &response = responses[i];
            const auto [first, newCpu] = cpuLinks.try_emplace(response.cpu.body, i);
            const auto [second, newGpu] = gpuLinks.try_emplace(response.gpuSlot, i);
            if (!newCpu)
                parents[root(i)] = root(first->second);
            if (!newGpu)
                parents[root(i)] = root(second->second);
        }
        for (size_t i = 0; i < parents.size(); ++i)
            parents[i] = root(i);
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return parents[a] < parents[b]; });

        native.clear();
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
            const auto [mapped, added] = cpuLinks.try_emplace(response.cpu.body, native.size());
            if (added) {
                native.push_back(response.native);
                ++group.w;
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

    void CaptureModels()
    {
        models.resize(native.size());
        for (size_t i = 0; i < native.size(); ++i) {
            const auto &body = Body(native[i]);
            const auto *motion = body.GetMotionProperties();
            const auto locks = ~static_cast<uint32_t>(motion->GetAllowedDOFs());
            auto &boundary = models[i];
            boundary.model = {};
            boundary.model.meta.z = MotionDynamic;
            boundary.model.mass = {motion->GetInverseMass(), float(locks & 7u), float((locks >> 3) & 7u), 0};
            boundary.model.invInertiaDiagonal = Pack(motion->GetInverseInertiaDiagonal());
            const auto inertia = motion->GetInertiaRotation();
            boundary.model.inertiaRotation = {inertia.GetX(), inertia.GetY(), inertia.GetZ(), inertia.GetW()};
            boundary.center = Pack(body.GetCenterOfMassPosition());
            const auto rotation = body.GetRotation();
            boundary.rotation = {rotation.GetX(), rotation.GetY(), rotation.GetZ(), rotation.GetW()};
        }
        const float dt = params.gravityDt.w;
        const float minImpact = step.Scene().System().GetPhysicsSettings().mMinVelocityForRestitution;
        auto &responses = step.Responses();
        for (size_t i = 0; i < order.size(); ++i) {
            auto &response = responses[order[i]];
            auto &patch = patches[i];
            const auto free = cpu.CoupledFreeVelocity(response.native);
            const auto v = XYZ(Pack(free.mLinear));
            const auto w = XYZ(Pack(free.mAngular));
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
            const bool modifies = port == GpuStates || port == CpuVelocities || port == Patches || port == Sleep;
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
            !Allocate(Patches, patches.size() * sizeof(DevicePatch)) || !Allocate(Impulses, velocityBytes) ||
            (!wakeSlots.empty() && !Allocate(WakeSlots, wakeSlots.size() * sizeof(uint32_t)))) {
            return {nullptr, [self = shared_from_this(), done = std::move(done)]() mutable {
                        return self->Install(std::move(done));
                    }};
        }
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

    JoltGpuControlWork Prepare(const JoltGpuStorage &storage, const JPH::GPU::StepParams &stepParams, uint32_t index,
                               std::function<void()> done)
    {
        substep = index;
        params.gravityDt = stepParams.gravityDt;
        buffers[GpuModels] = storage.Get(Resource::BodyModels);
        buffers[GpuStates] = storage.Get(Resource::BodyStates);
        buffers[Sleep] = storage.Get(Resource::SleepStates);
        buffers[Forces] = storage.Get(Resource::BodyForces);
        return discovery.Discover(
            storage, stepParams, substep, publication, step, policy,
            [self = shared_from_this(), solve = stepParams.counts.w, done = std::move(done)]() mutable {
                if (!solve)
                    self->step.Responses().clear();
                self->BuildWorkset();
                self->cpu.BeginCoupledSubstep(self->substep, self->native);
                if (self->native.empty()) {
                    self->cpu.AdvanceCoupled();
                    done();
                    return JoltGpuControlWork{};
                }
                self->CaptureModels();
                self->cpu.AdvanceCoupled(); // First CPU local round, before the first cross-domain response.
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

    JoltGpuControlWork Exchange(uint32_t iteration, uint32_t iterations, std::function<void()> done)
    {
        if (native.empty()) {
            done();
            return {};
        }
        velocities.clear();
        velocities.reserve(native.size());
        for (const auto id : native) {
            const auto &body = Body(id);
            velocities.push_back({Pack(body.GetLinearVelocity()), Pack(body.GetAngularVelocity())});
        }
        params.range = {0, static_cast<uint32_t>(groups.size()), 0, 0};
        const bool last = iteration + 1 == iterations;
        const bool readContacts = last && policy.Requested();
        std::vector<rhi::ComputeBufferRead> reads{{buffers[Impulses], 0, velocities.size() * sizeof(Velocity)}};
        if (readContacts)
            reads.push_back({buffers[Patches], 0, patches.size() * sizeof(DevicePatch)});
        std::vector<rhi::ComputeBufferUpdate> updates{{buffers[Params], 0, Bytes(&params, 1)},
                                                      {buffers[CpuVelocities], 0, Bytes(velocities)}};
        if (iteration == 0) {
            updates.push_back({buffers[CpuModels], 0, Bytes(models)});
            updates.push_back({buffers[Groups], 0, Bytes(groups)});
            updates.push_back({buffers[Patches], 0, Bytes(patches)});
        }
        auto packet =
            rhi::PrepareComputeBatch(owner.m_host, std::move(updates), {Dispatch(owner.m_exchange)}, std::move(reads));
        return {packet, [self = shared_from_this(), packet, last, readContacts, done = std::move(done)] {
                    self->receipts.resize(self->native.size());
                    packet->Readback(0)->ReadInto(self->receipts.data(), self->receipts.size() * sizeof(Velocity));
                    self->impulses.clear();
                    for (size_t i = 0; i < self->native.size(); ++i) {
                        const auto &value = self->receipts[i];
                        self->impulses.push_back({self->native[i],
                                                  JPH::Vec3(value.linear.x, value.linear.y, value.linear.z),
                                                  JPH::Vec3(value.angular.x, value.angular.y, value.angular.z)});
                    }
                    if (self->cpu.AdvanceCoupled(self->impulses) != last)
                        throw std::logic_error(
                            "CPU and GPU coupling rounds differ from their shared velocity settings");
                    if (readContacts) {
                        packet->Readback(1)->ReadInto(self->patches.data(), self->patches.size() * sizeof(DevicePatch));
                        self->PublishContacts();
                    }
                    done();
                    return JoltGpuControlWork{};
                }};
    }
};

decltype(JoltGpuTickControl::prepareCoupledContacts)
JoltMixedConstraintSolver::Prepare(const std::shared_ptr<const JoltGpuScene::Publication> &publication,
                                   JoltGpuMixedContacts &discovery, JoltMixedContactStep &step,
                                   JoltCpuContactPublication &cpu, const JoltMixedContactStep::Policy &policy)
{
    auto job = std::make_shared<Job>(*this, discovery, step, cpu, policy, publication);
    return [job](const JoltGpuStorage &storage, const JoltGpuStepInputs &, const JPH::GPU::StepParams &params,
                 uint32_t substep, uint32_t iteration, JoltGpuCoupledContactStage stage, std::function<void()> done) {
        return stage == JoltGpuCoupledContactStage::Prepare
                   ? job->Prepare(storage, params, substep, std::move(done))
                   : job->Exchange(iteration, params.counts.w, std::move(done));
    };
}
} // namespace infernux
