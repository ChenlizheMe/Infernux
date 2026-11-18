#include "JoltMixedContinuousContacts.h"
#include "JoltCpuContactPublication.h"
#include "JoltCpuScene.h"
#include "JoltGpuProgramSources.h"
#include <function/scene/Rigidbody.h>

#include <Jolt/GPU/ContinuousTypes.h>
#include <Jolt/Physics/Collision/ManifoldBetweenTwoFaces.h>
#include <Jolt/Physics/Collision/ShapeCast.h>
#include <Jolt/Physics/Collision/TransformedShape.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace infernux
{
namespace
{
using namespace JPH::GPU;
enum Entry : uint32_t
{
    Gather,
    RestrictMotion,
    RestrictReports,
    Apply
};
constexpr uint32_t PortCount = 24;
constexpr uint64_t TransferPageBytes = 32u * 1024u;
struct Parameters
{
    UInt4 range;
    Float4 gravityDt, contact;
};
template <typename T> std::vector<uint8_t> Bytes(const std::vector<T> &values)
{
    std::vector<uint8_t> bytes(values.size() * sizeof(T));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}
uint32_t TimeBits(float time)
{
    uint32_t bits;
    std::memcpy(&bits, &time, sizeof(bits));
    return bits;
}
float ImpactTime(uint32_t bits)
{
    float time;
    std::memcpy(&time, &bits, sizeof(time));
    return time;
}
template <typename V> Float4 Pack(const V &value)
{
    return {float(value.GetX()), float(value.GetY()), float(value.GetZ()), 0};
}
Float4 Pack(JPH::QuatArg value)
{
    return {value.GetX(), value.GetY(), value.GetZ(), value.GetW()};
}
JPH::Vec3 Vector(const Float4 &value)
{
    return {value.x, value.y, value.z};
}
glm::vec3 XYZ(const Float4 &value)
{
    return {value.x, value.y, value.z};
}
bool Overlaps(glm::vec3 lo, glm::vec3 hi, UInt4 lower, UInt4 upper)
{
    return glm::all(glm::lessThanEqual(lo, glm::uintBitsToFloat(glm::uvec3(upper.x, upper.y, upper.z)))) &&
           glm::all(glm::lessThanEqual(glm::uintBitsToFloat(glm::uvec3(lower.x, lower.y, lower.z)), hi));
}
template <typename Accept> class CastCollector final : public JPH::CastShapeCollector
{
  public:
    explicit CastCollector(Accept accept) : m_accept(std::move(accept))
    {
    }
    void AddHit(const JPH::ShapeCastResult &hit) override
    {
        if (m_accept(hit))
            UpdateEarlyOutFraction(hit.mFraction);
    }

  private:
    Accept m_accept;
};
} // namespace

JoltMixedContinuousContacts::JoltMixedContinuousContacts(rhi::ComputeHost &host, const ShaderTargetCode &target)
    : m_host(host)
{
    const std::array names{"gather_continuous_boundaries", "restrict_continuous_motion", "restrict_continuous_reports",
                           "apply_cross_domain_sweep"};
    for (size_t i = 0; i < names.size(); ++i) {
        const auto entry = std::find_if(target.entries.begin(), target.entries.end(),
                                        [&](const auto &code) { return code.name == names[i]; });
        if (entry == target.entries.end() || entry->workgroupSize != std::array<uint32_t, 3>{64, 1, 1})
            throw std::invalid_argument("Mixed continuous entry has a different workgroup layout");
        std::vector<rhi::ComputeBufferBinding> bindings;
        for (const auto &resource : target.resources) {
            const std::array<uint32_t, 2> key{resource.set, resource.binding.binding};
            if (std::find(entry->bindings.begin(), entry->bindings.end(), key) == entry->bindings.end())
                continue;
            if (resource.set != 0 || resource.binding.binding >= PortCount)
                throw std::invalid_argument("Mixed continuous entry uses an unknown port");
            m_programs[i].ports.push_back(resource.binding.binding);
            bindings.push_back({resource.binding.binding, resource.binding.type, resource.binding.readOnlyStorage});
        }
        m_programs[i].kernel = PrepareJoltGpuKernel(m_host, target.target, *entry, std::move(bindings));
    }
}

struct JoltMixedContinuousContacts::Job : std::enable_shared_from_this<Job>
{
    using Boundary = JoltGpuControlWork::ResumeAfter;
    JoltMixedContinuousContacts &owner;
    const JoltGpuStorage &storage;
    JoltGpuStepInputs &inputs;
    StepParams params;
    uint32_t substep;
    std::shared_ptr<const JoltGpuScene::Publication> publication;
    JoltGpuMixedContacts &discovery;
    JoltMixedContactStep &step;
    JoltCpuContactPublication &cpu;
    const JoltMixedContactStep::Policy &policy;
    std::function<void()> completed;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, PortCount> buffers;
    std::vector<UInt4> leaves;
    std::vector<uint32_t> slots;
    std::vector<ContinuousSample> samples;
    struct Hit
    {
        ContinuousMixedHit value;
        PhysicsContactPublication::Endpoint cpu, gpu;
        JPH::BodyID native;
        uint32_t sample;
    };
    std::vector<Hit> hits, selected;
    std::vector<ContinuousCpuBody> cpuBodies;
    std::vector<ContinuousMixedHit> deviceHits;
    std::vector<ContinuousRootPolicy> policies;
    std::vector<JPH::PhysicsSystem::CoupledContinuousUpdate> updates;
    std::unordered_map<uint32_t, uint32_t> cpuRows;
    std::vector<rhi::ComputeBufferUpdate> uploads;
    using AfterUpload = JoltGpuControlWork (Job::*)(std::vector<rhi::ComputeBufferUpdate>);
    size_t uploadIndex = 0;
    uint64_t uploadOffset = 0;
    uint32_t gatherFirst = 0, restrictFirst = 0, reportFirst = 0, applyFirst = 0;
    uint32_t leafCursor = 0, nodeCursor = 0, proxyCursor = 0, proxyEnd = 0;
    bool leafReady = false;
    glm::vec3 leafMin{}, leafMax{};
    uint32_t leafSample = 0;

    Job(JoltMixedContinuousContacts &owner, const JoltGpuStorage &storage, JoltGpuStepInputs &inputs,
        const StepParams &params, uint32_t substep, const std::shared_ptr<const JoltGpuScene::Publication> &publication,
        JoltGpuMixedContacts &discovery, JoltMixedContactStep &step, JoltCpuContactPublication &cpu,
        const JoltMixedContactStep::Policy &policy, std::function<void()> completed)
        : owner(owner), storage(storage), inputs(inputs), params(params), substep(substep), publication(publication),
          discovery(discovery), step(step), cpu(cpu), policy(policy), completed(std::move(completed))
    {
        buffers[1] = inputs.GetCaptured(Resource::BodyModels);
        buffers[15] = inputs.GetSweepBodies();
        buffers[16] = inputs.GetSweepChoices();
        buffers[22] = inputs.GetSweepContacts()->patches;
        buffers[23] = inputs.GetSweepContacts()->contacts;
    }
    const JPH::Body &Body(JPH::BodyID id) const
    {
        return *step.Scene().System().GetBodyLockInterfaceNoLock().TryGetBody(id);
    }
    bool Allocate(uint32_t port, uint64_t bytes)
    {
        if (buffers[port])
            return true;
        buffers[port] = rhi::ComputeBuffer::TryCreatePrivate(
            owner.m_host, {bytes / 4u, rhi::ComputeScalarType::UInt32, 1u, rhi::BufferUsageFlags::None});
        return bool(buffers[port]);
    }
    rhi::ComputeBufferUpdate Uniform(UInt4 range)
    {
        const Parameters value{range, params.gravityDt, params.contact};
        std::vector<uint8_t> bytes(sizeof(value));
        std::memcpy(bytes.data(), &value, sizeof(value));
        return {buffers[0], 0u, std::move(bytes)};
    }
    rhi::ComputeDispatchDesc Dispatch(Entry entry, uint32_t count, uint32_t first = 0u)
    {
        rhi::ComputeDispatchDesc dispatch;
        const auto &program = owner.m_programs[entry];
        dispatch.kernel = program.kernel;
        dispatch.groupCountX = (count + 63u) / 64u;
        rhi::ComputeDispatchProof proof;
        for (auto port : program.ports) {
            const auto &buffer = buffers[port];
            const uint32_t index = static_cast<uint32_t>(dispatch.buffers.size());
            dispatch.buffers.push_back(buffer);
            const bool output = (entry == Gather && port == 17u) || (entry == Entry::Apply && port == 21u);
            const bool mutablePort = (entry == RestrictMotion || entry == Entry::Apply) && port == 15u ||
                                     entry == RestrictReports && port == 22u;
            dispatch.bufferAccesses.push_back(output        ? rhi::ComputeBufferAccess::Write
                                              : mutablePort ? rhi::ComputeBufferAccess::ReadWrite
                                                            : rhi::ComputeBufferAccess::Read);
            if (output) {
                const uint64_t stride = entry == Gather ? sizeof(ContinuousSample) : sizeof(ContinuousReceipt);
                proof.writes.push_back({index, uint64_t(first) * stride, uint64_t(count) * stride});
            } else
                proof.reads.push_back({index, 0u, buffer->GetByteSize()});
        }
        dispatch.proof = std::move(proof);
        return dispatch;
    }
    JoltGpuControlWork Upload(AfterUpload next)
    {
        std::vector<rhi::ComputeBufferUpdate> pending;
        uint64_t remaining = TransferPageBytes;
        while (uploadIndex < uploads.size() && remaining > 0) {
            auto &source = uploads[uploadIndex];
            const uint64_t count = std::min(uint64_t(source.data.size()) - uploadOffset, remaining);
            if (uploadOffset == 0 && count == source.data.size()) {
                pending.push_back(std::move(source));
                ++uploadIndex;
            } else {
                const auto first = source.data.begin() + uploadOffset;
                pending.push_back({source.buffer, source.offset + uploadOffset, {first, first + count}});
                uploadOffset += count;
                if (uploadOffset == source.data.size()) {
                    ++uploadIndex;
                    uploadOffset = 0;
                }
            }
            remaining -= count;
        }
        if (uploadIndex == uploads.size()) {
            uploads.clear();
            uploadIndex = 0;
            return (this->*next)(std::move(pending));
        }
        auto packet = rhi::PrepareComputeBatch(owner.m_host, std::move(pending), {});
        return {packet, [self = shared_from_this(), next] { return self->Upload(next); }, Boundary::Acceptance};
    }
    JoltGpuControlWork Begin()
    {
        return discovery.DiscoverContinuous(
            storage, params, publication, step, cpu, policy,
            [self = shared_from_this()](UInt4 source) { self->leaves.push_back(source); },
            [self = shared_from_this()] { return self->PrepareSamples(); });
    }
    JoltGpuControlWork PrepareSamples()
    {
        if (leaves.empty())
            return Finish();
        if (slots.empty()) {
            std::sort(leaves.begin(), leaves.end(), [](auto a, auto b) { return a.x < b.x; });
            slots.reserve(leaves.size());
            for (const auto source : leaves)
                slots.push_back(publication->directory->leaves[source.x].bodySlot);
            std::sort(slots.begin(), slots.end());
            slots.erase(std::unique(slots.begin(), slots.end()), slots.end());
            samples.resize(slots.size());
        }
        if (!Allocate(0, sizeof(Parameters)) || !Allocate(14, slots.size() * sizeof(uint32_t)) ||
            !Allocate(17, slots.size() * sizeof(ContinuousSample)))
            return {nullptr, [self = shared_from_this()] { return self->PrepareSamples(); }};
        uploads = {{buffers[14], 0u, Bytes(slots)}};
        return Upload(&Job::GatherSamples);
    }
    JoltGpuControlWork GatherSamples(std::vector<rhi::ComputeBufferUpdate> pending = {})
    {
        if (gatherFirst == slots.size())
            return CastNext();
        const uint32_t count = std::min(uint32_t(slots.size()) - gatherFirst,
                                        uint32_t((TransferPageBytes - sizeof(Parameters)) / sizeof(ContinuousSample)));
        const uint64_t offset = uint64_t(gatherFirst) * sizeof(ContinuousSample);
        pending.push_back(Uniform({gatherFirst, count, params.selection.y, params.selectedCounts.x}));
        auto packet = rhi::PrepareComputeBatch(owner.m_host, std::move(pending), {Dispatch(Gather, count, gatherFirst)},
                                               {{buffers[17], offset, uint64_t(count) * sizeof(ContinuousSample)}});
        return {packet, [self = shared_from_this(), packet, count] {
                    packet->Readback(0)->ReadInto(self->samples.data() + self->gatherFirst,
                                                  count * sizeof(ContinuousSample));
                    self->gatherFirst += count;
                    return self->GatherSamples();
                }};
    }
    JoltGpuControlWork CastNext();
    void CastPair(uint32_t proxy);
    void Arbitrate();
    uint32_t CpuRow(JPH::BodyID);
    JoltGpuControlWork PrepareDecisions();
    JoltGpuControlWork Restrict(std::vector<rhi::ComputeBufferUpdate> pending = {});
    JoltGpuControlWork RestrictLocalReports();
    JoltGpuControlWork Apply(std::vector<rhi::ComputeBufferUpdate> pending = {});
    JoltGpuControlWork Finish()
    {
        cpu.AdvanceCoupledIntegration(updates);
        auto done = std::move(completed);
        done();
        return {};
    }
};

uint32_t JoltMixedContinuousContacts::Job::CpuRow(JPH::BodyID id)
{
    const auto found = cpuRows.find(id.GetIndexAndSequenceNumber());
    if (found != cpuRows.end())
        return found->second;
    const auto &body = Body(id);
    ContinuousCpuBody value{};
    value.model.meta.z = body.IsDynamic() ? MotionDynamic : MotionKinematic;
    value.state = {Pack(body.GetCenterOfMassPosition()), Pack(body.GetRotation()), Pack(body.GetLinearVelocity()),
                   Pack(body.GetAngularVelocity())};
    value.displacement = Pack(cpu.CoupledDisplacement(id));
    if (body.IsDynamic()) {
        const auto *motion = body.GetMotionProperties();
        const auto locks = ~static_cast<uint32_t>(motion->GetAllowedDOFs());
        value.model.mass = {motion->GetInverseMass(), float(locks & 7u), float((locks >> 3) & 7u), 0};
        value.model.invInertiaDiagonal = Pack(motion->GetInverseInertiaDiagonal());
        value.model.inertiaRotation = Pack(motion->GetInertiaRotation());
        value.model.damping = {0, 0, motion->GetMaxLinearVelocity(), motion->GetMaxAngularVelocity()};
    }
    const auto row = static_cast<uint32_t>(cpuBodies.size());
    cpuBodies.push_back(value);
    cpuRows.emplace(id.GetIndexAndSequenceNumber(), row);
    return row;
}

void JoltMixedContinuousContacts::Job::CastPair(uint32_t index)
{
    const auto &proxy = step.m_proxies[index];
    if (!proxy.allowResponse)
        return;
    const auto source = leaves[leafCursor];
    const auto &address = publication->directory->leaves[source.x];
    const auto &actor = *publication->directory->actors[address.bodySlot];
    if (!std::binary_search(step.m_gpuResponseBodies.begin(), step.m_gpuResponseBodies.end(), actor.bodyId) ||
        std::binary_search(step.m_suppressedPairs.begin(), step.m_suppressedPairs.end(),
                           JoltMixedContactStep::BodyPair{proxy.id, actor.bodyId}))
        return;
    const auto &leaf = actor.leaves[address.actorLeaf];
    if (leaf.sensor)
        return;
    const auto &shape = storage.GetModel().shapes[source.x];
    const auto &model = storage.GetModel().bodies[address.bodySlot];
    const bool gpuDynamic = model.meta.z == MotionDynamic;
    if (!proxy.dynamic && !gpuDynamic)
        return;
    const bool cpuContinuous = proxy.collisionDetectionMode != static_cast<int>(CollisionDetectionMode::Discrete);
    const bool cpuDynamicContinuous =
        proxy.collisionDetectionMode == static_cast<int>(CollisionDetectionMode::ContinuousDynamic);
    const float gpuRadius = model.invInertiaDiagonal.w;
    if (proxy.dynamic && gpuDynamic ? !cpuDynamicContinuous && gpuRadius >= 0.0f
        : proxy.dynamic             ? !cpuContinuous
                                    : gpuRadius == 0.0f)
        return;
    const auto &a = step.m_continuousPose[index];
    const auto da = step.m_continuousDisplacement[index];
    const auto &sample = samples[leafSample];
    const auto &b = sample.state;
    const auto db = Vector(sample.displacement);
    float radius = proxy.dynamic && cpuContinuous ? proxy.shape->GetInnerRadius() : std::abs(gpuRadius);
    if (proxy.dynamic && cpuContinuous && gpuRadius != 0.0f)
        radius = std::min(radius, std::abs(gpuRadius));
    const auto relative = da - db;
    if (relative.LengthSq() <= JPH::Square(radius * params.restitution.y))
        return;
    const JPH::Quat bodyRotation(b.rotation.x, b.rotation.y, b.rotation.z, b.rotation.w);
    const JPH::Quat leafRotation(shape.localRotation.x, shape.localRotation.y, shape.localRotation.z,
                                 shape.localRotation.w);
    const auto centerB = JPH::RVec3(Vector(b.positionCOM)) + bodyRotation * Vector(shape.localPositionCOM);
    const auto rotationB = bodyRotation * leafRotation;
    // Non-convex prescribed boundaries are targets, never convex cast drivers.
    const bool reverse = proxy.shape->MustBeStatic();
    const auto base = reverse ? centerB : a.center;
    const auto targetDelta = reverse ? da : db;
    const auto direction = reverse ? -relative : relative;
    const JPH::TransformedShape target(reverse ? a.center : centerB, reverse ? a.rotation : rotationB,
                                       reverse ? proxy.shape.GetPtr() : leaf.shape.GetPtr(), {});
    const JPH::RShapeCast cast(reverse ? leaf.shape.GetPtr() : proxy.shape.GetPtr(), JPH::Vec3::sOne(),
                               JPH::RMat44::sRotationTranslation(reverse ? rotationB : a.rotation, base), direction);
    JPH::ShapeCastSettings settings;
    settings.mUseShrunkenShapeAndConvexRadius = true;
    settings.mReturnDeepestPoint = true;
    settings.mCollectFacesMode = JPH::ECollectFacesMode::CollectFaces;
    settings.mBackFaceModeTriangles = JPH::EBackFaceMode::IgnoreBackFaces;
    settings.mBackFaceModeConvex = JPH::EBackFaceMode::IgnoreBackFaces;
    settings.mActiveEdgeMovementDirection = direction;
    settings.mActiveEdgeMode = step.Scene().System().GetPhysicsSettings().mCheckActiveEdges
                                   ? JPH::EActiveEdgeMode::CollideOnlyWithActive
                                   : JPH::EActiveEdgeMode::CollideWithAll;
    std::optional<Hit> best;
    CastCollector collector([&](const JPH::ShapeCastResult &result) {
        const auto cpuSubshape = reverse ? result.mSubShapeID2 : result.mSubShapeID1;
        const auto &surface = step.Scene().GetSurface(proxy.id, cpuSubshape);
        const JoltCpuScene::Surface other{leaf.surfaceTarget,
                                          leaf.colliderId,
                                          leaf.colliderGeneration,
                                          source.y,
                                          source.z,
                                          shape.material.x,
                                          shape.material.y,
                                          static_cast<PhysicsMaterialCombine>(shape.surface.z),
                                          static_cast<PhysicsMaterialCombine>(shape.surface.w),
                                          false};
        if (surface.sensor || !step.Scene().SurfacesCollide(surface, other))
            return false;
        const auto n = (reverse ? -1.0f : 1.0f) * result.mPenetrationAxis.Normalized();
        const float travel = n.Dot(relative);
        if (travel <= 0.0f)
            return false;
        JPH::ContactPoints first, second;
        JPH::ManifoldBetweenTwoFaces(result.mContactPointOn1, result.mContactPointOn2, result.mPenetrationAxis,
                                     step.Scene().System().GetPhysicsSettings().mManifoldTolerance, result.mShape1Face,
                                     result.mShape2Face, first, second JPH_IF_DEBUG_RENDERER(, base));
        JPH::Vec3 average = JPH::Vec3::sZero();
        for (const auto &point : second)
            average += point;
        average = second.empty() ? result.mContactPointOn2 : average / float(second.size());
        const auto point = base + average + result.mFraction * targetDelta;
        const auto armA = JPH::Vec3(point - (a.center + result.mFraction * da));
        const auto armB = JPH::Vec3(point - (JPH::RVec3(Vector(b.positionCOM)) + result.mFraction * db));
        const auto incoming =
            a.linear + a.angular.Cross(armA) - Vector(b.linearVelocity) - Vector(b.angularVelocity).Cross(armB);
        if (incoming.Dot(n) <= 0.0f)
            return false;
        Hit hit{};
        hit.cpu = {proxy.id,           cpuSubshape.GetValue(), surface.colliderId,
                   surface.generation, surface.target,         proxy.target};
        hit.gpu = {actor.bodyId,       leaf.subShapeId, leaf.colliderId, leaf.colliderGeneration,
                   leaf.surfaceTarget, actor.bodyTarget};
        hit.native = step.Scene().NativeBody(proxy.id);
        hit.sample = leafSample;
        hit.value.bodies = {0u, address.bodySlot, 0, 0};
        const auto normal = Pack(n);
        hit.value.normalFriction = {normal.x, normal.y, normal.z,
                                    JoltCpuScene::CombineSurface(surface.friction, other.friction,
                                                                 surface.frictionCombine, other.frictionCombine)};
        const auto position = Pack(point);
        hit.value.pointRestitution = {position.x, position.y, position.z,
                                      JoltCpuScene::CombineSurface(surface.restitution, other.restitution,
                                                                   surface.bounceCombine, other.bounceCombine)};
        hit.value.time = {result.mFraction, result.mFraction, params.restitution.x, 0};
        best = std::move(hit);
        return true;
    });
    target.CastShape(cast, settings, base, collector);
    if (best)
        hits.push_back(std::move(*best));
}

JoltGpuControlWork JoltMixedContinuousContacts::Job::CastNext()
{
    const auto &tree = step.m_continuousTree;
    const auto layout = tree[0];
    uint32_t operations = 0;
    while (operations++ < 256u) {
        if (leafCursor == leaves.size()) {
            Arbitrate();
            return PrepareDecisions();
        }
        if (!leafReady) {
            const auto &address = publication->directory->leaves[leaves[leafCursor].x];
            leafSample =
                static_cast<uint32_t>(std::lower_bound(slots.begin(), slots.end(), address.bodySlot) - slots.begin());
            const auto &shape = storage.GetModel().shapes[leaves[leafCursor].x];
            const auto &sample = samples[leafSample];
            const auto &motion = sample.state;
            const JPH::Quat q(motion.rotation.x, motion.rotation.y, motion.rotation.z, motion.rotation.w);
            const JPH::Quat local(shape.localRotation.x, shape.localRotation.y, shape.localRotation.z,
                                  shape.localRotation.w);
            const auto center = JPH::RVec3(Vector(motion.positionCOM)) + q * Vector(shape.localPositionCOM);
            const auto &leaf = publication->directory->actors[address.bodySlot]->leaves[address.actorLeaf];
            const auto bounds = leaf.shape->GetWorldSpaceBounds(JPH::RMat44::sRotationTranslation(q * local, center),
                                                                JPH::Vec3::sOne());
            const auto delta = XYZ(sample.displacement);
            const auto lo = Pack(bounds.mMin), hi = Pack(bounds.mMax);
            leafMin = XYZ(lo) + glm::min(delta, glm::vec3(0));
            leafMax = XYZ(hi) + glm::max(delta, glm::vec3(0));
            nodeCursor = proxyCursor = proxyEnd = 0;
            leafReady = true;
        }
        if (proxyCursor < proxyEnd) {
            const auto lo = tree[layout.z + 2u * proxyCursor], hi = tree[layout.z + 2u * proxyCursor + 1u];
            ++proxyCursor;
            if (Overlaps(leafMin, leafMax, lo, hi))
                CastPair(lo.w);
            continue;
        }
        if (nodeCursor == layout.x) {
            ++leafCursor;
            leafReady = false;
            continue;
        }
        const auto lo = tree[1u + 2u * nodeCursor], hi = tree[2u + 2u * nodeCursor];
        const auto escape = step.m_escape[nodeCursor];
        if (!Overlaps(leafMin, leafMax, lo, hi))
            nodeCursor = escape;
        else if ((hi.w & spatial::kBvhLeafBit) == 0)
            nodeCursor = lo.w;
        else {
            proxyCursor = lo.w;
            proxyEnd = lo.w + (hi.w & ~spatial::kBvhLeafBit);
            nodeCursor = escape;
        }
    }
    return {nullptr, [self = shared_from_this()] { return self->CastNext(); }, Boundary::HostWork};
}

void JoltMixedContinuousContacts::Job::Arbitrate()
{
    if (hits.empty())
        return;
    std::map<uint64_t, size_t> nodes;
    std::vector<size_t> parents;
    const auto node = [&](uint64_t key) {
        const auto [found, added] = nodes.emplace(key, parents.size());
        if (added)
            parents.push_back(parents.size());
        return found->second;
    };
    const auto root = [&](size_t index) {
        while (parents[index] != index) {
            parents[index] = parents[parents[index]];
            index = parents[index];
        }
        return index;
    };
    const auto join = [&](size_t a, size_t b) { parents[root(b)] = root(a); };
    const auto cpuKey = [](JPH::BodyID id) { return uint64_t(id.GetIndexAndSequenceNumber()) * 2u; };
    const auto gpuKey = [](uint32_t id) { return uint64_t(id) * 2u + 1u; };
    enum Kind
    {
        Cpu,
        Gpu,
        Mixed
    };
    struct Choice
    {
        Kind kind;
        size_t node, index;
        float time;
    };
    std::vector<Choice> choices;
    const auto nativeContacts = cpu.CoupledContinuousContacts();
    for (size_t i = 0; i < nativeContacts.size(); ++i) {
        const auto &contact = nativeContacts[i];
        const auto a = node(cpuKey(contact.mBody1));
        if (Body(contact.mBody2).IsDynamic())
            join(a, node(cpuKey(contact.mBody2)));
        choices.push_back({Cpu, a, i, contact.mFraction});
    }
    std::map<uint32_t, size_t> localGpu;
    for (size_t i = 0; i < samples.size(); ++i) {
        const auto &sample = samples[i];
        if (storage.GetModel().bodies[sample.identity.x].meta.z != MotionDynamic || sample.identity.z == UINT32_MAX ||
            !localGpu.emplace(sample.identity.y, i).second)
            continue;
        choices.push_back({Gpu, node(gpuKey(sample.identity.y)), i, ImpactTime(sample.identity.z)});
    }
    for (size_t i = 0; i < hits.size(); ++i) {
        const auto &hit = hits[i];
        const bool dynamicCpu = Body(hit.native).IsDynamic();
        const bool dynamicGpu = storage.GetModel().bodies[hit.value.bodies.y].meta.z == MotionDynamic;
        size_t endpoint = dynamicCpu ? node(cpuKey(hit.native)) : node(gpuKey(samples[hit.sample].identity.y));
        if (dynamicCpu && dynamicGpu)
            join(endpoint, node(gpuKey(samples[hit.sample].identity.y)));
        choices.push_back({Mixed, endpoint, i, hit.value.time.x});
    }
    std::map<size_t, size_t> best;
    std::set<size_t> touched;
    for (size_t i = 0; i < choices.size(); ++i) {
        const auto &choice = choices[i];
        const auto component = root(choice.node);
        if (choice.kind == Mixed)
            touched.insert(component);
        const auto [previous, added] = best.emplace(component, i);
        if (!added) {
            const auto &old = choices[previous->second];
            if (std::tie(choice.time, choice.kind, choice.index) < std::tie(old.time, old.kind, old.index))
                previous->second = i;
        }
    }
    for (const auto &[key, index] : nodes) {
        const auto component = root(index);
        if (touched.find(component) == touched.end())
            continue;
        const auto &winner = choices[best.at(component)];
        if (key & 1u) {
            const uint32_t gpuRoot = static_cast<uint32_t>(key / 2u);
            const bool keep = winner.kind == Gpu && samples[winner.index].identity.y == gpuRoot;
            // The mixed impact itself writes an untouched singleton's response.
            const bool direct = winner.kind == Mixed && samples[hits[winner.index].sample].identity.y == gpuRoot &&
                                samples[hits[winner.index].sample].identity.z == UINT32_MAX;
            if (!keep && !direct)
                policies.push_back({{gpuRoot, TimeBits(winner.time), 0, 0}});
        } else {
            const JPH::BodyID id(static_cast<uint32_t>(key / 2u));
            const bool keep = winner.kind == Cpu && nativeContacts[winner.index].mBody1 == id;
            updates.push_back({id, keep ? nativeContacts[winner.index].mTravelFraction : winner.time, !keep,
                               JPH::Vec3::sZero(), JPH::Vec3::sZero()});
        }
    }
    for (const auto component : touched) {
        const auto &winner = choices[best.at(component)];
        if (winner.kind == Mixed) {
            selected.push_back(hits[winner.index]);
            deviceHits.push_back(selected.back().value);
            deviceHits.back().bodies.x = CpuRow(selected.back().native);
        }
    }
    std::sort(policies.begin(), policies.end(),
              [](const auto &a, const auto &b) { return a.decision.x < b.decision.x; });
}

JoltGpuControlWork JoltMixedContinuousContacts::Job::PrepareDecisions()
{
    if (policies.empty() && selected.empty())
        return Finish();
    if ((!policies.empty() && !Allocate(18, policies.size() * sizeof(ContinuousRootPolicy))) ||
        (!selected.empty() && (!Allocate(19, cpuBodies.size() * sizeof(ContinuousCpuBody)) ||
                               !Allocate(20, deviceHits.size() * sizeof(ContinuousMixedHit)) ||
                               !Allocate(21, selected.size() * sizeof(ContinuousReceipt)))))
        return {nullptr, [self = shared_from_this()] { return self->PrepareDecisions(); }};
    if (!policies.empty())
        uploads.push_back({buffers[18], 0, Bytes(policies)});
    if (!selected.empty()) {
        uploads.push_back({buffers[19], 0, Bytes(cpuBodies)});
        uploads.push_back({buffers[20], 0, Bytes(deviceHits)});
    }
    return Upload(&Job::Restrict);
}

JoltGpuControlWork JoltMixedContinuousContacts::Job::Restrict(std::vector<rhi::ComputeBufferUpdate> pending)
{
    if (policies.empty())
        return Apply(std::move(pending));
    if (restrictFirst == params.selectedCounts.x)
        return RestrictLocalReports();
    const auto count = std::min(params.selectedCounts.x - restrictFirst, WorkItemLimit(Domain::Bodies));
    pending.push_back(Uniform({restrictFirst, count, params.selection.y, static_cast<uint32_t>(policies.size())}));
    auto packet = rhi::PrepareComputeBatch(owner.m_host, std::move(pending), {Dispatch(RestrictMotion, count)});
    return {packet,
            [self = shared_from_this(), count] {
                self->restrictFirst += count;
                return self->Restrict();
            },
            Boundary::Acceptance};
}

JoltGpuControlWork JoltMixedContinuousContacts::Job::RestrictLocalReports()
{
    const auto total = inputs.GetSweepContacts()->patchCount;
    if (policies.empty() || reportFirst == total)
        return Apply();
    const auto count = std::min(total - reportFirst, WorkItemLimit(Domain::Bodies));
    auto packet = rhi::PrepareComputeBatch(
        owner.m_host, {Uniform({reportFirst, count, params.selection.y, static_cast<uint32_t>(policies.size())})},
        {Dispatch(RestrictReports, count)});
    return {packet,
            [self = shared_from_this(), count] {
                self->reportFirst += count;
                return self->RestrictLocalReports();
            },
            Boundary::Acceptance};
}

JoltGpuControlWork JoltMixedContinuousContacts::Job::Apply(std::vector<rhi::ComputeBufferUpdate> pending)
{
    if (applyFirst == selected.size())
        return Finish();
    const auto count = std::min(uint32_t(selected.size()) - applyFirst,
                                uint32_t((TransferPageBytes - sizeof(Parameters)) / sizeof(ContinuousReceipt)));
    const uint64_t offset = uint64_t(applyFirst) * sizeof(ContinuousReceipt);
    pending.push_back(Uniform({applyFirst, count, params.selection.y, 0}));
    auto packet =
        rhi::PrepareComputeBatch(owner.m_host, std::move(pending), {Dispatch(Entry::Apply, count, applyFirst)},
                                 {{buffers[21], offset, uint64_t(count) * sizeof(ContinuousReceipt)}});
    return {packet, [self = shared_from_this(), packet, count] {
                std::vector<ContinuousReceipt> receipts(count);
                packet->Readback(0)->ReadInto(receipts.data(), count * sizeof(ContinuousReceipt));
                for (uint32_t i = 0; i < count; ++i) {
                    const auto &hit = self->selected[self->applyFirst + i];
                    const auto &receipt = receipts[i];
                    if (self->Body(hit.native).IsDynamic()) {
                        auto update =
                            std::lower_bound(self->updates.begin(), self->updates.end(), hit.native,
                                             [](const auto &value, JPH::BodyID id) { return value.mBody < id; });
                        update->mLinearImpulse += Vector(receipt.linear);
                        update->mAngularImpulse += Vector(receipt.angular);
                    }
                    if (self->policy.Observes(hit.cpu.SurfaceKey(), hit.gpu.SurfaceKey(), false)) {
                        auto &frame = self->step.ContactFrame(self->substep);
                        frame.manifolds.push_back({hit.cpu, hit.gpu, false, frame.points.size(), 1});
                        frame.points.push_back({XYZ(hit.value.pointRestitution), XYZ(hit.value.normalFriction),
                                                XYZ(receipt.relative), -XYZ(receipt.linear), 0, self->policy.impulses});
                    }
                }
                self->applyFirst += count;
                return self->Apply();
            }};
}

JoltGpuControlWork JoltMixedContinuousContacts::Prepare(
    const JoltGpuStorage &storage, JoltGpuStepInputs &inputs, const StepParams &params, uint32_t substep,
    const std::shared_ptr<const JoltGpuScene::Publication> &publication, JoltGpuMixedContacts &discovery,
    JoltMixedContactStep &step, JoltCpuContactPublication &cpu, const JoltMixedContactStep::Policy &policy,
    std::function<void()> completed)
{
    return std::make_shared<Job>(*this, storage, inputs, params, substep, publication, discovery, step, cpu, policy,
                                 std::move(completed))
        ->Begin();
}
} // namespace infernux
