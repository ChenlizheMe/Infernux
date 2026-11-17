#include "JoltMixedContacts.h"
#include "JoltCpuScene.h"
#include "JoltGpuProgramSources.h"

#include <Jolt/Physics/Collision/CollideShape.h>
#include <Jolt/Physics/Collision/ManifoldBetweenTwoFaces.h>
#include <Jolt/Physics/Collision/TransformedShape.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace infernux
{
namespace
{
using namespace JPH::GPU;
using Contact = PhysicsContactPublication;
constexpr uint32_t ResponseProxy = 2;
constexpr uint32_t DynamicProxy = 4;
constexpr uint32_t ObservedActive = 8;

template <class Vector> glm::vec3 Vector3(const Vector &v)
{
    return {float(v.GetX()), float(v.GetY()), float(v.GetZ())};
}

JPH::Vec3 Vector3(Float4 v)
{
    return {v.x, v.y, v.z};
}

uint32_t Bits(float value)
{
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

UInt4 EncodeBounds(glm::vec3 value, uint32_t metadata)
{
    return {Bits(value.x), Bits(value.y), Bits(value.z), metadata};
}

glm::vec3 DecodeBounds(UInt4 value)
{
    return glm::uintBitsToFloat(glm::uvec3(value.x, value.y, value.z));
}

bool Overlaps(const spatial::Bounds &bounds, UInt4 lower, UInt4 upper)
{
    return glm::all(glm::lessThanEqual(bounds.minimum, DecodeBounds(upper))) &&
           glm::all(glm::lessThanEqual(DecodeBounds(lower), bounds.maximum));
}

std::vector<uint8_t> Bytes(const void *data, size_t size)
{
    std::vector<uint8_t> bytes(size);
    std::memcpy(bytes.data(), data, size);
    return bytes;
}

void EscapeIndices(const spatial::Bvh &tree, std::vector<uint32_t> &indices, uint32_t node, uint32_t after)
{
    indices[node] = after;
    const auto &value = tree.nodes[node];
    if (!(value.second & spatial::kBvhLeafBit)) {
        EscapeIndices(tree, indices, value.first, value.second);
        EscapeIndices(tree, indices, value.second, after);
    }
}

template <class Observe> class CrossPairCollector final : public JPH::CollideShapeCollector
{
  public:
    explicit CrossPairCollector(Observe observe) : m_observe(std::move(observe))
    {
    }

    void AddHit(const JPH::CollideShapeResult &hit) override
    {
        m_observe(hit);
    }

  private:
    Observe m_observe;
};
} // namespace

void JoltMixedContactStep::Begin(const JoltCpuScene &scene, std::vector<CpuEndpoint> endpoints,
                                 std::vector<uint32_t> gpuResponseBodies, uint32_t substeps)
{
    m_scene = &scene;
    m_proxies.clear();
    m_responses.clear();
    std::sort(gpuResponseBodies.begin(), gpuResponseBodies.end());
    m_gpuResponseBodies = std::move(gpuResponseBodies);
    std::sort(endpoints.begin(), endpoints.end(), [](auto a, auto b) { return a.body < b.body; });
    std::vector<uint32_t> bodyIds;
    bodyIds.reserve(endpoints.size());
    const auto &bodies = scene.System().GetBodyLockInterfaceNoLock();
    for (const auto &endpoint : endpoints) {
        const auto found = scene.m_bodies.find(endpoint.body);
        if (found == scene.m_bodies.end() || found->second.surfaces.empty())
            continue;
        const auto &record = found->second;
        const auto &body = *bodies.TryGetBody(record.native);
        if (!body.IsInBroadPhase())
            continue;
        const bool sensor = std::any_of(record.surfaces.begin(), record.surfaces.end(),
                                        [](const auto &surface) { return surface.sensor; });
        m_proxies.push_back({endpoint.body, record.shapeRevision, record.shape, record.bodyTarget, sensor,
                             endpoint.allowResponse && !body.IsStatic(), body.IsDynamic()});
        bodyIds.push_back(endpoint.body);
    }
    if (m_ids != bodyIds) {
        m_ids = std::move(bodyIds);
        m_tree = {};
        m_escape.clear();
    }
    m_pose.resize(size_t(substeps) + 1);
    m_treeWords.resize(substeps);
    m_frames.resize(substeps);
    for (auto &frame : m_frames) {
        frame.manifolds.clear();
        frame.points.clear();
    }
    CaptureCpuBoundary(0);
}

void JoltMixedContactStep::CaptureCpuBoundary(uint32_t boundary)
{
    auto &poses = m_pose[boundary];
    poses.clear();
    poses.reserve(m_proxies.size());
    const auto &bodies = m_scene->System().GetBodyLockInterfaceNoLock();
    for (const auto &proxy : m_proxies) {
        const auto *body = bodies.TryGetBody(m_scene->NativeBody(proxy.id));
        poses.push_back({body->GetCenterOfMassPosition(), body->GetRotation(), body->GetLinearVelocity(),
                         body->GetAngularVelocity(),
                         proxy.shape->GetWorldSpaceBounds(body->GetCenterOfMassTransform(), JPH::Vec3::sOne()),
                         body->IsActive()});
    }
}

const std::vector<JPH::GPU::UInt4> &JoltMixedContactStep::EncodeTree(uint32_t substep, float predictedInterval)
{
    auto &words = m_treeWords[substep];
    if (m_proxies.empty())
        return words;
    const auto &start = m_pose[substep];
    const auto &end = m_pose[substep + 1];
    std::vector<spatial::Bounds> bounds;
    bounds.reserve(m_proxies.size());
    for (size_t i = 0; i < m_proxies.size(); ++i) {
        const auto &a = start[i];
        auto minimum = Vector3(a.bounds.mMin);
        auto maximum = Vector3(a.bounds.mMax);
        if (predictedInterval > 0) {
            const auto translation = predictedInterval * Vector3(a.linear);
            const auto center = Vector3(a.center);
            const float radius = glm::length(glm::max(glm::abs(minimum - center), glm::abs(maximum - center)));
            const float angular = radius * std::min(2.0f, predictedInterval * a.angular.Length());
            const glm::vec3 margin(angular + m_scene->System().GetPhysicsSettings().mSpeculativeContactDistance);
            minimum += glm::min(translation, glm::vec3(0)) - margin;
            maximum += glm::max(translation, glm::vec3(0)) + margin;
        } else {
            const auto &b = end[i];
            minimum = glm::min(minimum, Vector3(b.bounds.mMin));
            maximum = glm::max(maximum, Vector3(b.bounds.mMax));
            if (a.rotation != b.rotation) {
                // A sphere about the COM bounds every intermediate orientation.
                const auto local = m_proxies[i].shape->GetLocalBounds();
                const auto radius = glm::length(glm::max(glm::abs(Vector3(local.mMin)), glm::abs(Vector3(local.mMax))));
                minimum = glm::min(minimum, glm::min(Vector3(a.center), Vector3(b.center)) - glm::vec3(radius));
                maximum = glm::max(maximum, glm::max(Vector3(a.center), Vector3(b.center)) + glm::vec3(radius));
            }
        }
        bounds.push_back({minimum, maximum});
    }
    if (m_tree.nodes.empty()) {
        m_tree = spatial::BuildBvh(bounds);
        const auto count = static_cast<uint32_t>(m_tree.nodes.size());
        m_escape.resize((count + 3u) & ~3u);
        EscapeIndices(m_tree, m_escape, 0, count);
    } else {
        spatial::RefitBvh(m_tree, bounds);
    }
    const auto count = static_cast<uint32_t>(m_tree.nodes.size());
    const auto escapeBase = 1u + 2u * count;
    const auto boundsBase = escapeBase + static_cast<uint32_t>(m_escape.size() / 4);
    words.resize(boundsBase + 2 * bounds.size());
    words[0] = {count, escapeBase, boundsBase, static_cast<uint32_t>(bounds.size())};
    std::memcpy(words.data() + 1, m_tree.nodes.data(), count * sizeof(spatial::BvhNode));
    std::memcpy(words.data() + escapeBase, m_escape.data(), m_escape.size() * sizeof(uint32_t));
    for (size_t i = 0; i < bounds.size(); ++i) {
        const uint32_t proxy = m_tree.primitiveOrder[i];
        words[boundsBase + 2 * i] = EncodeBounds(bounds[proxy].minimum, proxy);
        const auto &source = m_proxies[proxy];
        const uint32_t flags = (source.sensor ? ShapeSensor : 0u) | (source.allowResponse ? ResponseProxy : 0u) |
                               (source.dynamic ? DynamicProxy : 0u);
        words[boundsBase + 2 * i + 1] = EncodeBounds(bounds[proxy].maximum, flags);
    }
    return words;
}

void JoltMixedContactStep::Collect(uint32_t substep, const JoltGpuScene::Publication &publication, const Policy &policy,
                                   const Candidate &candidate, const JoltGpuModel *responseModel)
{
    const auto &address = publication.directory->leaves[candidate.source.x];
    const auto &actor = *publication.directory->actors[address.bodySlot];
    const auto &leaf = actor.leaves[address.actorLeaf];
    const JoltCpuScene::Surface gpuSurface{leaf.surfaceTarget,
                                           leaf.colliderId,
                                           leaf.colliderGeneration,
                                           candidate.source.y,
                                           candidate.source.z,
                                           candidate.material.x,
                                           candidate.material.y,
                                           static_cast<PhysicsMaterialCombine>(Bits(candidate.material.z)),
                                           static_cast<PhysicsMaterialCombine>(Bits(candidate.material.w)),
                                           leaf.sensor};
    const JPH::RVec3 center(candidate.center.x, candidate.center.y, candidate.center.z);
    const JPH::Quat rotation(candidate.rotation.x, candidate.rotation.y, candidate.rotation.z, candidate.rotation.w);
    const JPH::TransformedShape gpu(center, rotation, leaf.shape.GetPtr(), {});
    auto &frame = m_frames[substep];
    const bool gpuActive = (candidate.source.w & ObservedActive) != 0;
    const bool gpuDynamic = (candidate.source.w & ~ObservedActive) == MotionDynamic;
    const bool gpuPermitsResponse =
        !leaf.sensor && std::binary_search(m_gpuResponseBodies.begin(), m_gpuResponseBodies.end(), actor.bodyId);
    const bool observeTriggers = policy.WantsTriggers();
    const auto &physicsSettings = m_scene->System().GetPhysicsSettings();
    const auto collect = [&](uint32_t index, bool mayRespond) {
        const auto &proxy = m_proxies[index];
        const auto &cpu = m_pose[substep][index];
        JPH::CollideShapeSettings settings;
        if (mayRespond) {
            settings.mCollectFacesMode = JPH::ECollectFacesMode::CollectFaces;
            settings.mMaxSeparationDistance = physicsSettings.mSpeculativeContactDistance;
        }
        CrossPairCollector collector([&](const JPH::CollideShapeResult &hit) {
            const auto &surface = m_scene->GetSurface(proxy.id, hit.mSubShapeID1);
            const bool trigger = surface.sensor || leaf.sensor;
            if (!m_scene->SurfacesCollide(surface, gpuSurface) ||
                (trigger ? hit.mPenetrationDepth < 0.0f ||
                               !policy.Observes({surface.colliderId, surface.generation},
                                                {leaf.colliderId, leaf.colliderGeneration}, true)
                         : !mayRespond))
                return;
            const auto normal = hit.mPenetrationAxis.Normalized();
            const auto velocity = [&](JPH::RVec3Arg position) {
                const auto va = cpu.linear + cpu.angular.Cross(JPH::Vec3(position - cpu.center));
                const auto vb =
                    Vector3(candidate.linear) + Vector3(candidate.angular).Cross(JPH::Vec3(position - center));
                return Vector3(vb - va);
            };
            const Contact::Endpoint a{proxy.id,           hit.mSubShapeID1.GetValue(),
                                      surface.colliderId, surface.generation,
                                      surface.target,     proxy.target};
            const Contact::Endpoint b{actor.bodyId,       leaf.subShapeId, leaf.colliderId, leaf.colliderGeneration,
                                      leaf.surfaceTarget, actor.bodyTarget};
            if (trigger || (!cpu.active && !gpuActive)) {
                if (!trigger &&
                    (hit.mPenetrationDepth < 0.0f || !policy.Observes(a.SurfaceKey(), b.SurfaceKey(), false)))
                    return;
                const auto position = cpu.center + 0.5f * (hit.mContactPointOn1 + hit.mContactPointOn2);
                frame.manifolds.push_back({a, b, trigger, frame.points.size(), 1});
                frame.points.push_back({Vector3(position), Vector3(normal), velocity(position), {}, 0, false});
                return;
            }

            JPH::ContactPoints first, second;
            JPH::ManifoldBetweenTwoFaces(hit.mContactPointOn1, hit.mContactPointOn2, hit.mPenetrationAxis,
                                         settings.mMaxSeparationDistance + physicsSettings.mManifoldTolerance,
                                         hit.mShape1Face, hit.mShape2Face, first,
                                         second JPH_IF_DEBUG_RENDERER(, cpu.center));
            if (first.size() > MaxContactPointsPerPair)
                JPH::PruneContactPoints(normal, first, second JPH_IF_DEBUG_RENDERER(, cpu.center));
            auto &patch = m_responses.emplace_back();
            patch.cpu = a;
            patch.gpu = b;
            patch.shapeRevisions = {proxy.shapeRevision, actor.shapeRevision};
            patch.gpuSlot = address.bodySlot;
            patch.cpuDynamic = proxy.dynamic;
            patch.gpuDynamic = gpuDynamic;
            patch.wakeGpu = gpuDynamic && !gpuActive;
            patch.native = m_scene->NativeBody(proxy.id);
            patch.pointCount = static_cast<uint32_t>(first.size());
            const auto n = Vector3(normal);
            patch.normalFriction = {n.x, n.y, n.z,
                                    JoltCpuScene::CombineSurface(surface.friction, gpuSurface.friction,
                                                                 surface.frictionCombine, gpuSurface.frictionCombine)};
            patch.restitution = JoltCpuScene::CombineSurface(surface.restitution, gpuSurface.restitution,
                                                             surface.bounceCombine, gpuSurface.bounceCombine);
            const auto &model = responseModel->shapes[candidate.source.x];
            const auto &r = model.localRotation;
            const auto &p = model.localPositionCOM;
            const JPH::Quat gpuInverse = JPH::Quat(r.x, r.y, r.z, r.w) * rotation.Conjugated();
            const JPH::Vec3 gpuLocalCenter(p.x, p.y, p.z);
            const auto cpuInverse = cpu.rotation.Conjugated();
            const JPH::Vec3 centerDifference(cpu.center - center);
            for (uint32_t i = 0; i < patch.pointCount; ++i) {
                const auto point = cpu.center + 0.5f * (first[i] + second[i]);
                const auto midpoint = Vector3(point);
                const auto vb = Vector3(candidate.linear) + Vector3(candidate.angular).Cross(JPH::Vec3(point - center));
                patch.points[i] = {{midpoint.x, midpoint.y, midpoint.z, (first[i] - second[i]).Dot(normal)},
                                   Vector3(vb),
                                   Vector3(cpuInverse * first[i]),
                                   Vector3(gpuLocalCenter + gpuInverse * (centerDifference + second[i]))};
            }
        });
        gpu.CollideShape(proxy.shape.GetPtr(), JPH::Vec3::sOne(),
                         JPH::RMat44::sRotationTranslation(cpu.rotation, cpu.center), settings, cpu.center, collector);
    };

    const auto &tree = m_treeWords[substep];
    const auto layout = tree[0];
    const spatial::Bounds sweep{{candidate.sweepMin.x, candidate.sweepMin.y, candidate.sweepMin.z},
                                {candidate.sweepMax.x, candidate.sweepMax.y, candidate.sweepMax.z}};
    for (uint32_t node = 0; node < layout.x;) {
        const auto lower = tree[1u + node * 2u], upper = tree[2u + node * 2u];
        if (Overlaps(sweep, lower, upper)) {
            if (!(upper.w & spatial::kBvhLeafBit)) {
                node = lower.w;
                continue;
            }
            const auto end = lower.w + (upper.w & ~spatial::kBvhLeafBit);
            for (uint32_t i = lower.w; i < end; ++i) {
                const auto proxyMin = tree[layout.z + 2u * i], proxyMax = tree[layout.z + 2u * i + 1u];
                const bool trigger = observeTriggers && (leaf.sensor || (proxyMax.w & ShapeSensor));
                const bool response =
                    gpuPermitsResponse && (proxyMax.w & ResponseProxy) && (gpuDynamic || (proxyMax.w & DynamicProxy));
                if ((trigger || response) && Overlaps(sweep, proxyMin, proxyMax))
                    collect(proxyMin.w, response);
            }
        }
        node = m_escape[node];
    }
}

void JoltMixedContactStep::AppendFrame(uint32_t substep, Frame &destination)
{
    if (!m_scene)
        return;
    auto &source = m_frames[substep];
    const auto offset = destination.points.size();
    destination.points.insert(destination.points.end(), source.points.begin(), source.points.end());
    for (auto &manifold : source.manifolds) {
        manifold.pointOffset += offset;
        destination.manifolds.push_back(std::move(manifold));
    }
}

void JoltMixedContactStep::ClearStep()
{
    m_scene = nullptr;
    m_proxies.clear();
    m_gpuResponseBodies.clear();
    m_responses.clear();
    for (auto &words : m_treeWords)
        words.clear();
    for (auto &frame : m_frames) {
        frame.manifolds.clear();
        frame.points.clear();
    }
}

JoltGpuMixedContacts::JoltGpuMixedContacts(rhi::ComputeHost &host, const ShaderTargetCode &target) : m_host(host)
{
    const auto entry = std::find_if(target.entries.begin(), target.entries.end(),
                                    [](const auto &code) { return code.name == "gather_cross_domain_candidates"; });
    if (entry == target.entries.end() || entry->workgroupSize != std::array<uint32_t, 3>{64, 1, 1})
        throw std::invalid_argument("Cross-domain candidate shader has a different workgroup layout");
    std::vector<rhi::ComputeBufferBinding> bindings;
    for (const auto &resource : target.resources) {
        const auto &binding = resource.binding;
        const std::array<uint32_t, 2> key{resource.set, binding.binding};
        if (std::find(entry->bindings.begin(), entry->bindings.end(), key) == entry->bindings.end())
            continue;
        if (resource.set != 0 || binding.binding > 8)
            throw std::invalid_argument("Cross-domain candidate shader uses an unknown port");
        m_ports.push_back(binding.binding);
        bindings.push_back({binding.binding, binding.type, binding.readOnlyStorage});
    }
    m_kernel = PrepareJoltGpuKernel(host, target.target, *entry, std::move(bindings));
}

struct JoltGpuMixedContacts::Job : std::enable_shared_from_this<Job>
{
    using Candidate = JoltMixedContactStep::Candidate;
    using Boundary = JoltGpuControlWork::ResumeAfter;
    static constexpr uint32_t LeavesPerDispatch = 2048;
    static constexpr uint64_t TransferPerPacket = 64 * 1024;

    struct Window
    {
        uint32_t substep;
        uint32_t leaves;
        std::shared_ptr<rhi::ComputeBuffer> candidates;
    };
    struct ReadRange
    {
        uint32_t substep;
        uint32_t count;
    };

    JoltGpuMixedContacts &owner;
    JoltMixedContactStep &step;
    const JoltMixedContactStep::Policy &policy;
    std::shared_ptr<const JoltGpuScene::Publication> publication;
    // Uniform, CPU BVH, GPU bounds/shapes/models/states, counts, hits, GPU sleep.
    std::array<std::shared_ptr<rhi::ComputeBuffer>, 9> buffers;
    std::vector<Window> windows;
    std::vector<uint32_t> counts;
    std::vector<Candidate> decoded;
    const std::vector<UInt4> *tree = nullptr;
    const JoltGpuModel *responseModel = nullptr;
    std::function<JoltGpuControlWork()> substepReady;
    bool discovery = false;
    UInt4 range{};
    uint64_t treeOffset = 0;
    uint32_t substep = 0, leafEnd = 0, readWindow = 0, readFirst = 0;
    uint64_t countBytes = 0;

    Job(JoltGpuMixedContacts &owner, JoltMixedContactStep &step, const JoltMixedContactStep::Policy &policy,
        std::shared_ptr<const JoltGpuScene::Publication> publication)
        : owner(owner), step(step), policy(policy), publication(std::move(publication))
    {
    }

    bool Allocate(size_t port, uint64_t bytes)
    {
        auto &slot = buffers[port];
        if (slot && slot->GetByteSize() >= bytes)
            return true;
        slot = rhi::ComputeBuffer::TryCreatePrivate(owner.m_host, {bytes / 4, rhi::ComputeScalarType::UInt32, 1, {}});
        return bool(slot);
    }

    JoltGpuControlWork Begin(const JoltGpuStorage &storage, const JPH::GPU::StepParams &params, uint32_t index,
                             bool response, std::function<JoltGpuControlWork()> done)
    {
        discovery = response;
        if (discovery) {
            responseModel = &storage.GetModel();
            step.Responses().clear();
            step.CaptureCpuBoundary(index);
        }
        tree = &step.EncodeTree(index, discovery ? params.gravityDt.w : 0.0f);
        const uint32_t leaves = params.selectedCounts.y;
        if (!leaves || tree->empty())
            return done();
        substep = index;
        substepReady = std::move(done);
        range = {params.selection.z, 0u, 0u, Bits(params.gravityDt.w)};
        leafEnd = range.x + leaves;
        treeOffset = 0;
        const uint64_t perSubstep = (uint64_t(leaves) + LeavesPerDispatch - 1u) / LeavesPerDispatch;
        countBytes = perSubstep * (discovery ? 1u : params.solve.y) * sizeof(uint32_t);
        buffers[2] = storage.Get(Resource::Bounds);
        buffers[3] = publication->shapeModels;
        buffers[4] = publication->bodyModels;
        buffers[5] = storage.Get(Resource::BodyStates);
        buffers[8] = storage.Get(Resource::SleepStates);
        return Upload();
    }

    JoltGpuControlWork Upload()
    {
        const uint64_t treeBytes = tree->size() * sizeof(UInt4);
        if (!Allocate(0, 2 * sizeof(range)) || !Allocate(1, treeBytes) || !Allocate(6, countBytes))
            return {nullptr, [self = shared_from_this()] { return self->Upload(); }};
        const uint64_t bytes = std::min(treeBytes - treeOffset, TransferPerPacket);
        const auto *data = reinterpret_cast<const uint8_t *>(tree->data());
        std::vector<rhi::ComputeBufferUpdate> updates{{buffers[1], treeOffset, Bytes(data + treeOffset, bytes)}};
        if (treeOffset + bytes == treeBytes)
            return Gather(std::move(updates));
        auto packet = rhi::PrepareComputeBatch(owner.m_host, std::move(updates), {});
        return {packet,
                [self = shared_from_this(), bytes] {
                    self->treeOffset += bytes;
                    return self->Upload();
                },
                Boundary::Acceptance};
    }

    JoltGpuControlWork Gather(std::vector<rhi::ComputeBufferUpdate> updates = {})
    {
        range.y = std::min(LeavesPerDispatch, leafEnd - range.x);
        range.z = static_cast<uint32_t>(windows.size());
        if (!Allocate(7, uint64_t(range.y) * sizeof(Candidate)))
            return {nullptr, [self = shared_from_this(), updates = std::move(updates)]() mutable {
                        return self->Gather(std::move(updates));
                    }};
        const uint32_t zero = 0;
        const uint32_t flags = (policy.WantsTriggers() ? 1u : 0u) | (discovery ? 2u : 0u);
        const std::array<UInt4, 2> uniform{range, UInt4{flags, 0, 0, 0}};
        updates.push_back({buffers[0], 0, Bytes(uniform.data(), sizeof(uniform))});
        updates.push_back({buffers[6], uint64_t(range.z) * sizeof(uint32_t), Bytes(&zero, sizeof(zero))});

        rhi::ComputeDispatchDesc dispatch;
        dispatch.kernel = owner.m_kernel;
        dispatch.groupCountX = (range.y + 63u) / 64u;
        for (const auto port : owner.m_ports) {
            dispatch.buffers.push_back(buffers[port]);
            dispatch.bufferAccesses.push_back(port == 6   ? rhi::ComputeBufferAccess::ReadWrite
                                              : port == 7 ? rhi::ComputeBufferAccess::Write
                                                          : rhi::ComputeBufferAccess::Read);
        }
        if (discovery && windows.empty() && range.y <= 32u) {
            auto packet =
                rhi::PrepareComputeBatch(owner.m_host, std::move(updates), {std::move(dispatch)},
                                         {{buffers[6], uint64_t(range.z) * sizeof(uint32_t), sizeof(uint32_t)},
                                          {buffers[7], 0, uint64_t(range.y) * sizeof(Candidate)}});
            return {packet, [self = shared_from_this(), packet] {
                        uint32_t count;
                        packet->Readback(0)->ReadInto(&count, sizeof(count));
                        if (count > self->range.y)
                            throw std::runtime_error("Cross-domain candidate count exceeds its leaf range");
                        if (count) {
                            self->decoded.resize(self->range.y);
                            packet->Readback(1)->ReadInto(self->decoded.data(),
                                                          self->decoded.size() * sizeof(Candidate));
                            for (uint32_t i = 0; i < count; ++i)
                                self->step.Collect(self->substep, *self->publication, self->policy, self->decoded[i],
                                                   self->responseModel);
                        }
                        auto ready = std::move(self->substepReady);
                        return ready();
                    }};
        }
        auto packet = rhi::PrepareComputeBatch(owner.m_host, std::move(updates), {std::move(dispatch)});
        return {packet,
                [self = shared_from_this()] {
                    self->windows.push_back({self->substep, self->range.y, std::move(self->buffers[7])});
                    self->range.x += self->range.y;
                    if (self->range.x < self->leafEnd)
                        return self->Gather();
                    if (self->discovery)
                        return self->ReadCounts();
                    auto ready = std::move(self->substepReady);
                    return ready();
                },
                Boundary::Acceptance};
    }

    JoltGpuControlWork ReadCounts()
    {
        if (windows.empty())
            return {};
        auto packet =
            rhi::PrepareComputeBatch(owner.m_host, {}, {}, {{buffers[6], 0, windows.size() * sizeof(uint32_t)}});
        return {packet, [self = shared_from_this(), packet] {
                    self->counts.resize(self->windows.size());
                    packet->Readback(0)->ReadInto(self->counts.data(), self->counts.size() * sizeof(uint32_t));
                    for (size_t i = 0; i < self->counts.size(); ++i)
                        if (self->counts[i] > self->windows[i].leaves)
                            throw std::runtime_error("Cross-domain candidate count exceeds its leaf range");
                    return self->ReadCandidates();
                }};
    }

    JoltGpuControlWork ReadCandidates()
    {
        std::vector<rhi::ComputeBufferRead> reads;
        std::vector<ReadRange> ranges;
        auto nextWindow = readWindow;
        auto nextFirst = readFirst;
        uint32_t remaining = TransferPerPacket / sizeof(Candidate);
        while (nextWindow < windows.size() && remaining) {
            const auto &window = windows[nextWindow];
            const auto count = std::min(remaining, counts[nextWindow] - nextFirst);
            if (count) {
                const uint64_t bytes = uint64_t(count) * sizeof(Candidate);
                reads.push_back({window.candidates, uint64_t(nextFirst) * sizeof(Candidate), bytes});
                ranges.push_back({window.substep, count});
                nextFirst += count;
                remaining -= count;
            }
            if (nextFirst == counts[nextWindow]) {
                ++nextWindow;
                nextFirst = 0;
            }
        }
        if (reads.empty()) {
            if (!discovery)
                return {};
            auto ready = std::move(substepReady);
            return ready();
        }
        auto packet = rhi::PrepareComputeBatch(owner.m_host, {}, {}, std::move(reads));
        return {packet, [self = shared_from_this(), packet, ranges = std::move(ranges), nextWindow, nextFirst] {
                    for (size_t i = 0; i < ranges.size(); ++i) {
                        const auto &range = ranges[i];
                        self->decoded.resize(range.count);
                        packet->Readback(i)->ReadInto(self->decoded.data(), range.count * sizeof(Candidate));
                        for (const auto &candidate : self->decoded)
                            self->step.Collect(range.substep, *self->publication, self->policy, candidate,
                                               self->responseModel);
                    }
                    self->readWindow = nextWindow;
                    self->readFirst = nextFirst;
                    return self->ReadCandidates();
                }};
    }
};

JoltGpuScene::ContactObserver
JoltGpuMixedContacts::Observe(const std::shared_ptr<const JoltGpuScene::Publication> &publication,
                              JoltMixedContactStep &step, const JoltMixedContactStep::Policy &policy)
{
    auto job = std::make_shared<Job>(*this, step, policy, publication);
    return {[job](const JoltGpuStorage &storage, const JoltGpuStepInputs &, const JPH::GPU::StepParams &params,
                  uint32_t substep, JoltGpuContactReportStage stage, std::function<void()> done) {
                if (stage == JoltGpuContactReportStage::BeforeVelocity)
                    return job->Begin(storage, params, substep, false, [done = std::move(done)] {
                        done();
                        return JoltGpuControlWork{};
                    });
                done();
                return JoltGpuControlWork{};
            },
            [job](const auto &) { return job->ReadCounts(); }};
}

JoltGpuControlWork JoltGpuMixedContacts::Discover(const JoltGpuStorage &storage, const JPH::GPU::StepParams &params,
                                                  uint32_t substep,
                                                  const std::shared_ptr<const JoltGpuScene::Publication> &publication,
                                                  JoltMixedContactStep &step,
                                                  const JoltMixedContactStep::Policy &policy,
                                                  std::function<JoltGpuControlWork()> ready)
{
    return std::make_shared<Job>(*this, step, policy, publication)
        ->Begin(storage, params, substep, true, std::move(ready));
}
} // namespace infernux
