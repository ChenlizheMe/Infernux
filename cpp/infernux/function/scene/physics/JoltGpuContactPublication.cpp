#include "JoltGpuContactPublication.h"

#include "JoltGpuContactReports.h"
#include <cmath>
#include <stdexcept>
#include <utility>

namespace infernux
{
namespace
{
using State = PhysicsContactPublication;
using namespace JPH::GPU;
constexpr uint32_t MaxPatchPoints = 4;

glm::vec3 Vector(Float4 value)
{
    return {value.x, value.y, value.z};
}

bool Finite(Float4 value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z) && std::isfinite(value.w);
}

uint32_t Flags(const State::Policy &policy)
{
    return (policy.events ? 1u : 0u) | (policy.events && policy.triggers ? 2u : 0u) | (policy.impulses ? 4u : 0u);
}

std::vector<uint32_t> LeafInterest(const JoltGpuScene::Publication &p, const State::Policy &policy)
{
    if (!p.directory)
        throw std::logic_error("Contact interest needs accepted directory");
    std::vector<uint32_t> result(p.leafCount);
    for (uint32_t i = 0; i < p.leafCount; ++i) {
        const auto &address = p.directory->leaves.at(p.leafBase + i);
        const auto &actor = p.directory->actors.at(address.bodySlot);
        if (!actor || address.actorLeaf >= actor->leaves.size())
            throw std::logic_error("Invalid accepted interest directory");
        const auto &leaf = actor->leaves[address.actorLeaf];
        const auto mask = policy.Interest({leaf.colliderId, leaf.colliderGeneration});
        constexpr uint8_t collisions = CollisionEnterInterest | CollisionStayInterest | CollisionExitInterest;
        constexpr uint8_t triggers = TriggerEnterInterest | TriggerStayInterest | TriggerExitInterest;
        result[i] = ((mask & collisions) ? 1u : 0u) | ((mask & triggers) ? 2u : 0u);
    }
    return result;
}

State::Endpoint DecodeEndpoint(const JoltGpuScene::Publication &p, uint32_t slot, uint32_t leaf)
{
    if (slot < p.bodyBase || slot - p.bodyBase >= p.bodyCount || leaf < p.leafBase ||
        leaf - p.leafBase >= p.leafCount || !p.directory || slot >= p.directory->actors.size() ||
        leaf >= p.directory->leaves.size())
        throw std::runtime_error("Contact endpoint outside retained publication range");
    const auto &actor = p.directory->actors[slot];
    const auto address = p.directory->leaves[leaf];
    if (!actor || address.bodySlot != slot || address.actorLeaf >= actor->leaves.size() ||
        slot >= p.directory->residentBodies.size() || !p.directory->residentBodies[slot])
        throw std::runtime_error("Contact endpoint differs from retained actor/leaf directory");
    const auto &surface = actor->leaves[address.actorLeaf];
    return {actor->bodyId,         surface.subShapeId, surface.colliderId, surface.colliderGeneration,
            surface.surfaceTarget, actor->bodyTarget};
}

State::Manifold DecodeManifold(const JoltGpuScene::Publication &p, const ContactReportPatch &r, uint32_t substep,
                               std::vector<State::Point> &points)
{
    if (r.meta.x != substep || !r.meta.z || r.meta.z > MaxPatchPoints ||
        (r.meta.y & ~(PatchActive | PatchSensor | PatchOverlapping)))
        throw std::runtime_error("Invalid GPU contact report metadata");
    const auto a = DecodeEndpoint(p, r.source.z, r.source.x);
    const auto b = DecodeEndpoint(p, r.source.w, r.source.y);
    if (a.body == b.body)
        throw std::runtime_error("GPU contact report contains a self pair");
    const bool trigger = (r.meta.y & PatchSensor) != 0;
    if (trigger ? !(r.meta.y & PatchOverlapping) : !(r.meta.y & PatchActive))
        throw std::runtime_error("GPU contact kind/active mismatch");
    const auto offset = points.size();
    for (uint32_t i = 0; i < r.meta.z; ++i) {
        const auto &v = r.points[i];
        const auto n = Vector(v.normal);
        if (v.feature.y != 1 || v.feature.z > 1 || v.feature.w || !Finite(v.pointDepth) || !Finite(v.normal) ||
            !Finite(v.relativeVelocity) || !Finite(v.impulse) || std::abs(glm::dot(n, n) - 1.0f) > 1.0e-3f ||
            (trigger && (v.impulse.x || v.impulse.y || v.impulse.z)))
            throw std::runtime_error("GPU contact report has invalid numerical point");
        points.push_back({Vector(v.pointDepth), n, Vector(v.relativeVelocity), Vector(v.impulse), v.feature.x});
    }
    return {a, b, trigger, offset, r.meta.z};
}

State::PairKey PairKey(const State::Manifold &manifold)
{
    const auto a = manifold.a.Key(), b = manifold.b.Key();
    return {std::min(a, b), std::max(a, b), manifold.trigger};
}
} // namespace

struct JoltGpuContactPublication::Pending
{
    PhysicsContactPublication::Policy policy;
    std::shared_ptr<JoltGpuContactReportResult> report;
};

JoltGpuScene::ContactObserver JoltGpuContactPublication::Prepare(JoltGpuContactReports &reports,
                                                                 const JoltGpuScene &scene, RuntimeWorld &world,
                                                                 const std::shared_ptr<const Publication> &publication,
                                                                 uint32_t substeps, State::Policy policy)
{
    auto &store = PhysicsECSStore::Instance();
    scene.VisitActorHandles(world, [&](PhysicsActorHandle handle) {
        if (store.IsValid(handle))
            policy.CaptureActor(world, store.GetActor(handle));
    });
    return Prepare(reports, publication, substeps, std::move(policy));
}

JoltGpuScene::ContactObserver JoltGpuContactPublication::Prepare(JoltGpuContactReports &reports,
                                                                 const std::shared_ptr<const Publication> &publication,
                                                                 uint32_t substeps, State::Policy policy)
{
    if (!policy.Requested())
        return {};
    auto pending = std::make_shared<Pending>();
    pending->policy = std::move(policy);
    auto capture =
        reports.Prepare(publication, substeps, Flags(pending->policy),
                        [pending](const Publication &accepted) { return LeafInterest(accepted, pending->policy); });
    pending->report = std::move(capture.result);
    m_pending = std::move(pending);
    return {std::move(capture.phase), std::move(capture.finish)};
}

void JoltGpuContactPublication::ValidateCompleted(const std::shared_ptr<const Publication> &publication,
                                                  uint32_t substeps) const
{
    if (!publication || !publication->isFixedStep)
        throw std::logic_error("GPU contact completion requires a Fixed publication");
    if (!m_pending)
        return;
    const auto &report = *m_pending->report;
    if (report.publication.get() != publication.get() || !report.complete || report.expectedSubsteps != substeps ||
        report.frames.size() != substeps)
        throw std::logic_error("GPU contact completion differs from its Fixed publication or substep count");
}

void JoltGpuContactPublication::AppendFrame(uint32_t substep, State::Frame &destination) const
{
    if (!m_pending)
        return;
    const auto &report = *m_pending->report;
    const auto &ordinary = report.frames.at(substep);
    const auto swept = report.sweepFrames.find(substep);
    size_t sweepRecords = 0;
    if (swept != report.sweepFrames.end())
        for (const auto &frame : swept->second)
            sweepRecords += frame.records.size();
    destination.manifolds.reserve(destination.manifolds.size() + ordinary.records.size() + sweepRecords);
    destination.points.reserve(destination.points.size() + ordinary.records.size() * MaxPatchPoints + sweepRecords);
    std::map<size_t, std::vector<State::Point>> extra;
    const auto append = [&](const auto &frame, std::map<State::PairKey, size_t> *existing) {
        if (frame.substep != substep || !frame.finished || !frame.seeded)
            throw std::logic_error("Contact frame sequence mismatch");
        for (const auto &record : frame.records) {
            if (record.meta.w >= frame.capacity)
                throw std::runtime_error("Contact source ordinal exceeds retained workset");
            auto manifold = DecodeManifold(*report.publication, record, substep, destination.points);
            if (existing) {
                const auto [found, inserted] = existing->emplace(PairKey(manifold), destination.manifolds.size());
                if (!inserted) {
                    const auto &prior = destination.manifolds[found->second];
                    auto point = destination.points.back();
                    destination.points.pop_back();
                    if (prior.a.Key() != manifold.a.Key()) {
                        point.normal = -point.normal;
                        point.relativeVelocity = -point.relativeVelocity;
                        point.impulse = -point.impulse;
                    }
                    extra[found->second].push_back(point);
                    continue;
                }
            }
            destination.manifolds.push_back(std::move(manifold));
        }
    };
    const auto ordinaryFirst = destination.manifolds.size();
    append(ordinary, nullptr);
    if (swept != report.sweepFrames.end()) {
        std::map<State::PairKey, size_t> existing;
        for (size_t i = ordinaryFirst; i < destination.manifolds.size(); ++i)
            existing.emplace(PairKey(destination.manifolds[i]), i);
        for (const auto &frame : swept->second)
            append(frame, &existing);
    }
    size_t joinedPoints = destination.points.size();
    for (const auto &[index, points] : extra)
        joinedPoints += destination.manifolds[index].pointCount + points.size();
    destination.points.reserve(joinedPoints);
    for (const auto &[index, points] : extra) {
        auto &manifold = destination.manifolds[index];
        const auto offset = destination.points.size();
        for (uint32_t i = 0; i < manifold.pointCount; ++i)
            destination.points.push_back(destination.points[manifold.pointOffset + i]);
        destination.points.insert(destination.points.end(), points.begin(), points.end());
        manifold.pointOffset = offset;
        manifold.pointCount += static_cast<uint32_t>(points.size());
    }
}

void JoltGpuContactPublication::AcceptCompleted(std::shared_ptr<const Publication> publication) noexcept
{
    m_pending.reset();
    m_publication = std::move(publication);
}

void JoltGpuContactPublication::Publish(PhysicsContactPublication &contacts,
                                        const std::shared_ptr<const Publication> &publication)
{
    const auto substeps = m_pending ? m_pending->report->expectedSubsteps : 0;
    ValidateCompleted(publication, substeps);
    const ContactStreamIdentity identity{publication->worldId, publication->simulationEpoch,
                                         publication->fixedStepIndex};
    try {
        if (m_pending) {
            contacts.Publish(identity, m_pending->policy, substeps, [&](uint32_t substep) {
                State::Frame decoded;
                AppendFrame(substep, decoded);
                return decoded;
            });
        } else {
            contacts.PublishEmpty(identity);
        }
    } catch (...) {
        CancelPending();
        throw;
    }
    AcceptCompleted(publication);
}

void JoltGpuContactPublication::CancelPending() noexcept
{
    m_pending.reset();
}

void JoltGpuContactPublication::Close() noexcept
{
    CancelPending();
    m_publication.reset();
}
} // namespace infernux
