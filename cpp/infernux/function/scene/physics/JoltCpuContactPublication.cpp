#include "JoltCpuContactPublication.h"
#include "JoltCpuScene.h"

#include <Jolt/Physics/Collision/CollideShape.h>
#include <Jolt/Physics/Collision/Shape/CompoundShape.h>
#include <Jolt/Physics/Collision/ShapeFilter.h>
#include <array>
#include <exception>
#include <mutex>
#include <stdexcept>

namespace infernux
{
namespace
{
using Reports = PhysicsContactPublication;
using NativePair = std::tuple<JPH::BodyID, JPH::BodyID, uint32_t, uint32_t>;
constexpr size_t MaxPoints = JPH::ContactConstraintManager::MaxContactPoints;
constexpr uint8_t TriggerInterest = TriggerEnterInterest | TriggerStayInterest | TriggerExitInterest;

template <class Vector> glm::vec3 ToGlm(const Vector &v)
{
    return {float(v.GetX()), float(v.GetY()), float(v.GetZ())};
}

JPH::SubShapeID SubShape(uint32_t value)
{
    JPH::SubShapeID id;
    id.SetValue(value);
    return id;
}

NativePair Key(const JPH::Body &a, JPH::SubShapeID sa, const JPH::Body &b, JPH::SubShapeID sb)
{
    if (a.GetID() < b.GetID())
        return {a.GetID(), b.GetID(), sa.GetValue(), sb.GetValue()};
    return {b.GetID(), a.GetID(), sb.GetValue(), sa.GetValue()};
}

struct Pose
{
    JPH::RefConst<JPH::Shape> shape;
    JPH::RVec3 center;
    JPH::Quat rotation;

    explicit Pose(const JPH::Body &body)
        : shape(body.GetShape()), center(body.GetCenterOfMassPosition()), rotation(body.GetRotation())
    {
    }

    bool Matches(const JPH::Body &body) const
    {
        return shape.GetPtr() == body.GetShape() && center == body.GetCenterOfMassPosition() &&
               rotation == body.GetRotation();
    }
};

struct RestingGeometry
{
    Pose a, b;
    float margin;
};

struct RetainedContact
{
    Reports::Manifold manifold;
    Reports::Point point;
    bool wasActive;
    std::optional<RestingGeometry> resting;
};

struct Sample
{
    NativePair key;
    size_t offset;
    uint32_t count;
    bool trigger;
    bool ccd;
};

class PairCollector final : public JPH::CollideShapeCollector
{
  public:
    PairCollector(JPH::SubShapeID a, JPH::SubShapeID b, float margin, bool exactLeaf)
        : m_a(a), m_b(b), m_margin(margin), m_exactLeaf(exactLeaf)
    {
    }

    void AddHit(const JPH::CollideShapeResult &hit) override
    {
        if (ShouldEarlyOut() || hit.mPenetrationDepth < -m_margin)
            return;
        if (hit.mSubShapeID1 == m_a && hit.mSubShapeID2 == m_b) {
            m_hit = hit;
            ForceEarlyOut();
        } else if (!m_exactLeaf &&
                   (!m_hit || hit.mPenetrationDepth > m_hit->mPenetrationDepth ||
                    (hit.mPenetrationDepth == m_hit->mPenetrationDepth && Leaves(hit) < Leaves(*m_hit)))) {
            m_hit = hit;
        }
    }

    std::optional<JPH::CollideShapeResult> m_hit;

  private:
    static std::pair<uint32_t, uint32_t> Leaves(const JPH::CollideShapeResult &hit)
    {
        return {hit.mSubShapeID1.GetValue(), hit.mSubShapeID2.GetValue()};
    }

    JPH::SubShapeID m_a, m_b;
    float m_margin;
    bool m_exactLeaf;
};

bool IsContainer(const JPH::Shape &shape)
{
    return shape.GetType() == JPH::EShapeType::Compound || shape.GetType() == JPH::EShapeType::Decorated;
}

template <class Accept> class PairShapeFilter final : public JPH::ShapeFilter
{
  public:
    PairShapeFilter(const JPH::SimShapeFilter &sim, const JPH::Body &a, const JPH::Body &b, Accept accept)
        : m_sim(sim), m_a(a), m_b(b), m_accept(accept)
    {
    }

    bool ShouldCollide(const JPH::Shape *a, const JPH::SubShapeID &sa, const JPH::Shape *b,
                       const JPH::SubShapeID &sb) const override
    {
        if (a->GetType() == JPH::EShapeType::Mesh && b->GetType() == JPH::EShapeType::Mesh)
            return false;
        return m_sim.ShouldCollide(m_a, a, sa, m_b, b, sb) && (IsContainer(*a) || IsContainer(*b) || m_accept(sa, sb));
    }

  private:
    const JPH::SimShapeFilter &m_sim;
    const JPH::Body &m_a, &m_b;
    const Accept m_accept;
};

template <class Record> class OverlapCollector final : public JPH::CollideShapeCollector
{
  public:
    explicit OverlapCollector(Record record) : m_record(record)
    {
    }

    void AddHit(const JPH::CollideShapeResult &hit) override
    {
        if (hit.mPenetrationDepth >= 0.0f)
            m_record(hit);
    }

  private:
    const Record m_record;
};

class ScopedListener final
{
  public:
    ScopedListener(JPH::PhysicsSystem &system, JPH::ContactListener &listener)
        : m_system(system), m_previous(system.GetContactListener())
    {
        m_system.SetContactListener(&listener);
    }

    ~ScopedListener()
    {
        m_system.SetContactListener(m_previous);
    }

  private:
    JPH::PhysicsSystem &m_system;
    JPH::ContactListener *m_previous;
};
} // namespace

struct JoltCpuContactPublication::Capture final : JPH::ContactListener
{
    void Begin(JoltCpuScene &scene, Reports::Policy nextPolicy, uint32_t substeps, bool sameEpoch,
               std::function<void(uint32_t)> afterSubstep)
    {
        m_scene = &scene;
        m_material = scene.System().GetContactListener();
        m_policy = std::move(nextPolicy);
        m_afterSubstep = std::move(afterSubstep);
        m_streamTriggers = m_policy.events && m_policy.triggers;
        m_observeTriggers = m_policy.WantsTriggers();
        m_failure = nullptr;
        m_frames.clear();
        m_frames.reserve(substeps);
        m_samples.clear();
        m_points.clear();
        m_previous = sameEpoch ? m_history : decltype(m_history){};
        for (auto &[key, retained] : m_previous) {
            const auto [a, b] = FindBodies(key);
            retained.wasActive = a && b && (a->IsActive() || b->IsActive());
        }
    }

    bool WantsContactStepReports() const noexcept override
    {
        return true;
    }

    bool WantsContactStepImpulses() const noexcept override
    {
        return m_policy.impulses;
    }

    JPH::ValidateResult OnContactValidate(const JPH::Body &a, const JPH::Body &b, JPH::RVec3Arg base,
                                          const JPH::CollideShapeResult &hit) override
    {
        return m_material->OnContactValidate(a, b, base, hit);
    }

    JPH::ValidateResult OnCCDContactValidate(const JPH::Body &a, const JPH::Body &b, JPH::RVec3Arg base,
                                             const JPH::CollideShapeResult &hit) override
    {
        return m_material->OnCCDContactValidate(a, b, base, hit);
    }

    void OnContactAdded(const JPH::Body &a, const JPH::Body &b, const JPH::ContactManifold &manifold,
                        JPH::ContactSettings &settings) override
    {
        m_material->OnContactAdded(a, b, manifold, settings);
        RecordDiscrete(a, b, manifold, settings.mIsSensor);
    }

    void OnContactPersisted(const JPH::Body &a, const JPH::Body &b, const JPH::ContactManifold &manifold,
                            JPH::ContactSettings &settings) override
    {
        m_material->OnContactPersisted(a, b, manifold, settings);
        RecordDiscrete(a, b, manifold, settings.mIsSensor);
    }

    void OnContactRemoved(const JPH::SubShapeIDPair &pair) override
    {
        m_material->OnContactRemoved(pair);
    }

    void OnContactPairObserved(const JPH::Body &a, const JPH::Body &b) noexcept override
    {
        if (!m_observeTriggers)
            return;
        const auto first = SummarizeTriggers(Record(a)), second = SummarizeTriggers(Record(b));
        if (!((first.sensor && second.interested) || (second.sensor && first.interested) || first.interestedSensor ||
              second.interestedSensor))
            return;
        // The same zero-separation query supplies fresh and cached overlaps.
        // A reduced or empty solver cache cannot represent every trigger leaf.
        const auto &system = m_scene->System();
        const PairShapeFilter filter(*system.GetSimShapeFilter(), a, b,
                                     [&](auto sa, auto sb) { return WantsTrigger(a, sa, b, sb); });
        const float sign = a.GetID() < b.GetID() ? 1.0f : -1.0f;
        const auto base = a.GetCenterOfMassPosition();
        OverlapCollector collector([&](const JPH::CollideShapeResult &hit) {
            const auto sa = hit.mSubShapeID1, sb = hit.mSubShapeID2;
            const auto p = base + 0.5f * (hit.mContactPointOn1 + hit.mContactPointOn2);
            const auto normal = hit.mPenetrationAxis.Normalized();
            const Reports::Point point{
                ToGlm(p), sign * ToGlm(normal), sign * ToGlm(b.GetPointVelocity(p) - a.GetPointVelocity(p)), {}, 0,
                false};
            Append(Key(a, sa, b, sb), &point, 1, true, false);
        });
        try {
            Collide(a, b, base, 0.0f, collector, filter);
        } catch (...) {
            const std::scoped_lock lock(m_samplesMutex);
            m_failure = std::current_exception();
        }
    }

    void OnCCDContactSolved(uint32_t, const JPH::Body &a, const JPH::SubShapeID &sa, const JPH::Body &b,
                            const JPH::SubShapeID &sb, JPH::RVec3Arg position, JPH::Vec3Arg normal,
                            JPH::Vec3Arg relativeVelocity, JPH::Vec3Arg impulse) noexcept override
    {
        if (!Observes(a, sa, b, sb, false))
            return;
        const float sign = a.GetID() < b.GetID() ? 1.0f : -1.0f;
        const Reports::Point point{
            ToGlm(position),  sign * ToGlm(normal), sign * ToGlm(relativeVelocity), sign * ToGlm(impulse), 0,
            m_policy.impulses};
        Append(Key(a, sa, b, sb), &point, 1, false, true);
    }

    void OnContactStepCompleted(uint32_t substep, const JPH::PhysicsSystem &system) noexcept override
    {
        if (m_failure)
            return;
        try {
            if (substep != m_frames.size())
                throw std::logic_error("CPU contact substep sequence mismatch");
            m_frames.push_back(BuildFrame(system));
            if (m_afterSubstep)
                m_afterSubstep(substep);
        } catch (...) {
            m_failure = std::current_exception();
        }
    }

    void Finish(uint32_t substeps, const JPH::PhysicsSystem &system)
    {
        // An idle Update bypasses the job graph and produces no native reports.
        if (m_frames.empty() && !m_failure)
            for (uint32_t step = 0; step < substeps; ++step)
                OnContactStepCompleted(step, system);
        m_afterSubstep = {};
        if (m_failure)
            std::rethrow_exception(m_failure);
        if (m_frames.size() != substeps)
            throw std::logic_error("Incomplete CPU contact Fixed");
    }

    const Reports::Policy &Policy() const noexcept
    {
        return m_policy;
    }

    uint32_t FrameCount() const noexcept
    {
        return static_cast<uint32_t>(m_frames.size());
    }

    Reports::Frame TakeFrame(uint32_t substep)
    {
        return std::move(m_frames.at(substep));
    }

    void AcceptHistory() noexcept
    {
        m_history = std::move(m_previous);
        m_frames.clear();
    }

    void DiscardFrames() noexcept
    {
        m_previous.clear();
        m_frames.clear();
        m_afterSubstep = {};
    }

  private:
    const JoltCpuScene::BodyRecord &Record(const JPH::Body &body) const
    {
        return m_scene->m_bodies.at(static_cast<uint32_t>(body.GetUserData()));
    }

    const JoltCpuScene::Surface &Surface(const JPH::Body &body, JPH::SubShapeID subshape) const
    {
        return m_scene->GetSurface(static_cast<uint32_t>(body.GetUserData()), subshape);
    }

    static Reports::Surface SurfaceKey(const JoltCpuScene::Surface &surface)
    {
        return {surface.colliderId, surface.generation};
    }

    static const JoltCpuScene::Surface *FindSurface(const JoltCpuScene::BodyRecord &record, Reports::Surface key)
    {
        const auto found = std::find_if(record.surfaces.begin(), record.surfaces.end(),
                                        [&](const auto &surface) { return SurfaceKey(surface) == key; });
        return found != record.surfaces.end() ? &*found : nullptr;
    }

    struct TriggerSummary
    {
        bool sensor = false, interested = false, interestedSensor = false;
    };

    TriggerSummary SummarizeTriggers(const JoltCpuScene::BodyRecord &record) const
    {
        TriggerSummary result;
        for (const auto &surface : record.surfaces) {
            const bool observes = m_streamTriggers || (m_policy.Interest(SurfaceKey(surface)) & TriggerInterest) != 0;
            result.sensor |= surface.sensor;
            result.interested |= observes;
            result.interestedSensor |= surface.sensor && observes;
        }
        return result;
    }

    bool Observes(const JPH::Body &a, JPH::SubShapeID sa, const JPH::Body &b, JPH::SubShapeID sb, bool trigger) const
    {
        const auto &first = Surface(a, sa), &second = Surface(b, sb);
        return m_policy.Observes(SurfaceKey(first), SurfaceKey(second), trigger);
    }

    bool WantsTrigger(const JPH::Body &a, JPH::SubShapeID sa, const JPH::Body &b, JPH::SubShapeID sb) const
    {
        const auto &first = Surface(a, sa), &second = Surface(b, sb);
        return (first.sensor || second.sensor) && m_policy.Observes(SurfaceKey(first), SurfaceKey(second), true);
    }

    Reports::Endpoint Endpoint(const JPH::Body &body, JPH::SubShapeID subshape) const
    {
        const auto &surface = Surface(body, subshape);
        return {static_cast<uint32_t>(body.GetUserData()),
                subshape.GetValue(),
                surface.colliderId,
                surface.generation,
                surface.target,
                Record(body).bodyTarget};
    }

    std::pair<const JPH::Body *, const JPH::Body *> FindBodies(const NativePair &key) const
    {
        const auto &bodies = m_scene->System().GetBodyLockInterfaceNoLock();
        return {bodies.TryGetBody(std::get<0>(key)), bodies.TryGetBody(std::get<1>(key))};
    }

    bool Recheck(const JPH::Body &first, const JPH::Body &second, float margin, NativePair &key,
                 RetainedContact &contact) const
    {
        // Use the simulation's ordering and relative COM frame for the narrowphase.
        const bool reverse = first.GetMotionType() < second.GetMotionType();
        const auto &a = reverse ? second : first, &b = reverse ? first : second;
        const auto &system = m_scene->System();
        auto &manifold = contact.manifold;
        const auto &oldA = reverse ? manifold.b : manifold.a, &oldB = reverse ? manifold.a : manifold.b;
        const auto surfaceA = oldA.SurfaceKey(), surfaceB = oldB.SurfaceKey();
        const PairShapeFilter filter(*system.GetSimShapeFilter(), a, b, [&](auto hitA, auto hitB) {
            return SurfaceKey(Surface(a, hitA)) == surfaceA && SurfaceKey(Surface(b, hitB)) == surfaceB;
        });
        const auto base = a.GetCenterOfMassPosition();
        // Trigger reports track exact leaves. A solver-reduced solid manifold
        // may be represented by another leaf of the same Collider after sleep.
        PairCollector hits(SubShape(oldA.subshape), SubShape(oldB.subshape), margin, manifold.trigger);
        Collide(a, b, base, margin, hits, filter);
        if (!hits.m_hit)
            return false;
        const auto &hit = *hits.m_hit;
        const auto p = base + 0.5f * (hit.mContactPointOn1 + hit.mContactPointOn2);
        const auto axis = hit.mPenetrationAxis.Normalized();
        contact.point = {ToGlm(p),
                         (reverse ? -1.0f : 1.0f) * ToGlm(axis),
                         ToGlm(second.GetPointVelocity(p) - first.GetPointVelocity(p)),
                         {},
                         0,
                         false};
        key = Key(a, hit.mSubShapeID1, b, hit.mSubShapeID2);
        manifold.a = Endpoint(first, reverse ? hit.mSubShapeID2 : hit.mSubShapeID1);
        manifold.b = Endpoint(second, reverse ? hit.mSubShapeID1 : hit.mSubShapeID2);
        return true;
    }

    void Collide(const JPH::Body &a, const JPH::Body &b, JPH::RVec3Arg base, float margin,
                 JPH::CollideShapeCollector &collector, const JPH::ShapeFilter &filter) const
    {
        const auto &system = m_scene->System();
        const auto &settings = system.GetPhysicsSettings();
        JPH::CollideShapeSettings collision;
        collision.mMaxSeparationDistance = margin;
        collision.mCollectFacesMode = JPH::ECollectFacesMode::CollectFaces;
        collision.mActiveEdgeMode = settings.mCheckActiveEdges ? JPH::EActiveEdgeMode::CollideOnlyWithActive
                                                               : JPH::EActiveEdgeMode::CollideWithAll;
        collision.mActiveEdgeMovementDirection = a.GetLinearVelocity() - b.GetLinearVelocity();
        collision.mInternalEdgeRemovalVertexToleranceSq = settings.mInternalEdgeRemovalVertexToleranceSq;
        const auto transformA = JPH::Mat44::sRotation(a.GetRotation());
        const auto transformB = b.GetCenterOfMassTransform().PostTranslated(-base).ToMat44();
        system.GetSimCollideBodyVsBody()(a, b, transformA, transformB, collision, collector, filter);
    }

    void RecordDiscrete(const JPH::Body &a, const JPH::Body &b, const JPH::ContactManifold &manifold,
                        bool trigger) noexcept
    {
        if (trigger || manifold.mIsCCD)
            return;
        const auto sa = manifold.mSubShapeID1, sb = manifold.mSubShapeID2;
        if (!Observes(a, sa, b, sb, false))
            return;
        const float sign = a.GetID() < b.GetID() ? 1.0f : -1.0f;
        const uint32_t count = m_policy.impulses ? static_cast<uint32_t>(manifold.mRelativeContactPointsOn1.size()) : 1;
        std::array<Reports::Point, MaxPoints> points;
        for (uint32_t i = 0; i < count; ++i) {
            const auto p = manifold.mBaseOffset +
                           0.5f * (manifold.mRelativeContactPointsOn1[i] + manifold.mRelativeContactPointsOn2[i]);
            points[i] = {ToGlm(p),
                         sign * ToGlm(manifold.mWorldSpaceNormal),
                         sign * ToGlm(b.GetPointVelocity(p) - a.GetPointVelocity(p)),
                         {},
                         i,
                         false};
        }
        Append(Key(a, sa, b, sb), points.data(), count, false, false);
    }

    void Append(const NativePair &key, const Reports::Point *points, uint32_t count, bool trigger, bool ccd) noexcept
    {
        const std::scoped_lock lock(m_samplesMutex);
        if (m_failure)
            return;
        try {
            const auto offset = m_points.size();
            m_points.insert(m_points.end(), points, points + count);
            m_samples.push_back({key, offset, count, trigger, ccd});
        } catch (...) {
            // The observed leaf fanout can exceed native solver capacity. Keep
            // scratch high-water capacity and discard this Fixed on allocation failure.
            m_failure = std::current_exception();
        }
    }

    void AttachImpulses(const JPH::PhysicsSystem &system)
    {
        for (const auto &row : system.GetAppliedContactImpulses()) {
            const NativePair key{JPH::BodyID(row.body1ID), JPH::BodyID(row.body2ID), row.subShapeID1, row.subShapeID2};
            const auto sample = std::lower_bound(m_samples.begin(), m_samples.end(), key,
                                                 [](const Sample &s, const NativePair &k) { return s.key < k; });
            // Native recording also includes rows of uninterested pairs.
            if (sample == m_samples.end() || sample->key != key || sample->ccd)
                continue;
            if (row.pointIndex >= sample->count || sample->trigger)
                throw std::logic_error("CPU contact impulse row differs from its manifold");
            auto &point = m_points[sample->offset + row.pointIndex];
            if (point.hasSolvedImpulse)
                throw std::logic_error("Duplicate CPU contact impulse row");
            point.impulse = {row.impulse.x, row.impulse.y, row.impulse.z};
            point.hasSolvedImpulse = true;
        }
    }

    bool Retain(NativePair &key, RetainedContact &contact)
    {
        const auto [a, b] = FindBodies(key);
        if (!a || !b || !a->IsInBroadPhase() || !b->IsInBroadPhase())
            return false;
        const bool active = a->IsActive() || b->IsActive();
        if (contact.wasActive && active)
            return false;
        const auto &system = m_scene->System();
        if (!system.GetObjectLayerPairFilter().ShouldCollide(a->GetObjectLayer(), b->GetObjectLayer()))
            return false;
        const auto &recordA = Record(*a), &recordB = Record(*b);
        const auto &old = contact.manifold;
        const auto *first = FindSurface(recordA, old.a.SurfaceKey());
        const auto *second = FindSurface(recordB, old.b.SurfaceKey());
        if (a->GetUserData() != old.a.body || b->GetUserData() != old.b.body || !first || !second ||
            (first->sensor || second->sensor) != old.trigger ||
            (!old.trigger && !recordA.authoredDynamic && !recordB.authoredDynamic) ||
            !m_scene->SurfacesCollide(*first, *second) ||
            !m_policy.Observes(old.a.SurfaceKey(), old.b.SurfaceKey(), old.trigger))
            return false;
        const float margin = old.trigger ? 0.0f : system.GetPhysicsSettings().mSpeculativeContactDistance;
        if (!contact.resting || contact.resting->margin != margin || !contact.resting->a.Matches(*a) ||
            !contact.resting->b.Matches(*b) || active || contact.wasActive) {
            if (!Recheck(*a, *b, margin, key, contact))
                return false;
            contact.resting = RestingGeometry{Pose(*a), Pose(*b), margin};
        }
        contact.wasActive = active;
        return true;
    }

    Reports::Frame BuildFrame(const JPH::PhysicsSystem &system)
    {
        std::sort(m_samples.begin(), m_samples.end(), [](const Sample &a, const Sample &b) {
            return std::tie(a.key, a.ccd, a.offset) < std::tie(b.key, b.ccd, b.offset);
        });
        if (m_policy.impulses)
            AttachImpulses(system);

        Reports::Frame frame;
        frame.manifolds.reserve(m_samples.size() + m_previous.size());
        frame.points.reserve(m_points.size() + m_previous.size());
        std::map<NativePair, RetainedContact> next;
        for (auto sample = m_samples.begin(); sample != m_samples.end();) {
            const auto key = sample->key;
            const auto [a, b] = FindBodies(key);
            Reports::Manifold manifold{Endpoint(*a, SubShape(std::get<2>(key))),
                                       Endpoint(*b, SubShape(std::get<3>(key))), sample->trigger, frame.points.size(),
                                       0};
            do {
                if (manifold.pointCount && (!sample->ccd || manifold.trigger))
                    throw std::logic_error("Duplicate CPU contact manifold");
                for (uint32_t i = 0; i < sample->count; ++i) {
                    auto point = m_points[sample->offset + i];
                    point.feature = manifold.pointCount++;
                    frame.points.push_back(point);
                }
                ++sample;
            } while (sample != m_samples.end() && sample->key == key);
            frame.manifolds.push_back(manifold);
            next.emplace(key, RetainedContact{manifold, frame.points[manifold.pointOffset],
                                              a->IsActive() || b->IsActive(), std::nullopt});
        }
        for (auto &[oldKey, retained] : m_previous) {
            auto key = oldKey;
            if (next.count(key) || !Retain(key, retained) || next.count(key))
                continue;
            retained.manifold.pointOffset = frame.points.size();
            retained.manifold.pointCount = 1;
            frame.manifolds.push_back(retained.manifold);
            frame.points.push_back(retained.point);
            next.emplace(key, std::move(retained));
        }
        m_previous = std::move(next);
        m_samples.clear();
        m_points.clear();
        return frame;
    }

    JoltCpuScene *m_scene = nullptr;
    JPH::ContactListener *m_material = nullptr;
    Reports::Policy m_policy;
    std::map<NativePair, RetainedContact> m_history, m_previous;
    std::vector<Reports::Frame> m_frames;
    std::vector<Sample> m_samples;
    std::vector<Reports::Point> m_points;
    std::mutex m_samplesMutex;
    std::exception_ptr m_failure;
    std::function<void(uint32_t)> m_afterSubstep;
    bool m_streamTriggers = false, m_observeTriggers = false;
};

struct JoltCpuContactPublication::CoupledStep
{
    CoupledStep(JoltCpuExecution &shared, JoltCpuScene &owner, float fixedDelta, uint32_t count,
                ContactStreamIdentity stepIdentity)
        : execution(shared), scene(owner), time(fixedDelta), substeps(count), identity(stepIdentity)
    {
    }

    ~CoupledStep()
    {
        context.reset();
        if (workspace)
            execution.ReleaseCoupledWorkspace(std::move(workspace));
    }

    JoltCpuExecution &execution;
    JoltCpuScene &scene;
    float time;
    uint32_t substeps;
    ContactStreamIdentity identity;
    std::unique_ptr<JoltCpuExecution::CoupledWorkspace> workspace;
    std::optional<JPH::PhysicsUpdateContext> context;
    std::optional<ScopedListener> listener;
};

JoltCpuContactPublication::JoltCpuContactPublication() = default;
JoltCpuContactPublication::~JoltCpuContactPublication()
{
    CancelCoupled();
}

void JoltCpuContactPublication::PrepareObservation(JoltCpuScene &scene, uint32_t substeps,
                                                   ContactStreamIdentity identity, Reports::Policy policy,
                                                   std::function<void(uint32_t)> afterSubstep)
{
    m_ready.reset();
    m_observing = policy.Requested() || bool(afterSubstep);
    scene.MarkActivePoseUpdates();
    if (m_observing) {
        if (!m_capture)
            m_capture = std::make_unique<Capture>();
        const bool sameEpoch = m_published && m_published->worldId == identity.worldId &&
                               m_published->simulationEpoch == identity.simulationEpoch;
        m_capture->Begin(scene, std::move(policy), substeps, sameEpoch, std::move(afterSubstep));
    }
}

void JoltCpuContactPublication::FinishObservation(JoltCpuScene &scene, uint32_t substeps,
                                                  ContactStreamIdentity identity)
{
    if (m_observing)
        m_capture->Finish(substeps, scene.System());
    scene.MarkActivePoseUpdates();
    m_ready = identity;
}

JPH::EPhysicsUpdateError JoltCpuContactPublication::Step(JoltCpuExecution &execution, JoltCpuScene &scene,
                                                         float deltaTime, uint32_t substeps,
                                                         ContactStreamIdentity identity, Reports::Policy policy,
                                                         std::function<void(uint32_t)> afterSubstep)
{
    PrepareObservation(scene, substeps, identity, std::move(policy), std::move(afterSubstep));
    JPH::EPhysicsUpdateError result;
    {
        std::optional<ScopedListener> listener;
        if (m_observing)
            listener.emplace(scene.System(), *m_capture);
        result = execution.Step(scene.System(), deltaTime, static_cast<int>(substeps));
    }
    if (result == JPH::EPhysicsUpdateError::None)
        FinishObservation(scene, substeps, identity);
    return result;
}

void JoltCpuContactPublication::ConfigureCoupledFixed(JoltCpuExecution &execution, JoltCpuScene &scene, float deltaTime,
                                                      uint32_t substeps, ContactStreamIdentity identity,
                                                      Reports::Policy policy)
{
    PrepareObservation(scene, substeps, identity, std::move(policy), {});
    m_coupled = std::make_unique<CoupledStep>(execution, scene, deltaTime, substeps, identity);
    if (m_observing)
        m_coupled->listener.emplace(scene.System(), *m_capture);
}

void JoltCpuContactPublication::BeginCoupledSubstep(uint32_t substep, const std::vector<JPH::BodyID> &endpoints)
{
    auto &fixed = *m_coupled;
    auto &system = fixed.scene.System();
    bool admitted = false;
    if (substep == 0) {
        fixed.workspace = fixed.execution.AcquireCoupledWorkspace();
        auto &context = fixed.context.emplace(fixed.workspace->scratch);
        admitted =
            system.BeginCoupledVelocityUpdate(fixed.time, static_cast<int>(fixed.substeps), endpoints.data(),
                                              static_cast<uint32_t>(endpoints.size()), context, fixed.workspace->jobs);
    } else {
        admitted =
            system.BeginNextCoupledSubstep(*fixed.context, endpoints.data(), static_cast<uint32_t>(endpoints.size()));
    }
    if (!admitted)
        throw std::logic_error("CPU coupling endpoints differ from the admitted Fixed");
}

bool JoltCpuContactPublication::AdvanceCoupled(const std::vector<JPH::PhysicsSystem::CoupledBodyImpulse> &impulses)
{
    auto &fixed = *m_coupled;
    auto &context = *fixed.context;
    auto &system = fixed.scene.System();
    const auto result =
        system.AdvanceCoupledVelocityUpdate(context, impulses.data(), static_cast<uint32_t>(impulses.size()));
    if (!result)
        return context.mAwaitingCoupledEndpoints;
    fixed.listener.reset();
    if (*result != JPH::EPhysicsUpdateError::None)
        throw std::runtime_error("CPU physics exceeded its configured native capacity");
    FinishObservation(fixed.scene, fixed.substeps, fixed.identity);
    m_coupled.reset();
    return true;
}

JPH::PhysicsUpdateContext::CoupledFreeVelocity JoltCpuContactPublication::CoupledFreeVelocity(JPH::BodyID body) const
{
    const auto &context = *m_coupled->context;
    const auto &bodies = context.mCoupledBodies;
    const auto found = std::lower_bound(bodies.begin(), bodies.end(), body);
    return context.mCoupledFreeVelocities[found - bodies.begin()];
}

void JoltCpuContactPublication::CancelCoupled() noexcept
{
    if (!m_coupled)
        return;
    if (m_coupled->context) {
        auto &context = *m_coupled->context;
        auto &system = m_coupled->scene.System();
        while (context.mCoupledUpdate) {
            if (context.mAwaitingCoupledEndpoints)
                system.BeginNextCoupledSubstep(context, nullptr, 0);
            system.AdvanceCoupledVelocityUpdate(context);
        }
    }
    m_coupled.reset();
}

void JoltCpuContactPublication::ValidateCompleted(ContactStreamIdentity identity, uint32_t substeps) const
{
    if (!m_ready || m_ready->worldId != identity.worldId || m_ready->simulationEpoch != identity.simulationEpoch ||
        m_ready->fixedStepIndex != identity.fixedStepIndex || (m_observing && m_capture->FrameCount() != substeps))
        throw std::logic_error("CPU contact completion differs from its Fixed identity or substep count");
}

Reports::Frame JoltCpuContactPublication::TakeFrame(uint32_t substep)
{
    return m_observing ? m_capture->TakeFrame(substep) : Reports::Frame{};
}

void JoltCpuContactPublication::AcceptCompleted() noexcept
{
    if (m_observing)
        m_capture->AcceptHistory();
    else
        m_capture.reset();
    m_published = std::exchange(m_ready, std::nullopt);
}

void JoltCpuContactPublication::Publish(Reports &publication)
{
    if (!m_ready)
        throw std::logic_error("CPU contact publication requires a completed Fixed");
    const auto identity = *m_ready;
    try {
        if (m_observing) {
            publication.Publish(identity, m_capture->Policy(), m_capture->FrameCount(),
                                [&](uint32_t substep) { return TakeFrame(substep); });
        } else {
            publication.PublishEmpty(identity);
        }
    } catch (...) {
        CancelPending();
        throw;
    }
    AcceptCompleted();
}

void JoltCpuContactPublication::CancelPending() noexcept
{
    CancelCoupled();
    m_ready.reset();
    if (m_capture)
        m_capture->DiscardFrames();
}

void JoltCpuContactPublication::Close() noexcept
{
    CancelCoupled();
    m_ready.reset();
    m_published.reset();
    m_capture.reset();
}
} // namespace infernux
