#include "JoltCpuScene.h"
#include "../Rigidbody.h"

#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/CollideShape.h>
#include <Jolt/Physics/Collision/Shape/CompoundShape.h>
#include <algorithm>
#include <core/config/EngineConfig.h>
#include <iterator>
#include <set>
#include <stdexcept>

namespace infernux
{
JoltCpuExecution::JoltCpuExecution(const EngineConfig &config)
    : m_scratch(static_cast<JPH::uint>(config.physicsTempAllocatorSize)), m_inlineJobs(config.physicsMaxJobs),
      m_workerJobs(config.physicsMaxJobs, config.physicsMaxBarriers, config.physicsMaxConcurrency)
{
}

JPH::EPhysicsUpdateError JoltCpuExecution::Step(JPH::PhysicsSystem &system, float deltaTime, int collisionSteps)
{
    auto &jobs = JobSystem::Get();
    JPH::EPhysicsUpdateError result;
    // Tiny rigid scenes have no useful range parallelism. Execute the same
    // Jolt job graph locally instead of paying shared-pool submission per job.
    // Soft-body work is not estimated by rigid body count.
    constexpr uint32_t inlineRigidBodyLimit = 128;
    const bool inlineStep = m_workerJobs.GetMaxConcurrency() == 1 ||
                            (system.GetNumActiveBodies(JPH::EBodyType::RigidBody) <= inlineRigidBodyLimit &&
                             system.GetNumActiveBodies(JPH::EBodyType::SoftBody) == 0);
    if (inlineStep) {
        jobs.RunInDomain(JobDomain::Physics,
                         [&] { result = system.Update(deltaTime, collisionSteps, &m_scratch, &m_inlineJobs); });
        return result;
    }

    auto group = jobs.CreateTaskGroup(JobDomain::Physics, JobPriority::Critical);
    m_workerJobs.BeginFrame(group);
    jobs.RunInDomain(JobDomain::Physics,
                     [&] { result = system.Update(deltaTime, collisionSteps, &m_scratch, &m_workerJobs); });
    m_workerJobs.EndFrame();
    group.Close();
    jobs.Wait(group);
    return result;
}

JoltCpuScene::JoltCpuScene(const EngineConfig &config, const JPH::BroadPhaseLayerInterface &layers,
                           const JPH::ObjectVsBroadPhaseLayerFilter &broadphaseFilter,
                           const JPH::ObjectLayerPairFilter &pairFilter)
{
    m_system.Init(config.physicsMaxBodies, 0, config.physicsMaxBodyPairs, config.physicsMaxContactConstraints, layers,
                  broadphaseFilter, pairFilter);
    JPH::PhysicsSettings settings;
    settings.mNumVelocitySteps = config.physicsVelocitySteps;
    settings.mNumPositionSteps = config.physicsPositionSteps;
    settings.mPenetrationSlop = config.physicsPenetrationSlop;
    settings.mSpeculativeContactDistance = config.physicsSpeculativeContactDistance;
    settings.mLinearCastMaxPenetration = config.physicsLinearCastMaxPenetration;
    settings.mLinearCastThreshold = config.physicsLinearCastThreshold;
    settings.mBaumgarte = config.physicsBaumgarte;
    settings.mMaxPenetrationDistance = config.physicsMaxPenetrationDistance;
    settings.mMinVelocityForRestitution = config.physicsMinVelocityForRestitution;
    settings.mTimeBeforeSleep = config.physicsTimeBeforeSleep;
    settings.mPointVelocitySleepThreshold = config.physicsPointVelocitySleepThreshold;
    m_system.SetPhysicsSettings(settings);
    const auto gravity = config.physicsGravity;
    m_system.SetGravity(JPH::Vec3(gravity.x, gravity.y, gravity.z));
    m_system.SetContactListener(this);
}

JoltCpuScene::~JoltCpuScene()
{
    m_system.SetContactListener(nullptr);
}

JPH::BodyID JoltCpuScene::CreateBody(uint32_t bodyId, const JPH::BodyCreationSettings &settings, BodyRecord record)
{
    auto [entry, inserted] = m_bodies.try_emplace(bodyId, std::move(record));
    if (!inserted)
        throw std::logic_error("CPU physics body identity is already owned");
    auto *body = m_system.GetBodyInterface().CreateBody(settings);
    if (!body) {
        m_bodies.erase(entry);
        throw std::length_error("CPU physics body capacity exhausted");
    }
    body->SetUserData(bodyId);
    entry->second.native = body->GetID();
    return entry->second.native;
}

void JoltCpuScene::DestroyBody(uint32_t bodyId)
{
    const auto native = NativeBody(bodyId);
    auto &bodies = m_system.GetBodyInterface();
    if (bodies.IsAdded(native))
        bodies.RemoveBody(native);
    bodies.DestroyBody(native);
    m_bodies.erase(bodyId);
}

JPH::BodyID JoltCpuScene::NativeBody(uint32_t bodyId) const
{
    return m_bodies.at(bodyId).native;
}

const JoltCpuScene::Surface &JoltCpuScene::GetSurface(uint32_t bodyId, JPH::SubShapeID subshape) const
{
    const auto &record = m_bodies.at(bodyId);
    uint32_t member = 0;
    if (record.memberCompound) {
        JPH::SubShapeID remainder;
        const auto index = record.memberCompound->GetSubShapeIndexFromID(subshape, remainder);
        member = record.memberCompound->GetCompoundUserData(index);
    }
    return record.surfaces[member];
}

void JoltCpuScene::SetIgnoredPairs(std::shared_ptr<const PhysicsColliderPairs> pairs)
{
    if (pairs == m_ignoredPairs)
        return;
    static const PhysicsColliderPairs empty;
    const auto &before = m_ignoredPairs ? *m_ignoredPairs : empty;
    const auto &after = pairs ? *pairs : empty;
    PhysicsColliderPairs changes;
    std::set_symmetric_difference(before.begin(), before.end(), after.begin(), after.end(),
                                  std::back_inserter(changes));
    std::set<std::array<uint64_t, 2>> changed;
    for (const auto &pair : changes) {
        changed.insert({pair[0], pair[1]});
        changed.insert({pair[2], pair[3]});
    }
    m_ignoredPairs = std::move(pairs);
    if (changed.empty())
        return;
    auto &bodies = m_system.GetBodyInterface();
    for (const auto &[id, record] : m_bodies) {
        const bool affected = std::any_of(record.surfaces.begin(), record.surfaces.end(), [&](const Surface &surface) {
            return changed.count({surface.colliderId, surface.generation}) != 0;
        });
        if (affected) {
            bodies.InvalidateContactCache(record.native);
            if (bodies.IsAdded(record.native))
                bodies.ActivateBody(record.native);
        }
    }
}

JPH::ValidateResult JoltCpuScene::OnContactValidate(const JPH::Body &a, const JPH::Body &b, JPH::RVec3Arg,
                                                    const JPH::CollideShapeResult &hit)
{
    const auto &surfaceA = GetSurface(static_cast<uint32_t>(a.GetUserData()), hit.mSubShapeID1);
    const auto &surfaceB = GetSurface(static_cast<uint32_t>(b.GetUserData()), hit.mSubShapeID2);
    if ((surfaceA.collisionMask & (1u << surfaceB.layer)) == 0 ||
        (surfaceB.collisionMask & (1u << surfaceA.layer)) == 0)
        return JPH::ValidateResult::RejectContact;
    if (m_ignoredPairs) {
        std::array<uint64_t, 2> first{surfaceA.colliderId, surfaceA.generation},
            second{surfaceB.colliderId, surfaceB.generation};
        if (second < first)
            std::swap(first, second);
        const PhysicsColliderPair key{first[0], first[1], second[0], second[1]};
        if (std::binary_search(m_ignoredPairs->begin(), m_ignoredPairs->end(), key))
            return JPH::ValidateResult::RejectContact;
    }
    // Acceptance is per leaf: accepting the whole body pair bypasses filters
    // on the other Colliders of a compound.
    return JPH::ValidateResult::AcceptContact;
}

JPH::ValidateResult JoltCpuScene::OnCCDContactValidate(const JPH::Body &a, const JPH::Body &b, JPH::RVec3Arg origin,
                                                       const JPH::CollideShapeResult &hit)
{
    if (a.IsDynamic() && b.IsDynamic() &&
        m_bodies.at(static_cast<uint32_t>(a.GetUserData())).collisionDetectionMode !=
            static_cast<int>(CollisionDetectionMode::ContinuousDynamic) &&
        m_bodies.at(static_cast<uint32_t>(b.GetUserData())).collisionDetectionMode !=
            static_cast<int>(CollisionDetectionMode::ContinuousDynamic))
        return JPH::ValidateResult::RejectContact;
    const auto &surfaceA = GetSurface(static_cast<uint32_t>(a.GetUserData()), hit.mSubShapeID1);
    const auto &surfaceB = GetSurface(static_cast<uint32_t>(b.GetUserData()), hit.mSubShapeID2);
    if (surfaceA.sensor || surfaceB.sensor)
        return JPH::ValidateResult::RejectContact;
    return OnContactValidate(a, b, origin, hit);
}

namespace
{
float CombineSurface(float a, float b, PhysicsMaterialCombine modeA, PhysicsMaterialCombine modeB)
{
    switch (std::max(modeA, modeB)) {
    case PhysicsMaterialCombine::Minimum:
        return std::min(a, b);
    case PhysicsMaterialCombine::Multiply:
        return a * b;
    case PhysicsMaterialCombine::Maximum:
        return std::max(a, b);
    case PhysicsMaterialCombine::Average:
        break;
    }
    return 0.5f * (a + b);
}
} // namespace

void JoltCpuScene::OnContactAdded(const JPH::Body &a, const JPH::Body &b, const JPH::ContactManifold &manifold,
                                  JPH::ContactSettings &settings)
{
    const auto &surfaceA = GetSurface(static_cast<uint32_t>(a.GetUserData()), manifold.mSubShapeID1);
    const auto &surfaceB = GetSurface(static_cast<uint32_t>(b.GetUserData()), manifold.mSubShapeID2);
    settings.mIsSensor = surfaceA.sensor || surfaceB.sensor;
    settings.mCombinedFriction =
        CombineSurface(surfaceA.friction, surfaceB.friction, surfaceA.frictionCombine, surfaceB.frictionCombine);
    settings.mCombinedRestitution =
        CombineSurface(surfaceA.restitution, surfaceB.restitution, surfaceA.bounceCombine, surfaceB.bounceCombine);
}

void JoltCpuScene::OnContactPersisted(const JPH::Body &a, const JPH::Body &b, const JPH::ContactManifold &manifold,
                                      JPH::ContactSettings &settings)
{
    OnContactAdded(a, b, manifold, settings);
}
} // namespace infernux
