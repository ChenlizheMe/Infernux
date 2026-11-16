#include "JoltCpuScene.h"

#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <core/config/EngineConfig.h>
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
}

JPH::BodyID JoltCpuScene::CreateBody(uint32_t bodyId, const JPH::BodyCreationSettings &settings)
{
    auto [entry, inserted] = m_bodies.try_emplace(bodyId);
    if (!inserted)
        throw std::logic_error("CPU physics body identity is already owned");
    auto *body = m_system.GetBodyInterface().CreateBody(settings);
    if (!body) {
        m_bodies.erase(entry);
        throw std::length_error("CPU physics body capacity exhausted");
    }
    body->SetUserData(bodyId);
    entry->second = body->GetID();
    return entry->second;
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
    return m_bodies.at(bodyId);
}
} // namespace infernux
