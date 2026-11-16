#include "JoltCpuScene.h"
#include "../Rigidbody.h"

#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/Body/MotionProperties.h>
#include <Jolt/Physics/Collision/CollideShape.h>
#include <Jolt/Physics/Collision/Shape/CompoundShape.h>
#include <algorithm>
#include <cmath>
#include <core/config/EngineConfig.h>
#include <iterator>
#include <set>
#include <stdexcept>
#include <tuple>

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
    m_system.SetSimShapeFilter(this);
}

JoltCpuScene::~JoltCpuScene()
{
    m_system.SetContactListener(nullptr);
    m_system.SetSimShapeFilter(nullptr);
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
    const bool added = bodies.IsAdded(native);
    JPH::AABox oldBounds;
    if (added) {
        JPH::BodyLockRead lock(m_system.GetBodyLockInterface(), native);
        oldBounds = lock.GetBody().GetWorldSpaceBounds();
    }
    if (added)
        bodies.RemoveBody(native);
    bodies.DestroyBody(native);
    m_bodies.erase(bodyId);
    if (added)
        WakeBodiesInBounds(oldBounds);
}

void JoltCpuScene::WakeBodiesInBounds(JPH::AABox bounds)
{
    bounds.ExpandBy(JPH::Vec3::sReplicate(m_system.GetPhysicsSettings().mSpeculativeContactDistance));
    m_system.GetBodyInterface().ActivateBodiesInAABox(bounds, JPH::BroadPhaseLayerFilter{}, JPH::ObjectLayerFilter{});
}

JPH::BodyID JoltCpuScene::NativeBody(uint32_t bodyId) const
{
    return m_bodies.at(bodyId).native;
}

void JoltCpuScene::ApplyBodyForce(uint32_t bodyId, ForceKind kind, ForceMode mode, JPH::Vec3Arg value,
                                  JPH::RVec3Arg point)
{
    const bool continuous = mode == ForceMode::Force || mode == ForceMode::Acceleration;
    const bool massIndependent = mode == ForceMode::Acceleration || mode == ForceMode::VelocityChange;
    if (!continuous && mode != ForceMode::Impulse && mode != ForceMode::VelocityChange)
        throw std::invalid_argument("Invalid CPU physics force mode");
    const auto &record = m_bodies.at(bodyId);
    if (!record.authoredDynamic || value == JPH::Vec3::sZero())
        return;

    bool wake = false;
    {
        JPH::BodyLockWrite lock(m_system.GetBodyLockInterface(), record.native);
        auto &body = lock.GetBody();
        // FreezeAll uses prescribed native motion. Pending continuous loads
        // still belong to the authored dynamic body if it unlocks this Fixed.
        if (!body.IsDynamic() && !continuous)
            return;

        auto force = JPH::Vec3::sZero(), torque = JPH::Vec3::sZero();
        if (kind == ForceKind::Torque) {
            torque = value;
            if (massIndependent) {
                const auto &mass = record.mass.value();
                const auto axes = body.GetRotation() * mass.axes;
                torque = axes * ((axes.Conjugated() * value) / mass.inverse);
            }
        } else {
            force = massIndependent ? value * record.mass.value().mass : value;
            if (kind == ForceKind::ForceAtPosition)
                torque = JPH::Vec3(point - body.GetCenterOfMassPosition()).Cross(force);
        }

        if (continuous) {
            auto &motion = *body.GetMotionProperties();
            motion.SetAccumulatedForceAndTorque(motion.GetAccumulatedForce() + force,
                                                motion.GetAccumulatedTorque() + torque);
        } else {
            if (kind != ForceKind::Torque)
                body.AddImpulse(force);
            if (kind != ForceKind::Force)
                body.AddAngularImpulse(torque);
        }
        // Native activation also resets an already-active body's sleep timer.
        wake = body.IsInBroadPhase() && (continuous || !body.IsActive());
    }
    if (wake)
        m_system.GetBodyInterface().ActivateBody(record.native);
}

void JoltCpuScene::SetBodyOriginPose(uint32_t bodyId, const std::optional<JPH::RVec3> &position,
                                     const std::optional<JPH::Quat> &rotation)
{
    const auto native = NativeBody(bodyId);
    auto &bodies = m_system.GetBodyInterface();
    JPH::RVec3 origin;
    JPH::Quat orientation;
    JPH::AABox oldBounds;
    bool added;
    {
        JPH::BodyLockRead lock(m_system.GetBodyLockInterface(), native);
        const auto &body = lock.GetBody();
        origin = position.value_or(body.GetPosition());
        orientation = rotation.value_or(body.GetRotation());
        if (origin == body.GetPosition() && (orientation == body.GetRotation() || orientation == -body.GetRotation()))
            return;
        added = body.IsInBroadPhase();
        oldBounds = body.GetWorldSpaceBounds();
    }

    // Jolt sets the body origin and applies the currently accepted local COM.
    bodies.SetPositionAndRotation(native, origin, orientation, JPH::EActivation::Activate);
    bodies.InvalidateContactCache(native);
    if (added) {
        JPH::AABox newBounds;
        {
            JPH::BodyLockRead lock(m_system.GetBodyLockInterface(), native);
            newBounds = lock.GetBody().GetWorldSpaceBounds();
        }
        WakeBodiesInBounds(oldBounds);
        if (!(oldBounds == newBounds))
            WakeBodiesInBounds(newBounds);
    }
}

void JoltCpuScene::SetBodyVelocity(uint32_t bodyId, const std::optional<JPH::Vec3> &linear,
                                   const std::optional<JPH::Vec3> &angular)
{
    const auto native = NativeBody(bodyId);
    bool wake = false;
    {
        JPH::BodyLockWrite lock(m_system.GetBodyLockInterface(), native);
        auto &body = lock.GetBody();
        if (body.IsStatic())
            return;
        if (linear)
            body.SetLinearVelocityClamped(*linear);
        if (angular)
            body.SetAngularVelocityClamped(*angular);
        const auto zero = JPH::Vec3::sZero();
        wake =
            !body.IsActive() && body.IsInBroadPhase() && ((linear && *linear != zero) || (angular && *angular != zero));
    }
    if (wake)
        m_system.GetBodyInterface().ActivateBody(native);
}

void JoltCpuScene::SetBodyAwake(uint32_t bodyId, bool awake)
{
    const auto native = NativeBody(bodyId);
    auto &bodies = m_system.GetBodyInterface();
    if (!awake)
        SetBodyVelocity(bodyId, JPH::Vec3::sZero(), JPH::Vec3::sZero());
    if (!bodies.IsAdded(native))
        return;
    if (awake)
        bodies.ActivateBody(native);
    else
        bodies.DeactivateBody(native);
}

void JoltCpuScene::ResetBodyMotion(uint32_t bodyId)
{
    const auto native = NativeBody(bodyId);
    {
        JPH::BodyLockWrite lock(m_system.GetBodyLockInterface(), native);
        auto &body = lock.GetBody();
        if (!body.IsStatic())
            body.GetMotionProperties()->ResetMotion();
    }
    m_system.ForgetBodyContacts(&native, 1);
    SetBodyAwake(bodyId, true);
}

void JoltCpuScene::StartKinematicDrive(uint32_t bodyId, JPH::RVec3Arg position, JPH::QuatArg rotation, float duration,
                                       float maxSpeed)
{
    if (!std::isfinite(duration) || duration <= 0.0f || !std::isfinite(maxSpeed) || maxSpeed < 0.0f)
        throw std::invalid_argument("Physics movement requires a positive duration and nonnegative speed limit");

    const auto &record = m_bodies.at(bodyId);
    auto reachable = rotation;
    JPH::Quat deltaRotation;
    JPH::RVec3 cappedOrigin;
    bool capped = false;
    {
        JPH::BodyLockRead lock(m_system.GetBodyLockInterface(), record.native);
        const auto &body = lock.GetBody();
        if (!body.IsInBroadPhase() || !body.IsKinematic() || record.authoredDynamic)
            throw std::logic_error("CPU kinematic drive requires a resident authored kinematic body");

        const auto &motion = *body.GetMotionProperties();
        const auto current = body.GetRotation();
        deltaRotation = (rotation * current.Conjugated()).Normalized();
        // At exactly pi, choose the same axis for q and -q.
        if (deltaRotation.GetW() == 0.0f) {
            if (std::tuple(deltaRotation.GetX(), deltaRotation.GetY(), deltaRotation.GetZ()) <
                std::tuple(0.0f, 0.0f, 0.0f))
                deltaRotation = -deltaRotation;
            deltaRotation.SetW(0.0f);
        }
        constexpr auto rotationDOFs =
            JPH::EAllowedDOFs::RotationX | JPH::EAllowedDOFs::RotationY | JPH::EAllowedDOFs::RotationZ;
        if ((motion.GetAllowedDOFs() & rotationDOFs) != rotationDOFs) {
            const auto step = motion.LockAngular(deltaRotation.GetAngularVelocity(duration)) * duration;
            const auto angle = step.Length();
            reachable = angle == 0.0f ? current : (JPH::Quat::sRotation(step / angle, angle) * current).Normalized();
        }

        const auto delta = JPH::Vec3(position - body.GetPosition());
        const float distance = delta.Length(), maximum = maxSpeed * duration;
        capped = maxSpeed > 0.0f && distance > maximum;
        if (capped) {
            const auto com = body.GetShape()->GetCenterOfMass();
            const auto wanted = position - JPH::RVec3(delta * (maximum / distance)) + reachable * com;
            cappedOrigin = body.GetCenterOfMassPosition() +
                           motion.LockTranslation(JPH::Vec3(wanted - body.GetCenterOfMassPosition())) - reachable * com;
            deltaRotation = JPH::Quat::sIdentity();
        }
    }

    if (capped)
        SetBodyOriginPose(bodyId, cappedOrigin, reachable);

    bool wake = false;
    {
        JPH::BodyLockWrite lock(m_system.GetBodyLockInterface(), record.native);
        auto &body = lock.GetBody();
        const auto targetCOM = position + reachable * body.GetShape()->GetCenterOfMass();
        // Native MoveKinematic applies DOFs without imposing manual velocity clamps.
        body.GetMotionProperties()->MoveKinematic(JPH::Vec3(targetCOM - body.GetCenterOfMassPosition()), deltaRotation,
                                                  duration);
        wake = !body.IsActive() && body.IsInBroadPhase() &&
               (body.GetLinearVelocity() != JPH::Vec3::sZero() || body.GetAngularVelocity() != JPH::Vec3::sZero());
    }
    if (wake)
        m_system.GetBodyInterface().ActivateBody(record.native);
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
            if (bodies.IsAdded(record.native)) {
                JPH::AABox bounds;
                {
                    JPH::BodyLockRead lock(m_system.GetBodyLockInterface(), record.native);
                    bounds = lock.GetBody().GetWorldSpaceBounds();
                }
                WakeBodiesInBounds(bounds);
            }
        }
    }
}

JPH::ValidateResult JoltCpuScene::OnContactValidate(const JPH::Body &a, const JPH::Body &b, JPH::RVec3Arg,
                                                    const JPH::CollideShapeResult &hit)
{
    const auto &surfaceA = GetSurface(static_cast<uint32_t>(a.GetUserData()), hit.mSubShapeID1);
    const auto &surfaceB = GetSurface(static_cast<uint32_t>(b.GetUserData()), hit.mSubShapeID2);
    // Acceptance is per leaf: accepting the whole body pair bypasses filters
    // on the other Colliders of a compound.
    return SurfacesCollide(surfaceA, surfaceB) ? JPH::ValidateResult::AcceptContact
                                               : JPH::ValidateResult::RejectContact;
}

bool JoltCpuScene::SurfacesCollide(const Surface &a, const Surface &b) const
{
    if ((a.collisionMask & (1u << b.layer)) == 0 || (b.collisionMask & (1u << a.layer)) == 0)
        return false;
    if (m_ignoredPairs) {
        std::array<uint64_t, 2> first{a.colliderId, a.generation}, second{b.colliderId, b.generation};
        if (second < first)
            std::swap(first, second);
        const PhysicsColliderPair key{first[0], first[1], second[0], second[1]};
        if (std::binary_search(m_ignoredPairs->begin(), m_ignoredPairs->end(), key))
            return false;
    }
    return true;
}

bool JoltCpuScene::ShouldCollide(const JPH::Body &a, const JPH::Shape *shapeA, const JPH::SubShapeID &subshapeA,
                                 const JPH::Body &b, const JPH::Shape *shapeB, const JPH::SubShapeID &subshapeB) const
{
    if (shapeA->GetSubType() == JPH::EShapeSubType::Empty || shapeB->GetSubType() == JPH::EShapeSubType::Empty)
        return false;
    const auto idA = static_cast<uint32_t>(a.GetUserData()), idB = static_cast<uint32_t>(b.GetUserData());
    const auto container = [](const JPH::Shape &shape) {
        return shape.GetType() == JPH::EShapeType::Compound || shape.GetType() == JPH::EShapeType::Decorated;
    };
    if (container(*shapeA) || container(*shapeB))
        return true;
    // Match the common component contract before native narrowphase dispatch:
    // two prescribed/static solid surfaces do not create response or events.
    const auto &surfaceA = GetSurface(idA, subshapeA), &surfaceB = GetSurface(idB, subshapeB);
    return (m_bodies.at(idA).authoredDynamic || m_bodies.at(idB).authoredDynamic || surfaceA.sensor ||
            surfaceB.sensor) &&
           SurfacesCollide(surfaceA, surfaceB);
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
