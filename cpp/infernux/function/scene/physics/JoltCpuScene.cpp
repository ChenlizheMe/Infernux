#include "JoltCpuScene.h"
#include "../Rigidbody.h"

#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/Body/MotionProperties.h>
#include <Jolt/Physics/Collision/CollideShape.h>
#include <Jolt/Physics/Collision/CollisionCollectorImpl.h>
#include <Jolt/Physics/Collision/Shape/CompoundShape.h>
#include <algorithm>
#include <cmath>
#include <core/config/EngineConfig.h>
#include <iterator>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

namespace infernux
{
JoltCpuExecution::JoltCpuExecution(const EngineConfig &config)
    : m_scratch(config.physicsTempAllocatorSize), m_inlineJobs(config.physicsMaxJobs),
      m_workerJobs(config.physicsMaxJobs, config.physicsMaxBarriers, config.physicsMaxConcurrency)
{
}

JPH::EPhysicsUpdateError JoltCpuExecution::Step(JPH::PhysicsSystem &system, float deltaTime, int collisionSteps)
{
    JPH_ASSERT(m_scratch.IsEmpty());
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
        JPH_ASSERT(m_scratch.IsEmpty());
        return result;
    }

    auto group = jobs.CreateTaskGroup(JobDomain::Physics, JobPriority::Critical);
    m_workerJobs.BeginFrame(group);
    jobs.RunInDomain(JobDomain::Physics,
                     [&] { result = system.Update(deltaTime, collisionSteps, &m_scratch, &m_workerJobs); });
    m_workerJobs.EndFrame();
    group.Close();
    jobs.Wait(group);
    JPH_ASSERT(m_scratch.IsEmpty());
    return result;
}

JPH::PhysicsSettings JoltCpuScene::CaptureSettings(const EngineConfig &config)
{
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
    return settings;
}

JoltCpuScene::JoltCpuScene(const EngineConfig &config, const JPH::BroadPhaseLayerInterface &layers,
                           const JPH::ObjectVsBroadPhaseLayerFilter &broadphaseFilter,
                           const JPH::ObjectLayerPairFilter &pairFilter)
{
    m_system.Init(config.physicsMaxBodies, 0, config.physicsMaxBodyPairs, config.physicsMaxContactConstraints, layers,
                  broadphaseFilter, pairFilter);
    m_system.SetPhysicsSettings(CaptureSettings(config));
    const auto gravity = config.physicsGravity;
    m_system.SetGravity(JPH::Vec3(gravity.x, gravity.y, gravity.z));
    m_system.SetBodyActivationListener(this);
    m_system.SetContactListener(this);
    m_system.SetSimShapeFilter(this);
}

JoltCpuScene::~JoltCpuScene()
{
    m_system.SetBodyActivationListener(nullptr);
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
    MarkPoseUpdate(bodyId, entry->second);
    return entry->second.native;
}

void JoltCpuScene::AddBodiesBatch(const std::vector<uint32_t> &bodyIds)
{
    if (bodyIds.empty())
        return;
    auto &bodies = m_system.GetBodyInterface();
    std::vector<JPH::BodyID> nativeIds;
    nativeIds.reserve(bodyIds.size());
    std::unordered_set<uint32_t> seen;
    seen.reserve(bodyIds.size());
    for (const auto id : bodyIds) {
        const auto native = NativeBody(id);
        if (seen.insert(id).second && !bodies.IsAdded(native))
            nativeIds.push_back(native);
    }
    if (nativeIds.empty())
        return;

    const auto count = static_cast<int>(nativeIds.size());
    const auto addState = bodies.AddBodiesPrepare(nativeIds.data(), count);
    // A standard Add explicitly wakes even a body that was sleeping during suspension.
    bodies.AddBodiesFinalize(nativeIds.data(), count, addState, JPH::EActivation::Activate);
    for (const auto native : nativeIds) {
        uint32_t id;
        JPH::AABox bounds;
        {
            JPH::BodyLockRead lock(m_system.GetBodyLockInterface(), native);
            const auto &body = lock.GetBody();
            id = static_cast<uint32_t>(body.GetUserData());
            bounds = body.GetWorldSpaceBounds();
        }
        MarkPoseUpdate(id, m_bodies.at(id));
        m_constraints.OnBodyResidenceChanged(id);
        WakeBodiesInBounds(bounds, native);
    }
}

void JoltCpuScene::RemoveBodyFromBroadphase(uint32_t bodyId, bool preserveVelocity)
{
    const auto found = m_bodies.find(bodyId);
    if (found == m_bodies.end())
        return;
    auto &record = found->second;
    const auto native = record.native;
    std::optional<JPH::AABox> bounds;
    {
        JPH::BodyLockWrite lock(m_system.GetBodyLockInterface(), native);
        auto &body = lock.GetBody();
        if (body.IsInBroadPhase())
            bounds = body.GetWorldSpaceBounds();
        else if (preserveVelocity)
            return;

        if (!body.IsStatic()) {
            record.awakeWhenDetached = !preserveVelocity || body.IsActive();
            if (!preserveVelocity) {
                body.SetLinearVelocityClamped(JPH::Vec3::sZero());
                body.SetAngularVelocityClamped(JPH::Vec3::sZero());
                body.ResetSleepTimer();
            }
        }
    }
    if (bounds) {
        // Both suspension modes retain native continuous force and torque.
        m_system.GetBodyInterface().RemoveBody(native, preserveVelocity);
        m_constraints.OnBodyResidenceChanged(bodyId);
        WakeBodiesInBounds(*bounds);
    }
}

bool JoltCpuScene::IsBodyInBroadphase(uint32_t bodyId) const
{
    const auto found = m_bodies.find(bodyId);
    return found != m_bodies.end() && m_system.GetBodyInterface().IsAdded(found->second.native);
}

void JoltCpuScene::DestroyBody(uint32_t bodyId)
{
    if (m_constraints.HasBodyConstraints(bodyId))
        throw std::logic_error("Retire the CPU body's constraints before destroying it");
    const auto &record = m_bodies.at(bodyId);
    const auto native = record.native;
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
    if (record.poseUpdateIndex != UINT32_MAX) {
        const auto tail = m_poseUpdates.back();
        m_poseUpdates[record.poseUpdateIndex] = tail;
        m_bodies.at(tail).poseUpdateIndex = record.poseUpdateIndex;
        m_poseUpdates.pop_back();
    }
    m_bodies.erase(bodyId);
    if (added)
        WakeBodiesInBounds(oldBounds);
}

void JoltCpuScene::WakeBodiesInBounds(JPH::AABox bounds, JPH::BodyID exclude)
{
    bounds.ExpandBy(JPH::Vec3::sReplicate(m_system.GetPhysicsSettings().mSpeculativeContactDistance));
    JPH::AllHitCollisionCollector<JPH::CollideShapeBodyCollector> collector;
    m_system.GetBroadPhaseQuery().CollideAABox(bounds, collector);
    auto &hits = collector.mHits;
    hits.erase(std::remove_if(hits.begin(), hits.end(),
                              [&](JPH::BodyID native) {
                                  JPH::BodyLockRead lock(m_system.GetBodyLockInterface(), native);
                                  if (native == exclude || !lock.SucceededAndIsInBroadPhase())
                                      return true;
                                  const auto id = static_cast<uint32_t>(lock.GetBody().GetUserData());
                                  auto &record = m_bodies.at(id);
                                  if (!record.authoredDynamic)
                                      return true;
                                  if (!lock.GetBody().IsActive())
                                      MarkPoseUpdate(id, record);
                                  return false;
                              }),
               hits.end());
    m_system.GetBodyInterface().ActivateBodies(hits.data(), static_cast<int>(hits.size()));
}

JPH::BodyID JoltCpuScene::NativeBody(uint32_t bodyId) const
{
    return m_bodies.at(bodyId).native;
}

bool JoltCpuScene::HasBody(uint32_t bodyId) const noexcept
{
    return m_bodies.find(bodyId) != m_bodies.end();
}

const PreparedBodyMass &JoltCpuScene::GetAcceptedMass(uint32_t bodyId) const
{
    const auto &mass = m_bodies.at(bodyId).mass;
    if (!mass)
        throw std::logic_error("Rigidbody has no accepted principal inertia");
    return *mass;
}

bool JoltCpuScene::AcceptedAwake(const BodyRecord &record, const JPH::Body &body)
{
    return !body.IsStatic() && (body.IsInBroadPhase() ? body.IsActive() : record.awakeWhenDetached);
}

void JoltCpuScene::WakeBody(BodyRecord &record, JPH::Body &body)
{
    if (body.IsInBroadPhase()) {
        const bool wasActive = body.IsActive();
        m_system.GetBodyInterfaceNoLock().ActivateBody(record.native);
        if (!wasActive && body.IsActive())
            MarkPoseUpdate(static_cast<uint32_t>(body.GetUserData()), record);
    } else {
        record.awakeWhenDetached = true;
        body.ResetSleepTimer();
    }
}

PhysicsBodyMotionState JoltCpuScene::ReadMotionState(uint32_t bodyId, PhysicsStateFields fields, bool *sleeping) const
{
    const auto &record = m_bodies.at(bodyId);
    JPH::BodyLockRead lock(m_system.GetBodyLockInterface(), record.native);
    return SampleMotionState(bodyId, record, lock.GetBody(), fields, sleeping);
}

PhysicsBodyMotionState JoltCpuScene::SampleMotionState(uint32_t bodyId, const BodyRecord &record, const JPH::Body &body,
                                                       PhysicsStateFields fields, bool *sleeping)
{
    const auto vector = [](const auto &value) {
        return glm::vec3(float(value.GetX()), float(value.GetY()), float(value.GetZ()));
    };
    const auto matrix = [&](const JPH::Mat44 &value) {
        return glm::mat3(vector(value.GetColumn3(0)), vector(value.GetColumn3(1)), vector(value.GetColumn3(2)));
    };
    PhysicsBodyMotionState state;
    state.bodyId = bodyId;
    state.shapeRevision = record.shapeRevision;
    if (sleeping)
        *sleeping = !AcceptedAwake(record, body);
    if (HasPhysicsStateField(fields, PhysicsStateFields::Position))
        state.position = vector(body.GetPosition());
    if (HasPhysicsStateField(fields, PhysicsStateFields::Rotation)) {
        const auto rotation = body.GetRotation();
        state.rotation = {rotation.GetW(), rotation.GetX(), rotation.GetY(), rotation.GetZ()};
    }
    if (HasPhysicsStateField(fields, PhysicsStateFields::CenterOfMass))
        state.centerOfMass = vector(body.GetCenterOfMassPosition());
    if (HasPhysicsStateField(fields, PhysicsStateFields::LinearVelocity))
        state.linearVelocity = vector(body.GetLinearVelocity());
    if (HasPhysicsStateField(fields, PhysicsStateFields::AngularVelocity))
        state.angularVelocity = vector(body.GetAngularVelocity());
    if (body.IsDynamic()) {
        const auto &motion = *body.GetMotionProperties();
        if (HasPhysicsStateField(fields, PhysicsStateFields::InverseMass))
            state.inverseMass = vector(motion.LockTranslation(JPH::Vec3::sReplicate(motion.GetInverseMass())));
        if (HasPhysicsStateField(fields, PhysicsStateFields::InverseInertia))
            state.inverseInertia =
                matrix(motion.GetInverseInertiaForRotation(JPH::Mat44::sRotation(body.GetRotation())));
    }
    // A fully locked authored dynamic body uses native prescribed motion, but
    // still has physical mass. Do not invert its singular response tensor.
    if (record.authoredDynamic && HasPhysicsStateField(fields, PhysicsStateFields::InverseInertia)) {
        const auto &mass = record.mass.value();
        const auto axes = JPH::Mat44::sRotation(body.GetRotation() * mass.axes);
        state.inertia = matrix(
            axes.Multiply3x3(JPH::Mat44::sScale(JPH::Vec3::sOne() / mass.inverse)).Multiply3x3RightTransposed(axes));
    }
    return state;
}

void JoltCpuScene::ApplyBodyForce(uint32_t bodyId, ForceKind kind, ForceMode mode, JPH::Vec3Arg value,
                                  JPH::RVec3Arg point)
{
    const bool continuous = mode == ForceMode::Force || mode == ForceMode::Acceleration;
    const bool massIndependent = mode == ForceMode::Acceleration || mode == ForceMode::VelocityChange;
    if (!continuous && mode != ForceMode::Impulse && mode != ForceMode::VelocityChange)
        throw std::invalid_argument("Invalid CPU physics force mode");
    auto &record = m_bodies.at(bodyId);
    if (!record.authoredDynamic || value == JPH::Vec3::sZero())
        return;

    JPH::BodyLockWrite lock(m_system.GetBodyLockInterface(), record.native);
    auto &body = lock.GetBody();
    if (continuous || !AcceptedAwake(record, body))
        WakeBody(record, body);
    // FreezeAll is prescribed motion but holds continuous loads for this Fixed.
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
}

void JoltCpuScene::SetBodyOriginPose(uint32_t bodyId, const std::optional<JPH::RVec3> &position,
                                     const std::optional<JPH::Quat> &rotation)
{
    auto &record = m_bodies.at(bodyId);
    const auto native = record.native;
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
    MarkPoseUpdate(bodyId, record);
    if (added) {
        JPH::AABox newBounds;
        {
            JPH::BodyLockRead lock(m_system.GetBodyLockInterface(), native);
            newBounds = lock.GetBody().GetWorldSpaceBounds();
        }
        WakeBodiesInBounds(oldBounds, native);
        if (!(oldBounds == newBounds))
            WakeBodiesInBounds(newBounds, native);
    } else {
        SetBodyAwake(bodyId, true);
    }
}

void JoltCpuScene::SetBodyVelocity(uint32_t bodyId, const std::optional<JPH::Vec3> &linear,
                                   const std::optional<JPH::Vec3> &angular)
{
    auto &record = m_bodies.at(bodyId);
    JPH::BodyLockWrite lock(m_system.GetBodyLockInterface(), record.native);
    auto &body = lock.GetBody();
    if (body.IsStatic())
        return;
    if (linear)
        body.SetLinearVelocityClamped(*linear);
    if (angular)
        body.SetAngularVelocityClamped(*angular);
    const auto zero = JPH::Vec3::sZero();
    if (!AcceptedAwake(record, body) && ((linear && *linear != zero) || (angular && *angular != zero)))
        WakeBody(record, body);
}

void JoltCpuScene::SetBodyAwake(uint32_t bodyId, bool awake)
{
    auto &record = m_bodies.at(bodyId);
    JPH::BodyLockWrite lock(m_system.GetBodyLockInterface(), record.native);
    auto &body = lock.GetBody();
    if (body.IsStatic())
        return;
    if (awake) {
        WakeBody(record, body);
    } else {
        body.SetLinearVelocityClamped(JPH::Vec3::sZero());
        body.SetAngularVelocityClamped(JPH::Vec3::sZero());
        if (body.IsInBroadPhase())
            m_system.GetBodyInterfaceNoLock().DeactivateBody(record.native);
        else
            record.awakeWhenDetached = false;
    }
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

uint32_t JoltCpuScene::BodyRecord::MemberIndex(JPH::SubShapeID subshape) const
{
    if (!memberCompound)
        return 0;
    JPH::SubShapeID remainder;
    const auto index = memberCompound->GetSubShapeIndexFromID(subshape, remainder);
    return memberCompound->GetCompoundUserData(index);
}

const JoltCpuScene::Surface &JoltCpuScene::GetSurface(uint32_t bodyId, JPH::SubShapeID subshape) const
{
    const auto &record = m_bodies.at(bodyId);
    return record.surfaces[record.MemberIndex(subshape)];
}

bool JoltCpuScene::SetIgnoredPairs(std::shared_ptr<const PhysicsColliderPairs> pairs)
{
    if (UsesIgnoredPairs(pairs))
        return false;
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
        return false;
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
    return true;
}

void JoltCpuScene::MarkPoseUpdate(uint32_t bodyId, BodyRecord &record)
{
    if (record.poseUpdateIndex != UINT32_MAX)
        return;
    m_poseUpdates.push_back(bodyId);
    record.poseUpdateIndex = static_cast<uint32_t>(m_poseUpdates.size() - 1);
}

void JoltCpuScene::MarkActivePoseUpdates()
{
    // The owner calls this outside Update, when the active array and body
    // directory are stable. Public retire/replacement cannot race this loop.
    const auto *active = m_system.GetActiveBodiesUnsafe(JPH::EBodyType::RigidBody);
    const auto count = m_system.GetNumActiveBodies(JPH::EBodyType::RigidBody);
    const auto &bodies = m_system.GetBodyLockInterfaceNoLock();
    for (uint32_t i = 0; i < count; ++i) {
        const auto native = active[i];
        const auto *body = bodies.TryGetBody(native);
        const auto id = static_cast<uint32_t>(body->GetUserData());
        const auto found = m_bodies.find(id);
        if (found != m_bodies.end() && found->second.native == native)
            MarkPoseUpdate(id, found->second);
    }
}

void JoltCpuScene::OnBodyDeactivated(const JPH::BodyID &native, JPH::uint64 userData) noexcept
{
    // Jolt can invoke this while holding body locks on a worker. Only record
    // identities here. The owner resolves them after the native update.
    const std::lock_guard lock(m_deactivatedPoseMutex);
    if (m_deactivatedPoseError)
        return;
    try {
        m_deactivatedPoses.push_back({static_cast<uint32_t>(userData), native});
    } catch (...) {
        m_deactivatedPoseError = std::current_exception();
    }
}

std::vector<JoltCpuScene::PoseUpdate> JoltCpuScene::TakePoseUpdates()
{
    const auto &bodies = m_system.GetBodyLockInterfaceNoLock();
    {
        const std::lock_guard lock(m_deactivatedPoseMutex);
        if (m_deactivatedPoseError)
            std::rethrow_exception(m_deactivatedPoseError);
        for (const auto &event : m_deactivatedPoses) {
            const auto found = m_bodies.find(event.bodyId);
            // Removal from broadphase also invokes Jolt's deactivation
            // listener. A detached body only publishes actual pose changes.
            if (found != m_bodies.end() && found->second.native == event.native &&
                bodies.TryGetBody(event.native)->IsInBroadPhase())
                MarkPoseUpdate(event.bodyId, found->second);
        }
        m_deactivatedPoses.clear();
    }

    constexpr auto fields = static_cast<PhysicsStateFields>(static_cast<uint32_t>(PhysicsStateFields::Position) |
                                                            static_cast<uint32_t>(PhysicsStateFields::Rotation));
    std::vector<PoseUpdate> updates;
    updates.reserve(m_poseUpdates.size());
    for (const auto id : m_poseUpdates) {
        auto &record = m_bodies.at(id);
        bool sleeping;
        const auto state = SampleMotionState(id, record, *bodies.TryGetBody(record.native), fields, &sleeping);
        updates.push_back({{id, state.position, state.rotation}, sleeping});
        record.poseUpdateIndex = UINT32_MAX;
    }
    m_poseUpdates.clear();
    return updates;
}

JPH::ValidateResult JoltCpuScene::OnContactValidate(const JPH::Body &a, const JPH::Body &b, JPH::RVec3Arg,
                                                    const JPH::CollideShapeResult &hit)
{
    const auto idA = static_cast<uint32_t>(a.GetUserData()), idB = static_cast<uint32_t>(b.GetUserData());
    if (m_constraints.SuppressesCollision(idA, idB))
        return JPH::ValidateResult::RejectAllContactsForThisBodyPair;
    const auto &surfaceA = GetSurface(idA, hit.mSubShapeID1);
    const auto &surfaceB = GetSurface(idB, hit.mSubShapeID2);
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
    if (m_constraints.SuppressesCollision(idA, idB))
        return false;
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
