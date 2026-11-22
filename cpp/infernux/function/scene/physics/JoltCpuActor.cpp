#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/Body/MotionProperties.h>
#include <Jolt/Physics/Collision/Shape/CompoundShape.h>
#include <Jolt/Physics/Collision/Shape/OffsetCenterOfMassShape.h>

#include "../Rigidbody.h"
#include "JoltCpuActor.h"
#include "JoltCpuScene.h"
#include "PhysicsLayers.h"
#include "PhysicsShapeAuthoring.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace infernux
{
namespace
{
JPH::ObjectLayer BodyLayer(const PhysicsActorAuthoring &actor)
{
    const bool moving = actor.body.has_value();
    if (actor.surfaces.empty())
        return PhysicsObjectLayers::Encode(actor.objectLayer, moving);
    const auto layer = actor.surfaces.front().layer;
    for (const auto &surface : actor.surfaces)
        if (surface.layer != layer)
            return moving ? PhysicsObjectLayers::MIXED_MOVING : PhysicsObjectLayers::MIXED_NON_MOVING;
    return PhysicsObjectLayers::Encode(layer, moving);
}
JPH::ObjectLayer StaticDriveLayer(JPH::ObjectLayer layer, bool moving)
{
    using namespace PhysicsObjectLayers;
    if (IsMixed(layer))
        return moving ? MIXED_MOVING : MIXED_NON_MOVING;
    return Encode(DecodeGameLayer(layer), moving);
}

JPH::BodyCreationSettings CreationSettings(const PreparedCpuActor &actor)
{
    auto settings = actor.settings;
    // Jolt's constructor requires already-projected and clamped velocities.
    settings.mLinearVelocity = settings.mAngularVelocity = JPH::Vec3::sZero();
    return settings;
}

void InstallInitialVelocities(JPH::Body &body, const JPH::BodyCreationSettings &author)
{
    body.SetLinearVelocityClamped(author.mLinearVelocity);
    body.SetAngularVelocityClamped(author.mAngularVelocity);
}

struct BodyMobility
{
    JPH::EMotionType motionType;
    JPH::EAllowedDOFs allowedDOFs = JPH::EAllowedDOFs::None;
    float inverseMass = 0.0f;
    JPH::Vec3 inverseInertia = JPH::Vec3::sZero();
    JPH::Quat inertiaRotation = JPH::Quat::sIdentity();

    bool operator==(const BodyMobility &other) const
    {
        return motionType == other.motionType && allowedDOFs == other.allowedDOFs && inverseMass == other.inverseMass &&
               inverseInertia == other.inverseInertia &&
               (inertiaRotation == other.inertiaRotation || inertiaRotation == -other.inertiaRotation);
    }
};

BodyMobility ReadMobility(const JPH::Body &body)
{
    BodyMobility result{body.GetMotionType()};
    if (body.IsStatic())
        return result;
    const auto &motion = *body.GetMotionProperties();
    result.allowedDOFs = motion.GetAllowedDOFs();
    if (body.IsDynamic()) {
        result.inverseMass = motion.GetInverseMass();
        result.inverseInertia = motion.GetInverseInertiaDiagonal();
        result.inertiaRotation = motion.GetInertiaRotation();
    }
    return result;
}
} // namespace

PreparedCpuActor PrepareJoltCpuActor(PhysicsActorAuthoring actor)
{
    PreparedCpuActor prepared;
    const auto layer = BodyLayer(actor);
    prepared.surfaces = std::move(actor.surfaces);
    prepared.publication = std::move(actor.publication);
    const auto &body = actor.body;
    const auto &position = actor.position;
    const auto &rotation = actor.rotation;
    const auto &scale = actor.scale;
    prepared.rootScale = JPH::Vec3(scale.x, scale.y, scale.z);
    // Jolt cannot solve a dynamic body with all six axes locked. Such a body
    // is an immovable kinematic boundary internally; the component stays a
    // dynamic Rigidbody, with its real mass available when axes are unlocked.
    const bool frozen = body && body->constraints == static_cast<int>(RigidbodyConstraints::FreezeAll);
    prepared.frozen = frozen;
    prepared.authoredDynamic = body && !body->kinematic;
    const auto motion = !body                       ? JPH::EMotionType::Static
                        : body->kinematic || frozen ? JPH::EMotionType::Kinematic
                                                    : JPH::EMotionType::Dynamic;
    auto &settings = prepared.settings;
    settings = JPH::BodyCreationSettings(actor.shape, JPH::RVec3(position.x, position.y, position.z),
                                         JPH::Quat(rotation.x, rotation.y, rotation.z, rotation.w), motion, layer);
    settings.mAllowDynamicOrKinematic = body.has_value();
    settings.mApplyGyroscopicForce = true;
    settings.mIsSensor = !prepared.surfaces.empty() && std::all_of(prepared.surfaces.begin(), prepared.surfaces.end(),
                                                                   [](const auto &surface) { return surface.sensor; });
    // Preserve per-Collider material/filter/trigger identity in compounds.
    settings.mUseManifoldReduction = prepared.surfaces.size() <= 1;
    settings.mCollideKinematicVsNonDynamic = true;
    if (!prepared.surfaces.empty()) {
        settings.mFriction = prepared.surfaces.front().friction;
        settings.mRestitution = prepared.surfaces.front().restitution;
    }
    if (body) {
        prepared.bodyTarget = body->target;
        prepared.mass = body->mass;
        if (prepared.mass) {
            const auto &mass = *prepared.mass;
            const auto axes = JPH::Mat44::sRotation(mass.axes);
            settings.mOverrideMassProperties = JPH::EOverrideMassProperties::MassAndInertiaProvided;
            settings.mMassPropertiesOverride.mMass = mass.mass;
            settings.mMassPropertiesOverride.mInertia =
                axes * JPH::Mat44::sScale(mass.inverse.Reciprocal()) * axes.Transposed3x3();
        }
        settings.mAllowedDOFs = static_cast<JPH::EAllowedDOFs>((~(body->constraints >> 1)) & 0x3f);
        prepared.collisionDetectionMode = body->collisionDetectionMode;
        settings.mMotionQuality =
            motion == JPH::EMotionType::Dynamic && !prepared.surfaces.empty() && prepared.collisionDetectionMode != 0
                ? JPH::EMotionQuality::LinearCast
                : JPH::EMotionQuality::Discrete;
        settings.mLinearDamping = body->drag;
        settings.mAngularDamping = body->angularDrag;
        settings.mMaxLinearVelocity = body->maxLinearVelocity;
        settings.mMaxAngularVelocity = body->maxAngularVelocity;
        settings.mGravityFactor = body->gravityFactor;

        if (const auto &linear = prepared.publication.linearVelocity; !frozen && linear)
            settings.mLinearVelocity = JPH::Vec3(linear->x, linear->y, linear->z);
        if (const auto &angular = prepared.publication.angularVelocity; !frozen && angular)
            settings.mAngularVelocity = JPH::Vec3(angular->x, angular->y, angular->z);
    }
    return prepared;
}

JoltCpuScene::BodyRecord JoltCpuScene::CaptureBodyRecord(const PreparedCpuActor &actor)
{
    BodyRecord record;
    record.shapeRevision = actor.publication.sharedGeometryRevision.value_or(0);
    record.shape = actor.settings.GetShape();
    const auto scale = actor.rootScale;
    if (!std::isfinite(scale.GetX()) || !std::isfinite(scale.GetY()) || !std::isfinite(scale.GetZ()))
        throw std::invalid_argument("CPU body root scale must be finite");
    record.rootScale = scale;
    if (actor.surfaces.size() > 1) {
        const auto *root = record.shape.GetPtr();
        if (root->GetSubType() == JPH::EShapeSubType::OffsetCenterOfMass)
            root = static_cast<const JPH::OffsetCenterOfMassShape *>(root)->GetInnerShape();
        record.memberCompound = static_cast<const JPH::CompoundShape *>(root);
    }
    record.surfaces = actor.surfaces;
    record.bodyTarget = actor.bodyTarget;
    record.mass = actor.mass;
    record.collisionDetectionMode = actor.collisionDetectionMode;
    record.authoredDynamic = actor.authoredDynamic;
    return record;
}

JPH::BodyID JoltCpuScene::CreateBody(uint32_t bodyId, const PreparedCpuActor &actor)
{
    const auto settings = CreationSettings(actor);
    auto record = CaptureBodyRecord(actor);
    record.awakeWhenDetached = settings.mMotionType != JPH::EMotionType::Static;
    const auto native = CreateBody(bodyId, settings, std::move(record));
    if (settings.mMotionType != JPH::EMotionType::Static) {
        JPH::BodyLockWrite lock(m_system.GetBodyLockInterfaceNoLock(), native);
        InstallInitialVelocities(lock.GetBody(), actor.settings);
    }
    return native;
}

bool JoltCpuScene::NeedsBodyReplacement(uint32_t bodyId, const PreparedCpuActor &actor) const
{
    if (!actor.settings.HasMassProperties())
        return false;
    JPH::BodyLockRead lock(m_system.GetBodyLockInterface(), NativeBody(bodyId));
    return !lock.GetBody().CanBeKinematicOrDynamic();
}

JPH::BodyID JoltCpuScene::ReplaceBody(uint32_t bodyId, const PreparedCpuActor &actor)
{
    auto &current = m_bodies.at(bodyId);
    if (m_constraints.HasBodyConstraints(bodyId))
        throw std::logic_error("Retire the CPU body's constraints before replacing its allocation");
    auto settings = CreationSettings(actor);
    if (settings.mMotionType == JPH::EMotionType::Static)
        throw std::logic_error("CPU body replacement requires a complete Rigidbody model");
    auto next = CaptureBodyRecord(actor);
    next.ReuseQueryGeometry(current);
    const auto previousId = current.native;
    std::optional<JPH::AABox> previousBounds;
    {
        JPH::BodyLockRead lock(m_system.GetBodyLockInterface(), previousId);
        const auto &body = lock.GetBody();
        if (!body.IsStatic() || body.CanBeKinematicOrDynamic())
            throw std::logic_error("CPU replacement requires the original collider-only allocation");
        settings.mPosition = body.GetPosition();
        settings.mRotation = body.GetRotation();
        if (body.IsInBroadPhase())
            previousBounds = body.GetWorldSpaceBounds();
    }

    auto &bodies = m_system.GetBodyInterface();
    // Allocate before changing the old actor. Assign the released slot after
    // preparing the replacement, so a full scene does not need extra ID capacity.
    auto *replacement = bodies.CreateBodyWithoutID(settings);
    replacement->SetUserData(bodyId);
    InstallInitialVelocities(*replacement, actor.settings);
    const auto newBounds = replacement->GetWorldSpaceBounds();
    if (previousBounds)
        bodies.RemoveBody(previousId);
    auto *previous = bodies.UnassignBodyID(previousId);
    // This owner has just released an ID slot and the replacement has no ID.
    const bool assigned = bodies.AssignBodyID(replacement);
    JPH_ASSERT(assigned);
    (void)assigned;
    next.native = replacement->GetID();
    next.awakeWhenDetached = true;
    next.shapeRevision = actor.publication.sharedGeometryRevision.value_or(current.shapeRevision + 1);
    next.poseUpdateIndex = current.poseUpdateIndex;
    const auto native = next.native;
    current = std::move(next);
    bodies.DestroyBodyWithoutID(previous);
    MarkPoseUpdate(bodyId, current);
    if (previousBounds) {
        bodies.AddBody(native, JPH::EActivation::Activate);
        WakeBodiesInBounds(*previousBounds, native);
        if (!(*previousBounds == newBounds))
            WakeBodiesInBounds(newBounds, native);
    }
    return native;
}

std::optional<JPH::AABox> JoltCpuScene::StartTemporaryStaticDrive(uint32_t bodyId, JPH::RVec3Arg position,
                                                                  JPH::QuatArg rotation, float duration, float maxSpeed)
{
    if (!std::isfinite(duration) || duration <= 0.0f || !std::isfinite(maxSpeed) || maxSpeed < 0.0f)
        throw std::invalid_argument("Physics movement requires a positive duration and nonnegative speed limit");
    auto &record = m_bodies.at(bodyId);
    if (!record.temporaryStaticDrive) {
        auto &bodies = m_system.GetBodyInterface();
        const auto native = record.native;
        std::optional<JPH::BodyCreationSettings> replacementSettings;
        JPH::ObjectLayer movingLayer;
        {
            JPH::BodyLockRead lock(m_system.GetBodyLockInterface(), native);
            const auto &body = lock.GetBody();
            if (!body.IsStatic() || !body.IsInBroadPhase() || record.authoredDynamic)
                throw std::logic_error("A temporary static drive requires a resident static body");
            if (position == body.GetPosition() && (rotation == body.GetRotation() || rotation == -body.GetRotation()))
                return std::nullopt;
            movingLayer = StaticDriveLayer(body.GetObjectLayer(), true);
            if (!body.CanBeKinematicOrDynamic())
                replacementSettings = body.GetBodyCreationSettings();
        }

        if (replacementSettings) {
            if (m_constraints.HasBodyConstraints(bodyId))
                throw std::logic_error("A collider-only CPU allocation cannot upgrade with attached constraints");
            auto &settings = *replacementSettings;
            const JPH::BodyCreationSettings defaults;
            settings.mMotionType = JPH::EMotionType::Kinematic;
            settings.mObjectLayer = movingLayer;
            // The native constructor initializes a prescribed body with
            // zero inverse mass and inertia through its None-DOF path.
            settings.mAllowedDOFs = JPH::EAllowedDOFs::None;
            settings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateMassAndInertia;
            settings.mMaxLinearVelocity = defaults.mMaxLinearVelocity;
            settings.mMaxAngularVelocity = defaults.mMaxAngularVelocity;

            auto *replacement = bodies.CreateBodyWithoutID(settings);
            replacement->SetUserData(bodyId);
            replacement->GetMotionProperties()->SetKinematicDOFs(JPH::EAllowedDOFs::All);
            // Reuse the released ID slot even when the scene is full. The
            // shape, surfaces and published revision stay in the same record.
            bodies.RemoveBody(native);
            auto *previous = bodies.UnassignBodyID(native);
            const bool assigned = bodies.AssignBodyID(replacement);
            JPH_ASSERT(assigned);
            (void)assigned;
            record.native = replacement->GetID();
            bodies.DestroyBodyWithoutID(previous);
            bodies.AddBody(record.native, JPH::EActivation::DontActivate);
        } else {
            bodies.SetMotionType(native, JPH::EMotionType::Kinematic, JPH::EActivation::DontActivate);
            bodies.SetObjectLayer(native, movingLayer);
            {
                JPH::BodyLockWrite lock(m_system.GetBodyLockInterface(), native);
                lock.GetBody().GetMotionProperties()->SetKinematicDOFs(JPH::EAllowedDOFs::All);
            }
            bodies.InvalidateContactCache(native);
        }
        record.temporaryStaticDrive = true;
    }

    return StartKinematicDrive(bodyId, position, rotation, duration, maxSpeed);
}

void JoltCpuScene::StopTemporaryStaticDrive(uint32_t bodyId)
{
    auto &record = m_bodies.at(bodyId);
    if (!record.temporaryStaticDrive)
        return;

    auto &bodies = m_system.GetBodyInterface();
    const auto native = record.native;
    const auto layer = StaticDriveLayer(bodies.GetObjectLayer(native), false);
    // Native demotion deactivates the body and clears both drive velocities.
    bodies.SetMotionType(native, JPH::EMotionType::Static, JPH::EActivation::DontActivate);
    bodies.SetObjectLayer(native, layer);
    bodies.InvalidateContactCache(native);
    record.temporaryStaticDrive = false;
    record.awakeWhenDetached = false;
    MarkPoseUpdate(bodyId, record);
}

JPH::Vec3 JoltCpuScene::UpdateBody(uint32_t bodyId, const PreparedCpuActor &actor)
{
    auto next = CaptureBodyRecord(actor);
    auto &record = m_bodies.at(bodyId);
    next.ReuseQueryGeometry(record);
    const auto native = record.native;
    const auto &settings = actor.settings;
    const auto deltaCOM = next.shape->GetCenterOfMass() - record.shape->GetCenterOfMass();
    auto &bodies = m_system.GetBodyInterface();
    const bool added = bodies.IsAdded(native);
    const bool retainLoads = record.authoredDynamic && actor.authoredDynamic;
    const bool hasJoints = m_constraints.HasBodyConstraints(bodyId);
    BodyMobility oldMobility{JPH::EMotionType::Static};
    bool mobilityChanged = false, poseChanged = false, wasActive = false;
    JPH::RVec3 oldOrigin;
    JPH::AABox oldBounds, newBounds;
    JPH::Vec3 force = JPH::Vec3::sZero(), torque = JPH::Vec3::sZero();
    {
        JPH::BodyLockRead lock(m_system.GetBodyLockInterface(), native);
        const auto &body = lock.GetBody();
        if (settings.HasMassProperties() && !body.CanBeKinematicOrDynamic())
            throw std::logic_error("Adding a Rigidbody to a CPU static actor requires body allocation");
        oldBounds = body.GetWorldSpaceBounds();
        oldOrigin = body.GetPosition();
        wasActive = body.IsActive();
        if (hasJoints)
            oldMobility = ReadMobility(body);
        if (retainLoads) {
            force = body.GetMotionProperties()->GetAccumulatedForce();
            torque = body.GetMotionProperties()->GetAccumulatedTorque();
        }
    }
    // SetShape retains the current body-origin pose while updating its COM and
    // broadphase bounds. Configuration edits never teleport to a stale author pose.
    bodies.SetShape(native, next.shape, false, JPH::EActivation::DontActivate);
    next.temporaryStaticDrive = record.temporaryStaticDrive && settings.mMotionType == JPH::EMotionType::Static;
    bodies.SetObjectLayer(native, next.temporaryStaticDrive ? StaticDriveLayer(settings.mObjectLayer, true)
                                                            : settings.mObjectLayer);
    const auto motionType = next.temporaryStaticDrive ? JPH::EMotionType::Kinematic : settings.mMotionType;
    bodies.SetMotionType(native, motionType, JPH::EActivation::DontActivate);
    {
        JPH::BodyLockWrite lock(m_system.GetBodyLockInterface(), native);
        auto &body = lock.GetBody();
        body.SetIsSensor(settings.mIsSensor);
        body.SetCollideKinematicVsNonDynamic(settings.mCollideKinematicVsNonDynamic);
        body.SetUseManifoldReduction(settings.mUseManifoldReduction);
        body.SetApplyGyroscopicForce(settings.mApplyGyroscopicForce);
        body.SetFriction(settings.mFriction);
        body.SetRestitution(settings.mRestitution);
        if (settings.HasMassProperties()) {
            auto &motion = *body.GetMotionProperties();
            if (actor.mass && !actor.frozen)
                motion.SetMassProperties(settings.mAllowedDOFs, settings.mMassPropertiesOverride);
            else
                motion.SetKinematicDOFs(settings.mAllowedDOFs);
            motion.SetLinearDamping(settings.mLinearDamping);
            motion.SetAngularDamping(settings.mAngularDamping);
            motion.SetMaxLinearVelocity(settings.mMaxLinearVelocity);
            motion.SetMaxAngularVelocity(settings.mMaxAngularVelocity);
            motion.SetGravityFactor(settings.mGravityFactor);
            if (body.IsKinematic()) {
                // A completed drive can exceed the manual speed limits.
                motion.ProjectVelocitiesToAllowedDOFs();
            } else {
                motion.SetLinearVelocityClamped(motion.GetLinearVelocity());
                motion.SetAngularVelocityClamped(motion.GetAngularVelocity());
            }
            motion.SetAccumulatedForceAndTorque(force, torque);
        }
        newBounds = body.GetWorldSpaceBounds();
        poseChanged = body.GetPosition() != oldOrigin || (wasActive && !body.IsActive());
        mobilityChanged = hasJoints && !(oldMobility == ReadMobility(body));
    }
    if (settings.HasMassProperties())
        bodies.SetMotionQuality(native, settings.mMotionQuality);
    next.native = native;
    next.shapeRevision = actor.publication.sharedGeometryRevision.value_or(record.shapeRevision + 1);
    const bool scaleChanged = next.rootScale != record.rootScale;
    next.poseUpdateIndex = record.poseUpdateIndex;
    record = std::move(next);
    // COM/scale edits normally preserve the body origin and leave the
    // Transform's authored scale alone. Track actual origin or sleep changes.
    if (poseChanged)
        MarkPoseUpdate(bodyId, record);
    m_constraints.NotifyBodyShapeChanged(bodyId, deltaCOM, scaleChanged);
    if (mobilityChanged)
        m_constraints.ResetBodyWarmStart(bodyId);
    bodies.InvalidateContactCache(native);
    // Accepting a nonstatic author model wakes it even while detached.
    if (settings.mMotionType != JPH::EMotionType::Static)
        SetBodyAwake(bodyId, true);
    if (added) {
        // Editing or removing a support must wake its sleeping neighbours too.
        // Keep the two queries local instead of spanning a distant COM change.
        WakeBodiesInBounds(oldBounds, native);
        if (!(oldBounds == newBounds))
            WakeBodiesInBounds(newBounds, native);
    }
    return deltaCOM;
}

} // namespace infernux
