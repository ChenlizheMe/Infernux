#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/Body/MotionProperties.h>
#include <Jolt/Physics/Collision/Shape/CompoundShape.h>
#include <Jolt/Physics/Collision/Shape/OffsetCenterOfMassShape.h>

#include "../Collider.h"
#include "../GameObject.h"
#include "../Rigidbody.h"
#include "../Transform.h"
#include "../TransformECSStore.h"
#include "JoltCpuActor.h"
#include "JoltCpuScene.h"
#include "PhysicsLayers.h"
#include "PhysicsShapeAuthoring.h"

#include <algorithm>
#include <stdexcept>

namespace infernux
{
namespace
{
JPH::ObjectLayer BodyLayer(const PhysicsActorData &actor, const std::vector<Collider *> &members, bool moving)
{
    if (members.empty())
        return PhysicsObjectLayers::Encode(actor.owner->GetLayer(), moving);
    const int layer = members.front()->GetGameObject()->GetLayer();
    for (const auto *member : members)
        if (member->GetGameObject()->GetLayer() != layer)
            return moving ? PhysicsObjectLayers::MIXED_MOVING : PhysicsObjectLayers::MIXED_NON_MOVING;
    return PhysicsObjectLayers::Encode(layer, moving);
}
} // namespace

std::optional<PreparedCpuActor> PrepareJoltCpuActor(const PhysicsActorData &actor)
{
    TransformECSStore::PhysicsPoseReadScope authoring(TransformECSStore::Instance());
    PreparedCpuActor prepared;
    std::vector<Collider *> members;
    auto shape = BuildActorShape(actor, nullptr, nullptr, &members);
    if (!shape)
        return std::nullopt;
    prepared.surfaces.reserve(members.size());
    for (const auto *collider : members) {
        const auto layer = collider->GetGameObject()->GetLayer();
        prepared.surfaces.push_back({PhysicsTargetReference(collider), collider->GetComponentID(),
                                     collider->GetLifetimeGeneration(), static_cast<uint32_t>(layer),
                                     TagLayerManager::Instance().GetLayerCollisionMask(layer), collider->GetFriction(),
                                     collider->GetBounciness(), collider->GetFrictionCombine(),
                                     collider->GetBounceCombine(), collider->IsTrigger()});
    }

    const auto *body = actor.rigidbody && actor.rigidbody->IsEnabled() ? actor.rigidbody : nullptr;
    prepared.publication = PhysicsActorPublication::Capture(actor, members, body);
    const auto *transform = actor.owner->GetTransform();
    const auto position = transform->GetPosition();
    const auto rotation = transform->GetWorldRotation();
    // Jolt cannot solve a dynamic body with all six axes locked. Such a body
    // is an immovable kinematic boundary internally; the component stays a
    // dynamic Rigidbody, with its real mass available when axes are unlocked.
    const bool frozen = body && body->GetConstraints() == static_cast<int>(RigidbodyConstraints::FreezeAll);
    prepared.frozen = frozen;
    prepared.authoredDynamic = body && !body->IsKinematic();
    const auto motion = !body                           ? JPH::EMotionType::Static
                        : body->IsKinematic() || frozen ? JPH::EMotionType::Kinematic
                                                        : JPH::EMotionType::Dynamic;
    auto &settings = prepared.settings;
    settings = JPH::BodyCreationSettings(shape, JPH::RVec3(position.x, position.y, position.z),
                                         JPH::Quat(rotation.x, rotation.y, rotation.z, rotation.w), motion,
                                         BodyLayer(actor, members, body != nullptr));
    settings.mAllowDynamicOrKinematic = body != nullptr;
    settings.mApplyGyroscopicForce = true;
    settings.mIsSensor = !prepared.surfaces.empty() && std::all_of(prepared.surfaces.begin(), prepared.surfaces.end(),
                                                                   [](const auto &surface) { return surface.sensor; });
    // Preserve per-Collider material/filter/trigger identity in compounds.
    settings.mUseManifoldReduction = members.size() <= 1;
    settings.mCollideKinematicVsNonDynamic = true;
    if (!prepared.surfaces.empty()) {
        settings.mFriction = prepared.surfaces.front().friction;
        settings.mRestitution = prepared.surfaces.front().restitution;
    }
    if (body) {
        const auto &properties = body->GetAuthoredMassProperties();
        // Prescribed non-convex motion needs DOFs and velocity, not invented
        // volume or inertia. A later dynamic conversion prepares real mass.
        if (!body->IsKinematic() || properties || !shape->MustBeStatic()) {
            prepared.mass = PrepareBodyMass(shape.GetPtr(), body->GetMass(), properties ? &*properties : nullptr,
                                            transform->GetWorldScale());
            const auto &mass = *prepared.mass;
            const auto axes = JPH::Mat44::sRotation(mass.axes);
            settings.mOverrideMassProperties = JPH::EOverrideMassProperties::MassAndInertiaProvided;
            settings.mMassPropertiesOverride.mMass = mass.mass;
            settings.mMassPropertiesOverride.mInertia =
                axes * JPH::Mat44::sScale(mass.inverse.Reciprocal()) * axes.Transposed3x3();
        }
        settings.mAllowedDOFs = static_cast<JPH::EAllowedDOFs>((~(body->GetConstraints() >> 1)) & 0x3f);
        prepared.collisionDetectionMode = body->GetCollisionDetectionMode();
        settings.mMotionQuality =
            motion == JPH::EMotionType::Dynamic && !members.empty() && prepared.collisionDetectionMode != 0
                ? JPH::EMotionQuality::LinearCast
                : JPH::EMotionQuality::Discrete;
        settings.mLinearDamping = body->GetDrag();
        settings.mAngularDamping = body->GetAngularDrag();
        settings.mMaxLinearVelocity = body->GetMaxLinearVelocity();
        settings.mMaxAngularVelocity = body->GetMaxAngularVelocity();
        settings.mGravityFactor = body->GetUseGravity() ? 1.0f : 0.0f;

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
    record.shape = actor.settings.GetShape();
    if (actor.surfaces.size() > 1) {
        const auto *root = record.shape.GetPtr();
        if (root->GetSubType() == JPH::EShapeSubType::OffsetCenterOfMass)
            root = static_cast<const JPH::OffsetCenterOfMassShape *>(root)->GetInnerShape();
        record.memberCompound = static_cast<const JPH::CompoundShape *>(root);
    }
    record.surfaces = actor.surfaces;
    record.collisionDetectionMode = actor.collisionDetectionMode;
    record.authoredDynamic = actor.authoredDynamic;
    return record;
}

JPH::BodyID JoltCpuScene::CreateBody(uint32_t bodyId, const PreparedCpuActor &actor)
{
    auto settings = actor.settings;
    // Body creation's raw setters require already-clamped velocities. Install
    // authored velocities through the same native constrained/clamped path as
    // later commands, before the body is exposed to broadphase.
    settings.mLinearVelocity = settings.mAngularVelocity = JPH::Vec3::sZero();
    const auto native = CreateBody(bodyId, settings, CaptureBodyRecord(actor));
    if (settings.HasMassProperties()) {
        JPH::BodyLockWrite lock(m_system.GetBodyLockInterfaceNoLock(), native);
        auto &motion = *lock.GetBody().GetMotionProperties();
        motion.SetLinearVelocityClamped(actor.settings.mLinearVelocity);
        motion.SetAngularVelocityClamped(actor.settings.mAngularVelocity);
    }
    return native;
}

JPH::Vec3 JoltCpuScene::UpdateBody(uint32_t bodyId, const PreparedCpuActor &actor)
{
    auto next = CaptureBodyRecord(actor);
    auto &record = m_bodies.at(bodyId);
    const auto native = record.native;
    const auto &settings = actor.settings;
    const auto deltaCOM = next.shape->GetCenterOfMass() - record.shape->GetCenterOfMass();
    auto &bodies = m_system.GetBodyInterface();
    const bool added = bodies.IsAdded(native);
    const bool retainLoads = record.authoredDynamic && actor.authoredDynamic;
    JPH::AABox oldBounds, newBounds;
    JPH::Vec3 force = JPH::Vec3::sZero(), torque = JPH::Vec3::sZero();
    {
        JPH::BodyLockRead lock(m_system.GetBodyLockInterface(), native);
        const auto &body = lock.GetBody();
        if (settings.HasMassProperties() && !body.CanBeKinematicOrDynamic())
            throw std::logic_error("Adding a Rigidbody to a CPU static actor requires body allocation");
        oldBounds = body.GetWorldSpaceBounds();
        if (retainLoads) {
            force = body.GetMotionProperties()->GetAccumulatedForce();
            torque = body.GetMotionProperties()->GetAccumulatedTorque();
        }
    }
    // SetShape retains the current body-origin pose while updating its COM and
    // broadphase bounds. Configuration edits never teleport to a stale author pose.
    bodies.SetShape(native, next.shape, false, JPH::EActivation::DontActivate);
    bodies.SetObjectLayer(native, settings.mObjectLayer);
    bodies.SetMotionType(native, settings.mMotionType, JPH::EActivation::DontActivate);
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
            motion.SetLinearVelocityClamped(actor.frozen ? JPH::Vec3::sZero() : motion.GetLinearVelocity());
            motion.SetAngularVelocityClamped(actor.frozen ? JPH::Vec3::sZero() : motion.GetAngularVelocity());
            if (retainLoads)
                motion.SetAccumulatedForceAndTorque(force, torque);
        }
        newBounds = body.GetWorldSpaceBounds();
    }
    if (settings.HasMassProperties())
        bodies.SetMotionQuality(native, settings.mMotionQuality);
    next.native = native;
    record = std::move(next);
    bodies.InvalidateContactCache(native);
    if (added) {
        bodies.ActivateBody(native);
        // Editing or removing a support must wake its sleeping neighbours too.
        // Keep the two queries local instead of spanning a distant COM change.
        WakeBodiesInBounds(oldBounds);
        if (!(oldBounds == newBounds))
            WakeBodiesInBounds(newBounds);
    }
    return deltaCOM;
}

} // namespace infernux
