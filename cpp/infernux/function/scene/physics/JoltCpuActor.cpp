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
    const auto *transform = actor.owner->GetTransform();
    const auto position = transform->GetPosition();
    const auto rotation = transform->GetWorldRotation();
    // Jolt cannot solve a dynamic body with all six axes locked. Such a body
    // is an immovable kinematic boundary internally; the component stays a
    // dynamic Rigidbody, with its real mass available when axes are unlocked.
    const bool frozen = body && body->GetConstraints() == static_cast<int>(RigidbodyConstraints::FreezeAll);
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
        prepared.mass = PrepareBodyMass(shape.GetPtr(), body->GetMass(), properties ? &*properties : nullptr,
                                        transform->GetWorldScale());
        const auto &mass = *prepared.mass;
        const auto axes = JPH::Mat44::sRotation(mass.axes);
        settings.mOverrideMassProperties = JPH::EOverrideMassProperties::MassAndInertiaProvided;
        settings.mMassPropertiesOverride.mMass = mass.mass;
        settings.mMassPropertiesOverride.mInertia =
            axes * JPH::Mat44::sScale(mass.inverse.Reciprocal()) * axes.Transposed3x3();
        settings.mAllowedDOFs =
            frozen ? JPH::EAllowedDOFs::All : static_cast<JPH::EAllowedDOFs>((~(body->GetConstraints() >> 1)) & 0x3f);
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

        prepared.initialRigidbody = body->GetECSHandle();
        const auto &data = PhysicsECSStore::Instance().GetRigidbody(prepared.initialRigidbody);
        prepared.hasInitialLinearVelocity = data.hasLinearVelocity;
        prepared.hasInitialAngularVelocity = data.hasAngularVelocity;
        if (!frozen && data.hasLinearVelocity)
            settings.mLinearVelocity = JPH::Vec3(data.linearVelocity.x, data.linearVelocity.y, data.linearVelocity.z);
        if (!frozen && data.hasAngularVelocity)
            settings.mAngularVelocity =
                JPH::Vec3(data.angularVelocity.x, data.angularVelocity.y, data.angularVelocity.z);
    }
    return prepared;
}

JPH::BodyID JoltCpuScene::CreateBody(uint32_t bodyId, const PreparedCpuActor &actor)
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
    auto settings = actor.settings;
    // Body creation's raw setters require already-clamped velocities. Install
    // authored velocities through the same native constrained/clamped path as
    // later commands, before the body is exposed to broadphase.
    settings.mLinearVelocity = settings.mAngularVelocity = JPH::Vec3::sZero();
    const auto native = CreateBody(bodyId, settings, std::move(record));
    if (actor.mass) {
        JPH::BodyLockWrite lock(m_system.GetBodyLockInterfaceNoLock(), native);
        auto &motion = *lock.GetBody().GetMotionProperties();
        motion.SetLinearVelocityClamped(actor.settings.mLinearVelocity);
        motion.SetAngularVelocityClamped(actor.settings.mAngularVelocity);
    }
    return native;
}

} // namespace infernux
