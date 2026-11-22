#include <Jolt/Jolt.h>
#include <Jolt/Physics/Collision/Shape/EmptyShape.h>
#include <Jolt/Physics/Collision/Shape/OffsetCenterOfMassShape.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>

#include "../Collider.h"
#include "../GameObject.h"
#include "../MeshCollider.h"
#include "../Rigidbody.h"
#include "../TagLayerManager.h"
#include "../Transform.h"
#include "../TransformECSStore.h"
#include "PhysicsECSStore.h"
#include "PhysicsShapeAuthoring.h"

#include <utility>

namespace infernux
{
namespace
{
PhysicsActorData *FindActorBinding(PhysicsECSStore &store, PhysicsActorHandle handle, uint32_t bodyId) noexcept
{
    if (bodyId == UINT32_MAX || !store.IsValid(handle))
        return nullptr;
    auto &actor = store.GetActor(handle);
    return actor.bodyId == bodyId ? &actor : nullptr;
}
} // namespace

PhysicsActorPublication PhysicsActorPublication::Capture(const PhysicsActorData &actor,
                                                         const std::vector<Collider *> &members, const Rigidbody *body)
{
    PhysicsActorPublication result;
    result.colliderCenters.reserve(actor.colliders.size());
    for (const auto *collider : actor.colliders)
        if (collider)
            result.colliderCenters.push_back({collider->GetECSHandle(), collider->GetCenterInRigidSpace()});
    for (const auto *member : members)
        if (const auto *mesh = dynamic_cast<const MeshCollider *>(member))
            result.meshPublications.push_back({mesh->GetECSHandle(), mesh->CaptureCookedGeometry()});
    if (body) {
        result.rigidbody = body->GetECSHandle();
        result.rigidbodyComponentId = body->GetComponentID();
        const auto &data = PhysicsECSStore::Instance().GetRigidbody(result.rigidbody);
        if (data.hasLinearVelocity)
            result.linearVelocity = data.linearVelocity;
        if (data.hasAngularVelocity)
            result.angularVelocity = data.angularVelocity;
    }
    return result;
}

void PhysicsActorPublication::AcceptGeometry(PhysicsActorHandle handle, uint32_t bodyId,
                                             uint64_t shapeRevision) const noexcept
{
    auto &store = PhysicsECSStore::Instance();
    auto *actor = FindActorBinding(store, handle, bodyId);
    if (!actor)
        return;
    if (sharedGeometryRevision && *sharedGeometryRevision < actor->shapeRevision)
        return;
    actor->shapeRevision = sharedGeometryRevision.value_or(shapeRevision);
    for (const auto &center : colliderCenters)
        if (store.IsValid(center.collider)) {
            auto &collider = store.GetCollider(center.collider);
            if (collider.actorHandle == handle)
                collider.publishedRigidCenter = center.center;
        }
    for (const auto &publication : meshPublications)
        if (store.IsValid(publication.collider)) {
            auto &collider = store.GetCollider(publication.collider);
            if (collider.actorHandle == handle)
                static_cast<MeshCollider *>(collider.owner)->PublishCookedGeometry(publication.geometry);
        }
}

void PhysicsActorPublication::ConsumeInitialVelocities(PhysicsActorHandle handle, uint32_t bodyId) const noexcept
{
    auto &store = PhysicsECSStore::Instance();
    if (!FindActorBinding(store, handle, bodyId) || !store.IsValid(rigidbody))
        return;
    auto &data = store.GetRigidbody(rigidbody);
    if (data.actorHandle != handle)
        return;
    if (linearVelocity && data.hasLinearVelocity && data.linearVelocity == *linearVelocity)
        data.hasLinearVelocity = false;
    if (angularVelocity && data.hasAngularVelocity && data.angularVelocity == *angularVelocity)
        data.hasAngularVelocity = false;
}

namespace
{
const Rigidbody *ActiveRigidbody(const PhysicsActorData &actor)
{
    return actor.rigidbody && actor.rigidbody->IsEnabled() ? actor.rigidbody : nullptr;
}

JPH::RefConst<JPH::Shape> BuildShape(const PhysicsActorData &actor, const Rigidbody *body,
                                     const glm::vec3 &actorPosition, const glm::quat &actorRotation,
                                     const glm::vec3 &actorScale, std::vector<Collider *> *members)
{
    auto *go = actor.owner;
    const auto inverseRotation = glm::inverse(actorRotation);
    std::vector<std::pair<Collider *, JPH::RefConst<JPH::Shape>>> childShapes;
    const auto &colliders = actor.colliders;
    childShapes.reserve(colliders.size());
    bool complete = true;

    for (auto *col : colliders) {
        if (!col || col->IsBeingDestroyed() || !col->IsEnabled() || !col->GetGameObject()->IsActiveInHierarchy()) {
            continue;
        }

        auto child = col->CreateJoltShape();
        if (child) {
            if (child->GetSubType() == JPH::EShapeSubType::Empty)
                continue; // Empty children have no geometry or compound mass.
            if (col->GetGameObject() != go) {
                const auto *childTransform = col->GetGameObject()->GetTransform();
                const auto position = inverseRotation * (childTransform->GetPosition() - actorPosition);
                const auto rotation = glm::normalize(inverseRotation * childTransform->GetWorldRotation());
                // Each primitive already bakes world scale. Only the rigid
                // relative pose belongs here; a full inverse matrix scales twice.
                child =
                    new JPH::RotatedTranslatedShape(JPH::Vec3(position.x, position.y, position.z),
                                                    JPH::Quat(rotation.x, rotation.y, rotation.z, rotation.w), child);
            }
            childShapes.emplace_back(col, std::move(child));
        } else {
            // A pending or rejected child must not silently disappear from
            // a replacement compound. Visit every child to start its cook.
            complete = false;
        }
    }

    if (!complete)
        return nullptr;

    if (members) {
        members->reserve(childShapes.size());
        for (const auto &child : childShapes)
            members->push_back(child.first);
    }

    JPH::RefConst<JPH::Shape> shape;
    if (childShapes.empty()) {
        shape = new JPH::EmptyShape();
    } else if (childShapes.size() == 1) {
        shape = childShapes.front().second;
    } else {
        JPH::StaticCompoundShapeSettings compoundSettings;
        for (size_t i = 0; i < childShapes.size(); ++i) {
            compoundSettings.AddShape(JPH::Vec3::sZero(), JPH::Quat::sIdentity(), childShapes[i].second,
                                      static_cast<uint32_t>(i));
        }
        auto result = compoundSettings.Create();
        if (result.HasError())
            return nullptr;
        shape = result.Get();
    }
    if (body) {
        if (const auto &properties = body->GetAuthoredMassProperties()) {
            const auto center = properties->center * actorScale;
            const JPH::Vec3 offset = JPH::Vec3(center.x, center.y, center.z) - shape->GetCenterOfMass();
            shape = new JPH::OffsetCenterOfMassShape(shape, offset);
        }
    }
    return shape;
}
} // namespace

JPH::RefConst<JPH::Shape> BuildActorShape(const PhysicsActorData &actor)
{
    const auto *transform = actor.owner->GetTransform();
    return BuildShape(actor, ActiveRigidbody(actor), transform->GetPosition(), transform->GetWorldRotation(),
                      transform->GetWorldScale(), nullptr);
}

std::optional<PhysicsActorAuthoring> PhysicsActorAuthoring::Capture(const PhysicsActorData &actor)
{
    TransformECSStore::PhysicsPoseReadScope authoring(TransformECSStore::Instance());
    PhysicsActorAuthoring result{};
    const auto *transform = actor.owner->GetTransform();
    result.position = transform->GetPosition();
    result.rotation = transform->GetWorldRotation();
    result.scale = transform->GetWorldScale();
    result.objectId = actor.owner->GetID();
    result.objectLayer = actor.owner->GetLayer();

    const auto *body = ActiveRigidbody(actor);
    std::vector<Collider *> members;
    result.shape = BuildShape(actor, body, result.position, result.rotation, result.scale, &members);
    if (!result.shape)
        return std::nullopt;

    auto &layers = TagLayerManager::Instance();
    result.surfaces.reserve(members.size());
    for (const auto *collider : members) {
        const auto *object = collider->GetGameObject();
        const auto layer = object->GetLayer();
        result.surfaces.push_back({PhysicsTargetReference(collider), collider->GetComponentID(),
                                   collider->GetLifetimeGeneration(), static_cast<uint32_t>(layer),
                                   layers.GetLayerCollisionMask(layer), collider->GetFriction(),
                                   collider->GetBounciness(), collider->GetFrictionCombine(),
                                   collider->GetBounceCombine(), collider->IsTrigger(), object->GetID()});
    }
    if (body) {
        result.body = PhysicsRigidbodyAuthoring{PhysicsTargetReference(body),
                                                std::nullopt,
                                                body->GetConstraints(),
                                                body->GetCollisionDetectionMode(),
                                                body->GetDrag(),
                                                body->GetAngularDrag(),
                                                body->GetMaxLinearVelocity(),
                                                body->GetMaxAngularVelocity(),
                                                body->IsKinematic(),
                                                body->GetUseGravity() ? 1.0f : 0.0f};
        const auto &properties = body->GetAuthoredMassProperties();
        // A prescribed non-convex boundary needs no dynamic inertia.
        if (!result.body->kinematic || properties || !result.shape->MustBeStatic())
            result.body->mass = PrepareBodyMass(result.shape.GetPtr(), body->GetMass(),
                                                properties ? &*properties : nullptr, result.scale);
    }
    result.publication = PhysicsActorPublication::Capture(actor, members, body);
    return result;
}

} // namespace infernux
