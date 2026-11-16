#include <Jolt/Jolt.h>
#include <Jolt/Physics/Collision/Shape/EmptyShape.h>
#include <Jolt/Physics/Collision/Shape/OffsetCenterOfMassShape.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>

#include "../Collider.h"
#include "../GameObject.h"
#include "../MeshCollider.h"
#include "../Rigidbody.h"
#include "../Transform.h"
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

JPH::RefConst<JPH::Shape> BuildActorShape(const PhysicsActorData &actor, const Collider *exclude, size_t *outShapeCount,
                                          std::vector<Collider *> *outMembers)
{
    auto *go = actor.owner;
    std::vector<std::pair<Collider *, JPH::RefConst<JPH::Shape>>> childShapes;
    const auto &colliders = actor.colliders;
    childShapes.reserve(colliders.size());
    bool complete = true;

    for (auto *col : colliders) {
        if (!col || col == exclude || col->IsBeingDestroyed() || !col->IsEnabled() ||
            !col->GetGameObject()->IsActiveInHierarchy()) {
            continue;
        }

        auto child = col->CreateJoltShape();
        if (child) {
            if (child->GetSubType() == JPH::EShapeSubType::Empty)
                continue; // Empty children have no geometry or compound mass.
            if (col->GetGameObject() != go) {
                const auto *actorTransform = go->GetTransform();
                const auto *childTransform = col->GetGameObject()->GetTransform();
                const auto inverseRotation = glm::inverse(actorTransform->GetWorldRotation());
                const auto position = inverseRotation * (childTransform->GetPosition() - actorTransform->GetPosition());
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

    if (!complete) {
        if (outShapeCount)
            *outShapeCount = 0;
        return nullptr;
    }

    if (outShapeCount)
        *outShapeCount = childShapes.size();
    if (outMembers) {
        outMembers->reserve(childShapes.size());
        for (const auto &child : childShapes)
            outMembers->push_back(child.first);
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
    if (auto *body = actor.rigidbody; body && body->IsEnabled()) {
        if (const auto &properties = body->GetAuthoredMassProperties()) {
            const auto center = properties->center * go->GetTransform()->GetWorldScale();
            const JPH::Vec3 offset = JPH::Vec3(center.x, center.y, center.z) - shape->GetCenterOfMass();
            shape = new JPH::OffsetCenterOfMassShape(shape, offset);
        }
    }
    return shape;
}

} // namespace infernux
