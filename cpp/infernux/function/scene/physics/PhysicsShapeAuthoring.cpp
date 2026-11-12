#include <Jolt/Jolt.h>
#include <Jolt/Physics/Collision/Shape/EmptyShape.h>
#include <Jolt/Physics/Collision/Shape/OffsetCenterOfMassShape.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>

#include "../Collider.h"
#include "../GameObject.h"
#include "../Rigidbody.h"
#include "../Transform.h"
#include "PhysicsECSStore.h"
#include "PhysicsShapeAuthoring.h"

#include <utility>

namespace infernux
{

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
