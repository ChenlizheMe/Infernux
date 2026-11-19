#include <Jolt/Jolt.h>

#include "../Collider.h"
#include "../GameObject.h"
#include "../Rigidbody.h"
#include "../TagLayerManager.h"
#include "../Transform.h"
#include "../TransformECSStore.h"
#include "JoltGpuActor.h"
#include "PhysicsECSStore.h"
#include "PhysicsMass.h"
#include "PhysicsShapeAuthoring.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace infernux
{

static_assert(static_cast<uint32_t>(PhysicsMaterialCombine::Average) == JPH::GPU::MaterialCombineAverage &&
              static_cast<uint32_t>(PhysicsMaterialCombine::Minimum) == JPH::GPU::MaterialCombineMinimum &&
              static_cast<uint32_t>(PhysicsMaterialCombine::Multiply) == JPH::GPU::MaterialCombineMultiply &&
              static_cast<uint32_t>(PhysicsMaterialCombine::Maximum) == JPH::GPU::MaterialCombineMaximum);

void RefreshGpuActorSurface(JPH::GPU::ShapeModel &leaf, const Collider &collider)
{
    const int layer = collider.GetGameObject()->GetLayer();
    leaf.material.x = collider.GetFriction();
    leaf.material.y = collider.GetBounciness();
    leaf.surface = {static_cast<uint32_t>(layer), TagLayerManager::Instance().GetLayerCollisionMask(layer),
                    static_cast<uint32_t>(collider.GetFrictionCombine()),
                    static_cast<uint32_t>(collider.GetBounceCombine())};
    leaf.meta.w = (leaf.meta.w & ~JPH::GPU::ShapeSensor) | (collider.IsTrigger() ? JPH::GPU::ShapeSensor : 0u);
}

namespace
{
PreparedGpuActor PrepareShapeModel(JPH::RefConst<JPH::Shape> shape, size_t memberCount, const glm::vec3 &position,
                                   const glm::quat &rotation, const glm::vec3 &scale, uint32_t motion,
                                   uint32_t worldIndex)
{
    if (memberCount > std::numeric_limits<uint32_t>::max())
        throw std::length_error("Jolt GPU actor exceeds its member index range");
    PreparedGpuActor prepared{};
    prepared.geometry = PrepareJoltGpuShape(*shape, static_cast<uint32_t>(memberCount));
    if (prepared.geometry.leaves.size() > std::numeric_limits<uint32_t>::max())
        throw std::length_error("Jolt GPU actor exceeds its leaf index range");
    prepared.rootScale = {scale.x, scale.y, scale.z, 0};
    prepared.SetOriginPose(position, rotation);
    prepared.body.meta = {0, static_cast<uint32_t>(prepared.geometry.leaves.size()), motion, worldIndex};
    const auto rootBounds = shape->GetLocalBounds();
    const auto rootExtent = rootBounds.GetExtent(), rootBoundsCenter = rootBounds.GetCenter();
    prepared.sleep.rootHalfExtent = {rootExtent.GetX(), rootExtent.GetY(), rootExtent.GetZ(), 0};
    prepared.sleep.rootBoundsCenter = {rootBoundsCenter.GetX(), rootBoundsCenter.GetY(), rootBoundsCenter.GetZ(), 0};
    prepared.sleep.policy = {1u, 0u, 0u, 0u};
    prepared.body.inertiaRotation = {0.0f, 0.0f, 0.0f, 1.0f};
    prepared.authoredShape = std::move(shape);
    return prepared;
}
} // namespace

void PreparedGpuActor::SetOriginPose(const glm::vec3 &position, const glm::quat &rotation)
{
    const auto center = geometry.centerOfMass;
    const auto positionCOM = position + rotation * glm::vec3(center.GetX(), center.GetY(), center.GetZ());
    initialState.positionCOM = {positionCOM.x, positionCOM.y, positionCOM.z, 0.0f};
    initialState.rotation = {rotation.x, rotation.y, rotation.z, rotation.w};
}

std::optional<PreparedGpuActor> PrepareJoltGpuActor(const PhysicsActorData &actor, uint32_t bodySlot,
                                                    uint32_t worldIndex)
{
    TransformECSStore::PhysicsPoseReadScope authoring(TransformECSStore::Instance());
    std::vector<Collider *> members;
    auto shape = BuildActorShape(actor, nullptr, nullptr, &members);
    if (!shape)
        return std::nullopt;

    const auto *rigidbody = actor.rigidbody && actor.rigidbody->IsEnabled() ? actor.rigidbody : nullptr;
    if (rigidbody && (rigidbody->GetConstraints() & ~126) != 0)
        throw std::invalid_argument("Rigidbody constraints contain unknown bits");
    const auto *transform = actor.owner->GetTransform();
    const auto motion = rigidbody ? (rigidbody->IsKinematic() ? JPH::GPU::MotionKinematic : JPH::GPU::MotionDynamic)
                                  : JPH::GPU::MotionStatic;
    auto prepared = PrepareShapeModel(shape, members.size(), transform->GetPosition(), transform->GetWorldRotation(),
                                      transform->GetWorldScale(), motion, worldIndex);
    prepared.publication = PhysicsActorPublication::Capture(actor, members, rigidbody);
    if (rigidbody) {
        const auto &properties = rigidbody->GetAuthoredMassProperties();
        if (!rigidbody->IsKinematic() || properties || !shape->MustBeStatic()) {
            const auto mass = PrepareBodyMass(shape.GetPtr(), rigidbody->GetMass(), properties ? &*properties : nullptr,
                                              transform->GetWorldScale());
            prepared.body.mass.x = 1.0f / mass.mass;
            prepared.body.invInertiaDiagonal = {mass.inverse.GetX(), mass.inverse.GetY(), mass.inverse.GetZ(), 0.0f};
            prepared.body.inertiaRotation = {mass.axes.GetX(), mass.axes.GetY(), mass.axes.GetZ(), mass.axes.GetW()};
        }
        prepared.body.mass.y = float((rigidbody->GetConstraints() >> 1) & 7);
        prepared.body.mass.z = float((rigidbody->GetConstraints() >> 4) & 7);
        prepared.body.mass.w = rigidbody->GetUseGravity() ? 1.0f : 0.0f;
        prepared.body.damping = {rigidbody->GetDrag(), rigidbody->GetAngularDrag(), rigidbody->GetMaxLinearVelocity(),
                                 rigidbody->GetMaxAngularVelocity()};
        const auto detection = static_cast<CollisionDetectionMode>(rigidbody->GetCollisionDetectionMode());
        if (motion == JPH::GPU::MotionDynamic && detection != CollisionDetectionMode::Discrete) {
            const float radius = shape->GetInnerRadius();
            prepared.body.invInertiaDiagonal.w =
                detection == CollisionDetectionMode::ContinuousDynamic ? -radius : radius;
        }

        if (const auto &linear = prepared.publication.linearVelocity)
            prepared.initialState.linearVelocity = {linear->x, linear->y, linear->z, 0.0f};
        if (const auto &angular = prepared.publication.angularVelocity)
            prepared.initialState.angularVelocity = {angular->x, angular->y, angular->z, 0.0f};
    }
    for (auto &leaf : prepared.geometry.leaves) {
        const auto *collider = members[leaf.memberIndex];
        leaf.model.meta.y = bodySlot;
        RefreshGpuActorSurface(leaf.model, *collider);
        const uint64_t id = collider->GetComponentID();
        leaf.model.source.x = static_cast<uint32_t>(id);
        leaf.model.source.y = static_cast<uint32_t>(id >> 32);
    }
    if (!rigidbody && !members.empty() &&
        std::all_of(members.begin(), members.end(), [](const Collider *member) { return member->IsTrigger(); }))
        prepared.sleep.policy.x |= 2u;
    prepared.members.reserve(members.size());
    for (const auto *member : members)
        prepared.members.push_back({PhysicsTargetReference(member), member->GetComponentID(),
                                    member->GetLifetimeGeneration(), member->GetGameObject()->GetID()});
    return prepared;
}

} // namespace infernux
