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
    leaf.material = {collider.GetFriction(), collider.GetBounciness(), 0.0f, 0.0f};
    leaf.surface = {static_cast<uint32_t>(layer), TagLayerManager::Instance().GetLayerCollisionMask(layer),
                    static_cast<uint32_t>(collider.GetFrictionCombine()),
                    static_cast<uint32_t>(collider.GetBounceCombine())};
    leaf.meta.w = collider.IsTrigger() ? JPH::GPU::ShapeSensor : 0;
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
    if (rigidbody) {
        if (rigidbody->GetConstraints() != 0)
            throw std::invalid_argument("Jolt GPU actor preparation does not yet support locked degrees of freedom");
        if (rigidbody->GetCollisionDetectionMode() != static_cast<int>(CollisionDetectionMode::Discrete))
            throw std::invalid_argument(
                "Jolt GPU actor preparation does not yet support continuous collision detection");
    }
    if (members.size() > std::numeric_limits<uint32_t>::max())
        throw std::length_error("Jolt GPU actor exceeds its member index range");
    PreparedGpuActor prepared{};
    prepared.geometry = PrepareJoltGpuShape(*shape, static_cast<uint32_t>(members.size()));
    if (prepared.geometry.leaves.size() > std::numeric_limits<uint32_t>::max())
        throw std::length_error("Jolt GPU actor exceeds its leaf index range");
    const auto *transform = actor.owner->GetTransform();
    const auto rotation = transform->GetWorldRotation();
    const auto position = transform->GetPosition();
    const auto center = prepared.geometry.centerOfMass;
    const auto positionCOM = position + rotation * glm::vec3(center.GetX(), center.GetY(), center.GetZ());
    prepared.initialState.positionCOM = {positionCOM.x, positionCOM.y, positionCOM.z, 0.0f};
    prepared.initialState.rotation = {rotation.x, rotation.y, rotation.z, rotation.w};
    const auto motion = rigidbody ? (rigidbody->IsKinematic() ? JPH::GPU::MotionKinematic : JPH::GPU::MotionDynamic)
                                  : JPH::GPU::MotionStatic;
    // Scene assigns the actual global range at the ordered model boundary.
    prepared.body.meta = {0, static_cast<uint32_t>(prepared.geometry.leaves.size()), motion, worldIndex};
    prepared.body.inertiaRotation = {0.0f, 0.0f, 0.0f, 1.0f};
    if (rigidbody) {
        const auto &properties = rigidbody->GetAuthoredMassProperties();
        const auto mass = PrepareBodyMass(shape.GetPtr(), rigidbody->GetMass(), properties ? &*properties : nullptr,
                                          transform->GetWorldScale());
        prepared.body.mass.x = 1.0f / mass.mass;
        prepared.body.mass.w = rigidbody->GetUseGravity() ? 1.0f : 0.0f;
        prepared.body.invInertiaDiagonal = {mass.inverse.GetX(), mass.inverse.GetY(), mass.inverse.GetZ(), 0.0f};
        prepared.body.inertiaRotation = {mass.axes.GetX(), mass.axes.GetY(), mass.axes.GetZ(), mass.axes.GetW()};
        prepared.body.damping = {rigidbody->GetDrag(), rigidbody->GetAngularDrag(), rigidbody->GetMaxLinearVelocity(),
                                 rigidbody->GetMaxAngularVelocity()};

        prepared.initialRigidbody = rigidbody->GetECSHandle();
        prepared.rigidbodyComponentId = rigidbody->GetComponentID();
        const auto &data = PhysicsECSStore::Instance().GetRigidbody(prepared.initialRigidbody);
        prepared.hasInitialLinearVelocity = data.hasLinearVelocity;
        prepared.hasInitialAngularVelocity = data.hasAngularVelocity;
        if (data.hasLinearVelocity) {
            const auto velocity = data.linearVelocity;
            prepared.initialState.linearVelocity = {velocity.x, velocity.y, velocity.z, 0.0f};
        }
        if (data.hasAngularVelocity) {
            const auto velocity = data.angularVelocity;
            prepared.initialState.angularVelocity = {velocity.x, velocity.y, velocity.z, 0.0f};
        }
    }
    for (auto &leaf : prepared.geometry.leaves) {
        const auto *collider = members[leaf.memberIndex];
        leaf.model.meta.y = bodySlot;
        RefreshGpuActorSurface(leaf.model, *collider);
        const uint64_t id = collider->GetComponentID();
        leaf.model.source.x = static_cast<uint32_t>(id);
        leaf.model.source.y = static_cast<uint32_t>(id >> 32);
    }
    prepared.members = std::move(members);
    prepared.colliderCenters.reserve(actor.colliders.size());
    for (const auto *collider : actor.colliders)
        prepared.colliderCenters.push_back({collider->GetECSHandle(), collider->GetCenterInRigidSpace()});
    prepared.authoredShape = std::move(shape);
    return prepared;
}

} // namespace infernux
