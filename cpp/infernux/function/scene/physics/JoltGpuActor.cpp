#include <Jolt/Jolt.h>

#include "../Collider.h"
#include "../GameObject.h"
#include "../Rigidbody.h"
#include "../TagLayerManager.h"
#include "../Transform.h"
#include "JoltGpuActor.h"
#include "PhysicsECSStore.h"
#include "PhysicsMass.h"
#include "PhysicsShapeAuthoring.h"

#include <stdexcept>

namespace infernux
{

static_assert(static_cast<uint32_t>(PhysicsMaterialCombine::Average) == JPH::GPU::MaterialCombineAverage &&
              static_cast<uint32_t>(PhysicsMaterialCombine::Minimum) == JPH::GPU::MaterialCombineMinimum &&
              static_cast<uint32_t>(PhysicsMaterialCombine::Multiply) == JPH::GPU::MaterialCombineMultiply &&
              static_cast<uint32_t>(PhysicsMaterialCombine::Maximum) == JPH::GPU::MaterialCombineMaximum);

void RefreshGpuActorSurface(JPH::GPU::BodyModel &body, const Collider *collider, const GameObject &owner)
{
    const int layer = collider ? collider->GetGameObject()->GetLayer() : owner.GetLayer();
    body.mass.y = collider ? collider->GetFriction() : 0.0f;
    body.mass.z = collider ? collider->GetBounciness() : 0.0f;
    body.surface = {static_cast<uint32_t>(layer), TagLayerManager::Instance().GetLayerCollisionMask(layer),
                    static_cast<uint32_t>(collider ? collider->GetFrictionCombine() : PhysicsMaterialCombine::Average),
                    static_cast<uint32_t>(collider ? collider->GetBounceCombine() : PhysicsMaterialCombine::Average)};
}

std::optional<PreparedGpuActor> PrepareJoltGpuActor(const PhysicsActorData &actor, uint32_t shapeSlot,
                                                    uint32_t worldIndex)
{
    std::vector<Collider *> members;
    auto shape = BuildActorShape(actor, nullptr, nullptr, &members);
    if (!shape)
        return std::nullopt;

    const auto *rigidbody = actor.rigidbody && actor.rigidbody->IsEnabled() ? actor.rigidbody : nullptr;
    if (rigidbody) {
        if (rigidbody->IsKinematic())
            throw std::invalid_argument("Jolt GPU actor preparation does not yet support kinematic motion");
        if (rigidbody->GetConstraints() != 0)
            throw std::invalid_argument("Jolt GPU actor preparation does not yet support locked degrees of freedom");
        if (rigidbody->GetCollisionDetectionMode() != static_cast<int>(CollisionDetectionMode::Discrete))
            throw std::invalid_argument(
                "Jolt GPU actor preparation does not yet support continuous collision detection");
    }
    for (const auto *collider : members)
        if (collider->IsTrigger())
            throw std::invalid_argument("Jolt GPU actor preparation does not yet support trigger contacts");

    PreparedGpuActor prepared{};
    prepared.geometry = PrepareJoltGpuShape(*shape);
    if (rigidbody && prepared.geometry.type == JPH::GPU::ShapeBox)
        throw std::invalid_argument("Jolt GPU actor preparation does not yet support dynamic Box collision");

    const auto *transform = actor.owner->GetTransform();
    const auto rotation = transform->GetWorldRotation();
    const auto position = transform->GetPosition();
    const auto center = prepared.geometry.centerOfMass;
    const auto positionCOM = position + rotation * glm::vec3(center.GetX(), center.GetY(), center.GetZ());
    prepared.initialState.positionCOM = {positionCOM.x, positionCOM.y, positionCOM.z, 0.0f};
    prepared.initialState.rotation = {rotation.x, rotation.y, rotation.z, rotation.w};
    prepared.body.meta = {shapeSlot, prepared.geometry.type,
                          rigidbody ? JPH::GPU::MotionDynamic : JPH::GPU::MotionStatic, worldIndex};
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

        const auto &data = PhysicsECSStore::Instance().GetRigidbody(rigidbody->GetECSHandle());
        if (data.hasLinearVelocity) {
            const auto velocity = data.linearVelocity;
            prepared.initialState.linearVelocity = {velocity.x, velocity.y, velocity.z, 0.0f};
        }
        if (data.hasAngularVelocity) {
            const auto velocity = data.angularVelocity;
            prepared.initialState.angularVelocity = {velocity.x, velocity.y, velocity.z, 0.0f};
        }
    }
    prepared.surfaceCollider = members.empty() ? nullptr : members.front();
    RefreshGpuActorSurface(prepared.body, prepared.surfaceCollider, *actor.owner);
    prepared.authoredShape = std::move(shape);
    return prepared;
}

} // namespace infernux
