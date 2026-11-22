#include <Jolt/Jolt.h>

#include "../Rigidbody.h"
#include "JoltGpuActor.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace infernux
{

static_assert(static_cast<uint32_t>(PhysicsMaterialCombine::Average) == JPH::GPU::MaterialCombineAverage &&
              static_cast<uint32_t>(PhysicsMaterialCombine::Minimum) == JPH::GPU::MaterialCombineMinimum &&
              static_cast<uint32_t>(PhysicsMaterialCombine::Multiply) == JPH::GPU::MaterialCombineMultiply &&
              static_cast<uint32_t>(PhysicsMaterialCombine::Maximum) == JPH::GPU::MaterialCombineMaximum);

namespace
{
void WriteSurface(JPH::GPU::ShapeModel &leaf, const PhysicsSurfaceAuthoring &surface)
{
    leaf.material.x = surface.friction;
    leaf.material.y = surface.restitution;
    leaf.surface = {surface.layer, surface.collisionMask, static_cast<uint32_t>(surface.frictionCombine),
                    static_cast<uint32_t>(surface.bounceCombine)};
    leaf.meta.w = (leaf.meta.w & ~JPH::GPU::ShapeSensor) | (surface.sensor ? JPH::GPU::ShapeSensor : 0u);
    leaf.source.x = static_cast<uint32_t>(surface.colliderId);
    leaf.source.y = static_cast<uint32_t>(surface.colliderId >> 32);
}

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

PreparedGpuActor PrepareJoltGpuActor(PhysicsActorAuthoring actor, uint32_t bodySlot, uint32_t worldIndex)
{
    const auto &body = actor.body;
    if (body && (body->constraints & ~126) != 0)
        throw std::invalid_argument("Rigidbody constraints contain unknown bits");
    const auto motion =
        body ? (body->kinematic ? JPH::GPU::MotionKinematic : JPH::GPU::MotionDynamic) : JPH::GPU::MotionStatic;
    auto prepared = PrepareShapeModel(std::move(actor.shape), actor.surfaces.size(), actor.position, actor.rotation,
                                      actor.scale, motion, worldIndex);
    prepared.publication = std::move(actor.publication);
    if (body) {
        if (body->mass) {
            const auto &mass = *body->mass;
            prepared.body.mass.x = 1.0f / mass.mass;
            prepared.body.invInertiaDiagonal = {mass.inverse.GetX(), mass.inverse.GetY(), mass.inverse.GetZ(), 0.0f};
            prepared.body.inertiaRotation = {mass.axes.GetX(), mass.axes.GetY(), mass.axes.GetZ(), mass.axes.GetW()};
        }
        prepared.body.mass.y = float((body->constraints >> 1) & 7);
        prepared.body.mass.z = float((body->constraints >> 4) & 7);
        prepared.body.mass.w = body->gravityFactor;
        prepared.body.damping = {body->drag, body->angularDrag, body->maxLinearVelocity, body->maxAngularVelocity};
        const auto detection = static_cast<CollisionDetectionMode>(body->collisionDetectionMode);
        if (motion == JPH::GPU::MotionDynamic && detection != CollisionDetectionMode::Discrete) {
            const float radius = prepared.authoredShape->GetInnerRadius();
            prepared.body.invInertiaDiagonal.w =
                detection == CollisionDetectionMode::ContinuousDynamic ? -radius : radius;
        }

        if (const auto &linear = prepared.publication.linearVelocity)
            prepared.initialState.linearVelocity = {linear->x, linear->y, linear->z, 0.0f};
        if (const auto &angular = prepared.publication.angularVelocity)
            prepared.initialState.angularVelocity = {angular->x, angular->y, angular->z, 0.0f};
    }
    for (auto &leaf : prepared.geometry.leaves) {
        leaf.model.meta.y = bodySlot;
        WriteSurface(leaf.model, actor.surfaces[leaf.memberIndex]);
    }
    if (!body && !actor.surfaces.empty() &&
        std::all_of(actor.surfaces.begin(), actor.surfaces.end(), [](const auto &surface) { return surface.sensor; }))
        prepared.sleep.policy.x |= 2u;
    prepared.members.reserve(actor.surfaces.size());
    for (const auto &surface : actor.surfaces)
        prepared.members.push_back({surface.target, surface.colliderId, surface.generation, surface.gameObjectId});
    return prepared;
}

} // namespace infernux
