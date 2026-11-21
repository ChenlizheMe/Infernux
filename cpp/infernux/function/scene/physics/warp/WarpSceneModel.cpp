// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#include "WarpSceneModel.h"

#include "../../Collider.h"
#include "../../GameObject.h"
#include "../../Rigidbody.h"
#include "../../TagLayerManager.h"
#include "../../Transform.h"
#include "../../TransformECSStore.h"
#include "../PhysicsShapeLeaves.h"

#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace infernux::physics::warp
{
namespace
{
glm::vec3 Vector(JPH::RVec3Arg v)
{
    return {static_cast<float>(v.GetX()), static_cast<float>(v.GetY()), static_cast<float>(v.GetZ())};
}

glm::quat Rotation(JPH::QuatArg q)
{
    return {q.GetW(), q.GetX(), q.GetY(), q.GetZ()};
}

ModelVector Point(const glm::vec3 &v)
{
    return {v.x, v.y, v.z, 0.0f};
}

ModelPose Pose(const glm::vec3 &v, const glm::quat &q)
{
    return {Point(v), {q.w, q.x, q.y, q.z}};
}

std::shared_ptr<const WarpConvexHull> CookHull(const JPH::ConvexHullShape &shape)
{
    auto hull = std::make_shared<WarpConvexHull>();
    hull->vertices.reserve(shape.GetNumPoints());
    for (uint32_t vertex = 0; vertex < shape.GetNumPoints(); ++vertex)
        hull->vertices.push_back(Point(Vector(JPH::RVec3(shape.GetPoint(vertex)))));
    hull->faces.reserve(shape.GetNumFaces());
    for (uint32_t index = 0; index < shape.GetNumFaces(); ++index) {
        auto &face = hull->faces.emplace_back();
        face.vertices.resize(shape.GetNumVerticesInFace(index));
        shape.GetFaceVertices(index, static_cast<uint32_t>(face.vertices.size()), face.vertices.data());
        face.normal = Point(Vector(JPH::RVec3(shape.GetPlanes()[index].GetNormal())));
    }
    return hull;
}

using Hulls = std::unordered_map<const JPH::Shape *, std::shared_ptr<const WarpConvexHull>>;

WarpGeometryType CookGeometry(const JPH::Shape &shape, WarpCollisionGeometry &geometry, Hulls &hulls)
{
    switch (shape.GetSubType()) {
    case JPH::EShapeSubType::Sphere: {
        const float radius = static_cast<const JPH::SphereShape &>(shape).GetRadius();
        geometry.size = {radius, 0.0f, 0.0f, 0.0f};
        geometry.halfExtent = {radius, radius, radius, 0.0f};
        geometry.radius = radius;
        return Sphere;
    }
    case JPH::EShapeSubType::Box: {
        const auto half = Vector(JPH::RVec3(static_cast<const JPH::BoxShape &>(shape).GetHalfExtent()));
        geometry.size = geometry.halfExtent = Point(half);
        geometry.radius = glm::length(half);
        return Box;
    }
    case JPH::EShapeSubType::Capsule: {
        const auto &capsule = static_cast<const JPH::CapsuleShape &>(shape);
        const float radius = capsule.GetRadius(), halfHeight = capsule.GetHalfHeightOfCylinder();
        geometry.size = {radius, halfHeight, 0.0f, 0.0f};
        geometry.halfExtent = {radius, radius, halfHeight + radius, 0.0f};
        geometry.radius = halfHeight + radius;
        return Capsule;
    }
    case JPH::EShapeSubType::Cylinder: {
        const auto &cylinder = static_cast<const JPH::CylinderShape &>(shape);
        const float radius = cylinder.GetRadius(), halfHeight = cylinder.GetHalfHeight();
        geometry.size = {radius, halfHeight, 0.0f, 0.0f};
        geometry.halfExtent = {radius, radius, halfHeight, 0.0f};
        geometry.radius = std::hypot(radius, halfHeight);
        return Cylinder;
    }
    case JPH::EShapeSubType::ConvexHull: {
        auto &hull = hulls[&shape];
        if (!hull)
            hull = CookHull(static_cast<const JPH::ConvexHullShape &>(shape));
        geometry.hull = hull;
        return Hull;
    }
    default:
        throw std::invalid_argument("Collider shape is not supported by the native Warp scene model");
    }
}
} // namespace

std::optional<WarpSceneCapture> CaptureWarpSceneActor(const PhysicsActorData &actor)
{
    TransformECSStore::PhysicsPoseReadScope authoring(TransformECSStore::Instance());
    std::vector<Collider *> members;
    auto shape = BuildActorShape(actor, nullptr, nullptr, &members);
    if (!shape)
        return std::nullopt;

    const auto *rigidbody = actor.rigidbody && actor.rigidbody->IsEnabled() ? actor.rigidbody : nullptr;
    const auto *transform = actor.owner->GetTransform();
    WarpSceneCapture result{};
    result.publication = PhysicsActorPublication::Capture(actor, members, rigidbody);
    auto &source = result.actor;
    source.objectId = actor.owner->GetID();
    source.shape = std::move(shape);
    source.position = transform->GetPosition();
    source.rotation = transform->GetWorldRotation();
    if (rigidbody) {
        source.kinematic = rigidbody->IsKinematic();
        source.useGravity = rigidbody->GetUseGravity();
        if (!source.kinematic) {
            const auto &properties = rigidbody->GetAuthoredMassProperties();
            source.mass = PrepareBodyMass(source.shape.GetPtr(), rigidbody->GetMass(),
                                          properties ? &*properties : nullptr, transform->GetWorldScale());
            source.linearVelocity = result.publication.linearVelocity.value_or(glm::vec3(0.0f));
            source.angularVelocity = result.publication.angularVelocity.value_or(glm::vec3(0.0f));
        }
    }
    source.colliders.reserve(members.size());
    for (const auto *collider : members) {
        const int layer = collider->GetGameObject()->GetLayer();
        source.colliders.push_back(
            {collider->GetComponentID(), collider->GetLifetimeGeneration(), static_cast<uint32_t>(layer),
             TagLayerManager::Instance().GetLayerCollisionMask(layer), collider->GetFriction(), collider->IsTrigger()});
    }
    return result;
}

WarpSceneModel BuildWarpSceneModel(const std::vector<WarpSceneActor> &actors)
{
    WarpSceneModel result;
    std::vector<ModelBodyDefinition> bodies;
    std::vector<ModelGeometryDefinition> geometries;
    std::vector<WarpCollisionGeometry> collision;
    Hulls hulls;
    bodies.reserve(actors.size());
    for (uint32_t index = 0; index < actors.size(); ++index) {
        const auto &actor = actors[index];
        const auto center = Vector(JPH::RVec3(actor.shape->GetCenterOfMass()));
        ModelBodyDefinition body{};
        body.pose = Pose(actor.position, actor.rotation);
        body.inertiaPose.position = Point(center);
        body.kinematic = actor.kinematic;
        if (actor.mass) {
            const auto &mass = *actor.mass;
            body.inertiaPose = Pose(center, Rotation(mass.axes));
            body.principalInertia = {1.0f / mass.inverse.GetX(), 1.0f / mass.inverse.GetY(),
                                     1.0f / mass.inverse.GetZ()};
            body.mass = mass.mass;
            body.gravityCompensation = actor.useGravity ? 0.0f : 1.0f;
            body.joints.push_back({ModelJointType::Free});
        }
        bodies.push_back(std::move(body));

        const auto leaves = CollectPhysicsShapeLeaves(*actor.shape, static_cast<uint32_t>(actor.colliders.size()));
        for (const auto &[leaf, member] : leaves) {
            const auto &surface = actor.colliders[member];
            auto &shape = collision.emplace_back();
            const auto kind = CookGeometry(*leaf.mShape, shape, hulls);
            shape.material.friction[0] = surface.friction;
            shape.trigger = surface.trigger;
            glm::quat rotation = Rotation(leaf.mShapeRotation);
            if (kind == Capsule || kind == Cylinder) {
                // Jolt axial primitives run along +Y. Warp primitives run along +Z.
                constexpr float halfRoot = 0.7071067811865475f;
                rotation *= glm::quat(halfRoot, -halfRoot, 0.0f, 0.0f);
            }
            geometries.push_back({index, kind, 1u << surface.layer, surface.collisionMask,
                                  Pose(center + Vector(leaf.mShapePositionCOM), rotation)});
            result.geometries.push_back({surface, leaf.mSubShapeIDCreator.GetID().GetValue()});
        }
    }
    result.model = BuildWarpModel(bodies, geometries);
    result.collision = BuildWarpCollisionModel(result.model, collision, {});
    result.bodies.reserve(actors.size());
    auto &initial = result.model.initialVelocities;
    initial.resize(result.model.dofs.size());
    for (size_t index = 0; index < actors.size(); ++index) {
        const auto &actor = actors[index];
        const auto row = result.model.bodyRows[index];
        const auto &body = result.model.bodies[row];
        const uint32_t firstJoint = body.hierarchy[1];
        const uint32_t qpos = firstJoint == NoModelIndex ? NoModelIndex : result.model.joints[firstJoint].indices[1];
        const uint32_t qvel = body.dofs[0];
        result.bodies.push_back({actor.objectId, row, qpos, qvel});
        if (qvel == NoModelIndex)
            continue;
        const auto center = actor.rotation * Vector(JPH::RVec3(actor.shape->GetCenterOfMass()));
        const auto linear = actor.linearVelocity - glm::cross(actor.angularVelocity, center);
        const auto angular = glm::inverse(actor.rotation) * actor.angularVelocity;
        for (uint32_t axis = 0; axis < 3; ++axis) {
            initial[qvel + axis] = linear[axis];
            initial[qvel + axis + 3] = angular[axis];
        }
    }
    return result;
}

} // namespace infernux::physics::warp
