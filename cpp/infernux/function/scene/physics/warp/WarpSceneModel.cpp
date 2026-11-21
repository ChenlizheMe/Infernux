// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#include "WarpSceneModel.h"

#include "../PhysicsShapeLeaves.h"

#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <cmath>
#include <cstring>
#include <map>
#include <set>
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

ModelWords Bits(ModelVector values)
{
    ModelWords words;
    std::memcpy(words.data(), values.data(), sizeof(words));
    return words;
}

struct SceneJointEndpoint
{
    uint32_t body = 0;
    glm::vec3 anchor{0.0f};
    glm::quat orientation{1.0f, 0.0f, 0.0f, 0.0f};
    float inverseMass = 0.0f;
    float inverseInertia = 0.0f;
};

SceneJointEndpoint BindSceneJointEndpoint(const WarpModel &model, const std::vector<WarpSceneActor> &actors,
                                          uint32_t actorIndex, const glm::vec3 &worldAnchor,
                                          const glm::quat &worldRotation)
{
    if (actorIndex == NoModelIndex)
        return {0, worldAnchor, worldRotation};
    const auto &actor = actors.at(actorIndex);
    const auto inverse = glm::conjugate(actor.rotation);
    SceneJointEndpoint endpoint{model.bodyRows[actorIndex], inverse * (worldAnchor - actor.position),
                                inverse * worldRotation};
    if (actor.mass) {
        const auto axes = actor.mass->inverse;
        endpoint.inverseInertia = (axes.GetX() + axes.GetY() + axes.GetZ()) / 3.0f;
        const auto offset = endpoint.anchor - Vector(JPH::RVec3(actor.shape->GetCenterOfMass()));
        endpoint.inverseMass =
            1.0f / actor.mass->mass + (2.0f / 3.0f) * glm::dot(offset, offset) * endpoint.inverseInertia;
    }
    return endpoint;
}

void BindSceneJoints(WarpSceneModel &scene, const std::vector<WarpSceneActor> &actors,
                     const std::vector<WarpSceneJoint> &joints)
{
    constexpr size_t descriptorWords = 10;
    auto &model = scene.model;
    scene.hingeRows.resize(joints.size(), NoModelIndex);
    std::vector<ModelWords> descriptors(1), support;
    for (size_t index = 0; index < joints.size(); ++index) {
        const auto &joint = joints[index];
        const auto &owner = actors.at(joint.ownerActor);
        const auto point = owner.position + owner.rotation * joint.ownerAnchor;
        glm::quat rotation = owner.rotation;
        if (joint.hinge) {
            if (glm::dot(joint.hinge->axis, joint.hinge->axis) <= 1e-12f ||
                (joint.hinge->useLimits && joint.hinge->minimum > joint.hinge->maximum))
                throw std::invalid_argument("Warp hinge requires a nonzero axis and ordered limits");
            const auto z = glm::normalize(joint.hinge->axis);
            const auto up = std::abs(z.y) < 0.9f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
            const auto x = glm::normalize(glm::cross(up, z));
            rotation *= glm::quat_cast(glm::mat3(x, glm::cross(z, x), z));
        }
        const auto a = BindSceneJointEndpoint(model, actors, joint.ownerActor, point, rotation);
        const auto b = BindSceneJointEndpoint(model, actors, joint.connectedActor, point, rotation);
        if (a.body == b.body)
            throw std::invalid_argument("Scene joint must join different Warp bodies");
        if (a.inverseMass + b.inverseMass == 0.0f)
            continue;

        const auto first = static_cast<uint32_t>(support.size());
        for (const auto &[bodyRow, side] : {std::pair{a.body, 1u}, std::pair{b.body, 2u}}) {
            const auto &dofs = model.bodies[bodyRow].dofs;
            for (uint32_t local = dofs[1]; local-- > 0;)
                support.push_back({dofs[0] + local, side, local >= 3 ? bodyRow : NoModelIndex, 0});
        }
        const auto id = static_cast<uint32_t>((descriptors.size() - 1) / descriptorWords);
        descriptors.push_back({id, a.body, b.body, static_cast<uint32_t>(support.size()) - first});
        descriptors.push_back({a.body, b.body, first, joint.hinge ? 5u : 6u});
        descriptors.push_back(Bits({0.95f, 0.99f, 0.001f, 0.5f}));
        descriptors.push_back(Bits({0.02f, 1.0f, 2.0f, 0.0f}));
        descriptors.push_back(Bits({a.anchor.x, a.anchor.y, a.anchor.z, 1.0f}));
        descriptors.push_back(Bits(Point(b.anchor)));
        descriptors.push_back(Bits(Pose({}, a.orientation).rotation));
        descriptors.push_back(Bits(Pose({}, b.orientation).rotation));
        auto weights = Bits({0.0f, 0.0f, a.inverseMass + b.inverseMass, a.inverseInertia + b.inverseInertia});
        weights[0] = model.bodies[a.body].subtree[0];
        weights[1] = model.bodies[b.body].subtree[0];
        descriptors.push_back(weights);
        ModelWords controls{NoModelIndex, 0, 0, 0};
        if (joint.hinge) {
            const auto &hinge = *joint.hinge;
            const auto row = static_cast<uint32_t>(model.initialHingeDrives.size());
            controls = Bits({0.0f, hinge.minimum, hinge.maximum, 0.0f});
            controls[0] = row;
            controls[3] = hinge.useLimits;
            scene.hingeRows[index] = row;
            model.initialHingeDrives.push_back(hinge.drive.Pack());
        }
        descriptors.push_back(controls);
    }
    const auto count = static_cast<uint32_t>((descriptors.size() - 1) / descriptorWords);
    if (count == 0)
        return;
    descriptors[0] = {count, static_cast<uint32_t>(model.initialHingeDrives.size()), 0, 0};
    for (size_t row = 2; row < descriptors.size(); row += descriptorWords)
        descriptors[row][2] += static_cast<uint32_t>(descriptors.size());
    descriptors.insert(descriptors.end(), support.begin(), support.end());
    model.sceneJoints = std::move(descriptors);
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

std::shared_ptr<const WarpTriangleMesh> CookTriangleMesh(const JPH::MeshShape &shape)
{
    JPH::Array<JPH::MeshShape::TriangleData> triangles;
    shape.GetTriangleData(triangles);
    auto mesh = std::make_shared<WarpTriangleMesh>();
    mesh->faces.reserve(triangles.size());
    mesh->features.reserve(triangles.size());
    std::map<std::array<float, 3>, uint32_t> vertexIndices;
    for (const auto &triangle : triangles) {
        auto &face = mesh->faces.emplace_back();
        const std::array points{triangle.mV0, triangle.mV1, triangle.mV2};
        for (size_t corner = 0; corner < points.size(); ++corner) {
            const auto point = Vector(JPH::RVec3(points[corner]));
            const auto [index, fresh] = vertexIndices.try_emplace(std::array{point.x, point.y, point.z},
                                                                  static_cast<uint32_t>(mesh->vertices.size()));
            if (fresh)
                mesh->vertices.push_back(point);
            face[corner] = index->second;
        }
        mesh->features.push_back(
            {triangle.mSubShapeID.GetValue(), triangle.mActiveEdges, shape.GetTriangleUserData(triangle.mSubShapeID)});
    }
    return mesh;
}

struct GeometryCache
{
    std::unordered_map<const JPH::Shape *, std::shared_ptr<const WarpConvexHull>> hulls;
    std::unordered_map<const JPH::Shape *, std::shared_ptr<const WarpTriangleMesh>> meshes;
};

WarpGeometryType CookGeometry(const JPH::TransformedShape &leaf, WarpCollisionGeometry &geometry, GeometryCache &cache)
{
    const auto &shape = *leaf.mShape;
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
        auto &hull = cache.hulls[&shape];
        if (!hull)
            hull = CookHull(static_cast<const JPH::ConvexHullShape &>(shape));
        geometry.hull = hull;
        return Hull;
    }
    case JPH::EShapeSubType::Mesh: {
        const auto &mesh = static_cast<const JPH::MeshShape &>(shape);
        const auto pathBits = leaf.mSubShapeIDCreator.GetNumBitsWritten();
        if (mesh.GetSubShapeIDBitsRecursive() + pathBits > JPH::SubShapeID::MaxBits)
            throw std::invalid_argument("Collider mesh subshape path exceeds the native ID size");
        auto &cooked = cache.meshes[&shape];
        if (!cooked)
            cooked = CookTriangleMesh(mesh);
        geometry.mesh = cooked;
        geometry.meshPath = leaf.mSubShapeIDCreator.GetID().GetValue();
        geometry.meshPathBits = pathBits;
        return TriangleMesh;
    }
    default:
        throw std::invalid_argument("Collider shape is not supported by the native Warp scene model");
    }
}
} // namespace

WarpSceneCapture PrepareWarpSceneActor(PhysicsActorAuthoring author)
{
    WarpSceneCapture result{};
    result.publication = std::move(author.publication);
    auto &source = result.actor;
    source.objectId = author.objectId;
    source.shape = std::move(author.shape);
    source.position = author.position;
    source.rotation = author.rotation;
    if (const auto &body = author.body) {
        source.kinematic = body->kinematic;
        source.useGravity = body->useGravity;
        if (!source.kinematic) {
            source.mass = body->mass;
            source.linearVelocity = result.publication.linearVelocity.value_or(glm::vec3(0.0f));
            source.angularVelocity = result.publication.angularVelocity.value_or(glm::vec3(0.0f));
        }
    }
    source.colliders.reserve(author.surfaces.size());
    result.surfaces.reserve(author.surfaces.size());
    for (const auto &surface : author.surfaces) {
        source.colliders.push_back({surface.colliderId, surface.generation, surface.layer, surface.collisionMask,
                                    surface.friction, surface.sensor, surface.restitution, surface.frictionCombine,
                                    surface.bounceCombine});
        result.surfaces.push_back({surface.target, surface.gameObjectId, surface.colliderId});
    }
    return result;
}

WarpSceneModel BuildWarpSceneModel(const std::vector<WarpSceneActor> &actors, const std::vector<WarpSceneJoint> &joints,
                                   const PhysicsColliderPairs &ignoredColliders)
{
    WarpSceneModel result;
    std::vector<ModelBodyDefinition> bodies;
    std::vector<ModelGeometryDefinition> geometries;
    std::vector<WarpCollisionGeometry> collision;
    GeometryCache cache;
    std::vector<uint32_t> firstGeometry;
    firstGeometry.reserve(actors.size() + 1);
    bodies.reserve(actors.size());
    for (uint32_t index = 0; index < actors.size(); ++index) {
        firstGeometry.push_back(static_cast<uint32_t>(geometries.size()));
        const auto &actor = actors[index];
        const auto center = Vector(JPH::RVec3(actor.shape->GetCenterOfMass()));
        ModelBodyDefinition body{};
        body.pose = Pose(actor.position, actor.rotation);
        body.inertiaPose.position = Point(center);
        body.kinematic = actor.kinematic;
        body.editableStatic = !actor.mass && !actor.kinematic;
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
            const auto kind = CookGeometry(leaf, shape, cache);
            shape.material.friction[0] = surface.friction;
            shape.material.extra[2] = surface.bounciness;
            shape.material.dimensions[2] = 1 + static_cast<int32_t>(surface.frictionCombine);
            shape.material.dimensions[3] = 1 + static_cast<int32_t>(surface.bounceCombine);
            shape.trigger = surface.trigger;
            shape.subShape = leaf.mSubShapeIDCreator.GetID().GetValue();
            glm::quat rotation = Rotation(leaf.mShapeRotation);
            if (kind == Capsule || kind == Cylinder) {
                // Jolt axial primitives run along +Y. Warp primitives run along +Z.
                constexpr float halfRoot = 0.7071067811865475f;
                rotation *= glm::quat(halfRoot, -halfRoot, 0.0f, 0.0f);
            }
            geometries.push_back({index, kind, 1u << surface.layer, surface.collisionMask,
                                  Pose(center + Vector(leaf.mShapePositionCOM), rotation)});
            result.geometries.push_back({surface, shape.subShape, member});
        }
    }
    firstGeometry.push_back(static_cast<uint32_t>(geometries.size()));
    result.model = BuildWarpModel(bodies, geometries);
    BindSceneJoints(result, actors, joints);
    std::set<std::pair<uint32_t, uint32_t>> ignored;
    for (const auto &joint : joints) {
        if (joint.enableCollision || joint.connectedActor == NoModelIndex)
            continue;
        for (uint32_t first = firstGeometry[joint.ownerActor]; first < firstGeometry[joint.ownerActor + 1]; ++first)
            for (uint32_t second = firstGeometry[joint.connectedActor];
                 second < firstGeometry[joint.connectedActor + 1]; ++second)
                ignored.emplace(std::minmax(first, second));
    }
    if (!ignoredColliders.empty()) {
        using ColliderKey = std::pair<uint64_t, uint64_t>;
        std::map<ColliderKey, std::vector<uint32_t>> geometryByCollider;
        for (uint32_t row = 0; row < result.geometries.size(); ++row) {
            const auto &collider = result.geometries[row].collider;
            geometryByCollider[{collider.componentId, collider.generation}].push_back(row);
        }
        for (const auto &[firstId, firstGeneration, secondId, secondGeneration] : ignoredColliders) {
            const auto first = geometryByCollider.find({firstId, firstGeneration});
            const auto second = geometryByCollider.find({secondId, secondGeneration});
            if (first == geometryByCollider.end() || second == geometryByCollider.end())
                continue;
            for (const auto a : first->second)
                for (const auto b : second->second)
                    ignored.emplace(std::minmax(a, b));
        }
    }
    std::vector<WarpCollisionPair> pairs;
    pairs.reserve(ignored.size());
    for (const auto &[a, b] : ignored)
        pairs.push_back({a, b, {}, -1, true});
    result.collision = BuildWarpCollisionModel(result.model, collision, std::move(pairs));
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
        result.bodies.push_back({actor.objectId, row, qpos, qvel, static_cast<uint32_t>(actor.colliders.size())});
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
