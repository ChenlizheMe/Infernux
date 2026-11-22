#include "PhysicsShapeLeaves.h"
#include "PhysicsShapeState.h"

#include <Jolt/Physics/Collision/CollisionCollectorImpl.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/CompoundShape.h>
#include <Jolt/Physics/Collision/Shape/DecoratedShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/ShapeFilter.h>
#include <algorithm>
#include <stdexcept>

namespace infernux
{
namespace
{
glm::vec3 Vector(JPH::Vec3Arg value)
{
    return {value.GetX(), value.GetY(), value.GetZ()};
}

uint32_t ColliderMember(const JPH::Shape &root, JPH::SubShapeID subShapeID, uint32_t memberCount)
{
    if (memberCount == 1)
        return 0;
    const JPH::Shape *shape = &root;
    while (shape->GetType() == JPH::EShapeType::Decorated)
        shape = static_cast<const JPH::DecoratedShape *>(shape)->GetInnerShape();
    if (memberCount == 0 || shape->GetType() != JPH::EShapeType::Compound)
        throw std::invalid_argument("Physics leaf has no contributing compound member");
    const auto &compound = static_cast<const JPH::CompoundShape &>(*shape);
    if (!compound.IsSubShapeIDValid(subShapeID))
        throw std::invalid_argument("Physics leaf has an invalid compound path");
    JPH::SubShapeID remainder;
    const auto index = compound.GetSubShapeIndexFromID(subShapeID, remainder);
    // The authoring compound stores Collider ordinals on its immediate children.
    const auto member = compound.GetCompoundUserData(index);
    if (member >= memberCount)
        throw std::invalid_argument("Physics compound member is out of range");
    return member;
}
} // namespace

std::vector<PhysicsShapeLeaf> CollectPhysicsShapeLeaves(const JPH::Shape &shape, uint32_t memberCount)
{
    JPH::AllHitCollisionCollector<JPH::TransformedShapeCollector> collector;
    shape.CollectTransformedShapes(JPH::AABox::sBiggest(), JPH::Vec3::sZero(), JPH::Quat::sIdentity(),
                                   JPH::Vec3::sReplicate(1.0f), JPH::SubShapeIDCreator(), collector,
                                   JPH::ShapeFilter());
    std::vector<PhysicsShapeLeaf> leaves;
    leaves.reserve(collector.mHits.size());
    for (const auto &leaf : collector.mHits) {
        if (leaf.mShape->GetSubType() == JPH::EShapeSubType::Empty)
            continue;
        if (leaf.GetShapeScale() != JPH::Vec3::sReplicate(1.0f))
            throw std::invalid_argument("Physics leaf scale must already be baked");
        leaves.push_back({leaf, ColliderMember(shape, leaf.mSubShapeIDCreator.GetID(), memberCount)});
    }
    std::sort(leaves.begin(), leaves.end(), [](const auto &a, const auto &b) {
        if (a.member != b.member)
            return a.member < b.member;
        return a.shape.mSubShapeIDCreator.GetID().GetValue() < b.shape.mSubShapeIDCreator.GetID().GetValue();
    });
    return leaves;
}

std::shared_ptr<const PhysicsShapeGeometry> PhysicsShapeGeometry::DescribeLeaf(const JPH::Shape &shape)
{
    auto geometry = std::make_shared<PhysicsShapeGeometry>();
    auto owner = std::make_shared<JPH::RefConst<JPH::Shape>>(&shape);
    geometry->shape = std::shared_ptr<const JPH::Shape>(std::move(owner), &shape);
    geometry->identity = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&shape));
    geometry->convex = shape.GetType() == JPH::EShapeType::Convex;
    const auto bounds = shape.GetLocalBounds();
    geometry->boundsMin = Vector(bounds.mMin);
    geometry->boundsMax = Vector(bounds.mMax);
    switch (shape.GetSubType()) {
    case JPH::EShapeSubType::Box: {
        const auto &box = static_cast<const JPH::BoxShape &>(shape);
        geometry->type = PhysicsShapeType::Box;
        geometry->dimensions = glm::vec4(2.0f * Vector(box.GetHalfExtent()), box.GetConvexRadius());
        break;
    }
    case JPH::EShapeSubType::Sphere:
        geometry->type = PhysicsShapeType::Sphere;
        geometry->dimensions.x = static_cast<const JPH::SphereShape &>(shape).GetRadius();
        break;
    case JPH::EShapeSubType::Capsule: {
        const auto &capsule = static_cast<const JPH::CapsuleShape &>(shape);
        const float radius = capsule.GetRadius();
        geometry->type = PhysicsShapeType::Capsule;
        geometry->dimensions = {radius, 2.0f * (capsule.GetHalfHeightOfCylinder() + radius), 1.0f, 0.0f};
        break;
    }
    default:
        // Cylinders and convex hulls expose their
        // immutable surface through GetTriangles.
        geometry->type = PhysicsShapeType::Mesh;
        break;
    }
    return geometry;
}

std::vector<glm::vec3> PhysicsShapeGeometry::GetTriangles() const
{
    if (triangle) {
        // A zero-margin triangle hull exposes both sides of the same face.
        const auto &[a, b, c] = *triangle;
        return {a, b, c, a, c, b};
    }
    if (!shape)
        throw std::logic_error("Shape geometry has no immutable authored shape");
    JPH::Shape::GetTrianglesContext context;
    const auto leafScale = JPH::Vec3::sReplicate(1.0f);
    shape->GetTrianglesStart(context, shape->GetLocalBounds().Scaled(leafScale), JPH::Vec3::sZero(),
                             JPH::Quat::sIdentity(), leafScale);
    constexpr int batchSize = 256;
    JPH::Float3 vertices[3 * batchSize];
    std::vector<glm::vec3> result;
    for (int count; (count = shape->GetTrianglesNext(context, batchSize, vertices)) != 0;)
        for (int i = 0; i < count * 3; ++i)
            result.emplace_back(vertices[i].x, vertices[i].y, vertices[i].z);
    return result;
}

} // namespace infernux
