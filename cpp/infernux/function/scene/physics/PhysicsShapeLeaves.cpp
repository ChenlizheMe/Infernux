#include "PhysicsShapeLeaves.h"

#include <Jolt/Physics/Collision/CollisionCollectorImpl.h>
#include <Jolt/Physics/Collision/Shape/CompoundShape.h>
#include <Jolt/Physics/Collision/Shape/DecoratedShape.h>
#include <Jolt/Physics/Collision/ShapeFilter.h>
#include <algorithm>
#include <stdexcept>

namespace infernux
{
namespace
{
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

} // namespace infernux
