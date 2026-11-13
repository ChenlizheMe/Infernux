#include <Jolt/Jolt.h>

#include <Jolt/Geometry/AABox.h>
#include <Jolt/Physics/Collision/CollisionCollectorImpl.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CompoundShape.h>
#include <Jolt/Physics/Collision/Shape/DecoratedShape.h>
#include <Jolt/Physics/Collision/Shape/Shape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/ShapeFilter.h>
#include <Jolt/Physics/Collision/TransformedShape.h>

#include "JoltGpuShape.h"

#include <stdexcept>
#include <utility>

namespace infernux
{
namespace
{
uint32_t MemberIndex(const JPH::Shape &root, JPH::SubShapeID subShapeID, uint32_t memberCount)
{
    if (memberCount == 1)
        return 0;
    const JPH::Shape *shape = &root;
    // Author COM and rigid-pose decorators preserve the child's path bits.
    while (shape->GetType() == JPH::EShapeType::Decorated)
        shape = static_cast<const JPH::DecoratedShape *>(shape)->GetInnerShape();
    if (memberCount == 0 || shape->GetType() != JPH::EShapeType::Compound)
        throw std::invalid_argument("Jolt GPU leaf has no contributing compound member");
    const auto *compound = static_cast<const JPH::CompoundShape *>(shape);
    if (!compound->IsSubShapeIDValid(subShapeID))
        throw std::invalid_argument("Jolt GPU leaf has an invalid compound path");
    JPH::SubShapeID remainder;
    const auto index = compound->GetSubShapeIndexFromID(subShapeID, remainder);
    // GetSubShapeUserData would read the terminal Shape's user data instead.
    // The author's outer compound stores the Collider member ordinal here.
    const auto member = compound->GetCompoundUserData(index);
    if (member >= memberCount)
        throw std::invalid_argument("Jolt GPU compound member ordinal is out of range");
    return member;
}
} // namespace

PreparedGpuShape PrepareJoltGpuShape(const JPH::Shape &shape, uint32_t memberCount)
{
    PreparedGpuShape prepared{};
    prepared.centerOfMass = shape.GetCenterOfMass();

    // Collect in root COM space. Jolt handles compound child COM positions,
    // rotated/translated decorators and explicit COM offsets itself.
    JPH::AllHitCollisionCollector<JPH::TransformedShapeCollector> collector;
    shape.CollectTransformedShapes(JPH::AABox::sBiggest(), JPH::Vec3::sZero(), JPH::Quat::sIdentity(),
                                   JPH::Vec3::sReplicate(1.0f), JPH::SubShapeIDCreator(), collector,
                                   JPH::ShapeFilter());
    prepared.leaves.reserve(collector.mHits.size());
    for (const auto &leaf : collector.mHits) {
        if (leaf.mShape->GetSubType() == JPH::EShapeSubType::Empty)
            continue;
        const auto scale = leaf.GetShapeScale();
        if (scale.GetX() != 1.0f || scale.GetY() != 1.0f || scale.GetZ() != 1.0f)
            throw std::invalid_argument("Jolt GPU leaf scale must be unit; component scale must already be baked");
        PreparedGpuLeaf output{};
        output.subShapeID = leaf.mSubShapeIDCreator.GetID();
        output.subShapeBits = leaf.mSubShapeIDCreator.GetNumBitsWritten();
        output.memberIndex = MemberIndex(shape, output.subShapeID, memberCount);
        output.model.meta.z = output.subShapeID.GetValue();
        output.model.source = {0, 0, JPH::GPU::InvalidGeometryPayload, output.subShapeBits};
        output.model.localPositionCOM = {float(leaf.mShapePositionCOM.GetX()), float(leaf.mShapePositionCOM.GetY()),
                                         float(leaf.mShapePositionCOM.GetZ()), 0.0f};
        output.model.localRotation = {leaf.mShapeRotation.GetX(), leaf.mShapeRotation.GetY(),
                                      leaf.mShapeRotation.GetZ(), leaf.mShapeRotation.GetW()};
        switch (leaf.mShape->GetSubType()) {
        case JPH::EShapeSubType::Sphere: {
            const float radius = static_cast<const JPH::SphereShape *>(leaf.mShape.GetPtr())->GetRadius();
            output.model.meta.x = JPH::GPU::ShapeSphere;
            output.model.geometry = {0.0f, 0.0f, 0.0f, radius};
            break;
        }
        case JPH::EShapeSubType::Box: {
            const auto *box = static_cast<const JPH::BoxShape *>(leaf.mShape.GetPtr());
            const auto halfExtent = box->GetHalfExtent();
            output.model.meta.x = JPH::GPU::ShapeBox;
            output.model.geometry = {halfExtent.GetX(), halfExtent.GetY(), halfExtent.GetZ(), box->GetConvexRadius()};
            break;
        }
        default:
            throw std::invalid_argument(
                "Jolt GPU shape preparation supports only Sphere and Box leaves; Capsule is not yet admitted");
        }
        prepared.leaves.push_back(std::move(output));
    }
    return prepared;
}

} // namespace infernux
