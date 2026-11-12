#include <Jolt/Jolt.h>

#include <Jolt/Geometry/AABox.h>
#include <Jolt/Physics/Collision/CollisionCollectorImpl.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/Shape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/ShapeFilter.h>
#include <Jolt/Physics/Collision/TransformedShape.h>

#include "JoltGpuShape.h"

#include <stdexcept>

namespace infernux
{

PreparedGpuShape PrepareJoltGpuShape(const JPH::Shape &shape)
{
    PreparedGpuShape prepared{};
    prepared.centerOfMass = shape.GetCenterOfMass();

    // Collect in root COM space. Jolt handles compound child COM positions,
    // rotated/translated decorators and explicit COM offsets itself.
    JPH::AllHitCollisionCollector<JPH::TransformedShapeCollector> collector;
    shape.CollectTransformedShapes(JPH::AABox::sBiggest(), JPH::Vec3::sZero(), JPH::Quat::sIdentity(),
                                   JPH::Vec3::sReplicate(1.0f), JPH::SubShapeIDCreator(), collector,
                                   JPH::ShapeFilter());
    if (collector.mHits.size() != 1)
        throw std::invalid_argument("Jolt GPU shape preparation requires exactly one primitive leaf");

    const auto &leaf = collector.mHits.front();
    const auto scale = leaf.GetShapeScale();
    if (scale.GetX() != 1.0f || scale.GetY() != 1.0f || scale.GetZ() != 1.0f)
        throw std::invalid_argument("Jolt GPU leaf scale must be unit; component scale must already be baked");
    prepared.subShapeID = leaf.mSubShapeIDCreator.GetID();
    prepared.model.localPositionCOM = {float(leaf.mShapePositionCOM.GetX()), float(leaf.mShapePositionCOM.GetY()),
                                       float(leaf.mShapePositionCOM.GetZ()), 0.0f};
    prepared.model.localRotation = {leaf.mShapeRotation.GetX(), leaf.mShapeRotation.GetY(), leaf.mShapeRotation.GetZ(),
                                    leaf.mShapeRotation.GetW()};
    switch (leaf.mShape->GetSubType()) {
    case JPH::EShapeSubType::Empty:
        prepared.type = JPH::GPU::ShapeEmpty;
        prepared.model.geometry = {};
        break;
    case JPH::EShapeSubType::Sphere: {
        const float radius = static_cast<const JPH::SphereShape *>(leaf.mShape.GetPtr())->GetRadius();
        prepared.type = JPH::GPU::ShapeSphere;
        prepared.model.geometry = {0.0f, 0.0f, 0.0f, radius};
        break;
    }
    case JPH::EShapeSubType::Box: {
        const auto *box = static_cast<const JPH::BoxShape *>(leaf.mShape.GetPtr());
        const auto halfExtent = box->GetHalfExtent();
        const float radius = box->GetConvexRadius();
        prepared.model.geometry = {halfExtent.GetX(), halfExtent.GetY(), halfExtent.GetZ(), radius};
        prepared.type = JPH::GPU::ShapeBox;
        break;
    }
    default:
        throw std::invalid_argument("Jolt GPU shape preparation supports only Sphere, Box and Empty leaves");
    }
    return prepared;
}

} // namespace infernux
