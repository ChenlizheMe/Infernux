#include <Jolt/Jolt.h>

#include <Jolt/Geometry/AABox.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/Shape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>

#include "JoltGpuShape.h"
#include "PhysicsShapeLeaves.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace infernux
{
namespace
{
struct MeshTriangleGeometry
{
    struct Triangle
    {
        JPH::SubShapeID id;
        JPH::Vec3 normal;
        uint32_t activeEdges = 0;
        std::shared_ptr<const JoltGpuGeometryRecord> geometry;
    };
    JPH::RefConst<JPH::MeshShape> mesh;
    std::vector<Triangle> triangles;
};
struct MeshTriangleMemo
{
    std::mutex mutex;
    std::unordered_map<const JPH::MeshShape *, std::weak_ptr<const MeshTriangleGeometry>> records;
};
std::shared_ptr<MeshTriangleMemo> TriangleMemo()
{
    static auto memo = std::make_shared<MeshTriangleMemo>();
    return memo;
}

uint32_t HullTriangleEdges(const JPH::ConvexHullShape &hull, const std::array<JPH::Vec3, 3> &vertices,
                           uint8_t activeEdges)
{
    if (hull.GetNumPoints() != 3 || hull.GetConvexRadius() != 0 || (activeEdges & ~7u))
        throw std::invalid_argument("Mesh triangle requires its exact zero-margin three-point hull");
    std::array<uint32_t, 3> source{};
    uint32_t used = 0;
    for (uint32_t point = 0; point < 3; ++point) {
        bool found = false;
        for (uint32_t original = 0; original < 3; ++original)
            // Native ConvexHullShape stores exactly inputPoint - native COM.
            // Establish a bijection; never infer edge identity from nearest points.
            if (!(used & (1u << original)) && hull.GetPoint(point) == vertices[original] - hull.GetCenterOfMass()) {
                source[point] = original;
                used |= 1u << original;
                found = true;
                break;
            }
        if (!found)
            throw std::invalid_argument("Native triangle hull changed its source vertices");
    }
    uint32_t result = 0;
    for (uint32_t edge = 0; edge < 3; ++edge) {
        const auto a = source[edge], b = source[(edge + 1) % 3];
        const auto originalEdge = (a + 1) % 3 == b ? a : b;
        if (activeEdges & (1u << originalEdge))
            result |= 1u << edge;
    }
    return result; // Bits correspond to full hull vertex edges (01,12,20).
}

std::shared_ptr<const MeshTriangleGeometry> EncodeMeshTriangles(const JPH::MeshShape &mesh)
{
    auto memo = TriangleMemo();
    {
        std::lock_guard lock(memo->mutex);
        if (const auto found = memo->records.find(&mesh); found != memo->records.end())
            if (auto existing = found->second.lock())
                return existing;
    }
    const std::weak_ptr<MeshTriangleMemo> weak = memo;
    auto result = std::shared_ptr<MeshTriangleGeometry>(
        new MeshTriangleGeometry, [weak, key = &mesh](MeshTriangleGeometry *value) {
            if (auto memo = weak.lock()) {
                std::lock_guard lock(memo->mutex);
                const auto found = memo->records.find(key);
                if (found != memo->records.end() && found->second.expired())
                    memo->records.erase(found);
            }
            delete value;
        });
    // The weak key cannot be reused while this immutable recipe is alive.
    result->mesh = &mesh;
    JPH::Array<JPH::MeshShape::TriangleData> triangles;
    mesh.GetTriangleData(triangles);
    result->triangles.reserve(triangles.size());
    for (const auto &triangle : triangles) {
        const std::array<JPH::Vec3, 3> vertices{triangle.mV0, triangle.mV1, triangle.mV2};
        const auto cross = (vertices[1] - vertices[0]).Cross(vertices[2] - vertices[0]);
        const float areaSquared = cross.LengthSq();
        if (!(areaSquared > 0) || !std::isfinite(areaSquared))
            throw std::invalid_argument("Cooked mesh triangle has no finite CCW normal");
        JPH::ConvexHullShapeSettings settings(vertices.data(), 3, 0.0f);
        const auto cooked = settings.Create();
        if (cooked.HasError())
            throw std::invalid_argument(std::string("Native mesh triangle hull: ") + cooked.GetError().c_str());
        const auto &hull = static_cast<const JPH::ConvexHullShape &>(*cooked.Get());
        const auto edges = HullTriangleEdges(hull, vertices, triangle.mActiveEdges);
        result->triangles.push_back(
            {triangle.mSubShapeID, cross / std::sqrt(areaSquared), edges, EncodeJoltGpuGeometry(hull, &vertices)});
    }
    {
        std::lock_guard lock(memo->mutex);
        auto &entry = memo->records[&mesh];
        if (auto existing = entry.lock())
            return existing;
        entry = result;
    }
    return result;
}

} // namespace

PreparedGpuShape PrepareJoltGpuShape(const JPH::Shape &shape, uint32_t memberCount)
{
    PreparedGpuShape prepared{};
    prepared.centerOfMass = shape.GetCenterOfMass();

    const auto leaves = CollectPhysicsShapeLeaves(shape, memberCount);
    prepared.leaves.reserve(leaves.size());
    for (const auto &[leaf, member] : leaves) {
        PreparedGpuLeaf output{};
        output.shape = leaf.mShape;
        output.subShapeID = leaf.mSubShapeIDCreator.GetID();
        output.subShapeBits = leaf.mSubShapeIDCreator.GetNumBitsWritten();
        output.memberIndex = member;
        output.model.meta.z = output.subShapeID.GetValue();
        // The actor capture fills the real Collider ID and lifetime generation.
        output.model.source = {};
        output.model.localPositionCOM = {float(leaf.mShapePositionCOM.GetX()), float(leaf.mShapePositionCOM.GetY()),
                                         float(leaf.mShapePositionCOM.GetZ()), 0.0f};
        output.model.localRotation = {leaf.mShapeRotation.GetX(), leaf.mShapeRotation.GetY(),
                                      leaf.mShapeRotation.GetZ(), leaf.mShapeRotation.GetW()};
        switch (leaf.mShape->GetSubType()) {
        case JPH::EShapeSubType::Mesh: {
            const auto &mesh = static_cast<const JPH::MeshShape &>(*leaf.mShape);
            const auto bits = mesh.GetSubShapeIDBitsRecursive();
            if (bits > JPH::SubShapeID::MaxBits - output.subShapeBits)
                throw std::invalid_argument("Mesh triangle path exceeds native SubShapeID capacity");
            const auto geometry = EncodeMeshTriangles(mesh);
            for (const auto &triangle : geometry->triangles) {
                auto item = output;
                JPH::SubShapeID remainder;
                // Native SubShapeID pads unwritten high bits with ones. Push
                // only its actual mesh bits, retaining the complete parent path.
                const auto id = leaf.mSubShapeIDCreator.PushID(triangle.id.PopID(bits, remainder), bits);
                item.subShapeID = id.GetID();
                item.subShapeBits = id.GetNumBitsWritten();
                item.model.meta.x = JPH::GPU::ShapeConvexHull;
                item.model.meta.z = item.subShapeID.GetValue();
                item.model.meta.w =
                    JPH::GPU::ShapeOneSidedTriangle | (triangle.activeEdges << JPH::GPU::ShapeTriangleActiveEdgeShift);
                item.model.geometry = {0, triangle.normal.GetX(), triangle.normal.GetY(), triangle.normal.GetZ()};
                item.shape = triangle.geometry->shape;
                const auto position = leaf.mShapePositionCOM + leaf.mShapeRotation * item.shape->GetCenterOfMass();
                item.model.localPositionCOM = {float(position.GetX()), float(position.GetY()), float(position.GetZ()),
                                               output.model.localPositionCOM.x};
                // Capture the common mesh origin before adding the triangle COM.
                item.model.material.z = output.model.localPositionCOM.y;
                item.model.material.w = output.model.localPositionCOM.z;
                // Keep the complete immutable mesh recipe through existing
                // geometry-record owners, without a new publication field.
                item.geometryRecord = std::shared_ptr<const JoltGpuGeometryRecord>(geometry, triangle.geometry.get());
                prepared.leaves.push_back(std::move(item));
            }
            continue;
        }
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
        case JPH::EShapeSubType::Capsule: {
            const auto *capsule = static_cast<const JPH::CapsuleShape *>(leaf.mShape.GetPtr());
            output.model.meta.x = JPH::GPU::ShapeCapsule;
            output.model.geometry = {0.0f, capsule->GetHalfHeightOfCylinder(), 0.0f, capsule->GetRadius()};
            break;
        }
        case JPH::EShapeSubType::Cylinder: {
            const auto *cylinder = static_cast<const JPH::CylinderShape *>(leaf.mShape.GetPtr());
            output.model.meta.x = JPH::GPU::ShapeCylinder;
            JPH::ConvexShape::SupportBuffer buffer;
            const auto *core = cylinder->GetSupportFunction(JPH::ConvexShape::ESupportMode::ExcludeConvexRadius, buffer,
                                                            JPH::Vec3::sReplicate(1.0f));
            output.model.geometry = {cylinder->GetRadius(), cylinder->GetHalfHeight(), core->GetConvexRadius(), 0};
            break;
        }
        case JPH::EShapeSubType::ConvexHull: {
            const auto *hull = static_cast<const JPH::ConvexHullShape *>(leaf.mShape.GetPtr());
            output.model.meta.x = JPH::GPU::ShapeConvexHull;
            output.model.geometry = {hull->GetConvexRadius(), 0, 0, 0};
            output.geometryRecord = EncodeJoltGpuGeometry(*hull);
            break;
        }
        default:
            throw std::invalid_argument("Jolt GPU shape preparation requires a supported final convex leaf");
        }
        prepared.leaves.push_back(std::move(output));
    }
    // Jolt's CPU tree traversal is spatially ordered. The device consumes a
    // flat leaf stream, so establish component member order once at cook time
    // for queries, batch exports and downstream consumers alike.
    std::sort(prepared.leaves.begin(), prepared.leaves.end(), [](const auto &left, const auto &right) {
        return left.memberIndex != right.memberIndex ? left.memberIndex < right.memberIndex
                                                     : left.subShapeID.GetValue() < right.subShapeID.GetValue();
    });
    return prepared;
}

} // namespace infernux
