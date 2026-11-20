#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Collision/Shape/Shape.h>
#include <array>
#include <function/scene/physics/gpu/ConvexGeometryTypes.h>
#include <memory>
#include <vector>

namespace infernux
{
/// Exact final immutable Jolt geometry. Words contain no pose/Collider/World identity.
/// COM and component scale were already applied by native author cooking.
struct JoltGpuGeometryRecord
{
    JPH::RefConst<JPH::Shape> shape;
    float convexRadius = 0;
    std::vector<JPH::GPU::UInt4> words;
};

/// Encode a native final ConvexHull, including its exact native shrunken core.
/// Analytic shapes have inline geometry and do not allocate a cold record.
/// Mesh triangles may append their three original mesh-local ray vertices.
[[nodiscard]] std::shared_ptr<const JoltGpuGeometryRecord>
EncodeJoltGpuGeometry(const JPH::Shape &shape, const std::array<JPH::Vec3, 3> *meshRayVertices = nullptr);
/// Validate the complete typed record, all topology and initialized padding.
void ValidateJoltGpuGeometry(const JoltGpuGeometryRecord &record);
} // namespace infernux
