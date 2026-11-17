#pragma once

#include <core/math/Bvh.h>

#include <cstdint>
#include <vector>

namespace infernux
{
struct Vertex;

namespace raytracing
{

inline constexpr uint32_t kInvalidRayIndex = UINT32_MAX;

struct alignas(16) RayPosition
{
    float x, y, z, reserved = 0.0f;
};

struct alignas(16) RayTriangle
{
    uint32_t a, b, c;
    uint32_t primitive;
};

struct alignas(16) RayInstance
{
    // World-to-object rows, including translation. The direction is transformed without normalization.
    std::array<float, 4> worldToObjectX;
    std::array<float, 4> worldToObjectY;
    std::array<float, 4> worldToObjectZ;
    uint32_t root;
    uint32_t surfaceId;
    uint32_t layerMask;
    uint32_t reserved = 0;
};

static_assert(sizeof(RayPosition) == 16 && sizeof(RayTriangle) == 16 && sizeof(RayInstance) == 64);
static_assert(offsetof(RayInstance, root) == 48);

/// Immutable object-space data for one contiguous indexed triangle slice.
struct RayMesh
{
    std::vector<spatial::BvhNode> nodes;
    std::vector<RayPosition> positions;
    std::vector<RayTriangle> triangles;
};

struct RayMeshInstance
{
    const RayMesh &mesh;
    glm::mat4 objectToWorld;
    uint32_t surfaceId;
    uint32_t layerMask;
};

/// All references in a page are page-relative. BLASes shared by instances occur once.
struct RayGeometryPage
{
    std::vector<spatial::BvhNode> nodes;
    std::vector<RayPosition> positions;
    std::vector<RayTriangle> triangles;
    std::vector<RayInstance> instances;
    uint32_t root = kInvalidRayIndex;
};

[[nodiscard]] RayMesh BuildRayMesh(const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices,
                                   uint32_t indexStart, uint32_t indexCount, int32_t vertexStart);
[[nodiscard]] RayGeometryPage BuildRayGeometryPage(const std::vector<RayMeshInstance> &instances);

} // namespace raytracing
} // namespace infernux
