#pragma once

#include <core/math/Bvh.h>

#include <cstdint>
#include <memory>
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

/// Kept apart from positions so the traversal only reads shading data on a hit.
struct alignas(16) RayVertexAttributes
{
    std::array<float, 4> uv;
    std::array<float, 4> normal;
    std::array<float, 4> tangent;
    std::array<float, 4> color;
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
    float orientation = 1.0f;
};

static_assert(sizeof(RayPosition) == 16 && sizeof(RayTriangle) == 16 && sizeof(RayInstance) == 64);
static_assert(sizeof(RayVertexAttributes) == 64);
static_assert(offsetof(RayInstance, root) == 48 && offsetof(RayInstance, orientation) == 60);

/// Immutable object-space data for one contiguous indexed triangle slice.
struct RayMesh
{
    std::vector<spatial::BvhNode> nodes;
    std::vector<RayPosition> positions;
    std::vector<RayVertexAttributes> attributes;
    std::vector<RayTriangle> triangles;
};

struct RayMeshPlacement
{
    spatial::Bounds bounds;
    uint32_t root = kInvalidRayIndex;
};

struct RayMeshInstance
{
    RayMeshPlacement mesh;
    glm::mat4 objectToWorld;
    uint32_t surfaceId;
    uint32_t layerMask;
};

/// Object-space slices live in this page independently of the instance transforms.
struct RayMeshPage
{
    std::vector<spatial::BvhNode> nodes;
    std::vector<RayPosition> positions;
    std::vector<RayTriangle> triangles;
    std::vector<RayVertexAttributes> attributes;
    std::vector<RayMeshPlacement> placements;
};

struct RayInstancePage
{
    std::vector<spatial::BvhNode> nodes;
    std::vector<RayInstance> instances;
};

/// The draw's streams remain alive and unchanged until the last continuation.
class RayMeshBuild
{
  public:
    RayMeshBuild(const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices, uint32_t indexStart,
                 uint32_t indexCount, int32_t vertexStart);
    ~RayMeshBuild();
    [[nodiscard]] bool Advance(uint32_t budget);
    [[nodiscard]] RayMesh Take();

  private:
    struct State;
    std::unique_ptr<State> m_state;
};

/// Packs each independent BLAS once. Source meshes remain alive through the build.
class RayMeshPageBuild
{
  public:
    explicit RayMeshPageBuild(const std::vector<RayMesh> &meshes);
    ~RayMeshPageBuild();
    [[nodiscard]] bool Advance(uint32_t budget);
    [[nodiscard]] RayMeshPage Take();

  private:
    struct State;
    std::unique_ptr<State> m_state;
};

class RayInstancePageBuild
{
  public:
    explicit RayInstancePageBuild(const std::vector<RayMeshInstance> &instances);
    ~RayInstancePageBuild();
    [[nodiscard]] bool Advance(uint32_t budget);
    [[nodiscard]] RayInstancePage Take();

  private:
    struct State;
    std::unique_ptr<State> m_state;
};

[[nodiscard]] RayMesh BuildRayMesh(const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices,
                                   uint32_t indexStart, uint32_t indexCount, int32_t vertexStart);

enum class RayGeometryStatus : uint32_t
{
    Pending,
    Ready,
    Unavailable,
};

using RayGeometryWord = std::array<uint32_t, 4>;

struct RayGeometryBytes
{
    const void *data;
    size_t bytes;
};

/// A page lives at its final offsets through the bounded upload, without a second packed copy.
struct RayPageUpload
{
    RayPageUpload(const RayMeshPage &page);
    RayPageUpload(const RayInstancePage &page, RayGeometryStatus status);
    [[nodiscard]] RayGeometryBytes Section(size_t index) const noexcept;
    static constexpr size_t sectionCount = 5;
    uint64_t bytes = 0;

  private:
    RayGeometryWord m_header{};
    std::array<RayGeometryBytes, sectionCount - 1> m_sections{};
};

} // namespace raytracing
} // namespace infernux
