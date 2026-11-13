#pragma once

#include "ParticleRuntimeApi.h"

#include <function/renderer/rhi/RhiCommand.h>
#include <function/renderer/rhi/RhiDevice.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace infernux::rhi
{
class ComputeBuffer;
}

namespace infernux::particle
{

enum class GpuParticleColliderType : uint32_t
{
    Box = 0,
    Sphere = 1,
    Capsule = 2,
    Mesh = 3,
    Terrain = 4,
};

enum GpuParticleColliderFlags : uint32_t
{
    GpuParticleColliderTrigger = 1u << 0,
    GpuParticleColliderDynamic = 1u << 1,
    GpuParticleColliderConvex = 1u << 2,
};

/// Row-major affine 3x4 transform stored as three std430 vec4 values. Collider
/// transforms never contain projection, so retaining a constant fourth row
/// would waste GPU bandwidth for every particle system.
struct alignas(16) GpuParticleAffineTransform
{
    std::array<float, 4> row0{};
    std::array<float, 4> row1{};
    std::array<float, 4> row2{};
};

/// Stable std430 record shared by every GPU emitter. Affine transforms map
/// between collider-local and world space; shape interpretation is selected by
/// metadata.x. The ABI deliberately contains no scene/component pointers.
struct alignas(16) GpuParticleColliderRecord
{
    GpuParticleAffineTransform colliderToWorld{};
    GpuParticleAffineTransform worldToCollider{};
    GpuParticleAffineTransform previousWorldToCollider{};
    std::array<float, 4> shape{};
    /// Mesh payload: vertex offset, reordered index offset, BVH root, triangle count.
    std::array<uint32_t, 4> geometry{};
    std::array<float, 4> linearVelocity{};
    std::array<float, 4> angularVelocity{};
    std::array<float, 4> centerOfMass{};
    std::array<float, 4> material{};
    std::array<float, 4> worldAabbMin{};
    std::array<float, 4> worldAabbMax{};
    std::array<float, 4> previousWorldAabbMin{};
    std::array<float, 4> previousWorldAabbMax{};
    std::array<uint32_t, 4> metadata{};
    std::array<uint32_t, 4> identity{};
};

struct alignas(16) GpuParticleCollisionSceneHeader
{
    uint32_t colliderCount = 0;
    uint32_t staticColliderCount = 0;
    uint32_t revisionLow = 0;
    uint32_t revisionHigh = 0;
    std::array<float, 4> gridMinInvCellSize{};
    std::array<uint32_t, 4> gridDimensions{};
    /// vertex count, triangle-index count, BVH node count, topology revision.
    std::array<uint32_t, 4> topology{};
};

struct alignas(16) GpuParticleCollisionBvhNode
{
    std::array<float, 4> boundsMin{};
    std::array<float, 4> boundsMax{};
    /// Internal: left/right node, count=0. Leaf: first index/count, left/right unused.
    std::array<uint32_t, 4> metadata{};
};

struct GpuParticleCollisionMeshGeometry
{
    uint64_t identity = 0;
    std::vector<std::array<float, 4>> positions;
    std::vector<uint32_t> indices;
};

/// Only immutable publication identity participates in history matching. A leaf
/// ordinal can move during repacking; neither its index nor a hash is an identity.
struct GpuParticleCollisionRowKey
{
    uint64_t colliderId = 0;
    uint32_t subShapeId = 0;
    uint32_t subShapeBits = 0;
    std::uintptr_t geometryIdentity = 0;

    [[nodiscard]] bool operator==(const GpuParticleCollisionRowKey &other) const noexcept
    {
        return std::tie(colliderId, subShapeId, subShapeBits, geometryIdentity) ==
               std::tie(other.colliderId, other.subShapeId, other.subShapeBits, other.geometryIdentity);
    }
    [[nodiscard]] bool operator<(const GpuParticleCollisionRowKey &other) const noexcept
    {
        return std::tie(colliderId, subShapeId, subShapeBits, geometryIdentity) <
               std::tie(other.colliderId, other.subShapeId, other.subShapeBits, other.geometryIdentity);
    }
};

/// Completed, frozen output of the common physics query/export work. Numeric
/// transforms, material and broadphase data stay on the device. The particle
/// submission copies these values into its existing collision tables.
struct GpuParticleCollisionDevicePayload
{
    std::shared_ptr<rhi::ComputeBuffer> records;
    std::shared_ptr<rhi::ComputeBuffer> header;
    std::shared_ptr<rhi::ComputeBuffer> gridOffsets;
    std::shared_ptr<rhi::ComputeBuffer> gridRefs;
    std::shared_ptr<const void> retained;
    uint32_t rowCount = 0;
    uint32_t staticCount = 0;
    uint32_t gridRefCount = 0;
    uint64_t revision = 0;
    uint64_t topologyRevision = 0;
    uint64_t simulationEpoch = 0;
    std::vector<GpuParticleCollisionRowKey> rows;
};

struct GpuParticleCollisionSceneSnapshot
{
    uint64_t revision = 0;
    uint64_t topologyRevision = 0;
    bool replaceMeshTopology = false;
    std::shared_ptr<const GpuParticleCollisionDevicePayload> device;
    std::vector<GpuParticleColliderRecord> staticColliders;
    std::vector<GpuParticleColliderRecord> dynamicColliders;
    std::vector<GpuParticleCollisionMeshGeometry> meshGeometries;
};

/// Owns one device-local collider table shared by all resident GPU particle
/// emitters. CPU publication prepares data; recording captures immutable upload
/// bytes, and NotifySubmission confirms or rejects that publication.
class ParticleGpuCollisionScene
{
  public:
    static constexpr uint32_t DefaultCapacity = 4096;
    static constexpr uint32_t GridAxisCells = 8;
    static constexpr uint32_t MaxGridCellCount = GridAxisCells * GridAxisCells * GridAxisCells;
    static constexpr uint32_t MaxGridReferenceCount = DefaultCapacity * MaxGridCellCount;
    static constexpr uint32_t DefaultMeshVertexCapacity = 131072;
    static constexpr uint32_t DefaultMeshIndexCapacity = 393216;
    static constexpr uint32_t DefaultMeshBvhNodeCapacity = 262144;

    ParticleGpuCollisionScene() = default;
    INFERNUX_PARTICLE_RUNTIME_API ~ParticleGpuCollisionScene();

    ParticleGpuCollisionScene(const ParticleGpuCollisionScene &) = delete;
    ParticleGpuCollisionScene &operator=(const ParticleGpuCollisionScene &) = delete;

    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool Create(rhi::Device &device, uint32_t capacity = DefaultCapacity,
                                                            uint32_t meshVertexCapacity = DefaultMeshVertexCapacity,
                                                            uint32_t meshIndexCapacity = DefaultMeshIndexCapacity,
                                                            uint32_t meshBvhNodeCapacity = DefaultMeshBvhNodeCapacity);
    INFERNUX_PARTICLE_RUNTIME_API void Destroy() noexcept;

    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool Publish(const GpuParticleCollisionSceneSnapshot &snapshot,
                                                             std::string *error = nullptr);
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool RecordPendingUpload(const rhi::TransferCommandEncoder &encoder);
    /// Report the actual submission result, not completion of command recording.
    /// A rejected submission keeps all prepared data available for the next frame.
    INFERNUX_PARTICLE_RUNTIME_API void NotifySubmission(bool submitted) noexcept;
    /// The manager retires this owner against the real particle submission's
    /// completion, since raw transfer reads are not ComputeBuffer readers.
    [[nodiscard]] std::shared_ptr<const GpuParticleCollisionDevicePayload> RecordedDevicePayload() const noexcept
    {
        return m_recordedDevicePayload;
    }

    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool IsValid() const noexcept;
    [[nodiscard]] bool HasPendingUpload() const noexcept
    {
        return m_uploadPending;
    }
    [[nodiscard]] uint64_t PublishedRevision() const noexcept
    {
        return m_publishedRevision;
    }
    [[nodiscard]] uint32_t PublishedColliderCount() const noexcept
    {
        return m_publishedColliderCount;
    }
    [[nodiscard]] uint32_t PublishedStaticColliderCount() const noexcept
    {
        return m_publishedStaticColliderCount;
    }
    [[nodiscard]] uint32_t PublishedDynamicColliderCount() const noexcept
    {
        return m_publishedColliderCount - m_publishedStaticColliderCount;
    }
    [[nodiscard]] uint32_t Capacity() const noexcept
    {
        return m_capacity;
    }
    [[nodiscard]] rhi::BufferHandle HeaderBuffer() const noexcept
    {
        return m_headerBuffer;
    }
    [[nodiscard]] rhi::BufferHandle ColliderBuffer() const noexcept
    {
        return m_colliderBuffer;
    }
    [[nodiscard]] rhi::BufferHandle GridOffsetBuffer() const noexcept
    {
        return m_gridOffsetBuffer;
    }
    [[nodiscard]] rhi::BufferHandle GridColliderIndexBuffer() const noexcept
    {
        return m_gridColliderIndexBuffer;
    }
    [[nodiscard]] rhi::BufferHandle MeshVertexBuffer() const noexcept
    {
        return m_meshVertexBuffer;
    }
    [[nodiscard]] rhi::BufferHandle MeshIndexBuffer() const noexcept
    {
        return m_meshIndexBuffer;
    }
    [[nodiscard]] rhi::BufferHandle MeshBvhBuffer() const noexcept
    {
        return m_meshBvhBuffer;
    }
    [[nodiscard]] uint32_t PublishedGridReferenceCount() const noexcept
    {
        return m_publishedGridReferenceCount;
    }
    [[nodiscard]] uint64_t PublishedTopologyRevision() const noexcept
    {
        return m_publishedTopologyRevision;
    }
    [[nodiscard]] uint32_t PublishedMeshVertexCount() const noexcept
    {
        return m_publishedMeshVertexCount;
    }
    [[nodiscard]] uint32_t PublishedMeshIndexCount() const noexcept
    {
        return m_publishedMeshIndexCount;
    }
    [[nodiscard]] uint32_t PublishedMeshBvhNodeCount() const noexcept
    {
        return m_publishedMeshBvhNodeCount;
    }

  private:
    rhi::Device *m_device = nullptr;
    rhi::BufferHandle m_headerBuffer;
    rhi::BufferHandle m_colliderBuffer;
    rhi::BufferHandle m_gridOffsetBuffer;
    rhi::BufferHandle m_gridColliderIndexBuffer;
    rhi::BufferHandle m_meshVertexBuffer;
    rhi::BufferHandle m_meshIndexBuffer;
    rhi::BufferHandle m_meshBvhBuffer;
    GpuParticleCollisionSceneHeader m_stagedHeader{};
    GpuParticleCollisionSceneHeader m_recordedHeader{};
    uint64_t m_recordedRevision = 0;
    uint64_t m_recordedTopologyRevision = 0;
    uint32_t m_recordedGridReferenceCount = 0;
    uint32_t m_capacity = 0;
    uint32_t m_meshVertexCapacity = 0;
    uint32_t m_meshIndexCapacity = 0;
    uint32_t m_meshBvhNodeCapacity = 0;
    uint32_t m_publishedColliderCount = 0;
    uint32_t m_publishedStaticColliderCount = 0;
    uint32_t m_publishedGridReferenceCount = 0;
    uint32_t m_publishedMeshVertexCount = 0;
    uint32_t m_publishedMeshIndexCount = 0;
    uint32_t m_publishedMeshBvhNodeCount = 0;
    uint64_t m_pendingRevision = 0;
    uint64_t m_publishedRevision = 0;
    uint64_t m_pendingTopologyRevision = 0;
    uint64_t m_publishedTopologyRevision = 0;
    std::vector<GpuParticleColliderRecord> m_stagedStaticColliders;
    std::vector<GpuParticleColliderRecord> m_stagedDynamicColliders;
    std::vector<uint32_t> m_stagedGridOffsets;
    std::vector<uint32_t> m_stagedGridColliderIndices;
    std::vector<std::array<float, 4>> m_stagedMeshVertices;
    std::vector<uint32_t> m_stagedMeshIndices;
    std::vector<GpuParticleCollisionBvhNode> m_stagedMeshBvhNodes;
    std::unordered_map<uint64_t, std::array<uint32_t, 4>> m_stagedGeometryByIdentity;
    std::shared_ptr<const GpuParticleCollisionDevicePayload> m_stagedDevicePayload;
    std::shared_ptr<const GpuParticleCollisionDevicePayload> m_recordedDevicePayload;
    bool m_pendingStaticUpload = false;
    bool m_pendingDynamicUpload = false;
    bool m_pendingTopologyUpload = false;
    bool m_uploadPending = false;
};

static_assert(sizeof(GpuParticleAffineTransform) == 48);
static_assert(offsetof(GpuParticleAffineTransform, row0) == 0);
static_assert(offsetof(GpuParticleAffineTransform, row1) == 16);
static_assert(offsetof(GpuParticleAffineTransform, row2) == 32);
static_assert(sizeof(GpuParticleColliderRecord) == 336);
static_assert(offsetof(GpuParticleColliderRecord, colliderToWorld) == 0);
static_assert(offsetof(GpuParticleColliderRecord, worldToCollider) == 48);
static_assert(offsetof(GpuParticleColliderRecord, previousWorldToCollider) == 96);
static_assert(offsetof(GpuParticleColliderRecord, shape) == 144);
static_assert(offsetof(GpuParticleColliderRecord, geometry) == 160);
static_assert(offsetof(GpuParticleColliderRecord, linearVelocity) == 176);
static_assert(offsetof(GpuParticleColliderRecord, angularVelocity) == 192);
static_assert(offsetof(GpuParticleColliderRecord, centerOfMass) == 208);
static_assert(offsetof(GpuParticleColliderRecord, material) == 224);
static_assert(offsetof(GpuParticleColliderRecord, worldAabbMin) == 240);
static_assert(offsetof(GpuParticleColliderRecord, worldAabbMax) == 256);
static_assert(offsetof(GpuParticleColliderRecord, previousWorldAabbMin) == 272);
static_assert(offsetof(GpuParticleColliderRecord, previousWorldAabbMax) == 288);
static_assert(offsetof(GpuParticleColliderRecord, metadata) == 304);
static_assert(offsetof(GpuParticleColliderRecord, identity) == 320);
static_assert(sizeof(GpuParticleCollisionSceneHeader) == 64);
static_assert(sizeof(GpuParticleCollisionBvhNode) == 48);

} // namespace infernux::particle
