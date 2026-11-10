/**
 * @file MeshCollider.h
 * @brief Triangle mesh / convex hull collider (Unity: MeshCollider).
 */

#pragma once

#include "Collider.h"

#include <cstdint>
#include <memory>
#include <utility>

namespace infernux
{

struct CookedGeometry;
struct MeshCookingResult;

class MeshCollider : public Collider
{
  public:
    MeshCollider() = default;
    ~MeshCollider() override = default;

    [[nodiscard]] const char *GetTypeName() const override
    {
        return "MeshCollider";
    }

    void Awake() override;

    [[nodiscard]] bool IsConvex() const
    {
        return m_convex;
    }
    void SetConvex(bool convex);

    [[nodiscard]] const std::string &GetShapeError() const
    {
        return m_shapeError;
    }

    [[nodiscard]] bool IsCooking() const
    {
        return m_cookingPending;
    }

    [[nodiscard]] JPH::RefConst<JPH::Shape> CreateJoltShape() const override;

    static void ClearCookingCache();
    [[nodiscard]] static std::pair<size_t, size_t> GetCookingCacheStats();
    [[nodiscard]] static size_t GetPendingCookingCount();
    [[nodiscard]] static size_t GetAsyncCookingSubmissionCount();

    /// Commit completed worker payloads at a main-thread physics safe point.
    /// An explicit physics sync passes waitForAll=true and acts as a barrier.
    static void FlushCompletedCooking(bool waitForAll = false);

    /// Called when the sibling MeshRenderer changes collision geometry.
    void OnMeshGeometryChanged();

    [[nodiscard]] nlohmann::json SerializeDocument() const override;
    static void ValidateSerializedDocument(const nlohmann::json &document);
    bool DeserializeDocument(const nlohmann::json &document) override;
    [[nodiscard]] std::unique_ptr<Component> Clone() const override;

    void AutoFitToMesh() override;

    /// Local-space debug hull from the mesh committed to the physics body.
    [[nodiscard]] const std::vector<glm::vec3> &GetConvexHullPositions() const;
    [[nodiscard]] const std::vector<uint32_t> &GetConvexHullEdges() const;
    [[nodiscard]] uint64_t GetCollisionGeometryRevision() const
    {
        return m_collisionGeometryRevision;
    }

  private:
    friend class PhysicsWorld;

    void CaptureMeshGeometry() const;
    bool CollectMeshGeometry(std::vector<glm::vec3> &outVertices, std::vector<uint32_t> &outIndices) const;
    void CompleteCooking(std::shared_ptr<const MeshCookingResult> result, uint64_t revision);
    void PublishCookedGeometry();
    void RequestShapeUpdate();
    void InvalidatePendingCooking() const;

    bool m_convex = false;
    // Unscaled source selected at initialization or an explicit recook.
    mutable bool m_hasGeometrySnapshot = false;
    mutable std::vector<glm::vec3> m_sourceVertices;
    mutable std::vector<uint32_t> m_sourceIndices;
    std::shared_ptr<const CookedGeometry> m_publishedGeometry;
    uint64_t m_collisionGeometryRevision = 0;
    mutable std::string m_shapeError;
    mutable uint64_t m_cookingRevision = 0;
    mutable uint64_t m_pendingHashA = 0;
    mutable uint64_t m_pendingHashB = 0;
    mutable size_t m_pendingVertexCount = 0;
    mutable size_t m_pendingIndexCount = 0;
    mutable bool m_pendingConvex = false;
    mutable bool m_cookingPending = false;
    // Consumer ownership is independent of reusable cache eviction. A compound
    // can defer publication after this child has already finished cooking.
    mutable std::shared_ptr<const MeshCookingResult> m_cookingResult;
};

} // namespace infernux
