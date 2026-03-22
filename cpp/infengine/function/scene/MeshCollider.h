/**
 * @file MeshCollider.h
 * @brief Triangle mesh / convex hull collider (Unity: MeshCollider).
 */

#pragma once

#include "Collider.h"

namespace infengine
{

class MeshCollider : public Collider
{
  public:
    MeshCollider() = default;
    ~MeshCollider() override = default;

    [[nodiscard]] const char *GetTypeName() const override
    {
        return "MeshCollider";
    }

    [[nodiscard]] bool IsConvex() const
    {
        return m_convex;
    }
    void SetConvex(bool convex);

    [[nodiscard]] void *CreateJoltShapeRaw() const override;

    [[nodiscard]] std::string Serialize() const override;
    bool Deserialize(const std::string &jsonStr) override;

    void AutoFitToMesh() override;

    /// Convex hull positions (local space) cached after shape creation.
    const std::vector<glm::vec3> &GetConvexHullPositions() const
    {
        return m_convexHullPositions;
    }
    /// Convex hull edge pairs [a0,b0, a1,b1, …] cached after shape creation.
    const std::vector<uint32_t> &GetConvexHullEdges() const
    {
        return m_convexHullEdges;
    }

  private:
    bool CollectMeshGeometry(std::vector<glm::vec3> &outVertices, std::vector<uint32_t> &outIndices) const;

    bool m_convex = false;
    mutable std::vector<glm::vec3> m_convexHullPositions;
    mutable std::vector<uint32_t> m_convexHullEdges;
};

} // namespace infengine
