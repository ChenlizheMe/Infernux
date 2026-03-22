/**
 * @file MeshCollider.cpp
 * @brief MeshCollider implementation — triangle mesh or convex hull shape creation.
 */

// Jolt/Jolt.h MUST be the very first include in this TU
#include <Jolt/Geometry/ConvexHullBuilder.h>
#include <Jolt/Jolt.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>

#include "MeshCollider.h"

#include "ComponentFactory.h"
#include "GameObject.h"
#include "MeshRenderer.h"
#include "Rigidbody.h"

#include <core/log/InfLog.h>
#include <nlohmann/json.hpp>
#include <unordered_map>

namespace infengine
{

// ---------------------------------------------------------------------------
// Mesh cooking — mirrors Unity's MeshCollider cooking options:
//   WeldColocatedVertices + EnableMeshCleaning.
// Merges near-duplicate vertices and removes degenerate triangles so that
// Jolt's MeshShape receives clean, well-formed geometry.
// ---------------------------------------------------------------------------
static void CookMeshGeometry(std::vector<glm::vec3> &vertices, std::vector<uint32_t> &indices)
{
    if (vertices.empty() || indices.size() < 3)
        return;

    // --- Step 1: Weld colocated vertices ---
    // Quantize vertex positions into a grid; vertices that fall into the same
    // cell are treated as identical.  This eliminates duplicates created by
    // different normals / UVs in the rendering mesh.
    constexpr float kWeldCellSize = 1e-4f;
    const float invCell = 1.0f / kWeldCellSize;

    struct IVec3Hash
    {
        size_t operator()(const glm::ivec3 &v) const
        {
            size_t h = std::hash<int>()(v.x);
            h ^= std::hash<int>()(v.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(v.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    std::unordered_map<glm::ivec3, uint32_t, IVec3Hash> posMap;
    std::vector<glm::vec3> newVerts;
    std::vector<uint32_t> remap(vertices.size());
    newVerts.reserve(vertices.size());

    for (size_t i = 0; i < vertices.size(); ++i) {
        glm::ivec3 key(static_cast<int>(std::round(vertices[i].x * invCell)),
                       static_cast<int>(std::round(vertices[i].y * invCell)),
                       static_cast<int>(std::round(vertices[i].z * invCell)));
        auto it = posMap.find(key);
        if (it != posMap.end()) {
            remap[i] = it->second;
        } else {
            uint32_t idx = static_cast<uint32_t>(newVerts.size());
            posMap[key] = idx;
            newVerts.push_back(vertices[i]);
            remap[i] = idx;
        }
    }

    for (auto &idx : indices)
        idx = remap[idx];
    vertices = std::move(newVerts);

    // --- Step 2: Remove degenerate triangles ---
    // Discard triangles with collapsed indices or near-zero area.
    constexpr float kMinTriAreaSq = 1e-12f;

    std::vector<uint32_t> clean;
    clean.reserve(indices.size());

    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        uint32_t i0 = indices[i], i1 = indices[i + 1], i2 = indices[i + 2];
        if (i0 == i1 || i1 == i2 || i0 == i2)
            continue;
        glm::vec3 e1 = vertices[i1] - vertices[i0];
        glm::vec3 e2 = vertices[i2] - vertices[i0];
        float areaSq = glm::dot(glm::cross(e1, e2), glm::cross(e1, e2));
        if (areaSq < kMinTriAreaSq)
            continue;
        clean.push_back(i0);
        clean.push_back(i1);
        clean.push_back(i2);
    }

    indices = std::move(clean);
}

INFENGINE_REGISTER_COMPONENT("MeshCollider", MeshCollider)

void MeshCollider::SetConvex(bool convex)
{
    if (m_convex == convex) {
        return;
    }
    m_convex = convex;
    RebuildShape();
}

void MeshCollider::AutoFitToMesh()
{
    // MeshCollider uses the actual mesh vertices directly,
    // so center should remain at origin (no offset needed).
    DataMut().center = glm::vec3(0.0f);
}

bool MeshCollider::CollectMeshGeometry(std::vector<glm::vec3> &outVertices, std::vector<uint32_t> &outIndices) const
{
    outVertices.clear();
    outIndices.clear();

    auto *go = GetGameObject();
    if (!go) {
        return false;
    }

    auto *mr = go->GetComponent<MeshRenderer>();
    glm::vec3 scale(1.0f);
    if (auto *tf = go->GetTransform()) {
        scale = tf->GetWorldScale();
    }

    if (mr && mr->HasInlineMesh() && !mr->GetInlineVertices().empty() && mr->GetInlineIndices().size() >= 3) {
        outVertices.reserve(mr->GetInlineVertices().size());
        for (const auto &vertex : mr->GetInlineVertices()) {
            outVertices.emplace_back(vertex.pos.x * scale.x, vertex.pos.y * scale.y, vertex.pos.z * scale.z);
        }
        outIndices = mr->GetInlineIndices();
        return true;
    }

    // PATH 2: Asset-managed mesh (loaded from .fbx/.obj/.gltf etc.)
    if (mr && mr->HasMeshAsset()) {
        auto mesh = mr->GetMeshAssetRef().Get();
        if (mesh && !mesh->GetVertices().empty() && mesh->GetIndices().size() >= 3) {
            outVertices.reserve(mesh->GetVertices().size());
            for (const auto &vertex : mesh->GetVertices()) {
                outVertices.emplace_back(vertex.pos.x * scale.x, vertex.pos.y * scale.y, vertex.pos.z * scale.z);
            }
            outIndices = mesh->GetIndices();
            return true;
        }
    }

    // PATH 3: Fallback — AABB box from MeshRenderer bounds
    glm::vec3 boundsMin(-0.5f, -0.5f, -0.5f);
    glm::vec3 boundsMax(0.5f, 0.5f, 0.5f);
    if (mr) {
        boundsMin = mr->GetLocalBoundsMin();
        boundsMax = mr->GetLocalBoundsMax();
    }

    glm::vec3 minScaled(boundsMin.x * scale.x, boundsMin.y * scale.y, boundsMin.z * scale.z);
    glm::vec3 maxScaled(boundsMax.x * scale.x, boundsMax.y * scale.y, boundsMax.z * scale.z);

    outVertices = {
        {minScaled.x, minScaled.y, minScaled.z}, {maxScaled.x, minScaled.y, minScaled.z},
        {maxScaled.x, maxScaled.y, minScaled.z}, {minScaled.x, maxScaled.y, minScaled.z},
        {minScaled.x, minScaled.y, maxScaled.z}, {maxScaled.x, minScaled.y, maxScaled.z},
        {maxScaled.x, maxScaled.y, maxScaled.z}, {minScaled.x, maxScaled.y, maxScaled.z},
    };

    outIndices = {
        0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 3, 2, 6, 3, 6, 7, 1, 5, 6, 1, 6, 2, 0, 3, 7, 0, 7, 4,
    };
    return true;
}

void *MeshCollider::CreateJoltShapeRaw() const
{
    std::vector<glm::vec3> vertices;
    std::vector<uint32_t> indices;
    if (!CollectMeshGeometry(vertices, indices) || vertices.empty()) {
        JPH::Shape *fallback = new JPH::BoxShape(JPH::Vec3(0.5f, 0.5f, 0.5f));
        glm::vec3 center = GetCenter();
        if (center != glm::vec3(0.0f)) {
            fallback = new JPH::RotatedTranslatedShape(JPH::Vec3(center.x, center.y, center.z), JPH::Quat::sIdentity(),
                                                       fallback);
        }
        return fallback;
    }

    bool useConvex = m_convex;
    if (auto *rb = GetCachedRigidbody(); rb && !rb->IsKinematic()) {
        useConvex = true;
    }

    INFLOG_INFO("MeshCollider: creating shape — verts=", vertices.size(), ", tris=", indices.size() / 3,
                ", convex=", (useConvex ? "true" : "false"), " (requested=", (m_convex ? "true" : "false"), ")");

    JPH::Shape *shape = nullptr;
    if (useConvex) {
        // Build a convex hull with at most 100 output vertices using all input points.
        // ConvexHullBuilder iteratively selects the best vertices, giving a much
        // better approximation than naively sub-sampling the input.
        constexpr int kMaxConvexVerts = 100;

        JPH::Array<JPH::Vec3> allPoints;
        allPoints.reserve(static_cast<int>(vertices.size()));
        for (const auto &v : vertices) {
            allPoints.push_back(JPH::Vec3(v.x, v.y, v.z));
        }

        JPH::ConvexHullBuilder builder(allPoints);
        const char *buildError = nullptr;
        auto buildResult = builder.Initialize(kMaxConvexVerts, 1.0e-3f, buildError);

        JPH::Array<JPH::Vec3> hullPoints;
        // Build a mapping from original index → compact hull index for gizmo edges
        std::unordered_map<int, uint32_t> origToCompact;

        if (buildResult == JPH::ConvexHullBuilder::EResult::Success ||
            buildResult == JPH::ConvexHullBuilder::EResult::MaxVerticesReached) {
            // Collect unique vertex indices from hull faces
            std::vector<bool> used(allPoints.size(), false);
            for (const auto *face : builder.GetFaces()) {
                const auto *edge = face->mFirstEdge;
                do {
                    used[static_cast<size_t>(edge->mStartIdx)] = true;
                    edge = edge->mNextEdge;
                } while (edge != face->mFirstEdge);
            }
            for (size_t i = 0; i < allPoints.size(); ++i) {
                if (used[i]) {
                    origToCompact[static_cast<int>(i)] = static_cast<uint32_t>(hullPoints.size());
                    hullPoints.push_back(allPoints[i]);
                }
            }

            // Store hull for gizmo display (un-scale to local space)
            glm::vec3 scale(1.0f);
            if (auto *go = GetGameObject()) {
                if (auto *tf = go->GetTransform()) {
                    scale = tf->GetWorldScale();
                }
            }
            const float invSx = (scale.x != 0.0f) ? (1.0f / scale.x) : 1.0f;
            const float invSy = (scale.y != 0.0f) ? (1.0f / scale.y) : 1.0f;
            const float invSz = (scale.z != 0.0f) ? (1.0f / scale.z) : 1.0f;

            m_convexHullPositions.clear();
            m_convexHullPositions.reserve(hullPoints.size());
            for (const auto &p : hullPoints) {
                m_convexHullPositions.emplace_back(p.GetX() * invSx, p.GetY() * invSy, p.GetZ() * invSz);
            }

            m_convexHullEdges.clear();
            for (const auto *face : builder.GetFaces()) {
                const auto *edge = face->mFirstEdge;
                do {
                    uint32_t a = origToCompact[edge->mStartIdx];
                    uint32_t b = origToCompact[edge->mNextEdge->mStartIdx];
                    m_convexHullEdges.push_back(a);
                    m_convexHullEdges.push_back(b);
                    edge = edge->mNextEdge;
                } while (edge != face->mFirstEdge);
            }
        }
        if (hullPoints.empty()) {
            hullPoints = std::move(allPoints);
            m_convexHullPositions.clear();
            m_convexHullEdges.clear();
        }

        JPH::ConvexHullShapeSettings settings(hullPoints);
        JPH::ShapeSettings::ShapeResult result = settings.Create();
        if (result.HasError()) {
            shape = new JPH::BoxShape(JPH::Vec3(0.5f, 0.5f, 0.5f));
        } else {
            shape = const_cast<JPH::Shape *>(result.Get().GetPtr());
            shape->AddRef();
        }
    } else {
        // Cook the mesh (Unity: WeldColocatedVertices + EnableMeshCleaning)
        size_t origVerts = vertices.size();
        size_t origTris = indices.size() / 3;
        CookMeshGeometry(vertices, indices);
        INFLOG_INFO("MeshCollider: mesh cooking — verts ", origVerts, "→", vertices.size(), ", tris ", origTris, "→",
                    indices.size() / 3);
        if (indices.size() < 3) {
            shape = new JPH::BoxShape(JPH::Vec3(0.5f, 0.5f, 0.5f));
        } else {
            JPH::MeshShapeSettings settings;
            settings.mTriangleVertices.reserve(static_cast<int>(vertices.size()));
            for (const auto &v : vertices) {
                settings.mTriangleVertices.emplace_back(v.x, v.y, v.z);
            }
            settings.mIndexedTriangles.reserve(static_cast<int>(indices.size() / 3));
            for (size_t i = 0; i + 2 < indices.size(); i += 3) {
                settings.mIndexedTriangles.emplace_back(indices[i], indices[i + 1], indices[i + 2]);
            }
            settings.SetEmbedded();

            JPH::ShapeSettings::ShapeResult result = settings.Create();
            if (result.HasError()) {
                shape = new JPH::BoxShape(JPH::Vec3(0.5f, 0.5f, 0.5f));
            } else {
                shape = const_cast<JPH::Shape *>(result.Get().GetPtr());
                shape->AddRef();
            }
        }
    }

    glm::vec3 center = GetCenter();
    if (auto *go = GetGameObject()) {
        if (auto *tf = go->GetTransform()) {
            center *= tf->GetWorldScale();
        }
    }
    if (center != glm::vec3(0.0f)) {
        shape = new JPH::RotatedTranslatedShape(JPH::Vec3(center.x, center.y, center.z), JPH::Quat::sIdentity(), shape);
    }

    return shape;
}

std::string MeshCollider::Serialize() const
{
    auto baseJson = nlohmann::json::parse(Collider::Serialize());
    baseJson["convex"] = m_convex;
    return baseJson.dump();
}

bool MeshCollider::Deserialize(const std::string &jsonStr)
{
    if (!Collider::Deserialize(jsonStr)) {
        return false;
    }

    try {
        auto j = nlohmann::json::parse(jsonStr);
        m_convex = j.value("convex", false);
        RebuildShape();
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace infengine
