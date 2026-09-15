#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <function/renderer/InxRenderStruct.h>
#include <unordered_map>
#include <vector>

namespace infernux
{

/**
 * @brief Primitive mesh data for built-in shapes.
 *
 * Provides vertex and index data for common primitives like cubes, spheres, etc.
 * All meshes are centered at origin with unit size.
 * All meshes include proper normals and tangents for lighting.
 */
class PrimitiveMeshes
{
  public:
    /// @brief Get cube vertices (24 vertices for proper normals per face)
    static const std::vector<Vertex> &GetCubeVertices()
    {
        static std::vector<Vertex> vertices = CreateCubeVertices();
        return vertices;
    }

    /// @brief Get cube indices
    static const std::vector<uint32_t> &GetCubeIndices()
    {
        static std::vector<uint32_t> indices = CreateCubeIndices();
        return indices;
    }

    /// @brief Get quad vertices (for UI, sprites, etc.)
    static const std::vector<Vertex> &GetQuadVertices()
    {
        static std::vector<Vertex> vertices = {
            Vertex::CreateFull({-0.5f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f},
                               {0.0f, 0.0f}),
            Vertex::CreateFull({0.5f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f},
                               {1.0f, 0.0f}),
            Vertex::CreateFull({0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f},
                               {1.0f, 1.0f}),
            Vertex::CreateFull({-0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f},
                               {0.0f, 1.0f}),
        };
        return vertices;
    }

    /// @brief Get quad indices
    static const std::vector<uint32_t> &GetQuadIndices()
    {
        static std::vector<uint32_t> indices = {0, 1, 2, 2, 3, 0};
        return indices;
    }

    /// @brief Get sphere vertices (subdivided icosahedron)
    static const std::vector<Vertex> &GetSphereVertices()
    {
        return GetGeodesicSphereMesh().vertices;
    }

    /// @brief Get sphere indices
    static const std::vector<uint32_t> &GetSphereIndices()
    {
        return GetGeodesicSphereMesh().indices;
    }

    /// @brief Get capsule vertices
    static const std::vector<Vertex> &GetCapsuleVertices()
    {
        return GetCapsuleMesh().vertices;
    }

    /// @brief Get capsule indices
    static const std::vector<uint32_t> &GetCapsuleIndices()
    {
        return GetCapsuleMesh().indices;
    }

    /// @brief Get plane vertices (XZ plane, facing up)
    static const std::vector<Vertex> &GetPlaneVertices()
    {
        static std::vector<Vertex> vertices = {
            Vertex::CreateFull({-0.5f, 0.0f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f},
                               {0.0f, 0.0f}),
            Vertex::CreateFull({0.5f, 0.0f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f},
                               {1.0f, 0.0f}),
            Vertex::CreateFull({0.5f, 0.0f, 0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f},
                               {1.0f, 1.0f}),
            Vertex::CreateFull({-0.5f, 0.0f, 0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f},
                               {0.0f, 1.0f}),
        };
        return vertices;
    }

    /// @brief Get plane indices
    static const std::vector<uint32_t> &GetPlaneIndices()
    {
        static std::vector<uint32_t> indices = {0, 2, 1, 0, 3, 2};
        return indices;
    }

    /// @brief Get cylinder vertices
    static const std::vector<Vertex> &GetCylinderVertices()
    {
        static std::vector<Vertex> vertices = CreateCylinderVertices(16, 0.5f, 1.0f);
        return vertices;
    }

    /// @brief Get cylinder indices
    static const std::vector<uint32_t> &GetCylinderIndices()
    {
        static std::vector<uint32_t> indices = CreateCylinderIndices(16);
        return indices;
    }

    /// @brief Get skybox cube vertices (unit cube, positions used as direction vectors)
    static const std::vector<Vertex> &GetSkyboxCubeVertices()
    {
        static std::vector<Vertex> vertices = CreateSkyboxCubeVertices();
        return vertices;
    }

    /// @brief Get skybox cube indices
    static const std::vector<uint32_t> &GetSkyboxCubeIndices()
    {
        static std::vector<uint32_t> indices = CreateSkyboxCubeIndices();
        return indices;
    }

    /// @brief Calculate tangent from normal (generates an orthogonal tangent)
    static glm::vec4 CalculateTangent(const glm::vec3 &normal)
    {
        // Find a vector not parallel to normal
        glm::vec3 up = std::abs(normal.y) < 0.999f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
        glm::vec3 tangent = glm::normalize(glm::cross(up, normal));
        return glm::vec4(tangent, 1.0f); // Handedness = 1
    }

  private:
    struct GeneratedMesh
    {
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
    };

    static const GeneratedMesh &GetGeodesicSphereMesh()
    {
        static const GeneratedMesh mesh = CreateGeodesicSphereMesh(4, 0.5f);
        return mesh;
    }

    static const GeneratedMesh &GetCapsuleMesh()
    {
        static const GeneratedMesh mesh = CreateCapsuleMesh(32, 8, 0.5f, 2.0f);
        return mesh;
    }

    static std::vector<Vertex> CreateCubeVertices()
    {
        // 24 vertices - 4 per face for proper normals
        std::vector<Vertex> vertices;
        vertices.reserve(24);

        // All faces use white vertex color - material baseColor should control color
        glm::vec3 white(1.0f, 1.0f, 1.0f);

        // Front face (z = 0.5) - Normal: (0, 0, 1), Tangent: (1, 0, 0)
        glm::vec3 frontNormal(0.0f, 0.0f, 1.0f);
        glm::vec4 frontTangent(1.0f, 0.0f, 0.0f, 1.0f);
        vertices.push_back(Vertex::CreateFull({-0.5f, -0.5f, 0.5f}, frontNormal, frontTangent, white, {0.0f, 0.0f}));
        vertices.push_back(Vertex::CreateFull({0.5f, -0.5f, 0.5f}, frontNormal, frontTangent, white, {1.0f, 0.0f}));
        vertices.push_back(Vertex::CreateFull({0.5f, 0.5f, 0.5f}, frontNormal, frontTangent, white, {1.0f, 1.0f}));
        vertices.push_back(Vertex::CreateFull({-0.5f, 0.5f, 0.5f}, frontNormal, frontTangent, white, {0.0f, 1.0f}));

        // Back face (z = -0.5) - Normal: (0, 0, -1), Tangent: (-1, 0, 0)
        glm::vec3 backNormal(0.0f, 0.0f, -1.0f);
        glm::vec4 backTangent(-1.0f, 0.0f, 0.0f, 1.0f);
        vertices.push_back(Vertex::CreateFull({0.5f, -0.5f, -0.5f}, backNormal, backTangent, white, {0.0f, 0.0f}));
        vertices.push_back(Vertex::CreateFull({-0.5f, -0.5f, -0.5f}, backNormal, backTangent, white, {1.0f, 0.0f}));
        vertices.push_back(Vertex::CreateFull({-0.5f, 0.5f, -0.5f}, backNormal, backTangent, white, {1.0f, 1.0f}));
        vertices.push_back(Vertex::CreateFull({0.5f, 0.5f, -0.5f}, backNormal, backTangent, white, {0.0f, 1.0f}));

        // Top face (y = 0.5) - Normal: (0, 1, 0), Tangent: (1, 0, 0)
        glm::vec3 topNormal(0.0f, 1.0f, 0.0f);
        glm::vec4 topTangent(1.0f, 0.0f, 0.0f, 1.0f);
        vertices.push_back(Vertex::CreateFull({-0.5f, 0.5f, 0.5f}, topNormal, topTangent, white, {0.0f, 0.0f}));
        vertices.push_back(Vertex::CreateFull({0.5f, 0.5f, 0.5f}, topNormal, topTangent, white, {1.0f, 0.0f}));
        vertices.push_back(Vertex::CreateFull({0.5f, 0.5f, -0.5f}, topNormal, topTangent, white, {1.0f, 1.0f}));
        vertices.push_back(Vertex::CreateFull({-0.5f, 0.5f, -0.5f}, topNormal, topTangent, white, {0.0f, 1.0f}));

        // Bottom face (y = -0.5) - Normal: (0, -1, 0), Tangent: (1, 0, 0)
        glm::vec3 bottomNormal(0.0f, -1.0f, 0.0f);
        glm::vec4 bottomTangent(1.0f, 0.0f, 0.0f, 1.0f);
        vertices.push_back(Vertex::CreateFull({-0.5f, -0.5f, -0.5f}, bottomNormal, bottomTangent, white, {0.0f, 0.0f}));
        vertices.push_back(Vertex::CreateFull({0.5f, -0.5f, -0.5f}, bottomNormal, bottomTangent, white, {1.0f, 0.0f}));
        vertices.push_back(Vertex::CreateFull({0.5f, -0.5f, 0.5f}, bottomNormal, bottomTangent, white, {1.0f, 1.0f}));
        vertices.push_back(Vertex::CreateFull({-0.5f, -0.5f, 0.5f}, bottomNormal, bottomTangent, white, {0.0f, 1.0f}));

        // Right face (x = 0.5) - Normal: (1, 0, 0), Tangent: (0, 0, -1)
        glm::vec3 rightNormal(1.0f, 0.0f, 0.0f);
        glm::vec4 rightTangent(0.0f, 0.0f, -1.0f, 1.0f);
        vertices.push_back(Vertex::CreateFull({0.5f, -0.5f, 0.5f}, rightNormal, rightTangent, white, {0.0f, 0.0f}));
        vertices.push_back(Vertex::CreateFull({0.5f, -0.5f, -0.5f}, rightNormal, rightTangent, white, {1.0f, 0.0f}));
        vertices.push_back(Vertex::CreateFull({0.5f, 0.5f, -0.5f}, rightNormal, rightTangent, white, {1.0f, 1.0f}));
        vertices.push_back(Vertex::CreateFull({0.5f, 0.5f, 0.5f}, rightNormal, rightTangent, white, {0.0f, 1.0f}));

        // Left face (x = -0.5) - Normal: (-1, 0, 0), Tangent: (0, 0, 1)
        glm::vec3 leftNormal(-1.0f, 0.0f, 0.0f);
        glm::vec4 leftTangent(0.0f, 0.0f, 1.0f, 1.0f);
        vertices.push_back(Vertex::CreateFull({-0.5f, -0.5f, -0.5f}, leftNormal, leftTangent, white, {0.0f, 0.0f}));
        vertices.push_back(Vertex::CreateFull({-0.5f, -0.5f, 0.5f}, leftNormal, leftTangent, white, {1.0f, 0.0f}));
        vertices.push_back(Vertex::CreateFull({-0.5f, 0.5f, 0.5f}, leftNormal, leftTangent, white, {1.0f, 1.0f}));
        vertices.push_back(Vertex::CreateFull({-0.5f, 0.5f, -0.5f}, leftNormal, leftTangent, white, {0.0f, 1.0f}));

        return vertices;
    }

    static std::vector<uint32_t> CreateCubeIndices()
    {
        std::vector<uint32_t> indices;
        indices.reserve(36);

        // 6 faces, 2 triangles each, 3 indices per triangle
        for (uint32_t face = 0; face < 6; ++face) {
            uint32_t base = face * 4;
            // First triangle
            indices.push_back(base + 0);
            indices.push_back(base + 1);
            indices.push_back(base + 2);
            // Second triangle
            indices.push_back(base + 2);
            indices.push_back(base + 3);
            indices.push_back(base + 0);
        }

        return indices;
    }

    // ========================================================================
    // Sphere generation (geodesic icosphere with seam-aware UV duplication)
    // ========================================================================
    static GeneratedMesh CreateGeodesicSphereMesh(int subdivisions, float radius)
    {
        const float PI = 3.14159265358979323846f;
        const float goldenRatio = (1.0f + std::sqrt(5.0f)) * 0.5f;
        std::vector<glm::vec3> positions = {
            {-1.0f, goldenRatio, 0.0f},  {1.0f, goldenRatio, 0.0f},   {-1.0f, -goldenRatio, 0.0f},
            {1.0f, -goldenRatio, 0.0f},  {0.0f, -1.0f, goldenRatio},  {0.0f, 1.0f, goldenRatio},
            {0.0f, -1.0f, -goldenRatio}, {0.0f, 1.0f, -goldenRatio},  {goldenRatio, 0.0f, -1.0f},
            {goldenRatio, 0.0f, 1.0f},   {-goldenRatio, 0.0f, -1.0f}, {-goldenRatio, 0.0f, 1.0f},
        };
        for (glm::vec3 &position : positions)
            position = glm::normalize(position) * radius;

        std::vector<std::array<uint32_t, 3>> faces = {
            {0, 11, 5},  {0, 5, 1},  {0, 1, 7},  {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4},
            {11, 10, 2}, {10, 7, 6}, {7, 1, 8},  {3, 9, 4},  {3, 4, 2},   {3, 2, 6}, {3, 6, 8},
            {3, 8, 9},   {4, 9, 5},  {2, 4, 11}, {6, 2, 10}, {8, 6, 7},   {9, 8, 1},
        };

        for (int level = 0; level < subdivisions; ++level) {
            std::unordered_map<uint64_t, uint32_t> midpointCache;
            auto midpoint = [&](uint32_t first, uint32_t second) {
                const uint32_t lower = std::min(first, second);
                const uint32_t upper = std::max(first, second);
                const uint64_t key = (static_cast<uint64_t>(lower) << 32U) | upper;
                const auto existing = midpointCache.find(key);
                if (existing != midpointCache.end())
                    return existing->second;
                const uint32_t index = static_cast<uint32_t>(positions.size());
                positions.push_back(glm::normalize(positions[first] + positions[second]) * radius);
                midpointCache.emplace(key, index);
                return index;
            };

            std::vector<std::array<uint32_t, 3>> subdivided;
            subdivided.reserve(faces.size() * 4);
            for (const auto &face : faces) {
                const uint32_t ab = midpoint(face[0], face[1]);
                const uint32_t bc = midpoint(face[1], face[2]);
                const uint32_t ca = midpoint(face[2], face[0]);
                subdivided.push_back({face[0], ab, ca});
                subdivided.push_back({face[1], bc, ab});
                subdivided.push_back({face[2], ca, bc});
                subdivided.push_back({ab, bc, ca});
            }
            faces = std::move(subdivided);
        }

        GeneratedMesh mesh;
        mesh.vertices.reserve(positions.size() + 64);
        mesh.indices.reserve(faces.size() * 3);
        std::unordered_map<uint64_t, uint32_t> renderVertexCache;
        for (auto face : faces) {
            const glm::vec3 &a = positions[face[0]];
            const glm::vec3 &b = positions[face[1]];
            const glm::vec3 &c = positions[face[2]];
            if (glm::dot(glm::cross(b - a, c - a), a + b + c) < 0.0f)
                std::swap(face[1], face[2]);

            std::array<float, 3> u{};
            std::array<float, 3> v{};
            std::array<bool, 3> seamShifted{};
            std::array<bool, 3> pole{};
            for (size_t corner = 0; corner < 3; ++corner) {
                const glm::vec3 normal = glm::normalize(positions[face[corner]]);
                u[corner] = std::atan2(normal.z, normal.x) / (2.0f * PI) + 0.5f;
                v[corner] = std::acos(std::clamp(normal.y, -1.0f, 1.0f)) / PI;
                pole[corner] = normal.x * normal.x + normal.z * normal.z < 1.0e-12f;
            }
            // Longitude is undefined at a pole. Choose the triangle's UV
            // interval first, then give its pole a face-local midpoint.
            for (size_t corner = 0; corner < 3; ++corner)
                if (pole[corner])
                    u[corner] = u[(corner + 1) % 3];
            if (*std::max_element(u.begin(), u.end()) - *std::min_element(u.begin(), u.end()) > 0.5f) {
                for (size_t corner = 0; corner < 3; ++corner) {
                    if (u[corner] < 0.5f) {
                        u[corner] += 1.0f;
                        seamShifted[corner] = true;
                    }
                }
            }

            for (size_t corner = 0; corner < 3; ++corner)
                if (pole[corner])
                    u[corner] = (u[(corner + 1) % 3] + u[(corner + 2) % 3]) * 0.5f;

            for (size_t corner = 0; corner < 3; ++corner) {
                const uint32_t sourceIndex = face[corner];
                const uint64_t cacheKey =
                    pole[corner] ? (uint64_t{1} << 63U) | static_cast<uint64_t>(mesh.indices.size())
                                 : (static_cast<uint64_t>(sourceIndex) << 1U) | (seamShifted[corner] ? 1U : 0U);
                auto existing = renderVertexCache.find(cacheKey);
                if (existing == renderVertexCache.end()) {
                    const glm::vec3 normal = glm::normalize(positions[sourceIndex]);
                    glm::vec3 tangent(-normal.z, 0.0f, normal.x);
                    if (glm::dot(tangent, tangent) < 1e-8f)
                        tangent = glm::vec3(CalculateTangent(normal));
                    else
                        tangent = glm::normalize(tangent);
                    const uint32_t renderIndex = static_cast<uint32_t>(mesh.vertices.size());
                    mesh.vertices.push_back(Vertex::CreateFull(positions[sourceIndex], normal, glm::vec4(tangent, 1.0f),
                                                               {1.0f, 1.0f, 1.0f}, {u[corner], v[corner]}));
                    existing = renderVertexCache.emplace(cacheKey, renderIndex).first;
                }
                mesh.indices.push_back(existing->second);
            }
        }
        return mesh;
    }

    // ========================================================================
    // Capsule generation (cylinder + hemispheres)
    // ========================================================================
    static GeneratedMesh CreateCapsuleMesh(int segments, int hemisphereRings, float radius, float height)
    {
        GeneratedMesh mesh;
        const float PI = 3.14159265358979323846f;
        const float halfCylinder = (height - 2.0f * radius) * 0.5f;
        const glm::vec3 white(1.0f);
        mesh.vertices.push_back(Vertex::CreateFull({0.0f, halfCylinder + radius, 0.0f}, {0.0f, 1.0f, 0.0f},
                                                   {1.0f, 0.0f, 0.0f, 1.0f}, white, {0.5f, 0.0f}));

        std::vector<uint32_t> ringStarts;
        auto appendRing = [&](float y, float ringRadius, float normalY, float normalRadius) {
            ringStarts.push_back(static_cast<uint32_t>(mesh.vertices.size()));
            for (int seg = 0; seg <= segments; ++seg) {
                const float theta = 2.0f * PI * static_cast<float>(seg) / static_cast<float>(segments);
                const float cosine = std::cos(theta);
                const float sine = std::sin(theta);
                const glm::vec3 normal(cosine * normalRadius, normalY, sine * normalRadius);
                const glm::vec3 tangent(-sine, 0.0f, cosine);
                const float u = static_cast<float>(seg) / static_cast<float>(segments);
                const float v = (height * 0.5f - y) / height;
                mesh.vertices.push_back(Vertex::CreateFull({cosine * ringRadius, y, sine * ringRadius}, normal,
                                                           glm::vec4(tangent, 1.0f), white, {u, v}));
            }
        };

        for (int ring = 1; ring <= hemisphereRings; ++ring) {
            const float phi = (PI * 0.5f) * static_cast<float>(ring) / static_cast<float>(hemisphereRings);
            appendRing(halfCylinder + std::cos(phi) * radius, std::sin(phi) * radius, std::cos(phi), std::sin(phi));
        }
        appendRing(-halfCylinder, radius, 0.0f, 1.0f);
        for (int ring = 1; ring < hemisphereRings; ++ring) {
            const float phi = (PI * 0.5f) * static_cast<float>(ring) / static_cast<float>(hemisphereRings);
            appendRing(-halfCylinder - std::sin(phi) * radius, std::cos(phi) * radius, -std::sin(phi), std::cos(phi));
        }

        const uint32_t bottomPole = static_cast<uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back(Vertex::CreateFull({0.0f, -halfCylinder - radius, 0.0f}, {0.0f, -1.0f, 0.0f},
                                                   {1.0f, 0.0f, 0.0f, 1.0f}, white, {0.5f, 1.0f}));

        const uint32_t firstRing = ringStarts.front();
        for (int segment = 0; segment < segments; ++segment) {
            mesh.indices.push_back(0);
            mesh.indices.push_back(firstRing + static_cast<uint32_t>(segment + 1));
            mesh.indices.push_back(firstRing + static_cast<uint32_t>(segment));
        }
        for (size_t ring = 0; ring + 1 < ringStarts.size(); ++ring) {
            const uint32_t currentStart = ringStarts[ring];
            const uint32_t nextStart = ringStarts[ring + 1];
            for (int segment = 0; segment < segments; ++segment) {
                const uint32_t current = currentStart + static_cast<uint32_t>(segment);
                const uint32_t next = current + 1;
                const uint32_t below = nextStart + static_cast<uint32_t>(segment);
                const uint32_t belowNext = below + 1;
                mesh.indices.insert(mesh.indices.end(), {current, next, below, next, belowNext, below});
            }
        }
        const uint32_t lastRing = ringStarts.back();
        for (int segment = 0; segment < segments; ++segment) {
            mesh.indices.push_back(lastRing + static_cast<uint32_t>(segment));
            mesh.indices.push_back(lastRing + static_cast<uint32_t>(segment + 1));
            mesh.indices.push_back(bottomPole);
        }
        return mesh;
    }

    // ========================================================================
    // Cylinder generation
    // ========================================================================
    static std::vector<Vertex> CreateCylinderVertices(int segments, float radius, float height)
    {
        std::vector<Vertex> vertices;
        const float PI = 3.14159265358979323846f;
        float halfHeight = height / 2.0f;

        // Top cap center
        glm::vec3 topNormal(0.0f, 1.0f, 0.0f);
        glm::vec4 topTangent(1.0f, 0.0f, 0.0f, 1.0f);
        vertices.push_back(
            Vertex::CreateFull({0.0f, halfHeight, 0.0f}, topNormal, topTangent, {1.0f, 1.0f, 1.0f}, {0.5f, 0.5f}));

        // Top cap edge
        for (int seg = 0; seg <= segments; ++seg) {
            float theta = 2.0f * PI * seg / segments;
            float x = std::cos(theta) * radius;
            float z = std::sin(theta) * radius;
            float u = std::cos(theta) * 0.5f + 0.5f;
            float v = std::sin(theta) * 0.5f + 0.5f;
            vertices.push_back(
                Vertex::CreateFull({x, halfHeight, z}, topNormal, topTangent, {1.0f, 1.0f, 1.0f}, {u, v}));
        }

        // Side top ring
        for (int seg = 0; seg <= segments; ++seg) {
            float theta = 2.0f * PI * seg / segments;
            float x = std::cos(theta) * radius;
            float z = std::sin(theta) * radius;
            glm::vec3 sideNormal = glm::normalize(glm::vec3(x, 0.0f, z));
            glm::vec4 sideTangent(0.0f, 1.0f, 0.0f, 1.0f);
            float u = static_cast<float>(seg) / segments;
            vertices.push_back(
                Vertex::CreateFull({x, halfHeight, z}, sideNormal, sideTangent, {1.0f, 1.0f, 1.0f}, {u, 0.0f}));
        }

        // Side bottom ring
        for (int seg = 0; seg <= segments; ++seg) {
            float theta = 2.0f * PI * seg / segments;
            float x = std::cos(theta) * radius;
            float z = std::sin(theta) * radius;
            glm::vec3 sideNormal = glm::normalize(glm::vec3(x, 0.0f, z));
            glm::vec4 sideTangent(0.0f, 1.0f, 0.0f, 1.0f);
            float u = static_cast<float>(seg) / segments;
            vertices.push_back(
                Vertex::CreateFull({x, -halfHeight, z}, sideNormal, sideTangent, {1.0f, 1.0f, 1.0f}, {u, 1.0f}));
        }

        // Bottom cap center
        glm::vec3 bottomNormal(0.0f, -1.0f, 0.0f);
        glm::vec4 bottomTangent(1.0f, 0.0f, 0.0f, 1.0f);
        vertices.push_back(Vertex::CreateFull({0.0f, -halfHeight, 0.0f}, bottomNormal, bottomTangent,
                                              {1.0f, 1.0f, 1.0f}, {0.5f, 0.5f}));

        // Bottom cap edge
        for (int seg = 0; seg <= segments; ++seg) {
            float theta = 2.0f * PI * seg / segments;
            float x = std::cos(theta) * radius;
            float z = std::sin(theta) * radius;
            float u = std::cos(theta) * 0.5f + 0.5f;
            float v = std::sin(theta) * 0.5f + 0.5f;
            vertices.push_back(
                Vertex::CreateFull({x, -halfHeight, z}, bottomNormal, bottomTangent, {1.0f, 1.0f, 1.0f}, {u, v}));
        }

        return vertices;
    }

    static std::vector<uint32_t> CreateCylinderIndices(int segments)
    {
        std::vector<uint32_t> indices;

        // Top cap (fan from center, CCW from above → normal up)
        uint32_t topCenter = 0;
        for (int seg = 0; seg < segments; ++seg) {
            indices.push_back(topCenter);
            indices.push_back(static_cast<uint32_t>(1 + seg + 1));
            indices.push_back(static_cast<uint32_t>(1 + seg));
        }

        // Side (CCW winding for outward-facing normals)
        uint32_t sideTopStart = static_cast<uint32_t>(1 + segments + 1);
        uint32_t sideBottomStart = static_cast<uint32_t>(sideTopStart + segments + 1);
        for (int seg = 0; seg < segments; ++seg) {
            uint32_t tl = sideTopStart + seg;
            uint32_t tr = tl + 1;
            uint32_t bl = sideBottomStart + seg;
            uint32_t br = bl + 1;

            indices.push_back(tl);
            indices.push_back(tr);
            indices.push_back(bl);

            indices.push_back(tr);
            indices.push_back(br);
            indices.push_back(bl);
        }

        // Bottom cap (fan from center, CCW from below → normal down)
        uint32_t bottomCenter = static_cast<uint32_t>(sideBottomStart + segments + 1);
        uint32_t bottomEdgeStart = bottomCenter + 1;
        for (int seg = 0; seg < segments; ++seg) {
            indices.push_back(bottomCenter);
            indices.push_back(static_cast<uint32_t>(bottomEdgeStart + seg));
            indices.push_back(static_cast<uint32_t>(bottomEdgeStart + seg + 1));
        }

        return indices;
    }

    /// @brief Create skybox cube vertices - 8 corner vertices of a unit cube
    /// Only positions matter (used as world direction in skybox shader)
    static std::vector<Vertex> CreateSkyboxCubeVertices()
    {
        glm::vec3 n(0.0f);  // normals unused for skybox
        glm::vec4 t(0.0f);  // tangents unused for skybox
        glm::vec3 c(1.0f);  // white vertex color
        glm::vec2 uv(0.0f); // UVs unused for skybox

        return {
            Vertex::CreateFull({-1.0f, -1.0f, -1.0f}, n, t, c, uv), // 0
            Vertex::CreateFull({1.0f, -1.0f, -1.0f}, n, t, c, uv),  // 1
            Vertex::CreateFull({1.0f, 1.0f, -1.0f}, n, t, c, uv),   // 2
            Vertex::CreateFull({-1.0f, 1.0f, -1.0f}, n, t, c, uv),  // 3
            Vertex::CreateFull({-1.0f, -1.0f, 1.0f}, n, t, c, uv),  // 4
            Vertex::CreateFull({1.0f, -1.0f, 1.0f}, n, t, c, uv),   // 5
            Vertex::CreateFull({1.0f, 1.0f, 1.0f}, n, t, c, uv),    // 6
            Vertex::CreateFull({-1.0f, 1.0f, 1.0f}, n, t, c, uv),   // 7
        };
    }

    /// @brief Create skybox cube indices - 12 triangles (36 indices), wound CW when viewed from outside
    /// (rendered with front-face culling so we see inside faces)
    static std::vector<uint32_t> CreateSkyboxCubeIndices()
    {
        return {
            // Front  (z = -1)
            0,
            1,
            2,
            2,
            3,
            0,
            // Back   (z = +1)
            5,
            4,
            7,
            7,
            6,
            5,
            // Left   (x = -1)
            4,
            0,
            3,
            3,
            7,
            4,
            // Right  (x = +1)
            1,
            5,
            6,
            6,
            2,
            1,
            // Top    (y = +1)
            3,
            2,
            6,
            6,
            7,
            3,
            // Bottom (y = -1)
            4,
            5,
            1,
            1,
            0,
            4,
        };
    }
};

} // namespace infernux
