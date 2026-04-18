#pragma once

#include <cmath>
#include <function/renderer/InxRenderStruct.h>
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

    /// @brief Get sphere vertices (UV sphere)
    static const std::vector<Vertex> &GetSphereVertices()
    {
        static std::vector<Vertex> vertices = CreateSphereVertices(64, 32);
        return vertices;
    }

    /// @brief Get sphere indices
    static const std::vector<uint32_t> &GetSphereIndices()
    {
        static std::vector<uint32_t> indices = CreateSphereIndices(64, 32);
        return indices;
    }

    /// @brief Get capsule vertices
    static const std::vector<Vertex> &GetCapsuleVertices()
    {
        static std::vector<Vertex> vertices = CreateCapsuleVertices(16, 8, 0.5f, 1.0f);
        return vertices;
    }

    /// @brief Get capsule indices
    static const std::vector<uint32_t> &GetCapsuleIndices()
    {
        static std::vector<uint32_t> indices = CreateCapsuleIndices(16, 8);
        return indices;
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
    /// @brief Shared ring-based index generation for sphere, capsule, cylinder side, etc.
    static void AppendRingIndices(std::vector<uint32_t> &indices, int segments, int rings, uint32_t baseVertex = 0)
    {
        int vertsPerRing = segments + 1;
        for (int ring = 0; ring < rings; ++ring) {
            for (int seg = 0; seg < segments; ++seg) {
                uint32_t current = baseVertex + static_cast<uint32_t>(ring * vertsPerRing + seg);
                uint32_t next = current + 1;
                uint32_t below = static_cast<uint32_t>(current + vertsPerRing);
                uint32_t belowNext = below + 1;

                indices.push_back(current);
                indices.push_back(next);
                indices.push_back(below);

                indices.push_back(next);
                indices.push_back(belowNext);
                indices.push_back(below);
            }
        }
    }

    /// @brief Shared hemisphere vertex generation for capsule top/bottom.
    /// @param ySign +1 for top hemisphere, -1 for bottom hemisphere
    /// @param yOffset  vertical center offset (±cylinderHeight/2)
    /// @param vBase    UV v-coordinate start (0.0 for top, 0.75 for bottom)
    /// @param vScale   UV v-coordinate range (0.25 for both hemispheres)
    static void GenerateHemisphereVertices(std::vector<Vertex> &vertices, int segments, int hemisphereRings, float radius,
                                           float ySign, float yOffset, float vBase, float vScale)
    {
        const float PI = 3.14159265358979323846f;
        for (int ring = 0; ring <= hemisphereRings; ++ring) {
            float phi = (ySign > 0)
                            ? (PI / 2.0f) * ring / hemisphereRings
                            : (PI / 2.0f) + (PI / 2.0f) * ring / hemisphereRings;
            float y = std::cos(phi) * radius + yOffset;
            float ringRadius = std::sin(phi) * radius;

            for (int seg = 0; seg <= segments; ++seg) {
                float theta = 2.0f * PI * seg / segments;
                float x = std::cos(theta) * ringRadius;
                float z = std::sin(theta) * ringRadius;

                glm::vec3 localPos(std::cos(theta) * std::sin(phi), std::cos(phi), std::sin(theta) * std::sin(phi));
                glm::vec3 normal = glm::normalize(localPos);
                glm::vec3 tangent = glm::normalize(glm::vec3(-std::sin(theta), 0.0f, std::cos(theta)));

                float u = static_cast<float>(seg) / segments;
                float v = vBase + vScale * ring / hemisphereRings;

                vertices.push_back(
                    Vertex::CreateFull({x, y, z}, normal, glm::vec4(tangent, 1.0f), {1.0f, 1.0f, 1.0f}, {u, v}));
            }
        }
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
    // Sphere generation (UV sphere)
    // ========================================================================
    static std::vector<Vertex> CreateSphereVertices(int segments, int rings)
    {
        std::vector<Vertex> vertices;
        const float PI = 3.14159265358979323846f;
        const float radius = 0.5f;

        for (int ring = 0; ring <= rings; ++ring) {
            float phi = PI * ring / rings;
            float y = std::cos(phi) * radius;
            float ringRadius = std::sin(phi) * radius;

            for (int seg = 0; seg <= segments; ++seg) {
                float theta = 2.0f * PI * seg / segments;
                float x = std::cos(theta) * ringRadius;
                float z = std::sin(theta) * ringRadius;

                // Normal is the normalized position for a unit sphere
                glm::vec3 normal = glm::normalize(glm::vec3(x, y, z));

                // Tangent follows the longitude direction. Near the poles fall back to
                // an orthogonal tangent to avoid unstable normal-map seams.
                glm::vec3 tangent = glm::vec3(CalculateTangent(normal));
                if (ringRadius > 1e-5f) {
                    tangent = glm::normalize(glm::vec3(-std::sin(theta), 0.0f, std::cos(theta)));
                }

                // Use white vertex color - material baseColor controls actual color
                glm::vec3 color(1.0f, 1.0f, 1.0f);

                float u = static_cast<float>(seg) / segments;
                float v = static_cast<float>(ring) / rings;

                vertices.push_back(Vertex::CreateFull({x, y, z}, normal, glm::vec4(tangent, 1.0f), color, {u, v}));
            }
        }
        return vertices;
    }

    static std::vector<uint32_t> CreateSphereIndices(int segments, int rings)
    {
        std::vector<uint32_t> indices;
        AppendRingIndices(indices, segments, rings);
        return indices;
    }

    // ========================================================================
    // Capsule generation (cylinder + hemispheres)
    // ========================================================================
    static std::vector<Vertex> CreateCapsuleVertices(int segments, int hemisphereRings, float radius, float height)
    {
        std::vector<Vertex> vertices;
        const float PI = 3.14159265358979323846f;
        float cylinderHeight = height - 2.0f * radius;
        if (cylinderHeight < 0)
            cylinderHeight = 0;

        // Top hemisphere
        GenerateHemisphereVertices(vertices, segments, hemisphereRings, radius, +1.0f, cylinderHeight / 2.0f, 0.0f,
                                   0.25f);

        // Cylinder body
        for (int ring = 0; ring <= 1; ++ring) {
            float y = (ring == 0) ? cylinderHeight / 2.0f : -cylinderHeight / 2.0f;
            for (int seg = 0; seg <= segments; ++seg) {
                float theta = 2.0f * PI * seg / segments;
                float x = std::cos(theta) * radius;
                float z = std::sin(theta) * radius;

                glm::vec3 normal = glm::normalize(glm::vec3(x, 0.0f, z));
                glm::vec3 tangent(0.0f, 1.0f, 0.0f);

                float u = static_cast<float>(seg) / segments;
                float v = 0.25f + 0.5f * ring;

                vertices.push_back(
                    Vertex::CreateFull({x, y, z}, normal, glm::vec4(tangent, 1.0f), {1.0f, 1.0f, 1.0f}, {u, v}));
            }
        }

        // Bottom hemisphere
        GenerateHemisphereVertices(vertices, segments, hemisphereRings, radius, -1.0f, -cylinderHeight / 2.0f, 0.75f,
                                   0.25f);

        return vertices;
    }

    static std::vector<uint32_t> CreateCapsuleIndices(int segments, int hemisphereRings)
    {
        std::vector<uint32_t> indices;
        int totalRings = hemisphereRings + 2 + hemisphereRings; // top + cylinder + bottom
        AppendRingIndices(indices, segments, totalRings);
        return indices;
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
