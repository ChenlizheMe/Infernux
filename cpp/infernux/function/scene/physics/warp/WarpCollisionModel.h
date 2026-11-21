// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "WarpModel.h"

#include <glm/vec3.hpp>
#include <memory>
#include <optional>

namespace infernux::physics::warp
{

enum WarpGeometryType : uint32_t
{
    Plane = 0,
    Heightfield = 1,
    Sphere = 2,
    Capsule = 3,
    Ellipsoid = 4,
    Cylinder = 5,
    Box = 6,
    Hull = 7,
    TriangleMesh = 8,
};

struct WarpContactMaterial
{
    ModelVector parameters{0.0f, 0.0f, 1.0f, 0.0f}; // Margin, gap, solver mix, adhesion.
    ModelVector reference{0.02f, 1.0f, 0.0f, 0.0f};
    ModelVector impedance{0.9f, 0.95f, 0.001f, 0.5f};
    ModelVector friction{1.0f, 0.005f, 0.0001f, 0.0f};
    ModelVector extra{2.0f, 0.0001f, 0.0f, 0.0f}; // Impedance power, friction 5, restitution, unused.
    // Contact dimension, reference priority, scene friction and bounce combines (enum + 1).
    std::array<int32_t, 4> dimensions{3, 0, 0, 0};
};

struct WarpHullFace
{
    std::vector<uint32_t> vertices; // Convex loop in outward winding order.
    ModelVector normal{};
};

struct WarpConvexHull
{
    std::vector<ModelVector> vertices;
    std::vector<WarpHullFace> faces;
};

struct WarpHeightfield
{
    uint32_t rows = 0;
    uint32_t columns = 0;
    std::vector<float> samples; // Row-major normalized elevations.
};

struct WarpTriangleMesh
{
    std::vector<glm::vec3> vertices;
    std::vector<std::array<uint32_t, 3>> faces;
    std::vector<std::array<uint32_t, 3>> features; // Native triangle ID, active edges and asset face index.
};

struct WarpCollisionGeometry
{
    ModelVector size{};
    ModelVector center{};
    ModelVector halfExtent{};
    float radius = 0.0f;
    WarpContactMaterial material;
    std::shared_ptr<const WarpConvexHull> hull;
    std::shared_ptr<const WarpHeightfield> heightfield;
    bool trigger = false;
    uint32_t subShape = 0;
    std::shared_ptr<const WarpTriangleMesh> mesh;
    uint32_t meshPath = 0;
    uint32_t meshPathBits = 0;
};

struct WarpCollisionPair
{
    uint32_t firstGeometry;
    uint32_t secondGeometry;
    std::optional<WarpContactMaterial> material; // An explicit physical pair overrides the default filter.
    int32_t sensor = -1;
    bool suppressContact = false;
};

struct WarpLocalBound
{
    ModelVector center;
    ModelVector halfExtent;
    ModelVector radii;
    ModelWords geometry;
};

struct WarpCollisionShape
{
    ModelVector size;
    ModelWords type;
};

struct WarpCollisionModel
{
    std::vector<WarpLocalBound> bounds;
    std::vector<WarpCollisionShape> shapes;
    std::vector<WarpContactMaterial> materials;
    std::vector<ModelWords> pairRules;
    std::vector<ModelWords> rayGeometry; // Body row, layer mask, flags, subshape.
    std::vector<ModelWords> convexData;
    bool filterConnectedBodies = true;
    bool convexPairs = false;
    bool heightfields = false;
    bool triangleMeshes = false;
};

/// Build rows in the same geometry order as model.geometries. Pair overrides can arrive in any order.
[[nodiscard]] WarpCollisionModel BuildWarpCollisionModel(const WarpModel &model,
                                                         const std::vector<WarpCollisionGeometry> &geometries,
                                                         std::vector<WarpCollisionPair> pairs,
                                                         bool filterConnectedBodies = true);

} // namespace infernux::physics::warp
