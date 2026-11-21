// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "InxPhyGpuModel.h"

#include <glm/vec3.hpp>
#include <memory>
#include <optional>

namespace infernux::physics::inxphygpu
{

enum InxPhyGpuGeometryType : uint32_t
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

struct InxPhyGpuContactMaterial
{
    ModelVector parameters{0.0f, 0.0f, 1.0f, 0.0f}; // Margin, gap, solver mix, adhesion.
    ModelVector reference{0.02f, 1.0f, 0.0f, 0.0f};
    ModelVector impedance{0.9f, 0.95f, 0.001f, 0.5f};
    ModelVector friction{1.0f, 0.005f, 0.0001f, 0.0f};
    ModelVector extra{2.0f, 0.0001f, 0.0f, 0.0f}; // Impedance power, friction 5, restitution, unused.
    // Contact dimension, reference priority, scene friction and bounce combines (enum + 1).
    std::array<int32_t, 4> dimensions{3, 0, 0, 0};
};

struct InxPhyGpuHullFace
{
    std::vector<uint32_t> vertices; // Convex loop in outward winding order.
    ModelVector normal{};
};

struct InxPhyGpuConvexHull
{
    std::vector<ModelVector> vertices;
    std::vector<InxPhyGpuHullFace> faces;
};

struct InxPhyGpuHeightfield
{
    uint32_t rows = 0;
    uint32_t columns = 0;
    std::vector<float> samples; // Row-major normalized elevations.
};

struct InxPhyGpuTriangleMesh
{
    std::vector<glm::vec3> vertices;
    std::vector<std::array<uint32_t, 3>> faces;
    std::vector<std::array<uint32_t, 3>> features; // Native triangle ID, active edges and asset face index.
};

struct InxPhyGpuCollisionGeometry
{
    ModelVector size{};
    ModelVector center{};
    ModelVector halfExtent{};
    float radius = 0.0f;
    InxPhyGpuContactMaterial material;
    std::shared_ptr<const InxPhyGpuConvexHull> hull;
    std::shared_ptr<const InxPhyGpuHeightfield> heightfield;
    bool trigger = false;
    uint32_t subShape = 0;
    std::shared_ptr<const InxPhyGpuTriangleMesh> mesh;
    uint32_t meshPath = 0;
    uint32_t meshPathBits = 0;
};

struct InxPhyGpuCollisionPair
{
    uint32_t firstGeometry;
    uint32_t secondGeometry;
    std::optional<InxPhyGpuContactMaterial> material; // An explicit physical pair overrides the default filter.
    int32_t sensor = -1;
    bool suppressContact = false;
};

struct InxPhyGpuLocalBound
{
    ModelVector center;
    ModelVector halfExtent;
    ModelVector radii;
    ModelWords geometry;
};

struct InxPhyGpuCollisionShape
{
    ModelVector size;
    ModelWords type;
};

struct InxPhyGpuCollisionModel
{
    std::vector<InxPhyGpuLocalBound> bounds;
    std::vector<InxPhyGpuCollisionShape> shapes;
    std::vector<InxPhyGpuContactMaterial> materials;
    std::vector<ModelWords> pairRules;
    std::vector<ModelWords> rayGeometry; // Body row, layer mask, flags, subshape.
    std::vector<ModelWords> convexData;
    bool filterConnectedBodies = true;
    bool convexPairs = false;
    bool heightfields = false;
    bool triangleMeshes = false;
};

/// Build rows in the same geometry order as model.geometries. Pair overrides can arrive in any order.
[[nodiscard]] InxPhyGpuCollisionModel
BuildInxPhyGpuCollisionModel(const InxPhyGpuModel &model, const std::vector<InxPhyGpuCollisionGeometry> &geometries,
                             std::vector<InxPhyGpuCollisionPair> pairs, bool filterConnectedBodies = true);

} // namespace infernux::physics::inxphygpu
