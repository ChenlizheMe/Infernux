// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#include "WarpCollisionModel.h"
#include "PhysicsRayIndex.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace infernux::physics::warp
{
static_assert(sizeof(WarpLocalBound) == 64);
static_assert(sizeof(WarpCollisionShape) == 32);
static_assert(sizeof(WarpContactMaterial) == 96);

namespace
{
ModelWords FloatWords(const ModelVector &value)
{
    ModelWords words;
    std::memcpy(words.data(), value.data(), sizeof(words));
    return words;
}

void AppendScalars(std::vector<ModelWords> &records, const std::vector<uint32_t> &scalars)
{
    const auto first = records.size();
    records.resize(first + (scalars.size() + 3) / 4);
    for (size_t index = 0; index < scalars.size(); ++index)
        records[first + index / 4][index % 4] = scalars[index];
}

struct PackedHull
{
    ModelWords shape;
    ModelWords topology;
    ModelVector center{};
    ModelVector halfExtent{};
    float radius = 0.0f;
};

struct PackedMesh
{
    uint32_t descriptor;
    ModelVector center;
    ModelVector halfExtent;
    float radius;
};

struct ConvexData
{
    explicit ConvexData(size_t geometries) : topology(geometries, {NoModelIndex, NoModelIndex, 0, 0})
    {
    }

    const PackedHull &AddHull(const WarpConvexHull &hull)
    {
        const auto cached = hulls.find(&hull);
        if (cached != hulls.end())
            return cached->second;
        const auto count = static_cast<uint32_t>(hull.vertices.size());
        const auto faces = static_cast<uint32_t>(hull.faces.size());
        if (count < 4 || faces < 4)
            throw std::invalid_argument("Warp convex hull has no closed surface");

        PackedHull packed;
        packed.shape = {Hull, static_cast<uint32_t>(vertices.size()), count, NoModelIndex};
        auto minimum = hull.vertices.front(), maximum = minimum;
        for (const auto &vertex : hull.vertices) {
            vertices.push_back(FloatWords(vertex));
            packed.radius = std::max(packed.radius, std::hypot(vertex[0], vertex[1], vertex[2]));
            for (uint32_t axis = 0; axis < 3; ++axis) {
                minimum[axis] = std::min(minimum[axis], vertex[axis]);
                maximum[axis] = std::max(maximum[axis], vertex[axis]);
            }
        }
        for (uint32_t axis = 0; axis < 3; ++axis) {
            packed.center[axis] = (maximum[axis] + minimum[axis]) * 0.5f;
            packed.halfExtent[axis] = (maximum[axis] - minimum[axis]) * 0.5f;
        }

        const auto faceStart = static_cast<uint32_t>(topology.size());
        const auto incidenceStart = faceStart + faces;
        packed.topology = {faceStart, incidenceStart, 0, 0};
        std::vector<std::vector<uint32_t>> incident(count);
        std::vector<std::vector<uint32_t>> neighbors(count);
        for (uint32_t face = 0; face < faces; ++face) {
            const auto &source = hull.faces[face];
            const auto size = static_cast<uint32_t>(source.vertices.size());
            if (size < 3 ||
                std::any_of(
                    source.vertices.begin(), source.vertices.end(), [count](uint32_t index) { return index >= count; }))
                throw std::invalid_argument("Invalid Warp convex hull face loop");
            polygonVertices = std::max(polygonVertices, size);
            topology.push_back({static_cast<uint32_t>(links.size()), size, static_cast<uint32_t>(vertices.size()), 0});
            vertices.push_back(FloatWords(source.normal));
            links.insert(links.end(), source.vertices.begin(), source.vertices.end());
            for (uint32_t i = 0; i < size; ++i) {
                const auto a = source.vertices[i];
                const auto b = source.vertices[(i + 1) % size];
                incident[a].push_back(face);
                neighbors[a].push_back(b);
                neighbors[b].push_back(a);
            }
        }
        for (const auto &adjacent : incident) {
            topology.push_back({static_cast<uint32_t>(links.size()), static_cast<uint32_t>(adjacent.size()), 0, 0});
            links.insert(links.end(), adjacent.begin(), adjacent.end());
        }
        if (count >= 10) {
            const auto graph = static_cast<uint32_t>(adjacency.size());
            packed.shape[3] = graph;
            adjacency.resize(adjacency.size() + 2 + 2 * uint64_t{count});
            adjacency[graph] = count;
            adjacency[graph + 1] = faces;
            const auto neighborStart = static_cast<uint32_t>(adjacency.size());
            for (uint32_t vertex = 0; vertex < count; ++vertex) {
                auto &adjacent = neighbors[vertex];
                std::sort(adjacent.begin(), adjacent.end());
                const auto unique = std::unique(adjacent.begin(), adjacent.end());
                adjacency[graph + 2 + vertex] = static_cast<uint32_t>(adjacency.size()) - neighborStart;
                adjacency[graph + 2 + count + vertex] = vertex;
                adjacency.insert(adjacency.end(), adjacent.begin(), unique);
                adjacency.push_back(NoModelIndex);
            }
        }
        return hulls.emplace(&hull, std::move(packed)).first->second;
    }

    uint32_t AddHeightfield(const WarpHeightfield &field, const ModelVector &size)
    {
        if (field.rows < 2 || field.columns < 2 || field.samples.size() != uint64_t{field.rows} * field.columns ||
            size[0] <= 0 || size[1] <= 0 || size[2] < 0 || size[3] < 0)
            throw std::invalid_argument("Invalid Warp heightfield grid");
        const auto [sample, inserted] = sampleOffsets.try_emplace(&field, static_cast<uint32_t>(samples.size()));
        if (inserted) {
            for (const float height : field.samples) {
                uint32_t word;
                std::memcpy(&word, &height, sizeof(word));
                samples.push_back(word);
            }
        }
        const auto row = static_cast<uint32_t>(heightfields.size() / 2);
        heightfields.push_back({field.rows, field.columns, sample->second, 0});
        heightfields.push_back(FloatWords(size));
        return row;
    }

    const PackedMesh &AddTriangleMesh(const WarpTriangleMesh &mesh)
    {
        const auto cached = triangleMeshes.find(&mesh);
        if (cached != triangleMeshes.end())
            return cached->second;
        if (mesh.faces.empty() || mesh.faces.size() != mesh.features.size())
            throw std::invalid_argument("Warp triangle mesh requires face features");
        const auto base = static_cast<uint32_t>(vertices.size());
        float radius = 0.0f;
        for (const auto &vertex : mesh.vertices) {
            vertices.push_back(FloatWords({vertex.x, vertex.y, vertex.z, 0.0f}));
            radius = std::max(radius, glm::length(vertex));
        }
        const auto descriptor = AppendPhysicsRayMesh(meshRecords, mesh.vertices, mesh.faces, base);
        const auto root = meshRecords[descriptor][0];
        const auto &lo = meshRecords[root];
        const auto &hi = meshRecords[root + 1];
        ModelVector lower, upper;
        std::memcpy(lower.data(), lo.data(), sizeof(lower));
        std::memcpy(upper.data(), hi.data(), sizeof(upper));
        PackedMesh packed{descriptor, {}, {}, radius};
        for (uint32_t axis = 0; axis < 3; ++axis) {
            packed.center[axis] = 0.5f * (upper[axis] + lower[axis]);
            packed.halfExtent[axis] = 0.5f * (upper[axis] - lower[axis]);
        }
        meshRecords[descriptor][3] = static_cast<uint32_t>(meshRecords.size());
        std::vector<uint32_t> metadata;
        metadata.reserve(mesh.features.size() * 3);
        for (const auto &feature : mesh.features)
            metadata.insert(metadata.end(), feature.begin(), feature.end());
        AppendScalars(meshRecords, metadata);
        return triangleMeshes.emplace(&mesh, packed).first->second;
    }

    std::vector<ModelWords> Finish() const
    {
        // The reference collision loop uses a bounded GJK and EPA budget.
        constexpr uint32_t iterations = 50;
        std::vector<ModelWords> result(5);
        result[0] = {0, iterations + 5, 6 + 12 * iterations, 128};
        result[1] = {FloatWords({1.0e-6f, 0, 0, 0})[0], iterations, iterations, polygonVertices};
        ModelWords offsets;
        offsets[0] = static_cast<uint32_t>(result.size());
        result.insert(result.end(), vertices.begin(), vertices.end());
        offsets[1] = static_cast<uint32_t>(result.size());
        AppendScalars(result, adjacency);
        offsets[2] = static_cast<uint32_t>(result.size());
        result.insert(result.end(), topology.begin(), topology.end());
        offsets[3] = static_cast<uint32_t>(result.size());
        AppendScalars(result, links);
        result[2] = offsets;
        result[3] = {static_cast<uint32_t>(result.size()), static_cast<uint32_t>(result.size() + heightfields.size()),
                     heightfields.empty() ? 0u : 50u, 0};
        result.insert(result.end(), heightfields.begin(), heightfields.end());
        AppendScalars(result, samples);
        result[4][0] = static_cast<uint32_t>(result.size());
        result.insert(result.end(), meshRecords.begin(), meshRecords.end());
        return result;
    }

    uint32_t polygonVertices = 4;
    std::vector<ModelWords> vertices, topology, heightfields, meshRecords;
    std::vector<uint32_t> adjacency, links, samples;
    std::unordered_map<const WarpConvexHull *, PackedHull> hulls;
    std::unordered_map<const WarpHeightfield *, uint32_t> sampleOffsets;
    std::unordered_map<const WarpTriangleMesh *, PackedMesh> triangleMeshes;
};
} // namespace

WarpCollisionModel BuildWarpCollisionModel(const WarpModel &model, const std::vector<WarpCollisionGeometry> &geometries,
                                           std::vector<WarpCollisionPair> pairs, bool filterConnectedBodies)
{
    const auto count = model.dimensions.layout[0];
    if (geometries.size() != count)
        throw std::invalid_argument("Warp collision geometry must follow the compiled model order");

    std::vector<uint32_t> welded(model.bodies.size(), 0);
    for (uint32_t body = 1; body < model.bodies.size(); ++body) {
        const auto &row = model.bodies[body];
        welded[body] = row.dofs[1] != 0 || row.hierarchy[3] != NoModelIndex ? body : welded[row.hierarchy[0]];
    }

    WarpCollisionModel result;
    result.filterConnectedBodies = filterConnectedBodies;
    result.bounds.reserve(count);
    result.shapes.reserve(count);
    result.materials.reserve(count + pairs.size());
    result.rayGeometry.reserve(count);
    result.pairRules.resize(uint64_t{count} + 1);
    result.pairRules.reserve(result.pairRules.size() + pairs.size());
    ConvexData convex(count);
    uint32_t boxCount = 0, cylinderCount = 0;
    bool hasCapsule = false;
    for (uint32_t geom = 0; geom < count; ++geom) {
        const auto &source = geometries[geom];
        const auto [body, type, collisionType, collisionAffinity] = model.geometries[geom].indices;
        const auto weld = welded[body];
        const auto weldParent = weld == 0 ? 0 : welded[model.bodies[weld].hierarchy[0]];
        const auto &material = source.material;
        result.bounds.push_back({source.center,
                                 source.halfExtent,
                                 {source.radius, material.parameters[0] + material.parameters[1], 0, 0},
                                 {type, body, collisionType, collisionAffinity}});
        result.shapes.push_back({source.size, {type, 0, 0, 0}});
        auto &bounds = result.bounds.back();
        auto &shape = result.shapes.back();
        if (type == Hull) {
            if (!source.hull)
                throw std::invalid_argument("Warp convex mesh requires its cooked hull");
            const auto &hull = convex.AddHull(*source.hull);
            shape.type = hull.shape;
            convex.topology[geom] = hull.topology;
            bounds.center = hull.center;
            bounds.halfExtent = hull.halfExtent;
            bounds.radii[0] = hull.radius;
        } else if (type == Heightfield) {
            if (!source.heightfield)
                throw std::invalid_argument("Warp heightfield requires its sample grid");
            shape.type[1] = convex.AddHeightfield(*source.heightfield, source.size);
            const auto [x, y, height, depth] = source.size;
            bounds.center = {0, 0, (height - depth) * 0.5f, 0};
            bounds.halfExtent = {x, y, (height + depth) * 0.5f, 0};
            bounds.radii[0] = std::hypot(x, y, std::max(height, depth));
            result.heightfields = true;
        } else if (type == TriangleMesh) {
            if (!source.mesh || model.bodies[weld].dofs[1] != 0)
                throw std::invalid_argument("Warp triangle mesh requires cooked geometry on a non-dynamic body");
            const auto &mesh = convex.AddTriangleMesh(*source.mesh);
            shape.type = {TriangleMesh, mesh.descriptor, source.meshPath, source.meshPathBits};
            bounds.center = mesh.center;
            bounds.halfExtent = mesh.halfExtent;
            bounds.radii[0] = mesh.radius;
            result.triangleMeshes = true;
        }
        result.convexPairs |= type == Hull || type == Ellipsoid;
        boxCount += uint32_t(type == Box);
        cylinderCount += uint32_t(type == Cylinder);
        hasCapsule |= type == Capsule;
        result.materials.push_back(material);
        const uint32_t flags = uint32_t(model.bodies[weld].dofs[1] != 0) | (uint32_t(source.trigger) << 1);
        result.pairRules[geom] = {weld, weldParent, flags, 0};
        const bool isStatic = (flags & 1) == 0 && model.bodies[weld].hierarchy[3] == NoModelIndex;
        const uint32_t rayFlags = uint32_t(isStatic) | (uint32_t(source.trigger) << 1);
        result.rayGeometry.push_back({body, collisionType, rayFlags, source.subShape});
    }
    result.convexPairs |= boxCount > 1 || cylinderCount > 1 || (cylinderCount != 0 && (boxCount != 0 || hasCapsule));
    if (result.convexPairs || result.heightfields || result.triangleMeshes) {
        if (cylinderCount != 0)
            convex.polygonVertices = std::max(convex.polygonVertices, 16u);
        result.convexData = convex.Finish();
    }

    for (auto &pair : pairs) {
        if (pair.firstGeometry > pair.secondGeometry)
            std::swap(pair.firstGeometry, pair.secondGeometry);
        if (pair.firstGeometry == pair.secondGeometry || pair.secondGeometry >= count ||
            (pair.suppressContact && pair.material))
            throw std::invalid_argument("Invalid Warp collision pair override");
    }
    const auto key = [](const auto &pair) { return std::tie(pair.firstGeometry, pair.secondGeometry); };
    std::sort(pairs.begin(), pairs.end(), [&](const auto &a, const auto &b) { return key(a) < key(b); });

    auto next = pairs.begin();
    for (uint32_t geom = 0; geom < count; ++geom) {
        result.pairRules[geom][3] = static_cast<uint32_t>(result.pairRules.size());
        while (next != pairs.end() && next->firstGeometry == geom) {
            const auto &pair = *next++;
            if (next != pairs.end() && key(pair) == key(*next))
                throw std::invalid_argument("Duplicate Warp collision pair override");
            int32_t policy = pair.suppressContact ? -2 : -1;
            float margin = 0.0f;
            if (pair.material) {
                policy = static_cast<int32_t>(result.materials.size() - count);
                result.materials.push_back(*pair.material);
                margin = pair.material->parameters[0] + pair.material->parameters[1];
                for (const auto id : {pair.firstGeometry, pair.secondGeometry})
                    result.bounds[id].radii[2] = std::max(result.bounds[id].radii[2], margin);
            }
            uint32_t marginBits;
            std::memcpy(&marginBits, &margin, sizeof(marginBits));
            result.pairRules.push_back(
                {pair.secondGeometry, static_cast<uint32_t>(policy), static_cast<uint32_t>(pair.sensor), marginBits});
        }
    }
    result.pairRules[count][3] = static_cast<uint32_t>(result.pairRules.size());
    return result;
}

} // namespace infernux::physics::warp
