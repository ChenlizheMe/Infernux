#include "RayGeometry.h"

#include <function/renderer/InxRenderStruct.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

namespace infernux::raytracing
{
namespace
{
using spatial::Bounds;
using spatial::BuildBvh;
using spatial::BvhNode;
using spatial::kBvhLeafBit;

bool IsFinite(const glm::vec3 &value) noexcept
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

uint32_t PageOffset(size_t size, size_t count)
{
    // The leaf tag occupies the upper bit of a node word.
    if (count > kBvhLeafBit - size)
        throw std::length_error("Ray geometry page exceeds its index range");
    return static_cast<uint32_t>(size);
}

Bounds TransformBounds(const BvhNode &root, const glm::mat4 &matrix)
{
    // Accumulate interval endpoints in double precision, then round outwards for the GPU slab test.
    Bounds result;
    for (unsigned row = 0; row < 3; ++row) {
        double low = matrix[3][row], high = low;
        for (unsigned axis = 0; axis < 3; ++axis) {
            const double coefficient = matrix[axis][row];
            const double a = coefficient * root.minimum[axis];
            const double b = coefficient * root.maximum[axis];
            low += std::min(a, b);
            high += std::max(a, b);
        }
        result.minimum[row] = std::nextafter(static_cast<float>(low), -std::numeric_limits<float>::infinity());
        result.maximum[row] = std::nextafter(static_cast<float>(high), std::numeric_limits<float>::infinity());
    }
    return result;
}

RayInstance EncodeInstance(const RayMeshInstance &instance, uint32_t root)
{
    const glm::dmat4 model(instance.objectToWorld);
    if (model[0].w != 0.0 || model[1].w != 0.0 || model[2].w != 0.0 || model[3].w != 1.0)
        throw std::invalid_argument("Ray geometry instance requires an affine transform");
    const glm::dmat3 inverseLinear = glm::inverse(glm::dmat3(model));
    const glm::dvec3 inverseTranslation = -(inverseLinear * glm::dvec3(model[3]));
    const auto row = [&](unsigned i) {
        std::array<float, 4> components{
            static_cast<float>(inverseLinear[0][i]), static_cast<float>(inverseLinear[1][i]),
            static_cast<float>(inverseLinear[2][i]), static_cast<float>(inverseTranslation[i])};
        if (!std::all_of(components.begin(), components.end(), [](float v) { return std::isfinite(v); }))
            throw std::invalid_argument("Ray geometry instance requires a finite inverse transform");
        return components;
    };
    return {row(0), row(1), row(2), root, instance.surfaceId, instance.layerMask};
}

void AppendNodes(std::vector<BvhNode> &destination, const std::vector<BvhNode> &nodes, uint32_t root,
                 uint32_t leafOffset)
{
    for (auto node : nodes) {
        if ((node.second & kBvhLeafBit) != 0) {
            node.first += leafOffset;
        } else {
            node.first += root;
            node.second += root;
        }
        destination.push_back(node);
    }
}

uint32_t AppendMesh(RayGeometryPage &page, const RayMesh &mesh)
{
    const auto root = PageOffset(page.nodes.size(), mesh.nodes.size());
    const auto positionOffset = PageOffset(page.positions.size(), mesh.positions.size());
    const auto triangleOffset = PageOffset(page.triangles.size(), mesh.triangles.size());
    AppendNodes(page.nodes, mesh.nodes, root, triangleOffset);
    page.positions.insert(page.positions.end(), mesh.positions.begin(), mesh.positions.end());
    for (const auto &triangle : mesh.triangles) {
        page.triangles.push_back({triangle.a + positionOffset, triangle.b + positionOffset, triangle.c + positionOffset,
                                  triangle.primitive});
    }
    return root;
}
} // namespace

RayMesh BuildRayMesh(const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices, uint32_t indexStart,
                     uint32_t indexCount, int32_t vertexStart)
{
    if (indexCount % 3 != 0 || indexStart > indices.size() || indexCount > indices.size() - indexStart)
        throw std::invalid_argument("Ray mesh requires a complete indexed triangle slice");

    RayMesh mesh;
    const auto triangleCount = indexCount / 3;
    std::vector<Bounds> bounds;
    std::vector<RayTriangle> triangles;
    std::unordered_map<uint32_t, uint32_t> positionIndices;
    bounds.reserve(triangleCount);
    triangles.reserve(triangleCount);
    for (uint32_t primitive = 0; primitive < triangleCount; ++primitive) {
        const size_t first = static_cast<size_t>(indexStart) + primitive * 3;
        uint32_t corners[3];
        Bounds triangleBounds;
        for (unsigned corner = 0; corner < 3; ++corner) {
            const int64_t vertex = static_cast<int64_t>(indices[first + corner]) + vertexStart;
            if (vertex < 0 || static_cast<uint64_t>(vertex) >= vertices.size() || vertex > UINT32_MAX)
                throw std::invalid_argument("Ray mesh index falls outside its vertex stream");
            const glm::vec3 &position = vertices[static_cast<size_t>(vertex)].pos;
            const auto [entry, inserted] = positionIndices.try_emplace(static_cast<uint32_t>(vertex),
                                                                       static_cast<uint32_t>(mesh.positions.size()));
            if (inserted) {
                if (!IsFinite(position))
                    throw std::invalid_argument("Ray mesh requires finite vertex positions");
                mesh.positions.push_back({position.x, position.y, position.z});
            }
            corners[corner] = entry->second;
            triangleBounds.Include({position, position});
        }
        bounds.push_back(triangleBounds);
        triangles.push_back({corners[0], corners[1], corners[2], primitive});
    }

    auto hierarchy = BuildBvh(bounds);
    mesh.nodes = std::move(hierarchy.nodes);
    mesh.triangles.reserve(triangleCount);
    for (uint32_t index : hierarchy.primitiveOrder)
        mesh.triangles.push_back(triangles[index]);
    return mesh;
}

RayGeometryPage BuildRayGeometryPage(const std::vector<RayMeshInstance> &instances)
{
    RayGeometryPage page;
    std::unordered_map<const RayMesh *, uint32_t> roots;
    std::vector<Bounds> bounds;
    std::vector<RayInstance> packed;
    bounds.reserve(instances.size());
    packed.reserve(instances.size());
    for (const auto &instance : instances) {
        if (instance.mesh.nodes.empty())
            continue;
        const auto [entry, inserted] = roots.try_emplace(&instance.mesh, 0);
        if (inserted)
            entry->second = AppendMesh(page, instance.mesh);
        bounds.push_back(TransformBounds(instance.mesh.nodes.front(), instance.objectToWorld));
        packed.push_back(EncodeInstance(instance, entry->second));
    }
    auto hierarchy = BuildBvh(bounds);
    if (hierarchy.nodes.empty())
        return page;

    page.root = PageOffset(page.nodes.size(), hierarchy.nodes.size());
    AppendNodes(page.nodes, hierarchy.nodes, page.root, 0);
    page.instances.reserve(packed.size());
    for (uint32_t index : hierarchy.primitiveOrder)
        page.instances.push_back(packed[index]);
    return page;
}

std::vector<RayGeometryWord> EncodeRayGeometryPage(const RayGeometryPage &page, RayGeometryStatus status)
{
    const uint64_t nodes = 2;
    const uint64_t positions = nodes + uint64_t{page.nodes.size()} * (sizeof(BvhNode) / sizeof(RayGeometryWord));
    const uint64_t triangles = positions + page.positions.size();
    const uint64_t instances = triangles + page.triangles.size();
    const uint64_t size = instances + uint64_t{page.instances.size()} * (sizeof(RayInstance) / sizeof(RayGeometryWord));
    if (size > kInvalidRayIndex)
        throw std::length_error("Ray geometry exceeds the uvec4 address range");
    std::vector<RayGeometryWord> storage(static_cast<size_t>(size));
    storage[0] = {page.root, static_cast<uint32_t>(nodes), static_cast<uint32_t>(positions),
                  static_cast<uint32_t>(triangles)};
    storage[1] = {static_cast<uint32_t>(instances), static_cast<uint32_t>(status), 0u, 0u};
    const auto append = [&](uint64_t offset, const auto &values) {
        using Value = typename std::decay_t<decltype(values)>::value_type;
        static_assert(sizeof(Value) % sizeof(RayGeometryWord) == 0);
        if (!values.empty())
            std::memcpy(storage.data() + static_cast<size_t>(offset), values.data(), values.size() * sizeof(Value));
    };
    append(nodes, page.nodes);
    append(positions, page.positions);
    append(triangles, page.triangles);
    append(instances, page.instances);
    return storage;
}

} // namespace infernux::raytracing
