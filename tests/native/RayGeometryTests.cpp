#include <function/renderer/InxRenderStruct.h>
#include <function/renderer/raytracing/RayGeometry.h>
#include <function/resources/InxMesh/InxMesh.h>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

using namespace infernux;
using namespace infernux::raytracing;
using namespace infernux::spatial;

namespace
{
Bounds Visit(const Bvh &tree, const std::vector<Bounds> &primitives, uint32_t nodeIndex, uint32_t depth)
{
    assert(depth <= kBvhMaximumDepth);
    const auto &node = tree.nodes.at(nodeIndex);
    Bounds result;
    if (node.second & kBvhLeafBit) {
        const auto count = node.second & ~kBvhLeafBit;
        assert(count > 0 && count <= kBvhLeafCapacity);
        for (uint32_t i = 0; i < count; ++i)
            result.Include(primitives.at(tree.primitiveOrder.at(node.first + i)));
    } else {
        result.Include(Visit(tree, primitives, node.first, depth + 1));
        result.Include(Visit(tree, primitives, node.second, depth + 1));
    }
    for (unsigned i = 0; i < 3; ++i) {
        assert(result.minimum[i] == node.minimum[i]);
        assert(result.maximum[i] == node.maximum[i]);
    }
    return result;
}

void CheckHierarchy(const std::vector<Bounds> &bounds)
{
    const auto result = BuildBvh(bounds);
    auto repeat = BuildBvh(bounds);
    assert(result.primitiveOrder == repeat.primitiveOrder && result.nodes.size() == repeat.nodes.size());
    assert(std::memcmp(result.nodes.data(), repeat.nodes.data(), result.nodes.size() * sizeof(BvhNode)) == 0);
    Visit(result, bounds, 0, 0);
    auto order = result.primitiveOrder;
    std::sort(order.begin(), order.end());
    for (size_t i = 0; i < order.size(); ++i)
        assert(order[i] == i);

    auto moved = bounds;
    moved.back().minimum.x -= 2.0f;
    moved.back().maximum.y += 3.0f;
    RefitBvh(repeat, moved);
    assert(result.primitiveOrder == repeat.primitiveOrder);
    for (size_t i = 0; i < result.nodes.size(); ++i) {
        assert(result.nodes[i].first == repeat.nodes[i].first);
        assert(result.nodes[i].second == repeat.nodes[i].second);
    }
    Visit(repeat, moved, 0, 0);
}
} // namespace

int main()
{
    assert(BuildBvh({}).nodes.empty());
    // Equal centroids need a stable median. Exponential separation pressures the depth limit.
    CheckHierarchy(std::vector<Bounds>(80, Bounds{glm::vec3(-1.0f), glm::vec3(1.0f)}));
    std::vector<Bounds> unbalanced;
    for (int exponent = -90; exponent < 96; ++exponent) {
        const float x = std::ldexp(1.0f, exponent);
        unbalanced.push_back({glm::vec3(x, 0, 0), glm::vec3(x, 1, 1)});
    }
    CheckHierarchy(unbalanced);

    // Two submeshes in the canonical asset stream. Geometry is addressed by the stored index,
    // so a nonzero submesh vertex-range start is never added to an asset index.
    std::vector<Vertex> source(7);
    source[3].pos = {0.0f, 0.0f, 0.0f};
    source[4].pos = {1.0f, 0.0f, 0.0f};
    source[5].pos = {0.0f, 1.0f, 0.0f};
    source[6].pos = {1.0f, 1.0f, 0.0f};
    const auto mesh = BuildRayMesh(source, {0, 1, 2, 3, 4, 5, 4, 6, 5}, 3, 6, 0);
    const auto basedMesh = BuildRayMesh(source, {0, 1, 2, 1, 3, 2}, 0, 6, 3);
    assert(mesh.positions.size() == 4 && mesh.triangles.size() == 2);
    assert(std::memcmp(mesh.triangles.data(), basedMesh.triangles.data(), 2 * sizeof(RayTriangle)) == 0);
    assert(std::memcmp(mesh.positions.data(), basedMesh.positions.data(), 4 * sizeof(RayPosition)) == 0);

    glm::mat4 mirrored{1.0f};
    mirrored[0].x = -2.0f;
    mirrored[1].y = 4.0f;
    mirrored[3] = {4.0f, 8.0f, 12.0f, 1.0f};
    const auto page = BuildRayGeometryPage({{mesh, glm::mat4(1.0f), 1u, 2u}, {mesh, mirrored, 2u, 4u}});
    assert(page.root == mesh.nodes.size() && page.nodes.size() == mesh.nodes.size() + 1);
    assert(page.instances.size() == 2 && page.positions.size() == 4 && page.triangles.size() == 2);
    assert(page.instances[0].root == page.instances[1].root);
    assert(page.instances[1].worldToObjectX[0] == -0.5f && page.instances[1].worldToObjectX[3] == 2.0f);
    assert(page.instances[1].worldToObjectY[1] == 0.25f && page.instances[1].worldToObjectY[3] == -2.0f);
    assert(page.instances[1].surfaceId == 2u && page.instances[1].layerMask == 4u);
    assert(BuildRayGeometryPage({}).root == kInvalidRayIndex);

    auto asset = std::make_shared<MeshGeometry>();
    asset->vertices = source;
    asset->indices = {0, 1, 2, 3, 4, 5, 4, 6, 5};
    DrawCall first;
    first.indexStart = 3;
    first.indexCount = 6;
    first.meshDataOwner = asset;
    first.meshVertices = &asset->vertices;
    first.meshIndices = &asset->indices;
    first.surfaceId = 1;
    first.layerMask = 2;
    DrawCall offscreen = first;
    offscreen.frustumVisible = false;
    offscreen.worldMatrix = mirrored;
    offscreen.surfaceId = 2;
    offscreen.layerMask = 4;
    DrawCallResult world;
    world.drawCalls = {first, offscreen};
    const auto packed = BuildWorldRayGeometry(world);
    assert(packed == EncodeRayGeometryPage(page, RayGeometryStatus::Ready));
    assert(packed[0][0] == page.root && packed[1][0] + 8 == packed.size());
    assert(packed[packed[1][0] + 3][1] == first.surfaceId);
    assert(packed[packed[1][0] + 7][1] == offscreen.surfaceId);

    const auto stripped = std::make_shared<const MeshGeometry>();
    world.drawCalls[1].meshDataOwner = stripped;
    world.drawCalls[1].meshVertices = &stripped->vertices;
    world.drawCalls[1].meshIndices = &stripped->indices;
    const auto unavailable = BuildWorldRayGeometry(world);
    assert(unavailable == EncodeRayGeometryPage({}, RayGeometryStatus::Unavailable));
    const auto empty = BuildWorldRayGeometry({});
    assert(empty == EncodeRayGeometryPage({}, RayGeometryStatus::Ready));
    assert(empty != unavailable);
}
