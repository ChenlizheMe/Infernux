#include <function/renderer/InxRenderStruct.h>
#include <function/renderer/raytracing/RayGeometry.h>

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
    for (const uint32_t window : {1u, 17u}) {
        BvhBuild build(bounds);
        assert(!build.Advance(0));
        assert(!build.Advance(window));
        while (!build.Advance(window)) {
        }
        const auto resumed = build.Take();
        assert(result.primitiveOrder == resumed.primitiveOrder && result.nodes.size() == resumed.nodes.size());
        assert(std::memcmp(result.nodes.data(), resumed.nodes.data(), result.nodes.size() * sizeof(BvhNode)) == 0);
    }
    Visit(result, bounds, 0, 0);
    auto order = result.primitiveOrder;
    std::sort(order.begin(), order.end());
    for (size_t i = 0; i < order.size(); ++i)
        assert(order[i] == i);

    auto moved = bounds;
    moved.back().minimum.x -= 2.0f;
    moved.back().maximum.y += 3.0f;
    auto repeat = result;
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
    const std::vector<Bounds> emptyBounds;
    BvhBuild emptyBuild(emptyBounds);
    assert(emptyBuild.Advance(0) && emptyBuild.Take().nodes.empty());
    assert(BuildBvh(emptyBounds).nodes.empty());
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
    source[4].texCoord = {0.125f, -0.5f};
    source[4].texCoord1 = {3.0f, 0.25f};
    source[4].normal = {0.0f, 0.0f, 1.0f};
    source[4].tangent = {1.0f, 0.0f, 0.0f, -1.0f};
    source[4].color = {2.0f, 0.5f, 4.0f};
    const auto mesh = BuildRayMesh(source, {0, 1, 2, 3, 4, 5, 4, 6, 5}, 3, 6, 0);
    const auto basedMesh = BuildRayMesh(source, {0, 1, 2, 1, 3, 2}, 0, 6, 3);
    assert(mesh.positions.size() == 4 && mesh.triangles.size() == 2);
    assert(std::memcmp(mesh.triangles.data(), basedMesh.triangles.data(), 2 * sizeof(RayTriangle)) == 0);
    assert(std::memcmp(mesh.positions.data(), basedMesh.positions.data(), 4 * sizeof(RayPosition)) == 0);
    assert(std::memcmp(mesh.attributes.data(), basedMesh.attributes.data(), 4 * sizeof(RayVertexAttributes)) == 0);
    assert((mesh.attributes[1].uv == std::array{0.125f, -0.5f, 3.0f, 0.25f}));
    assert(mesh.attributes[1].tangent[3] == -1.0f && mesh.attributes[1].color[2] == 4.0f);

    glm::mat4 mirrored{1.0f};
    mirrored[0].x = -2.0f;
    mirrored[1].y = 4.0f;
    mirrored[3] = {4.0f, 8.0f, 12.0f, 1.0f};
    const std::vector<RayMesh> meshes{mesh, {}, basedMesh};
    RayMeshPageBuild pack(meshes);
    while (!pack.Advance(1u)) {
    }
    const auto meshPage = pack.Take();
    assert(meshPage.placements[0].root == 0u);
    assert(meshPage.placements[1].root == kInvalidRayIndex);
    assert(meshPage.placements[2].root == mesh.nodes.size());
    assert(meshPage.nodes.size() == mesh.nodes.size() + basedMesh.nodes.size());
    assert(meshPage.positions.size() == 8u && meshPage.attributes.size() == 8u);
    assert(meshPage.triangles.size() == 4u && meshPage.triangles[2].a >= 4u);

    std::vector<RayMeshInstance> instances{{meshPage.placements[0], glm::mat4(1.0f), 1u, 2u},
                                           {meshPage.placements[0], mirrored, 2u, 4u}};
    RayInstancePageBuild top(instances);
    while (!top.Advance(1u)) {
    }
    const auto page = top.Take();
    assert(page.nodes.size() == 1u && page.instances.size() == 2u);
    assert(page.instances[0].root == page.instances[1].root);
    assert(page.instances[1].worldToObjectX[0] == -0.5f && page.instances[1].worldToObjectX[3] == 2.0f);
    assert(page.instances[1].worldToObjectY[1] == 0.25f && page.instances[1].worldToObjectY[3] == -2.0f);
    assert(page.instances[1].surfaceId == 2u && page.instances[1].layerMask == 4u);
    assert(page.instances[0].orientation == 1.0f && page.instances[1].orientation == -1.0f);

    const RayPageUpload meshUpload(meshPage);
    const RayPageUpload instanceUpload(page, RayGeometryStatus::Ready);
    const auto header = [](const RayPageUpload &upload) {
        return *static_cast<const RayGeometryWord *>(upload.Section(0).data);
    };
    assert(header(meshUpload)[0] == 1u);
    assert(header(meshUpload)[3] + 4 * meshPage.attributes.size() == meshUpload.bytes / sizeof(RayGeometryWord));
    assert((header(instanceUpload) == RayGeometryWord{0u, 1u, 3u, 1u}));
    assert(instanceUpload.bytes == 16u + sizeof(BvhNode) + 2 * sizeof(RayInstance));

    instances[1].objectToWorld[3].z += 7.0f;
    RayInstancePageBuild moving(instances);
    while (!moving.Advance(1u)) {
    }
    const auto movedPage = moving.Take();
    assert(movedPage.instances[1].root == page.instances[1].root);
    assert(movedPage.nodes[0].maximum[2] > page.nodes[0].maximum[2]);
    assert(movedPage.instances[1].worldToObjectZ[3] != page.instances[1].worldToObjectZ[3]);

    const std::vector<uint32_t> indices{0, 1, 2, 3, 4, 5, 4, 6, 5};
    RayMeshBuild meshBuild(source, indices, 3, 6, 0);
    while (!meshBuild.Advance(1u)) {
    }
    const auto resumedMesh = meshBuild.Take();
    assert(resumedMesh.positions.size() == mesh.positions.size());
    assert(std::memcmp(resumedMesh.triangles.data(), mesh.triangles.data(), 2 * sizeof(RayTriangle)) == 0);

    const RayInstancePage empty;
    const RayPageUpload unavailable(empty, RayGeometryStatus::Unavailable);
    const RayPageUpload emptyReady(empty, RayGeometryStatus::Ready);
    assert((header(unavailable) == RayGeometryWord{kInvalidRayIndex, 1u, 1u, 2u}));
    assert(header(emptyReady)[3] == 1u);
}
