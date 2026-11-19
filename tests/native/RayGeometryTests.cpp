#include <function/renderer/InxRenderStruct.h>
#include <function/renderer/raytracing/RayEmissive.h>
#include <function/renderer/raytracing/RayGeometry.h>
#include <function/renderer/raytracing/RayShadowChanges.h>

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

    BvhTopology topology;
    topology.primitiveOrder = result.primitiveOrder;
    for (const auto &node : result.nodes)
        topology.links.push_back({node.first, node.second});
    BvhRefit refit(topology, moved);
    assert(!refit.Advance(0));
    while (!refit.Advance(1)) {
    }
    assert(refit.Cost() >= 1.0);
    const auto incremental = refit.Take();
    assert(incremental.size() == repeat.nodes.size());
    assert(std::memcmp(incremental.data(), repeat.nodes.data(), incremental.size() * sizeof(BvhNode)) == 0);
}

void CheckVisibilityHistory(const RayInstancePage &page)
{
    const RenderLightingSource lights;
    std::vector<RayShadowInstance> instances{{1u, 1u, true, true, glm::mat4(1.0f), {}},
                                             {2u, 1u, true, true, glm::mat4(1.0f), {}}};
    const auto publish = [&](const RayShadowState *previous) {
        RayShadowChanges changes(previous, instances, page, lights, 3);
        uint32_t budget = 1024;
        assert(changes.Advance(budget));
        return changes.Take();
    };
    const auto initial = publish(nullptr);
    instances[0].objectToWorld[3].x += 0.1f;
    const auto moved = publish(&initial);
    instances[0].objectToWorld[3].x += 0.1f;
    const auto skipped = publish(&moved);
    assert(skipped.stamps[0][1] == skipped.serial && skipped.stamps[0][2] == initial.stamps[0][2]);
    instances[1].objectToWorld[3].x += 0.1f;
    const auto switched = publish(&skipped);
    assert(switched.stamps[0][1] == skipped.serial && switched.stamps[0][2] == switched.serial);
    instances[1].objectToWorld[3].x += 0.1f;
    instances[0].objectToWorld[3].x += 0.1f;
    const auto both = publish(&switched);
    assert(both.stamps[0][1] == both.serial && both.stamps[0][2] == both.serial);
    instances[0].objectToWorld[0].x = 2.0f;
    const auto scale = publish(&both);
    assert(scale.stamps[0][1] == scale.serial && scale.stamps[0][2] == both.serial);
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
    CheckVisibilityHistory(page);
    assert(page.nodes.size() == 1u && page.instances.size() == 2u);
    assert(page.instances[0].root == page.instances[1].root);
    assert(page.instances[1].worldToObjectX[0] == -0.5f && page.instances[1].worldToObjectX[3] == 2.0f);
    assert(page.instances[1].worldToObjectY[1] == 0.25f && page.instances[1].worldToObjectY[3] == -2.0f);
    assert(page.instances[1].surfaceId == 2u && page.instances[1].layerMask == 4u);
    assert(page.instances[0].orientation == 1.0f && page.instances[1].orientation == -1.0f);
    assert((page.instances[1].objectToWorldX == std::array<float, 4>{-2.0f, 0.0f, 0.0f, 4.0f}));
    assert((page.instances[1].objectToWorldY == std::array<float, 4>{0.0f, 4.0f, 0.0f, 8.0f}));
    assert((page.instances[1].objectToWorldZ == std::array<float, 4>{0.0f, 0.0f, 1.0f, 12.0f}));

    const RayPageUpload meshUpload(meshPage);
    const std::vector<RayGeometryWord> shadowStamps{{0u, 17u, 0u, 0u}};
    const RayPageUpload instanceUpload(page, RayGeometryStatus::Ready, 17u, shadowStamps);
    const auto header = [](const RayPageUpload &upload) {
        return *static_cast<const RayGeometryWord *>(upload.Section(0).data);
    };
    assert(header(meshUpload)[0] == 1u);
    assert(header(meshUpload)[3] + 4 * meshPage.attributes.size() == meshUpload.bytes / sizeof(RayGeometryWord));
    assert((header(instanceUpload) == RayGeometryWord{0u, 3u, 5u, 1u}));
    const auto *instanceHeaders = static_cast<const RayGeometryWord *>(instanceUpload.Section(0).data);
    assert((instanceHeaders[1] == RayGeometryWord{5u + 2u * kRayInstanceWords, 17u, 0u, 0u}));
    assert((instanceHeaders[2] == RayGeometryWord{6u + 2u * kRayInstanceWords, 6u + 2u * kRayInstanceWords, 0u, 0u}));
    assert(instanceUpload.Section(3).data == shadowStamps.data());
    assert(instanceUpload.bytes == 64u + sizeof(BvhNode) + 2 * sizeof(RayInstance));

    auto emissive = page;
    const std::vector<RayEmissiveSource> sources{{2u, 0u, 2u, 4.0f}, {1u, 0u, 2u, 1.0f}};
    const std::vector<float> areas{1.0f, 3.0f};
    RayEmissiveBuild distribution(sources, areas, emissive, 3u);
    while (!distribution.Advance(1u)) {
    }
    assert((emissive.emissiveRanges[2] == RayGeometryWord{0u, 2u, 0u, 1u}));
    assert((emissive.emissiveRanges[1] == RayGeometryWord{2u, 2u, 0u, 0u}));
    assert(emissive.emissiveTriangles.size() == 4u);
    const std::array<float, 4> expected{0.2f, 0.8f, 0.85f, 1.0f};
    for (size_t i = 0; i != expected.size(); ++i) {
        float cdf;
        std::memcpy(&cdf, &emissive.emissiveTriangles[i][3], sizeof(cdf));
        assert(cdf == expected[i]);
    }
    const RayPageUpload emitterUpload(emissive, RayGeometryStatus::Ready);
    const auto *emitterHeaders = static_cast<const RayGeometryWord *>(emitterUpload.Section(0).data);
    assert(emitterHeaders[2][2] == 4u && emitterHeaders[2][3] == 3u);
    assert(emitterUpload.Section(4).data == emissive.emissiveRanges.data());
    assert(emitterUpload.Section(5).data == emissive.emissiveTriangles.data());

    instances[1].objectToWorld[3].z += 7.0f;
    RayInstancePageBuild moving(instances, page.topology);
    while (!moving.Advance(1u)) {
    }
    const auto movedPage = moving.Take();
    assert(movedPage.topology == page.topology);
    assert(movedPage.instances[1].root == page.instances[1].root);
    assert(movedPage.nodes[0].maximum[2] > page.nodes[0].maximum[2]);
    assert(movedPage.instances[1].worldToObjectZ[3] != page.instances[1].worldToObjectZ[3]);

    const auto clustered = [&](bool interleaved) {
        std::vector<RayMeshInstance> result;
        for (uint32_t index = 0; index < 32u; ++index) {
            const bool right = interleaved ? (index % 2u != 0u) : index >= 16u;
            glm::mat4 transform(1.0f);
            transform[3].x = float(index % 16u) * 0.7f + (right ? 128.0f : 0.0f);
            result.push_back({meshPage.placements[0], transform, index + 1u, 1u});
        }
        return result;
    };
    const auto sorted = clustered(false);
    RayInstancePageBuild stable(sorted);
    while (!stable.Advance(7)) {
    }
    const auto old = stable.Take();
    const auto reordered = clustered(true);
    RayInstancePageBuild degraded(reordered, old.topology);
    while (!degraded.Advance(7)) {
    }
    const auto rebuilt = degraded.Take();
    assert(rebuilt.topology != old.topology);
    assert(rebuilt.topology->buildCost <= old.topology->buildCost * 1.5);
    assert(rebuilt.instances.size() == old.instances.size());
    for (size_t i = 0; i < rebuilt.instances.size(); ++i) {
        const auto sourceIndex = rebuilt.topology->primitiveOrder[i];
        assert(rebuilt.instances[i].surfaceId == reordered[sourceIndex].surfaceId);
    }

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
    assert((header(unavailable) == RayGeometryWord{kInvalidRayIndex, 3u, 3u, 2u}));
    assert(header(emptyReady)[3] == 1u);
}
