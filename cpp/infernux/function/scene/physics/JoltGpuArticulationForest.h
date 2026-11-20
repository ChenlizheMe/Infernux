#pragma once
#include "JoltGpuArticulations.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <function/scene/physics/gpu/ArticulationIntegrationTypes.h>
#include <numeric>

namespace infernux
{
// Cold accepted-author boundary. Never reads BodyState, runtime frames or q.
inline std::shared_ptr<JoltGpuArticulationState> BuildJoltGpuArticulationForest(const JoltGpuModel &model,
                                                                                uint32_t world, uint64_t byteLimit)
{
    using namespace JPH::GPU;
    auto result = std::make_shared<JoltGpuArticulationState>();
    result->worldRow = world;
    const auto &range = model.worlds.at(world);
    const uint32_t bodyBase = range.bodies.x, bodies = range.bodies.y;
    const uint32_t jointBase = range.leaves.z, joints = range.leaves.w;
    // Accepted model already validates native rows and type/payload agreement.
    // A lowering capacity refusal is an empty candidate, not a World failure.
    const auto empty = [&] {
        result->header = {{0, 0, 0, 0}, {0, 48, 48, 48}, {48, 48, 48, 48}};
        result->prefix.resize(48);
        std::memcpy(result->prefix.data(), &result->header, 48);
        return result;
    };
    if (!bodies || !joints || 48ull + uint64_t(bodies) * 4u + uint64_t(joints) * 20u > byteLimit)
        return empty();
    std::vector<uint32_t> parent(bodies), incoming(bodies, ArtNoParent), degree(bodies, 0);
    std::vector<uint8_t> bad(bodies, 0);
    std::vector<std::vector<uint32_t>> children(bodies);
    std::iota(parent.begin(), parent.end(), 0u);
    const auto root = [&](uint32_t a) {
        while (parent[a] != a) {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        return a;
    };
    const auto local = [&](uint32_t row) { return row >= bodyBase && row - bodyBase < bodies; };
    const auto dynamic = [](const BodyModel &m) {
        return m.meta.z == MotionDynamic && std::isfinite(m.mass.x) && m.mass.x > 0 && m.mass.y == 0 && m.mass.z == 0 &&
               std::isfinite(m.invInertiaDiagonal.x) && m.invInertiaDiagonal.x > 0 &&
               std::isfinite(m.invInertiaDiagonal.y) && m.invInertiaDiagonal.y > 0 &&
               std::isfinite(m.invInertiaDiagonal.z) && m.invInertiaDiagonal.z > 0 && m.invInertiaDiagonal.w == 0;
    };
    for (uint32_t offset = 0; offset < joints; ++offset) {
        const uint32_t row = jointBase + offset;
        const auto &j = model.joints[row];
        if (!(j.bodies.w & JointEnabled))
            continue;
        const bool pa = local(j.bodies.x), ch = local(j.bodies.y);
        if (!pa || !ch || j.bodies.x == j.bodies.y) {
            if (pa)
                bad[j.bodies.x - bodyBase] = 1;
            if (ch)
                bad[j.bodies.y - bodyBase] = 1;
            continue;
        }
        const uint32_t p = j.bodies.x - bodyBase, c = j.bodies.y - bodyBase;
        parent[root(c)] = root(p);
        ++degree[c];
        incoming[c] = row;
        children[p].push_back(c); // accepted row order is deterministic
        bool supported = j.identity.z == world &&
                         (j.bodies.z == JointFixed || j.bodies.z == JointHinge || j.bodies.z == JointSlider) &&
                         !(j.bodies.w & (JointLimitsEnabled | JointMotorEnabled)) && dynamic(model.bodies[j.bodies.y]);
        if (j.bodies.z == JointHinge) {
            const auto &p = std::get<JointHingeAuthor>(j.payload).passive;
            supported &= p.x == 0 && p.y == 0 && std::isfinite(p.z) && p.z >= 0;
        }
        if (!supported)
            bad[c] = 1;
    }
    std::vector<std::vector<uint32_t>> components(bodies);
    for (uint32_t b = 0; b < bodies; ++b)
        components[root(b)].push_back(b);
    std::vector<uint32_t> bodyMap(bodies, ArtNoParent), jointMap(joints, ArtNoParent), traversal(bodies, ArtNoParent);
    for (const auto &component : components) {
        if (component.size() < 2)
            continue;
        uint32_t r = ArtNoParent, roots = 0;
        bool supported = true;
        uint64_t edges = 0;
        for (const auto b : component) {
            supported &= !bad[b] && degree[b] <= 1u;
            edges += degree[b];
            if (!degree[b]) {
                r = b;
                ++roots;
            }
        }
        if (!supported || roots != 1 || edges != component.size() - 1u)
            continue;
        const auto &rootModel = model.bodies[bodyBase + r];
        const bool fixedRoot = rootModel.meta.z == MotionStatic;
        if (!fixedRoot && !dynamic(rootModel))
            continue;
        const uint32_t tree = static_cast<uint32_t>(result->trees.size());
        const uint32_t first = static_cast<uint32_t>(result->links.size());
        std::vector<uint32_t> order{r};
        traversal[r] = 0;
        result->links.push_back({{bodyBase + r, ArtNoParent, ArtNoParent, ArtNoParent}, {0, 0, world, 0}});
        for (uint32_t i = 0; i < order.size(); ++i) {
            const uint32_t p = order[i];
            bodyMap[p] = tree;
            for (const uint32_t c : children[p]) {
                // Single parent + N-1 + one root prove no repeat in this connected component.
                traversal[c] = static_cast<uint32_t>(order.size());
                order.push_back(c);
                const uint32_t row = incoming[c];
                const auto &j = model.joints[row];
                result->links.push_back(
                    {{bodyBase + c, traversal[p], row, j.bodies.z}, {j.identity.x, j.identity.y, world, 0}});
                jointMap[row - jointBase] = tree;
            }
        }
        result->trees.push_back({{first, static_cast<uint32_t>(order.size()), 0, first},
                                 {uint32_t(fixedRoot), tree, world, bodyBase + r},
                                 {ArtPending, ArtInitialize, 0, 0}});
    }
    if (result->trees.empty())
        return empty();
    const auto align = [](uint64_t v) { return (v + 15u) & ~uint64_t(15u); };
    const uint64_t bodyMapByte = 48, jointMapByte = bodyMapByte + uint64_t(bodies) * 4u;
    const uint64_t flagsByte = jointMapByte + uint64_t(joints) * 4u;
    const uint64_t reasonsByte = flagsByte + uint64_t(result->trees.size()) * 4u;
    const uint64_t preRotorByte = align(reasonsByte + uint64_t(result->trees.size()) * 4u);
    const uint64_t scratchByte = preRotorByte + uint64_t(joints) * 16u;
    const uint64_t totalBytes = scratchByte + uint64_t(result->links.size()) * ArtScratchWords * 16u;
    if (totalBytes > byteLimit || totalBytes > UINT32_MAX) {
        result->links.clear();
        result->trees.clear();
        return empty();
    }
    result->header = {
        {bodyBase, bodies, jointBase, joints},
        {uint32_t(result->trees.size()), uint32_t(bodyMapByte), uint32_t(jointMapByte), uint32_t(flagsByte)},
        {uint32_t(reasonsByte), uint32_t(preRotorByte), uint32_t(scratchByte), uint32_t(totalBytes)}};
    result->prefix.resize(size_t(scratchByte), 0);
    std::memcpy(result->prefix.data(), &result->header, sizeof(result->header));
    std::memcpy(result->prefix.data() + bodyMapByte, bodyMap.data(), bodyMap.size() * 4u);
    std::memcpy(result->prefix.data() + jointMapByte, jointMap.data(), jointMap.size() * 4u);
    for (uint32_t tree = 0; tree < result->trees.size(); ++tree) {
        const uint32_t flags = ArtControlDirty;
        std::memcpy(result->prefix.data() + flagsByte + 4u * tree, &flags, 4);
    }
    for (auto &tree : result->trees)
        tree.range.z = uint32_t(scratchByte / 16u + uint64_t(tree.range.x) * ArtScratchWords);
    return result;
}
} // namespace infernux
