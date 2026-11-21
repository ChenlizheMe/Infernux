// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#include "PhysicsRayIndex.h"
#include "InxPhyGpuCollisionModel.h"

#include <algorithm>
#include <glm/gtc/quaternion.hpp>
#include <utility>

namespace infernux::physics::inxphygpu
{
namespace
{
constexpr uint32_t RefitSubtreeSize = 128;

struct NodeInfo
{
    uint32_t height = 0;
    uint32_t primitives = 0;
};

using RefitLevels = std::vector<std::vector<uint32_t>>;

uint32_t AppendIndices(std::vector<RayIndexWord> &records, const std::vector<uint32_t> &indices)
{
    const auto offset = static_cast<uint32_t>(records.size());
    records.resize(records.size() + (indices.size() + 3) / 4);
    for (size_t index = 0; index < indices.size(); ++index)
        records[offset + index / 4][index % 4] = indices[index];
    return offset;
}

std::vector<uint32_t> BuildEscapeLinks(const std::vector<spatial::BvhNode> &nodes)
{
    const auto count = static_cast<uint32_t>(nodes.size());
    std::vector<uint32_t> links(count, count);
    for (uint32_t index = 0; index < count; ++index) {
        const auto &node = nodes[index];
        if (!(node.second & spatial::kBvhLeafBit)) {
            links[node.first] = node.second;
            links[node.second] = links[index];
        }
    }
    return links;
}

void ScheduleSubtrees(const std::vector<spatial::BvhNode> &nodes, const std::vector<NodeInfo> &info,
                      const std::vector<uint32_t> &escape, uint32_t root, std::vector<RefitLevels> &subtrees,
                      RefitLevels &upper)
{
    const auto &entry = info[root];
    if (entry.primitives <= RefitSubtreeSize) {
        auto &levels = subtrees.emplace_back(entry.height + 1);
        for (auto index = root; index < escape[root]; ++index)
            levels[info[index].height].push_back(index);
        return;
    }

    upper[entry.height].push_back(root);
    ScheduleSubtrees(nodes, info, escape, nodes[root].first, subtrees, upper);
    ScheduleSubtrees(nodes, info, escape, nodes[root].second, subtrees, upper);
}

} // namespace

PhysicsRayIndex BuildPhysicsRayIndex(const std::vector<spatial::Bounds> &finiteBounds,
                                     const std::vector<uint32_t> &geometryIds, const std::vector<uint32_t> &planeIds)
{
    const spatial::Bvh tree = spatial::BuildBvh(finiteBounds);
    PhysicsRayIndex result;
    const auto count = static_cast<uint32_t>(tree.nodes.size());
    result.nodeCount = count;
    auto &records = result.records;
    records.resize(2 + count);
    std::vector<NodeInfo> info(count);
    const auto escape = BuildEscapeLinks(tree.nodes);
    for (uint32_t end = count; end > 0; --end) {
        const auto &node = tree.nodes[end - 1];
        auto &entry = info[end - 1];
        if (node.second & spatial::kBvhLeafBit) {
            entry.primitives = node.second & ~spatial::kBvhLeafBit;
        } else {
            entry.height = 1 + std::max(info[node.first].height, info[node.second].height);
            entry.primitives = info[node.first].primitives + info[node.second].primitives;
        }
    }
    for (uint32_t index = 0; index < count; ++index) {
        const auto &node = tree.nodes[index];
        records[2 + index] = {node.first, node.second, escape[index], 0};
    }

    std::vector<uint32_t> ordered;
    ordered.reserve(geometryIds.size());
    for (const auto index : tree.primitiveOrder)
        ordered.push_back(geometryIds[index]);
    const auto order = AppendIndices(records, ordered);
    const auto planes = AppendIndices(records, planeIds);
    records[0] = {2, order, planes, static_cast<uint32_t>(planeIds.size())};

    std::vector<RefitLevels> batches;
    if (count > 0) {
        RefitLevels upper(info.front().height + 1);
        ScheduleSubtrees(tree.nodes, info, escape, 0, batches, upper);
        result.refitSubtrees = static_cast<uint32_t>(batches.size());
        result.refitRoot = result.refitSubtrees > 1;
        if (result.refitRoot)
            batches.push_back(std::move(upper));
    }

    const auto batchStart = static_cast<uint32_t>(records.size());
    records[1] = {count, batchStart, result.refitSubtrees, static_cast<uint32_t>(result.refitRoot)};
    records.resize(records.size() + batches.size());
    for (size_t batch = 0; batch < batches.size(); ++batch) {
        auto &levels = batches[batch];
        levels.erase(std::remove_if(levels.begin(), levels.end(), [](const auto &level) { return level.empty(); }),
                     levels.end());
        const auto levelStart = static_cast<uint32_t>(records.size());
        records.resize(records.size() + levels.size());
        records[batchStart + batch] = {levelStart, static_cast<uint32_t>(levels.size()), 0, 0};
        for (size_t level = 0; level < levels.size(); ++level) {
            const auto offset = AppendIndices(records, levels[level]);
            records[levelStart + level] = {offset, static_cast<uint32_t>(levels[level].size()), 0, 0};
        }
    }
    return result;
}

PhysicsRayIndex BuildInxPhyGpuRayIndex(const InxPhyGpuModel &model, const InxPhyGpuCollisionModel &collision)
{
    struct Frame
    {
        glm::vec3 position{0.0f};
        glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};

        Frame Append(const ModelPose &pose) const
        {
            const auto &p = pose.position;
            const auto &q = pose.rotation;
            return {position + rotation * glm::vec3(p[0], p[1], p[2]), rotation * glm::quat(q[0], q[1], q[2], q[3])};
        }
    };
    std::vector<Frame> bodies(model.bodies.size());
    for (uint32_t index = 1; index < model.bodies.size(); ++index) {
        const auto &body = model.bodies[index];
        bodies[index] = bodies[body.hierarchy[0]].Append(body.pose);
    }

    std::vector<spatial::Bounds> bounds;
    std::vector<uint32_t> finiteIds, planes;
    bounds.reserve(model.geometries.size());
    finiteIds.reserve(model.geometries.size());
    for (uint32_t index = 0; index < model.geometries.size(); ++index) {
        const auto &geometry = model.geometries[index];
        if (geometry.indices[1] == Plane) {
            planes.push_back(index);
            continue;
        }
        const auto frame = bodies[geometry.indices[0]].Append(geometry.pose);
        const auto &shape = collision.bounds[index];
        const auto &p = shape.center;
        const auto &h = shape.halfExtent;
        const auto rotation = glm::mat3_cast(frame.rotation);
        const auto center = frame.position + rotation * glm::vec3(p[0], p[1], p[2]);
        const auto extent = glm::abs(rotation[0]) * h[0] + glm::abs(rotation[1]) * h[1] + glm::abs(rotation[2]) * h[2];
        bounds.push_back({center - extent, center + extent});
        finiteIds.push_back(index);
    }
    return BuildPhysicsRayIndex(bounds, finiteIds, planes);
}

uint32_t AppendPhysicsRayMesh(std::vector<RayIndexWord> &records, const std::vector<glm::vec3> &vertices,
                              const std::vector<std::array<uint32_t, 3>> &faces, uint32_t vertexBase)
{
    std::vector<spatial::Bounds> bounds;
    bounds.reserve(faces.size());
    for (const auto &face : faces) {
        const auto &a = vertices[face[0]];
        const auto &b = vertices[face[1]];
        const auto &c = vertices[face[2]];
        bounds.push_back({glm::min(a, glm::min(b, c)), glm::max(a, glm::max(b, c))});
    }
    const spatial::Bvh tree = spatial::BuildBvh(bounds);
    const auto escape = BuildEscapeLinks(tree.nodes);
    const auto descriptor = static_cast<uint32_t>(records.size());
    const auto nodeStart = descriptor + 1;
    const auto nodeCount = static_cast<uint32_t>(tree.nodes.size());
    const auto faceStart = nodeStart + 2 * nodeCount;
    records.push_back({nodeStart, faceStart, nodeCount, 0});
    for (uint32_t index = 0; index < nodeCount; ++index) {
        const auto &node = tree.nodes[index];
        const auto &lo = node.minimum;
        const auto &hi = node.maximum;
        records.push_back(
            {glm::floatBitsToUint(lo[0]), glm::floatBitsToUint(lo[1]), glm::floatBitsToUint(lo[2]), node.first});
        records.push_back({glm::floatBitsToUint(hi[0]), glm::floatBitsToUint(hi[1]), glm::floatBitsToUint(hi[2]),
                           (node.second & spatial::kBvhLeafBit) ? node.second : escape[index]});
    }
    for (const auto index : tree.primitiveOrder) {
        const auto &face = faces[index];
        records.push_back({vertexBase + face[0], vertexBase + face[1], vertexBase + face[2], index});
    }
    return descriptor;
}

} // namespace infernux::physics::inxphygpu
