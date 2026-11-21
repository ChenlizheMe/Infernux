// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#include "InxPhyGpuModel.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace infernux::physics::inxphygpu
{
ModelWords InxPhyGpuJointLimits::Pack() const noexcept
{
    ModelWords words{static_cast<uint32_t>(enabled), 0u, 0u, 0u};
    std::memcpy(&words[1], &minimum, sizeof(minimum));
    std::memcpy(&words[2], &maximum, sizeof(maximum));
    return words;
}

namespace
{
static_assert(sizeof(ModelDimensions) == 32);
static_assert(sizeof(ModelBody) == 144);
static_assert(sizeof(ModelJoint) == 48);
static_assert(sizeof(ModelGeometry) == 48);
static_assert(sizeof(ModelDof) == 32);
static_assert(sizeof(ModelPose) == 32);

constexpr uint32_t CompactMassBlock = 0xfffffffeu;
constexpr uint32_t SmallMassWidth = 8;
constexpr uint32_t TiledMassWidth = 64;

using MassEntry = std::array<uint32_t, 2>; // Other DOF and packed mass address.

template <class T> uint32_t Count(const std::vector<T> &values)
{
    return static_cast<uint32_t>(values.size());
}

ModelVector WithW(const std::array<float, 3> &xyz, float w)
{
    return {xyz[0], xyz[1], xyz[2], w};
}

void AppendJoint(InxPhyGpuModel &model, const ModelJointDefinition &joint, uint32_t body, uint32_t &parentDof,
                 uint32_t &massSize)
{
    const uint32_t position = Count(model.initialPositions);
    uint32_t width = 1;
    switch (joint.type) {
    case ModelJointType::Free: {
        const auto &pose = model.bodies[body].pose;
        model.initialPositions.insert(model.initialPositions.end(), pose.position.begin(), pose.position.begin() + 3);
        model.initialPositions.insert(model.initialPositions.end(), pose.rotation.begin(), pose.rotation.end());
        width = 6;
        break;
    }
    case ModelJointType::Ball:
        model.initialPositions.insert(model.initialPositions.end(), {1.0f, 0.0f, 0.0f, 0.0f});
        width = 3;
        break;
    case ModelJointType::Slide:
    case ModelJointType::Hinge:
        model.initialPositions.push_back(joint.reference);
        break;
    }

    model.joints.push_back({{static_cast<uint32_t>(joint.type), position, Count(model.dofs), body},
                            WithW(joint.anchor, 0.0f),
                            WithW(joint.axis, joint.reference)});
    for (uint32_t axis = 0; axis < width; ++axis) {
        const uint32_t depth = parentDof == NoModelIndex ? 1 : model.dofs[parentDof].indices[3] + 1;
        const uint32_t current = Count(model.dofs);
        model.dofs.push_back({{body, parentDof, massSize, depth},
                              joint.armature[axis],
                              static_cast<uint32_t>(joint.gravityCompensatedByActuator)});
        massSize += depth;
        parentDof = current;
    }
}

void BuildBranches(InxPhyGpuModel &model, const std::vector<uint32_t> &leaves)
{
    model.branches.resize(leaves.size() + 1);
    const uint32_t payload = Count(model.branches);
    for (uint32_t branch = 0; branch < Count(leaves); ++branch) {
        const uint32_t start = Count(model.branches);
        model.branches[branch] = start - payload;
        for (uint32_t node = leaves[branch]; node != 0; node = model.bodies[node].hierarchy[0]) {
            model.branches.push_back(node);
            auto &owner = model.bodies[node].subtree[2];
            if (owner == NoModelIndex)
                owner = branch;
        }
        std::reverse(model.branches.begin() + start, model.branches.end());
    }
    model.branches[leaves.size()] = Count(model.branches) - payload;
}

void BuildMassAdjacency(InxPhyGpuModel &model)
{
    const uint32_t dofs = Count(model.dofs);
    auto &adjacency = model.massAdjacency;
    adjacency.resize(dofs + 1);
    for (uint32_t row = 0; row < dofs; ++row) {
        adjacency[row + 1] += model.dofs[row].indices[3];
        for (uint32_t col = model.dofs[row].indices[1]; col != NoModelIndex; col = model.dofs[col].indices[1])
            ++adjacency[col + 1];
    }
    for (uint32_t row = 0; row < dofs; ++row)
        adjacency[row + 1] += adjacency[row];
    adjacency.resize(dofs + 1 + 2ull * adjacency[dofs]);
    std::vector<uint32_t> cursors(adjacency.begin(), adjacency.begin() + dofs);
    const auto write = [&](uint32_t row, uint32_t col, uint32_t mass) {
        const uint32_t index = dofs + 1 + 2 * cursors[row]++;
        adjacency[index] = col;
        adjacency[index + 1] = mass;
    };
    for (uint32_t row = 0; row < dofs; ++row) {
        const auto &dof = model.dofs[row].indices;
        uint32_t packed = dof[2] + dof[3] - 1;
        write(row, row, packed);
        for (uint32_t col = dof[1]; col != NoModelIndex; col = model.dofs[col].indices[1]) {
            --packed;
            write(row, col, packed);
            write(col, row, packed);
        }
    }
}

ModelMassFactors BuildMassFactors(const InxPhyGpuModel &model, uint32_t massSize, bool referencePose)
{
    ModelMassFactors factors;
    const auto &adjacency = model.massAdjacency;
    const uint32_t entries = Count(model.dofs) + 1;
    const auto entry = [&](uint32_t index) {
        const uint32_t address = entries + 2 * index;
        return MassEntry{adjacency[address], adjacency[address + 1]};
    };
    uint32_t dense = 0;
    std::vector<ModelWords> sparseBlocks;
    std::vector<std::array<uint32_t, 2>> levels, contributions;
    std::vector<std::array<uint32_t, 3>> sparseRows;
    for (uint32_t first = 0; first < Count(model.dofs);) {
        uint32_t end = first + 1;
        while (end < Count(model.dofs) && model.dofs[end].indices[1] != NoModelIndex)
            ++end;
        const uint32_t width = end - first;
        if (!referencePose && width == 1) {
            auto &blocks = factors.smallBlocks;
            if (!blocks.empty() && blocks.back()[2] == CompactMassBlock && blocks.back()[0] + blocks.back()[1] == first)
                ++blocks.back()[1];
            else
                blocks.push_back({first, 1, CompactMassBlock, 0});
        } else if (!referencePose && width <= SmallMassWidth && model.dofs[end - 1].indices[3] == width) {
            factors.smallBlocks.push_back({first, width, dense, 0});
            dense += width * width;
        } else if (!referencePose && width <= TiledMassWidth) {
            const uint32_t gather = Count(factors.tileElements);
            factors.tiledBlocks.push_back({first, width, dense, gather});
            factors.tileElements.resize(gather + width * width, NoModelIndex);
            for (uint32_t row = first; row < end; ++row)
                for (uint32_t i = adjacency[row]; i < adjacency[row + 1]; ++i) {
                    const auto [column, mass] = entry(i);
                    factors.tileElements[gather + (row - first) * width + column - first] = mass;
                }
            dense += width * width;
        } else {
            std::vector<std::vector<uint32_t>> depth;
            for (uint32_t row = first; row < end; ++row) {
                const uint32_t level = model.dofs[row].indices[3];
                if (depth.size() < level)
                    depth.resize(level);
                depth[level - 1].push_back(row);
            }
            sparseBlocks.push_back({Count(levels), Count(depth), first, width});
            for (const auto &level : depth) {
                levels.push_back({Count(sparseRows), Count(level)});
                for (const uint32_t row : level) {
                    const uint32_t start = Count(contributions);
                    for (uint32_t i = adjacency[row]; i < adjacency[row + 1]; ++i) {
                        const auto [column, mass] = entry(i);
                        if (column > row)
                            contributions.push_back({column, mass});
                    }
                    sparseRows.push_back({row, start, Count(contributions) - start});
                }
            }
        }
        first = end;
    }
    factors.sparseBlockCount = Count(sparseBlocks);
    factors.layout = {dense + (sparseBlocks.empty() ? 0 : massSize), dense, Count(factors.smallBlocks), 0};
    if (sparseBlocks.empty())
        return factors;
    const uint32_t levelOffset = 4 + 4 * Count(sparseBlocks);
    const uint32_t rowOffset = levelOffset + 2 * Count(levels);
    const uint32_t contributionOffset = rowOffset + 3 * Count(sparseRows);
    auto &topology = factors.sparseTopology;
    topology.reserve(contributionOffset + 2 * contributions.size());
    topology.insert(topology.end(), {levelOffset, rowOffset, contributionOffset, 0});
    const auto append = [&](const auto &records) {
        for (const auto &record : records)
            topology.insert(topology.end(), record.begin(), record.end());
    };
    append(sparseBlocks);
    append(levels);
    append(sparseRows);
    append(contributions);
    return factors;
}
} // namespace

InxPhyGpuModel BuildInxPhyGpuModel(const std::vector<ModelBodyDefinition> &bodies,
                                   const std::vector<ModelGeometryDefinition> &geometries)
{
    const uint32_t world = Count(bodies);
    std::vector<std::vector<uint32_t>> children(bodies.size() + 1);
    for (uint32_t body = 0; body < world; ++body) {
        const uint32_t parent = bodies[body].parent;
        if (parent != NoModelIndex && parent >= world)
            throw std::invalid_argument("inxphygpu model body parent is outside the model");
        children[parent == NoModelIndex ? world : parent].push_back(body);
    }

    InxPhyGpuModel model;
    model.bodies.reserve(bodies.size() + 1);
    model.bodies.emplace_back();
    model.bodies[0].hierarchy = {NoModelIndex, NoModelIndex, 0, NoModelIndex};
    model.bodies[0].subtree = {0, 1, NoModelIndex, 0};
    model.bodies[0].dofs[0] = NoModelIndex;
    model.bodyRows.resize(bodies.size());
    std::vector<uint32_t> pending(children[world].rbegin(), children[world].rend());
    std::vector<uint32_t> leaves;
    std::vector<uint32_t> ancestorDof{NoModelIndex};
    ancestorDof.reserve(bodies.size() + 1);
    uint32_t massSize = 0;
    while (!pending.empty()) {
        const uint32_t source = pending.back();
        pending.pop_back();
        const auto &authored = bodies[source];
        const uint32_t row = Count(model.bodies);
        model.bodyRows[source] = row;
        const uint32_t parent = authored.parent == NoModelIndex ? 0 : model.bodyRows[authored.parent];
        if (authored.kinematic && (parent != 0 || !authored.joints.empty()))
            throw std::invalid_argument("inxphygpu model kinematic body must be a jointless root");
        if (authored.editableStatic &&
            (parent != 0 || authored.kinematic || !authored.joints.empty() || authored.mass != 0.0f))
            throw std::invalid_argument("inxphygpu model editable static body must be a massless, jointless root");
        ModelBody &body = model.bodies.emplace_back();
        body.hierarchy = {parent, authored.joints.empty() ? NoModelIndex : Count(model.joints), Count(authored.joints),
                          authored.kinematic ? Count(model.initialKinematicPoses) : NoModelIndex};
        body.subtree = {parent == 0 ? row : model.bodies[parent].subtree[0], 1, NoModelIndex, 0};
        body.pose = authored.pose;
        body.inertiaPose = authored.inertiaPose;
        body.inertiaMass = WithW(authored.principalInertia, authored.mass);
        body.subtreeMass = {authored.mass, authored.gravityCompensation, 0.0f, 0.0f};
        const uint32_t firstDof = Count(model.dofs);
        uint32_t lastDof = ancestorDof[parent];
        for (const auto &joint : authored.joints) {
            if (joint.type == ModelJointType::Free && (parent != 0 || authored.joints.size() != 1))
                throw std::invalid_argument("inxphygpu model free joint must be the root body's only joint");
            AppendJoint(model, joint, row, lastDof, massSize);
        }
        const uint32_t dofCount = Count(model.dofs) - firstDof;
        body.dofs = {dofCount == 0 ? NoModelIndex : firstDof, dofCount, 0, 0};
        if (authored.editableStatic) {
            model.initialStaticPoses.push_back(authored.pose);
            body.dofs[2] = Count(model.initialStaticPoses); // Zero denotes no editable static pose.
        }
        ancestorDof.push_back(lastDof);
        if (authored.kinematic)
            model.initialKinematicPoses.push_back(authored.pose);
        if (children[source].empty())
            leaves.push_back(row);
        pending.insert(pending.end(), children[source].rbegin(), children[source].rend());
    }
    if (model.bodies.size() != bodies.size() + 1)
        throw std::invalid_argument("inxphygpu model body hierarchy contains a cycle");
    model.bodies[0].dofs[3] = Count(model.initialStaticPoses);
    for (uint32_t row = Count(model.bodies) - 1; row != 0; --row) {
        const auto &body = model.bodies[row];
        auto &parent = model.bodies[body.hierarchy[0]];
        parent.subtree[1] += body.subtree[1];
        parent.subtreeMass[0] += body.subtreeMass[0];
    }
    model.geometries.reserve(geometries.size());
    for (const auto &geometry : geometries) {
        const uint32_t body = geometry.body == NoModelIndex ? 0 : model.bodyRows.at(geometry.body);
        model.geometries.push_back(
            {{body, geometry.type, geometry.collisionType, geometry.collisionAffinity}, geometry.pose});
    }
    BuildBranches(model, leaves);
    BuildMassAdjacency(model);
    model.massFactors = BuildMassFactors(model, massSize, false);
    model.dimensions.state = {Count(model.initialPositions), Count(model.dofs), Count(model.bodies),
                              Count(model.joints)};
    model.dimensions.layout = {Count(model.geometries), Count(model.initialKinematicPoses), Count(leaves), massSize};
    return model;
}

ModelMassFactors BuildInxPhyGpuReferenceMassFactors(const InxPhyGpuModel &model)
{
    return BuildMassFactors(model, model.dimensions.layout[3], true);
}
} // namespace infernux::physics::inxphygpu
