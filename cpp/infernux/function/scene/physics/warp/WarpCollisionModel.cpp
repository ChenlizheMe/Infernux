// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#include "WarpCollisionModel.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <tuple>

namespace infernux::physics::warp
{
static_assert(sizeof(WarpLocalBound) == 64);
static_assert(sizeof(WarpCollisionShape) == 32);
static_assert(sizeof(WarpContactMaterial) == 96);

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
    result.pairRules.resize(uint64_t{count} + 1);
    result.pairRules.reserve(result.pairRules.size() + pairs.size());
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
        result.materials.push_back(material);
        result.pairRules[geom] = {weld, weldParent, uint32_t(model.bodies[weld].dofs[1] != 0), 0};
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
