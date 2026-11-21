// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../../../core/math/Bvh.h"

namespace infernux::physics::warp
{

using RayIndexWord = std::array<uint32_t, 4>;

struct PhysicsRayIndex
{
    std::vector<RayIndexWord> records;
    uint32_t nodeCount = 0;
    uint32_t refitSubtrees = 0;
    bool refitRoot = false;
};

/// Scene cooking passes finite geometry bounds and their corresponding scene IDs.
/// Planes are tested separately because their collision bounds can be infinite.
[[nodiscard]] PhysicsRayIndex BuildPhysicsRayIndex(const std::vector<spatial::Bounds> &finiteBounds,
                                                   const std::vector<uint32_t> &geometryIds,
                                                   const std::vector<uint32_t> &planeIds);

/// Append a mesh-local tree and return its descriptor index. Face indices address
/// the same immutable vertex array as the convex and ray geometry programs.
[[nodiscard]] uint32_t AppendPhysicsRayMesh(std::vector<RayIndexWord> &records, const std::vector<glm::vec3> &vertices,
                                            const std::vector<std::array<uint32_t, 3>> &faces);

} // namespace infernux::physics::warp
