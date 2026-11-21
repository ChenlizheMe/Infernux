#pragma once

#include "PhysicsTargetReference.h"

#include <glm/glm.hpp>

namespace infernux
{
/// Hit from one accepted scene state. Retained consumers use target and the
/// captured IDs. The raw convenience pointers are resolved at materialization.
struct RaycastHit
{
    glm::vec3 point{0.0f};
    glm::vec3 normal{0.0f};
    float distance = 0.0f;
    uint32_t bodyId = UINT32_MAX;        ///< Stable body identity, or no hit.
    uint32_t subShapeId = 0;             ///< Subshape path inside the body.
    uint32_t triangleIndex = UINT32_MAX; ///< Cooked triangle index for mesh hits.
    GameObject *gameObject = nullptr;
    Collider *collider = nullptr;
    PhysicsTargetReference target;
    uint64_t gameObjectId = 0;
    uint64_t colliderId = 0;
};
} // namespace infernux
