#pragma once

#include "../FoundationApi.h"
#include <glm/glm.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace infernux::spatial
{

inline constexpr uint32_t kBvhLeafBit = 1u << 31u;
inline constexpr uint32_t kBvhLeafCapacity = 4;
inline constexpr uint32_t kBvhMaximumDepth = 48;

struct Bounds
{
    glm::vec3 minimum{std::numeric_limits<float>::infinity()};
    glm::vec3 maximum{-std::numeric_limits<float>::infinity()};

    INFERNUX_FOUNDATION_API void Include(const Bounds &other) noexcept;
    [[nodiscard]] INFERNUX_FOUNDATION_API glm::vec3 Center() const noexcept;
    [[nodiscard]] INFERNUX_FOUNDATION_API double Area() const noexcept;
};

/// Two GPU-aligned words. Leaves refer to consecutive entries in primitiveOrder.
struct alignas(16) BvhNode
{
    std::array<float, 3> minimum;
    uint32_t first;
    std::array<float, 3> maximum;
    uint32_t second;
};

static_assert(sizeof(BvhNode) == 32);
static_assert(offsetof(BvhNode, first) == 12 && offsetof(BvhNode, second) == 28);

struct Bvh
{
    std::vector<BvhNode> nodes;
    std::vector<uint32_t> primitiveOrder;
};

/// Nodes are stored in depth-first order with each parent before its children.
[[nodiscard]] INFERNUX_FOUNDATION_API Bvh BuildBvh(const std::vector<Bounds> &primitiveBounds);
/// Update bounds for the same primitives, retaining all addresses and their ordering.
INFERNUX_FOUNDATION_API void RefitBvh(Bvh &tree, const std::vector<Bounds> &primitiveBounds);

} // namespace infernux::spatial
