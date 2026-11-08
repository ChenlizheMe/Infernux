#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace infernux
{
/// Captured before publication can generate, retire or reparent scene nodes.
struct InstantiatedObjectIds
{
    uint64_t object = 0;
    uint64_t transform = 0;
    std::vector<uint64_t> components; // Source component order, including Python.
};
using InstantiationIds = std::unordered_map<uint64_t, InstantiatedObjectIds>;
} // namespace infernux
