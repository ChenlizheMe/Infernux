#pragma once

#include <string_view>

namespace infernux::particle
{

/// Stable four-bit rank for one 256-lane workgroup. The sixteen-lane tiles
/// are logical ranges, independent of a device's physical subgroup width.
inline constexpr std::string_view SlangParticleRadixRank = R"slang(
groupshared uint radix_digits[256];
groupshared uint radix_tile_counts[256];

uint inx_radix_rank(uint lane, uint digit, bool active) {
    radix_digits[lane] = active ? digit : 0xffffffffu;
    GroupMemoryBarrierWithGroupSync();

    uint tile = lane / 16u;
    uint bin = lane % 16u;
    uint count = 0u;
    for (uint index = 0u; index < 16u; ++index)
        count += radix_digits[tile * 16u + index] == bin ? 1u : 0u;
    radix_tile_counts[lane] = count;
    GroupMemoryBarrierWithGroupSync();

    uint rank = 0u;
    for (uint prior = 0u; prior < tile; ++prior)
        rank += radix_tile_counts[prior * 16u + digit];
    for (uint prior = tile * 16u; prior < lane; ++prior)
        rank += radix_digits[prior] == digit ? 1u : 0u;
    return rank;
}
)slang";

} // namespace infernux::particle
