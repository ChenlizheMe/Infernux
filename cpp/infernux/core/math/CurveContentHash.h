#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace infernux::curve
{
inline uint32_t FloatBits(float value) noexcept
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// Ordered 32-bit content identity. Wrap modes do not enter this accumulator;
// the legacy tangent metadata does. Preserve the sign bit of zero as authored.
template <typename Key> int32_t ContentHash(const Key *keys, size_t count) noexcept
{
    uint32_t hash = count ? 17u : 0u;
    for (size_t index = 0; index < count; ++index) {
        const auto &key = keys[index];
        hash = hash * 23u + FloatBits(key.time);
        hash = hash * 23u + FloatBits(key.value);
        hash = hash * 23u + FloatBits(key.inTangent);
        hash = hash * 23u + FloatBits(key.outTangent);
        hash = hash * 23u + key.weightedMode;
        hash = hash * 23u + FloatBits(key.inWeight);
        hash = hash * 23u + FloatBits(key.outWeight);
        hash = hash * 23u + static_cast<uint32_t>(key.tangentMode);
    }
    int32_t result;
    std::memcpy(&result, &hash, sizeof(result));
    return result;
}
} // namespace infernux::curve
