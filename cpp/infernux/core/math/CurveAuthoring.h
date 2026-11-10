#pragma once

#include <cstddef>

namespace infernux::curve
{
template <typename Key> float Slope(const Key &left, const Key &right)
{
    return left.time == right.time ? 0.f : (right.value - left.value) / (right.time - left.time);
}

// The caller supplies an existing key index. Authoring retains float32
// intermediate rounding, including on targets that support fused multiply-add.
template <typename Key> float SmoothedTangent(const Key *keys, size_t count, size_t index, float weight)
{
    if (index == 0)
        return Slope(keys[0], keys[1]);
    if (index + 1 == count)
        return Slope(keys[index - 1], keys[index]);
    const float left = Slope(keys[index - 1], keys[index]);
    const float right = Slope(keys[index], keys[index + 1]);
    const volatile float leftTerm = left * ((1.f + weight) * .5f);
    const volatile float rightTerm = right * ((1.f - weight) * .5f);
    return leftTerm + rightTerm;
}
} // namespace infernux::curve
