#pragma once

#include <array>
#include <cmath>

namespace infernux::transform_math
{
// Rotate local coordinates before advancing to the next hierarchy node.
// Keep the original vector separate from the rotation's delta matrix. Forming
// a full 3x3 first changes rounding, even though the real-number map is equal.
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
inline std::array<float, 3>
RotateLocal(const std::array<float, 4> &q, const std::array<float, 3> &v)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const float x = q[0], y = q[1], z = q[2], w = q[3];
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, xz = x * z, yz = y * z, wx = w * x, wy = w * y, wz = w * z;
    const float a0 = (-2.0f * (yy + zz)) * v[0], a1 = (2.0f * (xy + wz)) * v[0], a2 = (2.0f * (xz - wy)) * v[0];
    const float b0 = (2.0f * (xy - wz)) * v[1], b1 = (-2.0f * (xx + zz)) * v[1], b2 = (2.0f * (yz + wx)) * v[1];
    const float c0 = (2.0f * (xz + wy)) * v[2], c1 = (2.0f * (yz - wx)) * v[2], c2 = (-2.0f * (xx + yy)) * v[2];
    return {(a0 + v[0]) + (b0 + c0), (a1 + v[1]) + (b1 + c1), (a2 + v[2]) + (b2 + c2)};
}

inline std::array<float, 3> InverseLocalScale(const std::array<float, 3> &scale)
{
    return {std::abs(scale[0]) < 1e-9f ? 0.0f : 1.0f / scale[0], std::abs(scale[1]) < 1e-9f ? 0.0f : 1.0f / scale[1],
            std::abs(scale[2]) < 1e-9f ? 0.0f : 1.0f / scale[2]};
}
} // namespace infernux::transform_math
