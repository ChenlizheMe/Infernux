#include "QuaternionEuler.h"
#include <algorithm>
#include <cmath>

namespace infernux::rotation
{
std::array<float, 4> FromEulerYXZ(float x, float y, float z)
{
    constexpr float radians = 0.017453292519943295769f;
    x *= radians;
    y *= radians;
    z *= radians;
    const float cx = std::cos(x * .5f), sx = std::sin(x * .5f);
    const float cy = std::cos(y * .5f), sy = std::sin(y * .5f);
    const float cz = std::cos(z * .5f), sz = std::sin(z * .5f);
    return {cy * sx * cz + sy * cx * sz, sy * cx * cz - cy * sx * sz, cy * cx * sz - sy * sx * cz,
            cy * cx * cz + sy * sx * sz};
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
std::array<float, 4>
MultiplyPublic(const std::array<float, 4> &left, const std::array<float, 4> &right)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const double x = left[0], y = left[1], z = left[2], w = left[3];
    const double a = right[0], b = right[1], c = right[2], d = right[3];
    return {
        static_cast<float>(((w * a + x * d) + y * c) - z * b), static_cast<float>(((w * b + y * d) + z * a) - x * c),
        static_cast<float>(((w * c + z * d) + x * b) - y * a), static_cast<float>(((w * d - x * a) - y * b) - z * c)};
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
std::array<float, 3>
RotatePublic(const std::array<float, 4> &q, const std::array<float, 3> &point)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const float x2 = q[0] * 2.0f, y2 = q[1] * 2.0f, z2 = q[2] * 2.0f;
    const double xx = static_cast<float>(q[0] * x2), yy = static_cast<float>(q[1] * y2);
    const double zz = static_cast<float>(q[2] * z2), xy = static_cast<float>(q[0] * y2);
    const double xz = static_cast<float>(q[0] * z2), yz = static_cast<float>(q[1] * z2);
    const double wx = static_cast<float>(q[3] * x2), wy = static_cast<float>(q[3] * y2);
    const double wz = static_cast<float>(q[3] * z2);
    return {static_cast<float>((1.0 - (yy + zz)) * point[0] + (xy - wz) * point[1] + (xz + wy) * point[2]),
            static_cast<float>((xy + wz) * point[0] + (1.0 - (xx + zz)) * point[1] + (yz - wx) * point[2]),
            static_cast<float>((xz - wy) * point[0] + (yz + wx) * point[1] + (1.0 - (xx + yy)) * point[2])};
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
std::array<float, 4>
MultiplyTransform(const std::array<float, 4> &left, const std::array<float, 4> &right)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const float x = left[0], y = left[1], z = left[2], w = left[3];
    const float a = right[0], b = right[1], c = right[2], d = right[3];
    return {((y * c - z * b) + w * a) + x * d, ((z * a - x * c) + w * b) + y * d, ((w * c - y * a) + z * d) + x * b,
            ((w * d - x * a) - z * c) - y * b};
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
std::array<float, 4>
NormalizeTransform(float x, float y, float z, float w)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const float xx = x * x, yy = y * y, zz = z * z, ww = w * w;
    const float magnitudeXZ = std::sqrt((xx + yy) + (zz + ww));
    const float magnitudeYW = std::sqrt((yy + zz) + (ww + xx));
    if (magnitudeXZ == 0.0f)
        return {0, 0, 0, 1};
    return {x / magnitudeXZ, y / magnitudeYW, z / magnitudeXZ, w / magnitudeYW};
}

std::array<float, 4> NormalizePublic(float x, float y, float z, float w)
{
    // The Unity 2022 Mono public-property reference retains precision during
    // the dot product, stores its result as float, then divides each component
    // by the float magnitude. A reciprocal multiply changes public components.
    const float norm = static_cast<float>(static_cast<double>(x) * x + static_cast<double>(y) * y +
                                          static_cast<double>(z) * z + static_cast<double>(w) * w);
    const float magnitude = std::sqrt(norm);
    if (magnitude == 0.0f)
        return {0, 0, 0, 1};
    return {x / magnitude, y / magnitude, z / magnitude, w / magnitude};
}

std::array<float, 3> ToEulerYXZ(float x, float y, float z, float w)
{
    float xx = x * x, yy = y * y, zz = z * z, ww = w * w;
    float norm = xx + yy;
    norm += zz;
    norm += ww;
    const float magnitude = std::sqrt(norm);
    if (magnitude < 1e-5f)
        return {0, 0, 0};
    x /= magnitude;
    y /= magnitude;
    z /= magnitude;
    w /= magnitude;
    xx = x * x;
    yy = y * y;
    zz = z * z;
    ww = w * w;
    const float xw = x * w, yz = y * z;
    const float test = xw - yz;
    constexpr float degrees = 57.29577951308232f;
    const float pitch = std::asin(std::clamp(2.0f * test, -1.0f, 1.0f));
    float yaw, roll;
    if (std::abs(test) >= 0.499999f) {
        yaw = 2.0f * std::atan2(y, w);
        roll = 0;
    } else {
        const float xz = x * z, yw = y * w, xy = x * y, zw = z * w;
        float yawDenominator = -xx + zz;
        yawDenominator -= yy;
        yawDenominator += ww;
        float rollDenominator = yy - zz;
        rollDenominator -= xx;
        rollDenominator += ww;
        yaw = std::atan2(2.0f * (xz + yw), yawDenominator);
        roll = std::atan2(2.0f * (xy + zw), rollDenominator);
    }
    return {pitch * degrees, yaw * degrees, roll * degrees};
}
} // namespace infernux::rotation
