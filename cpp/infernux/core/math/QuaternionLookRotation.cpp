#include "QuaternionLookRotation.h"

#include <cmath>

namespace infernux::rotation
{
namespace
{
using Vector = std::array<float, 3>;
using Quaternion = std::array<float, 4>;

float Length(const Vector &value)
{
    return std::sqrt((value[0] * value[0] + value[1] * value[1]) + value[2] * value[2]);
}

Vector Divide(const Vector &value, float length)
{
    return {value[0] / length, value[1] / length, value[2] / length};
}

Vector Cross(const Vector &a, const Vector &b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

Quaternion FromBasis(const Vector &right, const Vector &up, const Vector &forward)
{
    const float trace = (right[0] + up[1]) + forward[2];
    if (trace > 0) {
        const float root = std::sqrt(trace + 1);
        const float scale = .5f / root;
        return {(up[2] - forward[1]) * scale, (forward[0] - right[2]) * scale, (right[1] - up[0]) * scale, .5f * root};
    }
    const std::array<Vector, 3> basis{right, up, forward};
    int i = 0;
    if (up[1] > right[0])
        i = 1;
    if (forward[2] > basis[i][i])
        i = 2;
    const int j = (i + 1) % 3, k = (j + 1) % 3;
    const float root = std::sqrt(((basis[i][i] - basis[j][j]) - basis[k][k]) + 1);
    const float scale = .5f / root;
    Quaternion result;
    result[i] = .5f * root;
    result[j] = (basis[i][j] + basis[j][i]) * scale;
    result[k] = (basis[i][k] + basis[k][i]) * scale;
    result[3] = (basis[j][k] - basis[k][j]) * scale;
    return result;
}
} // namespace

std::array<float, 4> LookRotation(const Vector &forward, const Vector &up)
{
    constexpr float minimumLength = 1e-5f;
    const float length = Length(forward);
    if (length < minimumLength)
        return {0, 0, 0, 1};
    Vector direction = Divide(forward, length);
    Vector right = Cross(up, direction);
    const float rightLength = Length(right);
    if (rightLength >= minimumLength) {
        right = Divide(right, rightLength);
        return FromBasis(right, Cross(direction, right), direction);
    }

    // Collinear/zero up uses the shortest rotation from +Z. Near -Z, use
    // the reference's fixed X axis. atan2 avoids cancellation in 1 + z.
    if (direction[2] < -.999999f)
        return {1, 0, 0, 0};
    const float xy = std::sqrt(direction[0] * direction[0] + direction[1] * direction[1]);
    if (xy == 0)
        return {0, 0, 0, 1};
    const double half = std::atan2(double(xy), double(direction[2])) * .5;
    const float sine = static_cast<float>(std::sin(half));
    const float cosine = static_cast<float>(std::cos(half));
    const float x = -direction[1] / xy * sine, y = direction[0] / xy * sine;
    right = {1 - 2 * y * y, 2 * x * y, -2 * y * cosine};
    const Vector correctedUp{2 * x * y, 1 - 2 * x * x, 2 * x * cosine};
    direction = {2 * y * cosine, -2 * x * cosine, 1 - 2 * (x * x + y * y)};
    return FromBasis(right, correctedUp, direction);
}
} // namespace infernux::rotation
