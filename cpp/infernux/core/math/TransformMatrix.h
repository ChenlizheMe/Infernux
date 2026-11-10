#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace infernux::transform_math
{
using Matrix = std::array<float, 16>; // column-major affine matrix
using Point = std::array<float, 3>;

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
inline float
Dot3(const Point &a, const Point &b)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const float x = a[0] * b[0], y = a[1] * b[1], z = a[2] * b[2];
    return x + (y + z);
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
// A Transform's inverse is built from its authored TRS, including collapsed
// axes. A generic inverse of its singular 4x4 matrix cannot retain those axes.
inline Matrix
InverseTRS(const Point &position, const std::array<float, 4> &rotation, const Point &scale)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const float x = rotation[0], y = rotation[1], z = rotation[2], w = rotation[3];
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, xz = x * z, yz = y * z;
    const float wx = w * x, wy = w * y, wz = w * z;
    const std::array<Point, 3> rows{{{{1.0f - 2.0f * (yy + zz), 2.0f * (xy + wz), 2.0f * (xz - wy)}},
                                     {{2.0f * (xy - wz), 1.0f - 2.0f * (xx + zz), 2.0f * (yz + wx)}},
                                     {{2.0f * (xz + wy), 2.0f * (yz - wx), 1.0f - 2.0f * (xx + yy)}}}};
    Matrix result{};
    result[15] = 1.0f;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        // Actual Transform measurements pin the inclusive 1e-9 boundary.
        if (std::abs(scale[axis]) < 1e-9f)
            continue;
        const float reciprocal = 1.0f / scale[axis];
        Point row;
        for (std::size_t column = 0; column < 3; ++column) {
            row[column] = rows[axis][column] * reciprocal;
            result[column * 4 + axis] = row[column];
        }
        result[12 + axis] = -Dot3(row, position);
    }
    return result;
}

// Compose affine matrices in column-major storage. Translation is an offset
// added after the 3x3 dot, rather than a fourth term in the matrix reduction.
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
inline Matrix
ComposeAffine(const float *leftMatrix, const float *rightMatrix)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    Matrix result{};
    result[15] = 1.0f;
    for (std::size_t row = 0; row < 3; ++row) {
        const Point left{{leftMatrix[row], leftMatrix[4 + row], leftMatrix[8 + row]}};
        for (std::size_t column = 0; column < 4; ++column) {
            const std::size_t offset = column * 4;
            result[offset + row] =
                Dot3(left, {{rightMatrix[offset], rightMatrix[offset + 1], rightMatrix[offset + 2]}});
        }
        result[12 + row] += leftMatrix[12 + row];
    }
    return result;
}
} // namespace infernux::transform_math
