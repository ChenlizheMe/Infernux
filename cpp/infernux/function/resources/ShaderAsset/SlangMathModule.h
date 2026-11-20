#pragma once

#include "SlangCompiler.h"

namespace infernux
{

// Logical matrices use row/column indexing on every target. Slang owns the
// physical representation; these helpers never reinterpret uploaded storage.
inline const ShaderSourceModule &GetSlangMathModule()
{
    static const ShaderSourceModule module{"inx_math", "<infernux/shader/math>", R"slang(
namespace inx_math
{
[ForceInline]
public int floor_div(int a, int b)
{
    int q = a / b;
    int r = a - q * b;
    return q - int(r != 0 && (r < 0) != (b < 0));
}

[ForceInline]
public uint floor_div(uint a, uint b)
{
    return a / b;
}

[ForceInline]
public float floor_div(float a, float b)
{
    return floor(a / b);
}

[ForceInline]
public float2x2 inverse(float2x2 m)
{
    return float2x2(m[1][1], -m[0][1], -m[1][0], m[0][0]) / determinant(m);
}

[ForceInline]
public float3x3 inverse(float3x3 m)
{
    float3 c0 = cross(m[1], m[2]);
    float3 c1 = cross(m[2], m[0]);
    float3 c2 = cross(m[0], m[1]);
    return transpose(float3x3(c0, c1, c2)) / dot(m[0], c0);
}

[ForceInline]
public float4x4 inverse(float4x4 m)
{
    // The twelve 2x2 minors are shared by all sixteen adjugate entries.
    float s0 = m[0][0] * m[1][1] - m[1][0] * m[0][1];
    float s1 = m[0][0] * m[1][2] - m[1][0] * m[0][2];
    float s2 = m[0][0] * m[1][3] - m[1][0] * m[0][3];
    float s3 = m[0][1] * m[1][2] - m[1][1] * m[0][2];
    float s4 = m[0][1] * m[1][3] - m[1][1] * m[0][3];
    float s5 = m[0][2] * m[1][3] - m[1][2] * m[0][3];
    float c0 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
    float c1 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
    float c2 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
    float c3 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
    float c4 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
    float c5 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
    float det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
    return float4x4(
        m[1][1] * c5 - m[1][2] * c4 + m[1][3] * c3,
        -m[0][1] * c5 + m[0][2] * c4 - m[0][3] * c3,
        m[3][1] * s5 - m[3][2] * s4 + m[3][3] * s3,
        -m[2][1] * s5 + m[2][2] * s4 - m[2][3] * s3,
        -m[1][0] * c5 + m[1][2] * c2 - m[1][3] * c1,
        m[0][0] * c5 - m[0][2] * c2 + m[0][3] * c1,
        -m[3][0] * s5 + m[3][2] * s2 - m[3][3] * s1,
        m[2][0] * s5 - m[2][2] * s2 + m[2][3] * s1,
        m[1][0] * c4 - m[1][1] * c2 + m[1][3] * c0,
        -m[0][0] * c4 + m[0][1] * c2 - m[0][3] * c0,
        m[3][0] * s4 - m[3][1] * s2 + m[3][3] * s0,
        -m[2][0] * s4 + m[2][1] * s2 - m[2][3] * s0,
        -m[1][0] * c3 + m[1][1] * c1 - m[1][2] * c0,
        m[0][0] * c3 - m[0][1] * c1 + m[0][2] * c0,
        -m[3][0] * s3 + m[3][1] * s1 - m[3][2] * s0,
        m[2][0] * s3 - m[2][1] * s1 + m[2][2] * s0) / det;
}

[ForceInline]
public matrix<float, R, C> matrix_comp_mult<let R : int, let C : int>(
    matrix<float, R, C> a, matrix<float, R, C> b)
{
    return a * b;
}

[ForceInline]
public matrix<float, R, C> outer_product<let R : int, let C : int>(
    vector<float, R> column, vector<float, C> row)
{
    matrix<float, R, C> result;
    [ForceUnroll]
    for (int i = 0; i < R; ++i)
        [ForceUnroll]
        for (int j = 0; j < C; ++j)
            result[i][j] = column[i] * row[j];
    return result;
}
}
)slang"};
    return module;
}

} // namespace infernux
