// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

struct LocalBound
{
    float4 center;
    float4 halfExtent;
    float4 radii; // bounding radius, margin + gap, maximum explicit pair margin + gap
    uint4 geometry; // type, body row, collision type mask, collision affinity mask
};

struct ProjectedBound
{
    float4 originRadius;
    float4 centerUpper;
    float4 axisX;
    float4 axisY;
    float4 axisZ;
    float4 aabbHalfMargin;
    uint4 geometry;
};
