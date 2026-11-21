// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Adapted from MuJoCo Warp _src/collision_convex.py and _src/ray.py.

struct HeightfieldGrid
{
    uint4 layout;
    float4 size;
};

HeightfieldGrid heightfieldGrid(StructuredBuffer<uint4> model, uint field)
{
    const uint offset = model[3].x + 2u * field;
    return {model[offset], asfloat(model[offset + 1u])};
}

float heightfieldElevation(StructuredBuffer<uint4> model, HeightfieldGrid grid, uint2 cell)
{
    const uint sample = grid.layout.z + cell.y * grid.layout.y + cell.x;
    return asfloat(model[model[3].y + sample / 4u][sample % 4u]);
}

float3 heightfieldVertex(StructuredBuffer<uint4> model, HeightfieldGrid grid, uint2 cell, float2 step, float margin)
{
    return float3(float2(cell) * step - grid.size.xy, heightfieldElevation(model, grid, cell) * grid.size.z + margin);
}
