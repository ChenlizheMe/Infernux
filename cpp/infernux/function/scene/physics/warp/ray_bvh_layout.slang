// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

static const uint BvhLeaf = 0x80000000u;

struct RayBvhBounds
{
    float4 low;
    float4 high;
};

uint rayIndexAt(StructuredBuffer<uint4> records, uint offset, uint index)
{
    return records[offset + index / 4u][index % 4u];
}

struct RaySlabs
{
    float3 origin;
    float3 inverse;
    bool3 parallel;
};

RaySlabs raySlabs(float3 origin, float3 direction)
{
    const bool3 invertible = abs(direction) >= asfloat(0x00800000u);
    const float3 inverse = 1.0 / select(invertible, direction, float3(1.0));
    return {origin, select(invertible, inverse, float3(0.0)), direction == float3(0.0)};
}

bool rayBoundsOverlap(RaySlabs ray, float3 low, float3 high, float limit)
{
    const bool3 outside = (ray.origin < low) | (ray.origin > high);
    if (any(ray.parallel & outside))
        return false;

    const float3 a = (low - ray.origin) * ray.inverse;
    const float3 b = (high - ray.origin) * ray.inverse;
    const bool3 open = ray.inverse == 0.0;
    const float3 first = select(open, float3(0.0), min(a, b));
    const float3 last = select(open, float3(limit), max(a, b));
    const float entry = max(first.x, max(first.y, first.z));
    const float exit = min(last.x, min(last.y, last.z));
    // Round the interval outward to keep edge hits on both Vulkan and WebGPU.
    const float rounding = 3.5762787e-7;
    return entry * (1.0 - rounding) <= min(limit, exit * (1.0 + rounding)) && exit >= 0.0;
}
