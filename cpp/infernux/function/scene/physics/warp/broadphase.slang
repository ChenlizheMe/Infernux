// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/collision_driver.py and _src/collision_core.py sweep and prune.

#include "model.slang"

static const float UnboundedRadius = 1.0e10;
static const uint FilterPlane = 1;
static const uint FilterSphere = 2;
static const uint FilterAabb = 4;
static const uint FilterObb = 8;

struct LocalBound
{
    float4 center;
    float4 halfExtent;
    float4 radii; // bounding radius, margin + gap, maximum explicit pair margin + gap
    uint4 geometry; // MuJoCo geometry type
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

struct SortKey
{
    float lower;
    uint geometry;
};

struct CollisionPairRule
{
    int pair;
    int sensor;
    float margin;
    uint unused;
};

[[vk::binding(0, 0)]] StructuredBuffer<ModelDimensions> model;
[[vk::binding(1, 0)]] StructuredBuffer<uint4> sweepSettings; // merge/scan stride, capacity per world, filter, workers
[[vk::binding(2, 0)]] StructuredBuffer<LocalBound> localBounds;
[[vk::binding(3, 0)]] StructuredBuffer<CartesianFrame> geomFrame;
[[vk::binding(4, 0)]] RWStructuredBuffer<ProjectedBound> worldBounds;
[[vk::binding(5, 0)]] StructuredBuffer<SortKey> sortInput;
[[vk::binding(6, 0)]] RWStructuredBuffer<SortKey> sortOutput;
[[vk::binding(7, 0)]] StructuredBuffer<uint> scanInput;
[[vk::binding(8, 0)]] RWStructuredBuffer<uint> scanOutput;
[[vk::binding(9, 0)]] StructuredBuffer<CollisionPairRule> pairRules;
[[vk::binding(10, 0)]] RWStructuredBuffer<Atomic<uint>> collisionCount;
[[vk::binding(11, 0)]] RWStructuredBuffer<uint4> collisionPair; // geoms, explicit pair id, sensor id

[shader("compute")]
[numthreads(64, 1, 1)]
void sap_project(uint3 id : SV_DispatchThreadID)
{
    const uint count = model[0].layout.x;
    if (id.x >= count)
        return;
    const uint index = id.y * count + id.x;
    const CartesianFrame frame = geomFrame[index];
    const LocalBound shape = localBounds[id.x];
    const float3 x = float3(frame.row0.x, frame.row1.x, frame.row2.x);
    const float3 y = float3(frame.row0.y, frame.row1.y, frame.row2.y);
    const float3 z = float3(frame.row0.z, frame.row1.z, frame.row2.z);
    const float3 h = shape.halfExtent.xyz;
    const float3 center = frame.position.xyz + x * shape.center.x + y * shape.center.y + z * shape.center.z;
    const float3 halfExtent = abs(x) * h.x + abs(y) * h.y + abs(z) * h.z;
    const float radius = shape.radii.x == 0.0 ? UnboundedRadius : shape.radii.x;
    const float margin = shape.radii.y;
    const float projectedRadius = radius + max(margin, shape.radii.z);
    const float projection = dot(normalize(float3(0.5935, 0.7790, 0.1235)), frame.position.xyz);
    const float lower = isnan(projection) ? UnboundedRadius : projection - projectedRadius;
    const float upper = isnan(projection) ? UnboundedRadius : projection + projectedRadius;
    ProjectedBound bound;
    bound.originRadius = float4(frame.position.xyz, shape.radii.x);
    bound.centerUpper = float4(center, upper);
    bound.axisX = float4(x, h.x);
    bound.axisY = float4(y, h.y);
    bound.axisZ = float4(z, h.z);
    bound.aabbHalfMargin = float4(halfExtent, margin);
    bound.geometry = shape.geometry;
    worldBounds[index] = bound;
    sortOutput[index] = {lower, id.x};
}

bool precedes(SortKey a, SortKey b)
{
    return a.lower < b.lower || (a.lower == b.lower && a.geometry < b.geometry);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sap_merge(uint3 id : SV_DispatchThreadID)
{
    const uint count = model[0].layout.x;
    if (id.x >= count)
        return;
    const uint base = id.y * count;
    const uint width = sweepSettings[0].x;
    const uint left = id.x / (2 * width) * (2 * width);
    const uint mid = min(left + width, count);
    const uint right = min(mid + width, count);
    const bool inLeft = id.x < mid;
    const SortKey key = sortInput[base + id.x];
    const uint other = inLeft ? mid : left;
    uint lo = other;
    uint hi = inLeft ? right : mid;
    while (lo < hi)
    {
        const uint split = (lo + hi) / 2;
        if (precedes(sortInput[base + split], key))
            lo = split + 1;
        else
            hi = split;
    }
    const uint ownRank = id.x - (inLeft ? left : mid);
    sortOutput[base + left + ownRank + lo - other] = key;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sap_find_ranges(uint3 id : SV_DispatchThreadID)
{
    const uint count = model[0].layout.x;
    if (id.x >= count)
        return;
    const uint base = id.y * count;
    const float upper = worldBounds[base + sortInput[base + id.x].geometry].centerUpper.w;
    uint lo = id.x + 1;
    uint hi = count;
    while (lo < hi)
    {
        const uint split = (lo + hi) / 2;
        if (sortInput[base + split].lower > upper)
            hi = split;
        else
            lo = split + 1;
    }
    const uint range = min(count - 1, lo) - id.x;
    scanOutput[base + id.x] = range;
    if (id.x == 0)
        collisionCount[id.y].store(0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sap_scan_ranges(uint3 id : SV_DispatchThreadID)
{
    const uint count = model[0].layout.x;
    if (id.x >= count)
        return;
    const uint index = id.y * count + id.x;
    const uint stride = sweepSettings[0].x;
    const uint prefix = id.x < stride ? 0 : scanInput[index - stride];
    scanOutput[index] = scanInput[index] + prefix;
}

float obbRadius(ProjectedBound shape, float3 axis)
{
    return abs(dot(shape.axisX.xyz, axis)) * shape.axisX.w + abs(dot(shape.axisY.xyz, axis)) * shape.axisY.w +
           abs(dot(shape.axisZ.xyz, axis)) * shape.axisZ.w;
}

bool separated(ProjectedBound a, ProjectedBound b, float3 offset, float3 axis, float margin)
{
    return obbRadius(a, axis) + obbRadius(b, axis) + margin < abs(dot(offset, axis));
}

bool boundsOverlap(ProjectedBound a, ProjectedBound b, CollisionPairRule rule, uint filters)
{
    const float margin = rule.pair >= 0 ? rule.margin : a.aabbHalfMargin.w + b.aabbHalfMargin.w;
    const float radiusA = a.originRadius.w;
    const float radiusB = b.originRadius.w;
    const float3 offset = b.originRadius.xyz - a.originRadius.xyz;
    if (radiusA == 0.0 || radiusB == 0.0)
    {
        if ((filters & FilterPlane) == 0)
            return true;
        return radiusA == 0.0 ? dot(offset, a.axisZ.xyz) <= radiusB + margin :
                                dot(-offset, b.axisZ.xyz) <= radiusA + margin;
    }
    const float radius = radiusA + radiusB + margin;
    if ((filters & FilterSphere) != 0 && dot(offset, offset) > radius * radius)
        return false;
    const float3 centerOffset = b.centerUpper.xyz - a.centerUpper.xyz;
    if ((filters & FilterAabb) != 0 &&
        any(abs(centerOffset) > a.aabbHalfMargin.xyz + b.aabbHalfMargin.xyz + margin))
        return false;
    if ((filters & FilterObb) != 0)
    {
        return !(separated(a, b, centerOffset, a.axisX.xyz, margin) ||
                 separated(a, b, centerOffset, a.axisY.xyz, margin) ||
                 separated(a, b, centerOffset, a.axisZ.xyz, margin) ||
                 separated(a, b, centerOffset, b.axisX.xyz, margin) ||
                 separated(a, b, centerOffset, b.axisY.xyz, margin) ||
                 separated(a, b, centerOffset, b.axisZ.xyz, margin));
    }
    return true;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sap_emit_pairs(uint3 id : SV_DispatchThreadID)
{
    const uint4 settings = sweepSettings[0];
    const uint count = model[0].layout.x;
    const uint base = id.y * count;
    const uint workCount = scanInput[base + count - 1];
    if (id.x >= settings.w)
        return;
    for (uint work = id.x; work < workCount; work += settings.w)
    {
        uint lo = 0;
        uint hi = count;
        while (lo < hi)
        {
            const uint split = (lo + hi) / 2;
            if (scanInput[base + split] > work)
                hi = split;
            else
                lo = split + 1;
        }
        const uint previous = lo == 0 ? 0 : scanInput[base + lo - 1];
        const uint first = sortInput[base + lo].geometry;
        const uint second = sortInput[base + lo + work - previous + 1].geometry;
        const uint a = min(first, second);
        const uint b = max(first, second);
        const uint pairIndex = a * (2 * count - 1 - a) / 2 + b - a - 1;
        const CollisionPairRule rule = pairRules[pairIndex];
        if (rule.pair < -1 && rule.sensor < 0)
            continue;
        const ProjectedBound one = worldBounds[base + first];
        const ProjectedBound two = worldBounds[base + second];
        if (rule.sensor < 0 && !boundsOverlap(one, two, rule, settings.z))
            continue;
        const uint destination = collisionCount[id.y].add(1);
        if (destination < settings.y)
        {
            const uint2 pair = one.geometry.x <= two.geometry.x ? uint2(first, second) : uint2(second, first);
            collisionPair[id.y * settings.y + destination] = uint4(pair, asuint(rule.pair), asuint(rule.sensor));
        }
    }
}
