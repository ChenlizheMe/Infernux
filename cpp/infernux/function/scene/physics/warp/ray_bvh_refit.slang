// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Scene BVH refit adapted from MuJoCo Warp _src/bvh.py.

#include "collision_bounds.slang"
#include "collision_geometry.slang"
#include "ray_bvh_layout.slang"

static const uint RefitGroupSize = 64;

[[vk::binding(0, 0)]] StructuredBuffer<uint4> refitSettings; // geometry count and first batch
[[vk::binding(1, 0)]] StructuredBuffer<LocalBound> refitGeometry;
[[vk::binding(2, 0)]] StructuredBuffer<uint4> refitIndex;
[[vk::binding(3, 0)]] RWStructuredBuffer<RayBvhBounds> refitBounds;
[[vk::binding(4, 0)]] StructuredBuffer<CartesianFrame> refitFrames;

RayBvhBounds refitLeaf(uint first, uint count, uint order, uint geomBase)
{
    float3 low = float3(3.402823e38);
    float3 high = -low;
    for (uint i = first; i < first + count; ++i)
    {
        const uint geometry = rayIndexAt(refitIndex, order, i);
        const LocalBound shape = refitGeometry[geometry];
        const CartesianFrame frame = refitFrames[geomBase + geometry];
        const float3 half = shape.halfExtent.xyz;
        const float3 center = frame.position.xyz + rotatePoint(frame, shape.center.xyz);
        const float3 extent = abs(frameAxis(frame, 0)) * half.x +
                              abs(frameAxis(frame, 1)) * half.y + abs(frameAxis(frame, 2)) * half.z;
        const float3 padded = extent + (abs(center) + extent) * 4.7683716e-7;
        low = min(low, center - padded);
        high = max(high, center + padded);
    }
    return {float4(low, 0.0), float4(high, 0.0)};
}

void syncRefit()
{
    __target_switch
    {
    case wgsl: __intrinsic_asm "storageBarrier()";
    default: DeviceMemoryBarrierWithGroupSync();
    }
}

[shader("compute")]
[numthreads(RefitGroupSize, 1, 1)]
void refit_ray_bvh(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    const uint4 settings = refitSettings[0];
    const uint4 index = refitIndex[0];
    const uint4 plan = refitIndex[1];
    const uint2 batch = refitIndex[plan.y + settings.y + group.x].xy;
    const uint boundsBase = group.y * plan.x;
    const uint geomBase = group.y * settings.x;
    for (uint level = 0; level < batch.y; ++level)
    {
        const uint2 work = refitIndex[batch.x + level].xy;
        for (uint item = lane; item < work.y; item += RefitGroupSize)
        {
            const uint node = rayIndexAt(refitIndex, work.x, item);
            const uint2 links = refitIndex[index.x + node].xy;
            if ((links.y & BvhLeaf) != 0)
            {
                refitBounds[boundsBase + node] = refitLeaf(links.x, links.y & ~BvhLeaf, index.y, geomBase);
            }
            else
            {
                const RayBvhBounds left = refitBounds[boundsBase + links.x];
                const RayBvhBounds right = refitBounds[boundsBase + links.y];
                refitBounds[boundsBase + node] = {min(left.low, right.low), max(left.high, right.high)};
            }
        }
        syncRefit();
    }
}
