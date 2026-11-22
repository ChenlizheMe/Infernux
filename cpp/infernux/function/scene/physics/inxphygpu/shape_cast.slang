// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#include "convex_sweep.slang"

static const uint CastWorkers = 16;

[[vk::binding(0, 0)]] StructuredBuffer<CartesianFrame> castFrames;
[[vk::binding(1, 0)]] StructuredBuffer<CollisionShape> castShapes;
[[vk::binding(2, 0)]] StructuredBuffer<uint4> castGeometry;
[[vk::binding(3, 0)]] StructuredBuffer<uint4> castModel;
[[vk::binding(4, 0)]] StructuredBuffer<uint4> castRequest;
[[vk::binding(5, 0)]] RWStructuredBuffer<ShapeCastHit> castResult;
[[vk::binding(6, 0)]] RWStructuredBuffer<uint4> castScratch;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> castHorizon;

groupshared ShapeCastHit castMatches[CastWorkers];

[shader("compute")]
[numthreads(CastWorkers, 1, 1)]
void cast_scene_shapes(uint lane : SV_GroupIndex)
{
    const float4 start = asfloat(castRequest[0]);
    const float4 motion = asfloat(castRequest[3]);
    const uint4 filter = castRequest[4]; // count, layers, flags, shape type.
    const uint identities = 5u + (filter.x - 1u) / 128u + 1u;
    const CollisionShape probe = {asfloat(castRequest[2]), uint4(filter.w, 0, 0, 0)};
    ConvexGeometry moving = queryConvexGeometry(cartesianFrame(float3(0.0), asfloat(castRequest[1])), probe);
    float3 minimum, maximum;
    castBounds(moving, castModel, minimum, maximum);
    const float3 extent = max(abs(minimum), abs(maximum));
    ShapeCastHit best = {float4(0.0, 0.0, 0.0, start.w), uint4(NoIndex), float4(0.0)};
    const ConvexPolytope workspace = castWorkspace(lane);
    const RaySlabs ray = raySlabs(float3(0.0), motion.xyz);
    for (uint row = lane; row < filter.x; row += CastWorkers)
    {
        const uint4 metadata = castGeometry[row];
        if ((metadata.y & filter.y) == 0u || ((filter.z & 1u) == 0u && (metadata.z & 2u) != 0u))
            continue;
        if ((filter.z & 2u) != 0u &&
            (castRequest[5 + row / 128u][(row / 32u) % 4u] & (1u << (row % 32u))) != 0u)
            continue;
        const uint body = castRequest[identities + row / 4u][row % 4u];
        const uint4 identity = uint4(row, body, metadata.w, NoIndex);
        const CollisionShape shape = castShapes[row];
        CartesianFrame frame = castFrames[row];
        frame.position.xyz -= start.xyz;
        if (shape.type.x == GeomTriangleMesh)
            castMesh(moving, frame, shape, extent, motion.xyz, motion.w, identity, workspace,
                     castModel, castScratch, castHorizon, best);
        else
        {
            ConvexGeometry target = queryConvexGeometry(frame, shape);
            float3 low, high;
            castBounds(target, castModel, low, high);
            if (rayBoundsOverlap(ray, low - extent, high + extent, best.normalDistance.w))
                castConvexPair(moving, target, motion.xyz, motion.w, float3(0.0), 0u, identity,
                               workspace, castModel, castScratch, castHorizon, best);
        }
    }
    castMatches[lane] = best;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = CastWorkers / 2; stride != 0; stride /= 2)
    {
        if (lane < stride)
        {
            const ShapeCastHit other = castMatches[lane + stride];
            const float failed = max(best.point.w, other.point.w);
            if (castPrecedes(other, best))
                best = other;
            best.point.w = failed;
            castMatches[lane] = best;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0)
    {
        best.point.xyz += start.xyz;
        castResult[0] = best;
    }
}
