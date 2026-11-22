// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#include "convex_penetration.slang"
#include "ray_bvh_layout.slang"
#include "triangle_mesh_geometry.slang"

static const uint CastWorkers = 16, CastVertices = 101, CastFaces = 1158, CastHorizon = 512;
static const uint CastWorkerRecords = 2 * CastVertices + CastFaces;

struct ShapeCastHit
{
    float4 normalDistance;
    uint4 identity;
    float4 point;
};

[[vk::binding(0, 0)]] StructuredBuffer<CartesianFrame> castFrames;
[[vk::binding(1, 0)]] StructuredBuffer<CollisionShape> castShapes;
[[vk::binding(2, 0)]] StructuredBuffer<uint4> castGeometry;
[[vk::binding(3, 0)]] StructuredBuffer<uint4> castModel;
[[vk::binding(4, 0)]] StructuredBuffer<uint4> castRequest;
[[vk::binding(5, 0)]] RWStructuredBuffer<ShapeCastHit> castResult;
[[vk::binding(6, 0)]] RWStructuredBuffer<uint4> castScratch;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> castHorizon;

groupshared ShapeCastHit castMatches[CastWorkers];

bool castPrecedes(ShapeCastHit a, ShapeCastHit b)
{
    return a.normalDistance.w < b.normalDistance.w ||
           (a.normalDistance.w == b.normalDistance.w &&
            (a.identity.y < b.identity.y || (a.identity.y == b.identity.y && a.identity.z < b.identity.z)));
}

void acceptCast(float3 normal, float distance, float3 witness, uint4 identity, inout ShapeCastHit best)
{
    const ShapeCastHit hit = {float4(normal, distance), identity, float4(witness, best.point.w)};
    if (castPrecedes(hit, best))
        best = hit;
}

void castSpheres(float radius, float3 target, float targetRadius, float3 direction, float closingLimit,
                 uint4 identity, inout ShapeCastHit best)
{
    const float combined = radius + targetRadius;
    float distance = 0.0;
    if (dot(target, target) > combined * combined)
    {
        const float along = dot(target, direction);
        const float3 closest = target - along * direction;
        const float discriminant = combined * combined - dot(closest, closest);
        if (along <= 0.0 || discriminant < 0.0)
            return;
        distance = along - sqrt(discriminant);
        if (distance > best.normalDistance.w)
            return;
    }
    const float3 radial = distance * direction - target;
    const float3 normal = dot(radial, radial) > 0.0 ? normalize(radial) : -direction;
    const float closing = -dot(normal, direction);
    if (closingLimit > 0.0 ? closing > closingLimit : closing >= 0.0)
        acceptCast(normal, distance, target + targetRadius * normal, identity, best);
}

void castConvexPair(ConvexGeometry moving, ConvexGeometry target, float3 direction, float closingLimit,
                    float3 faceNormal, uint edges, uint4 identity, ConvexPolytope workspace, inout ShapeCastHit best)
{
    if (moving.shape.type.x == GeomSphere && target.shape.type.x == GeomSphere)
    {
        castSpheres(moving.shape.size.x, target.frame.position.xyz, target.shape.size.x, direction, closingLimit,
                    identity, best);
        return;
    }
    const float supportTolerance = 1.0e-6;
    const float contactTolerance = 1.0e-5;
    float travel = 0.0;
    float3 outward = -direction;
    float3 witness = float3(0.0);
    bool reached = false;
    for (uint step = 0; step < 64; ++step)
    {
        moving.frame.position.xyz = direction * travel;
        bool expand;
        const ConvexDistance seed = convexDistancePhase(moving, target, supportTolerance, ConvexFarDistance,
                                                        64, castModel, expand, true);
        ConvexPenetration metric = {seed.first, seed.second, seed.distance, NoIndex, 0};
        if (expand && step == 0)
        {
            metric = expandConvexPenetration(seed, moving, target, supportTolerance, CastVertices - 5,
                                              workspace, castModel, castScratch, castHorizon);
            if (metric.status == EpaCapacity || metric.status == EpaDegenerate)
            {
                best.point.w = 1.0;
                return;
            }
        }

        const float3 delta = metric.first - metric.second;
        const float squared = dot(delta, delta);
        if (squared > supportTolerance * supportTolerance)
            outward = delta * ((metric.distance < 0.0 ? -1.0 : 1.0) * rsqrt(squared));
        if (metric.distance <= contactTolerance)
        {
            witness = metric.second;
            reached = true;
            break;
        }

        // This support plane bounds distance from below. Its projected travel
        // keeps the cast on the approach side even with an approximate simplex.
        const ConvexSupportPoint a = convexSupport(moving, -outward, castModel);
        const ConvexSupportPoint b = convexSupport(target, outward, castModel);
        const float gap = dot(outward, a.point - b.point);
        const float closing = -dot(outward, direction);
        if (closing <= 0.0)
            return;
        const float next = travel + max(0.0, gap) / closing;
        if (next > best.normalDistance.w)
            return;
        if (next <= travel)
        {
            best.point.w = 1.0;
            return;
        }
        travel = next;
    }
    if (!reached)
    {
        best.point.w = 1.0;
        return;
    }
    if (target.shape.type.x == GeomTriangle)
        outward = dot(outward, faceNormal) < 0.0 ? faceNormal :
            meshContactNormal(target.frame.row0.xyz, target.frame.row1.xyz, target.frame.row2.xyz, witness, outward,
                              faceNormal, edges);
    const float closing = -dot(outward, direction);
    if (closingLimit > 0.0 ? closing > closingLimit : closing >= 0.0)
        acceptCast(outward, travel, witness, identity, best);
}

void castMesh(ConvexGeometry moving, CartesianFrame frame, CollisionShape shape, float3 extent, float3 direction,
              float closingLimit, uint4 identity, ConvexPolytope workspace, inout ShapeCastHit best)
{
    const TriangleMeshTree tree = triangleMeshTree(castModel, shape);
    const float3 localExtent = abs(inverseRotatePoint(frame, float3(extent.x, 0.0, 0.0))) +
                               abs(inverseRotatePoint(frame, float3(0.0, extent.y, 0.0))) +
                               abs(inverseRotatePoint(frame, float3(0.0, 0.0, extent.z)));
    const RaySlabs ray = raySlabs(inverseRotatePoint(frame, -frame.position.xyz), inverseRotatePoint(frame, direction));
    uint node = 0;
    while (node < tree.nodeCount)
    {
        const uint4 lo = castModel[tree.nodeBase + 2 * node];
        const uint4 hi = castModel[tree.nodeBase + 2 * node + 1];
        const bool crosses = rayBoundsOverlap(ray, asfloat(lo.xyz) - localExtent, asfloat(hi.xyz) + localExtent,
                                               best.normalDistance.w);
        if ((hi.w & BvhLeaf) == 0)
        {
            node = crosses ? lo.w : hi.w;
            continue;
        }
        if (crosses)
        {
            const uint end = lo.w + (hi.w & ~BvhLeaf);
            for (uint i = lo.w; i < end; ++i)
            {
                const uint4 indices = castModel[tree.faceBase + i];
                const float3 a = frame.position.xyz + rotatePoint(frame, convexVertex(castModel, indices.x));
                const float3 b = frame.position.xyz + rotatePoint(frame, convexVertex(castModel, indices.y));
                const float3 c = frame.position.xyz + rotatePoint(frame, convexVertex(castModel, indices.z));
                const float3 normal = normalize(cross(b - a, c - a));
                if (dot(direction, normal) > 0.0)
                    continue;
                const CartesianFrame triangle = {float4((a + b + c) / 3.0, 0.0), float4(a, 0.0),
                                                 float4(b, 0.0), float4(c, 0.0)};
                const CollisionShape triangleShape = {float4(0.0), uint4(GeomTriangle, 0, 0, 0)};
                identity.z = triangleMeshSubShape(castModel, tree, indices.w);
                identity.w = triangleMeshFeature(castModel, tree, indices.w, 2);
                const uint edges = triangleMeshFeature(castModel, tree, indices.w, 1);
                castConvexPair(moving, queryConvexGeometry(triangle, triangleShape), direction, closingLimit, normal,
                               edges, identity, workspace, best);
            }
        }
        ++node;
    }
}

void castBounds(inout ConvexGeometry shape, out float3 low, out float3 high)
{
    for (uint axis = 0; axis < 3; ++axis)
    {
        float3 direction = float3(0.0);
        direction[axis] = 1.0;
        low[axis] = convexSupport(shape, -direction, castModel).point[axis];
        high[axis] = convexSupport(shape, direction, castModel).point[axis];
    }
}

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
    castBounds(moving, minimum, maximum);
    const float3 extent = max(abs(minimum), abs(maximum));
    ShapeCastHit best = {float4(0.0, 0.0, 0.0, start.w), uint4(NoIndex), float4(0.0)};
    const uint base = lane * CastWorkerRecords;
    const ConvexPolytope workspace = {base, base + 2 * CastVertices, lane * CastHorizon,
                                     CastVertices, CastFaces, CastHorizon, 0, 0, float3(0.0)};
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
            castMesh(moving, frame, shape, extent, motion.xyz, motion.w, identity, workspace, best);
        else
        {
            ConvexGeometry target = queryConvexGeometry(frame, shape);
            float3 low, high;
            castBounds(target, low, high);
            if (rayBoundsOverlap(ray, low - extent, high + extent, best.normalDistance.w))
                castConvexPair(moving, target, motion.xyz, motion.w, float3(0.0), 0u, identity, workspace, best);
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
