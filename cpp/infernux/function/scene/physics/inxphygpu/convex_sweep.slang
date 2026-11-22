// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "convex_penetration.slang"
#include "ray_bvh_layout.slang"
#include "triangle_mesh_geometry.slang"

static const uint CastVertices = 101, CastFaces = 1158, CastHorizon = 512;
static const uint CastWorkerRecords = 2 * CastVertices + CastFaces;

struct ShapeCastHit
{
    float4 normalDistance;
    uint4 identity;
    float4 point;
};

ConvexPolytope castWorkspace(uint worker)
{
    const uint base = worker * CastWorkerRecords;
    return {base, base + 2 * CastVertices, worker * CastHorizon, CastVertices, CastFaces, CastHorizon, 0, 0,
            float3(0.0)};
}

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
                    float3 faceNormal, uint edges, uint4 identity, ConvexPolytope workspace,
                    StructuredBuffer<uint4> geometryWords, RWStructuredBuffer<uint4> scratch,
                    RWStructuredBuffer<uint> horizon, inout ShapeCastHit best)
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
                                                        64, geometryWords, expand, true);
        ConvexPenetration metric = {seed.first, seed.second, seed.distance, NoIndex, 0};
        if (expand && step == 0)
        {
            metric = expandConvexPenetration(seed, moving, target, supportTolerance, CastVertices - 5,
                                              workspace, geometryWords, scratch, horizon);
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
        const ConvexSupportPoint a = convexSupport(moving, -outward, geometryWords);
        const ConvexSupportPoint b = convexSupport(target, outward, geometryWords);
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
              float closingLimit, uint4 identity, ConvexPolytope workspace,
                    StructuredBuffer<uint4> geometryWords, RWStructuredBuffer<uint4> scratch,
                    RWStructuredBuffer<uint> horizon, inout ShapeCastHit best)
{
    const TriangleMeshTree tree = triangleMeshTree(geometryWords, shape);
    const float3 localExtent = abs(inverseRotatePoint(frame, float3(extent.x, 0.0, 0.0))) +
                               abs(inverseRotatePoint(frame, float3(0.0, extent.y, 0.0))) +
                               abs(inverseRotatePoint(frame, float3(0.0, 0.0, extent.z)));
    const RaySlabs ray = raySlabs(inverseRotatePoint(frame, -frame.position.xyz), inverseRotatePoint(frame, direction));
    uint node = 0;
    while (node < tree.nodeCount)
    {
        const uint4 lo = geometryWords[tree.nodeBase + 2 * node];
        const uint4 hi = geometryWords[tree.nodeBase + 2 * node + 1];
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
                const uint4 indices = geometryWords[tree.faceBase + i];
                const float3 a = frame.position.xyz + rotatePoint(frame, convexVertex(geometryWords, indices.x));
                const float3 b = frame.position.xyz + rotatePoint(frame, convexVertex(geometryWords, indices.y));
                const float3 c = frame.position.xyz + rotatePoint(frame, convexVertex(geometryWords, indices.z));
                const float3 normal = normalize(cross(b - a, c - a));
                if (dot(direction, normal) > 0.0)
                    continue;
                const CartesianFrame triangle = {float4((a + b + c) / 3.0, 0.0), float4(a, 0.0),
                                                 float4(b, 0.0), float4(c, 0.0)};
                const CollisionShape triangleShape = {float4(0.0), uint4(GeomTriangle, 0, 0, 0)};
                identity.z = triangleMeshSubShape(geometryWords, tree, indices.w);
                identity.w = triangleMeshFeature(geometryWords, tree, indices.w, 2);
                const uint edges = triangleMeshFeature(geometryWords, tree, indices.w, 1);
                castConvexPair(moving, queryConvexGeometry(triangle, triangleShape), direction, closingLimit, normal,
                               edges, identity, workspace, geometryWords, scratch, horizon, best);
            }
        }
        ++node;
    }
}

void castBounds(inout ConvexGeometry shape, StructuredBuffer<uint4> geometryWords, out float3 low, out float3 high)
{
    for (uint axis = 0; axis < 3; ++axis)
    {
        float3 direction = float3(0.0);
        direction[axis] = 1.0;
        low[axis] = convexSupport(shape, -direction, geometryWords).point[axis];
        high[axis] = convexSupport(shape, direction, geometryWords).point[axis];
    }
}
