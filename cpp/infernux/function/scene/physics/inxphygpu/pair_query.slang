// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#include "convex_penetration.slang"
#include "triangle_mesh_geometry.slang"

[[vk::binding(0, 0)]] StructuredBuffer<uint4> pairModel;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint4> pairScratch;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> pairHorizon;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> pairResult;

ConvexGeometry pairOperand(uint address)
{
    const CollisionShape shape = {asfloat(pairModel[address]), pairModel[address + 1]};
    const CartesianFrame frame = cartesianFrame(asfloat(pairModel[address + 2].xyz), asfloat(pairModel[address + 3]));
    return {frame, shape, 0.0, -1};
}

bool projectPairTriangle(inout ConvexGeometry geometry, float3 otherCenter)
{
    if (geometry.shape.type.x != GeomTriangleMesh)
        return true;
    const TriangleMeshTree tree = triangleMeshTree(pairModel, geometry.shape);
    const uint3 indices = pairModel[tree.faceBase].xyz;
    const float3 origin = geometry.frame.position.xyz;
    const float3 a = origin + rotatePoint(geometry.frame, convexVertex(pairModel, indices.x));
    const float3 b = origin + rotatePoint(geometry.frame, convexVertex(pairModel, indices.y));
    const float3 c = origin + rotatePoint(geometry.frame, convexVertex(pairModel, indices.z));
    const float3 face = cross(b - a, c - a);
    geometry.frame = {float4((a + b + c) / 3.0, 0.0), float4(a, 0.0), float4(b, 0.0), float4(c, 0.0)};
    geometry.shape.type.x = GeomTriangle;
    return dot(otherCenter - a, face) >= 0.0;
}

void storePenetration(float3 normal, float depth, float3 first, float3 second)
{
    pairResult[0] = float4(normal, depth);
    pairResult[1] = float4(first, 0.0);
    pairResult[2] = float4(second, 0.0);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void compute_penetration()
{
    storePenetration(float3(0.0), 0.0, float3(0.0), float3(0.0));
    const uint operands = pairModel[0].x;
    ConvexGeometry first = pairOperand(operands);
    ConvexGeometry second = pairOperand(operands + 4);
    const float3 origin = first.frame.position.xyz;
    first.frame.position.xyz = float3(0.0);
    second.frame.position.xyz -= origin;
    if (!projectPairTriangle(first, second.frame.position.xyz) ||
        !projectPairTriangle(second, first.frame.position.xyz))
        return;
    if (first.shape.type.x == GeomSphere && second.shape.type.x == GeomSphere)
    {
        const float3 delta = second.frame.position.xyz;
        const float lengthToSecond = length(delta);
        const float radiusA = first.shape.size.x;
        const float radiusB = second.shape.size.x;
        const float depth = radiusA + radiusB - lengthToSecond;
        if (depth > 0.0)
        {
            const float3 toward = lengthToSecond > 0.0 ? delta / lengthToSecond : float3(1.0, 0.0, 0.0);
            storePenetration(-toward, depth, origin + toward * radiusA, origin + delta - toward * radiusB);
        }
        return;
    }
    bool expand;
    const float tolerance = 1.0e-6;
    const ConvexDistance seed = convexDistancePhase(first, second, tolerance, 0.0, 64, pairModel, expand, true);
    ConvexPenetration result = {seed.first, seed.second, seed.distance, NoIndex, 0};
    if (expand)
    {
        uint scratchRows, horizonRows, stride;
        pairScratch.GetDimensions(scratchRows, stride);
        pairHorizon.GetDimensions(horizonRows, stride);
        const uint vertices = 101u;
        const ConvexPolytope workspace = {0, 2 * vertices, 0, vertices, scratchRows - 2 * vertices,
                                          horizonRows, 0, 0, float3(0.0)};
        result = expandConvexPenetration(seed, first, second, tolerance, vertices - 5, workspace, pairModel,
                                         pairScratch, pairHorizon);
    }
    if (result.status == EpaCapacity || result.status == EpaDegenerate)
        pairResult[0].w = -1.0;
    if (result.distance < 0.0)
        storePenetration(normalize(result.second - result.first), -result.distance,
                         result.first + origin, result.second + origin);
}
