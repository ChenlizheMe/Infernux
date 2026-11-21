// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Adapted from MuJoCo Warp _src/collision_gjk.py, EPA polytope storage and seeding.

#include "convex_status.slang"

static const uint EpaVertexMask = 0x3ffu;
static const uint EpaDeletedFace = 0x80000000u;
static const uint EpaInvalidFace = 0x40000000u;

struct ConvexPolytope
{
    uint vertexBase;
    uint faceBase;
    uint horizonBase;
    uint vertexCapacity;
    uint faceCapacity;
    uint horizonCapacity;
    uint vertexCount;
    uint faceCount;
    float3 center;
};

uint packEpaFace(uint3 vertices)
{
    return vertices.x | (vertices.y << 10) | (vertices.z << 20);
}

uint3 unpackEpaFace(uint face)
{
    return (uint3(face, face >> 10, face >> 20) & EpaVertexMask);
}

void storeEpaVertex(ConvexPolytope polytope, uint index, ConvexSimplexVertex vertex, RWStructuredBuffer<uint4> scratch)
{
    const uint base = polytope.vertexBase + 2 * index;
    scratch[base] = uint4(asuint(vertex.first), asuint(vertex.indices.x));
    scratch[base + 1] = uint4(asuint(vertex.second), asuint(vertex.indices.y));
}

ConvexSimplexVertex loadEpaVertex(ConvexPolytope polytope, uint index, RWStructuredBuffer<uint4> scratch)
{
    const uint base = polytope.vertexBase + 2 * index;
    const uint4 a = scratch[base];
    const uint4 b = scratch[base + 1];
    return {asfloat(a.xyz), asfloat(b.xyz), int2(asint(a.w), asint(b.w))};
}

float3 epaPoint(ConvexPolytope polytope, uint index, RWStructuredBuffer<uint4> scratch)
{
    const uint base = polytope.vertexBase + 2 * index;
    return asfloat(scratch[base].xyz) - asfloat(scratch[base + 1].xyz);
}

float3 sampleEpaVertex(ConvexPolytope polytope, uint index, ConvexGeometry first, ConvexGeometry second,
                       float3 direction, StructuredBuffer<uint4> model,
                       RWStructuredBuffer<uint4> scratch, out int2 cached)
{
    const ConvexSupportPoint a = convexSupport(first, direction, model);
    const ConvexSupportPoint b = convexSupport(second, -direction, model);
    cached = int2(a.cachedVertex, b.cachedVertex);
    const ConvexSimplexVertex vertex = {a.point, b.point, int2(a.vertex, b.vertex)};
    storeEpaVertex(polytope, index, vertex, scratch);
    return a.point - b.point;
}

float attachEpaFace(inout ConvexPolytope polytope, uint3 vertices, RWStructuredBuffer<uint4> scratch)
{
    if (polytope.faceCount == polytope.faceCapacity)
        return -1.0;
    const float3 a = epaPoint(polytope, vertices.x, scratch);
    const float3 b = epaPoint(polytope, vertices.y, scratch);
    const float3 c = epaPoint(polytope, vertices.z, scratch);
    float3 projection;
    if (!projectConvexPlane(c, b, a, projection))
        return 0.0;
    if (dot(projection, a - polytope.center) < 0.0)
        projection = -projection;
    scratch[polytope.faceBase + polytope.faceCount++] = uint4(asuint(projection), packEpaFace(vertices));
    return dot(projection, projection);
}

bool sameEpaSide(float3 a, float3 b, float3 c, float3 d)
{
    const float3 normal = cross(b - a, c - a);
    return sameConvexSign(dot(normal, d - a), dot(normal, -a));
}

bool epaTetrahedronContainsOrigin(float3 a, float3 b, float3 c, float3 d)
{
    return sameEpaSide(a,b,c,d) && sameEpaSide(b,c,d,a) && sameEpaSide(c,d,a,b) && sameEpaSide(d,a,b,c);
}

bool epaOnTriangle(float3 a, float3 b, float3 c, float3 point)
{
    float3 weights;
    if (!convexTriangleCoordinates(a,b,c,point,weights) || any(weights < float3(0.0)))
        return false;
    return length(weights.x * a + weights.y * b + weights.z * c - point) < MinValue;
}

void epaSelectTriangle(inout ConvexPolytope polytope, uint3 face, RWStructuredBuffer<uint4> scratch)
{
    const ConvexSimplexVertex a = loadEpaVertex(polytope, face.x, scratch);
    const ConvexSimplexVertex b = loadEpaVertex(polytope, face.y, scratch);
    const ConvexSimplexVertex c = loadEpaVertex(polytope, face.z, scratch);
    storeEpaVertex(polytope, 0, a, scratch);
    storeEpaVertex(polytope, 1, b, scratch);
    storeEpaVertex(polytope, 2, c, scratch);
    polytope.vertexCount = 3;
    polytope.faceCount = 0;
}

uint seedEpaTriangle(inout ConvexPolytope polytope, float distance, ConvexGeometry first, ConvexGeometry second,
                     StructuredBuffer<uint4> model, RWStructuredBuffer<uint4> scratch)
{
    const float3 a = epaPoint(polytope, 0, scratch);
    const float3 b = epaPoint(polytope, 1, scratch);
    const float3 c = epaPoint(polytope, 2, scratch);
    polytope.center = (a + b + c) / 3.0;
    const float3 normal = cross(b - a, c - a);
    const float norm = length(normal);
    if (norm < MinValue)
        return EpaBoundary;
    const float3 direction = normal / norm;
    int2 cached;
    const float3 d = sampleEpaVertex(polytope, 3, first, second, -direction, model, scratch, cached);
    const float3 e = sampleEpaVertex(polytope, 4, first, second, direction, model, scratch, cached);
    if (epaOnTriangle(a,b,c,d) || epaOnTriangle(a,b,c,e))
        return EpaBoundary;
    if (distance > 1.0e-5 && !epaTetrahedronContainsOrigin(a,b,c,d) && !epaTetrahedronContainsOrigin(a,b,c,e))
        return EpaBoundary;
    const uint3 faces[6] = {uint3(4,0,1), uint3(4,2,0), uint3(4,1,2), uint3(3,1,0), uint3(3,0,2), uint3(3,2,1)};
    for (uint i = 0; i < 6; ++i)
    {
        if (attachEpaFace(polytope, faces[i], scratch) < 1.0e-10)
            return EpaBoundary;
    }
    polytope.vertexCount = 5;
    return 0;
}

uint seedEpaSegment(inout ConvexPolytope polytope, float distance, ConvexGeometry first, ConvexGeometry second,
                    StructuredBuffer<uint4> model, RWStructuredBuffer<uint4> scratch)
{
    const float3 a = epaPoint(polytope, 0, scratch);
    const float3 b = epaPoint(polytope, 1, scratch);
    polytope.center = 0.5 * (a + b);
    const float3 edge = b - a;
    const float edgeLength = length(edge);
    if (edgeLength < MinValue)
        return EpaBoundary;
    const float3 magnitude = abs(edge);
    const uint least = magnitude.x <= magnitude.y && magnitude.x <= magnitude.z ? 0 :
                       magnitude.y <= magnitude.z ? 1 : 2;
    float3 basis = float3(0.0);
    basis[least] = 1.0;
    const float3 axis = edge / edgeLength;
    const float3 direction = normalize(cross(basis, edge));
    const float3 tangent = cross(axis, direction) * 0.86602540378;
    int2 cached;
    const float3 c = sampleEpaVertex(polytope, 2, first, second, direction, model, scratch, cached);
    const float3 d = sampleEpaVertex(polytope, 3, first, second, tangent - 0.5 * direction,
                                    model, scratch, cached);
    const float3 e = sampleEpaVertex(polytope, 4, first, second, -tangent - 0.5 * direction,
                                    model, scratch, cached);
    const uint3 faces[6] = {uint3(0,2,3), uint3(0,4,2), uint3(0,3,4), uint3(1,3,2), uint3(1,2,4), uint3(1,4,3)};
    for (uint i = 0; i < 6; ++i)
    {
        if (attachEpaFace(polytope, faces[i], scratch) < 1.0e-10)
        {
            epaSelectTriangle(polytope, faces[i], scratch);
            return seedEpaTriangle(polytope, distance, first, second, model, scratch);
        }
    }
    const float3 signs = float3(dot(c-a, cross(d-a, edge)), dot(d-a, cross(e-a, edge)), dot(e-a, cross(c-a, edge)));
    if (!all(signs >= float3(0.0)) && !all(signs <= float3(0.0)))
        return EpaBoundary;
    polytope.vertexCount = 5;
    return 0;
}

uint seedEpaPolytope(inout ConvexPolytope polytope, ConvexDistance seed, ConvexGeometry first, ConvexGeometry second,
                     StructuredBuffer<uint4> model, RWStructuredBuffer<uint4> scratch)
{
    polytope.vertexCount = seed.count;
    polytope.faceCount = 0;
    if (polytope.vertexCapacity < 5 || polytope.faceCapacity < 6)
        return EpaCapacity;
    for (uint i = 0; i < seed.count; ++i)
        storeEpaVertex(polytope, i, seed.simplex[i], scratch);
    if (seed.count == 2)
        return seedEpaSegment(polytope, seed.distance, first, second, model, scratch);
    if (seed.count == 3)
        return seedEpaTriangle(polytope, seed.distance, first, second, model, scratch);

    const float3 a = epaPoint(polytope, 0, scratch);
    const float3 b = epaPoint(polytope, 1, scratch);
    const float3 c = epaPoint(polytope, 2, scratch);
    const float3 d = epaPoint(polytope, 3, scratch);
    polytope.center = 0.25 * (a + b + c + d);
    const uint3 faces[4] = {uint3(0,1,2), uint3(0,3,1), uint3(0,2,3), uint3(3,2,1)};
    float nearest = ConvexFarDistance;
    uint closest = 0;
    for (uint i = 0; i < 4; ++i)
    {
        const float distance = attachEpaFace(polytope, faces[i], scratch);
        if (distance < 1.0e-17)
        {
            epaSelectTriangle(polytope, faces[i], scratch);
            return seedEpaTriangle(polytope, seed.distance, first, second, model, scratch);
        }
        if (distance < nearest || (i == 1 && distance == nearest))
        {
            nearest = distance;
            closest = i;
        }
    }
    if (epaTetrahedronContainsOrigin(a,b,c,d))
        return 0;
    if (nearest > MinValue)
        return EpaBoundary;
    epaSelectTriangle(polytope, faces[closest], scratch);
    return seedEpaTriangle(polytope, seed.distance, first, second, model, scratch);
}
