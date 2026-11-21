// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Adapted from MuJoCo Warp _src/collision_gjk.py, EPA expansion and witnesses.

#include "convex_distance.slang"
#include "convex_polytope.slang"

static const uint EpaDegenerate = 3;
static const uint EpaIterationLimit = 4;

struct ConvexPenetration
{
    float3 first;
    float3 second;
    float distance;
    uint face; // the retained face in this worker's scratch lease
    uint status;
};

ConvexPenetration failedConvexPenetration(uint status)
{
    return {float3(0.0), float3(0.0), ConvexFarDistance, NoIndex, status};
}

bool addEpaHorizonEdge(ConvexPolytope polytope, uint a, uint b, inout uint size, RWStructuredBuffer<uint> horizon)
{
    const uint edge = (min(a, b) << 10) | max(a, b);
    const uint base = polytope.horizonBase;
    for (uint i = 0; i < size; ++i)
    {
        if (horizon[base + i] == edge)
        {
            --size;
            horizon[base + i] = horizon[base + size];
            return true;
        }
    }
    if (size == polytope.horizonCapacity)
        return false;
    horizon[base + size++] = edge;
    return true;
}

bool addEpaHorizonFace(ConvexPolytope polytope, uint face, inout uint size, RWStructuredBuffer<uint> horizon)
{
    const uint3 ids = unpackEpaFace(face);
    return addEpaHorizonEdge(polytope, ids.x, ids.y, size, horizon) &&
           addEpaHorizonEdge(polytope, ids.y, ids.z, size, horizon) &&
           addEpaHorizonEdge(polytope, ids.z, ids.x, size, horizon);
}

ConvexPenetration epaWitness(ConvexPolytope polytope, uint face, uint status, RWStructuredBuffer<uint4> scratch)
{
    const uint4 entry = scratch[polytope.faceBase + face];
    const uint3 ids = unpackEpaFace(entry.w);
    const ConvexSimplexVertex a = loadEpaVertex(polytope, ids.x, scratch);
    const ConvexSimplexVertex b = loadEpaVertex(polytope, ids.y, scratch);
    const ConvexSimplexVertex c = loadEpaVertex(polytope, ids.z, scratch);
    const float3 projection = asfloat(entry.xyz);
    float3 weights;
    if (!convexTriangleCoordinates(a.first - a.second, b.first - b.second, c.first - c.second, projection, weights))
        return failedConvexPenetration(EpaDegenerate);
    const float3 first = a.first * weights.x + b.first * weights.y + c.first * weights.z;
    const float3 second = a.second * weights.x + b.second * weights.y + c.second * weights.z;
    return {first, second, -length(projection), face, status};
}

ConvexPenetration expandConvexPenetration(ConvexDistance seed, ConvexGeometry first, ConvexGeometry second,
                                           float tolerance, uint iterations, ConvexPolytope polytope,
                                           StructuredBuffer<uint4> model,
                                           RWStructuredBuffer<uint4> scratch, RWStructuredBuffer<uint> horizon)
{
    const uint initial = seedEpaPolytope(polytope, seed, first, second, model, scratch);
    if (initial == EpaBoundary)
        return {seed.first, seed.second, seed.distance, NoIndex, EpaBoundary};
    if (initial == EpaCapacity)
        return failedConvexPenetration(EpaCapacity);

    const bool discrete = convexDiscreteGeometry(first) && convexDiscreteGeometry(second);
    const float epsilon = discrete ? 1.0e-7 : tolerance;
    const uint limit = min(iterations, 1000);
    float upper = ConvexFarDistance;
    float upper2 = ConvexFarDistance;
    uint nearest = NoIndex;
    uint valid = polytope.faceCount;
    uint step = 0;
    for (; step < limit; ++step)
    {
        const uint previous = nearest;
        nearest = NoIndex;
        float lower2 = ConvexFarDistance;
        for (uint i = 0; i < polytope.faceCount; ++i)
        {
            const uint4 face = scratch[polytope.faceBase + i];
            if ((face.w & (EpaDeletedFace | EpaInvalidFace)) != 0)
                continue;
            const float3 projection = asfloat(face.xyz);
            const float norm2 = dot(projection, projection);
            if (norm2 < lower2)
            {
                lower2 = norm2;
                nearest = i;
            }
        }
        if (nearest == NoIndex || lower2 > upper2)
        {
            nearest = previous;
            break;
        }
        if (lower2 <= 0.0)
            break;

        const uint newVertex = polytope.vertexCount;
        if (newVertex >= min(polytope.vertexCapacity, EpaVertexMask + 1))
            return failedConvexPenetration(EpaCapacity);
        const uint faceAddress = polytope.faceBase + nearest;
        uint4 face = scratch[faceAddress];
        const float3 projection = asfloat(face.xyz);
        const float lower = sqrt(lower2);
        int2 cached;
        const float3 point = sampleEpaVertex(polytope, newVertex, first, second, projection / lower,
                                            model, scratch, cached);
        first.cachedVertex = cached.x;
        second.cachedVertex = cached.y;
        ++polytope.vertexCount;
        upper = min(upper, dot(projection, point) / lower);
        upper2 = upper * upper;
        if (upper - lower < epsilon)
            break;

        if (discrete)
        {
            const int2 features = loadEpaVertex(polytope, newVertex, scratch).indices;
            bool repeated = false;
            for (uint i = 0; i < newVertex; ++i)
            {
                if (all(loadEpaVertex(polytope, i, scratch).indices == features))
                {
                    repeated = true;
                    break;
                }
            }
            if (repeated)
                break;
        }

        face.w |= EpaDeletedFace;
        scratch[faceAddress] = face;
        --valid;
        uint edgeCount = 0;
        if (!addEpaHorizonFace(polytope, face.w, edgeCount, horizon))
            return failedConvexPenetration(EpaCapacity);
        for (uint i = 0; i < polytope.faceCount; ++i)
        {
            const uint index = polytope.faceBase + i;
            uint4 neighbor = scratch[index];
            if ((neighbor.w & EpaDeletedFace) != 0)
                continue;
            const float3 normal = asfloat(neighbor.xyz);
            if (dot(normal, point) - dot(normal, normal) <= 1.0e-10)
                continue;
            if ((neighbor.w & EpaInvalidFace) == 0)
                --valid;
            neighbor.w |= EpaDeletedFace;
            scratch[index] = neighbor;
            if (!addEpaHorizonFace(polytope, neighbor.w, edgeCount, horizon))
                return failedConvexPenetration(EpaCapacity);
        }
        for (uint i = 0; i < edgeCount; ++i)
        {
            const uint edge = horizon[polytope.horizonBase + i];
            const uint3 ids = uint3(newVertex, edge & EpaVertexMask, edge >> 10);
            const float face2 = attachEpaFace(polytope, ids, scratch);
            if (face2 < 0.0)
                return failedConvexPenetration(EpaCapacity);
            if (face2 == 0.0)
                return failedConvexPenetration(EpaDegenerate);
            if (face2 >= lower2 && face2 <= upper2)
                ++valid;
            else
            {
                const uint address = polytope.faceBase + polytope.faceCount - 1;
                scratch[address].w |= EpaInvalidFace;
            }
        }
        if (valid == 0)
            break;
    }
    if (nearest == NoIndex)
        return failedConvexPenetration(EpaDegenerate);
    return epaWitness(polytope, nearest, step == limit ? EpaIterationLimit : 0, scratch);
}
