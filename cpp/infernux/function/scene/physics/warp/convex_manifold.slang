// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Adapted from MuJoCo Warp _src/collision_gjk.py, multicontact polygon clipping.

#include "convex_penetration.slang"
#include "convex_features.slang"


struct ConvexManifold
{
    float4 points[4]; // rebased midpoint and signed separation from the unclipped faces
    float3 direction;
    uint count;
    uint status;
};

struct ConvexPolygon
{
    uint begin;
    uint count;
};

float convexQuadArea(uint address, uint4 indices, RWStructuredBuffer<uint4> scratch)
{
    const float3 ac = loadConvexClipPoint(scratch, address + indices.x) -
                      loadConvexClipPoint(scratch, address + indices.z);
    const float3 bd = loadConvexClipPoint(scratch, address + indices.y) -
                      loadConvexClipPoint(scratch, address + indices.w);
    return length(cross(ac, bd));
}

bool advanceConvexQuad(ConvexPolygon polygon, uint moved, inout uint4 indices, inout uint4 best, inout float area,
                        RWStructuredBuffer<uint4> scratch)
{
    uint4 next = indices;
    next[moved] = (next[moved] + 1) % polygon.count;
    const float value = convexQuadArea(polygon.begin, next, scratch);
    if (value <= area)
        return false;
    area = value;
    indices = next;
    best = indices;
    return true;
}

void optimizeConvexQuad(ConvexPolygon polygon, inout uint4 indices, inout uint4 best, inout float area,
                        RWStructuredBuffer<uint4> scratch)
{
    while (advanceConvexQuad(polygon, 3, indices, best, area, scratch))
    {
        while (advanceConvexQuad(polygon, 2, indices, best, area, scratch)) {}
        while (advanceConvexQuad(polygon, 1, indices, best, area, scratch)) {}
    }
}

uint4 reduceConvexQuad(ConvexPolygon polygon, RWStructuredBuffer<uint4> scratch)
{
    uint4 indices = uint4(0,1,2,3);
    uint4 best = indices;
    float area = convexQuadArea(polygon.begin, indices, scratch);
    for (uint i = 0; i < polygon.count; ++i)
    {
        indices.x = i;
        optimizeConvexQuad(polygon, indices, best, area, scratch);
        for (uint j = 1; j < 4 && indices[j] == indices[j-1]; ++j)
            indices[j] = (indices[j] + 1) % polygon.count;
    }
    return best;
}

uint2 convexClipExtremes(ConvexPolygon polygon, RWStructuredBuffer<uint4> scratch)
{
    uint2 result = uint2(0,1);
    float best = 0.0;
    for (uint i = 0; i < polygon.count; ++i)
    {
        const float3 a = loadConvexClipPoint(scratch, polygon.begin + i);
        for (uint j = i + 1; j < polygon.count; ++j)
        {
            const float3 offset = loadConvexClipPoint(scratch, polygon.begin + j) - a;
            const float distance = dot(offset, offset);
            if (distance > best)
            {
                best = distance;
                result = uint2(i, j);
            }
        }
    }
    return result;
}

bool clipConvexPolygon(ConvexPolygon reference, inout ConvexPolygon polygon, uint output, uint capacity,
                       float3 normal, float margin, RWStructuredBuffer<uint4> scratch)
{
    for (uint edge = 0; edge < reference.count; ++edge)
    {
        const float3 a = loadConvexClipPoint(scratch, reference.begin + edge);
        const float3 b = loadConvexClipPoint(scratch, reference.begin + (edge + 1) % reference.count);
        const float3 inward = cross(b - a, normal);
        const float offset = dot(inward, a);
        uint written = 0;
        for (uint i = 0; i < polygon.count; ++i)
        {
            const float3 p = loadConvexClipPoint(scratch, polygon.begin + i);
            const float3 q = loadConvexClipPoint(scratch, polygon.begin + (i + 1) % polygon.count);
            const bool inP = dot(p - a, inward) > -1.0e-10;
            const bool inQ = dot(q - a, inward) > -1.0e-10;
            if (inP != inQ)
            {
                const float3 delta = q - p;
                const float denominator = dot(inward, delta);
                if (abs(denominator) >= 1.0e-10)
                {
                    const float t = (offset - dot(inward, p)) / denominator;
                    if (t > -3.0e-7 && t < 1.0000003)
                    {
                        if (written == capacity)
                            return false;
                        storeConvexClipPoint(scratch, output + written++, p + saturate(t) * delta);
                    }
                }
            }
            if (inQ)
            {
                if (written == capacity)
                    return false;
                storeConvexClipPoint(scratch, output + written++, q);
            }
        }
        const uint previous = polygon.begin;
        polygon = {output, written};
        output = previous;
        if (written == 0)
            break;
    }

    const float3 plane = loadConvexClipPoint(scratch, reference.begin);
    uint kept = 0;
    for (uint i = 0; i < polygon.count; ++i)
    {
        const float3 point = loadConvexClipPoint(scratch, polygon.begin + i);
        if (dot(point - plane, normal) <= margin)
            storeConvexClipPoint(scratch, polygon.begin + kept++, point);
    }
    polygon.count = kept;
    return true;
}

bool findConvexFaceEdge(ConvexGeometry faceGeometry, ConvexFaceChoices choices, uint2 faceMesh,
                        ConvexGeometry edgeGeometry, ConvexFeature feature, uint2 edgeMesh, float3 witness,
                        float3 direction, StructuredBuffer<uint4> model, out uint face, out float3 normal, out ConvexEdge segment)
{
    face = 0;
    normal = float3(0.0);
    segment = {float3(0.0), float3(0.0)};
    const uint count = convexEdgeCount(edgeGeometry, feature, edgeMesh, model);
    for (uint i = 0; i < choices.count; ++i)
    {
        const uint index = convexFaceId(choices, i, model);
        const float3 candidate = convexFaceNormal(faceGeometry, index, faceMesh, model);
        if (dot(candidate, direction) <= MinValue)
            continue;
        for (uint j = 0; j < count; ++j)
        {
            const ConvexEdge edge = convexEdge(edgeGeometry, feature, j, witness, edgeMesh, model);
            const float3 axis = edge.second - edge.first;
            if (abs(dot(axis, candidate)) < ConvexEdgeTolerance * length(axis))
            {
                face = index;
                normal = candidate;
                segment = edge;
                return true;
            }
        }
    }
    return false;
}

void buildConvexManifold(inout ConvexManifold manifold, ConvexSimplexVertex support[3],
                         ConvexGeometry first, ConvexGeometry second, float3 witnessA, float3 witnessB,
                         uint2 meshA, uint2 meshB, uint clipBase, uint maxFaceVertices,
                         StructuredBuffer<uint4> model, RWStructuredBuffer<uint4> scratch)
{
    const ConvexFeature featureA = convexFeature(support, true);
    const ConvexFeature featureB = convexFeature(support, false);
    const float3 direction = witnessB - witnessA;
    const ConvexFaceChoices choicesA = convexFaceChoices(first, featureA, -direction, meshA, model);
    const ConvexFaceChoices choicesB = convexFaceChoices(second, featureB, direction, meshB, model);
    const float directionEpsilon = ConvexEdgeTolerance * length(direction);
    float best = -ConvexFaceTolerance;
    float3 normalA = float3(0.0);
    float3 normalB = float3(0.0);
    uint faceA = 0;
    uint faceB = 0;
    bool found = false;
    for (uint i = 0; i < choicesA.count; ++i)
    {
        const uint a = convexFaceId(choicesA, i, model);
        const float3 nA = convexFaceNormal(first, a, meshA, model);
        if (dot(nA, direction) >= -directionEpsilon)
            continue;
        for (uint j = 0; j < choicesB.count; ++j)
        {
            const uint b = convexFaceId(choicesB, j, model);
            const float3 nB = convexFaceNormal(second, b, meshB, model);
            if (dot(nB, direction) <= directionEpsilon)
                continue;
            const float alignment = dot(nA, nB);
            if (alignment < best)
            {
                best = alignment;
                normalA = nA;
                normalB = nB;
                faceA = a;
                faceB = b;
                found = true;
            }
        }
    }
    bool edgeA = false;
    bool edgeB = false;
    ConvexEdge segment;
    if (!found)
    {
        if (featureA.dimension < 3 && choicesB.count != 0 &&
            (featureA.dimension <= featureB.dimension || choicesA.count == 0))
        {
            edgeA = findConvexFaceEdge(second, choicesB, meshB, first, featureA, meshA, witnessA, direction,
                                       model, faceB, normalB, segment);
            if (!edgeA)
                return;
        }
        else if (featureB.dimension < 3)
        {
            edgeB = findConvexFaceEdge(first, choicesA, meshA, second, featureB, meshB, witnessB, -direction,
                                       model, faceA, normalA, segment);
            if (!edgeB)
                return;
        }
        else
            return;
    }

    const uint countA = edgeA ? 2u : convexFaceVertexCount(first, faceA, meshA, model);
    const uint countB = edgeB ? 2u : convexFaceVertexCount(second, faceB, meshB, model);
    if (max(countA, countB) > maxFaceVertices)
    {
        manifold.status = ConvexClipCapacity;
        return;
    }
    const uint subject = clipBase + maxFaceVertices;
    ConvexPolygon reference = {clipBase, edgeA ? countB : countA};
    ConvexPolygon polygon = {subject, edgeA ? countA : countB};
    if (edgeA)
        storeConvexFace(second, faceB, meshB, countB, clipBase, model, scratch);
    else
        storeConvexFace(first, faceA, meshA, countA, clipBase, model, scratch);
    if (edgeA || edgeB)
    {
        storeConvexClipPoint(scratch, subject, segment.first);
        storeConvexClipPoint(scratch, subject + 1, segment.second);
    }
    else
        storeConvexFace(second, faceB, meshB, countB, subject, model, scratch);

    const float3 normal = edgeA ? normalB : normalA;
    const float3 projectionDir = edgeA ? -normalB : (edgeB ? -normalA : normalB);
    const float3 marginDir = edgeB ? normalA : -normalB;
    const float marginA = first.shape.type.x == GeomCapsule ? 0.0 : 0.5 * first.margin;
    const float marginB = second.shape.type.x == GeomCapsule ? 0.0 : 0.5 * second.margin;
    const float margin = marginA + marginB;
    if (!clipConvexPolygon(reference, polygon, subject + 2 * maxFaceVertices, 2 * maxFaceVertices,
                           normal, margin, scratch))
    {
        manifold.status = ConvexClipCapacity;
        return;
    }
    if (polygon.count == 0)
        return;
    uint4 indices = uint4(0,1,2,3);
    const uint count = edgeA || edgeB ? min(polygon.count, 2) : min(polygon.count, 4);
    if ((edgeA || edgeB) && polygon.count > 2)
        indices.xy = convexClipExtremes(polygon, scratch);
    else if (polygon.count > 4)
        indices = reduceConvexQuad(polygon, scratch);

    const float3 plane = loadConvexClipPoint(scratch, reference.begin);
    for (uint i = 0; i < count; ++i)
    {
        const float3 point = loadConvexClipPoint(scratch, polygon.begin + indices[i]);
        const float3 projected = point - dot(point - plane, normal) * normal;
        const float depth = dot(projected - point, projectionDir);
        const float3 partner = projected - depth * projectionDir;
        const float3 a = (edgeA ? partner : projected) + marginA * marginDir;
        const float3 b = (edgeA ? projected : partner) - marginB * marginDir;
        manifold.points[i] = float4(0.5 * (a + b), depth - margin);
        if (i == 0)
            manifold.direction = a - b;
    }
    manifold.count = count;
}
