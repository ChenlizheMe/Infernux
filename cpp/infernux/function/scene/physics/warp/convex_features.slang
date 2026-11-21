// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Adapted from MuJoCo Warp _src/collision_gjk.py, convex manifold feature recovery.

static const float ConvexFaceTolerance = 0.9960509968;
static const float ConvexEdgeTolerance = 0.0887829470;

struct ConvexFeature
{
    int3 ids;
    float3 point;
    uint dimension;
};

struct ConvexFaceChoices
{
    uint3 ids;
    uint count;
    uint linkOffset;
    bool linked;
};

struct ConvexEdge
{
    float3 first;
    float3 second;
};

ConvexFeature convexFeature(ConvexSimplexVertex vertices[3], bool first)
{
    ConvexFeature feature;
    feature.point = first ? vertices[0].first : vertices[0].second;
    const uint side = first ? 0 : 1;
    feature.ids = int3(vertices[0].indices[side], vertices[1].indices[side], vertices[2].indices[side]);
    if (feature.ids.x != feature.ids.y)
        feature.dimension = feature.ids.z == feature.ids.x || feature.ids.z == feature.ids.y ? 2 : 3;
    else
    {
        feature.ids.y = feature.ids.z;
        feature.dimension = feature.ids.x != feature.ids.z ? 2 : 1;
    }
    return feature;
}

float3 convexBoxCorner(ConvexGeometry geometry, uint id)
{
    const float3 corner = geometry.shape.size.xyz *
        float3((id & 1u) != 0 ? 1.0 : -1.0, (id & 2u) != 0 ? 1.0 : -1.0, (id & 4u) != 0 ? 1.0 : -1.0);
    return geometry.frame.position.xyz + rotatePoint(geometry.frame, corner);
}

float3 convexBoxNormal(ConvexGeometry geometry, uint face)
{
    return frameAxis(geometry.frame, face / 2) * ((face & 1u) == 0 ? 1.0 : -1.0);
}

bool convexHasLink(uint2 range, uint value, StructuredBuffer<uint4> model)
{
    for (uint i = 0; i < range.y; ++i)
        if (convexLink(model, range.x + i) == value)
            return true;
    return false;
}

ConvexFaceChoices convexFaceChoices(ConvexGeometry geometry, ConvexFeature feature, float3 direction,
                                     uint2 mesh, StructuredBuffer<uint4> model)
{
    ConvexFaceChoices choices = {uint3(0), 0, 0, false};
    switch (geometry.shape.type.x)
    {
    case GeomBox:
        uint mask = 7u;
        if (feature.dimension >= 2)
            mask &= ~uint(feature.ids.x ^ feature.ids.y);
        if (feature.dimension == 3)
            mask &= ~uint(feature.ids.x ^ feature.ids.z);
        for (uint axis = 0; axis < 3; ++axis)
        {
            const uint bit = 1u << axis;
            if ((mask & bit) != 0)
                choices.ids[choices.count++] = 2 * axis + ((uint(feature.ids.x) & bit) != 0 ? 0u : 1u);
        }
        if ((feature.dimension == 2 && (choices.count == 1 || choices.count == 2)) ||
            (feature.dimension == 3 && choices.count == 1) || feature.dimension == 1)
            return choices;
        choices.count = 0;
        const float norm = length(direction);
        for (uint side = 0; side < 6; ++side)
        {
            if (dot(direction, convexBoxNormal(geometry, side)) > ConvexFaceTolerance * norm)
            {
                choices.ids.x = side;
                choices.count = 1;
                break;
            }
        }
        break;
    case GeomCylinder:
        if (feature.dimension == 1)
        {
            choices.ids.x = uint(feature.ids.x);
            choices.count = 1;
        }
        break;
    case GeomMesh:
        const uint2 first = convexTopology(model, mesh.y + uint(feature.ids.x)).xy;
        if (feature.dimension == 1)
        {
            choices.linkOffset = first.x;
            choices.count = first.y;
            choices.linked = true;
            break;
        }
        const uint2 second = convexTopology(model, mesh.y + uint(feature.ids.y)).xy;
        const uint2 third = convexTopology(model, mesh.y + uint(feature.ids.z)).xy;
        const uint limit = feature.dimension == 3 ? 1u : 2u;
        for (uint i = 0; i < first.y && choices.count < limit; ++i)
        {
            const uint face = convexLink(model, first.x + i);
            if (convexHasLink(second, face, model) && (feature.dimension == 2 || convexHasLink(third, face, model)))
                choices.ids[choices.count++] = face;
        }
        break;
    }
    return choices;
}

uint convexFaceId(ConvexFaceChoices choices, uint index, StructuredBuffer<uint4> model)
{
    return choices.linked ? convexLink(model, choices.linkOffset + index) : choices.ids[index];
}

float3 convexFaceNormal(ConvexGeometry geometry, uint face, uint2 mesh, StructuredBuffer<uint4> model)
{
    if (geometry.shape.type.x == GeomBox)
        return convexBoxNormal(geometry, face);
    if (geometry.shape.type.x == GeomCylinder)
        return frameAxis(geometry.frame, 2) * (face == 0 ? 1.0 : -1.0);
    return rotatePoint(geometry.frame, convexVertex(model, convexTopology(model, mesh.x + face).z));
}

uint convexEdgeCount(ConvexGeometry geometry, ConvexFeature feature, uint2 mesh, StructuredBuffer<uint4> model)
{
    if (geometry.shape.type.x == GeomBox)
        return feature.dimension == 1 ? 3 : 1;
    if (geometry.shape.type.x == GeomMesh && feature.dimension == 1)
        return convexTopology(model, mesh.y + uint(feature.ids.x)).y;
    return 1;
}

float3 convexMeshVertex(ConvexGeometry geometry, uint id, StructuredBuffer<uint4> model)
{
    return geometry.frame.position.xyz + rotatePoint(geometry.frame, convexVertex(model, geometry.shape.type.y + id));
}

ConvexEdge convexEdge(ConvexGeometry geometry, ConvexFeature feature, uint index, float3 witness, uint2 mesh,
                      StructuredBuffer<uint4> model)
{
    const float3 center = geometry.frame.position.xyz;
    const float3 axis = frameAxis(geometry.frame, 2);
    const float halfLength = geometry.shape.size.y;
    switch (geometry.shape.type.x)
    {
    case GeomBox:
        const uint first = uint(feature.ids.x);
        const uint second = feature.dimension == 1 ? (first ^ (1u << index)) : uint(feature.ids.y);
        return {convexBoxCorner(geometry, first), convexBoxCorner(geometry, second)};
    case GeomMesh:
        const uint vertex = uint(feature.ids.x);
        uint end = uint(feature.ids.y);
        if (feature.dimension == 1)
        {
            const uint face = convexLink(model, convexTopology(model, mesh.y + vertex).x + index);
            const uint2 corners = convexTopology(model, mesh.x + face).xy;
            for (uint i = 0; i < corners.y; ++i)
            {
                if (convexLink(model, corners.x + i) == vertex)
                {
                    end = convexLink(model, corners.x + (i == 0 ? corners.y - 1 : i - 1));
                    break;
                }
            }
        }
        return {convexMeshVertex(geometry, vertex, model), convexMeshVertex(geometry, end, model)};
    case GeomCylinder:
        const float sign = feature.ids.x == 0 ? -1.0 : 1.0;
        float3 point = feature.point;
        if (geometry.margin > 0.0)
        {
            const float2 local = inverseRotatePoint(geometry.frame, point - center).xy;
            const float norm = length(local);
            const float2 radial = norm >= MinValue ? local * (geometry.shape.size.x / norm) : float2(0.0);
            point = center + rotatePoint(geometry.frame, float3(radial, -sign * halfLength));
        }
        return {point, point + sign * axis * (2.0 * halfLength)};
    default: // capsule edge through the already inflated witness
        const float3 mid = witness - axis * dot(witness - center, axis);
        return {mid - axis * halfLength, mid + axis * halfLength};
    }
}

void storeConvexClipPoint(RWStructuredBuffer<uint4> scratch, uint address, float3 point)
{
    scratch[address] = uint4(asuint(point), 0);
}

float3 loadConvexClipPoint(RWStructuredBuffer<uint4> scratch, uint address)
{
    return asfloat(scratch[address].xyz);
}

uint convexFaceVertexCount(ConvexGeometry geometry, uint face, uint2 mesh, StructuredBuffer<uint4> model)
{
    if (geometry.shape.type.x == GeomBox)
        return 4;
    if (geometry.shape.type.x == GeomCylinder)
        return 16;
    return convexTopology(model, mesh.x + face).y;
}

void storeConvexFace(ConvexGeometry geometry, uint face, uint2 mesh, uint count, uint base,
                     StructuredBuffer<uint4> model, RWStructuredBuffer<uint4> scratch)
{
    if (geometry.shape.type.x == GeomBox)
    {
        const uint4 corners[6] = {uint4(7,3,1,5), uint4(2,6,4,0), uint4(2,3,7,6),
                                  uint4(4,5,1,0), uint4(6,7,5,4), uint4(3,2,0,1)};
        for (uint i = 0; i < 4; ++i)
            storeConvexClipPoint(scratch, base + i, convexBoxCorner(geometry, corners[face][i]));
    }
    else if (geometry.shape.type.x == GeomCylinder)
    {
        const float2 unitCircle[16] = {
            float2(1.0,0.0), float2(0.9238795325,0.3826834324), float2(0.7071067812,0.7071067812),
            float2(0.3826834324,0.9238795325), float2(0.0,1.0), float2(-0.3826834324,0.9238795325),
            float2(-0.7071067812,0.7071067812), float2(-0.9238795325,0.3826834324), float2(-1.0,0.0),
            float2(-0.9238795325,-0.3826834324), float2(-0.7071067812,-0.7071067812), float2(-0.3826834324,-0.9238795325),
            float2(0.0,-1.0), float2(0.3826834324,-0.9238795325), float2(0.7071067812,-0.7071067812),
            float2(0.9238795325,-0.3826834324)};
        const float sign = face == 0 ? 1.0 : -1.0;
        const float3 center = geometry.frame.position.xyz + frameAxis(geometry.frame, 2) * (sign * geometry.shape.size.y);
        const float3 u = frameAxis(geometry.frame, 0) * geometry.shape.size.x;
        const float3 v = frameAxis(geometry.frame, 1) * (sign * geometry.shape.size.x);
        for (uint i = 0; i < 16; ++i)
            storeConvexClipPoint(scratch, base + i, center + u * unitCircle[i].x - v * unitCircle[i].y);
    }
    else
    {
        const uint offset = convexTopology(model, mesh.x + face).x;
        for (uint i = 0; i < count; ++i)
            storeConvexClipPoint(scratch, base + i, convexMeshVertex(geometry, convexLink(model, offset + count - 1 - i), model));
    }
}
