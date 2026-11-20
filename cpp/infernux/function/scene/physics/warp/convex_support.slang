// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Adapted from MuJoCo Warp _src/collision_gjk.py, support.

#include "collision_geometry.slang"

struct ConvexGeometry
{
    CartesianFrame frame;
    CollisionShape shape;
    float margin;
    int cachedVertex;
};

struct ConvexSupportPoint
{
    float3 point;
    int cachedVertex;
    int vertex;
};

float convexSign(float value)
{
    // Warp uses the positive side for both signs of zero.
    return value < 0.0 ? -1.0 : 1.0;
}

ConvexSupportPoint hullSupport(uint4 shape, int cachedVertex, float3 direction,
                               StructuredBuffer<float4> vertices, StructuredBuffer<int> adjacency)
{
    const int first = int(shape.y);
    const int count = int(shape.z);
    int vertex = 0;
    if (shape.w == NoIndex || count < 10)
    {
        vertex = max(cachedVertex - first, 0);
        float best = dot(vertices[first + vertex].xyz, direction);
        for (int i = 0; i < count; ++i)
        {
            const float projection = dot(vertices[first + i].xyz, direction);
            if (projection > best)
            {
                best = projection;
                vertex = i;
            }
        }
        return {vertices[first + vertex].xyz, first + vertex, vertex};
    }

    const int graph = int(shape.w);
    const int hullCount = adjacency[graph];
    const int neighborOffsets = graph + 2;
    const int hullVertices = neighborOffsets + hullCount;
    const int neighbors = hullVertices + hullCount;
    int current = max(cachedVertex, 0);
    float best = dot(direction, vertices[first + adjacency[hullVertices + current]].xyz);
    int previous = -1;
    while (previous != current)
    {
        previous = current;
        int edge = neighbors + adjacency[neighborOffsets + current];
        for (int neighbor = adjacency[edge]; neighbor >= 0; neighbor = adjacency[++edge])
        {
            const int candidate = adjacency[hullVertices + neighbor];
            const float projection = dot(direction, vertices[first + candidate].xyz);
            if (projection > best)
            {
                current = neighbor;
                best = projection;
            }
        }
    }
    vertex = adjacency[hullVertices + current];
    return {vertices[first + vertex].xyz, current, vertex};
}

ConvexSupportPoint convexSupport(ConvexGeometry geometry, float3 direction, StructuredBuffer<float4> vertices,
                                 StructuredBuffer<int> adjacency)
{
    const uint kind = geometry.shape.type.x;
    const float3 size = geometry.shape.size.xyz;
    const float margin = 0.5 * geometry.margin;
    const float3 center = geometry.frame.position.xyz;
    if (kind == GeomSphere)
        return {center + (size.x + margin) * direction, -1, -1};

    const float3 local = inverseRotatePoint(geometry.frame, direction);
    ConvexSupportPoint result = {float3(0.0), -1, -1};
    switch (kind)
    {
    case GeomBox:
        result.point = size * float3(convexSign(local.x), convexSign(local.y), convexSign(local.z));
        result.vertex = (local.x >= 0.0 ? 1 : 0) | (local.y >= 0.0 ? 2 : 0) | (local.z >= 0.0 ? 4 : 0);
        break;
    case GeomCapsule:
        result.point = local * size.x;
        result.point.z += convexSign(local.z) * size.y;
        break;
    case GeomEllipsoid:
        const float3 ellipse = local * size;
        const float ellipseLength = length(ellipse);
        result.point = ellipseLength > 0.0 ? (ellipse / ellipseLength) * size : float3(0.0);
        break;
    case GeomCylinder:
        const float radial = length(local.xy);
        result.point.xy = radial > MinValue ? local.xy * (size.x / radial) : float2(0.0);
        result.point.z = convexSign(local.z) * size.y;
        result.vertex = local.z >= 0.0 ? 0 : 1;
        break;
    case GeomMesh:
        result = hullSupport(geometry.shape.type, geometry.cachedVertex, local, vertices, adjacency);
        break;
    }
    result.point = rotatePoint(geometry.frame, result.point) + center;
    if (margin > 0.0)
        result.point += margin * direction;
    return result;
}
