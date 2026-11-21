// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#include "model.slang"

static const uint GeomPlane = 0;
static const uint GeomHeightfield = 1;
static const uint GeomSphere = 2;
static const uint GeomCapsule = 3;
static const uint GeomEllipsoid = 4;
static const uint GeomCylinder = 5;
static const uint GeomBox = 6;
static const uint GeomMesh = 7;
static const uint GeomTriangleMesh = 8;
static const uint GeomTriangle = 9; // A transient support shape, never a model row.
static const float MinValue = 1.0e-15;

struct CollisionShape
{
    float4 size;
    uint4 type; // geometry type, hull vertex start or heightfield id, hull vertex count, adjacency offset or NoIndex
};

float3 frameAxis(CartesianFrame frame, uint axis)
{
    return float3(frame.row0[axis], frame.row1[axis], frame.row2[axis]);
}

float3 rotatePoint(CartesianFrame frame, float3 point)
{
    return float3(dot(frame.row0.xyz, point), dot(frame.row1.xyz, point), dot(frame.row2.xyz, point));
}

float3 inverseRotatePoint(CartesianFrame frame, float3 point)
{
    return float3(dot(frameAxis(frame, 0), point), dot(frameAxis(frame, 1), point), dot(frameAxis(frame, 2), point));
}

CartesianFrame relativeCollisionFrame(CartesianFrame origin, CartesianFrame geometry)
{
    const float3 x = inverseRotatePoint(origin, frameAxis(geometry, 0));
    const float3 y = inverseRotatePoint(origin, frameAxis(geometry, 1));
    const float3 z = inverseRotatePoint(origin, frameAxis(geometry, 2));
    return {float4(inverseRotatePoint(origin, geometry.position.xyz - origin.position.xyz), 0.0),
            float4(x.x, y.x, z.x, 0.0), float4(x.y, y.y, z.y, 0.0), float4(x.z, y.z, z.z, 0.0)};
}

float3 triangleVertex(CartesianFrame triangle, uint index)
{
    return index == 0 ? triangle.row0.xyz : (index == 1 ? triangle.row1.xyz : triangle.row2.xyz);
}
