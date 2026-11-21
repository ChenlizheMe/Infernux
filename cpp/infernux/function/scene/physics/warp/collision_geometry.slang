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
