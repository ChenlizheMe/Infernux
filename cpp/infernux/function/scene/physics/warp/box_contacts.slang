// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Adapted from MuJoCo Warp _src/collision_primitive_core.py, box_box.

static const uint MaxBoxVertices = 8;
static const float BoxSeparationEpsilon = 1.0e-6;
static const float BoxParallelEpsilon = 1.0e-7;
static const float BoxSupportEpsilon = 1.0e-5;
static const float BoxDuplicateEpsilon = 1.0e-10;
static const float BoxEdgeBias = 1.0e-6;

struct BoxManifold
{
    float3 normal;
    float4 points[MaxBoxVertices]; // world position and signed distance
    uint count;
};

struct BoxPolygon
{
    float3 vertices[MaxBoxVertices]; // reference plane coordinates and signed depth
    uint count;
};

float3 matrixColumn(float3x3 matrix, uint column)
{
    return float3(matrix[0][column], matrix[1][column], matrix[2][column]);
}

BoxPolygon clipBoxPolygon(BoxPolygon source, uint coordinate, float sign, float limit)
{
    BoxPolygon clipped;
    clipped.count = 0;
    for (uint i = 0; i < source.count; ++i)
    {
        const float3 p = source.vertices[i];
        const float3 q = source.vertices[i + 1 == source.count ? 0 : i + 1];
        const float dp = sign * p[coordinate] - limit;
        const float dq = sign * q[coordinate] - limit;
        if (dp <= 0.0 && clipped.count < MaxBoxVertices)
            clipped.vertices[clipped.count++] = p;
        const bool crosses = (dp < 0.0 && dq > 0.0) || (dp > 0.0 && dq < 0.0);
        if (crosses && clipped.count < MaxBoxVertices)
            clipped.vertices[clipped.count++] = p + (dp / (dp - dq)) * (q - p);
    }
    return clipped;
}

BoxManifold boxFaceManifold(CartesianFrame reference, float3 referenceExtent, float3 incidentExtent,
                           float3 incidentCenter, float3x3 incidentAxes, uint axis, bool referenceIsFirst,
                           float margin)
{
    const float sign = incidentCenter[axis] >= 0.0 ? 1.0 : -1.0;
    uint face = 0;
    for (uint i = 1; i < 3; ++i)
    {
        if (abs(incidentAxes[axis][i]) > abs(incidentAxes[axis][face]))
            face = i;
    }
    const float side = sign * incidentAxes[axis][face] > 0.0 ? -1.0 : 1.0;
    const uint x = (axis + 1) % 3;
    const uint y = (axis + 2) % 3;
    const uint u = (face + 1) % 3;
    const uint v = (face + 2) % 3;

    float3 center = incidentCenter + side * incidentExtent[face] * matrixColumn(incidentAxes, face);
    float3 spanU = incidentExtent[u] * matrixColumn(incidentAxes, u);
    float3 spanV = incidentExtent[v] * matrixColumn(incidentAxes, v);
    center = float3(center[x], center[y], sign * center[axis] - referenceExtent[axis]);
    spanU = float3(spanU[x], spanU[y], sign * spanU[axis]);
    spanV = float3(spanV[x], spanV[y], sign * spanV[axis]);

    BoxPolygon poly;
    poly.count = 4;
    poly.vertices[0] = center + spanU + spanV;
    poly.vertices[1] = center - spanU + spanV;
    poly.vertices[2] = center - spanU - spanV;
    poly.vertices[3] = center + spanU - spanV;
    poly = clipBoxPolygon(poly, 0, 1.0, referenceExtent[x]);
    poly = clipBoxPolygon(poly, 0, -1.0, referenceExtent[x]);
    poly = clipBoxPolygon(poly, 1, 1.0, referenceExtent[y]);
    poly = clipBoxPolygon(poly, 1, -1.0, referenceExtent[y]);

    BoxManifold manifold;
    manifold.count = 0;
    manifold.normal = (referenceIsFirst ? sign : -sign) * frameAxis(reference, axis);
    float2 accepted[MaxBoxVertices];
    const float2 faceExtent = float2(referenceExtent[x], referenceExtent[y]);
    const float duplicateRadius2 = BoxDuplicateEpsilon * dot(faceExtent, faceExtent);
    for (uint i = 0; i < poly.count; ++i)
    {
        const float3 vertex = poly.vertices[i];
        if (vertex.z > margin)
            continue;
        bool duplicate = false;
        for (uint j = 0; j < manifold.count; ++j)
        {
            const float2 delta = accepted[j] - vertex.xy;
            if (dot(delta, delta) < duplicateRadius2)
            {
                duplicate = true;
                break;
            }
        }
        if (duplicate)
            continue;
        accepted[manifold.count] = vertex.xy;
        float3 position;
        position[x] = vertex.x;
        position[y] = vertex.y;
        position[axis] = sign * (referenceExtent[axis] + 0.5 * vertex.z);
        manifold.points[manifold.count++] =
            float4(reference.position.xyz + rotatePoint(reference, position), vertex.z);
    }
    return manifold;
}

BoxManifold boxEdgeManifold(CartesianFrame first, float3 extentA, float3 extentB, float3 centerB,
                           float3x3 axesB, uint edgeA, uint edgeB, float3 axis, float tolerance)
{
    axis = dot(axis, centerB) < 0.0 ? -axis : axis;
    const float3 axisB = mul(axis, axesB);
    const uint a1 = (edgeA + 1) % 3;
    const uint a2 = (edgeA + 2) % 3;
    const uint b1 = (edgeB + 1) % 3;
    const uint b2 = (edgeB + 2) % 3;
    const uint ambA = abs(axis[a1]) < BoxSupportEpsilon ? a1 : (abs(axis[a2]) < BoxSupportEpsilon ? a2 : 3);
    const uint ambB = abs(axisB[b1]) < BoxSupportEpsilon ? b1 : (abs(axisB[b2]) < BoxSupportEpsilon ? b2 : 3);

    float3 supportA = float3(0.0);
    supportA[a1] = axis[a1] >= 0.0 ? extentA[a1] : -extentA[a1];
    supportA[a2] = axis[a2] >= 0.0 ? extentA[a2] : -extentA[a2];
    float3 supportB = float3(0.0);
    supportB[b1] = axisB[b1] >= 0.0 ? -extentB[b1] : extentB[b1];
    supportB[b2] = axisB[b2] >= 0.0 ? -extentB[b2] : extentB[b2];
    const float3 directionB = matrixColumn(axesB, edgeB);
    const float angle = directionB[edgeA];
    const float determinant = 1.0 - angle * angle;
    const float halfA = extentA[edgeA];
    const float halfB = extentB[edgeB];
    float best = 1.0e38;
    float3 bestA = float3(0.0);
    float3 bestB = float3(0.0);
    for (uint i = 0; i < (ambA < 3 ? 2u : 1u); ++i)
    {
        float3 cornerA = supportA;
        if (i != 0)
            cornerA[ambA] = -cornerA[ambA];
        for (uint j = 0; j < (ambB < 3 ? 2u : 1u); ++j)
        {
            float3 localB = supportB;
            if (j != 0)
                localB[ambB] = -localB[ambB];
            const float3 cornerB = mul(axesB, localB) + centerB;
            const float3 delta = cornerB - cornerA;
            const float projectedA = delta[edgeA];
            const float projectedB = dot(directionB, delta);
            const float s = clamp(determinant >= MinValue ? (projectedA - angle * projectedB) / determinant : 0.0,
                                  -halfA, halfA);
            const float t = clamp(angle * s - projectedB, -halfB, halfB);
            float3 witnessA = cornerA;
            witnessA[edgeA] = clamp(projectedA + angle * t, -halfA, halfA);
            const float3 witnessB = cornerB + directionB * t;
            const float3 gap = witnessB - witnessA;
            const float gap2 = dot(gap, gap);
            if (gap2 < best)
            {
                best = gap2;
                bestA = witnessA;
                bestB = witnessB;
            }
        }
    }

    BoxManifold manifold;
    const float distance = dot(bestB - bestA, axis);
    manifold.count = 0;
    if (distance <= tolerance)
    {
        manifold.count = 1;
        manifold.points[0] = float4(first.position.xyz + rotatePoint(first, 0.5 * (bestA + bestB)), distance);
        manifold.normal = rotatePoint(first, axis);
    }
    return manifold;
}

BoxManifold boxBoxContact(CartesianFrame first, float3 extentA, CartesianFrame second, float3 extentB, float margin)
{
    BoxManifold empty;
    empty.count = 0;
    const float3 displacement = second.position.xyz - first.position.xyz;
    const float3 centerB = inverseRotatePoint(first, displacement);
    const float3 centerA = inverseRotatePoint(second, -displacement);
    const float3x3 axesB = transpose(float3x3(inverseRotatePoint(first, frameAxis(second, 0)),
                                              inverseRotatePoint(first, frameAxis(second, 1)),
                                              inverseRotatePoint(first, frameAxis(second, 2))));
    const float tolerance = margin + BoxSeparationEpsilon * dot(extentA + extentB, float3(1.0));

    // Test face axes in reference order. An edge axis must improve on the best face axis.
    float best = -1.0e38;
    uint face = 0;
    for (uint i = 0; i < 6; ++i)
    {
        const uint axis = i % 3;
        const float separation = i < 3 ? abs(centerB[axis]) - extentA[axis] - dot(abs(axesB[axis]), extentB) :
                                  abs(centerA[axis]) - extentB[axis] - dot(abs(matrixColumn(axesB, axis)), extentA);
        if (separation > tolerance)
            return empty;
        if (separation > best)
        {
            best = separation;
            face = i;
        }
    }

    const float faceSeparation = best;
    uint2 edge = uint2(0);
    float3 edgeAxis = float3(0.0);
    bool edgeWon = false;
    for (uint i = 0; i < 3; ++i)
    {
        const uint i1 = (i + 1) % 3;
        const uint i2 = (i + 2) % 3;
        for (uint j = 0; j < 3; ++j)
        {
            float2 axis = float2(-axesB[i2][j], axesB[i1][j]);
            const float norm2 = dot(axis, axis);
            if (norm2 < BoxParallelEpsilon)
                continue;
            axis /= sqrt(norm2);
            const uint j1 = (j + 1) % 3;
            const uint j2 = (j + 2) % 3;
            const float radiusA = extentA[i1] * abs(axis.x) + extentA[i2] * abs(axis.y);
            const float radiusB = extentB[j1] * abs(axis.x * axesB[i1][j1] + axis.y * axesB[i2][j1]) +
                                  extentB[j2] * abs(axis.x * axesB[i1][j2] + axis.y * axesB[i2][j2]);
            const float separation = abs(axis.x * centerB[i1] + axis.y * centerB[i2]) - radiusA - radiusB;
            if (separation > tolerance)
                return empty;
            if (separation - BoxEdgeBias * abs(separation) > best && separation > faceSeparation)
            {
                best = separation;
                edge = uint2(i, j);
                edgeAxis = float3(0.0);
                edgeAxis[i1] = axis.x;
                edgeAxis[i2] = axis.y;
                edgeWon = true;
            }
        }
    }
    if (edgeWon)
    {
        // A near-parallel edge axis adds little when the face has nearly the same depth.
        const float alignment = face < 3 ? abs(edgeAxis[face]) : abs(dot(edgeAxis, matrixColumn(axesB, face - 3)));
        if (alignment <= 0.99 || best >= faceSeparation + 0.05 * abs(faceSeparation) + MinValue)
            return boxEdgeManifold(first, extentA, extentB, centerB, axesB, edge.x, edge.y, edgeAxis, tolerance);
    }

    if (face < 3)
        return boxFaceManifold(first, extentA, extentB, centerB, axesB, face, true, margin);
    return boxFaceManifold(second, extentB, extentA, centerA, transpose(axesB), face - 3, false, margin);
}
