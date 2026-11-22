// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Adapted from MuJoCo Warp _src/collision_gjk.py, signed-volume distance subalgorithm.

bool sameConvexSign(float first, float second)
{
    return (first > 0.0 && second > 0.0) || (first < 0.0 && second < 0.0);
}

float2 projectConvexSegment(float3 first, float3 second, out float3 point)
{
    const float3 edge = second - first;
    const float edge2 = dot(edge, edge);
    if (edge2 > 0.0)
    {
        const float t = -dot(first, edge) / edge2;
        if (t > 0.0 && t < 1.0)
        {
            // Preserve the normal projection on long edges near the origin.
            point = cross(edge, cross(first, edge)) / edge2;
            return float2(1.0 - t, t);
        }
    }
    // The newest vertex is the support point that can reduce the distance.
    point = second;
    return float2(0.0, 1.0);
}

bool projectConvexPlane(float3 first, float3 second, float3 third, out float3 projection)
{
    const float3 ab = second - first;
    const float3 ac = third - first;
    const float3 bc = third - second;
    float3 normal = cross(bc, ab);
    float offset = dot(normal, second);
    float normal2 = dot(normal, normal);
    projection = float3(0.0);
    if (normal2 == 0.0)
        return false;
    if (offset == 0.0 || normal2 <= MinValue)
    {
        normal = cross(ab, ac);
        offset = dot(normal, first);
        normal2 = dot(normal, normal);
        if (normal2 == 0.0)
            return false;
        if (offset == 0.0 || normal2 <= MinValue)
        {
            normal = cross(ac, bc);
            offset = dot(normal, third);
            normal2 = dot(normal, normal);
        }
    }
    if (normal2 == 0.0)
        return false;
    projection = normal * (offset / normal2);
    return true;
}

bool convexTriangleCoordinates(float3 first, float3 second, float3 third, float3 point, out float3 weights)
{
    // Measure both signed areas from the same origin. This keeps the weights
    // normalized even when two supports nearly coincide on a box face.
    const float3 ab = second - first;
    const float3 ac = third - first;
    const float3 area = cross(ab, ac);
    const float3 magnitude = abs(area);
    const uint axis = magnitude.x >= magnitude.y && magnitude.x >= magnitude.z ? 0 :
                      magnitude.y >= magnitude.z ? 1 : 2;
    weights = float3(0.0);
    if (area[axis] == 0.0)
        return false;
    const float3 delta = point - first;
    const float weightB = cross(delta, ac)[axis] / area[axis];
    const float weightC = cross(ab, delta)[axis] / area[axis];
    weights = float3(1.0 - weightB - weightC, weightB, weightC);
    return true;
}

float3 projectConvexTriangle(float3 vertices[3], out float3 projection)
{
    if (!projectConvexPlane(vertices[0], vertices[1], vertices[2], projection))
        return float3(projectConvexSegment(vertices[0], vertices[1], projection), 0.0);
    float3 projected;
    if (!convexTriangleCoordinates(vertices[0], vertices[1], vertices[2], projection, projected))
        return float3(projectConvexSegment(vertices[0], vertices[1], projection), 0.0);
    const bool3 interior = projected > float3(0.0);
    if (all(interior))
        return projected;

    float bestDistance = ConvexFarDistance;
    float3 weights = float3(0.0);
    for (uint excluded = 0; excluded < 3; ++excluded)
    {
        if (interior[excluded])
            continue;
        const uint first = excluded == 0 ? 1 : 0;
        const uint second = excluded <= 1 ? 2 : 1;
        float3 witness;
        const float2 edgeWeights = projectConvexSegment(vertices[first], vertices[second], witness);
        const float distance2 = dot(witness, witness);
        if (distance2 < bestDistance)
        {
            bestDistance = distance2;
            projection = witness;
            weights = float3(0.0);
            weights[first] = edgeWeights.x;
            weights[second] = edgeWeights.y;
        }
    }
    return weights;
}

float3 projectConvexTriangle(float3 vertices[3])
{
    float3 projection;
    return projectConvexTriangle(vertices, projection);
}

float4 projectConvexTetrahedron(float3 vertices[4], out float3 projection)
{
    const float3 a = vertices[0];
    const float3 b = vertices[1];
    const float3 c = vertices[2];
    const float3 d = vertices[3];
    const float4 cofactors = float4(-dot(b, cross(c, d)), dot(a, cross(c, d)),
                                    -dot(a, cross(b, d)), dot(a, cross(b, c)));
    const float volume = cofactors.x + cofactors.y + cofactors.z + cofactors.w;
    const bool4 interior = bool4(sameConvexSign(volume, cofactors.x), sameConvexSign(volume, cofactors.y),
                                 sameConvexSign(volume, cofactors.z), sameConvexSign(volume, cofactors.w));
    projection = float3(0.0);
    if (all(interior))
        return cofactors / volume;

    float bestDistance = ConvexFarDistance;
    float4 weights = float4(0.0);
    for (uint excluded = 0; excluded < 4; ++excluded)
    {
        if (interior[excluded])
            continue;
        const uint3 indices = uint3(excluded == 0 ? 1 : 0, excluded <= 1 ? 2 : 1, excluded <= 2 ? 3 : 2);
        float3 face[3] = {vertices[indices.x], vertices[indices.y], vertices[indices.z]};
        float3 witness;
        const float3 coord = projectConvexTriangle(face, witness);
        const float distance2 = dot(witness, witness);
        if (distance2 < bestDistance)
        {
            bestDistance = distance2;
            projection = witness;
            weights = float4(0.0);
            weights[indices.x] = coord.x;
            weights[indices.y] = coord.y;
            weights[indices.z] = coord.z;
        }
    }
    return weights;
}

float4 projectConvexSimplex(uint count, float3 vertices[4], out float3 projection)
{
    switch (count)
    {
    case 2:
        return float4(projectConvexSegment(vertices[0], vertices[1], projection), 0.0, 0.0);
    case 3:
        float3 face[3] = {vertices[0], vertices[1], vertices[2]};
        return float4(projectConvexTriangle(face, projection), 0.0);
    case 4:
        return projectConvexTetrahedron(vertices, projection);
    default:
        projection = vertices[0];
        return float4(1.0, 0.0, 0.0, 0.0);
    }
}
