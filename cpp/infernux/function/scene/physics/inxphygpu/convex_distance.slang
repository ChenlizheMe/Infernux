// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Adapted from MuJoCo Warp _src/collision_gjk.py, gjk and gjk_phase.

#include "convex_support.slang"

static const float ConvexFarDistance = 1.0e30;

#include "convex_simplex.slang"

struct ConvexSimplexVertex
{
    float3 first;
    float3 second;
    int2 indices;
};

struct ConvexDistance
{
    ConvexSimplexVertex simplex[4];
    float3 first;
    float3 second;
    float distance;
    uint count;
    bool separated;
};

bool convexDiscreteGeometry(ConvexGeometry geometry)
{
    return geometry.margin == 0.0 && (geometry.shape.type.x == GeomBox || geometry.shape.type.x == GeomMesh ||
                                     geometry.shape.type.x == GeomHeightfield || geometry.shape.type.x == GeomTriangle);
}

bool convexFaceGeometry(uint kind)
{
    return kind == GeomBox || kind == GeomMesh || kind == GeomCylinder || kind == GeomTriangle;
}

float3 tuneConvexDirection(float3 direction, float norm, float3 simplex[4], uint count, out float3 search)
{
    const float3 unitDirection = direction / norm;
    search = direction;
    if (count == 2)
    {
        const float3 edge = simplex[1] - simplex[0];
        const float edge2 = dot(edge, edge);
        if (edge2 > MinValue * MinValue)
        {
            const float3 adjusted = unitDirection - (dot(unitDirection, edge) / edge2) * edge;
            const float adjustedNorm = length(adjusted);
            if (adjustedNorm <= MinValue)
                return adjusted;
            const float3 unit = adjusted / adjustedNorm;
            search = unit * norm;
            return unit;
        }
    }
    else if (count == 3)
    {
        const float3 normal = cross(simplex[1] - simplex[0], simplex[2] - simplex[0]);
        const float normalLength = length(normal);
        if (normalLength > MinValue)
        {
            const float3 unit = normal * (convexSign(dot(unitDirection, normal)) / normalLength);
            search = unit * norm;
            return unit;
        }
    }
    return unitDirection;
}

ConvexDistance gjkDistance(inout ConvexGeometry first, inout ConvexGeometry second, float tolerance,
                           float cutoff, uint iterations, StructuredBuffer<uint4> model)
{
    const bool discrete = convexDiscreteGeometry(first) && convexDiscreteGeometry(second);
    const float gapTolerance = discrete ? 0.0 : 0.5 * tolerance * tolerance;
    const float minNorm = discrete ? MinValue : tolerance;
    float3 direction = first.frame.position.xyz - second.frame.position.xyz;
    float norm = length(direction);
    float previousNorm = 0.0;
    float4 weights = float4(1.0, 0.0, 0.0, 0.0);
    float3 simplex[4];
    ConvexDistance result;
    result.count = 0;
    for (uint step = 0; step < iterations; ++step)
    {
        if (norm < minNorm || abs(previousNorm - norm) < MinValue)
            break;
        float3 search = direction;
        float3 supportDirection = direction / norm;
        if (discrete && norm < 1.0e-4)
            supportDirection = tuneConvexDirection(direction, norm, simplex, result.count, search);
        const ConvexSupportPoint a = convexSupport(first, -supportDirection, model);
        const ConvexSupportPoint b = convexSupport(second, supportDirection, model);
        first.cachedVertex = a.cachedVertex;
        second.cachedVertex = b.cachedVertex;
        const uint next = result.count;
        result.simplex[next] = {a.point, b.point, int2(a.vertex, b.vertex)};
        simplex[next] = a.point - b.point;

        // Frank–Wolfe gap and the support-plane lower bound use the searched direction.
        if (dot(search, search - simplex[next]) < gapTolerance)
            break;
        const float lower = dot(search, simplex[next]);
        if (lower > 0.0 && (cutoff == 0.0 || (cutoff < ConvexFarDistance && lower >= cutoff * norm)))
        {
            result.count = 0;
            result.first = float3(0.0);
            result.second = float3(0.0);
            result.distance = ConvexFarDistance;
            result.separated = true;
            return result;
        }

        weights = projectConvexSimplex(next + 1, simplex);
        result.count = 0;
        float3 closest = float3(0.0);
        for (uint i = 0; i <= next; ++i)
        {
            if (weights[i] == 0.0)
                continue;
            const uint index = result.count++;
            result.simplex[index] = result.simplex[i];
            simplex[index] = simplex[i];
            weights[index] = weights[i];
            closest += weights[index] * simplex[index];
        }
        if (result.count == 0)
            break;
        direction = closest;
        previousNorm = step == 0 ? 0.0 : norm;
        norm = length(direction);
        if (result.count == 4)
            break;
    }

    result.first = result.count == 0 ? first.frame.position.xyz : float3(0.0);
    result.second = result.count == 0 ? second.frame.position.xyz : float3(0.0);
    for (uint i = 0; i < result.count; ++i)
    {
        result.first += weights[i] * result.simplex[i].first;
        result.second += weights[i] * result.simplex[i].second;
    }
    result.separated = false;
    if (norm > 0.0)
    {
        const float3 unitDirection = direction / norm;
        const ConvexSupportPoint a = convexSupport(first, -unitDirection, model);
        const ConvexSupportPoint b = convexSupport(second, unitDirection, model);
        result.separated = dot(direction, a.point - b.point) > 0.0;
    }
    result.distance = result.count == 4 && !result.separated ? 0.0 : norm;
    return result;
}

float shrinkConvexGeometry(inout ConvexGeometry geometry, bool polygonalPair)
{
    const uint kind = geometry.shape.type.x;
    if (kind == GeomSphere || kind == GeomCapsule)
    {
        const float thickness = geometry.shape.size.x + 0.5 * geometry.margin;
        geometry.shape.size.x = 0.0;
        geometry.margin = 0.0;
        return thickness;
    }
    if (polygonalPair)
    {
        const float thickness = 0.5 * geometry.margin;
        geometry.margin = 0.0;
        return thickness;
    }
    return 0.0;
}

// Use frames rebased to the first shape's origin, including for the EPA consumer.
ConvexDistance convexDistancePhase(inout ConvexGeometry first, inout ConvexGeometry second, float tolerance,
                                   float cutoff, uint iterations, StructuredBuffer<uint4> model, out bool needsPenetration)
{
    needsPenetration = false;
    if (first.margin > 0.0 || second.margin > 0.0 || first.shape.type.x <= GeomCapsule ||
        second.shape.type.x <= GeomCapsule)
    {
        const bool faceA = convexFaceGeometry(first.shape.type.x);
        const bool faceB = convexFaceGeometry(second.shape.type.x);
        const bool polygonalPair = (faceA && (faceB || second.shape.type.x == GeomCapsule)) ||
                                   (faceB && first.shape.type.x == GeomCapsule);
        ConvexGeometry reducedA = first;
        ConvexGeometry reducedB = second;
        const float inflateA = shrinkConvexGeometry(reducedA, polygonalPair);
        const float inflateB = shrinkConvexGeometry(reducedB, polygonalPair);
        const float thickness = inflateA + inflateB;
        if (thickness > 0.0)
        {
            ConvexDistance result = gjkDistance(reducedA, reducedB, tolerance, cutoff + thickness, iterations,
                                               model);
            first.cachedVertex = reducedA.cachedVertex;
            second.cachedVertex = reducedB.cachedVertex;
            if (result.distance > tolerance)
            {
                if (result.distance < ConvexFarDistance)
                {
                    const float3 delta = result.second - result.first;
                    const float norm2 = dot(delta, delta);
                    const float3 normal = norm2 > 0.0 ? delta / sqrt(norm2) : float3(0.0);
                    result.first += inflateA * normal;
                    result.second -= inflateB * normal;
                    result.distance -= thickness;
                }
                return result;
            }
        }
    }
    const ConvexDistance result = gjkDistance(first, second, tolerance, cutoff, iterations, model);
    needsPenetration = result.distance <= tolerance && result.count >= 2 && !result.separated;
    return result;
}
