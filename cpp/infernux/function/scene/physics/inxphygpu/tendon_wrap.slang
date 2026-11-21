// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/util_misc.py tendon wrapping.

struct TendonCircleSolution
{
    float2 pointA;
    float2 pointB;
    float arcLength;
};

struct TendonWrapSolution
{
    float3 pointA;
    float3 pointB;
    float arcLength;
};

static const TendonCircleSolution ClearTendonCircle = {float2(0.0), float2(0.0), -1.0};
static const TendonWrapSolution ClearTendonWrap = {float3(0.0), float3(0.0), -1.0};

float tendonCross2(float2 a, float2 b)
{
    return a.x * b.y - a.y * b.x;
}

bool tendonTangentsIntersect(float2 a, float2 tangentA, float2 b, float2 tangentB)
{
    const float2 u = tangentA - a;
    const float2 v = tangentB - b;
    const float determinant = tendonCross2(u, v);
    if (abs(determinant) < TendonMinLength)
        return false;
    const float orientation = sign(determinant);
    const float2 starts = a - b;
    const float2 ends = tangentA - tangentB;
    return tendonCross2(v, starts) * orientation >= 0.0 &&
           tendonCross2(v, ends) * orientation < 0.0 &&
           tendonCross2(u, starts) * orientation >= 0.0 &&
           tendonCross2(u, ends) * orientation < 0.0;
}

float tendonWinding(float2 a, float2 b)
{
    return atan2(tendonCross2(a, b), dot(a, b));
}

TendonCircleSolution wrapTendonCircle(float2 a, float2 b, float radius, bool guided, float2 side)
{
    const float radiusSquared = radius * radius;
    const float squaredA = dot(a, a);
    const float squaredB = dot(b, b);
    const float2 line = b - a;
    const float squaredLine = dot(line, line);
    if (squaredA < radiusSquared || squaredB < radiusSquared ||
        radius < TendonMinLength || squaredLine < TendonMinLength)
        return ClearTendonCircle;

    const float2 closest = a + saturate(-dot(line, a) / squaredLine) * line;
    const bool clear = dot(closest, closest) > radiusSquared;
    const float winding = guided ? tendonWinding(a, side) + tendonWinding(side, b) : 0.0;
    if (clear && (!guided || abs(winding - tendonWinding(a, b)) < PhysicsPi))
        return ClearTendonCircle;

    const float2 radialA = a * radiusSquared;
    const float2 radialB = b * radiusSquared;
    const float2 normalA = float2(a.y, -a.x) * radius * sqrt(squaredA - radiusSquared);
    const float2 normalB = float2(b.y, -b.x) * radius * sqrt(squaredB - radiusSquared);
    const float2 clockwiseA = (radialA + normalA) / squaredA;
    const float2 clockwiseB = (radialB - normalB) / squaredB;
    const float2 counterA = (radialA - normalA) / squaredA;
    const float2 counterB = (radialB + normalB) / squaredB;
    const bool intersectsClockwise = tendonTangentsIntersect(a, clockwiseA, b, clockwiseB);
    const bool intersectsCounter = tendonTangentsIntersect(a, counterA, b, counterB);
    const float2 clockwiseChord = clockwiseA - clockwiseB;
    const float2 counterChord = counterA - counterB;
    const float clockwiseScore = intersectsClockwise ? -10000.0 : -dot(clockwiseChord, clockwiseChord);
    const float counterScore = intersectsCounter ? -10000.0 : -dot(counterChord, counterChord);
    const bool counter = guided ? winding > 0.0 : clockwiseScore <= counterScore;
    if (counter ? intersectsCounter : intersectsClockwise)
        return ClearTendonCircle;

    const float2 pointA = counter ? counterA : clockwiseA;
    const float2 pointB = counter ? counterB : clockwiseB;
    if (guided)
        return {pointA, pointB, radius * abs(winding - tendonWinding(a, pointA) - tendonWinding(pointB, b))};

    const float crossValue = tendonCross2(pointA, pointB);
    float angle = atan2(abs(crossValue), dot(pointA, pointB));
    if (angle > 1.0e-6 && (counter ? crossValue < 0.0 : crossValue > 0.0))
        angle = 2.0 * PhysicsPi - angle;
    return {pointA, pointB, radius * angle};
}

TendonCircleSolution wrapTendonInside(float2 a, float2 b, float radius)
{
    const float lengthA = length(a);
    const float lengthB = length(b);
    if (radius < TendonMinLength || lengthA <= radius || lengthB <= radius)
        return ClearTendonCircle;

    const float2 line = b - a;
    const float squaredLine = dot(line, line);
    if (squaredLine > TendonMinLength)
    {
        const float fraction = -dot(line, a) / squaredLine;
        if (fraction > 0.0 && fraction < 1.0 && length(a + fraction * line) <= radius)
            return ClearTendonCircle;
    }

    const float cosine = (lengthA * lengthA + lengthB * lengthB - squaredLine) / (2.0 * lengthA * lengthB);
    if (cosine <= -1.0)
        return ClearTendonCircle;
    const float2 midpoint = radius * normalize(a + b);
    const TendonCircleSolution midpointSolution = {midpoint, midpoint, 0.0};
    if (cosine >= 1.0)
        return midpointSolution;

    const float2 ratio = radius / float2(lengthA, lengthB);
    const float2 ratioSquared = ratio * ratio;
    const float angle = acos(cosine);
    const float tolerance = 1.0e-6;
    float phi = 0.5 * PhysicsPi;
    float z = 1.0;
    float2 endpointAngles = asin(ratio);
    float residual = endpointAngles.x + endpointAngles.y - PhysicsPi + angle;
    if (residual > 0.0)
        return midpointSolution;

    // Solve in angle space to avoid the singular slope of asin at z = 1.
    uint iteration = 0;
    while (iteration < 20u && abs(residual) > tolerance)
    {
        const float2 slopes = ratio / max(float2(TendonMinLength), sqrt(1.0 - z * z * ratioSquared));
        const float derivative = cos(phi) * (slopes.x + slopes.y) - 2.0;
        if (derivative > -TendonMinLength)
            return midpointSolution;
        const float next = phi - residual / derivative;
        if (next > phi)
            return midpointSolution;
        phi = next;
        z = sin(phi);
        endpointAngles = asin(ratio * z);
        residual = endpointAngles.x + endpointAngles.y - 2.0 * phi + angle;
        if (residual > tolerance)
            return midpointSolution;
        ++iteration;
    }
    if (iteration == 20u)
        return midpointSolution;

    const bool fromA = tendonCross2(a, b) > 0.0;
    const float2 endpoint = normalize(fromA ? a : b);
    const float turn = phi - (fromA ? endpointAngles.x : endpointAngles.y);
    const float2 point = radius * (cos(turn) * endpoint + sin(turn) * float2(-endpoint.y, endpoint.x));
    return {point, point, 0.0};
}

TendonWrapSolution solveTendonWrap(float3 pointA, float3 pointB, float3 center, float4 rotation,
                                  float radius, bool cylinder, bool guided, float3 guide)
{
    const float4 inverseRotation = float4(rotation.x, -rotation.yzw);
    const float3 localA = rotateVector(pointA - center, inverseRotation);
    const float3 localB = rotateVector(pointB - center, inverseRotation);
    const float lengthA = length(localA);
    if (lengthA < TendonMinLength || length(localB) < TendonMinLength)
        return ClearTendonWrap;

    float3 axisX = float3(1.0, 0.0, 0.0);
    float3 axisY = float3(0.0, 1.0, 0.0);
    if (!cylinder)
    {
        axisX = localA / lengthA;
        float3 normal = cross(localA, localB);
        if (length(normal) < TendonMinLength)
        {
            const float3 magnitude = abs(axisX);
            float3 direction = float3(0.0, 1.0, 1.0);
            if (magnitude.y > max(magnitude.x, magnitude.z))
                direction = float3(1.0, 0.0, 1.0);
            if (magnitude.z > max(magnitude.x, magnitude.y))
                direction = float3(1.0, 1.0, 0.0);
            normal = cross(axisX, direction);
        }
        axisY = normalize(cross(normalize(normal), axisX));
    }

    const float2 a = float2(dot(localA, axisX), dot(localA, axisY));
    const float2 b = float2(dot(localB, axisX), dot(localB, axisY));
    TendonCircleSolution arc;
    if (guided)
    {
        const float3 localGuide = rotateVector(guide - center, inverseRotation);
        if (length(localGuide) < radius)
            arc = wrapTendonInside(a, b, radius);
        else
        {
            const float2 side = float2(dot(localGuide, axisX), dot(localGuide, axisY));
            const float sideLength = length(side);
            const float2 projectedSide = radius * (sideLength < TendonMinLength ? float2(1.0, 0.0) : side / sideLength);
            arc = wrapTendonCircle(a, b, radius, true, projectedSide);
        }
    }
    else
        arc = wrapTendonCircle(a, b, radius, false, float2(0.0));
    if (arc.arcLength < 0.0)
        return ClearTendonWrap;

    float3 entry = axisX * arc.pointA.x + axisY * arc.pointA.y;
    float3 exit = axisX * arc.pointB.x + axisY * arc.pointB.y;
    float arcLength = arc.arcLength;
    if (cylinder)
    {
        const float tangentA = length(localA.xy - entry.xy);
        const float tangentB = length(localB.xy - exit.xy);
        const float developedLength = tangentA + arcLength + tangentB;
        const float slope = (localB.z - localA.z) / (developedLength == 0.0 ? TendonMinLength : developedLength);
        entry.z = localA.z + slope * tangentA;
        exit.z = localA.z + slope * (tangentA + arcLength);
        arcLength = length(float2(arcLength, exit.z - entry.z));
    }
    return {center + rotateVector(entry, rotation), center + rotateVector(exit, rotation), arcLength};
}
