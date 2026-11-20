// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Adapted from MuJoCo Warp _src/collision_primitive_core.py, capsule_box.

static const int NoCapsuleBoxFeature = -4;

struct CapsuleBoxFeature
{
    float distance2;
    float capsuleParameter;
    float edgeParameter;
    int type; // negative: capsule end; nonnegative: 3 * box edge region + capsule region
    int face;
    uint edge;
    uint corner;
};

float capsuleBoxDivide(float numerator, float denominator)
{
    return numerator / (denominator == 0.0 ? MinValue : denominator);
}

CapsuleBoxFeature nearestCapsuleBoxFeature(float3 center, float3 halfAxis, float halfLength, float3 extent)
{
    CapsuleBoxFeature best;
    best.distance2 = 1.0e32;
    best.type = NoCapsuleBoxFeature;
    for (int side = -1; side <= 1; side += 2)
    {
        const float3 tip = center + float(side) * halfAxis;
        const float3 projected = clamp(tip, -extent, extent);
        uint outside = 0;
        int face = -1;
        for (uint axis = 0; axis < 3; ++axis)
        {
            if (abs(tip[axis]) > extent[axis])
            {
                ++outside;
                face = int(axis);
            }
        }
        if (outside > 1)
            continue;
        const float3 delta = projected - tip;
        const float distance2 = dot(delta, delta);
        if (distance2 < best.distance2)
        {
            best.distance2 = distance2;
            best.capsuleParameter = float(side);
            best.type = side - 2;
            best.face = face;
        }
    }

    const float capsuleSquared = halfLength * halfLength;
    for (uint corner = 0; corner < 8; ++corner)
    {
        const float3 signs = float3((corner & 1u) != 0 ? 1.0 : -1.0,
                                    (corner & 2u) != 0 ? 1.0 : -1.0,
                                    (corner & 4u) != 0 ? 1.0 : -1.0);
        for (uint edge = 0; edge < 3; ++edge)
        {
            if ((corner & (1u << edge)) != 0)
                continue;
            float3 edgeCenter = signs * extent;
            edgeCenter[edge] = 0.0;
            float3 difference = edgeCenter - center;
            const float halfEdge = extent[edge];
            const float u = -halfEdge * difference[edge];
            const float v = dot(halfAxis, difference);
            const float edgeSquared = halfEdge * halfEdge;
            const float crossTerm = -halfEdge * halfAxis[edge];
            const float determinant = edgeSquared * capsuleSquared - crossTerm * crossTerm;
            if (abs(determinant) < MinValue)
                continue;
            const float inverse = 1.0 / determinant;
            float edgeParameter = (capsuleSquared * u - crossTerm * v) * inverse;
            float capsuleParameter = (edgeSquared * v - crossTerm * u) * inverse;
            int edgeRegion = 1;
            int capsuleRegion = 1;
            if (abs(edgeParameter) > 1.0)
            {
                edgeRegion = edgeParameter > 0.0 ? 2 : 0;
                edgeParameter = clamp(edgeParameter, -1.0, 1.0);
                capsuleParameter = capsuleBoxDivide(v - crossTerm * edgeParameter, capsuleSquared);
            }
            if (abs(capsuleParameter) > 1.0)
            {
                capsuleRegion = capsuleParameter > 0.0 ? 2 : 0;
                capsuleParameter = clamp(capsuleParameter, -1.0, 1.0);
                edgeParameter = capsuleBoxDivide(u - crossTerm * capsuleParameter, edgeSquared);
                if (abs(edgeParameter) > 1.0)
                {
                    edgeRegion = edgeParameter > 0.0 ? 2 : 0;
                    edgeParameter = clamp(edgeParameter, -1.0, 1.0);
                }
            }
            difference -= halfAxis * capsuleParameter;
            difference[edge] += halfEdge * edgeParameter;
            const float distance2 = dot(difference, difference);
            if (distance2 < best.distance2 - MinValue)
            {
                best.distance2 = distance2;
                best.capsuleParameter = capsuleParameter;
                best.edgeParameter = edgeParameter;
                best.type = 3 * edgeRegion + capsuleRegion;
                best.edge = edge;
                best.corner = corner + (1u << edge) * uint(best.type / 6);
            }
        }
    }
    return best;
}

float secondCapsuleBoxOffset(CapsuleBoxFeature feature, float3 center, float3 axis, float3 halfAxis, float3 extent)
{
    const float absent = -4.0;
    if (feature.type < 0)
    {
        if (feature.face < 0)
            return absent;
        const float direction = feature.type == -3 ? 1.0 : -1.0;
        const float3 tip = center - direction * halfAxis;
        float offset = 2.0;
        for (uint i = 0; i < 3; ++i)
        {
            if (int(i) == feature.face)
                continue;
            const float inverse = capsuleBoxDivide(direction, halfAxis[i]);
            const float upper = (extent[i] - tip[i]) * inverse;
            const float lower = (-extent[i] - tip[i]) * inverse;
            if (upper > 0.0)
                offset = min(offset, upper);
            if (lower > 0.0)
                offset = min(offset, lower);
        }
        return direction * offset;
    }

    const uint axisSigns = (halfAxis.x > 0.0 ? 1u : 0u) | (halfAxis.y > 0.0 ? 2u : 0u) |
                           (halfAxis.z > 0.0 ? 4u : 0u);
    uint orientation = axisSigns ^ feature.corner;
    if (feature.type / 3 != 1)
    {
        // A corner feature can contribute a second contact if the axis extends along a face or edge.
        if (orientation == 0 || orientation == 7)
            return absent;
        const bool forward = (orientation & (orientation - 1)) == 0;
        const float direction = forward ? 1.0 : -1.0;
        orientation = forward ? orientation : 7u - orientation;
        const uint along = uint(firstbitlow(orientation));
        if (axis[along] * axis[along] > 0.5)
        {
            const float bound = 2.0 * capsuleBoxDivide(extent[along], abs(halfAxis[along]));
            return direction * min(1.0 - direction * feature.capsuleParameter, bound);
        }
        const uint i = (along + 1) % 3;
        const uint j = (along + 2) % 3;
        const float bound = 2.0 * min(capsuleBoxDivide(extent[i], abs(halfAxis[i])),
                                      capsuleBoxDivide(extent[j], abs(halfAxis[j])));
        return -direction * min(1.0 + direction * feature.capsuleParameter, bound);
    }

    // With an edge feature, the other point is bounded by the adjacent face and edge endpoint.
    const uint edge = feature.edge;
    orientation &= 7u - (1u << edge);
    if (orientation == 0 || (orientation & (orientation - 1)) != 0)
        return absent;
    const uint i = (edge + 1) % 3;
    const uint j = (edge + 2) % 3;
    const uint face = abs(axis[i]) > abs(axis[j]) ? i : j;
    const bool forward = (orientation & (1u << face)) != 0;
    const float direction = forward ? 1.0 : -1.0;
    const float remaining = 1.0 - direction * feature.capsuleParameter;
    const float faceBound = 2.0 * capsuleBoxDivide(extent[face], abs(halfAxis[face]));
    const bool towardUpper = ((axisSigns & (1u << edge)) != 0) == forward;
    const float edgeRemaining = 1.0 + (towardUpper ? -feature.edgeParameter : feature.edgeParameter);
    const float edgeBound = extent[edge] * capsuleBoxDivide(edgeRemaining, abs(halfAxis[edge]));
    return direction * min(remaining, min(faceBound, edgeBound));
}

uint capsuleBoxContacts(CartesianFrame capsule, float2 size, CartesianFrame box, float3 extent,
                        out PrimitivePoint first, out PrimitivePoint second)
{
    const float3 center = inverseRotatePoint(box, capsule.position.xyz - box.position.xyz);
    const float3 axis = inverseRotatePoint(box, frameAxis(capsule, 2));
    const float3 halfAxis = axis * size.y;
    const CapsuleBoxFeature feature = nearestCapsuleBoxFeature(center, halfAxis, size.y, extent);
    first = {float3(0.0), float3(0.0), 0.0};
    second = first;
    if (feature.type == NoCapsuleBoxFeature)
        return 0;
    const float3 firstCenter = center + halfAxis * feature.capsuleParameter;
    first = sphereBoxContact(box.position.xyz + rotatePoint(box, firstCenter), size.x, box, extent);
    const float offset = secondCapsuleBoxOffset(feature, center, axis, halfAxis, extent);
    if (offset <= -3.0)
        return 1;
    const float3 secondCenter = center + halfAxis * (feature.capsuleParameter + offset);
    second = sphereBoxContact(box.position.xyz + rotatePoint(box, secondCenter), size.x, box, extent);
    return 2;
}
