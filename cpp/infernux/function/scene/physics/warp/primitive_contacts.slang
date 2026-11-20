// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/collision_{core,primitive,primitive_core}.py.

#include "model.slang"
#include "contact.slang"

static const uint GeomPlane = 0;
static const uint GeomSphere = 2;
static const uint GeomCapsule = 3;
static const uint GeomEllipsoid = 4;
static const uint GeomCylinder = 5;
static const uint GeomBox = 6;
static const float MinFriction = 1.0e-5;
static const float MinValue = 1.0e-15;

struct PrimitiveShape
{
    float4 size;
    uint4 type;
};

struct ContactMaterial
{
    float4 parameters; // margin, gap, solmix, adhesion
    float4 reference; // solref, solreffriction
    float4 impedance;
    float4 friction;
    float4 extra; // fifth solimp, fifth friction
    int4 dimensions; // condim, geom priority
};

struct ContactBasis
{
    float3 normal;
    float3 tangent;
    float3 bitangent;
};

struct PrimitivePoint
{
    float3 point;
    float3 normal;
    float distance;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint4> contactSettings; // geom count, candidate capacity, contact capacity, workers
[[vk::binding(1, 0)]] StructuredBuffer<CartesianFrame> geometryFrames;
[[vk::binding(2, 0)]] StructuredBuffer<PrimitiveShape> shapes;
[[vk::binding(3, 0)]] StructuredBuffer<uint4> collisionPair;
[[vk::binding(4, 0)]] RWStructuredBuffer<Atomic<uint>> collisionCount;
[[vk::binding(5, 0)]] StructuredBuffer<ContactMaterial> materials;
[[vk::binding(6, 0)]] RWStructuredBuffer<Contact> contacts;
[[vk::binding(7, 0)]] RWStructuredBuffer<Atomic<uint>> contactCount;

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

#include "box_contacts.slang"

PrimitivePoint sphereContact(float3 centerA, float radiusA, float3 centerB, float radiusB)
{
    const float3 difference = centerB - centerA;
    const float separation = length(difference);
    const float3 normal = separation == 0.0 ? float3(1, 0, 0) : difference / separation;
    const float distance = separation - (radiusA + radiusB);
    return {centerA + normal * (radiusA + 0.5 * distance), normal, distance};
}

PrimitivePoint sphereCylinderContact(float3 sphere, float radius, CartesianFrame cylinder, float2 size)
{
    const float3 center = cylinder.position.xyz;
    const float3 axis = frameAxis(cylinder, 2);
    const float3 offset = sphere - center;
    const float axial = dot(offset, axis);
    const float3 axialOffset = axis * axial;
    const float3 radialOffset = offset - axialOffset;
    const float radial2 = dot(radialOffset, radialOffset);
    const bool withinCaps = abs(axial) < size.y;
    const bool withinWall = radial2 < size.x * size.x;
    if (withinCaps && (!withinWall || size.x - sqrt(radial2) <= size.y - abs(axial)))
        return sphereContact(sphere, radius, center + axialOffset, size.x);

    if (withinWall)
    {
        const float3 normal = axial > 0.0 ? axis : -axis;
        const float distance = dot(offset, normal) - size.y - radius;
        return {sphere - normal * (radius + 0.5 * distance), -normal, distance};
    }

    const float3 rim = center + axis * (sign(axial) * size.y) + radialOffset * (size.x / sqrt(radial2));
    return sphereContact(sphere, radius, rim, 0.0);
}

PrimitivePoint sphereBoxContact(float3 sphere, float radius, CartesianFrame box, float3 extent)
{
    const float3 center = inverseRotatePoint(box, sphere - box.position.xyz);
    const float3 clamped = clamp(center, -extent, extent);
    const float3 delta = clamped - center;
    const float distance = length(delta);
    if (distance > MinValue)
    {
        const float3 normal = delta / distance;
        const float3 point = 0.5 * (clamped + center + normal * radius);
        return {box.position.xyz + rotatePoint(box, point), rotatePoint(box, normal), distance - radius};
    }

    // Center inside the box: choose the nearest face in Warp's fixed traversal order.
    float closest = 2.0 * (extent.x + extent.y + extent.z);
    uint face = 0;
    for (uint i = 0; i < 6; ++i)
    {
        const uint axis = i / 2;
        const float side = (i & 1u) != 0 ? 1.0 : -1.0;
        const float faceDistance = abs(side * extent[axis] - center[axis]);
        if (faceDistance < closest)
        {
            closest = faceDistance;
            face = i;
        }
    }
    float3 normal = float3(0.0);
    normal[face / 2] = (face & 1u) != 0 ? -1.0 : 1.0;
    const float3 point = center + 0.5 * (radius - closest) * normal;
    return {box.position.xyz + rotatePoint(box, point), rotatePoint(box, normal), -closest - radius};
}

#include "capsule_box.slang"

float2 clampSegmentWeights(float s, float t, float s0, float s1)
{
    if (t < 0.0)
        return float2(saturate(s0), 0.0);
    if (t > 1.0)
        return float2(saturate(s1), 1.0);
    return float2(s, t);
}

uint segmentWeights(float3 a, float3 ab, float3 c, float3 cd, out float2 first, out float2 second)
{
    const float3 ca = a - c;
    const float abSquared = dot(ab, ab);
    const float cdSquared = dot(cd, cd);
    const float cdCa = dot(cd, ca);
    const float abCa = dot(ab, ca);
    const float abCd = dot(ab, cd);
    second = float2(0.0);
    if (abSquared <= MinValue)
    {
        first = float2(0.0, cdSquared <= MinValue ? 0.0 : saturate(cdCa / cdSquared));
        return 1;
    }
    if (cdSquared <= MinValue)
    {
        first = float2(saturate(-abCa / abSquared), 0.0);
        return 1;
    }
    const float s0 = -abCa / abSquared;
    const float s1 = (abCd - abCa) / abSquared;
    const float3 area = cross(ab, cd);
    const float determinant = dot(area, area);
    if (determinant > 1.0e-8 * abSquared * cdSquared)
    {
        const float s = saturate(dot(area, cross(cd, ca)) / determinant);
        first = clampSegmentWeights(s, (abCd * s + cdCa) / cdSquared, s0, s1);
        return 1;
    }
    first = clampSegmentWeights(0.0, cdCa / cdSquared, s0, s1);
    second = clampSegmentWeights(1.0, (abCd + cdCa) / cdSquared, s0, s1);
    return 2;
}

float3 basisReference(float3 normal)
{
    return abs(normal.y) < 0.5 ? float3(0, 1, 0) : float3(0, 0, 1);
}

ContactBasis makeBasis(float3 normal)
{
    normal = normalize(normal);
    const float3 reference = basisReference(normal);
    const float3 tangent = normalize(reference - normal * dot(normal, reference));
    return {normal, tangent, cross(normal, tangent)};
}

ContactMaterial resolveMaterial(uint4 pair, uint geomCount)
{
    const int explicitPair = asint(pair.z);
    if (explicitPair >= 0)
        return materials[geomCount + uint(explicitPair)];

    const ContactMaterial a = materials[pair.x];
    const ContactMaterial b = materials[pair.y];
    const float wa = a.parameters.z;
    const float wb = b.parameters.z;
    float weight = 0.5;
    if (wa >= MinValue || wb >= MinValue)
        weight = wa < MinValue ? 0.0 : (wb < MinValue ? 1.0 : wa / (wa + wb));
    int dim = max(a.dimensions.x, b.dimensions.x);
    float3 friction = max(a.friction.xyz, b.friction.xyz);
    float adhesion = a.parameters.w + b.parameters.w;
    if (a.dimensions.y != b.dimensions.y)
    {
        const ContactMaterial preferred = a.dimensions.y > b.dimensions.y ? a : b;
        weight = a.dimensions.y > b.dimensions.y ? 1.0 : 0.0;
        dim = preferred.dimensions.x;
        friction = preferred.friction.xyz;
        adhesion = preferred.parameters.w;
    }

    ContactMaterial result;
    result.dimensions = int4(dim, 0, 0, 0);
    result.parameters = float4(a.parameters.xy + b.parameters.xy, 0.0, adhesion);
    const float2 reference = a.reference.x > 0.0 && b.reference.x > 0.0 ?
                                 lerp(b.reference.xy, a.reference.xy, weight) : min(a.reference.xy, b.reference.xy);
    result.reference = float4(reference, 0.0, 0.0);
    result.impedance = lerp(b.impedance, a.impedance, weight);
    result.friction = float4(friction.x, friction.x, friction.y, friction.z);
    result.extra = float4(lerp(b.extra.x, a.extra.x, weight), friction.z, 0.0, 0.0);
    return result;
}

void writeContact(uint world, uint4 pair, uint ordinal, float distance, float3 point, ContactBasis basis,
                  ContactMaterial material, uint capacity)
{
    const float margin = material.parameters.x;
    const bool detected = distance < margin + material.parameters.y;
    const bool constraint = asint(pair.z) >= -1 && detected;
    const bool sensor = asint(pair.w) >= 0;
    if (!constraint && !sensor)
        return;

    const uint slot = contactCount[world].add(1u);
    if (slot >= capacity)
        return;
    const uint dim = material.parameters.w != 0.0 && distance >= margin ? 1u : uint(material.dimensions.x);
    Contact contact;
    contact.positionDistance = float4(point, distance);
    contact.normalMargin = float4(basis.normal, margin);
    contact.tangentAdhesion = float4(basis.tangent, material.parameters.w);
    contact.bitangent = float4(basis.bitangent, 0.0);
    contact.reference = material.reference;
    contact.impedance = material.impedance;
    contact.friction = max(material.friction, float4(MinFriction));
    contact.extra = float4(material.extra.x, max(material.extra.y, MinFriction), 0.0, 0.0);
    contact.geometry = uint4(pair.xy, dim, (constraint ? ContactConstraint : 0u) | (sensor ? ContactSensor : 0u));
    contact.identity = uint4(ordinal, pair.w, 0, 0);
    contacts[world * capacity + slot] = contact;
}

void writePlaneCylinderContacts(uint world, uint4 pair, float3 plane, ContactBasis basis,
                                CartesianFrame cylinder, float2 size, ContactMaterial material, uint capacity)
{
    const float3 normal = basis.normal;
    float3 axis = frameAxis(cylinder, 2);
    const float projection = dot(normal, axis);
    axis = projection > 0.0 ? -axis : axis;
    const float towardPlane = -abs(projection);
    const float3 endOffset = axis * size.y;
    const float centerDistance = dot(cylinder.position.xyz - plane, normal);
    const float capOffset = towardPlane * size.y;
    float3 radial = axis * towardPlane - normal;
    if (dot(radial, radial) < 1.0e-12)
    {
        // A flat end cap admits any in-plane direction. Keep the ring on rotated caps as well.
        const float3 preferred = float3(1.0, 0.0, 0.0) - axis * axis.x;
        radial = dot(preferred, preferred) >= 1.0e-12 ? preferred : basis.tangent;
    }
    radial = normalize(radial) * size.x;
    const float radialDistance = dot(radial, normal);
    const float3 lateral = normalize(cross(radial, axis)) * (size.x * sqrt(3.0) * 0.5);
    const float3 nearCap = cylinder.position.xyz + endOffset;
    const float nearDistance = centerDistance + capOffset + radialDistance;
    const float farDistance = centerDistance - capOffset + radialDistance;
    const float sideDistance = centerDistance + capOffset - 0.5 * radialDistance;
    const float3 sidePoint = nearCap - 0.5 * radial - normal * (0.5 * sideDistance);
    writeContact(world, pair, 0, nearDistance, nearCap + radial - normal * (0.5 * nearDistance), basis, material,
                 capacity);
    writeContact(world, pair, 1, farDistance,
                 cylinder.position.xyz - endOffset + radial - normal * (0.5 * farDistance), basis, material, capacity);
    writeContact(world, pair, 2, sideDistance, sidePoint + lateral, basis, material, capacity);
    writeContact(world, pair, 3, sideDistance, sidePoint - lateral, basis, material, capacity);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void clear_contact_count(uint3 id : SV_DispatchThreadID)
{
    contactCount[id.y].store(0u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void collide_primitives(uint3 id : SV_DispatchThreadID)
{
    const uint4 settings = contactSettings[0];
    if (id.x >= settings.w)
        return;
    const uint count = min(collisionCount[id.y].load(), settings.y);
    const uint geometryBase = id.y * settings.x;
    for (uint item = id.x; item < count; item += settings.w)
    {
        const uint4 pair = collisionPair[id.y * settings.y + item];
        const PrimitiveShape a = shapes[pair.x];
        const PrimitiveShape b = shapes[pair.y];
        const uint typeA = a.type.x;
        const uint typeB = b.type.x;
        const bool planePair = typeA == GeomPlane && (typeB >= GeomSphere && typeB <= GeomBox);
        const bool spherePair = typeA == GeomSphere &&
                                (typeB == GeomSphere || typeB == GeomCapsule || typeB == GeomCylinder || typeB == GeomBox);
        const bool capsulePair = typeA == GeomCapsule && (typeB == GeomCapsule || typeB == GeomBox);
        const bool boxPair = typeA == GeomBox && typeB == GeomBox;
        if (!planePair && !spherePair && !capsulePair && !boxPair)
            continue;

        const CartesianFrame frameA = geometryFrames[geometryBase + pair.x];
        const CartesianFrame frameB = geometryFrames[geometryBase + pair.y];
        const float3 centerA = frameA.position.xyz;
        const float3 centerB = frameB.position.xyz;
        const ContactMaterial material = resolveMaterial(pair, settings.x);
        if (boxPair)
        {
            const float margin = material.parameters.x + material.parameters.y;
            const BoxManifold manifold = boxBoxContact(frameA, a.size.xyz, frameB, b.size.xyz, margin);
            if (manifold.count != 0)
            {
                const ContactBasis basis = makeBasis(manifold.normal);
                for (uint i = 0; i < manifold.count; ++i)
                {
                    const float4 point = manifold.points[i];
                    writeContact(id.y, pair, i, point.w, point.xyz, basis, material, settings.z);
                }
            }
            continue;
        }
        if (spherePair)
        {
            PrimitivePoint result;
            if (typeB == GeomBox)
                result = sphereBoxContact(centerA, a.size.x, frameB, b.size.xyz);
            else if (typeB == GeomCylinder)
                result = sphereCylinderContact(centerA, a.size.x, frameB, b.size.xy);
            else
            {
                float3 closest = centerB;
                if (typeB == GeomCapsule)
                {
                    const float3 axis = frameAxis(frameB, 2);
                    const float along = clamp(dot(centerA - centerB, axis), -b.size.y, b.size.y);
                    closest += axis * along;
                }
                result = sphereContact(centerA, a.size.x, closest, b.size.x);
            }
            writeContact(id.y, pair, 0, result.distance, result.point, makeBasis(result.normal), material, settings.z);
            continue;
        }
        if (capsulePair)
        {
            if (typeB == GeomBox)
            {
                PrimitivePoint first, second;
                const uint points = capsuleBoxContacts(frameA, a.size.xy, frameB, b.size.xyz, first, second);
                for (uint i = 0; i < points; ++i)
                {
                    const PrimitivePoint result = i == 0 ? first : second;
                    writeContact(id.y, pair, i, result.distance, result.point, makeBasis(result.normal), material,
                                 settings.z);
                }
                continue;
            }
            const float3 halfA = frameAxis(frameA, 2) * a.size.y;
            const float3 halfB = frameAxis(frameB, 2) * b.size.y;
            const float3 startA = centerA - halfA;
            const float3 startB = centerB - halfB;
            const float3 spanA = (centerA + halfA) - startA;
            const float3 spanB = (centerB + halfB) - startB;
            float2 first, second;
            const uint points = segmentWeights(startA, spanA, startB, spanB, first, second);
            for (uint i = 0; i < points; ++i)
            {
                const float2 weights = i == 0 ? first : second;
                const PrimitivePoint result = sphereContact(startA + spanA * weights.x, a.size.x,
                                                            startB + spanB * weights.y, b.size.x);
                writeContact(id.y, pair, i, result.distance, result.point, makeBasis(result.normal), material, settings.z);
            }
            continue;
        }

        const float3 normal = frameAxis(frameA, 2);
        ContactBasis basis = makeBasis(normal);
        if (typeB == GeomCylinder)
        {
            writePlaneCylinderContacts(id.y, pair, centerA, basis, frameB, b.size.xy, material, settings.z);
            continue;
        }
        const float3 capsuleAxis = frameAxis(frameB, 2);
        uint pointCount = 1;
        if (typeB == GeomCapsule)
        {
            const float3 projected = capsuleAxis - normal * dot(normal, capsuleAxis);
            const float projectedLength = length(projected);
            const float3 tangent = projectedLength < 0.5 ? basisReference(normal) : projected / projectedLength;
            basis = {normal, tangent, cross(normal, tangent)};
            pointCount = 2;
        }
        else if (typeB == GeomBox)
            pointCount = 8;

        for (uint pointId = 0; pointId < pointCount; ++pointId)
        {
            float3 closest = centerB;
            if (typeB == GeomSphere || typeB == GeomCapsule)
            {
                if (typeB == GeomCapsule)
                    closest += capsuleAxis * (pointId == 0 ? b.size.y : -b.size.y);
                closest -= normal * b.size.x;
            }
            else if (typeB == GeomEllipsoid)
            {
                const float3 localNormal = inverseRotatePoint(frameB, normal);
                closest += rotatePoint(frameB, -normalize(localNormal * b.size.xyz) * b.size.xyz);
            }
            else
            {
                const float3 corner = float3((pointId & 1u) != 0 ? b.size.x : -b.size.x,
                                             (pointId & 2u) != 0 ? b.size.y : -b.size.y,
                                             (pointId & 4u) != 0 ? b.size.z : -b.size.z);
                closest += rotatePoint(frameB, corner);
            }
            const float distance = dot(closest - centerA, normal);
            const float3 point = closest - 0.5 * distance * normal;
            writeContact(id.y, pair, pointId, distance, point, basis, material, settings.z);
        }
    }
}
