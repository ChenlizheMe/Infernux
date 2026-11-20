// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/collision_{core,primitive,primitive_core}.py.

#include "model.slang"

static const uint GeomPlane = 0;
static const uint GeomSphere = 2;
static const uint GeomCapsule = 3;
static const uint GeomEllipsoid = 4;
static const uint GeomBox = 6;
static const uint ContactConstraint = 1;
static const uint ContactSensor = 2;
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

struct Contact
{
    float4 positionDistance;
    float4 normalMargin;
    float4 tangentAdhesion;
    float4 bitangent;
    float4 reference;
    float4 impedance;
    float4 friction;
    float4 extra;
    uint4 geometry; // first geometry, second geometry, contact dimension, constraint/sensor flags
    uint4 identity; // per-pair contact ordinal, sensor id
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
        const bool planePair = typeA == GeomPlane &&
                               (typeB == GeomSphere || typeB == GeomCapsule || typeB == GeomEllipsoid || typeB == GeomBox);
        const bool spherePair = typeA == GeomSphere && (typeB == GeomSphere || typeB == GeomCapsule);
        if (!planePair && !spherePair)
            continue;

        const CartesianFrame frameA = geometryFrames[geometryBase + pair.x];
        const CartesianFrame frameB = geometryFrames[geometryBase + pair.y];
        const float3 centerA = frameA.position.xyz;
        const float3 centerB = frameB.position.xyz;
        const ContactMaterial material = resolveMaterial(pair, settings.x);
        if (spherePair)
        {
            float3 closest = centerB;
            if (typeB == GeomCapsule)
            {
                const float3 axis = frameAxis(frameB, 2);
                const float along = clamp(dot(centerA - centerB, axis), -b.size.y, b.size.y);
                closest += axis * along;
            }
            const float3 difference = closest - centerA;
            const float lengthAB = length(difference);
            const float3 normal = lengthAB == 0.0 ? float3(1, 0, 0) : difference / lengthAB;
            const float distance = lengthAB - (a.size.x + b.size.x);
            const float3 point = centerA + normal * (a.size.x + 0.5 * distance);
            writeContact(id.y, pair, 0, distance, point, makeBasis(normal), material, settings.z);
            continue;
        }

        const float3 normal = frameAxis(frameA, 2);
        ContactBasis basis = makeBasis(normal);
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
                const float3 localNormal = float3(dot(normal, frameAxis(frameB, 0)), dot(normal, frameAxis(frameB, 1)),
                                                  dot(normal, frameAxis(frameB, 2)));
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
