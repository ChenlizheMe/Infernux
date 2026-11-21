// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/collision_{core,primitive,primitive_core}.py.

#include "contact.slang"
#include "collision_candidate.slang"

static const float MinFriction = 1.0e-5;
static const uint UseConvexBoxes = 1;
static const uint UseConvexMulticontact = 2;

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

// First record: geom count, batch candidate capacity, contact capacity per world, batch convex worker capacity.
// Second record: flags.
[[vk::binding(0, 0)]] StructuredBuffer<uint4> contactSettings;
[[vk::binding(1, 0)]] StructuredBuffer<CartesianFrame> geometryFrames;
[[vk::binding(2, 0)]] StructuredBuffer<CollisionShape> shapes;
[[vk::binding(3, 0)]] StructuredBuffer<CollisionCandidate> collisionPair;
[[vk::binding(4, 0)]] StructuredBuffer<uint> collisionCount;
[[vk::binding(5, 0)]] StructuredBuffer<ContactMaterial> materials;
[[vk::binding(6, 0)]] RWStructuredBuffer<Contact> contacts;
[[vk::binding(7, 0)]] RWStructuredBuffer<Atomic<uint>> contactCount;

uint collisionWorkCount(uint4 settings)
{
    return min(collisionCount[0], settings.y);
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
                  ContactMaterial material, uint capacity, uint secondSubShape = 0)
{
    const float margin = material.parameters.x;
    const bool detected = distance < margin + material.parameters.y;
    const bool constraint = asint(pair.z) >= -1 && detected;
    const bool sensor = asint(pair.w) >= 0 && (asint(pair.z) != OverlapTrigger || detected);
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
    contact.identity = uint4(ordinal, pair.w, 0, secondSubShape);
    contacts[world * capacity + slot] = contact;
}
