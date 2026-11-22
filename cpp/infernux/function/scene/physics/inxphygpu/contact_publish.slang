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
    float4 extra; // fifth solimp, fifth friction, restitution
    int4 dimensions; // contact dimension, priority, scene friction and restitution combines (enum + 1)
};

struct ContactBasis
{
    float3 normal;
    float3 tangent;
    float3 bitangent;
};

// First record: geom count, batch candidate capacity, contact capacity per world, batch convex worker capacity.
// Second record: flags, minimum restitution speed as float bits.
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

float combineSceneSurface(float a, float b, int mode)
{
    switch (mode)
    {
    case 2: return min(a, b);
    case 3: return a * b;
    case 4: return max(a, b);
    default: return 0.5 * (a + b);
    }
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
    float restitution = 0.0;
    const bool sceneSurface = a.dimensions.z != 0;
    if (sceneSurface)
    {
        friction.x = combineSceneSurface(a.friction.x, b.friction.x, max(a.dimensions.z, b.dimensions.z));
        restitution = combineSceneSurface(a.extra.z, b.extra.z, max(a.dimensions.w, b.dimensions.w));
        if (friction.x == 0.0)
            dim = 1;
    }
    result.dimensions = int4(dim, 0, int(sceneSurface), 0);
    result.parameters = float4(a.parameters.xy + b.parameters.xy, 0.0, adhesion);
    if (sceneSurface && explicitPair == OverlapTrigger)
        result.parameters.y = 0.0;
    const float2 reference = a.reference.x > 0.0 && b.reference.x > 0.0 ?
                                 lerp(b.reference.xy, a.reference.xy, weight) : min(a.reference.xy, b.reference.xy);
    result.reference = float4(reference, 0.0, 0.0);
    result.impedance = lerp(b.impedance, a.impedance, weight);
    result.friction = float4(friction.x, friction.x, friction.y, friction.z);
    result.extra = float4(lerp(b.extra.x, a.extra.x, weight), friction.z, restitution, 0.0);
    return result;
}

void writeContact(uint world, uint4 pair, uint ordinal, float distance, float3 point, ContactBasis basis,
                  ContactMaterial material, uint capacity, uint secondSubShape = NoIndex)
{
    const float margin = material.parameters.x;
    const bool detected = distance < margin + material.parameters.y;
    const bool solid = asint(pair.z) >= -1;
    // Scene gaps extend contact observation while keeping island and solver rows geometric.
    const bool proximity = solid && detected && material.dimensions.z != 0 && distance >= margin;
    const bool constraint = solid && detected && !proximity;
    const bool sensor = asint(pair.w) >= 0 && (asint(pair.z) != OverlapTrigger || detected);
    if (!constraint && !sensor && !proximity)
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
    const float minimumRestitutionSpeed = material.extra.z > 0.0 ? asfloat(contactSettings[1].y) : 0.0;
    contact.extra = float4(material.extra.x, max(material.extra.y, MinFriction), material.extra.z, minimumRestitutionSpeed);
    const uint flags = (constraint ? ContactConstraint : 0u) | (sensor ? ContactSensor : 0u) |
                       (proximity ? ContactProximity : 0u);
    contact.geometry = uint4(pair.xy, dim, flags);
    contact.identity = uint4(ordinal, pair.w, NoIndex, secondSubShape);
    contacts[world * capacity + slot] = contact;
}
