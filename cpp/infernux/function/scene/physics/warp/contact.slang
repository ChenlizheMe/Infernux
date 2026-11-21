// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang contact layout for MuJoCo Warp collision and constraint stages.

static const uint ContactConstraint = 1;
static const uint ContactSensor = 2;

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
    uint4 identity; // per-pair contact ordinal, sensor id, first subshape, second subshape
};

float contactFriction(Contact contact, uint axis)
{
    return axis < 4 ? contact.friction[axis] : contact.extra.y;
}

uint contactConstraintRows(Contact contact, bool elliptic)
{
    const uint dimension = contact.geometry.z;
    return elliptic || dimension == 1 ? dimension : 2 * (dimension - 1);
}
