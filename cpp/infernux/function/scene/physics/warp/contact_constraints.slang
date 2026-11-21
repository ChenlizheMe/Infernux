// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/constraint.py rigid contact rows.

#include "model.slang"
#include "contact.slang"
#include "constraint_data.slang"
#include "constraint_parameters.slang"
#include "constraint_stream.slang"

static const uint NegativeJacobian = 0x80000000u;

[[vk::binding(1, 0)]] StructuredBuffer<Contact> contacts;
[[vk::binding(2, 0)]] RWStructuredBuffer<Atomic<uint>> contactCount;
[[vk::binding(3, 0)]] StructuredBuffer<uint4> topology;
[[vk::binding(4, 0)]] StructuredBuffer<DofModel> dofs;
[[vk::binding(6, 0)]] RWStructuredBuffer<uint4> reservations; // first row, rows, first column, columns per row
[[vk::binding(9, 0)]] StructuredBuffer<SpatialMotion> centeredDof;
[[vk::binding(10, 0)]] StructuredBuffer<float4> subtreeCom;
[[vk::binding(11, 0)]] StructuredBuffer<float> qvel;
[[vk::binding(12, 0)]] StructuredBuffer<float> smoothAcceleration;
[[vk::binding(13, 0)]] StructuredBuffer<float> warmstartAcceleration;
[[vk::binding(14, 0)]] RWStructuredBuffer<float> acceleration;
[[vk::binding(15, 0)]] RWStructuredBuffer<float> residuals;
[[vk::binding(16, 0)]] RWStructuredBuffer<ConstraintEvaluation> evaluation;

// settings[0] holds geometry count and per-world contact, row and column capacities.
// settings[1] holds body count, DOF count, worker count and cone/reference flags.
// The remaining records hold each world's timestep and inverse square root of impratio as float bits.
// topology stores one geom-to-body record per geom, then body records with last DOF of the welded body,
// root body and translational inverse weight as float bits.
uint4 contactBody(uint geometry)
{
    return topology[settings[0].x + topology[geometry].x];
}

uint nextContactDof(uint first, uint second)
{
    return first == NoIndex ? second : (second == NoIndex ? first : max(first, second));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void clear_constraint_counts(uint3 id : SV_DispatchThreadID)
{
    rowCounts[2 * id.y].store(0u);
    rowCounts[2 * id.y + 1].store(0u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void reserve_contact_rows(uint3 id : SV_DispatchThreadID)
{
    const uint4 limits = settings[0];
    const uint4 config = settings[1];
    const uint count = min(contactCount[id.y].load(), limits.y);
    const uint base = id.y * limits.y;
    for (uint i = id.x; id.x < config.z && i < count; i += config.z)
    {
        const uint index = base + i;
        const Contact contact = contacts[index];
        const float penetration = contact.positionDistance.w - contact.normalMargin.w;
        reservations[index] = uint4(0u);
        if ((contact.geometry.w & ContactConstraint) == 0 || (penetration >= 0.0 && contact.tangentAdhesion.w == 0.0))
            continue;

        const uint firstDof = contactBody(contact.geometry.x).x;
        const uint secondDof = contactBody(contact.geometry.y).x;
        uint a = firstDof;
        uint b = secondDof;
        uint width = 0;
        while (a != b)
        {
            const uint next = nextContactDof(a, b);
            if (next == a)
                a = dofs[a].indices.y;
            else
                b = dofs[b].indices.y;
            ++width;
        }

        const uint dimension = contact.geometry.z;
        const uint height = dimension == 1 || (config.w & EllipticCone) != 0 ? dimension : 2 * (dimension - 1);
        const uint2 block = reserveConstraintBlock(id.y, height, width);
        if (block.x == NoIndex)
            continue;
        reservations[index] = uint4(block.x, height, block.y, width);

        a = firstDof;
        b = secondDof;
        const uint columnBase = id.y * limits.w + block.y;
        for (uint j = 0; j < width; ++j)
        {
            const uint next = nextContactDof(a, b);
            const bool negative = next == a;
            columns[columnBase + j] = {next | (negative ? NegativeJacobian : 0u), 0.0};
            if (negative)
                a = dofs[a].indices.y;
            else
                b = dofs[b].indices.y;
        }
    }
}

float3 contactDirection(Contact contact, uint axis)
{
    switch (axis)
    {
    case 0: return contact.normalMargin.xyz;
    case 1: return contact.tangentAdhesion.xyz;
    default: return contact.bitangent.xyz;
    }
}

float jacobianValue(Contact contact, uint axis, float3 pointVelocity, float3 angularVelocity)
{
    return axis < 3 ? dot(contactDirection(contact, axis), pointVelocity) :
                      dot(contactDirection(contact, axis - 3), angularVelocity);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void project_contact_rows(uint3 id : SV_DispatchThreadID)
{
    const uint4 limits = settings[0];
    const uint4 config = settings[1];
    const uint count = min(contactCount[id.y].load(), limits.y);
    const uint bodyBase = id.y * config.x;
    const uint dofBase = id.y * config.y;
    const bool elliptic = (config.w & EllipticCone) != 0;
    for (uint i = id.x; id.x < config.z && i < count; i += config.z)
    {
        const uint index = id.y * limits.y + i;
        const uint4 span = reservations[index];
        if (span.y == 0)
            continue;
        const Contact contact = contacts[index];
        const float3 offsetA = contact.positionDistance.xyz -
                               subtreeCom[bodyBase + contactBody(contact.geometry.x).y].xyz;
        const float3 offsetB = contact.positionDistance.xyz -
                               subtreeCom[bodyBase + contactBody(contact.geometry.y).y].xyz;
        const uint base = id.y * limits.w + span.z;
        for (uint j = 0; j < span.w; ++j)
        {
            const uint encoded = columns[base + j].dof;
            const uint dof = encoded & ~NegativeJacobian;
            const bool negative = (encoded & NegativeJacobian) != 0;
            const SpatialMotion coordinate = centeredDof[dofBase + dof];
            const float sign = negative ? -1.0 : 1.0;
            const float3 angular = sign * coordinate.angular.xyz;
            const float3 linear = sign * coordinate.linear.xyz + cross(angular, negative ? offsetA : offsetB);
            const float normal = dot(contact.normalMargin.xyz, linear);
            for (uint row = 0; row < span.y; ++row)
            {
                float value = normal;
                if (elliptic)
                    value = jacobianValue(contact, row, linear, angular);
                else if (span.y > 1)
                {
                    const uint axis = row / 2 + 1;
                    const float direction = (row & 1u) == 0 ? 1.0 : -1.0;
                    value += direction * contactFriction(contact, axis - 1) * jacobianValue(contact, axis, linear, angular);
                }
                columns[base + row * span.w + j] = {dof, value};
            }
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void finish_contact_rows(uint3 id : SV_DispatchThreadID)
{
    const uint4 limits = settings[0];
    const uint4 config = settings[1];
    const float2 world = asfloat(settings[2 + id.y].xy);
    const uint count = min(contactCount[id.y].load(), limits.y);
    const bool elliptic = (config.w & EllipticCone) != 0;
    const bool unclamped = (config.w & UnclampedReference) != 0;
    for (uint i = id.x; id.x < config.z && i < count; i += config.z)
    {
        const uint index = id.y * limits.y + i;
        const uint4 span = reservations[index];
        if (span.y == 0)
            continue;
        const Contact contact = contacts[index];
        float weight = asfloat(contactBody(contact.geometry.x).z) + asfloat(contactBody(contact.geometry.y).z);
        const float ratio = world.y * world.y;
        const float friction = contact.friction.x;
        if (!elliptic && span.y > 1)
            weight = (weight + friction * friction * weight) * 2.0 * friction * friction * ratio;

        const float position = contact.positionDistance.w - contact.normalMargin.w;
        const float3 kbi = impedanceParameters(contact.reference.xy, contact.impedance, contact.extra.x, position,
                                               world.x, unclamped);
        const float diagonal = constraintDiagonal(weight, kbi.z);
        const float2 frictionReference = any(contact.reference.zw != float2(0.0)) ?
                                            contact.reference.zw : contact.reference.xy;
        const float tangentialDamping = impedanceParameters(frictionReference, contact.impedance, contact.extra.x,
                                                            position, world.x, unclamped).y;
        const uint base = id.y * limits.w + span.z;
        const uint dofBase = id.y * config.y;
        for (uint row = 0; row < span.y; ++row)
        {
            float velocity = 0.0;
            const uint columnStart = base + row * span.w;
            for (uint j = 0; j < span.w; ++j)
            {
                const ConstraintColumn column = columns[columnStart + j];
                velocity += column.value * qvel[dofBase + column.dof];
            }
            float d = diagonal;
            float acceleration = -kbi.x * kbi.z * position - kbi.y * velocity;
            if (elliptic && row > 0)
            {
                const float mu = contactFriction(contact, row - 1);
                const float scale = row == 1 ? ratio : ratio * friction * friction / (mu * mu);
                d /= scale;
                acceleration = -tangentialDamping * velocity;
            }
            else if (contact.tangentAdhesion.w != 0.0)
                acceleration += contact.tangentAdhesion.w / (d * (elliptic ? 1 : span.y));

            ConstraintRow result;
            result.layout = uint4(span.z + row * span.w, span.w, i, row);
            result.state = float4(elliptic && row > 0 ? contact.normalMargin.w : contact.positionDistance.w,
                                  contact.normalMargin.w, velocity, acceleration);
            result.diagonal = d;
            result.frictionLoss = 0.0;
            result.kind = ContactRow;
            rows[id.y * limits.z + span.x + row] = result;
        }
    }
}

uint activeConstraintRows(uint world, uint4 limits)
{
    const uint rowCount = rowCounts[2 * world].load();
    const uint columnCount = rowCounts[2 * world + 1].load();
    // Keep the overflow in the counters and skip any incomplete matrix.
    return rowCount <= limits.z && columnCount <= limits.w ? rowCount : NoIndex;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void seed_contact_acceleration(uint3 id : SV_DispatchThreadID)
{
    const uint4 config = settings[1];
    if (id.x >= config.y)
        return;
    const uint count = activeConstraintRows(id.y, settings[0]);
    if (count == NoIndex)
        return;
    const uint index = id.y * config.y + id.x;
    const bool warmstart = (config.w & DisableWarmstart) == 0 && count != 0u;
    acceleration[index] = warmstart ? warmstartAcceleration[index] : smoothAcceleration[index];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void contact_residual(uint3 id : SV_DispatchThreadID)
{
    const uint4 limits = settings[0];
    const uint count = activeConstraintRows(id.y, limits);
    if (count == NoIndex || id.x >= count)
        return;
    const uint index = id.y * limits.z + id.x;
    const ConstraintRow row = rows[index];
    const uint columnBase = id.y * limits.w + row.layout.x;
    const uint dofBase = id.y * settings[1].y;
    float residual = -row.state.w;
    for (uint j = 0; j < row.layout.y; ++j)
    {
        const ConstraintColumn entry = columns[columnBase + j];
        residual += entry.value * acceleration[dofBase + entry.dof];
    }
    residuals[index] = residual;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void evaluate_contact_objective(uint3 id : SV_DispatchThreadID)
{
    const uint4 limits = settings[0];
    const uint count = activeConstraintRows(id.y, limits);
    if (count == NoIndex || id.x >= count)
        return;
    const uint index = id.y * limits.z + id.x;
    const ConstraintRow row = rows[index];
    const float residual = residuals[index];
    const float diagonal = row.diagonal;
    const uint kind = row.kind;
    if (kind == EqualityRow)
    {
        evaluation[index] = evaluateQuadratic(residual, diagonal);
        return;
    }
    if (isFrictionConstraint(kind))
    {
        evaluation[index] = evaluateFriction(residual, diagonal, row.frictionLoss);
        return;
    }
    const ConstraintEvaluation satisfied = {0.0, 0.0, SatisfiedConstraint};
    const bool elliptic = (settings[1].w & EllipticCone) != 0;
    if (kind != ContactRow || !elliptic)
    {
        evaluation[index] = residual >= 0.0 ? satisfied : evaluateQuadratic(residual, diagonal);
        return;
    }
    const Contact contact = contacts[id.y * limits.y + row.layout.z];
    if (contact.geometry.z == 1)
    {
        evaluation[index] = residual >= 0.0 ? satisfied : evaluateQuadratic(residual, diagonal);
        return;
    }

    const uint ordinal = row.layout.w;
    const uint first = index - ordinal;
    const float mu = contact.friction.x * asfloat(settings[2 + id.y].y);
    const float normal = residuals[first] * mu;
    float tangentSquared = 0.0;
    for (uint j = 1; j < contact.geometry.z; ++j)
    {
        const float value = residuals[first + j] * contactFriction(contact, j - 1);
        tangentSquared += value * value;
    }
    const float tangent = sqrt(tangentSquared);
    if (normal >= mu * tangent)
        evaluation[index] = satisfied;
    else if (mu * normal + tangent <= 0.0 || tangent == 0.0)
        evaluation[index] = evaluateQuadratic(residual, diagonal);
    else
    {
        const float dm = rows[first].diagonal / max(ConstraintMinValue, mu * mu * (1.0 + mu * mu));
        const float gap = normal - mu * tangent;
        const float force = -dm * gap * mu;
        if (ordinal == 0)
            evaluation[index] = {force, 0.5 * dm * gap * gap, EllipticMiddleState};
        else
        {
            const float friction = contactFriction(contact, ordinal - 1);
            evaluation[index] = {-force * residual * friction * friction / max(ConstraintMinValue, tangent),
                                 0.0, EllipticMiddleState};
        }
    }
}
