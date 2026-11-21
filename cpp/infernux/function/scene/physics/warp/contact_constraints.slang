// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/constraint.py rigid contact rows.

#include "model.slang"
#include "contact.slang"
#include "constraint_data.slang"
#include "constraint_parameters.slang"
#include "constraint_stream.slang"
#include "constraint_transpose.slang"
#include "solver_worlds.slang"
#include "prescribed_motion.slang"

static const uint NegativeJacobian = 0x80000000u;
static const uint ContactTileSize = 256;
static const uint ContactRowGroupSize = 64;

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
[[vk::binding(17, 0)]] StructuredBuffer<ModelDimensions> motionModel;
[[vk::binding(18, 0)]] StructuredBuffer<PrescribedMotion> prescribedMotion;
[[vk::binding(19, 0)]] StructuredBuffer<uint> fixedSteps;
[[vk::binding(20, 0)]] StructuredBuffer<uint> contactOrder;
[[vk::binding(21, 0)]] RWStructuredBuffer<uint> sortedContacts;
[[vk::binding(22, 0)]] StructuredBuffer<uint> contactMergeWidth;

struct ContactOrderKey
{
    uint4 pairAndSubshapes;
    uint feature;
    uint source;
};

groupshared ContactOrderKey contactKeys[ContactTileSize];
groupshared uint activeContacts;
groupshared uint2 contactRowScan[2][ContactRowGroupSize];

// settings[0] holds geometry count and per-world contact, row and column capacities.
// settings[1] holds body count, DOF count, worker count and cone/reference flags.
// The remaining records hold each world's timestep and inverse square root of impratio as float bits.
// topology stores one geom-to-body record per geom, then body records with last DOF of the welded body,
// root body, translational inverse weight as float bits and subtree body count.
uint4 contactBody(uint geometry)
{
    return topology[settings[0].x + topology[geometry].x];
}

uint nextContactDof(uint first, uint second)
{
    return first == NoIndex ? second : (second == NoIndex ? first : max(first, second));
}

ContactOrderKey readContactKey(uint base, uint source)
{
    const Contact contact = contacts[base + source];
    return {uint4(contact.geometry.xy, contact.identity.zw), contact.identity.x, source};
}

bool precedesContact(ContactOrderKey a, ContactOrderKey b)
{
    for (uint i = 0; i < 4; ++i)
        if (a.pairAndSubshapes[i] != b.pairAndSubshapes[i])
            return a.pairAndSubshapes[i] < b.pairAndSubshapes[i];
    return a.feature < b.feature || (a.feature == b.feature && a.source < b.source);
}

uint groupContactCount(uint world, uint lane, uint capacity)
{
    __target_switch
    {
    case wgsl:
        if (lane == 0)
            activeContacts = min(contactCount[world].load(), capacity);
        return workgroupUniformLoad(activeContacts);
    default:
        return min(contactCount[world].load(), capacity);
    }
}

[shader("compute")]
[numthreads(ContactRowGroupSize, 1, 1)]
void sort_contact_tiles(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    const uint capacity = settings[0].y;
    const uint count = groupContactCount(group.y, lane, capacity);
    const uint first = group.x * ContactTileSize;
    if (first >= count)
        return;
    const uint size = min(count - first, ContactTileSize);
    const uint base = group.y * capacity;
    uint length = 1u;
    while (length < size)
        length *= 2;
    const ContactOrderKey empty = {uint4(NoIndex), NoIndex, NoIndex};
    for (uint i = lane; i < length; i += ContactRowGroupSize)
        contactKeys[i] = i < size ? readContactKey(base, first + i) : empty;
    GroupMemoryBarrierWithGroupSync();
    for (uint span = 2; span <= length; span *= 2)
    {
        for (uint stride = span / 2; stride > 0; stride /= 2)
        {
            for (uint i = lane; i < length; i += ContactRowGroupSize)
            {
                if ((i & stride) == 0)
                {
                    const uint other = i | stride;
                    const ContactOrderKey a = contactKeys[i], b = contactKeys[other];
                    if (precedesContact(a, b) != ((i & span) == 0))
                    {
                        contactKeys[i] = b;
                        contactKeys[other] = a;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
    for (uint i = lane; i < size; i += ContactRowGroupSize)
        sortedContacts[base + first + i] = contactKeys[i].source;
}

[shader("compute")]
[numthreads(ContactRowGroupSize, 1, 1)]
void merge_contact_tiles(uint3 id : SV_DispatchThreadID)
{
    const uint capacity = settings[0].y;
    const uint count = min(contactCount[id.y].load(), capacity);
    if (id.x >= count)
        return;
    const uint base = id.y * capacity;
    const uint width = contactMergeWidth[0];
    const uint start = id.x / (2 * width) * (2 * width);
    const uint mid = min(start + width, count);
    const uint end = min(mid + width, count);
    const bool left = id.x < mid;
    const uint other = left ? mid : start;
    uint lo = other;
    uint hi = left ? end : mid;
    const uint source = contactOrder[base + id.x];
    const ContactOrderKey key = readContactKey(base, source);
    while (lo < hi)
    {
        const uint split = (lo + hi) / 2;
        if (precedesContact(readContactKey(base, contactOrder[base + split]), key))
            lo = split + 1;
        else
            hi = split;
    }
    const uint rank = id.x - (left ? start : mid) + lo - other;
    sortedContacts[base + start + rank] = source;
}

uint contactColumnCount(uint a, uint b)
{
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
    return width;
}

void writeContactColumns(uint first, uint second, uint base, uint width)
{
    for (uint i = 0; i < width; ++i)
    {
        const uint next = nextContactDof(first, second);
        const bool negative = next == first;
        columns[base + i] = {next | (negative ? NegativeJacobian : 0u), 0.0};
        if (negative)
            first = dofs[first].indices.y;
        else
            second = dofs[second].indices.y;
    }
}

[shader("compute")]
[numthreads(ContactRowGroupSize, 1, 1)]
void reserve_contact_rows(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    const uint4 limits = settings[0];
    const bool elliptic = (settings[1].w & EllipticCone) != 0;
    const uint world = group.y;
    const uint count = groupContactCount(world, lane, limits.y);
    const uint base = world * limits.y;
    uint2 totals = uint2(0u);
    for (uint first = 0; first < count; first += ContactRowGroupSize)
    {
        const bool live = first + lane < count;
        uint index = 0u;
        uint firstDof = NoIndex, secondDof = NoIndex;
        uint width = 0, height = 0;
        if (live)
        {
            index = base + contactOrder[base + first + lane];
            const Contact contact = contacts[index];
            const bool separated = contact.positionDistance.w - contact.normalMargin.w >= 0.0;
            if ((contact.geometry.w & ContactConstraint) != 0 && (!separated || contact.tangentAdhesion.w != 0.0))
            {
                height = contactConstraintRows(contact, elliptic);
                firstDof = contactBody(contact.geometry.x).x;
                secondDof = contactBody(contact.geometry.y).x;
                width = contactColumnCount(firstDof, secondDof);
            }
        }
        const uint2 size = uint2(height, height * width);
        contactRowScan[0][lane] = size;
        GroupMemoryBarrierWithGroupSync();
        uint read = 0;
        for (uint stride = 1; stride < ContactRowGroupSize; stride *= 2)
        {
            const uint2 earlier = lane < stride ? uint2(0u) : contactRowScan[read][lane - stride];
            contactRowScan[1 - read][lane] = contactRowScan[read][lane] + earlier;
            GroupMemoryBarrierWithGroupSync();
            read = 1 - read;
        }
        const uint2 finish = totals + contactRowScan[read][lane];
        totals += contactRowScan[read][ContactRowGroupSize - 1];
        GroupMemoryBarrierWithGroupSync();
        if (live)
        {
            const bool accepted = height != 0 && all(finish <= limits.zw);
            const uint2 block = finish - size;
            reservations[index] = accepted ? uint4(block.x, height, block.y, width) : uint4(0u);
            if (accepted)
                writeContactColumns(firstDof, secondDof, world * limits.w + block.y, width);
        }
    }
    if (lane == 0)
    {
        rowCounts[2 * world].store(totals.x);
        rowCounts[2 * world + 1].store(totals.y);
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

float contactRowVelocity(Contact contact, uint row, uint count, bool elliptic, float3 linear, float3 angular)
{
    if (elliptic)
        return jacobianValue(contact, row, linear, angular);
    float value = dot(contact.normalMargin.xyz, linear);
    if (count > 1)
    {
        const uint axis = row / 2 + 1;
        const float direction = (row & 1u) == 0 ? 1.0 : -1.0;
        value += direction * contactFriction(contact, axis - 1) * jacobianValue(contact, axis, linear, angular);
    }
    return value;
}

SpatialMotion contactPrescribedMotion(uint geometry, uint world, uint step, float3 point)
{
    const uint row = topology[geometry].y;
    if (row != NoIndex)
    {
        const PrescribedMotion motion = prescribedMotion[world * motionModel[0].layout.y + row];
        if (motion.step == step)
            return spatial(motion.angular.xyz, motion.linear.xyz + cross(motion.angular.xyz, point - motion.center));
    }
    return spatial(float3(0.0), float3(0.0));
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
            for (uint row = 0; row < span.y; ++row)
                columns[base + row * span.w + j] = {dof, contactRowVelocity(contact, row, span.y, elliptic, linear, angular)};
        }
    }
}

void finishContactRows<let Prescribed : bool>(uint3 id)
{
    const uint4 limits = settings[0];
    const uint4 config = settings[1];
    const float2 world = asfloat(settings[2 + id.y].xy);
    const uint count = min(contactCount[id.y].load(), limits.y);
    const bool elliptic = (config.w & EllipticCone) != 0;
    const bool unclamped = (config.w & (UnclampedReference | DiscreteConstraints)) != 0;
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
        const float4 response = constraintResponse(ContactRow, kbi, weight, world.x, config.w, contact.reference.x);
        const float2 frictionReference = any(contact.reference.zw != float2(0.0)) ?
                                            contact.reference.zw : contact.reference.xy;
        const float tangentialDamping = impedanceParameters(frictionReference, contact.impedance, contact.extra.x,
                                                            position, world.x, unclamped).y;
        const float tangentialFactor = (config.w & DiscreteConstraints) != 0 ? 1.0 + world.x * tangentialDamping : 1.0;
        const uint base = id.y * limits.w + span.z;
        const uint dofBase = id.y * config.y;
        float3 prescribedLinear = float3(0.0);
        float3 prescribedAngular = float3(0.0);
        if (Prescribed)
        {
            const uint step = fixedSteps[id.y] + 1u;
            const float3 point = contact.positionDistance.xyz;
            const SpatialMotion a = contactPrescribedMotion(contact.geometry.x, id.y, step, point);
            const SpatialMotion b = contactPrescribedMotion(contact.geometry.y, id.y, step, point);
            prescribedLinear = b.linear.xyz - a.linear.xyz;
            prescribedAngular = b.angular.xyz - a.angular.xyz;
        }
        for (uint row = 0; row < span.y; ++row)
        {
            float velocity = Prescribed ? contactRowVelocity(contact, row, span.y, elliptic, prescribedLinear,
                                                              prescribedAngular) : 0.0;
            const uint columnStart = base + row * span.w;
            for (uint j = 0; j < span.w; ++j)
            {
                const ConstraintColumn column = columns[columnStart + j];
                velocity += column.value * qvel[dofBase + column.dof];
            }
            float d = response.x;
            float acceleration = constraintAcceleration(response, position, velocity, 0.0);
            if (elliptic && row > 0)
            {
                const float mu = contactFriction(contact, row - 1);
                const float scale = row == 1 ? ratio : ratio * friction * friction / (mu * mu);
                d /= scale;
                acceleration = -tangentialDamping * velocity / tangentialFactor;
            }
            else if (elliptic && contact.extra.z > 0.0 && velocity < -contact.extra.w)
            {
                // Restitution targets a separating velocity. The incoming speed
                // is sampled before applying this step's accelerations.
                d = constraintDiagonal(weight, MaxImpedance);
                acceleration = -(1.0 + contact.extra.z) * velocity / world.x;
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

[shader("compute")]
[numthreads(64, 1, 1)]
void finish_contact_rows(uint3 id : SV_DispatchThreadID)
{
    finishContactRows<false>(id);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void finish_controlled_contact_rows(uint3 id : SV_DispatchThreadID)
{
    finishContactRows<true>(id);
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
    id = solverDispatchId(id);
    const uint4 config = settings[1];
    if (id.x >= config.y)
        return;
    const uint nv = config.y;
    transposeAddress[transposeCursorBase(id.y, nv) + id.x].store(0u);
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
    id = solverDispatchId(id);
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
    id = solverDispatchId(id);
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
