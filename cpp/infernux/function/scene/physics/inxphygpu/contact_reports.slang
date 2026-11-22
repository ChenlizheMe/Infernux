// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#include "model.slang"
#include "contact.slang"
#include "contact_report_layout.slang"
#include "constraint_data.slang"
#include "prescribed_motion.slang"

struct ContactReportDispatch
{
    uint worldCount;
    uint substep;
    uint fixedStepOffset;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint4> settings;
[[vk::binding(1, 0)]] StructuredBuffer<Contact> contacts;
[[vk::binding(2, 0)]] StructuredBuffer<uint> counts;
[[vk::binding(3, 0)]] StructuredBuffer<uint4> topology;
[[vk::binding(4, 0)]] StructuredBuffer<uint4> reservations;
[[vk::binding(5, 0)]] StructuredBuffer<ConstraintSolveState> constraintState;
[[vk::binding(6, 0)]] StructuredBuffer<SpatialMotion> bodyVelocity;
[[vk::binding(7, 0)]] StructuredBuffer<float4> treeCom;
[[vk::binding(8, 0)]] RWStructuredBuffer<uint> stepStatus;
[[vk::binding(9, 0)]] RWStructuredBuffer<ContactReport> reports;
[[vk::binding(10, 0)]] ConstantBuffer<ContactReportDispatch> reportDispatch;
[[vk::binding(11, 0)]] StructuredBuffer<PrescribedMotion> prescribed;
[[vk::binding(12, 0)]] StructuredBuffer<uint> fixedSteps;
[[vk::binding(13, 0)]] StructuredBuffer<ContactReport> sweptReports;

float3 pointSpeed<let HasPrescribed : bool>(uint geometry, uint world, float3 point)
{
    const uint4 limits = settings[0];
    const uint4 layout = settings[1];
    const uint2 address = topology[geometry].xy;
    if (HasPrescribed && address.y != NoIndex)
    {
        uint rows, stride;
        prescribed.GetDimensions(rows, stride);
        const PrescribedMotion motion = prescribed[world * (rows / reportDispatch.worldCount) + address.y];
        if (motion.step == fixedSteps[world] + reportDispatch.fixedStepOffset)
            return motion.linear.xyz + cross(motion.angular.xyz, point - motion.center);
    }
    const uint root = topology[limits.x + address.x].y;
    const uint base = world * layout.x;
    const SpatialMotion velocity = bodyVelocity[base + address.x];
    return velocity.linear.xyz + cross(velocity.angular.xyz, point - treeCom[base + root].xyz);
}

void reportContacts<let HasPrescribed : bool>(uint3 id)
{
    const uint4 limits = settings[0];
    const bool accepted = stepStatus[id.y] == 0u;
    const uint count = accepted ? min(counts[id.y], limits.y) : 0u;
    const uint worlds = reportDispatch.worldCount;
    const uint output = id.y + reportDispatch.substep * worlds;
    const uint stride = limits.y + settings[2 + worlds].x;
    if (id.x == 0)
        stepStatus[worlds + output] = count;
    if (id.x >= count)
        return;

    const uint index = id.y * limits.y + id.x;
    const Contact source = contacts[index];
    const uint2 span = reservations[index].xy;
    const bool trigger = (source.geometry.w & (ContactConstraint | ContactProximity)) == 0u;
    float3 force = float3(0.0);
    if (span.y != 0u)
    {
        const uint base = id.y * limits.z + span.x;
        force = constraintState[base].force * source.normalMargin.xyz;
        // The native scene solver uses an elliptic cone. Components 3..5
        // are angular torques; the public linear contact impulse excludes them.
        if (span.y >= 3u)
            force += constraintState[base + 1u].force * source.tangentAdhesion.xyz +
                     constraintState[base + 2u].force * source.bitangent.xyz;
    }
    const float3 point = source.positionDistance.xyz;
    const float3 relative = pointSpeed<HasPrescribed>(source.geometry.y, id.y, point) -
                            pointSpeed<HasPrescribed>(source.geometry.x, id.y, point);
    const float dt = asfloat(settings[2 + id.y].x);
    reports[output * stride + id.x] = {source.positionDistance, float4(source.normalMargin.xyz, 0.0), float4(relative, 0.0),
                      float4(dt * force, 0.0), uint4(source.geometry.xy, source.identity.zw),
                      uint4(source.identity.x, uint(trigger), uint(span.y != 0u), 0u)};
}

[shader("compute")]
[numthreads(64, 1, 1)]
void capture_contact_reports(uint3 id : SV_DispatchThreadID)
{
    reportContacts<false>(id);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void capture_controlled_contact_reports(uint3 id : SV_DispatchThreadID)
{
    reportContacts<true>(id);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void append_swept_contact_reports(uint3 id : SV_DispatchThreadID)
{
    if (stepStatus[id.y] != 0u)
        return;
    const uint worlds = reportDispatch.worldCount;
    const uint output = id.y + reportDispatch.substep * worlds;
    const uint capacity = settings[0].y;
    const uint count = min(counts[id.y], capacity);
    const uint sweptCapacity = settings[2 + worlds].x;
    const uint sweptBase = id.y * (sweptCapacity + 1u);
    const uint sweptCount = sweptReports[sweptBase].identity.x;
    if (id.x == 0)
        stepStatus[worlds + output] = count + sweptCount;
    if (id.x < sweptCount)
        reports[output * (capacity + sweptCapacity) + count + id.x] = sweptReports[sweptBase + 1u + id.x];
}
