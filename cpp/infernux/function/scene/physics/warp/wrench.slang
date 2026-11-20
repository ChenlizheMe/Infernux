// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/support.py apply_ft and _src/passive.py gravity forces.

#include "model.slang"

struct BodyWrench
{
    float4 angular;
    float4 linear;
};

struct WrenchAtTreeCenter
{
    BodyWrench applied;
    BodyWrench gravityCompensation;
};

[[vk::binding(0, 0)]] StructuredBuffer<ModelDimensions> model;
[[vk::binding(1, 0)]] StructuredBuffer<BodyModel> bodies;
[[vk::binding(2, 0)]] StructuredBuffer<CartesianFrame> inertialFrame;
[[vk::binding(3, 0)]] StructuredBuffer<float4> subtreeCom;
[[vk::binding(4, 0)]] StructuredBuffer<float4> gravity;
[[vk::binding(5, 0)]] StructuredBuffer<BodyWrench> bodyWrench;
[[vk::binding(6, 0)]] RWStructuredBuffer<WrenchAtTreeCenter> centeredWrench;
[[vk::binding(7, 0)]] StructuredBuffer<DofModel> dofs;
[[vk::binding(8, 0)]] StructuredBuffer<SpatialMotion> centeredDof;
[[vk::binding(9, 0)]] RWStructuredBuffer<float> qfrcSmooth;

[shader("compute")]
[numthreads(64, 1, 1)]
void center_body_wrenches(uint3 id : SV_DispatchThreadID)
{
    const uint count = model[0].state.z;
    if (id.x == 0 || id.x >= count)
        return;
    const uint base = id.y * count;
    const uint index = base + id.x;
    const BodyModel body = bodies[id.x];
    const float3 offset = inertialFrame[index].position.xyz - subtreeCom[base + body.subtree.x].xyz;
    BodyWrench applied = bodyWrench[index];
    applied.angular.xyz += cross(offset, applied.linear.xyz);
    const float3 gravityForce = -gravity[id.y].xyz * (body.inertiaMass.w * body.subtreeMass.y);
    const BodyWrench compensation = {float4(cross(offset, gravityForce), 0.0), float4(gravityForce, 0.0)};
    centeredWrench[index] = {applied, compensation};
}

[shader("compute")]
[numthreads(64, 1, 1)]
void add_generalized_wrenches(uint3 id : SV_DispatchThreadID)
{
    const ModelDimensions size = model[0];
    if (id.x >= size.state.y)
        return;
    const uint dofIndex = id.y * size.state.y + id.x;
    const DofModel dof = dofs[id.x];
    const SpatialMotion coordinate = centeredDof[dofIndex];
    const uint bodyBase = id.y * size.state.z;
    const uint first = dof.indices.x;
    const uint end = first + bodies[first].subtree.y;
    float3 force = float3(0.0);
    float3 torque = float3(0.0);
    for (uint body = first; body < end; ++body)
    {
        const WrenchAtTreeCenter wrench = centeredWrench[bodyBase + body];
        force += wrench.applied.linear.xyz + wrench.gravityCompensation.linear.xyz;
        torque += wrench.applied.angular.xyz + wrench.gravityCompensation.angular.xyz;
    }
    qfrcSmooth[dofIndex] += dot(coordinate.linear.xyz, force) + dot(coordinate.angular.xyz, torque);
}
