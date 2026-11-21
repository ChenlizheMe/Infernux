// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/forward.py semi-implicit advancement.

#include "model.slang"
#include "damping.slang"
#include "integrate_position.slang"

[[vk::binding(0, 0)]] StructuredBuffer<ModelDimensions> model;
[[vk::binding(1, 0)]] StructuredBuffer<JointModel> joints;
[[vk::binding(2, 0)]] StructuredBuffer<float> timestep;
[[vk::binding(3, 0)]] StructuredBuffer<float> qacc;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> qvel;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> qpos;
[[vk::binding(6, 0)]] RWStructuredBuffer<float> qaccWarmstart;
[[vk::binding(7, 0)]] RWStructuredBuffer<float> time;
[[vk::binding(8, 0)]] StructuredBuffer<float> solvedAcceleration;
[[vk::binding(9, 0)]] StructuredBuffer<DofModel> dofs;
[[vk::binding(10, 0)]] StructuredBuffer<float4> dofDamping;
[[vk::binding(11, 0)]] StructuredBuffer<float> massMatrix;
[[vk::binding(12, 0)]] RWStructuredBuffer<float> eulerMassMatrix;

[shader("compute")]
[numthreads(64, 1, 1)]
void assemble_euler_mass(uint3 id : SV_DispatchThreadID)
{
    const ModelDimensions size = model[0];
    if (id.x >= size.state.y)
        return;
    const uint4 row = dofs[id.x].indices;
    const uint base = id.y * size.layout.w + row.z;
    for (uint entry = 0; entry + 1 < row.w; ++entry)
        eulerMassMatrix[base + entry] = massMatrix[base + entry];
    const uint diagonal = base + row.w - 1;
    const float derivative = dampingDerivative(dofDamping[id.x].xyz, qvel[id.y * size.state.y + id.x]);
    eulerMassMatrix[diagonal] = massMatrix[diagonal] + timestep[id.y] * derivative;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void advance_velocity(uint3 id : SV_DispatchThreadID)
{
    const uint count = model[0].state.y;
    if (id.x >= count)
        return;
    const uint index = id.y * count + id.x;
    qvel[index] += qacc[index] * timestep[id.y];
    qaccWarmstart[index] = solvedAcceleration[index];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void advance_position(uint3 id : SV_DispatchThreadID)
{
    const ModelDimensions size = model[0];
    if (id.x >= size.state.w)
        return;
    const JointModel joint = joints[id.x];
    const float dt = timestep[id.y];
    const uint type = joint.indices.x;
    const uint pos = id.y * size.state.x + joint.indices.y;
    const uint velocity = id.y * size.state.y + joint.indices.z;
    float3 linear = float3(0.0);
    float3 angular = float3(0.0);
    if (type == JointFree)
    {
        linear = float3(qvel[velocity], qvel[velocity + 1], qvel[velocity + 2]);
        angular = float3(qvel[velocity + 3], qvel[velocity + 4], qvel[velocity + 5]);
    }
    else if (type == JointBall)
        angular = float3(qvel[velocity], qvel[velocity + 1], qvel[velocity + 2]);
    else
        linear.x = qvel[velocity];
    integrateJointPosition(type, pos, linear, angular, dt, qpos, qpos);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void advance_time(uint3 id : SV_DispatchThreadID)
{
    time[id.y] += timestep[id.y];
}
