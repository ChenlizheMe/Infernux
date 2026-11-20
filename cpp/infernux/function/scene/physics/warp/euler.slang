// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/forward.py semi-implicit advancement.

#include "model.slang"

[[vk::binding(0, 0)]] StructuredBuffer<ModelDimensions> model;
[[vk::binding(1, 0)]] StructuredBuffer<JointModel> joints;
[[vk::binding(2, 0)]] StructuredBuffer<float> timestep;
[[vk::binding(3, 0)]] StructuredBuffer<float> qacc;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> qvel;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> qpos;
[[vk::binding(6, 0)]] RWStructuredBuffer<float> qaccWarmstart;
[[vk::binding(7, 0)]] RWStructuredBuffer<float> time;

[shader("compute")]
[numthreads(64, 1, 1)]
void advance_velocity(uint3 id : SV_DispatchThreadID)
{
    const uint count = model[0].state.y;
    if (id.x >= count)
        return;
    const uint index = id.y * count + id.x;
    qvel[index] += qacc[index] * timestep[id.y];
    qaccWarmstart[index] = qacc[index];
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
    uint pos = id.y * size.state.x + joint.indices.y;
    uint velocity = id.y * size.state.y + joint.indices.z;
    if (type == JointFree)
    {
        qpos[pos] += dt * qvel[velocity];
        qpos[pos + 1] += dt * qvel[velocity + 1];
        qpos[pos + 2] += dt * qvel[velocity + 2];
        pos += 3;
        velocity += 3;
    }
    if (type == JointFree || type == JointBall)
    {
        const float4 orientation = float4(qpos[pos], qpos[pos + 1], qpos[pos + 2], qpos[pos + 3]);
        const float3 angular = float3(qvel[velocity], qvel[velocity + 1], qvel[velocity + 2]);
        const float4 next = integrateQuat(orientation, angular, dt);
        qpos[pos] = next.x;
        qpos[pos + 1] = next.y;
        qpos[pos + 2] = next.z;
        qpos[pos + 3] = next.w;
    }
    else
        qpos[pos] += dt * qvel[velocity];
}

[shader("compute")]
[numthreads(1, 1, 1)]
void advance_time(uint3 id : SV_DispatchThreadID)
{
    time[id.y] += timestep[id.y];
}
