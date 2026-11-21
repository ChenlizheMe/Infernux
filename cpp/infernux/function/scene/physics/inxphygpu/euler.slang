// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/forward.py semi-implicit advancement.

#include "model.slang"
#include "damping.slang"
#include "integrate_position.slang"
#include "linear_dispatch.slang"
#include "scene_motion.slang"

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
[[vk::binding(13, 0)]] StructuredBuffer<uint> awakeCounts;
[[vk::binding(14, 0)]] StructuredBuffer<uint2> awakeWork;
[[vk::binding(15, 0)]] StructuredBuffer<uint> contactStepStatus;
[[vk::binding(16, 0)]] RWStructuredBuffer<uint> fixedSteps;
[[vk::binding(17, 0)]] StructuredBuffer<BodyModel> sceneBodies;
[[vk::binding(18, 0)]] StructuredBuffer<SceneRigidMotion> sceneMotion;

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

void advanceVelocity(uint dof, uint world)
{
    const uint count = model[0].state.y;
    const uint index = world * count + dof;
    qvel[index] += qacc[index] * timestep[world];
    qaccWarmstart[index] = solvedAcceleration[index];
}

void advancePosition(uint jointIndex, uint world)
{
    const ModelDimensions size = model[0];
    const JointModel joint = joints[jointIndex];
    const float dt = timestep[world];
    const uint type = joint.indices.x;
    const uint pos = world * size.state.x + joint.indices.y;
    const uint velocity = world * size.state.y + joint.indices.z;
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

void advanceSceneBody(uint jointIndex, uint world)
{
    const ModelDimensions size = model[0];
    const JointModel joint = joints[jointIndex];
    const uint body = joint.indices.w;
    const uint pos = world * size.state.x + joint.indices.y;
    const uint vel = world * size.state.y + joint.indices.z;
    const float dt = timestep[world];
    const SceneRigidMotion motion = sceneMotion[body];
    const float3 center = sceneBodies[body].inertialPosition.xyz;
    const float4 rotation = float4(qpos[pos + 3], qpos[pos + 4], qpos[pos + 5], qpos[pos + 6]);
    const float3 offset = rotateVector(center, rotation);
    const float3 before = float3(qvel[vel + 3], qvel[vel + 4], qvel[vel + 5]);
    const float3 linearAccel = float3(qacc[vel], qacc[vel + 1], qacc[vel + 2]);
    const float3 spinAccel = float3(qacc[vel + 3], qacc[vel + 4], qacc[vel + 5]);
    const float3 originVelocity = float3(qvel[vel], qvel[vel + 1], qvel[vel + 2]) + dt * linearAccel;
    const float3 integratedSpin = before + dt * spinAccel;
    const float3 integratedCenter = originVelocity + cross(rotateVector(integratedSpin, rotation), offset) +
        dt * centerTransport(rotateVector(before, rotation), offset);
    const float3 centerVelocity = clampSceneSpeed(integratedCenter, motion.velocityLimit.x);
    const float3 spin = clampSceneSpeed(integratedSpin, motion.velocityLimit.y);
    // Integrate the center of mass. The generalized translation lives at the body origin.
    const float4 next = integrateQuat(rotation, spin, dt);
    const float3 nextOffset = rotateVector(center, next);
    const float3 nextVelocity = centerVelocity - cross(rotateVector(spin, next), nextOffset);
    for (uint axis = 0; axis < 3; ++axis)
    {
        qpos[pos + axis] += dt * centerVelocity[axis] + offset[axis] - nextOffset[axis];
        qvel[vel + axis] = nextVelocity[axis];
        qvel[vel + 3 + axis] = spin[axis];
        qaccWarmstart[vel + axis] = linearAccel[axis];
        qaccWarmstart[vel + 3 + axis] = spinAccel[axis];
    }
    for (uint lane = 0; lane < 4; ++lane)
        qpos[pos + 3 + lane] = next[lane];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void advance_scene_position(uint3 id : SV_DispatchThreadID)
{
    if (id.x < model[0].state.w)
        advanceSceneBody(id.x, id.y);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void advance_scene_contact_position(uint3 id : SV_DispatchThreadID)
{
    if (id.x < model[0].state.w && contactStepStatus[id.y] == 0)
        advanceSceneBody(id.x, id.y);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void advance_velocity(uint3 id : SV_DispatchThreadID)
{
    if (id.x < model[0].state.y)
        advanceVelocity(id.x, id.y);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void advance_position(uint3 id : SV_DispatchThreadID)
{
    if (id.x < model[0].state.w)
        advancePosition(id.x, id.y);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void advance_awake_velocity(uint3 id : SV_DispatchThreadID)
{
    const uint index = linearWorkIndex<64>(id);
    if (index < awakeCounts[0])
    {
        const uint2 work = awakeWork[index];
        advanceVelocity(work.x, work.y);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void advance_awake_position(uint3 id : SV_DispatchThreadID)
{
    const uint index = linearWorkIndex<64>(id);
    if (index < awakeCounts[1])
    {
        const uint2 work = awakeWork[index];
        advancePosition(work.x, work.y);
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void advance_time(uint3 id : SV_DispatchThreadID)
{
    time[id.y] += timestep[id.y];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void advance_contact_velocity(uint3 id : SV_DispatchThreadID)
{
    if (id.x < model[0].state.y && contactStepStatus[id.y] == 0)
        advanceVelocity(id.x, id.y);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void advance_contact_position(uint3 id : SV_DispatchThreadID)
{
    if (id.x < model[0].state.w && contactStepStatus[id.y] == 0)
        advancePosition(id.x, id.y);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void advance_contact_time(uint3 id : SV_DispatchThreadID)
{
    if (contactStepStatus[id.y] == 0)
        time[id.y] += timestep[id.y];
}

void advanceControlledClock(uint world)
{
    time[world] += timestep[world];
    fixedSteps[world] += 1u;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void advance_controlled_time(uint3 id : SV_DispatchThreadID)
{
    advanceControlledClock(id.y);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void advance_contact_controlled_time(uint3 id : SV_DispatchThreadID)
{
    if (contactStepStatus[id.y] == 0)
        advanceControlledClock(id.y);
}
