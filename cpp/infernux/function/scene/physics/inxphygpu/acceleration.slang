// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/forward.py smooth force assembly.

#include "model.slang"
#include "scene_motion.slang"

[[vk::binding(0, 0)]] StructuredBuffer<ModelDimensions> model;
[[vk::binding(1, 0)]] StructuredBuffer<float> qfrcPassive;
[[vk::binding(2, 0)]] StructuredBuffer<float> qfrcBias;
[[vk::binding(3, 0)]] StructuredBuffer<float> qfrcActuator;
[[vk::binding(4, 0)]] StructuredBuffer<float> qfrcApplied;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> qfrcSmooth;
[[vk::binding(6, 0)]] StructuredBuffer<BodyModel> sceneBodies;
[[vk::binding(7, 0)]] StructuredBuffer<Pose> scenePoses;
[[vk::binding(8, 0)]] StructuredBuffer<float> sceneVelocity;
[[vk::binding(9, 0)]] RWStructuredBuffer<float> sceneAcceleration;
[[vk::binding(10, 0)]] StructuredBuffer<float> sceneTimestep;
[[vk::binding(11, 0)]] StructuredBuffer<SceneRigidMotion> sceneMotion;

[shader("compute")]
[numthreads(64, 1, 1)]
void assemble_smooth_force(uint3 id : SV_DispatchThreadID)
{
    const uint nv = model[0].state.y;
    if (id.x >= nv)
        return;
    const uint index = id.y * nv + id.x;
    qfrcSmooth[index] = qfrcPassive[index] - qfrcBias[index] + qfrcActuator[index] + qfrcApplied[index];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void damp_scene_free_motion(uint3 id : SV_DispatchThreadID)
{
    const ModelDimensions size = model[0];
    if (id.x >= size.state.z)
        return;
    const BodyModel body = sceneBodies[id.x];
    if (body.dofs.y == 0)
        return;

    const SceneRigidMotion motion = sceneMotion[id.x];
    const float dt = sceneTimestep[id.y];
    const uint dof = id.y * size.state.y + body.dofs.x;
    const float4 orientation = scenePoses[id.y * size.state.z + id.x].rotation;
    const float4 inverse = float4(orientation.x, -orientation.yzw);
    const float3 offset = rotateVector(body.inertialPosition.xyz, orientation);
    float before[6];
    for (uint i = 0; i < 6; ++i)
        before[i] = sceneVelocity[dof + i] + dt * sceneAcceleration[dof + i];
    const float3 originVelocity = float3(before[0], before[1], before[2]);
    const float3 spin = float3(before[3], before[4], before[5]);
    const float3 previousSpin =
        rotateVector(float3(sceneVelocity[dof + 3], sceneVelocity[dof + 4], sceneVelocity[dof + 5]), orientation);
    const float3 linear =
        originVelocity + cross(rotateVector(spin, orientation), offset) + dt * centerTransport(previousSpin, offset);
    const float2 damping = max(float2(0.0), 1.0 - dt * motion.damping);
    const float3 linearDelta = clampSceneSpeed(linear * damping.x, motion.velocityLimit.x) - linear;
    const float3 spinDelta = clampSceneSpeed(spin * damping.y, motion.velocityLimit.y) - spin;
    if (all(linearDelta == 0.0) && all(spinDelta == 0.0))
        return;

    const float3 originAccel = (linearDelta - cross(rotateVector(spinDelta, orientation), offset)) / dt;
    const float3 spinAccel = spinDelta / dt;
    const float3 forceDelta = (body.inertiaMass.w / dt) * linearDelta;
    const float4 inertia = body.inertialRotation;
    const float3 principalAccel = rotateVector(spinAccel, float4(inertia.x, -inertia.yzw));
    const float3 torqueDelta = rotateVector(principalAccel * body.inertiaMass.xyz, inertia) +
        rotateVector(cross(offset, forceDelta), inverse);
    for (uint axis = 0; axis < 3; ++axis)
    {
        sceneAcceleration[dof + axis] += originAccel[axis];
        sceneAcceleration[dof + 3 + axis] += spinAccel[axis];
        // Keep M * qacc_smooth and qfrc_smooth consistent for the contact solve.
        qfrcSmooth[dof + axis] += forceDelta[axis];
        qfrcSmooth[dof + 3 + axis] += torqueDelta[axis];
    }
}
