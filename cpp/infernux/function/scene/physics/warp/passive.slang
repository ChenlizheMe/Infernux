// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of joint spring/damper forces in MuJoCo Warp _src/passive.py.

#include "model.slang"

[[vk::binding(0, 0)]] StructuredBuffer<ModelDimensions> model;
[[vk::binding(1, 0)]] StructuredBuffer<JointModel> joints;
[[vk::binding(2, 0)]] StructuredBuffer<float4> jointSpring; // linear, quadratic, cubic
[[vk::binding(3, 0)]] StructuredBuffer<float4> dofDamping; // linear, quadratic, cubic
[[vk::binding(4, 0)]] StructuredBuffer<float> springReference;
[[vk::binding(5, 0)]] StructuredBuffer<float> qpos;
[[vk::binding(6, 0)]] StructuredBuffer<float> qvel;
[[vk::binding(7, 0)]] RWStructuredBuffer<float> qfrcPassive;

float polynomialCoefficient(float3 coefficients, float displacement)
{
    return coefficients.x + coefficients.y * displacement + coefficients.z * displacement * displacement;
}

float3 springRotation(uint position, uint reference)
{
    const float4 current = normalize(float4(qpos[position], qpos[position + 1], qpos[position + 2],
                                            qpos[position + 3]));
    const float4 conjugateReference = float4(springReference[reference], -springReference[reference + 1],
                                             -springReference[reference + 2], -springReference[reference + 3]);
    const float4 relative = mulQuat(conjugateReference, current);
    const float sinHalf = length(relative.yzw);
    if (sinHalf == 0.0)
        return float3(0.0);
    const float pi = 3.14159265358979323846;
    float angle = 2.0 * atan2(sinHalf, relative.x);
    if (angle > pi)
        angle -= 2.0 * pi;
    return relative.yzw * (angle / sinHalf);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void joint_passive_force(uint3 id : SV_DispatchThreadID)
{
    const ModelDimensions size = model[0];
    if (id.x >= size.state.w)
        return;

    const JointModel joint = joints[id.x];
    const uint type = joint.indices.x;
    const float3 stiffness = jointSpring[id.x].xyz;
    uint reference = joint.indices.y;
    uint position = id.y * size.state.x + reference;
    uint firstDof = joint.indices.z;
    const uint dofBase = id.y * size.state.y;
    const uint count = type == JointFree ? 6 : (type == JointBall ? 3 : 1);
    for (uint i = 0; i < count; ++i)
    {
        const float velocity = qvel[dofBase + firstDof + i];
        qfrcPassive[dofBase + firstDof + i] =
            -velocity * polynomialCoefficient(dofDamping[firstDof + i].xyz, abs(velocity));
    }

    if (!any(stiffness != float3(0.0)))
        return;
    if (type == JointFree)
    {
        const float3 translation = float3(qpos[position] - springReference[reference],
                                         qpos[position + 1] - springReference[reference + 1],
                                         qpos[position + 2] - springReference[reference + 2]);
        const float3 force = -translation * polynomialCoefficient(stiffness, length(translation));
        for (uint axis = 0; axis < 3; ++axis)
            qfrcPassive[dofBase + firstDof + axis] += force[axis];
        reference += 3;
        position += 3;
        firstDof += 3;
    }
    if (type == JointFree || type == JointBall)
    {
        const float3 angular = springRotation(position, reference);
        const float3 torque = -angular * polynomialCoefficient(stiffness, length(angular));
        for (uint axis = 0; axis < 3; ++axis)
            qfrcPassive[dofBase + firstDof + axis] += torque[axis];
    }
    else
    {
        const float displacement = qpos[position] - springReference[reference];
        qfrcPassive[dofBase + firstDof] -= displacement * polynomialCoefficient(stiffness, displacement);
    }
}
