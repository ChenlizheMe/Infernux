// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Adapted from MuJoCo Warp _src/forward.py, rungekutta4, and _src/math.py.

#include "model.slang"
#include "integrate_position.slang"
#include "rk4_tableau.slang"

struct Rk4DofState
{
    float initialVelocity;
    float stageVelocity;
    float weightedVelocity;
    float weightedAcceleration;
};

[[vk::binding(0, 0)]] StructuredBuffer<ModelDimensions> model;
[[vk::binding(1, 0)]] StructuredBuffer<JointModel> joints;
[[vk::binding(2, 0)]] StructuredBuffer<float> timestep;
[[vk::binding(3, 0)]] StructuredBuffer<uint4> rkStage;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> qacc;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> qvel;
[[vk::binding(6, 0)]] RWStructuredBuffer<float> qpos;
[[vk::binding(7, 0)]] RWStructuredBuffer<float> initialPosition;
[[vk::binding(8, 0)]] RWStructuredBuffer<Rk4DofState> dofState;
[[vk::binding(9, 0)]] RWStructuredBuffer<float> qaccWarmstart;
[[vk::binding(10, 0)]] RWStructuredBuffer<float> time;

float3 rotationVectorRate(float3 rotation, float3 velocity)
{
    float angleSquared = dot(rotation, rotation);
    if (angleSquared > PhysicsPi * PhysicsPi)
    {
        const float turn = 2.0 * PhysicsPi;
        const float angle = sqrt(angleSquared);
        rotation *= (angle - turn * floor(angle / turn + 0.5)) / angle;
        angleSquared = dot(rotation, rotation);
    }

    float coefficient;
    if (angleSquared < 3.0625)
    {
        // FP32 minimax coefficients from MuJoCo's Bortz exponential-map derivative.
        coefficient = 1.26339661e-09;
        coefficient = -1.64186869e-08 + angleSquared * coefficient;
        coefficient = 8.78585737e-08 + angleSquared * coefficient;
        coefficient = -2.48791736e-07 + angleSquared * coefficient;
        coefficient = 4.23758962e-07 + angleSquared * coefficient;
        coefficient = 4.57548566e-07 + angleSquared * coefficient;
        coefficient = 3.32480922e-05 + angleSquared * coefficient;
        coefficient = 0.00138884981 + angleSquared * coefficient;
        coefficient = 0.0833333358 + angleSquared * coefficient;
    }
    else
    {
        const float half = 0.5 * sqrt(angleSquared);
        coefficient = (1.0 - half / tan(half)) / angleSquared;
    }
    const float3 first = cross(rotation, velocity);
    return velocity + 0.5 * first + coefficient * cross(rotation, first);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void advance_rk4_stage(uint3 id : SV_DispatchThreadID)
{
    const ModelDimensions dimensions = model[0];
    const uint stage = rkStage[0].x;
    const bool last = stage == Rk4FinalStage;
    const float dt = timestep[id.y];
    if (last && id.x == 0)
        time[id.y] += dt;
    if (id.x >= dimensions.state.w)
        return;
    const uint4 joint = joints[id.x].indices;
    const uint type = joint.x;
    const bool rotates = type == JointFree || type == JointBall;
    const uint pos = id.y * dimensions.state.x + joint.y;
    const uint dof = id.y * dimensions.state.y + joint.z;
    const uint angularOffset = type == JointFree ? 3u : 0u;
    const uint dofs = rotates ? angularOffset + 3u : 1u;
    const float scale = rk4StepScale(stage);
    const float weight = rk4Weight(stage);

    if (stage == 0)
    {
        const uint positions = rotates ? dofs + 1u : 1u;
        for (uint i = 0; i < positions; ++i)
            initialPosition[pos + i] = qpos[pos + i];
    }

    float3 angleRate = float3(0.0);
    if (rotates)
    {
        const uint angular = dof + angularOffset;
        angleRate = float3(qvel[angular], qvel[angular + 1], qvel[angular + 2]);
        if (stage > 0)
        {
            const float3 previous = float3(dofState[angular].stageVelocity,
                                           dofState[angular + 1].stageVelocity,
                                           dofState[angular + 2].stageVelocity);
            angleRate = rotationVectorRate((dt * rk4StepScale(stage - 1)) * previous, angleRate);
        }
    }

    float3 linear = float3(0.0);
    float3 angular = float3(0.0);
    for (uint i = 0; i < dofs; ++i)
    {
        const uint index = dof + i;
        const bool isAngular = rotates && i >= angularOffset;
        float rate = qvel[index];
        if (isAngular)
            rate = angleRate[i - angularOffset];
        Rk4DofState state = {qvel[index], 0.0, 0.0, 0.0};
        if (stage > 0)
            state = dofState[index];
        const float acceleration = qacc[index];
        state.stageVelocity = rate;
        state.weightedVelocity += weight * rate;
        state.weightedAcceleration += weight * acceleration;
        if (last)
        {
            rate = state.weightedVelocity;
            qvel[index] = state.initialVelocity + dt * state.weightedAcceleration;
            qacc[index] = state.weightedAcceleration;
            qaccWarmstart[index] = acceleration;
        }
        else
        {
            qvel[index] = state.initialVelocity + (dt * scale) * acceleration;
            dofState[index] = state;
        }
        if (isAngular)
            angular[i - angularOffset] = rate * scale;
        else
            linear[i] = rate * scale;
    }
    integrateJointPosition(type, pos, linear, angular, dt, initialPosition, qpos);
}
