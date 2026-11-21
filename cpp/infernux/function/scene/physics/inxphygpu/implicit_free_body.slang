// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Adapted from MuJoCo Warp _src/forward.py, _implicit_free_body, and _src/math.py.

#include "model.slang"
#include "damping.slang"

[[vk::binding(0, 0)]] StructuredBuffer<ModelDimensions> model;
[[vk::binding(1, 0)]] StructuredBuffer<uint4> freeBodies; // header count and damping flag, then root and first DOF
[[vk::binding(2, 0)]] StructuredBuffer<float> timestep;
[[vk::binding(3, 0)]] StructuredBuffer<float> qvel;
[[vk::binding(4, 0)]] StructuredBuffer<Pose> bodyPose;
[[vk::binding(5, 0)]] StructuredBuffer<SpatialInertia> compositeInertia;
[[vk::binding(6, 0)]] StructuredBuffer<float4> subtreeCom;
[[vk::binding(7, 0)]] StructuredBuffer<DofModel> dofs;
[[vk::binding(8, 0)]] StructuredBuffer<float4> damping;
[[vk::binding(9, 0)]] StructuredBuffer<float> massMatrix;
[[vk::binding(10, 0)]] StructuredBuffer<float> rhs;
[[vk::binding(11, 0)]] RWStructuredBuffer<float> acceleration;
[[vk::binding(12, 0)]] RWStructuredBuffer<float> symmetricMass;
[[vk::binding(13, 0)]] StructuredBuffer<uint> coupledDofs;

[shader("compute")]
[numthreads(64, 1, 1)]
void restore_implicit_free_mass(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= freeBodies[0].x)
        return;
    const uint first = freeBodies[id.x + 1].y;
    const uint base = id.y * model[0].layout.w;
    for (uint row = first; row < first + 6; ++row)
    {
        const uint2 span = dofs[row].indices.zw;
        for (uint col = 0; col < span.y; ++col)
        {
            const uint index = base + span.x + col;
            symmetricMass[index] = massMatrix[index];
        }
    }
}

void solveFreeBody(inout float matrix[6][6], inout float value[6])
{
    // Eliminate the right-hand side along with the pivoted local matrix.
    for (uint row = 0; row < 6; ++row)
    {
        uint pivot = row;
        for (uint next = row + 1; next < 6; ++next)
            if (abs(matrix[next][row]) > abs(matrix[pivot][row]))
                pivot = next;

        if (pivot != row)
        {
            for (uint col = row; col < 6; ++col)
            {
                const float saved = matrix[row][col];
                matrix[row][col] = matrix[pivot][col];
                matrix[pivot][col] = saved;
            }
            const float saved = value[row];
            value[row] = value[pivot];
            value[pivot] = saved;
        }
        const float inverse = 1.0 / matrix[row][row];
        for (uint next = row + 1; next < 6; ++next)
        {
            const float factor = matrix[next][row] * inverse;
            for (uint col = row + 1; col < 6; ++col)
                matrix[next][col] -= factor * matrix[row][col];
            value[next] -= factor * value[row];
        }
    }
    for (int row = 5; row >= 0; --row)
    {
        for (int col = row + 1; col < 6; ++col)
            value[row] -= matrix[row][col] * value[col];
        value[row] /= matrix[row][row];
    }
}

void solveFreeBodyGyro<let Discrete : bool>(uint3 id)
{
    const uint2 configuration = freeBodies[0].xy; // body count and damping enabled
    if (id.x >= configuration.x)
        return;
    const ModelDimensions dimensions = model[0];
    const uint2 root = freeBodies[id.x + 1].xy;
    const uint bodyIndex = id.y * dimensions.state.z + root.x;
    const uint dofBase = id.y * dimensions.state.y + root.y;
    if (Discrete)
    {
        for (uint i = 0; i < 6; ++i)
            if (coupledDofs[dofBase + i] != 0)
                return;
    }
    const uint massBase = id.y * dimensions.layout.w;
    const float dt = timestep[id.y];
    float matrix[6][6];
    float value[6];
    for (uint row = 0; row < 6; ++row)
    {
        for (uint col = 0; col <= row; ++col)
        {
            matrix[row][col] = 0.0;
            matrix[col][row] = 0.0;
        }
    }
    for (uint row = 0; row < 6; ++row)
    {
        // Compact rows can store only the diagonal.
        const uint index = root.y + row;
        const uint2 span = dofs[index].indices.zw;
        uint columnDof = index;
        for (uint entryIndex = span.y; entryIndex > 0; --entryIndex)
        {
            const uint col = columnDof - root.y;
            const float entry = massMatrix[massBase + span.x + entryIndex - 1];
            matrix[row][col] = entry;
            matrix[col][row] = entry;
            columnDof = dofs[columnDof].indices.y;
        }
        if (!Discrete && configuration.y != 0)
            matrix[row][row] += dt * dampingDerivative(damping[index].xyz, qvel[dofBase + row]);
        value[row] = rhs[dofBase + row];
    }

    const Pose pose = bodyPose[bodyIndex];
    const CartesianFrame world = cartesianFrame(pose.position.xyz, pose.rotation);
    const SpatialInertia inertia = compositeInertia[bodyIndex];
    const float mass = inertia.firstMomentMass.w;
    const float3x3 inertiaMatrix = float3x3(inertia.rotation.x, inertia.rotation.w, inertia.products.x,
                                           inertia.rotation.w, inertia.rotation.y, inertia.products.y,
                                           inertia.products.x, inertia.products.y, inertia.rotation.z);
    const float3x3 rotation = float3x3(world.row0.xyz, world.row1.xyz, world.row2.xyz);
    const float3 offset = subtreeCom[bodyIndex].xyz - pose.position.xyz;
    const float3 velocity = mul(rotation, float3(qvel[dofBase + 3], qvel[dofBase + 4], qvel[dofBase + 5]));
    const float3 inertiaVelocity = mul(inertiaMatrix, velocity);
    const float3 crossOffset = cross(velocity, offset);
    const float dotOffset = dot(velocity, offset);
    for (uint col = 0; col < 3; ++col)
    {
        const float3 axis = float3(rotation[0][col], rotation[1][col], rotation[2][col]);
        const float3 linear = offset * dot(velocity, axis) - axis * dotOffset + cross(crossOffset, axis);
        const float3 torque = -mass * cross(offset, linear) + cross(velocity, mul(inertiaMatrix, axis)) -
                              cross(inertiaVelocity, axis);
        const float3 angular = mul(transpose(rotation), torque);
        for (uint row = 0; row < 3; ++row)
        {
            matrix[row][3 + col] -= dt * mass * linear[row];
            matrix[3 + row][3 + col] += dt * angular[row];
        }
    }
    solveFreeBody(matrix, value);
    for (uint row = 0; row < 6; ++row)
        acceleration[dofBase + row] = value[row];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void solve_implicit_free_bodies(uint3 id : SV_DispatchThreadID)
{
    solveFreeBodyGyro<false>(id);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void solve_discrete_free_bodies(uint3 id : SV_DispatchThreadID)
{
    solveFreeBodyGyro<true>(id);
}
