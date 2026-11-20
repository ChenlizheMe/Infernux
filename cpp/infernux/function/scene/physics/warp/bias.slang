// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/smooth.py com_vel and rne bias forces.

#include "model.slang"

[[vk::binding(0, 0)]] StructuredBuffer<ModelDimensions> model;
[[vk::binding(1, 0)]] StructuredBuffer<BodyModel> bodies;
[[vk::binding(2, 0)]] StructuredBuffer<JointModel> joints;
[[vk::binding(3, 0)]] StructuredBuffer<uint> branches;
[[vk::binding(4, 0)]] StructuredBuffer<float> qvel;
[[vk::binding(5, 0)]] StructuredBuffer<SpatialMotion> centeredDof;
[[vk::binding(6, 0)]] RWStructuredBuffer<SpatialMotion> centeredDofDot;
[[vk::binding(7, 0)]] RWStructuredBuffer<SpatialMotion> bodyVelocity;
[[vk::binding(8, 0)]] StructuredBuffer<float4> gravity;
[[vk::binding(9, 0)]] RWStructuredBuffer<SpatialMotion> biasAcceleration;
[[vk::binding(10, 0)]] StructuredBuffer<SpatialInertia> centeredInertia;
[[vk::binding(11, 0)]] RWStructuredBuffer<SpatialMotion> bodyForce;
[[vk::binding(12, 0)]] RWStructuredBuffer<SpatialMotion> subtreeForce;
[[vk::binding(13, 0)]] StructuredBuffer<DofModel> dofs;
[[vk::binding(14, 0)]] RWStructuredBuffer<float> qfrcBias;

[shader("compute")]
[numthreads(64, 1, 1)]
void comvel_branch(uint3 id : SV_DispatchThreadID)
{
    const ModelDimensions size = model[0];
    const uint branchCount = size.layout.z;
    if (id.x >= branchCount)
        return;
    const uint bodyBase = id.y * size.state.z;
    const uint dofBase = id.y * size.state.y;
    const SpatialMotion zero = spatial(float3(0.0), float3(0.0));
    SpatialMotion velocity = zero;
    if (id.x == 0)
        bodyVelocity[bodyBase] = zero;

    for (uint index = branches[id.x]; index < branches[id.x + 1]; ++index)
    {
        const uint bodyIndex = branches[branchCount + 1 + index];
        const BodyModel body = bodies[bodyIndex];
        const bool publish = body.subtree.z == id.x;
        for (uint local = 0; local < body.hierarchy.z; ++local)
        {
            const JointModel joint = joints[body.hierarchy.y + local];
            uint first = dofBase + joint.indices.z;
            const uint type = joint.indices.x;
            if (type == JointFree)
            {
                for (uint axis = 0; axis < 3; ++axis)
                {
                    const uint dof = first + axis;
                    addMotion(velocity, centeredDof[dof], qvel[dof]);
                    if (publish)
                        centeredDofDot[dof] = zero;
                }
                first += 3;
            }
            const uint count = type == JointFree || type == JointBall ? 3 : 1;
            const SpatialMotion jointVelocity = velocity;
            for (uint axis = 0; axis < count; ++axis)
            {
                const uint dof = first + axis;
                const SpatialMotion coordinate = centeredDof[dof];
                if (publish)
                    centeredDofDot[dof] = motionCross(jointVelocity, coordinate);
                addMotion(velocity, coordinate, qvel[dof]);
            }
        }
        if (publish)
            bodyVelocity[bodyBase + bodyIndex] = velocity;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void bias_acceleration_branch(uint3 id : SV_DispatchThreadID)
{
    const ModelDimensions size = model[0];
    const uint branchCount = size.layout.z;
    if (id.x >= branchCount)
        return;
    const uint bodyBase = id.y * size.state.z;
    const uint dofBase = id.y * size.state.y;
    SpatialMotion acceleration = spatial(float3(0.0), -gravity[id.y].xyz);
    if (id.x == 0)
        biasAcceleration[bodyBase] = acceleration;
    for (uint index = branches[id.x]; index < branches[id.x + 1]; ++index)
    {
        const uint bodyIndex = branches[branchCount + 1 + index];
        const BodyModel body = bodies[bodyIndex];
        for (uint offset = 0; offset < body.dofs.y; ++offset)
        {
            const uint dof = dofBase + body.dofs.x + offset;
            addMotion(acceleration, centeredDofDot[dof], qvel[dof]);
        }
        if (body.subtree.z == id.x)
            biasAcceleration[bodyBase + bodyIndex] = acceleration;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void bias_force_body(uint3 id : SV_DispatchThreadID)
{
    const uint count = model[0].state.z;
    if (id.x >= count)
        return;
    const uint index = id.y * count + id.x;
    if (id.x == 0)
    {
        bodyForce[index] = spatial(float3(0.0), float3(0.0));
        return;
    }
    const SpatialInertia inertia = centeredInertia[index];
    const SpatialMotion velocity = bodyVelocity[index];
    SpatialMotion force = inertiaTimesMotion(inertia, biasAcceleration[index]);
    addMotion(force, motionCrossForce(velocity, inertiaTimesMotion(inertia, velocity)), 1.0);
    bodyForce[index] = force;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void bias_force_subtree(uint3 id : SV_DispatchThreadID)
{
    const uint count = model[0].state.z;
    if (id.x == 0 || id.x >= count)
        return;
    const uint base = id.y * count;
    SpatialMotion sum = bodyForce[base + id.x];
    const uint end = id.x + bodies[id.x].subtree.y;
    for (uint child = id.x + 1; child < end; ++child)
        addMotion(sum, bodyForce[base + child], 1.0);
    subtreeForce[base + id.x] = sum;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void bias_force_world(uint3 id : SV_DispatchThreadID)
{
    const uint count = model[0].state.z;
    const uint base = id.y * count;
    SpatialMotion sum = spatial(float3(0.0), float3(0.0));
    for (uint root = 1; root < count; root += bodies[root].subtree.y)
        addMotion(sum, subtreeForce[base + root], 1.0);
    subtreeForce[base] = sum;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void project_bias(uint3 id : SV_DispatchThreadID)
{
    const ModelDimensions size = model[0];
    if (id.x >= size.state.y)
        return;
    const uint dof = id.y * size.state.y + id.x;
    const uint body = id.y * size.state.z + dofs[id.x].indices.x;
    qfrcBias[dof] = spatialDot(centeredDof[dof], subtreeForce[body]);
}
