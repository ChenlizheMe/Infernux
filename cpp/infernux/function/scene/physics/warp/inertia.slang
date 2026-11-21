// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/smooth.py com_pos and crb.

#include "model.slang"

[[vk::binding(0, 0)]] StructuredBuffer<ModelDimensions> model;
[[vk::binding(1, 0)]] StructuredBuffer<BodyModel> bodies;
[[vk::binding(2, 0)]] StructuredBuffer<CartesianFrame> inertialFrame;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> subtreeCom;
[[vk::binding(4, 0)]] RWStructuredBuffer<SpatialInertia> centeredInertia;
[[vk::binding(5, 0)]] StructuredBuffer<JointModel> joints;
[[vk::binding(6, 0)]] StructuredBuffer<Pose> bodyPose;
[[vk::binding(7, 0)]] StructuredBuffer<JointPose> jointPose;
[[vk::binding(8, 0)]] RWStructuredBuffer<SpatialMotion> centeredDof;
[[vk::binding(9, 0)]] RWStructuredBuffer<SpatialInertia> compositeInertia;
[[vk::binding(10, 0)]] StructuredBuffer<DofModel> dofs;
[[vk::binding(11, 0)]] RWStructuredBuffer<float> massMatrix;

[shader("compute")]
[numthreads(64, 1, 1)]
void subtree_com(uint3 id : SV_DispatchThreadID)
{
    const uint count = model[0].state.z;
    if (id.x == 0 || id.x >= count)
        return;
    const uint base = id.y * count;
    const BodyModel body = bodies[id.x];
    float3 weighted = float3(0.0);
    for (uint child = id.x; child < id.x + body.subtree.y; ++child)
        weighted += inertialFrame[base + child].position.xyz * bodies[child].inertiaMass.w;
    const float mass = body.subtreeMass.x;
    const float3 result = mass < 1.0e-15 ? inertialFrame[base + id.x].position.xyz : weighted / mass;
    subtreeCom[base + id.x] = float4(result, 0.0);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void subtree_com_world(uint3 id : SV_DispatchThreadID)
{
    const uint count = model[0].state.z;
    const uint base = id.y * count;
    float3 weighted = inertialFrame[base].position.xyz * bodies[0].inertiaMass.w;
    for (uint root = 1; root < count; root += bodies[root].subtree.y)
        weighted += subtreeCom[base + root].xyz * bodies[root].subtreeMass.x;
    const float mass = bodies[0].subtreeMass.x;
    const float3 result = mass < 1.0e-15 ? inertialFrame[base].position.xyz : weighted / mass;
    subtreeCom[base] = float4(result, 0.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cinert(uint3 id : SV_DispatchThreadID)
{
    const uint count = model[0].state.z;
    if (id.x >= count)
        return;
    const uint base = id.y * count;
    const BodyModel body = bodies[id.x];
    const CartesianFrame frame = inertialFrame[base + id.x];
    const float3 inertia = body.inertiaMass.xyz;
    const float mass = body.inertiaMass.w;
    const float3 offset = frame.position.xyz - subtreeCom[base + body.subtree.x].xyz;
    const float3 row0 = frame.row0.xyz * inertia;
    const float3 row1 = frame.row1.xyz * inertia;
    const float3 row2 = frame.row2.xyz * inertia;
    SpatialInertia value;
    value.rotation = float4(dot(row0, frame.row0.xyz), dot(row1, frame.row1.xyz), dot(row2, frame.row2.xyz),
                            dot(row0, frame.row1.xyz));
    value.rotation.xyz += mass * (offset.yzx * offset.yzx + offset.zxy * offset.zxy);
    value.rotation.w -= mass * offset.x * offset.y;
    value.products = float4(dot(row0, frame.row2.xyz) - mass * offset.x * offset.z,
                            dot(row1, frame.row2.xyz) - mass * offset.y * offset.z, 0.0, 0.0);
    value.firstMomentMass = float4(mass * offset, mass);
    centeredInertia[base + id.x] = value;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cdof(uint3 id : SV_DispatchThreadID)
{
    const ModelDimensions size = model[0];
    if (id.x >= size.state.w)
        return;
    const JointModel joint = joints[id.x];
    const uint body = joint.indices.w;
    const uint bodyBase = id.y * size.state.z;
    const uint type = joint.indices.x;
    uint first = id.y * size.state.y + joint.indices.z;
    const JointPose pose = jointPose[id.y * size.state.w + id.x];
    const float3 offset = subtreeCom[bodyBase + bodies[body].subtree.x].xyz - pose.anchor.xyz;
    if (type == JointFree)
    {
        centeredDof[first] = spatial(float3(0.0), float3(1.0, 0.0, 0.0));
        centeredDof[first + 1] = spatial(float3(0.0), float3(0.0, 1.0, 0.0));
        centeredDof[first + 2] = spatial(float3(0.0), float3(0.0, 0.0, 1.0));
        first += 3;
    }
    if (type == JointFree || type == JointBall)
    {
        const float4 rotation = bodyPose[bodyBase + body].rotation;
        const float3 x = rotateVector(float3(1.0, 0.0, 0.0), rotation);
        const float3 y = rotateVector(float3(0.0, 1.0, 0.0), rotation);
        const float3 z = rotateVector(float3(0.0, 0.0, 1.0), rotation);
        centeredDof[first] = spatial(x, cross(x, offset));
        centeredDof[first + 1] = spatial(y, cross(y, offset));
        centeredDof[first + 2] = spatial(z, cross(z, offset));
    }
    else if (type == JointSlide)
        centeredDof[first] = spatial(float3(0.0), pose.axis.xyz);
    else
        centeredDof[first] = spatial(pose.axis.xyz, cross(pose.axis.xyz, offset));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void crb_subtree(uint3 id : SV_DispatchThreadID)
{
    const uint count = model[0].state.z;
    if (id.x >= count)
        return;
    const uint base = id.y * count;
    SpatialInertia value = centeredInertia[base + id.x];
    if (id.x != 0)
    {
        const uint end = id.x + bodies[id.x].subtree.y;
        for (uint child = id.x + 1; child < end; ++child)
        {
            const SpatialInertia next = centeredInertia[base + child];
            value.rotation += next.rotation;
            value.products += next.products;
            value.firstMomentMass += next.firstMomentMass;
        }
    }
    compositeInertia[base + id.x] = value;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void assemble_mass(uint3 id : SV_DispatchThreadID)
{
    const ModelDimensions size = model[0];
    if (id.x >= size.state.y)
        return;
    const uint base = id.y * size.state.y;
    const DofModel dof = dofs[id.x];
    const uint row = id.y * size.layout.w + dof.indices.z;
    const SpatialMotion coordinate = centeredDof[base + id.x];
    const SpatialMotion response = inertiaTimesMotion(compositeInertia[id.y * size.state.z + dof.indices.x],
                                                     coordinate);
    uint offset = dof.indices.w - 1;
    massMatrix[row + offset] = dof.armature + spatialDot(coordinate, response);
    uint parent = dof.indices.y;
    while (offset != 0 && parent != NoIndex)
    {
        --offset;
        massMatrix[row + offset] = spatialDot(centeredDof[base + parent], response);
        parent = dofs[parent].indices.y;
    }
}
