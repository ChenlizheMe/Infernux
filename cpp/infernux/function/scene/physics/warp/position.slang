// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/smooth.py position kernels.

#include "model.slang"

[[vk::binding(0, 0)]] StructuredBuffer<ModelDimensions> model;
[[vk::binding(1, 0)]] StructuredBuffer<BodyModel> bodies;
[[vk::binding(2, 0)]] StructuredBuffer<JointModel> joints;
[[vk::binding(3, 0)]] StructuredBuffer<float> qpos;
[[vk::binding(4, 0)]] StructuredBuffer<uint> branches;
[[vk::binding(5, 0)]] StructuredBuffer<Pose> mocap;
[[vk::binding(6, 0)]] RWStructuredBuffer<Pose> bodyPose;
[[vk::binding(7, 0)]] RWStructuredBuffer<JointPose> jointPose;
[[vk::binding(8, 0)]] RWStructuredBuffer<CartesianFrame> inertialFrame;
[[vk::binding(9, 0)]] StructuredBuffer<GeomModel> geoms;
[[vk::binding(10, 0)]] RWStructuredBuffer<CartesianFrame> geomFrame;
[[vk::binding(11, 0)]] StructuredBuffer<float4> publishedBodies;
[[vk::binding(12, 0)]] StructuredBuffer<uint4> publishedRaySettings;

float4 loadQuat(uint first)
{
    return normalize(float4(qpos[first], qpos[first + 1], qpos[first + 2], qpos[first + 3]));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void kinematics_branch(uint3 id : SV_DispatchThreadID)
{
    const ModelDimensions size = model[0];
    const uint branchCount = size.layout.z;
    if (id.x >= branchCount)
        return;
    const uint bodyBase = id.y * size.state.z;
    const uint jointBase = id.y * size.state.w;
    const uint qposBase = id.y * size.state.x;
    // Warp's leaf branches can share a prefix. Carry each branch's parent
    // pose locally and let its selected writer publish shared ancestors.
    Pose parentPose = bodyPose[bodyBase];
    for (uint index = branches[id.x]; index < branches[id.x + 1]; ++index)
    {
        const uint bodyIndex = branches[branchCount + 1 + index];
        const BodyModel body = bodies[bodyIndex];
        const uint parent = body.hierarchy.x;
        const uint firstJoint = body.hierarchy.y;
        const uint jointCount = body.hierarchy.z;
        const bool publish = body.subtree.z == id.x;
        Pose pose;
        if (jointCount == 1)
        {
            const JointModel joint = joints[firstJoint];
            if (joint.indices.x == JointFree)
            {
                const uint input = qposBase + joint.indices.y;
                pose.position = float4(qpos[input], qpos[input + 1], qpos[input + 2], 0.0);
                pose.rotation = loadQuat(input + 3);
                JointPose frame;
                frame.anchor = pose.position;
                frame.axis = float4(joint.axis.xyz, 0.0);
                if (publish)
                {
                    bodyPose[bodyBase + bodyIndex] = pose;
                    jointPose[jointBase + firstJoint] = frame;
                }
                parentPose = pose;
                continue;
            }
        }

        pose.position = body.position;
        pose.rotation = body.rotation;
        if (body.hierarchy.w != NoIndex)
            pose = mocap[id.y * size.layout.y + body.hierarchy.w];
        if (parent != NoIndex)
        {
            pose.position.xyz = parentPose.position.xyz + rotateVector(pose.position.xyz, parentPose.rotation);
            pose.rotation = mulQuat(parentPose.rotation, pose.rotation);
        }
        for (uint local = 0; local < jointCount; ++local)
        {
            const uint jointIndex = firstJoint + local;
            const JointModel joint = joints[jointIndex];
            const uint input = qposBase + joint.indices.y;
            const uint type = joint.indices.x;
            JointPose frame;
            frame.anchor = float4(pose.position.xyz + rotateVector(joint.anchor.xyz, pose.rotation), 0.0);
            frame.axis = float4(rotateVector(joint.axis.xyz, pose.rotation), 0.0);
            if (type == JointSlide)
            {
                pose.position.xyz += frame.axis.xyz * (qpos[input] - joint.axis.w);
            }
            else
            {
                float4 localRotation;
                if (type == JointBall)
                    localRotation = loadQuat(input);
                else
                    localRotation = axisAngleQuat(joint.axis.xyz, qpos[input] - joint.axis.w);
                pose.rotation = mulQuat(pose.rotation, localRotation);
                pose.position.xyz = frame.anchor.xyz - rotateVector(joint.anchor.xyz, pose.rotation);
            }
            if (publish)
                jointPose[jointBase + jointIndex] = frame;
        }
        pose.rotation = normalize(pose.rotation);
        if (publish)
            bodyPose[bodyBase + bodyIndex] = pose;
        parentPose = pose;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void body_inertial_frames(uint3 id : SV_DispatchThreadID)
{
    const uint count = model[0].state.z;
    if (id.x >= count)
        return;
    const uint index = id.y * count + id.x;
    const BodyModel body = bodies[id.x];
    const Pose pose = bodyPose[index];
    inertialFrame[index] = cartesianFrame(
        pose.position.xyz + rotateVector(body.inertialPosition.xyz, pose.rotation),
        mulQuat(pose.rotation, body.inertialRotation));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void geom_local_to_global(uint3 id : SV_DispatchThreadID)
{
    const ModelDimensions size = model[0];
    if (id.x >= size.layout.x)
        return;
    const GeomModel geom = geoms[id.x];
    const Pose pose = bodyPose[id.y * size.state.z + geom.indices.x];
    geomFrame[id.y * size.layout.x + id.x] = cartesianFrame(
        pose.position.xyz + rotateVector(geom.position.xyz, pose.rotation), mulQuat(pose.rotation, geom.rotation));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void published_geom_local_to_global(uint3 id : SV_DispatchThreadID)
{
    const uint4 settings = publishedRaySettings[0]; // Geometry count, rays, input worlds, first body row.
    if (id.x >= settings.x)
        return;
    const GeomModel geom = geoms[id.x];
    const uint base = 5 * (settings.w + geom.indices.x); // Published position, rotation, COM and two velocities.
    const float4 position = publishedBodies[base];
    const float4 rotation = publishedBodies[base + 1];
    geomFrame[id.x] = cartesianFrame(position.xyz + rotateVector(geom.position.xyz, rotation),
                                     mulQuat(rotation, geom.rotation));
}
