// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/smooth.py site transmission.

struct SiteActuatorModel
{
    uint4 span; // actuator, first sparse column, sparse column count
    uint4 bodies; // site body and tree root, reference body and tree root
    Pose site;
    Pose reference;
};

[[vk::binding(18, 0)]] StructuredBuffer<SiteActuatorModel> siteActuators;
[[vk::binding(19, 0)]] StructuredBuffer<uint2> siteMomentColumns; // DOF, reference column flag
[[vk::binding(20, 0)]] StructuredBuffer<Pose> siteBodyPose;
[[vk::binding(21, 0)]] StructuredBuffer<float4> siteTreeCom;
[[vk::binding(22, 0)]] StructuredBuffer<SpatialMotion> siteCenteredDof;
[[vk::binding(23, 0)]] RWStructuredBuffer<float> siteActuatorMoment;

Pose actuatorSitePose(uint body, Pose local)
{
    const Pose parent = siteBodyPose[body];
    return {float4(parent.position.xyz + rotateVector(local.position.xyz, parent.rotation), 0.0),
            mulQuat(parent.rotation, local.rotation)};
}

[shader("compute")]
[numthreads(64, 1, 1)]
void evaluate_site_transmission(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= actuatorDimensions[1].x)
        return;

    const SiteActuatorModel site = siteActuators[id.x];
    const ActuatorModel actuator = actuators[site.span.x];
    const uint bodyBase = id.y * model[0].state.z;
    const uint dofBase = id.y * model[0].state.y;
    const uint momentBase = id.y * actuatorDimensions[0].w;
    const Pose driven = actuatorSitePose(bodyBase + site.bodies.x, site.site);
    const float3 drivenOffset = driven.position.xyz - siteTreeCom[bodyBase + site.bodies.y].xyz;
    float3 referenceOffset = float3(0.0);
    float4 wrenchFrame = driven.rotation;
    float coordinate = 0.0;
    if ((actuator.indices.w & ReferenceSite) != 0)
    {
        const Pose reference = actuatorSitePose(bodyBase + site.bodies.z, site.reference);
        const float4 inverse = float4(reference.rotation.x, -reference.rotation.yzw);
        if (any(actuator.gearFirst.xyz != float3(0.0)))
        {
            const float3 displacement = rotateVector(driven.position.xyz - reference.position.xyz, inverse);
            coordinate = dot(displacement, actuator.gearFirst.xyz);
        }
        if (any(actuator.gearLast.xyz != float3(0.0)))
        {
            const float3 rotation = rotationVector(mulQuat(inverse, driven.rotation));
            coordinate += dot(rotation, actuator.gearLast.xyz);
        }
        referenceOffset = reference.position.xyz - siteTreeCom[bodyBase + site.bodies.w].xyz;
        wrenchFrame = reference.rotation;
    }

    const float3 force = rotateVector(actuator.gearFirst.xyz, wrenchFrame);
    const float3 torque = rotateVector(actuator.gearLast.xyz, wrenchFrame);
    float velocity = 0.0;
    for (uint column = site.span.y; column < site.span.y + site.span.z; ++column)
    {
        const uint2 entry = siteMomentColumns[column];
        const bool reference = entry.y != 0;
        const uint dof = dofBase + entry.x;
        const SpatialMotion motion = siteCenteredDof[dof];
        const float3 offset = reference ? referenceOffset : drivenOffset;
        const float3 translation = motion.linear.xyz + cross(motion.angular.xyz, offset);
        const float moment = (reference ? -1.0 : 1.0) * (dot(translation, force) + dot(motion.angular.xyz, torque));
        siteActuatorMoment[momentBase + column] = moment;
        velocity += moment * qvel[dof];
    }
    const uint index = id.y * actuatorDimensions[0].x + site.span.x;
    const ActuatorStep result = {float4(0.0), float4(0.0), float4(coordinate, velocity, 0.0, 0.0)};
    actuatorStep[index] = result;
}
