// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/smooth.py body transmission.

#include "contact.slang"

static const uint BodyMomentWidth = 64;

struct BodyActuatorContact
{
    uint2 bodies;
    float3 offsetA;
    float3 offsetB;
    float3 normal;
};

[[vk::binding(26, 0)]] StructuredBuffer<uint4> bodyActuators; // actuator, moment column start, driven body
[[vk::binding(27, 0)]] StructuredBuffer<Contact> bodyActuatorContacts;
[[vk::binding(28, 0)]] RWStructuredBuffer<Atomic<uint>> bodyActuatorContactCount;
[[vk::binding(29, 0)]] StructuredBuffer<uint4> bodyActuatorTopology; // contact topology with subtree counts in body records

groupshared BodyActuatorContact bodyContactTile[BodyMomentWidth];
groupshared float bodyMomentSpeed[BodyMomentWidth];
groupshared uint bodyTransmissionContactCount;

BodyActuatorContact loadBodyActuatorContact(uint index, uint target, uint nGeom, uint bodyBase)
{
    const Contact contact = bodyActuatorContacts[index];
    BodyActuatorContact result = {uint2(NoIndex), float3(0.0), float3(0.0), float3(0.0)};
    if ((contact.geometry.w & ContactConstraint) == 0)
        return result;

    const uint a = bodyActuatorTopology[contact.geometry.x].x;
    const uint b = bodyActuatorTopology[contact.geometry.y].x;
    if (a != target && b != target)
        return result;

    result.bodies = uint2(a, b);
    const uint rootA = bodyActuatorTopology[nGeom + a].y;
    const uint rootB = bodyActuatorTopology[nGeom + b].y;
    result.offsetA = contact.positionDistance.xyz - transmissionTreeCom[bodyBase + rootA].xyz;
    result.offsetB = contact.positionDistance.xyz - transmissionTreeCom[bodyBase + rootB].xyz;
    result.normal = contact.normalMargin.xyz;
    return result;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void evaluate_body_transmission(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    const uint4 target = bodyActuators[group.x];
    const ModelDimensions dimensions = model[0];
    const uint nGeom = dimensions.layout.x;
    const uint nv = dimensions.state.y;
    const uint bodyBase = group.y * dimensions.state.z;
    const uint dofBase = group.y * nv;
    const uint momentBase = group.y * actuatorDimensions[0].w + target.y;
    const uint contactStride = actuatorDimensions[2].x;
    const uint contactBase = group.y * contactStride;
    if (lane == 0)
        bodyTransmissionContactCount = min(bodyActuatorContactCount[group.y].load(), contactStride);
    GroupMemoryBarrierWithGroupSync();
    const uint contactCount = workgroupUniformLoad(bodyTransmissionContactCount);

    float velocity = 0.0;
    for (uint dofBlock = 0; dofBlock < nv; dofBlock += BodyMomentWidth)
    {
        const uint dof = dofBlock + lane;
        uint firstBody = 0;
        uint endBody = 0;
        SpatialMotion motion = spatial(float3(0.0), float3(0.0));
        if (dof < nv)
        {
            firstBody = actuationDofs[dof].indices.x;
            endBody = firstBody + bodyActuatorTopology[nGeom + firstBody].w;
            motion = transmissionCenteredDof[dofBase + dof];
        }
        float sum = 0.0;
        uint matched = 0;
        for (uint block = 0; block < contactCount; block += BodyMomentWidth)
        {
            const uint width = min(contactCount - block, BodyMomentWidth);
            if (lane < width)
                bodyContactTile[lane] = loadBodyActuatorContact(contactBase + block + lane, target.z, nGeom, bodyBase);
            GroupMemoryBarrierWithGroupSync();
            const uint activeWidth = dof < nv ? width : 0;
            for (uint j = 0; j < activeWidth; ++j)
            {
                const BodyActuatorContact contact = bodyContactTile[j];
                if (contact.bodies.x == NoIndex)
                    continue;
                ++matched;
                const bool a = contact.bodies.x >= firstBody && contact.bodies.x < endBody;
                const bool b = contact.bodies.y >= firstBody && contact.bodies.y < endBody;
                if (a != b)
                {
                    const float3 translation = motion.linear.xyz +
                        cross(motion.angular.xyz, a ? contact.offsetA : contact.offsetB);
                    sum += (a ? 1.0 : -1.0) * dot(contact.normal, translation);
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (dof < nv)
        {
            const float moment = sum / float(max(matched, 1u));
            actuatorMoments[momentBase + dof] = moment;
            velocity += moment * qvel[dofBase + dof];
        }
    }
    bodyMomentSpeed[lane] = velocity;
    GroupMemoryBarrierWithGroupSync();
    for (uint width = BodyMomentWidth / 2; width > 0; width >>= 1u)
    {
        if (lane < width)
            bodyMomentSpeed[lane] += bodyMomentSpeed[lane + width];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0)
    {
        const ActuatorStep result = {float4(0.0), float4(0.0), float4(0.0, bodyMomentSpeed[0], 0.0, 0.0)};
        actuatorStep[group.y * actuatorDimensions[0].x + target.x] = result;
    }
}
