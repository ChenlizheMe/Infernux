// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Shared projection for MuJoCo Warp _src/smooth.py and _src/passive.py contact forces.

#include "contact.slang"

static const uint ContactProjectionWidth = 64;

struct ContactProjectionWorld
{
    uint geometryCount;
    uint dofCount;
    uint bodyBase;
    uint dofBase;
    uint contactBase;
    uint contactCount;
};

struct NormalContact
{
    uint2 bodies;
    float3 offsetA;
    float3 offsetB;
    float3 force;
};

struct ContactForceSum
{
    float force;
    uint count;
};

[[vk::binding(21, 0)]] StructuredBuffer<float4> projectionTreeCom;
[[vk::binding(22, 0)]] StructuredBuffer<SpatialMotion> projectionCenteredDof;
[[vk::binding(24, 0)]] StructuredBuffer<DofModel> projectionDofs;
[[vk::binding(27, 0)]] StructuredBuffer<Contact> projectionContacts;
[[vk::binding(28, 0)]] RWStructuredBuffer<Atomic<uint>> projectionContactCount;
[[vk::binding(29, 0)]] StructuredBuffer<uint4> projectionTopology;

groupshared NormalContact projectionContactTile[ContactProjectionWidth];
groupshared uint projectionUniformCount;

ContactProjectionWorld contactProjectionWorld(ModelDimensions model, uint stride, uint world, uint lane)
{
    if (lane == 0)
        projectionUniformCount = min(projectionContactCount[world].load(), stride);
    GroupMemoryBarrierWithGroupSync();
    const uint count = workgroupUniformLoad(projectionUniformCount);
    return {model.layout.x, model.state.y, world * model.state.z, world * model.state.y, world * stride, count};
}

NormalContact loadNormalContact(ContactProjectionWorld world, uint index, uint target, bool material)
{
    const Contact contact = projectionContacts[world.contactBase + index];
    NormalContact result = {uint2(NoIndex), float3(0.0), float3(0.0), float3(0.0)};
    const float strength = material ? contact.tangentAdhesion.w : 1.0;
    if ((contact.geometry.w & ContactConstraint) == 0 || strength == 0.0)
        return result;

    const uint a = projectionTopology[contact.geometry.x].x;
    const uint b = projectionTopology[contact.geometry.y].x;
    if (!material && a != target && b != target)
        return result;

    const uint rootA = projectionTopology[world.geometryCount + a].y;
    const uint rootB = projectionTopology[world.geometryCount + b].y;
    result.bodies = uint2(a, b);
    result.offsetA = contact.positionDistance.xyz - projectionTreeCom[world.bodyBase + rootA].xyz;
    result.offsetB = contact.positionDistance.xyz - projectionTreeCom[world.bodyBase + rootB].xyz;
    result.force = strength * contact.normalMargin.xyz;
    return result;
}

ContactForceSum projectContactNormals(ContactProjectionWorld world, uint lane, uint dof, uint target, bool material)
{
    uint firstBody = 0;
    uint endBody = 0;
    SpatialMotion motion = spatial(float3(0.0), float3(0.0));
    if (dof < world.dofCount)
    {
        firstBody = projectionDofs[dof].indices.x;
        endBody = firstBody + projectionTopology[world.geometryCount + firstBody].w;
        motion = projectionCenteredDof[world.dofBase + dof];
    }
    ContactForceSum sum = {0.0, 0u};
    for (uint block = 0; block < world.contactCount; block += ContactProjectionWidth)
    {
        const uint width = min(world.contactCount - block, ContactProjectionWidth);
        if (lane < width)
            projectionContactTile[lane] = loadNormalContact(world, block + lane, target, material);
        GroupMemoryBarrierWithGroupSync();
        const uint activeWidth = dof < world.dofCount ? width : 0;
        for (uint j = 0; j < activeWidth; ++j)
        {
            const NormalContact contact = projectionContactTile[j];
            if (contact.bodies.x == NoIndex)
                continue;
            ++sum.count;
            const bool a = contact.bodies.x >= firstBody && contact.bodies.x < endBody;
            const bool b = contact.bodies.y >= firstBody && contact.bodies.y < endBody;
            if (a != b)
            {
                const float3 translation = motion.linear.xyz +
                    cross(motion.angular.xyz, a ? contact.offsetA : contact.offsetB);
                sum.force += (a ? 1.0 : -1.0) * dot(contact.force, translation);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    return sum;
}
