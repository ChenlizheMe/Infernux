// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/smooth.py body transmission.

[[vk::binding(26, 0)]] StructuredBuffer<uint4> bodyActuators; // actuator, moment column start, driven body

groupshared float bodyMomentSpeed[ContactProjectionWidth];

[shader("compute")]
[numthreads(64, 1, 1)]
void evaluate_body_transmission(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    const uint4 target = bodyActuators[group.x];
    const ContactProjectionWorld world = contactProjectionWorld(model[0], actuatorDimensions[2].x, group.y, lane);
    const uint momentBase = group.y * actuatorDimensions[0].w + target.y;

    float velocity = 0.0;
    for (uint dofBlock = 0; dofBlock < world.dofCount; dofBlock += ContactProjectionWidth)
    {
        const uint dof = dofBlock + lane;
        const ContactForceSum sum = projectContactNormals(world, lane, dof, target.z, false);
        if (dof < world.dofCount)
        {
            const float moment = sum.force / float(max(sum.count, 1u));
            actuatorMoments[momentBase + dof] = moment;
            velocity += moment * qvel[world.dofBase + dof];
        }
    }
    bodyMomentSpeed[lane] = velocity;
    GroupMemoryBarrierWithGroupSync();
    for (uint width = ContactProjectionWidth / 2; width > 0; width >>= 1u)
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
