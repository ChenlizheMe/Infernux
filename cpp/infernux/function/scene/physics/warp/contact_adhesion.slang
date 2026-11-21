// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/passive.py material adhesion.

#include "model.slang"
#include "contact_force_projection.slang"

[[vk::binding(0, 0)]] StructuredBuffer<ModelDimensions> model;
[[vk::binding(1, 0)]] StructuredBuffer<uint4> constraintSettings;
[[vk::binding(6, 0)]] RWStructuredBuffer<float> passiveForce;

[shader("compute")]
[numthreads(64, 1, 1)]
void add_contact_adhesion(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    const ContactProjectionWorld world = contactProjectionWorld(model[0], constraintSettings[0].y, group.y, lane);
    const uint dof = group.x * ContactProjectionWidth + lane;
    const ContactForceSum sum = projectContactNormals(world, lane, dof, NoIndex, true);
    if (dof < world.dofCount)
        passiveForce[world.dofBase + dof] += sum.force;
}
