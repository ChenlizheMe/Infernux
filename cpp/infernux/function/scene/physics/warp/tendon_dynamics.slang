// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// MuJoCo Warp tendon generalized force and armature projection.

#include "tendon_data.slang"

[[vk::binding(0, 0)]] StructuredBuffer<uint4> tendonDimensions;
[[vk::binding(1, 0)]] StructuredBuffer<TendonModel> tendons;
[[vk::binding(2, 0)]] StructuredBuffer<float> tendonJacobian;
[[vk::binding(3, 0)]] StructuredBuffer<TendonState> tendonState;
[[vk::binding(4, 0)]] StructuredBuffer<uint> dofOffsets;
[[vk::binding(5, 0)]] StructuredBuffer<uint2> dofContributions;
[[vk::binding(6, 0)]] StructuredBuffer<uint> massOffsets;
[[vk::binding(7, 0)]] StructuredBuffer<uint4> massContributions;
[[vk::binding(8, 0)]] RWStructuredBuffer<float> qfrcPassive;
[[vk::binding(9, 0)]] RWStructuredBuffer<float> qfrcBias;
[[vk::binding(10, 0)]] RWStructuredBuffer<float> massMatrix;

[shader("compute")]
[numthreads(64, 1, 1)]
void add_tendon_forces(uint3 id : SV_DispatchThreadID)
{
    const uint nv = tendonDimensions[1].y;
    if (id.x >= nv)
        return;
    const uint4 size = tendonDimensions[0];
    const uint tendonBase = id.y * size.x;
    const uint jacobianBase = id.y * size.z;
    float2 sum = float2(0.0);
    for (uint i = dofOffsets[id.x]; i < dofOffsets[id.x + 1]; ++i)
    {
        const uint2 link = dofContributions[i]; // tendon, packed Jacobian index
        const TendonState tendon = tendonState[tendonBase + link.x];
        sum += tendonJacobian[jacobianBase + link.y] * float2(tendon.passiveForce, tendon.inertialBias);
    }
    const uint index = id.y * nv + id.x;
    qfrcPassive[index] += sum.x;
    qfrcBias[index] += sum.y;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void add_tendon_armature(uint3 id : SV_DispatchThreadID)
{
    const uint nM = tendonDimensions[1].w;
    if (id.x >= nM)
        return;
    const uint base = id.y * tendonDimensions[0].z;
    float sum = 0.0;
    for (uint i = massOffsets[id.x]; i < massOffsets[id.x + 1]; ++i)
    {
        const uint4 link = massContributions[i]; // tendon, two packed Jacobian indices
        sum += tendons[link.x].springArmature.z * tendonJacobian[base + link.y] * tendonJacobian[base + link.z];
    }
    massMatrix[id.y * nM + id.x] += sum;
}
