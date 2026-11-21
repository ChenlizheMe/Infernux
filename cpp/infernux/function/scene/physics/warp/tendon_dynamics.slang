// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// MuJoCo Warp tendon generalized force and armature projection.

#include "tendon_data.slang"
#include "damping.slang"

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
[[vk::binding(11, 0)]] StructuredBuffer<float> timestep;
[[vk::binding(12, 0)]] RWStructuredBuffer<float2> discreteTendonTerms;
[[vk::binding(13, 0)]] StructuredBuffer<uint4> discreteMetricLayout;
[[vk::binding(14, 0)]] RWStructuredBuffer<Atomic<uint>> coupledDofs;
[[vk::binding(15, 0)]] RWStructuredBuffer<float> discreteForce;

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

[shader("compute")]
[numthreads(64, 1, 1)]
void add_implicit_tendon_damping(uint3 id : SV_DispatchThreadID)
{
    const uint nM = tendonDimensions[1].w;
    if (id.x >= nM)
        return;
    const uint4 sizes = tendonDimensions[0];
    const uint base = id.y * sizes.z;
    float derivative = 0.0;
    for (uint i = massOffsets[id.x]; i < massOffsets[id.x + 1]; ++i)
    {
        const uint4 link = massContributions[i];
        const float velocity = tendonState[id.y * sizes.x + link.x].velocity;
        const float damping = dampingDerivative(tendons[link.x].damping.xyz, velocity);
        derivative += damping * tendonJacobian[base + link.y] * tendonJacobian[base + link.z];
    }
    massMatrix[id.y * nM + id.x] += timestep[id.y] * derivative;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void evaluate_discrete_tendon_terms(uint3 id : SV_DispatchThreadID)
{
    const uint count = tendonDimensions[0].x;
    if (id.x >= count)
        return;
    const uint index = id.y * count + id.x;
    const TendonModel tendon = tendons[id.x];
    const TendonState state = tendonState[index];
    const float h = timestep[id.y];
    const float offset = state.length - clamp(state.length, tendon.springArmature.x, tendon.springArmature.y);
    const float stiffness = offset != 0.0 ? max(0.0, polynomialForceDerivative(tendon.stiffness.xyz, offset)) : 0.0;
    const float damping = max(0.0, dampingDerivative(tendon.damping.xyz, state.velocity));
    const float shift = h * stiffness;
    discreteTendonTerms[index] = float2(h * (shift + damping), -shift * state.velocity);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void add_discrete_tendon_shift(uint3 id : SV_DispatchThreadID)
{
    const uint dofCount = tendonDimensions[1].y;
    if (id.x >= dofCount)
        return;
    const uint4 size = tendonDimensions[0];
    const uint tendonBase = id.y * size.x;
    const uint jacobianBase = id.y * size.z;
    float force = 0.0;
    bool coupled = false;
    for (uint i = dofOffsets[id.x]; i < dofOffsets[id.x + 1]; ++i)
    {
        const uint2 link = dofContributions[i];
        const float2 terms = discreteTendonTerms[tendonBase + link.x];
        force += terms.y * tendonJacobian[jacobianBase + link.y];
        coupled = coupled || terms.x != 0.0;
    }
    const uint index = id.y * dofCount + id.x;
    discreteForce[index] += force;
    if (coupled)
        coupledDofs[index].store(1u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void add_discrete_tendon_metric(uint3 id : SV_DispatchThreadID)
{
    const uint count = discreteMetricLayout[0].y;
    if (id.x >= count)
        return;
    const uint4 size = tendonDimensions[0];
    const uint tendonBase = id.y * size.x;
    const uint jacobianBase = id.y * size.z;
    float value = 0.0;
    for (uint i = massOffsets[id.x]; i < massOffsets[id.x + 1]; ++i)
    {
        const uint4 link = massContributions[i];
        value += discreteTendonTerms[tendonBase + link.x].x *
                 tendonJacobian[jacobianBase + link.y] * tendonJacobian[jacobianBase + link.z];
    }
    massMatrix[id.y * count + id.x] += value;
}
