// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// MuJoCo Warp _src/constraint.py tendon equality, limit and friction rows.

#include "model.slang"
#include "constraint_data.slang"
#include "constraint_parameters.slang"
#include "constraint_stream.slang"
#include "tendon_data.slang"

struct TendonConstraintModel
{
    uint4 source; // kind, first tendon, second tendon or NoIndex, source id
    ConstraintParameters parameters;
    float4 data; // equality coefficients or limit range and margin
    float4 extra; // fifth equality coefficient, inverse weight, friction loss
};

[[vk::binding(1, 0)]] StructuredBuffer<uint4> tendonConstraintDimensions; // candidate count, tendon count, Jacobian count, equality count
[[vk::binding(2, 0)]] StructuredBuffer<TendonConstraintModel> tendonConstraints;
[[vk::binding(3, 0)]] StructuredBuffer<TendonModel> tendons;
[[vk::binding(4, 0)]] StructuredBuffer<uint> equalityActive;
[[vk::binding(6, 0)]] StructuredBuffer<TendonState> tendonState;
[[vk::binding(9, 0)]] StructuredBuffer<uint4> tendonColumns;
[[vk::binding(10, 0)]] StructuredBuffer<float> tendonJacobian;

uint mergedJacobianWidth(uint2 first, uint2 second)
{
    uint count = 0;
    while (first.y != 0 && second.y != 0)
    {
        const uint a = tendonColumns[first.x].x;
        const uint b = tendonColumns[second.x].x;
        if (a <= b)
        {
            ++first.x;
            --first.y;
        }
        if (b <= a)
        {
            ++second.x;
            --second.y;
        }
        ++count;
    }
    return count + first.y + second.y;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void emit_tendon_constraint_rows(uint3 id : SV_DispatchThreadID)
{
    const uint4 dimensions = tendonConstraintDimensions[0];
    if (id.x >= dimensions.x)
        return;
    const TendonConstraintModel constraint = tendonConstraints[id.x];
    const uint4 source = constraint.source;
    if (source.x == EqualityRow && equalityActive[id.y * dimensions.w + source.w] == 0)
        return;

    const uint tendonBase = id.y * dimensions.y;
    const TendonModel firstModel = tendons[source.y];
    const TendonState firstState = tendonState[tendonBase + source.y];
    uint2 first = firstModel.layout.zw;
    uint2 second = uint2(0);
    float velocity = firstState.velocity;
    float distance = 0.0;
    float margin = 0.0;
    float firstScale = 1.0;
    float secondScale = 0.0;
    if (source.x == TendonLimitRow)
    {
        const float lower = firstState.length - constraint.data.x;
        const float upper = constraint.data.y - firstState.length;
        margin = constraint.data.z;
        distance = min(lower, upper) - margin;
        if (distance >= 0.0)
            return;
        firstScale = lower < upper ? 1.0 : -1.0;
        velocity *= firstScale;
    }
    else if (source.x == EqualityRow)
    {
        float target = constraint.data.x;
        if (source.z != NoIndex)
        {
            const TendonModel other = tendons[source.z];
            const TendonState value = tendonState[tendonBase + source.z];
            const float x = value.length - other.springArmature.w;
            const float2 coupling = constraintCoupling(constraint.data, constraint.extra.x, x);
            target = coupling.x;
            secondScale = -coupling.y;
            if (secondScale != 0.0)
                second = other.layout.zw;
            velocity += value.velocity * secondScale;
        }
        distance = firstState.length - firstModel.springArmature.w - target;
    }

    const uint width = mergedJacobianWidth(first, second);
    const uint2 block = reserveConstraintBlock(id.y, 1, width);
    if (block.x == NoIndex)
        return;
    const uint4 limits = settings[0];
    const uint jBase = id.y * dimensions.z;
    const uint columnBase = id.y * limits.w + block.y;
    uint activeWidth = 0;
    while (first.y != 0 || second.y != 0)
    {
        const uint a = first.y != 0 ? tendonColumns[first.x].x : NoIndex;
        const uint b = second.y != 0 ? tendonColumns[second.x].x : NoIndex;
        float value = 0.0;
        if (a <= b)
        {
            value = firstScale * tendonJacobian[jBase + first.x];
            ++first.x;
            --first.y;
        }
        if (b <= a)
        {
            value += secondScale * tendonJacobian[jBase + second.x];
            ++second.x;
            --second.y;
        }
        if (source.x != EqualityRow || value != 0.0)
            columns[columnBase + activeWidth++] = {min(a, b), value};
    }
    const uint flags = settings[1].w;
    const float timestep = asfloat(settings[2 + id.y].x);
    const float3 kbi = impedanceParameters(constraint.parameters, distance, timestep,
                                           (flags & (UnclampedReference | DiscreteConstraints)) != 0);
    rows[id.y * limits.z + block.x] = makeConstraintRow(uint4(block.y, activeWidth, source.w, 0), source.x,
        kbi, distance, margin, velocity, constraint.extra.y, constraint.extra.z, 0.0,
        timestep, flags, constraint.parameters.reference.x);
}
