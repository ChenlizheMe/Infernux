// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/constraint.py scalar joint and DOF rows.

#include "model.slang"
#include "constraint_data.slang"
#include "constraint_parameters.slang"
#include "constraint_stream.slang"

struct JointConstraintModel
{
    uint4 source; // row kind, first joint or friction DOF, second joint or NoIndex, source id
    ConstraintParameters parameters;
    float4 data; // first four equality coefficients or (limit lower, upper, margin)
    float4 extra; // fifth equality coefficient, inverse weight, friction loss
};

[[vk::binding(1, 0)]] StructuredBuffer<uint4> jointConstraintDimensions; // candidates, nq, per-world equality count
[[vk::binding(2, 0)]] StructuredBuffer<JointConstraintModel> jointConstraints;
[[vk::binding(3, 0)]] StructuredBuffer<JointModel> joints;
[[vk::binding(4, 0)]] StructuredBuffer<uint> equalityActive;
[[vk::binding(9, 0)]] StructuredBuffer<float> qpos;
[[vk::binding(10, 0)]] StructuredBuffer<float> qvel;

[shader("compute")]
[numthreads(64, 1, 1)]
void emit_joint_constraint_rows(uint3 id : SV_DispatchThreadID)
{
    const uint4 counts = jointConstraintDimensions[0];
    if (id.x >= counts.x)
        return;
    const JointConstraintModel constraint = jointConstraints[id.x];
    const uint4 source = constraint.source;
    const uint positionBase = id.y * counts.y;
    uint3 dof = uint3(source.y, 0, 0);
    uint width = 1;
    float3 jacobian = float3(1, 0, 0);
    float distance = 0.0;
    float margin = 0.0;
    if (source.x == EqualityRow)
    {
        if (equalityActive[id.y * counts.z + source.w] == 0)
            return;
        const JointModel first = joints[source.y];
        dof.x = first.indices.z;
        float target = constraint.data.x;
        if (source.z != NoIndex)
        {
            const JointModel second = joints[source.z];
            const float x = qpos[positionBase + second.indices.y] - second.axis.w;
            const float4 coefficients = constraint.data;
            const float fourth = constraint.extra.x;
            target += x * (coefficients.y + x * (coefficients.z + x * (coefficients.w + x * fourth)));
            jacobian.y = -(coefficients.y + x * (2.0 * coefficients.z + x * (3.0 * coefficients.w + x * 4.0 * fourth)));
            dof.y = second.indices.z;
            width = 2;
        }
        distance = qpos[positionBase + first.indices.y] - first.axis.w - target;
    }
    else if (source.x == LimitRow)
    {
        const uint4 joint = joints[source.y].indices;
        const uint q = positionBase + joint.y;
        dof.x = joint.z;
        margin = constraint.data.z;
        if (joint.x == JointBall)
        {
            const float4 orientation = normalize(float4(qpos[q], qpos[q + 1], qpos[q + 2], qpos[q + 3]));
            const float3 rotation = rotationVector(orientation);
            const float angle = length(rotation);
            distance = max(constraint.data.x, constraint.data.y) - angle - margin;
            if (distance >= 0.0)
                return;
            jacobian = -rotation / max(angle, ConstraintMinValue);
            dof = joint.z + uint3(0, 1, 2);
            width = 3;
        }
        else
        {
            const float lower = qpos[q] - constraint.data.x;
            const float upper = constraint.data.y - qpos[q];
            distance = min(lower, upper) - margin;
            if (distance >= 0.0)
                return;
            jacobian.x = lower < upper ? 1.0 : -1.0;
        }
    }

    const uint2 block = reserveConstraintBlock(id.y, 1, width);
    if (block.x == NoIndex)
        return;
    const uint4 limits = settings[0];
    const uint4 config = settings[1];
    const uint columnBase = id.y * limits.w + block.y;
    const uint dofBase = id.y * config.y;
    float velocity = 0.0;
    for (uint i = 0; i < width; ++i)
    {
        columns[columnBase + i] = {dof[i], jacobian[i]};
        velocity += jacobian[i] * qvel[dofBase + dof[i]];
    }
    const float3 kbi = impedanceParameters(constraint.parameters, distance, asfloat(settings[2 + id.y].x),
                                           (config.w & UnclampedReference) != 0);
    rows[id.y * limits.z + block.x] = makeConstraintRow(uint4(block.y, width, source.w, 0), source.x, kbi,
        distance, margin, velocity, constraint.extra.y, constraint.extra.z, 0.0);
}
