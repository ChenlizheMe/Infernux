// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Adapted from MuJoCo Warp _src/constraint.py, regularize_constraint.

#include "model.slang"
#include "contact.slang"
#include "constraint_data.slang"
#include "constraint_parameters.slang"
#include "constraint_stream.slang"

[[vk::binding(1, 0)]] StructuredBuffer<Contact> contacts;
[[vk::binding(2, 0)]] StructuredBuffer<DofModel> dofs;
[[vk::binding(3, 0)]] StructuredBuffer<float> massMatrix;
[[vk::binding(4, 0)]] StructuredBuffer<float> metric;
[[vk::binding(6, 0)]] RWStructuredBuffer<Atomic<uint>> coupledDofs;
[[vk::binding(9, 0)]] StructuredBuffer<ModelDimensions> model;

[shader("compute")]
[numthreads(64, 1, 1)]
void clear_discrete_coupling(uint3 id : SV_DispatchThreadID)
{
    const uint dofCount = settings[1].y;
    if (id.x < dofCount)
        coupledDofs[id.y * dofCount + id.x].store(0u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void regularize_discrete_constraints(uint3 id : SV_DispatchThreadID)
{
    const uint4 limits = settings[0];
    const uint count = rowCounts[2 * id.y].load();
    if (id.x >= count || count > limits.z || rowCounts[2 * id.y + 1].load() > limits.w)
        return;

    const uint index = id.y * limits.z + id.x;
    const bool contactRow = rows[index].kind == ContactRow;
    if (contactRow && rows[index].layout.w != 0)
        return;

    const ConstraintRow row = rows[index];
    const uint4 config = settings[1];
    const bool elliptic = (config.w & EllipticCone) != 0;
    uint blockRows = 1;
    float adhesion = 0.0;
    if (contactRow)
    {
        const Contact contact = contacts[id.y * limits.y + row.layout.z];
        blockRows = contactConstraintRows(contact, elliptic);
        adhesion = contact.tangentAdhesion.w / (elliptic ? 1 : blockRows);
    }
    const uint dofBase = id.y * config.y;
    const uint columnBase = id.y * limits.w;
    const uint massBase = id.y * model[0].layout.w;
    float2 weights = float2(0.0);
    for (uint i = 0; i < row.layout.y; ++i)
    {
        const ConstraintColumn column = columns[columnBase + row.layout.x + i];
        if (column.value == 0.0)
            continue;
        coupledDofs[dofBase + column.dof].store(1u);
        const uint2 massRow = dofs[column.dof].indices.zw;
        const uint diagonal = massBase + massRow.x + massRow.y - 1;
        weights += column.value * column.value / max(float2(ConstraintMinValue),
                                                      float2(metric[diagonal], massMatrix[diagonal]));
    }
    // Use the normal row's scale for the full contact block.
    const float scale = weights.x / max(weights.y, ConstraintMinValue);
    for (uint offset = 0; offset < blockRows; ++offset)
    {
        const uint address = index + offset;
        const ConstraintRow current = rows[address];
        if (offset != 0)
        {
            const uint start = columnBase + current.layout.x;
            for (uint i = 0; i < current.layout.y; ++i)
            {
                const ConstraintColumn column = columns[start + i];
                if (column.value != 0.0)
                    coupledDofs[dofBase + column.dof].store(1u);
            }
        }
        if (weights.y > ConstraintMinValue)
        {
            const float oldDiagonal = current.diagonal;
            const float diagonal = min(oldDiagonal / max(scale, ConstraintMinValue), 1.0 / ConstraintMinValue);
            if (adhesion != 0.0 && (!elliptic || offset == 0) &&
                oldDiagonal > ConstraintMinValue && diagonal > ConstraintMinValue)
                rows[address].state.w += (1.0 / diagonal - 1.0 / oldDiagonal) * adhesion;
            rows[address].diagonal = diagonal;
        }
    }
}
