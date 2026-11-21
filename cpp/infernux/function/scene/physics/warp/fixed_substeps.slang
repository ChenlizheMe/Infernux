// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#include "model.slang"

[[vk::binding(0, 0)]] StructuredBuffer<ModelDimensions> model;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> qpos;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> qvel;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> warmstart;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> previous;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> fixedSteps; // Accepted ordinals, then the current substep.
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> status;

uint cursor()
{
    uint size, stride;
    fixedSteps.GetDimensions(size, stride);
    return size - 1u;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void begin_substeps()
{
    fixedSteps[cursor()] = 0u;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void advance_substep()
{
    const uint row = cursor();
    fixedSteps[row] += 1u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void checkpoint_fixed_step(uint3 id : SV_DispatchThreadID)
{
    const uint2 width = model[0].state.xy;
    const uint stride = width.x + 2u * width.y;
    const uint base = id.y * stride;
    if (id.x == 0)
        status[id.y] = 0u;
    if (id.x < width.x)
        previous[base + id.x] = qpos[id.y * width.x + id.x];
    else if (id.x < stride)
    {
        const uint row = id.x - width.x;
        const uint dof = id.y * width.y;
        previous[base + id.x] = row < width.y ? qvel[dof + row] : warmstart[dof + row - width.y];
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void restore_rejected_fixed_step(uint3 id : SV_DispatchThreadID)
{
    if (status[id.y] == 0u)
        return;
    const uint2 width = model[0].state.xy;
    const uint stride = width.x + 2u * width.y;
    const uint base = id.y * stride;
    if (id.x < width.x)
        qpos[id.y * width.x + id.x] = previous[base + id.x];
    else if (id.x < stride)
    {
        const uint row = id.x - width.x;
        const uint dof = id.y * width.y;
        if (row < width.y)
            qvel[dof + row] = previous[base + id.x];
        else
            warmstart[dof + row - width.y] = previous[base + id.x];
    }
}
