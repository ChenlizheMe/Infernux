// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#include "model.slang"
#include "linear_dispatch.slang"

[[vk::binding(0, 0)]] StructuredBuffer<ModelDimensions> model;
[[vk::binding(1, 0)]] StructuredBuffer<float> qpos;
[[vk::binding(2, 0)]] StructuredBuffer<float> qvel;
[[vk::binding(3, 0)]] StructuredBuffer<float> time;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> snapshot;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> scratchWords;
[[vk::binding(6, 0)]] StructuredBuffer<uint> stepStatus;

[shader("compute")]
[numthreads(64, 1, 1)]
void initialize_step_scratch(uint3 id : SV_DispatchThreadID)
{
    uint count, stride;
    scratchWords.GetDimensions(count, stride);
    const uint index = linearWorkIndex<64>(id);
    if (index < count)
        scratchWords[index] = 0;
}

void writeSnapshot<let IncludeStatus : bool>(uint3 id)
{
    const uint2 state = model[0].state.xy;
    uint worlds, stride;
    time.GetDimensions(worlds, stride);
    const uint positionSize = worlds * state.x;
    const uint velocitySize = worlds * state.y;

    if (id.x < state.x)
    {
        const uint index = id.y * state.x + id.x;
        snapshot[index] = asuint(qpos[index]);
    }
    if (id.x < state.y)
    {
        const uint index = id.y * state.y + id.x;
        snapshot[positionSize + index] = asuint(qvel[index]);
    }
    if (id.x == 0)
    {
        snapshot[positionSize + velocitySize + id.y] = asuint(time[id.y]);
        if (IncludeStatus)
            snapshot[positionSize + velocitySize + worlds + id.y] = stepStatus[id.y];
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void snapshot_generalized_state(uint3 id : SV_DispatchThreadID)
{
    writeSnapshot<false>(id);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void snapshot_contact_state(uint3 id : SV_DispatchThreadID)
{
    writeSnapshot<true>(id);
}
