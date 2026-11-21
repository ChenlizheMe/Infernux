// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#include "model.slang"

[[vk::binding(0, 0)]] StructuredBuffer<ModelDimensions> model;
[[vk::binding(1, 0)]] StructuredBuffer<float> qpos;
[[vk::binding(2, 0)]] StructuredBuffer<float> qvel;
[[vk::binding(3, 0)]] StructuredBuffer<float> time;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> snapshot;

[shader("compute")]
[numthreads(64, 1, 1)]
void snapshot_generalized_state(uint3 id : SV_DispatchThreadID)
{
    const uint2 state = model[0].state.xy;
    uint worlds, stride;
    time.GetDimensions(worlds, stride);
    const uint positionSize = worlds * state.x;
    const uint velocitySize = worlds * state.y;

    if (id.x < state.x)
    {
        const uint index = id.y * state.x + id.x;
        snapshot[index] = qpos[index];
    }
    if (id.x < state.y)
    {
        const uint index = id.y * state.y + id.x;
        snapshot[positionSize + index] = qvel[index];
    }
    if (id.x == 0)
        snapshot[positionSize + velocitySize + id.y] = time[id.y];
}
