// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/forward.py smooth force assembly.

#include "model.slang"

[[vk::binding(0, 0)]] StructuredBuffer<ModelDimensions> model;
[[vk::binding(1, 0)]] StructuredBuffer<float> qfrcPassive;
[[vk::binding(2, 0)]] StructuredBuffer<float> qfrcBias;
[[vk::binding(3, 0)]] StructuredBuffer<float> qfrcActuator;
[[vk::binding(4, 0)]] StructuredBuffer<float> qfrcApplied;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> qfrcSmooth;

[shader("compute")]
[numthreads(64, 1, 1)]
void assemble_smooth_force(uint3 id : SV_DispatchThreadID)
{
    const uint nv = model[0].state.y;
    if (id.x >= nv)
        return;
    const uint index = id.y * nv + id.x;
    qfrcSmooth[index] = qfrcPassive[index] - qfrcBias[index] + qfrcActuator[index] + qfrcApplied[index];
}
