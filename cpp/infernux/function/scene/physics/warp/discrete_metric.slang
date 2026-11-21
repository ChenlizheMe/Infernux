// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

[[vk::binding(0, 0)]] StructuredBuffer<uint4> metricLayout; // original packed count, effective packed count
[[vk::binding(1, 0)]] StructuredBuffer<float> jointMetric;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> metric;

[shader("compute")]
[numthreads(64, 1, 1)]
void initialize_discrete_metric(uint3 id : SV_DispatchThreadID)
{
    const uint2 size = metricLayout[0].xy;
    if (id.x >= size.y)
        return;
    float value = 0.0;
    if (id.x < size.x)
        value = jointMetric[id.y * size.x + id.x];
    metric[id.y * size.y + id.x] = value;
}
