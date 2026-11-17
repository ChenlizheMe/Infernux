import inx_specialization;

extern static const int probe_bias;

[vk::binding(0, 0)] StructuredBuffer<int> values;
[vk::binding(1, 0)] RWStructuredBuffer<int> results;

[shader("compute")]
[numthreads(4, 1, 1)]
void platform_probe(uint3 thread : SV_DispatchThreadID)
{
    uint count, stride;
    values.GetDimensions(count, stride);
    if (thread.x < count)
        results[thread.x] = values[thread.x] * 3 + probe_bias;
}
