[[vk::binding(0, 0)]] RWStructuredBuffer<uint> values;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 thread : SV_DispatchThreadID)
{
    uint3 groups = WorkgroupCount();
    uint index = thread.x + groups.x * (thread.y + groups.y * thread.z);
    values[index] = index * 3u + 7u;
}

[[vk::binding(1, 0)]] StructuredBuffer<uint> request;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> count_output;
[[vk::binding(3, 0)]] StructuredBuffer<uint> count_input;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> payload_output;
[[vk::binding(5, 0)]] StructuredBuffer<uint> payload_input;
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> result;

static const uint capacity = 67u;
static const uint padding = 4u;

[shader("compute")]
[numthreads(1, 1, 1)]
void make_count()
{
    const uint selector = request[0];
    count_output[1] = selector == 0u ? 0u : selector * 2u - 1u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void reset_result(uint3 thread : SV_DispatchThreadID)
{
    if (thread.x < capacity)
        result[thread.x] = 999999u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void make_payload(uint3 thread : SV_DispatchThreadID)
{
    if (thread.x >= min(count_input[1], capacity)) return;
    payload_output[padding + thread.x] = thread.x * 3u + 7u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void read_payload(uint3 thread : SV_DispatchThreadID)
{
    if (thread.x >= min(count_input[1], capacity)) return;
    result[thread.x] = payload_input[padding + thread.x] * 2u;
}
