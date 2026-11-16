[[vk::binding(0, 0)]] RWStructuredBuffer<uint> values;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 thread : SV_DispatchThreadID)
{
    uint3 groups = WorkgroupCount();
    uint index = thread.x + groups.x * (thread.y + groups.y * thread.z);
    values[index] = index * 3u + 7u;
}
