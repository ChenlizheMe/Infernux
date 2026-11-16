[[vk::binding(0, 0)]] StructuredBuffer<uint> source_words;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> destination_words;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 thread : SV_DispatchThreadID)
{
    destination_words[thread.x] = source_words[thread.x] * 7u + 13u;
}
