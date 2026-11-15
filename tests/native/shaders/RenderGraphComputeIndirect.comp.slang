struct DrawArguments { uint vertexCount; uint instanceCount; uint firstVertex; uint firstInstance; };
[[vk::binding(0, 0)]] RWStructuredBuffer<DrawArguments> args;
[shader("compute")]
[numthreads(1, 1, 1)]
void main() { args[0] = {3u, 1u, 0u, 0u}; }
