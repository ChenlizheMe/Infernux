[[vk::binding(0, 0)]] Sampler2D<float4> source_texture;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> pixels;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 thread : SV_DispatchThreadID)
{
    uint2 extent;
    source_texture.GetDimensions(extent.x, extent.y);
    if (any(thread.xy >= extent))
        return;
    float2 uv = (float2(thread.xy) + 0.5) / float2(extent);
    pixels[thread.y * extent.x + thread.x] = source_texture.SampleLevel(uv, 0.0);
}
