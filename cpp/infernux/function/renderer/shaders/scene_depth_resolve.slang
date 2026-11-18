struct ResolveConstants {
    uint width;
    uint height;
    uint sample_count;
    uint reserved;
};
ConstantBuffer<ResolveConstants> parameters;
DepthTexture2DMS source_depth;
[format("r32f")] WTexture2D<float> resolved_depth;

[shader("compute")]
[numthreads(8, 8, 1)]
void resolve_depth(uint3 id : SV_DispatchThreadID)
{
    uint2 coordinate = id.xy;
    if (coordinate.x >= parameters.width || coordinate.y >= parameters.height) return;
    int2 pixel = int2(coordinate);
    float depth = source_depth.Load(pixel, 0);
    for (uint sample_index = 1; sample_index < parameters.sample_count; ++sample_index)
        // Reversed-Z: retain the nearest covered surface for soft-particle edges.
        depth = max(depth, source_depth.Load(pixel, int(sample_index)));
    resolved_depth.Store(pixel, depth);
}
