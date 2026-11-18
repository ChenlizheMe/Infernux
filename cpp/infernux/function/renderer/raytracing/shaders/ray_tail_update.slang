import inx_ray_tail_cache;

ByteAddressBuffer cache_read;
RWByteAddressBuffer cache_write;
RWByteAddressBuffer cache_requests;
ByteAddressBuffer cache_control;
Texture2D<uint4> request_0;
Texture2D<uint4> request_1;
Texture2D<float4> tail_samples;

[shader("compute")]
[numthreads(64, 1, 1)]
void prepare_main(uint3 tid : SV_DispatchThreadID)
{
    uint4 c = cache_control.Load4(0);
    uint4 work = cache_control.Load4(16);
    if (work.z == 0) {
        uint index = work.x + tid.x;
        if (tid.x < work.y) {
            uint base = 32 + index * 128;
            bool valid = cache_read.Load(0) == c.x && cache_read.Load(4) == c.z;
            for (uint b = 0; b < 8; ++b)
                cache_write.Store4(base + b * 16, valid ? cache_read.Load4(base + b * 16) : uint4(0));
        }
        if (tid.x == 0) {
            cache_write.Store4(0, uint4(c.x, c.z, c.w, 1));
            uint4 metric = cache_control.Load4(32), world = cache_control.Load4(48);
            cache_write.Store4(16, uint4(metric.xy, world.z, 0));
        }
        return;
    }
    // One bounded allocator establishes unique slots before parallel merge.
    if (tid.x != 0) return;
    for (uint block = 0; block < 4; ++block)
        cache_requests.Store4(block * 16, cache_control.Load4(block * 16));
    uint4 metric = cache_control.Load4(32);
    uint width, height; request_0.GetDimensions(width, height);
    for (uint i = 0; i < c.y; ++i) {
        uint offset = 64 + i * 64;
        cache_requests.Store4(offset, uint4(0));
        cache_requests.Store4(offset + 16, uint4(0));
        cache_requests.Store4(offset + 32, uint4(0));
        cache_requests.Store4(offset + 48, uint4(0));
        if (i >= width || height < 2) continue;
        uint4 r0 = request_0.Load(int3(i, 0, 0)), r1 = request_1.Load(int3(i, 0, 0));
        uint4 r2 = request_0.Load(int3(i, 1, 0)), r3 = request_1.Load(int3(i, 1, 0));
        if ((r1.w & 1u) == 0) continue;
        int3 cell = ray_tail_cell(asfloat(r1.xyz), asfloat(metric.x));
        uint direction = ray_tail_direction(asfloat(r2.xyz), metric.y);
        uint hash = ray_tail_hash(cell, 0, r2.w, r3.w, direction);
        uint selected = 0xffffffffu, empty = 0xffffffffu, oldest = 0xffffffffu;
        uint oldestAge = 0;
        bool reuse = false, duplicateKey = false;
        for (uint probe = 0; probe < 4; ++probe) {
            uint slot = (hash + probe) % c.x;
            uint base = 32 + slot * 128;
            uint4 key = cache_write.Load4(base), identity = cache_write.Load4(base + 16);
            bool valid = identity.z == c.z && (identity.w & 1u) != 0;
            bool reserved = identity.z == c.z && (identity.w & 2u) != 0;
            bool claimed = false;
            for (uint prior = 0; prior < i; ++prior) {
                uint flags = cache_requests.Load(64 + prior * 64 + 28);
                claimed |= (flags & 1u) != 0 && (flags >> 2) == slot;
            }
            bool sameKey = all(key.xyz == asuint(cell)) && key.w == 0 && identity.x == r2.w &&
                           identity.y == r3.w && identity.z == c.z && cache_write.Load(base + 60) == direction;
            if (sameKey && (valid || reserved)) {
                if (claimed || reserved) duplicateKey = true;
                else {
                    selected = slot;
                    uint2 exact = cache_write.Load2(base + 112);
                    float cone = unpackHalf2x16ToFloat(r1.w).y;
                    reuse = all(exact == r0.xy) && ray_tail_representative(
                        asfloat(cache_write.Load3(base + 32)), asfloat(cache_write.Load3(base + 48)),
                        asfloat(cache_write.Load3(base + 96)), asfloat(r1.xyz), asfloat(r2.xyz),
                        asfloat(r3.xyz), asfloat(metric.x), cone);
                }
                break;
            }
            if (claimed || reserved) continue;
            if (!valid && empty == 0xffffffffu) empty = slot;
            uint age = c.w - cache_write.Load(base + 124);
            if (valid && (oldest == 0xffffffffu || age > oldestAge)) {
                oldest = slot; oldestAge = age;
            }
        }
        if (duplicateKey) continue;
        if (selected == 0xffffffffu) selected = empty != 0xffffffffu ? empty : oldest;
        if (selected == 0xffffffffu) continue;
        uint base = 32 + selected * 128;
        if (reuse) {
            uint4 identity = cache_write.Load4(base + 16);
            uint4 anchor = cache_write.Load4(base + 32), outgoing = cache_write.Load4(base + 48);
            uint4 bary = cache_write.Load4(base + 64), normal = cache_write.Load4(base + 96);
            uint4 exact = cache_write.Load4(base + 112);
            r0 = uint4(exact.xy, bary.xy);
            r1 = uint4(anchor.xyz, 1u | (selected << 2));
            r2 = uint4(outgoing.xyz, identity.x);
            r3 = uint4(normal.xyz, identity.y);
        } else {
            cache_write.Store4(base, uint4(asuint(cell), 0));
            cache_write.Store4(base + 16, uint4(r2.w, r3.w, c.z, 2));
            cache_write.Store4(base + 32, uint4(r1.xyz, 0));
            cache_write.Store4(base + 48, uint4(r2.xyz, direction));
            cache_write.Store4(base + 64, uint4(r0.zw, 0, 0));
            cache_write.Store4(base + 80, uint4(0));
            cache_write.Store4(base + 96, uint4(r3.xyz, 0));
            cache_write.Store4(base + 112, uint4(r0.xy, 0, 0));
            r1.w = 3u | (selected << 2);
        }
        cache_requests.Store4(offset, r0); cache_requests.Store4(offset + 16, r1);
        cache_requests.Store4(offset + 32, r2); cache_requests.Store4(offset + 48, r3);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void merge_main(uint3 tid : SV_DispatchThreadID)
{
    uint4 c = cache_control.Load4(0), work = cache_control.Load4(16);
    if (work.z == 0 || tid.x >= c.y) return;
    if (tid.x == 0) {
        cache_write.Store4(0, uint4(c.x, c.z, c.w, 1));
        uint4 metric = cache_control.Load4(32), world = cache_control.Load4(48);
        cache_write.Store4(16, uint4(metric.xy, world.z, 0));
    }
    uint requestFlags = cache_requests.Load(64 + tid.x * 64 + 28);
    if ((requestFlags & 1u) == 0) return;
    uint base = 32 + (requestFlags >> 2) * 128;
    uint width, height; tail_samples.GetDimensions(width, height);
    float4 value = tail_samples.Load(int3(tid.x % width, tid.x / width, 0));
    if (value.w < 0.0) {
        if ((requestFlags & 2u) != 0) cache_write.Store(base + 28, 0);
        return;
    }
    uint n = cache_write.Load(base + 44);
    float count = float(n + 1);
    float3 oldMean = asfloat(cache_write.Load3(base + 80));
    float3 mean = oldMean + (value.xyz - oldMean) / count;
    float2 oldDistance = asfloat(cache_write.Load2(base + 72));
    float2 distance = oldDistance + (float2(value.w, value.w * value.w) - oldDistance) / count;
    float low = n == 0 ? value.w : min(asfloat(cache_write.Load(base + 92)), value.w);
    float high = n == 0 ? value.w : max(asfloat(cache_write.Load(base + 108)), value.w);
    float3 luminance = float3(0.2126, 0.7152, 0.0722);
    float delta = dot(value.xyz - oldMean, luminance);
    float m2 = asfloat(cache_write.Load(base + 120)) + delta * dot(value.xyz - mean, luminance);
    cache_write.Store(base + 28, 1); cache_write.Store(base + 44, n + 1);
    cache_write.Store2(base + 72, asuint(distance));
    cache_write.Store4(base + 80, asuint(float4(mean, low)));
    cache_write.Store(base + 108, asuint(high));
    cache_write.Store2(base + 120, uint2(asuint(m2), c.w));

}
