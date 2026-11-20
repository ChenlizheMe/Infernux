// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

#ifndef JPH_GPU_KEY_SORT_SLANG
#define JPH_GPU_KEY_SORT_SLANG

static const uint JphSortLanes = 64u;
groupshared uint4 jph_sort_keys[JphSortLanes];

bool jph_key_less(uint4 a, uint4 b)
{
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    if (a.z != b.z) return a.z < b.z;
    return a.w < b.w;
}

uint4 jph_sort_tile(uint lane, uint4 key)
{
    jph_sort_keys[lane] = key;
    GroupMemoryBarrierWithGroupSync();
    for (uint size = 2u; size <= JphSortLanes; size *= 2u)
    {
        for (uint stride = size / 2u; stride != 0u; stride /= 2u)
        {
            uint other = lane ^ stride;
            if (lane < other)
            {
                uint4 a = jph_sort_keys[lane], b = jph_sort_keys[other];
                bool ascending = (lane & size) == 0u;
                if (ascending ? jph_key_less(b, a) : jph_key_less(a, b))
                {
                    jph_sort_keys[lane] = b;
                    jph_sort_keys[other] = a;
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
    return jph_sort_keys[lane];
}

#endif
