// Fixed-depth directional outgoing indirect radiance, not irradiance or Le.
public int3 ray_tail_cell(float3 position, float cellSize)
{
    return int3(floor(position / cellSize));
}

public uint ray_tail_direction(float3 w, uint grid)
{
    w /= abs(w.x) + abs(w.y) + abs(w.z);
    float2 p = w.xy;
    if (w.z < 0.0)
        p = (1.0 - abs(p.yx)) * float2(p.x >= 0.0 ? 1.0 : -1.0, p.y >= 0.0 ? 1.0 : -1.0);
    uint2 cell = min(uint2((p * 0.5 + 0.5) * float(grid)), uint2(grid - 1));
    return cell.x + grid * cell.y;
}

public uint ray_tail_hash(int3 cell, uint lod, uint surface, uint primitive, uint direction)
{
    uint h = asuint(cell.x) * 0x8da6b343u ^ asuint(cell.y) * 0xd8163841u ^ asuint(cell.z) * 0xcb1ab31fu;
    h ^= surface * 0x9e3779b9u ^ primitive * 0x85ebca6bu ^ direction * 0xc2b2ae35u ^ lod;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu;
    return h ^ (h >> 16);
}

public bool ray_tail_same(uint4 key, uint4 identity, int3 cell, uint surface, uint primitive, uint epoch)
{
    return all(key.xyz == asuint(cell)) && key.w == 0 && identity.x == surface &&
           identity.y == primitive && identity.z == epoch && (identity.w & 1u) != 0;
}

public bool ray_tail_representative(float3 anchor, float3 wo, float3 ng,
                                    float3 requestedAnchor, float3 requestedWo, float3 requestedNg,
                                    float cellSize, float cone)
{
    float3 delta = anchor - requestedAnchor;
    return dot(delta, delta) <= 0.0625 * cellSize * cellSize &&
           dot(ng, requestedNg) >= 0.9982 && dot(wo, requestedWo) >= 1.0 - 0.5 * cone * cone;
}
