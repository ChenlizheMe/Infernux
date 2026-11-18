struct CanonicalLightData {
    float4 position_range;
    float4 direction_spot;
    float4 color_intensity;
    float4 attenuation;
    float4 area_right_width;
    float4 area_up_height;
    uint4 metadata;
    uint4 identity_shadow;
};

ByteAddressBuffer canonical_lights;
RWStructuredBuffer<uint4> tile_headers;
RWStructuredBuffer<Atomic<uint>> tile_light_masks;

struct ForwardPlusGridConstants {
    // The canonical host camera stores column vectors. Keep that data
    // contract explicit and independent of target matrix storage rules.
    float4 view_projection_columns[4];
    float4 viewport_projection_scale;
    uint4 grid_lights;
    uint4 domain_mask_words;
};
ConstantBuffer<ForwardPlusGridConstants> pc;

// Pull the tile's clip inequalities back to world-space planes. This avoids
// dividing a sphere's center by w and guessing a symmetric projected radius,
// which can exclude lit pixels for off-axis or shifted projections. All light
// lanes share the six planes and their normal lengths, computed once per tile.
groupshared float4 tile_planes[6];
groupshared float tile_plane_lengths[6];

void prepare_tile_plane(uint index, uint2 tile) {
    float4x4 rows = transpose(float4x4(pc.view_projection_columns[0], pc.view_projection_columns[1],
                                      pc.view_projection_columns[2], pc.view_projection_columns[3]));
    float2 viewport = pc.viewport_projection_scale.xy;
    float2 pixel_min = float2(tile) * float(pc.grid_lights.w);
    float2 pixel_max = min(pixel_min + float(pc.grid_lights.w), viewport);
    float2 ndc_min = 2.0 * pixel_min / viewport - 1.0;
    float2 ndc_max = 2.0 * pixel_max / viewport - 1.0;
    float4 plane;
    if (index == 0u) plane = rows[0] - ndc_min.x * rows[3];
    else if (index == 1u) plane = ndc_max.x * rows[3] - rows[0];
    else if (index == 2u) plane = rows[1] - ndc_min.y * rows[3];
    else if (index == 3u) plane = ndc_max.y * rows[3] - rows[1];
    // Vulkan clip depth is [0,w], also for reversed or infinite-far matrices.
    else if (index == 4u) plane = rows[2];
    else plane = rows[3] - rows[2];
    tile_planes[index] = plane;
    // Do not normalize: an infinite far plane can have a zero normal.
    tile_plane_lengths[index] = length(plane.xyz);
}

bool overlaps_tile(CanonicalLightData light) {
    float4 center = float4(light.position_range.xyz, 1.0);
    float radius = max(light.position_range.w, 0.0);
    if (light.metadata.x == 3u) {
        radius += 0.5 * length(float2(light.area_right_width.w, light.area_up_height.w));
    }
    for (uint index = 0u; index < 6u; ++index) {
        float4 plane = tile_planes[index];
        float support = radius * tile_plane_lengths[index];
        // Outward rounding for float dot products keeps a tangent sphere in
        // the list; this is relative to the terms, not a world-unit padding.
        float roundoff = 0.000001 * (dot(abs(plane), abs(center)) + support);
        if (dot(plane, center) + support < -roundoff) return false;
    }
    return true;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void build_light_grid(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    uint2 tile = group.xy;
    uint tile_index = tile.y * pc.grid_lights.x + tile.x;
    uint offset = tile_index * pc.domain_mask_words.y;
    if (lane < 6u) prepare_tile_plane(lane, tile);
    for (uint word = lane; word < pc.domain_mask_words.y; word += 64u) {
        tile_light_masks[offset + word].store(0u);
    }
    // Both shared planes and storage mask words must be visible before the
    // other lanes test lights and atomically accumulate into the cleared mask.
    AllMemoryBarrierWithGroupSync();

    uint4 counts_generation = canonical_lights.Load<uint4>(0);
    uint directional_count = counts_generation.x;
    uint available_local_count = min(counts_generation.y, pc.grid_lights.z);
    for (uint local_index = lane; local_index < available_local_count; local_index += 64u) {
        CanonicalLightData light = canonical_lights.Load<CanonicalLightData>(
            16u + (directional_count + local_index) * sizeof(CanonicalLightData));
        if ((light.metadata.w & pc.domain_mask_words.x) == 0u || !overlaps_tile(light)) continue;
        tile_light_masks[offset + (local_index >> 5u)].or(1u << (local_index & 31u));
    }
    AllMemoryBarrierWithGroupSync();
    if (lane == 0u) {
        if (tile_index == 0u) {
            tile_headers[0] = uint4(pc.grid_lights.xy, pc.grid_lights.w, pc.domain_mask_words.x);
        }
        tile_headers[tile_index + 1u] =
            uint4(offset, pc.domain_mask_words.y, available_local_count, pc.domain_mask_words.x);
    }
}
