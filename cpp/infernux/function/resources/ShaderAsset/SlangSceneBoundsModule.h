#pragma once

#include "SlangCompiler.h"

namespace infernux
{

/// Fit native directional shadows from the same accepted world bounds used
/// by geometry. No host observation of a GPU pose is required.
inline const ShaderSourceModule &GetSlangSceneBoundsModule()
{
    static const ShaderSourceModule module{"inx_scene_bounds", "<infernux/renderer/scene_bounds>", R"slang(
import inx_geometry;

struct ShadowFitParameters {
    float4 ray_direction;
    uint4 counts;
};
[[vk::binding(0, 0)]] StructuredBuffer<InxWorldBounds> bounds;
[[vk::binding(1, 0)]] RWStructuredBuffer<InxWorldBounds> tiles;
[[vk::binding(2, 0)]] RWStructuredBuffer<InxModelMatrix> shadow_camera;
[[vk::binding(3, 0)]] ConstantBuffer<ShadowFitParameters> parameters;

groupshared float3 tile_minimum[64];
groupshared float3 tile_maximum[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void reduce_bounds(uint3 index : SV_DispatchThreadID, uint3 lane : SV_GroupThreadID, uint3 group : SV_GroupID) {
    tile_minimum[lane.x] = float3(3.402823466e+38);
    tile_maximum[lane.x] = float3(-3.402823466e+38);
    if (index.x < parameters.counts.x) {
        tile_minimum[lane.x] = bounds[index.x].minimum.xyz;
        tile_maximum[lane.x] = bounds[index.x].maximum.xyz;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 32u; stride != 0u; stride >>= 1u) {
        if (lane.x < stride) {
            tile_minimum[lane.x] = min(tile_minimum[lane.x], tile_minimum[lane.x + stride]);
            tile_maximum[lane.x] = max(tile_maximum[lane.x], tile_maximum[lane.x + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane.x == 0u) {
        InxWorldBounds value;
        value.minimum = float4(tile_minimum[0], 0.0);
        value.maximum = float4(tile_maximum[0], 0.0);
        tiles[group.x] = value;
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void fit_shadow_camera() {
    float3 minimum = tiles[0].minimum.xyz;
    float3 maximum = tiles[0].maximum.xyz;
    for (uint i = 1u; i < parameters.counts.y; ++i) {
        minimum = min(minimum, tiles[i].minimum.xyz);
        maximum = max(maximum, tiles[i].maximum.xyz);
    }
    float3 center = (minimum + maximum) * 0.5;
    float radius = max(2.0, length(maximum - minimum) * 0.6);
    float3 forward = normalize(parameters.ray_direction.xyz);
    float3 up = abs(forward.y) > 0.95 ? float3(1, 0, 0) : float3(0, 1, 0);
    float3 right = normalize(cross(forward, up));
    up = cross(right, forward);
    float3 eye = center - forward * (radius * 2.0);
    float4x4 view = float4x4(
        right.x, right.y, right.z, -dot(right, eye),
        up.x, up.y, up.z, -dot(up, eye),
        -forward.x, -forward.y, -forward.z, dot(forward, eye),
        0, 0, 0, 1);
    float near_plane = radius * 4.5;
    float far_plane = 0.1;
    float4x4 projection = float4x4(
        1.0 / radius, 0, 0, 0,
        0, 1.0 / radius, 0, 0,
        0, 0, 1.0 / (near_plane - far_plane), near_plane / (near_plane - far_plane),
        0, 0, 0, 1);
    InxModelMatrix result;
    result.value = mul(projection, view);
    shadow_camera[0] = result;
}
)slang"};
    return module;
}

} // namespace infernux
