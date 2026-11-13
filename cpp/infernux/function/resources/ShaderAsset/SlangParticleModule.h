#pragma once

#include "SlangCompiler.h"

#include <array>
#include <string_view>

namespace infernux
{

inline std::string_view GetSlangParticleSharedSource()
{
    return R"slang(#ifndef INX_ROOT_CONSTANTS
#define INX_ROOT_CONSTANTS [[vk::push_constant]]
#endif
struct ParticleRenderInstance {
    float4 position_size;
    float4 color;
    float4 rotation_custom;
    float4 scale_custom;
    uint4 ribbon_data;
    float4 custom_data;
    float4 previous_position_history;
};
struct ViewConstants {
    column_major float4x4 view_projection;
    column_major float4x4 previous_view_projection;
    float4 camera_right;
    float4 camera_up;
    float4 material_tint;
    float4 depth_reconstruct;
    float4 lighting_control;
    float4 rendering_control;
    float4 alignment_reference;
};
INX_ROOT_CONSTANTS ConstantBuffer<ViewConstants> view;
)slang";
}

struct SlangParticleGraphicsModule
{
    std::string_view renderer;
    std::string_view stage;
    ShaderSourceModule module;
};

inline const std::array<SlangParticleGraphicsModule, 9> &GetSlangParticleGraphicsModules()
{
    static const auto modules = [] {
        const std::string geometry = std::string(GetSlangParticleSharedSource()) + R"slang(
[[vk::binding(0, 0)]] StructuredBuffer<ParticleRenderInstance> instances;
[[vk::binding(1, 0)]] StructuredBuffer<uint> render_indices;
struct ParticleVertexOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_position : TEXCOORD0;
    [[vk::location(1)]] float3 normal : TEXCOORD1;
    [[vk::location(2)]] float4 tangent : TEXCOORD2;
    [[vk::location(3)]] float3 color : TEXCOORD3;
    [[vk::location(4)]] float2 uv : TEXCOORD4;
    [[vk::location(5)]] float view_depth : TEXCOORD5;
    [[vk::location(6)]] float4 line_color : TEXCOORD6;
    [[vk::location(9)]] float2 particle_local_uv : TEXCOORD9;
    [[vk::location(10)]] float2 particle_next_uv : TEXCOORD10;
    [[vk::location(11)]] float particle_blend : TEXCOORD11;
    [[vk::location(12)]] float particle_age : TEXCOORD12;
    [[vk::location(13)]] nointerpolation uint particle_id : TEXCOORD13;
    [[vk::location(14)]] float particle_alpha : TEXCOORD14;
    [[vk::location(15)]] nointerpolation uint layer_mask : TEXCOORD15;
};
struct ParticleMotionOutput {
    float4 position : SV_Position;
    [[vk::location(4)]] float2 uv : TEXCOORD4;
    [[vk::location(14)]] float alpha : TEXCOORD14;
    [[vk::location(15)]] float2 motion : TEXCOORD15;
};
#if INX_PARTICLE_MOTION_PASS
typedef ParticleMotionOutput ParticleOutput;
#else
typedef ParticleVertexOutput ParticleOutput;
#endif
ParticleOutput particle_output(ParticleRenderInstance point, float3 position, float3 normal, float4 tangent,
    float4 color, float2 uv, float2 local_uv, float2 next_uv, float blend) {
    ParticleOutput output;
    float4 clip = mul(view.view_projection, float4(position, 1.0));
    output.position = clip;
    output.position.y = -output.position.y;
    output.uv = uv;
#if INX_PARTICLE_MOTION_PASS
    float3 previous = point.previous_position_history.xyz + position - point.position_size.xyz;
    float4 previous_clip = mul(view.previous_view_projection, float4(previous, 1.0));
    output.motion = (clip.xy / max(abs(clip.w), 1e-6) -
        previous_clip.xy / max(abs(previous_clip.w), 1e-6)) * float2(0.5, -0.5);
    output.alpha = color.a;
#else
    output.world_position = position;
    output.normal = normal;
    output.tangent = tangent;
    output.color = color.rgb;
    output.view_depth = clip.w;
    output.line_color = float4(1.0);
    output.particle_local_uv = local_uv;
    output.particle_next_uv = next_uv;
    output.particle_blend = blend;
    output.particle_age = saturate(point.scale_custom.w);
    output.particle_id = point.ribbon_data.w;
    output.particle_alpha = color.a;
    output.layer_mask = asuint(view.lighting_control.w);
#endif
    return output;
}
float3 particle_axis(float3 value, float3 fallback) {
    float squared = dot(value, value);
    return squared > 1e-10 ? value * rsqrt(squared) : fallback;
}
)slang";
        const std::string billboard = geometry + R"slang(
static const float2 corners[6] = {
    float2(-1,-1), float2(-1,1), float2(1,1), float2(-1,-1), float2(1,1), float2(1,-1)
};
static const float2 uvs[6] = {
    float2(0,1), float2(0,0), float2(1,0), float2(0,1), float2(1,0), float2(1,1)
};
void billboard_basis(float3 position, float3 custom, out float3 right, out float3 up) {
    float3 camera_right = particle_axis(view.camera_right.xyz, float3(1,0,0));
    float3 camera_up = particle_axis(view.camera_up.xyz, float3(0,1,0));
    float3 camera_normal = particle_axis(cross(camera_right, camera_up), float3(0,0,1));
    int alignment = int(round(view.alignment_reference.w));
    right = camera_right;
    up = camera_up;
    if (alignment == 1) {
        float3 to_camera = particle_axis(view.alignment_reference.xyz - position, camera_normal);
        right = particle_axis(cross(camera_up, to_camera), camera_right);
        up = particle_axis(cross(to_camera, right), camera_up);
    } else if (alignment == 2 || alignment == 3) {
        up = particle_axis(alignment == 2 ? view.alignment_reference.xyz : custom, camera_up);
        float3 projected = cross(up, camera_normal);
        if (dot(projected, projected) <= 1e-10) projected = camera_right - up * dot(camera_right, up);
        right = particle_axis(projected, camera_right);
        up = particle_axis(cross(camera_normal, right), up);
    }
}
[shader("vertex")]
ParticleOutput main(uint vertex : SV_VulkanVertexID, uint instance : SV_VulkanInstanceID) {
    ParticleRenderInstance point = instances[render_indices[instance]];
    float2 corner = corners[vertex % 6u];
    float3 right, up;
    billboard_basis(point.position_size.xyz, point.custom_data.yzw, right, up);
    float cosine = cos(point.rotation_custom.x), sine = sin(point.rotation_custom.x);
    float2 rotated = float2(cosine * corner.x + sine * corner.y, -sine * corner.x + cosine * corner.y);
    float2 scale = point.scale_custom.xy;
    float3 position = point.position_size.xyz + (right * rotated.x * scale.x + up * rotated.y * scale.y) * point.position_size.w;
    float size_sign = point.position_size.w < 0.0 ? -1.0 : 1.0;
    float3 tangent = (right * (cosine * scale.x) - up * (sine * scale.y)) * size_sign;
    float magnitude = max(max(abs(tangent.x), abs(tangent.y)), abs(tangent.z));
    tangent = magnitude > 0.0 ? normalize(tangent / magnitude) :
        particle_axis((right * cosine - up * sine) * size_sign, right);
    float handedness = (scale.x < 0.0) != (scale.y < 0.0) ? -1.0 : 1.0;
    float2 grid = max(view.rendering_control.zw, float2(1.0));
    float count = grid.x * grid.y;
    float frame_position = fmod(max(point.custom_data.x, 0.0), count);
    float frame = floor(frame_position), next = fmod(frame + 1.0, count);
    float2 local_uv = uvs[vertex % 6u];
    float2 cell = float2(fmod(frame, grid.x), floor(frame / grid.x));
    float2 next_cell = float2(fmod(next, grid.x), floor(next / grid.x));
    return particle_output(point, position, particle_axis(cross(right, up), float3(0,0,1)),
        float4(tangent, handedness), point.color, (local_uv + cell) / grid, local_uv,
        (local_uv + next_cell) / grid, frac(frame_position));
}
)slang";
        const std::string mesh = geometry + R"slang(
struct ParticleMeshVertex { float4 position; float4 normal; float4 tangent; float4 color; float4 uv; };
[[vk::binding(2, 0)]] StructuredBuffer<ParticleMeshVertex> mesh_vertices;
[[vk::binding(3, 0)]] StructuredBuffer<uint> mesh_indices;
[shader("vertex")]
ParticleOutput main(uint vertex_id : SV_VulkanVertexID, uint instance : SV_VulkanInstanceID) {
    ParticleRenderInstance point = instances[render_indices[instance]];
    ParticleMeshVertex vertex = mesh_vertices[mesh_indices[vertex_id]];
    float3 c = cos(point.rotation_custom.yzw), s = sin(point.rotation_custom.yzw);
    float3x3 rx = float3x3(1,0,0, 0,c.x,-s.x, 0,s.x,c.x);
    float3x3 ry = float3x3(c.y,0,s.y, 0,1,0, -s.y,0,c.y);
    float3x3 rz = float3x3(c.z,-s.z,0, s.z,c.z,0, 0,0,1);
    float3x3 rotation = mul(mul(rz, ry), rx);
    float3 scale = point.scale_custom.xyz * point.position_size.w;
    float3 position = point.position_size.xyz + mul(rotation, vertex.position.xyz * scale);
    float3 transformed = mul(rotation, vertex.normal.xyz * sign(scale) / max(abs(scale), float3(1e-6)));
    float squared = dot(transformed, transformed);
    float3 normal = squared > 1e-12 ? transformed * rsqrt(squared) : float3(0,1,0);
    float4 unbiased_clip = mul(view.view_projection, float4(position, 1.0));
    if (view.rendering_control.y > 0.5) {
        float3 to_light = view.camera_right.w > 0.5 ? normalize(view.camera_right.xyz - position) : normalize(view.camera_right.xyz);
        float perspective = view.camera_right.w > 0.5 ? clamp(abs(unbiased_clip.w) / max(view.camera_up.w, 1e-6), .001, 1.0) : 1.0;
        float texel = max(view.camera_up.z * perspective, 1e-6);
        position -= to_light * (view.camera_up.x * texel);
        position -= normal * (view.camera_up.y * texel * (1.0 - saturate(dot(normal, to_light))));
    }
    ParticleOutput output = particle_output(point, position, normal, float4(mul(rotation, vertex.tangent.xyz), vertex.tangent.w),
        point.color * vertex.color, vertex.uv.xy, vertex.uv.xy, vertex.uv.xy, 0.0);
    if (view.rendering_control.y > 0.5 && view.camera_right.w < 0.5) output.position.z = max(output.position.z, 0.0);
    return output;
}
)slang";
        const std::string picking = std::string(GetSlangParticleSharedSource()) + R"slang(
[shader("fragment")]
uint2 main([[vk::location(14)]] float alpha : TEXCOORD14) : SV_Target0 {
    if (!(alpha > 0.0)) discard;
    return uint2(asuint(view.material_tint.x), asuint(view.material_tint.y));
}
)slang";
        const std::string billboardMotion = R"slang(
[[vk::binding(2, 0)]] Sampler2D albedo_map;
[shader("fragment")]
float2 main([[vk::location(4)]] float2 uv : TEXCOORD4,
    [[vk::location(14)]] float alpha : TEXCOORD14,
    [[vk::location(15)]] float2 motion : TEXCOORD15) : SV_Target0 {
    if (albedo_map.Sample(uv).a * alpha <= .0001) discard;
    return motion;
}
)slang";
        const std::string meshMotion = R"slang(
[shader("fragment")]
float2 main([[vk::location(14)]] float alpha : TEXCOORD14,
    [[vk::location(15)]] float2 motion : TEXCOORD15) : SV_Target0 {
    if (!(alpha > 0.0)) discard;
    return motion;
}
)slang";
        const auto module = [](std::string_view renderer, std::string_view stage, std::string source) {
            const auto name = "particle_" + std::string(renderer) + "_" + std::string(stage);
            return SlangParticleGraphicsModule{renderer, stage, {name, "<infernux/" + name + ">", std::move(source)}};
        };
        return std::array<SlangParticleGraphicsModule, 9>{
            module("billboard", "vertex", billboard),
            module("billboard", "picking_fragment", picking),
            module("billboard", "motion_vertex", "#define INX_PARTICLE_MOTION_PASS 1\n" + billboard),
            module("billboard", "motion_fragment", billboardMotion),
            module("mesh", "vertex", mesh),
            module("mesh", "shadow_fragment", "[shader(\"fragment\")] void main() {}\n"),
            module("mesh", "picking_fragment", picking),
            module("mesh", "motion_vertex", "#define INX_PARTICLE_MOTION_PASS 1\n" + mesh),
            module("mesh", "motion_fragment", meshMotion),
        };
    }();
    return modules;
}

} // namespace infernux
