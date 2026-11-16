#pragma once
#include "SlangCompiler.h"
#include <array>
#include <string_view>

namespace infernux
{
inline const ShaderSourceModule &GetSlangParticleModule()
{
    static const ShaderSourceModule module{"inx_particle", "<infernux/shader/particle>", R"slang(import inx_camera;
import inx_geometry;
public struct InxParticleInput
{
    public float4 position : SV_Position;
    [[vk::location(0)]] public float3 position_ws : TEXCOORD0;
    [[vk::location(1)]] public float3 normal_ws : TEXCOORD1;
    [[vk::location(2)]] public float4 tangent_ws : TEXCOORD2;
    [[vk::location(3)]] public float3 color : TEXCOORD3;
    [[vk::location(4)]] public float2 uv0 : TEXCOORD4;
    [[vk::location(5)]] public float view_depth : TEXCOORD5;
    [[vk::location(6)]] public float4 line_color : TEXCOORD6;
    [[vk::location(7)]] public nointerpolation float2 flipbook_grid : TEXCOORD7;
    [[vk::location(8)]] public float2 motion_vector : TEXCOORD8;
    [[vk::location(9)]] public float2 local_uv : TEXCOORD9;
    [[vk::location(10)]] public float2 next_uv : TEXCOORD10;
    [[vk::location(11)]] public float frame_blend : TEXCOORD11;
    [[vk::location(12)]] public float normalized_age : TEXCOORD12;
    [[vk::location(13)]] public nointerpolation uint particle_id : TEXCOORD13;
    [[vk::location(14)]] public float alpha : TEXCOORD14;
    [[vk::location(15)]] public nointerpolation uint layer_mask : TEXCOORD15;
};

public struct InxParticleInstance {
    public float4 position_size;
    public float4 color;
    public float4 rotation_custom;
    public float4 scale_custom;
    public uint4 ribbon_data;
    public float4 custom_data;
    public float4 previous_position_history;
};
public struct InxParticleDrawValues {
    public float4 camera_right;
    public float4 camera_up;
    public uint4 object_data;
    public float4 depth_reconstruct;
    public float4 lighting_control;
    public float4 rendering_control;
    public float4 alignment_reference;
};

[[vk::push_constant]] ConstantBuffer<InxParticleDrawValues> inx_native_particle_draw;
[[vk::binding(999, 0)]] ConstantBuffer<InxParticleDrawValues> inx_portable_particle_draw;
public InxParticleDrawValues inx_particle_draw()
{
    __target_switch {
    case spirv: return inx_native_particle_draw;
    case wgsl: return inx_portable_particle_draw;
    }
}
public float4 inx_particle_clip(float3 position)
{
    return mul(inx_view.projection, mul(inx_view.view, float4(position, 1.0)));
}
public float4 inx_particle_previous_clip(float3 position)
{
    return mul(inx_view.previous_view_projection, float4(position, 1.0));
}
public uint2 inx_particle_owner_id()
{
    return inx_particle_draw().object_data.xy;
}
[[vk::binding(7, 1)]] ConstantBuffer<InxShadowParameters> inx_particle_shadow_parameters;
public InxParticleInput inx_particle_shadow_output(InxParticleInput input)
{
    // Same reversed-depth directional near-plane convention as mesh casters.
    // Bias comes from the shared light/view contract, never camera basis lanes.
    if (inx_particle_shadow_parameters.light_vector.w < 0.5)
        input.position.z = min(input.position.z, input.position.w);
    input.view_depth = 0.0;
    return input;
}

[[vk::binding(5, 2)]] DepthTexture2D inx_particle_scene_depth;
[[vk::binding(6, 2)]] Texture2D<float> inx_particle_scene_color_depth;
public float inx_particle_eye_depth(float depth)
{
    float4 projection = inx_particle_draw().depth_reconstruct;
    float numerator = projection.y - depth * projection.w;
    float denominator = depth * projection.z - projection.x;
    return abs(numerator / (abs(denominator) > 1e-7 ? denominator : 1e-7));
}
public float inx_particle_soft_coverage<let depth_plane : bool>(InxParticleInput input)
{
    uint width, height;
    float scene_depth;
    if (depth_plane) {
        inx_particle_scene_depth.GetDimensions(width, height);
        int2 pixel = clamp(int2(input.position.xy), int2(0), int2(width, height) - 1);
        scene_depth = inx_particle_scene_depth.Load(int3(pixel, 0));
    } else {
        inx_particle_scene_color_depth.GetDimensions(width, height);
        int2 pixel = clamp(int2(input.position.xy), int2(0), int2(width, height) - 1);
        scene_depth = inx_particle_scene_color_depth.Load(int3(pixel, 0));
    }
    float distance = inx_particle_eye_depth(scene_depth) - inx_particle_eye_depth(input.position.z);
    return smoothstep(0.0, 1.0, saturate(distance / max(inx_particle_draw().camera_right.w, 1e-4)));
}
public float4 inx_particle_apply_soft_coverage(float4 color, float coverage)
{
    color.a *= coverage;
    if (inx_particle_draw().rendering_control.y > 0.5)
        color.rgb *= coverage;
    return color;
}

)slang"};
    return module;
}

inline std::string_view GetSlangParticleGeometrySource()
{
    return R"slang(import inx_particle;
import inx_camera;
import inx_geometry;
// Match GpuParticleViewConstants::objectData.w.
static const uint INX_PARTICLE_MOTION_USE_CURRENT_POSITION = 1u;
static const uint INX_PARTICLE_MOTION_RESET = 2u;

float3 particle_axis(float3 value, float3 fallback) {
    float squared = dot(value, value);
    return squared > 1e-10 ? value * rsqrt(squared) : fallback;
}
InxParticleInput particle_output(InxParticleInstance point, float3 position, float3 normal, float4 tangent,
    float4 color, float2 uv, float2 local_uv, float2 next_uv, float blend, bool motion) {
    InxParticleInput output;
    float4 clip = inx_particle_clip(position);
    output.position = clip;
    output.position.y = -output.position.y;
    output.position_ws = position;
    output.normal_ws = normal;
    output.tangent_ws = tangent;
    output.color = color.rgb;
    output.uv0 = uv;
    output.view_depth = mul(inx_view.view, float4(position, 1.0)).z;
    output.line_color = float4(1.0);
    output.flipbook_grid = inx_particle_draw().rendering_control.zw;
    output.local_uv = local_uv;
    output.next_uv = next_uv;
    output.frame_blend = blend;
    output.normalized_age = saturate(point.scale_custom.w);
    output.particle_id = point.ribbon_data.w;
    output.alpha = color.a;
    output.layer_mask = inx_particle_draw().object_data.z;
    output.motion_vector = float2(0.0);
    uint motion_flags = inx_particle_draw().object_data.w;
    if (motion && (motion_flags & INX_PARTICLE_MOTION_RESET) == 0u) {
        float3 previous = (motion_flags & INX_PARTICLE_MOTION_USE_CURRENT_POSITION) != 0u
            ? position : point.previous_position_history.xyz + position - point.position_size.xyz;
        float4 previous_clip = inx_particle_previous_clip(previous);
        output.motion_vector = (clip.xy / max(abs(clip.w), 1e-6) -
            previous_clip.xy / max(abs(previous_clip.w), 1e-6)) * float2(0.5, -0.5);
    }
    return output;
}

)slang";
}

struct SlangParticleGraphicsModule
{
    std::string_view renderer;
    ShaderSourceModule module;
};

inline const std::array<SlangParticleGraphicsModule, 3> &GetSlangParticleGraphicsModules()
{
    static const std::array<SlangParticleGraphicsModule, 3> modules = {{
        {"billboard",
         {"particle_billboard", "<infernux/particle/billboard>",
          std::string(GetSlangParticleGeometrySource()) +
              R"slang([[vk::binding(1, 2)]] StructuredBuffer<InxParticleInstance> instances;
[[vk::binding(2, 2)]] StructuredBuffer<uint> render_indices;
static const float2 corners[6] = {
    float2(-1,-1), float2(-1,1), float2(1,1), float2(-1,-1), float2(1,1), float2(1,-1)
};
static const float2 uvs[6] = {
    float2(0,1), float2(0,0), float2(1,0), float2(0,1), float2(1,0), float2(1,1)
};
void billboard_basis(float3 position, float3 custom, out float3 right, out float3 up) {
    float3 camera_right = particle_axis(inx_particle_draw().camera_right.xyz, float3(1,0,0));
    float3 camera_up = particle_axis(inx_particle_draw().camera_up.xyz, float3(0,1,0));
    float3 camera_normal = particle_axis(cross(camera_right, camera_up), float3(0,0,1));
    int alignment = int(round(inx_particle_draw().alignment_reference.w));
    right = camera_right;
    up = camera_up;
    if (alignment == 1) {
        float3 to_camera = particle_axis(inx_particle_draw().alignment_reference.xyz - position, camera_normal);
        right = particle_axis(cross(camera_up, to_camera), camera_right);
        up = particle_axis(cross(to_camera, right), camera_up);
    } else if (alignment == 2 || alignment == 3) {
        up = particle_axis(alignment == 2 ? inx_particle_draw().alignment_reference.xyz : custom, camera_up);
        float3 projected = cross(up, camera_normal);
        if (dot(projected, projected) <= 1e-10) projected = camera_right - up * dot(camera_right, up);
        right = particle_axis(projected, camera_right);
        up = particle_axis(cross(camera_normal, right), up);
    }
}
InxParticleInput particle_billboard_vertex(uint vertex : SV_VulkanVertexID, uint instance : SV_VulkanInstanceID, bool motion) {
    InxParticleInstance point = instances[render_indices[instance]];
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
    float2 grid = max(inx_particle_draw().rendering_control.zw, float2(1.0));
    float count = grid.x * grid.y;
    float frame_position = fmod(max(point.custom_data.x, 0.0), count);
    float frame = floor(frame_position), next = fmod(frame + 1.0, count);
    float2 local_uv = uvs[vertex % 6u];
    float2 cell = float2(fmod(frame, grid.x), floor(frame / grid.x));
    float2 next_cell = float2(fmod(next, grid.x), floor(next / grid.x));
    return particle_output(point, position, particle_axis(cross(right, up), float3(0,0,1)),
        float4(tangent, handedness), point.color, (local_uv + cell) / grid, local_uv,
        (local_uv + next_cell) / grid, frac(frame_position), motion);
}

[shader("vertex")]
InxParticleInput vertex_main(uint vertex : SV_VulkanVertexID, uint instance : SV_VulkanInstanceID) {
    return particle_billboard_vertex(vertex, instance, false);
}
[shader("vertex")]
InxParticleInput vertex_motion(uint vertex : SV_VulkanVertexID, uint instance : SV_VulkanInstanceID) {
    return particle_billboard_vertex(vertex, instance, true);
}
[shader("vertex")]
InxParticleInput vertex_shadow(uint vertex : SV_VulkanVertexID, uint instance : SV_VulkanInstanceID) {
    return inx_particle_shadow_output(particle_billboard_vertex(vertex, instance, false));
}
)slang"}},
        {"mesh",
         {"particle_mesh", "<infernux/particle/mesh>",
          std::string(GetSlangParticleGeometrySource()) +
              R"slang([[vk::binding(1, 2)]] StructuredBuffer<InxParticleInstance> instances;
[[vk::binding(2, 2)]] StructuredBuffer<uint> render_indices;
struct ParticleMeshVertex { float4 position; float4 normal; float4 tangent; float4 color; float4 uv; };
[[vk::binding(3, 2)]] StructuredBuffer<ParticleMeshVertex> mesh_vertices;
[[vk::binding(4, 2)]] StructuredBuffer<uint> mesh_indices;
InxParticleInput particle_mesh_vertex(uint vertex_id : SV_VulkanVertexID, uint instance : SV_VulkanInstanceID, bool motion) {
    InxParticleInstance point = instances[render_indices[instance]];
    ParticleMeshVertex vertex = mesh_vertices[mesh_indices[vertex_id]];
    float3 c = cos(point.rotation_custom.yzw), s = sin(point.rotation_custom.yzw);
    float3x3 rx = float3x3(1,0,0, 0,c.x,-s.x, 0,s.x,c.x);
    float3x3 ry = float3x3(c.y,0,s.y, 0,1,0, -s.y,0,c.y);
    float3x3 rz = float3x3(c.z,-s.z,0, s.z,c.z,0, 0,0,1);
    float3x3 rotation = mul(mul(rz, ry), rx);
    float3 scale = point.scale_custom.xyz * point.position_size.w;
    float4x4 model = float4x4(
        rotation[0][0] * scale.x, rotation[0][1] * scale.y, rotation[0][2] * scale.z, point.position_size.x,
        rotation[1][0] * scale.x, rotation[1][1] * scale.y, rotation[1][2] * scale.z, point.position_size.y,
        rotation[2][0] * scale.x, rotation[2][1] * scale.y, rotation[2][2] * scale.z, point.position_size.z,
        0.0, 0.0, 0.0, 1.0);
    InxVertexInput authored = (InxVertexInput)0;
    authored.position = vertex.position.xyz;
    authored.normal = vertex.normal.xyz;
    authored.tangent = vertex.tangent;
    InxInstanceData owner = (InxInstanceData)0;
    InxMeshInput transformed = inx_transform_vertex(authored, model, inx_view.view, inx_view.projection, owner);
    InxParticleInput output = particle_output(point, transformed.position_ws, transformed.normal_ws,
        transformed.tangent_ws, point.color * vertex.color, vertex.uv.xy, vertex.uv.xy, vertex.uv.xy, 0.0, motion);
    return output;
}

[shader("vertex")]
InxParticleInput vertex_main(uint vertex : SV_VulkanVertexID, uint instance : SV_VulkanInstanceID) {
    return particle_mesh_vertex(vertex, instance, false);
}
[shader("vertex")]
InxParticleInput vertex_motion(uint vertex : SV_VulkanVertexID, uint instance : SV_VulkanInstanceID) {
    return particle_mesh_vertex(vertex, instance, true);
}
[shader("vertex")]
InxParticleInput vertex_shadow(uint vertex : SV_VulkanVertexID, uint instance : SV_VulkanInstanceID) {
    return inx_particle_shadow_output(particle_mesh_vertex(vertex, instance, false));
}
)slang"}},
        {"ribbon",
         {"particle_ribbon", "<infernux/particle/ribbon>",
          std::string(GetSlangParticleGeometrySource()) + R"slang(struct DrawArguments {
    uint vertex_count;
    uint instance_count;
    uint first_vertex;
    uint first_instance;
};
[[vk::binding(1, 2)]] StructuredBuffer<InxParticleInstance> instances;
[[vk::binding(2, 2)]] StructuredBuffer<uint> sorted_indices;
[[vk::binding(3, 2)]] StructuredBuffer<uint> visible_segments;
[[vk::binding(4, 2)]] StructuredBuffer<DrawArguments> source_arguments;
static const uint endpoint_for_vertex[6] = {0u, 0u, 1u, 0u, 1u, 1u};
static const float side_for_vertex[6] = {-1.0, 1.0, 1.0, -1.0, 1.0, -1.0};
bool ribbon_points_connected(uint4 first, uint4 second) {
    return first.x == second.x && (second.z & 1u) == 0u;
}
float3 ribbon_side(float3 first, float3 second, float3 fallback_side) {
    float3 tangent = second - first;
    float tangent_length = length(tangent);
    tangent = tangent_length > 1e-7 ? tangent / tangent_length : normalize(inx_particle_draw().camera_up.xyz);
    float3 camera_forward = cross(inx_particle_draw().camera_right.xyz, inx_particle_draw().camera_up.xyz);
    float3 side = cross(camera_forward, tangent);
    float side_length = length(side);
    return side_length > 1e-7 ? side / side_length : fallback_side;
}
float3 ribbon_join_offset(float3 first_side, float3 second_side, float half_width) {
    float3 sum = first_side + second_side;
    float sum_length = length(sum);
    if (sum_length <= 1e-5) return second_side * half_width;
    float3 join = sum / sum_length;
    float projection = max(abs(dot(join, second_side)), 1e-4);
    return join * (half_width * min(1.0 / projection, 2.0));
}
struct RibbonSegment {
    InxParticleInstance first;
    InxParticleInstance second;
    InxParticleInstance neighbor;
    uint endpoint;
    uint corner;
    bool connected;
    bool neighbor_connected;
};
RibbonSegment ribbon_segment(uint vertex, uint instance) {
    bool compacted = inx_particle_draw().alignment_reference.z > 0.5;
    uint segment = compacted ? visible_segments[instance] : vertex / 6u;
    RibbonSegment result;
    result.corner = vertex % 6u;
    result.endpoint = endpoint_for_vertex[result.corner];
    result.first = instances[sorted_indices[segment]];
    result.second = instances[sorted_indices[segment + 1u]];
    result.connected = ribbon_points_connected(result.first.ribbon_data, result.second.ribbon_data);
    result.neighbor = result.first;
    result.neighbor_connected = false;
    if (result.connected && result.endpoint == 0u && segment > 0u) {
        result.neighbor = instances[sorted_indices[segment - 1u]];
        result.neighbor_connected = ribbon_points_connected(result.neighbor.ribbon_data, result.first.ribbon_data);
    } else if (result.connected && result.endpoint == 1u && segment + 2u < source_arguments[0].instance_count) {
        result.neighbor = instances[sorted_indices[segment + 2u]];
        result.neighbor_connected = ribbon_points_connected(result.second.ribbon_data, result.neighbor.ribbon_data);
    }
    return result;
}
float3 ribbon_point_position(InxParticleInstance point, bool previous) {
    return previous ? point.previous_position_history.xyz : point.position_size.xyz;
}
struct RibbonFrame {
    float3 position;
    float3 tangent;
    float3 side;
};
RibbonFrame ribbon_frame(RibbonSegment segment, bool previous) {
    float3 first = ribbon_point_position(segment.first, previous);
    float3 second = segment.connected ? ribbon_point_position(segment.second, previous) : first;
    InxParticleInstance point = segment.endpoint == 0u ? segment.first : segment.second;
    float3 center = segment.endpoint == 0u ? first : second;
    float half_width = abs(point.position_size.w) * 0.5;
    RibbonFrame result;
    result.side = ribbon_side(first, second, normalize(inx_particle_draw().camera_right.xyz));
    result.tangent = normalize(second - first + normalize(inx_particle_draw().camera_up.xyz) * 1e-8);
    float3 offset = result.side * half_width;
    if (segment.neighbor_connected) {
        float3 neighbor = ribbon_point_position(segment.neighbor, previous);
        if (segment.endpoint == 0u)
            offset = ribbon_join_offset(ribbon_side(neighbor, first, result.side), result.side, half_width);
        else
            offset = ribbon_join_offset(result.side, ribbon_side(second, neighbor, result.side), half_width);
    }
    result.position = center + offset * side_for_vertex[segment.corner];
    return result;
}

InxParticleInput particle_ribbon_vertex(uint vertex, uint instance, bool motion) {
    RibbonSegment segment = ribbon_segment(vertex, instance);
    RibbonFrame frame = ribbon_frame(segment, false);
    InxParticleInstance point = segment.endpoint == 0u ? segment.first : segment.second;
    float order_coordinate = float(point.ribbon_data.y);
    float uv_scale = max(inx_particle_draw().alignment_reference.x, 1e-6);
    float2 uv = float2(inx_particle_draw().alignment_reference.y > 0.5 ? order_coordinate * uv_scale : order_coordinate / uv_scale,
                      side_for_vertex[segment.corner] > 0.0 ? 1.0 : 0.0);
    float2 local_uv = float2(float(segment.endpoint), side_for_vertex[segment.corner] > 0.0 ? 1.0 : 0.0);
    InxParticleInput output = particle_output(point, frame.position, normalize(cross(frame.side, frame.tangent)),
        float4(frame.tangent, 1.0), point.color, uv, local_uv, local_uv, 0.0, false);
    if (!segment.connected) output.alpha = 0.0;
    uint motion_flags = inx_particle_draw().object_data.w;
    if (motion && (motion_flags & INX_PARTICLE_MOTION_RESET) == 0u) {
        float3 previous = (motion_flags & INX_PARTICLE_MOTION_USE_CURRENT_POSITION) != 0u
            ? frame.position : ribbon_frame(segment, true).position;
        float4 current_clip = inx_particle_clip(frame.position);
        float4 previous_clip = inx_particle_previous_clip(previous);
        output.motion_vector = (current_clip.xy / max(abs(current_clip.w), 1e-6) -
            previous_clip.xy / max(abs(previous_clip.w), 1e-6)) * float2(0.5, -0.5);
    }
    return output;
}
[shader("vertex")]
InxParticleInput vertex_main(uint vertex : SV_VulkanVertexID, uint instance : SV_VulkanInstanceID) {
    return particle_ribbon_vertex(vertex, instance, false);
}
[shader("vertex")]
InxParticleInput vertex_motion(uint vertex : SV_VulkanVertexID, uint instance : SV_VulkanInstanceID) {
    return particle_ribbon_vertex(vertex, instance, true);
}
[shader("vertex")]
InxParticleInput vertex_shadow(uint vertex : SV_VulkanVertexID, uint instance : SV_VulkanInstanceID) {
    return inx_particle_shadow_output(particle_ribbon_vertex(vertex, instance, false));
}
)slang"}},
    }};
    return modules;
}
} // namespace infernux
