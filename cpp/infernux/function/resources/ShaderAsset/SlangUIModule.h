#pragma once

#include "SlangCompiler.h"
#include <string_view>

namespace infernux
{

inline constexpr std::string_view SlangUIVertexSource = R"slang(struct VertexInput {
    [[vk::location(0)]] float2 position : TEXCOORD0;
    [[vk::location(1)]] float2 uv : TEXCOORD1;
    [[vk::location(2)]] float4 color : TEXCOORD2;
};
struct VertexOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float4 color : TEXCOORD0;
    [[vk::location(1)]] float2 uv : TEXCOORD1;
};

[shader("vertex")]
VertexOutput vertex_main(VertexInput input) {
    VertexOutput output;
    float2 position = input.position * pc.scale + pc.translate;
    output.position = float4(position.x, -position.y, 0.0, 1.0);
    output.uv = input.uv;
    output.color = input.color;
    return output;
}

)slang";

/// UI geometry and camera matrices use the engine's top-left viewport
/// convention. Each vertex entry converts it to the compiler's Y-up clip
/// space; the target then performs its usual viewport conversion.
inline const ShaderSourceModule &GetSlangScreenUIModule()
{
    static const ShaderSourceModule module{"inx_screen_ui", "<infernux/renderer/screen_ui>", R"slang(
struct ScreenConstants {
    float2 scale;
    float2 translate;
    float encode_sample;
    float padding0;
    float padding1;
    float padding2;
    float4 material_color;
    float alpha_clip_threshold;
    float alpha_clip_enabled;
    float2 tail_padding;
};
INX_ROOT_CONSTANTS ConstantBuffer<ScreenConstants> pc;
[[vk::binding(0, 0)]] Sampler2D<float4> ui_texture;

)slang" + std::string(SlangUIVertexSource) + R"slang(
[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target0 {
    float4 sampled = ui_texture.Sample(input.uv);
    if (pc.encode_sample > 0.5) {
        float3 rgb = max(sampled.rgb, float3(0.0));
        sampled.rgb = select(rgb <= float3(0.0031308), 12.92 * rgb,
                             1.055 * pow(rgb, float3(1.0 / 2.4)) - 0.055);
    }
    float4 color = input.color * sampled * pc.material_color;
    if (pc.alpha_clip_enabled > 0.5 && color.a < pc.alpha_clip_threshold)
        discard;
    return color;
}
)slang"};
    return module;
}

/// Editor draw data shares one root block across vertex and fragment entries:
/// scale at byte 0, translation at byte 8, and texture encoding at byte 16.
inline const ShaderSourceModule &GetSlangEditorUIModule()
{
    static const ShaderSourceModule module{"inx_editor_ui", "<infernux/renderer/editor_ui>",
                                           R"slang(struct EditorConstants {
    float2 scale;
    float2 translate;
    int encode_sample;
};
INX_ROOT_CONSTANTS ConstantBuffer<EditorConstants> pc;
[[vk::binding(0, 0)]] Sampler2D<float4> ui_texture;
)slang" + std::string(SlangUIVertexSource) + R"slang(
[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target0 {
    float4 sampled = ui_texture.Sample(input.uv);
    if (pc.encode_sample != 0) {
        float3 value = sampled.rgb;
        sampled.rgb = select(value <= float3(.0031308), value * 12.92,
            1.055 * pow(max(value, float3(0.0)), float3(1.0 / 2.4)) - .055);
    }
    return input.color * sampled;
}
)slang"};
    return module;
}

inline const ShaderSourceModule &GetSlangWorldUIModule()
{
    static const ShaderSourceModule module{"inx_world_ui", "<infernux/renderer/world_ui>", R"slang(
struct WorldConstants {
    column_major float4x4 view_projection;
    float4 material_color;
    float alpha_clip_threshold;
    float alpha_clip_enabled;
    float2 screen_scale;
    float4 camera_right;
    float4 camera_up;
};
INX_ROOT_CONSTANTS ConstantBuffer<WorldConstants> pc;
[[vk::binding(0, 0)]] Sampler2D<float4> ui_texture;

struct VertexInput {
    [[vk::location(0)]] float3 position : TEXCOORD0;
    [[vk::location(1)]] float2 uv : TEXCOORD1;
    [[vk::location(2)]] float4 color : TEXCOORD2;
    [[vk::location(3)]] float2 local_position : TEXCOORD3;
    [[vk::location(4)]] float3 anchor : TEXCOORD4;
    [[vk::location(5)]] float3 local_offset : TEXCOORD5;
    [[vk::location(6)]] float policy : TEXCOORD6;
};
struct VertexOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float4 color : TEXCOORD0;
    [[vk::location(1)]] float2 uv : TEXCOORD1;
    [[vk::location(2)]] float2 local_position : TEXCOORD2;
};

[shader("vertex")]
VertexOutput vertex_main(VertexInput input) {
    VertexOutput output;
    float3 position = input.position;
    uint policy = uint(input.policy + 0.5);
    if (policy != 0u) {
        float3 offset = input.position - input.anchor;
        if ((policy & 1u) != 0u)
            offset = pc.camera_right.xyz * input.local_offset.x
                   + pc.camera_up.xyz * input.local_offset.y
                   + cross(pc.camera_right.xyz, pc.camera_up.xyz) * input.local_offset.z;
        if ((policy & 2u) != 0u) {
            float clip_w = mul(pc.view_projection, float4(input.anchor, 1.0)).w;
            offset *= max(clip_w, 0.0) * pc.screen_scale.x * 100.0;
        }
        position = input.anchor + offset;
    }
    output.position = mul(pc.view_projection, float4(position, 1.0));
    output.position.y = -output.position.y;
    output.color = input.color;
    output.uv = input.uv;
    output.local_position = input.local_position;
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target0 {
    float4 color = input.color * ui_texture.Sample(input.uv) * pc.material_color;
    // Empty coverage must not publish world depth; preserve antialiased edges.
    if ((pc.alpha_clip_enabled > 0.5 && color.a < pc.alpha_clip_threshold) || color.a <= 0.0)
        discard;
    return color;
}
)slang"};
    return module;
}

} // namespace infernux
