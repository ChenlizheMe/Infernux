#pragma once

#include "SlangCompiler.h"
#include <core/types/ShaderTypes.h>

namespace infernux
{

// Pure geometry declarations shared by native programs, static Python authors
// and their ShaderLibs. Resource ownership stays in the program, never here.
inline const ShaderSourceModule &GetSlangGeometryModule()
{
    static const ShaderSourceModule module{"inx_geometry", "<infernux/shader/geometry>",
                                           R"slang(
public struct InxVertexInput
{
    [[vk::location(0)]] public float3 position : POSITION;
    [[vk::location(1)]] public float3 normal : NORMAL;
    [[vk::location(2)]] public float4 tangent : TANGENT;
    [[vk::location(3)]] public float3 color : COLOR;
    [[vk::location(4)]] public float2 uv0 : TEXCOORD0;
    [[vk::location(5)]] public uint4 bone_indices : BLENDINDICES;
    [[vk::location(6)]] public float4 bone_weights : BLENDWEIGHT;
    [[vk::location(7)]] public float2 uv1 : TEXCOORD1;
};
public struct InxMeshInput
{
    public float4 position : SV_Position;
    public float3 position_ws : TEXCOORD0;
    public float3 normal_ws : TEXCOORD1;
    public float4 tangent_ws : TEXCOORD2;
    public float3 color : TEXCOORD3;
    public float2 uv0 : TEXCOORD4;
    public float2 uv1 : TEXCOORD5;
    public float view_depth : TEXCOORD6;
    public float4 line_color : TEXCOORD7;
    public nointerpolation uint2 object_id : TEXCOORD8;
    public nointerpolation uint object_flags : TEXCOORD9;
    public nointerpolation uint layer_mask : TEXCOORD10;
    public float2 motion_vector : TEXCOORD11;
};
public struct InxSurfaceData
{
    public float3 albedo;
    public float3 normal_ws;
    public float metallic;
    public float smoothness;
    public float occlusion;
    public float3 emission;
    public float alpha;
    public float specular_highlights;
    public float shading_param0;
    public float shading_param1;
};
public InxSurfaceData inx_surface_data(float3 albedo, float3 normal_ws, float metallic, float smoothness,
                                      float occlusion, float3 emission, float alpha, float specular_highlights,
                                      float shading_param0, float shading_param1)
{
    InxSurfaceData result;
    result.albedo = albedo;
    result.normal_ws = normal_ws;
    result.metallic = metallic;
    result.smoothness = smoothness;
    result.occlusion = occlusion;
    result.emission = emission;
    result.alpha = alpha;
    result.specular_highlights = specular_highlights;
    result.shading_param0 = shading_param0;
    result.shading_param1 = shading_param1;
    return result;
}
public struct InxCameraData
{
    public column_major float4x4 model;
    public column_major float4x4 view;
    public column_major float4x4 projection;
    public column_major float4x4 previous_view_projection;
    public column_major float4x4 inverse_view_projection;
    public float4 projection_params;
    public float4 depth_params;
};
public struct InxShadowParameters
{
    public float4 light_vector;
    public float4 bias;
};
public struct InxModelMatrix { public column_major float4x4 value; };
public struct InxInstanceData
{
    public column_major float4x4 previous_model;
    public uint2 object_id;
    public uint object_flags;
    public uint layer_mask;
};
public struct InxSkinState
{
    public uint bone_offset;
    public uint bone_count;
    public uint flags;
    public uint previous_bone_offset;
};
public float3 inx_transform_normal(float4x4 matrix, float3 normal)
{
    float3 a = float3(matrix[0][0], matrix[1][0], matrix[2][0]);
    float3 b = float3(matrix[0][1], matrix[1][1], matrix[2][1]);
    float3 c = float3(matrix[0][2], matrix[1][2], matrix[2][2]);
    float3 value = cross(b, c) * normal.x + cross(c, a) * normal.y + cross(a, b) * normal.z;
    return normalize(value * sign(dot(a, cross(b, c))));
}
public InxVertexInput inx_skin_vertex(InxVertexInput input, InxSkinState skin,
                                     StructuredBuffer<InxModelMatrix> bones)
{
    if ((skin.flags & 1u) != 0u && skin.bone_count != 0u && input.bone_indices.w != 0x4C494E45u)
    {
        uint4 index = skin.bone_offset + min(input.bone_indices, skin.bone_count - 1u);
        float4x4 matrix = input.bone_weights.x * bones[index.x].value
                       + input.bone_weights.y * bones[index.y].value
                       + input.bone_weights.z * bones[index.z].value
                       + input.bone_weights.w * bones[index.w].value;
        input.position = mul(matrix, float4(input.position, 1.0)).xyz;
        input.normal = inx_transform_normal(matrix, input.normal);
        input.tangent.xyz = normalize(mul((float3x3)matrix, input.tangent.xyz));
        input.tangent.w *= sign(determinant((float3x3)matrix));
    }
    return input;
}
public InxMeshInput inx_transform_vertex(InxVertexInput input, float4x4 model, float4x4 view, float4x4 projection,
                                        InxInstanceData instance)
{
    InxMeshInput result;
    float4 world = mul(model, float4(input.position, 1.0));
    float3 normal = inx_transform_normal(model, input.normal);
    float3 tangent = normalize(mul((float3x3)model, input.tangent.xyz));
    float tangent_sign = input.tangent.w * sign(determinant((float3x3)model));
    bool line = input.bone_indices.w == 0x4C494E45u;
    if (line)
    {
        float3 right = view[0].xyz;
        float3 up = view[1].xyz;
        float3 facing = input.bone_indices.z == 0u ? -view[2].xyz : normal;
        float3 fallback = right - tangent * dot(right, tangent);
        if (dot(fallback, fallback) < 1.0e-8)
            fallback = up - tangent * dot(up, tangent);
        if (dot(fallback, fallback) < 1.0e-10)
            fallback = abs(tangent.x) < 0.9 ? cross(tangent, float3(1, 0, 0)) : cross(tangent, float3(0, 1, 0));
        fallback = normalize(fallback);
        float3 side = cross(facing, tangent);
        float side_length = length(side);
        if (side_length > 0.20)
            side /= side_length;
        else if (side_length > 1.0e-6)
        {
            side /= side_length;
            if (dot(fallback, side) < 0.0) fallback = -fallback;
            side = normalize(lerp(fallback, side, smoothstep(0.025, 0.20, side_length)));
        }
        else side = fallback;
        world.xyz += side * input.bone_weights.x;
        normal = input.bone_indices.y != 0u ? facing : normal;
        tangent_sign = 1.0;
    }
    result.position_ws = world.xyz;
    result.normal_ws = normal;
    result.tangent_ws = float4(tangent, tangent_sign);
    result.color = input.color;
    result.uv0 = input.uv0;
    result.uv1 = input.uv1;
    result.view_depth = mul(view, world).z;
    result.line_color = line ? float4(input.color, input.bone_weights.y) : float4(1.0);
    result.object_id = instance.object_id;
    result.object_flags = instance.object_flags;
    result.layer_mask = instance.layer_mask;
    result.motion_vector = float2(0.0);
    result.position = mul(projection, mul(view, world));
    // The existing shared camera buffer stores a Vulkan viewport projection.
    // Return canonical Y-up before the common target compiler's clip lowering.
    result.position.y = -result.position.y;
    return result;
}
public InxSurfaceData inx_prepare_surface(InxSurfaceData surface, InxMeshInput input, bool front_facing)
{
    surface.albedo *= input.line_color.rgb;
    surface.alpha *= input.line_color.a;
    float3 normal = dot(surface.normal_ws, surface.normal_ws) > 1e-8 ? surface.normal_ws : input.normal_ws;
    surface.normal_ws = normalize(normal) * (front_facing ? 1.0 : -1.0);
    return surface;
}
public struct InxGBufferOutput
{
    public float4 albedo_alpha : SV_Target0;
    public float4 normal_smoothness : SV_Target1;
    public float4 material : SV_Target2;
    public float4 emission : SV_Target3;
    public uint4 metadata : SV_Target4;
};
public InxGBufferOutput inx_pack_gbuffer(InxSurfaceData surface, InxMeshInput input)
{
    InxGBufferOutput result;
    result.albedo_alpha = float4(surface.albedo, surface.alpha);
    result.normal_smoothness = float4(surface.normal_ws * 0.5 + 0.5, surface.smoothness);
    result.material = float4(saturate(surface.metallic), saturate(surface.occlusion),
                             saturate(surface.specular_highlights), surface.shading_param0);
    result.emission = float4(surface.emission, surface.shading_param1);
    result.metadata = uint4(input.layer_mask, )slang" +
                                               std::to_string(ShadingModelId("PBR")) + R"slang(u,
                            (input.object_flags & 2u) == 0u ? 1u : 0u, 0u);
    return result;
}
)slang"};
    return module;
}

} // namespace infernux
