"""Resource value types consumed by the static Shader frontends."""

RESOURCE_TYPES = {"Color2D": "InxColor2D", "Depth2D": "InxDepth2D", "UInt2D": "InxUInt2D", "History2D": "InxHistory2D"}
RESOURCE_BINDINGS = RESOURCE_TYPES | {"History2D": "InxColor2D"}
RESOURCE_ELEMENTS = {"InxColor2D": "float4", "InxDepth2D": "float", "InxUInt2D": "uint4", "InxHistory2D": "float4"}
SURFACE_FIELDS = {
    "albedo": ("float3", "float3(1.0)"), "normal_ws": ("float3", "float3(0.0)"),
    "metallic": ("float", "0.0"), "smoothness": ("float", "0.5"), "occlusion": ("float", "1.0"),
    "emission": ("float3", "float3(0.0)"), "alpha": ("float", "1.0"), "specular_highlights": ("float", "1.0"),
    "shading_param0": ("float", "0.0"), "shading_param1": ("float", "0.0"),
}
STRUCT_TYPES = {"fragment_input": "InxFragmentInput", "mesh_input": "InxMeshInput", "SurfaceContext": "InxMeshInput",
                "SurfaceData": "InxSurfaceData", "ShadingContext": "InxShadingContext", "LightData": "InxLight",
                "VertexData": "InxVertexData", "VertexContext": "InxVertexContext", "ViewContext": "InxViewContext",
                "ProjectionContext": "InxProjectionContext", "CameraContext": "InxCameraContext",
                "FragmentOutput": "InxFragmentOutput"}
STRUCT_FIELDS = {"InxFragmentInput": {"position": "float4", "uv": "float2"},
                 "InxFragmentOutput": {"color": "float4", "depth": "float"},
                 "InxCameraContext": {"view_matrix": "float4x4", "projection_matrix": "float4x4",
                                      "view_projection_matrix": "float4x4", "previous_view_projection_matrix": "float4x4",
                                      "inverse_view_projection_matrix": "float4x4",
                                      "projection_params": "float4", "depth_params": "float4"},
                 "InxViewContext": {"size": "float2", "pixel_size": "float2", "camera_position_ws": "float3",
                                    "near_clip": "float", "far_clip": "float", "depth_params": "float4",
                                    "dithering": "bool", "stop_nans": "bool", "opaque_output": "bool"},
                 "InxHistory2D": {"valid": "bool"},
                 "InxMeshInput": {"position": "float4", "position_ws": "float3", "normal_ws": "float3",
                                  "tangent_ws": "float4", "color": "float3", "uv0": "float2", "uv1": "float2",
                                  "view_depth": "float", "line_color": "float4", "object_id": "uint2",
                                  "object_flags": "uint", "layer_mask": "uint"},
                 "InxSurfaceData": {name: kind for name, (kind, _) in SURFACE_FIELDS.items()},
                 "InxShadingContext": {"position_ws": "float3", "geometric_normal_ws": "float3", "view_depth": "float",
                                       "pixel_position": "float2", "layer_mask": "uint", "receives_shadows": "bool"},
                 "InxLight": {"direction": "float3", "color": "float3", "attenuation": "float", "shadow": "float"},
                 "InxVertexData": {"position_os": "float3", "normal_os": "float3", "tangent_os": "float4",
                                   "color": "float3", "uv0": "float2", "uv1": "float2"},
                 "InxVertexContext": {"time": "float", "delta_time": "float", "object_id": "uint2",
                                      "layer_mask": "uint", "model_matrix": "float4x4"},
                 "InxProjectionContext": {"view_matrix": "float4x4", "projection_matrix": "float4x4"}}

# A logical image carries its sampler through typed library calls. Slang
# flattens the pair for each target; neither authors nor helpers name slots.
RESOURCE_MODULE = ("inx_resource_types", "<infernux/shader/resources>", """
public struct InxColor2D
{
    public Texture2D<float4> texture;
    public SamplerState sampler;
};
public struct InxHistory2D
{
    public Texture2D<float4> texture;
    public SamplerState sampler;
    public bool valid;
};
public InxHistory2D inx_history(InxColor2D source, bool valid)
{
    InxHistory2D result;
    result.texture = source.texture;
    result.sampler = source.sampler;
    result.valid = valid;
    return result;
}
public struct InxDepth2D
{
    public Texture2D<float> texture;
    public SamplerState sampler;
};
public struct InxUInt2D
{
    [format("unknown")] public Texture2D<uint4> texture;
};
public struct InxFragmentInput
{
    public float4 position : SV_Position;
    public float2 uv : TEXCOORD0;
};
""" + "\n".join("""
public uint2 inx_image_dimensions(TYPE image, uint mip)
{
    uint width, height, levels;
    image.texture.GetDimensions(mip, width, height, levels);
    return uint2(width, height);
}
""".replace("TYPE", kind) for kind in RESOURCE_TYPES.values()))
