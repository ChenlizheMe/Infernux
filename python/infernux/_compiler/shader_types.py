"""Resource value types consumed by the static Shader frontends."""

RESOURCE_TYPES = {kind: "Inx" + kind for kind in (
    "Color2D", "Depth2D", "UInt2D", "History2D", "Color3D", "Color2DMS", "Depth2DMS", "UInt2DMS",
)}
RESOURCE_BINDINGS = RESOURCE_TYPES | {"History2D": "InxColor2D"}
RESOURCE_ELEMENTS = {value: "uint4" if key.startswith("UInt") else "float" if key.startswith("Depth") else "float4"
                     for key, value in RESOURCE_TYPES.items()}
RESOURCE_DIMENSIONS = {value: 3 if "3D" in key else 2 for key, value in RESOURCE_TYPES.items()}
MULTISAMPLED_TYPES = {value for key, value in RESOURCE_TYPES.items() if key.endswith("MS")}
BUFFER_TYPES = {kind + (str(width) if width > 1 else ""): "InxBuffer_" + kind + str(width)
                for kind in ("float", "int", "uint") for width in range(1, 5)}
BUFFER_ELEMENTS = {value: key for key, value in BUFFER_TYPES.items()}
SURFACE_FIELDS = {
    "albedo": ("float3", "float3(1.0)"), "normal_ws": ("float3", "float3(0.0)"),
    "metallic": ("float", "0.0"), "smoothness": ("float", "0.5"), "occlusion": ("float", "1.0"),
    "emission": ("float3", "float3(0.0)"), "alpha": ("float", "1.0"), "specular_highlights": ("float", "1.0"),
    "shading_param0": ("float", "0.0"), "shading_param1": ("float", "0.0"),
}
STRUCT_TYPES = {"fragment_input": "InxFragmentInput", "multisample_input": "InxMultisampleInput", "mesh_input": "InxMeshInput", "SurfaceContext": "InxMeshInput",
                "SurfaceData": "InxSurfaceData", "ShadingContext": "InxShadingContext", "LightData": "InxLight",
                "VertexData": "InxVertexData", "VertexContext": "InxVertexContext", "ViewContext": "InxViewContext",
                "ProjectionContext": "InxProjectionContext", "CameraContext": "InxCameraContext",
                "FragmentOutput": "InxFragmentOutput"}
STRUCT_FIELDS = {"InxFragmentInput": {"position": "float4", "uv": "float2"},
                 "InxMultisampleInput": {"position": "float4", "uv": "float2", "sample_index": "uint"},
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
public struct InxMultisampleInput
{
    public float4 position : SV_Position;
    public float2 uv : TEXCOORD0;
    public uint sample_index : SV_SampleIndex;
};
public struct InxColor3D
{
    public Texture3D<float4> texture;
    public SamplerState sampler;
};
""" + "\n".join("""
public struct TYPE { public Texture2DMS<ELEMENT> texture; };
public uint2 inx_image_dimensions(TYPE image)
{
    uint width, height, samples;
    image.texture.GetDimensions(width, height, samples);
    return uint2(width, height);
}
public uint inx_image_samples(TYPE image)
{
    uint width, height, samples;
    image.texture.GetDimensions(width, height, samples);
    return samples;
}
""".replace("TYPE", kind).replace("ELEMENT", RESOURCE_ELEMENTS[kind]) for kind in sorted(MULTISAMPLED_TYPES)) + """
public uint3 inx_image_dimensions(InxColor3D image, uint mip)
{
    uint width, height, depth, levels;
    image.texture.GetDimensions(mip, width, height, depth, levels);
    return uint3(width, height, depth);
}
""" + "\n".join("""
public uint2 inx_image_dimensions(TYPE image, uint mip)
{
    uint width, height, levels;
    image.texture.GetDimensions(mip, width, height, levels);
    return uint2(width, height);
}
""".replace("TYPE", kind) for kind in RESOURCE_TYPES.values() if kind not in MULTISAMPLED_TYPES and kind != "InxColor3D") + "\n".join(
    """
public struct WRAPPER
{
    public ByteAddressBuffer data;
    public uint byte_offset;
    public uint count;
};
public WRAPPER inx_buffer_ELEMENT(ByteAddressBuffer data, uint byte_offset, uint count)
{
    WRAPPER result;
    result.data = data;
    result.byte_offset = byte_offset;
    result.count = count;
    return result;
}
public ELEMENT inx_buffer_load(WRAPPER source, uint index)
{
    return source.data.Load<ELEMENT>(source.byte_offset + index * STRIDE);
}
public uint inx_buffer_length(WRAPPER source) { return source.count; }
""".replace("WRAPPER", wrapper).replace("ELEMENT", element)
    .replace("STRIDE", str(4 * (int(element[-1]) if element[-1].isdigit() else 1)) + "u")
    for element, wrapper in BUFFER_TYPES.items()))
