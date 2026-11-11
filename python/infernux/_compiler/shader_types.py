"""Resource value types consumed by the static Shader frontends."""

RESOURCE_TYPES = {"Color2D": "InxColor2D"}
STRUCT_TYPES = {"fragment_input": "InxFragmentInput", "mesh_input": "InxMeshInput"}
STRUCT_FIELDS = {"InxFragmentInput": {"position": "float4", "uv": "float2"},
                 "InxMeshInput": {"position": "float4", "position_ws": "float3", "normal_ws": "float3",
                                  "tangent_ws": "float4", "color": "float3", "uv0": "float2", "uv1": "float2",
                                  "view_depth": "float", "line_color": "float4"}}

# A logical image carries its sampler through typed library calls. Slang
# flattens the pair for each target; neither authors nor helpers name slots.
RESOURCE_MODULE = ("inx_resource_types", "<infernux/shader/resources>", """
public struct InxColor2D
{
    public Texture2D<float4> texture;
    public SamplerState sampler;
};
public struct InxFragmentInput
{
    public float4 position : SV_Position;
    public float2 uv : TEXCOORD0;
};
public uint2 inx_image_dimensions(InxColor2D image, uint mip)
{
    uint width, height, levels;
    image.texture.GetDimensions(mip, width, height, levels);
    return uint2(width, height);
}
""")
