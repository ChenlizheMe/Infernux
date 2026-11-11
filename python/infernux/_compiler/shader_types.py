"""Resource value types consumed by the static Shader frontends."""

RESOURCE_TYPES = {"Color2D": "InxColor2D"}

# A logical image carries its sampler through typed library calls. Slang
# flattens the pair for each target; neither authors nor helpers name slots.
RESOURCE_MODULE = ("inx_resource_types", "<infernux/shader/resources>", """
public struct InxColor2D
{
    public Texture2D<float4> texture;
    public SamplerState sampler;
};
""")
