// Engine-owned object layer forwarded from InstanceAuxData. This is shared by
// Forward, Forward+, and GBuffer so Light.culling_mask has identical semantics.
layout(location = 15) flat in uvec2 _inx_ObjectRenderData;
#define _inx_ObjectLayerMask _inx_ObjectRenderData.x
#define _inx_ReceivesShadows (_inx_ObjectRenderData.y != 0u && INX_MATERIAL_RECEIVES_SHADOWS != 0)
#define INX_GEOMETRY_SHADOW_CONTROL 1
