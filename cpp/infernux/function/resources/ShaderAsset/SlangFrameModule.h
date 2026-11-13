#pragma once

#include "SlangCompiler.h"

namespace infernux
{

// Imported only by graphics programs that consume the current view's globals.
inline const ShaderSourceModule &GetSlangFrameModule()
{
    static const ShaderSourceModule module{"inx_frame", "<infernux/shader/frame>", R"slang(
import inx_geometry;
[[vk::binding(0, 2)]] public ConstantBuffer<InxEngineGlobals> inx_globals;
public float inx_frame_time() { return inx_globals.time.x; }
public float inx_frame_delta_time() { return inx_globals.time.w; }
public struct InxViewContext
{
    public float2 size;
    public float2 pixel_size;
    public float3 camera_position_ws;
    public float near_clip;
    public float far_clip;
    public float4 depth_params;
    public bool dithering;
    public bool stop_nans;
    public bool opaque_output;
};
public InxViewContext inx_view_context()
{
    InxViewContext result;
    result.size = inx_globals.screen_params.xy;
    result.pixel_size = inx_globals.screen_params.zw;
    result.camera_position_ws = inx_globals.camera_position.xyz;
    result.near_clip = inx_globals.projection_params.x;
    result.far_clip = inx_globals.projection_params.y;
    result.depth_params = inx_globals.depth_params;
    result.dithering = inx_globals.view_options.x != 0u;
    result.stop_nans = inx_globals.view_options.y != 0u;
    result.opaque_output = inx_globals.view_options.z != 0u;
    return result;
}
)slang"};
    return module;
}

} // namespace infernux
