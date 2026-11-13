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
)slang"};
    return module;
}

} // namespace infernux
