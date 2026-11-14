#pragma once

#include <function/renderer/rhi/RhiHandles.h>

namespace infernux
{

/// Borrowed resources owned by one camera or light view. The frame buffer is
/// the engine's camera-local globals, shared by every shader in this view.
struct ShaderViewBindings
{
    rhi::BindingLayoutHandle layout;
    rhi::BindGroupHandle group;
    rhi::BufferHandle globals;
};

} // namespace infernux
