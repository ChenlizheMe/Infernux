#pragma once

#include "RenderGraphDescription.h"
#include "rhi/RenderViewContext.h"
#include "rhi/RhiCommand.h"

#include <algorithm>

namespace infernux
{

inline rhi::BufferCopyRegion GraphBufferCopyRegion(const GraphCommandDesc &command, const GraphBufferDesc &from,
                                                   const GraphBufferDesc &to)
{
    return {from.byteOffset, to.byteOffset,
            command.copyBytes ? command.copyBytes : std::min(from.byteSize, to.byteSize)};
}

inline rhi::TextureCopyRegion GraphTextureCopyRegion(const GraphTextureDesc &from, const GraphTextureDesc &to,
                                                     const rhi::RenderViewContext &view)
{
    const auto [fromWidth, fromHeight] = from.ResolveExtent(view.width, view.height);
    const auto [toWidth, toHeight] = to.ResolveExtent(view.width, view.height);
    return {rhi::TextureBufferCopyRegion::AspectFor(from.EffectiveFormat(view)),
            0,
            0,
            0,
            0,
            std::min(fromWidth, toWidth),
            std::min(fromHeight, toHeight),
            1};
}

inline rhi::TextureBufferCopyRegion GraphTextureBufferCopyRegion(const GraphCommandDesc &command,
                                                                 const GraphTextureDesc &texture,
                                                                 const GraphBufferDesc &buffer,
                                                                 const rhi::RenderViewContext &view)
{
    const bool toBuffer = command.type == GraphCommandType::CopyTextureToBuffer;
    const auto [width, height] = texture.ResolveExtent(view.width, view.height);
    return {rhi::TextureBufferCopyRegion::AspectFor(texture.EffectiveFormat(view)),
            0,
            0,
            toBuffer ? command.copyWidth : width,
            toBuffer ? command.copyHeight : height,
            buffer.byteOffset + command.copyBufferOffset};
}

} // namespace infernux
