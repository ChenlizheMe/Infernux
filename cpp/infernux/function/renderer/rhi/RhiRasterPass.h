#pragma once

#include "RhiDescriptors.h"
#include <array>

namespace infernux::rhi
{

enum class AttachmentLoad : uint8_t
{
    Load,
    Clear,
};

enum class AttachmentStore : uint8_t
{
    Store,
    Discard,
};

struct RasterColorAttachment
{
    TextureViewHandle view;
    TextureViewHandle resolve;
    PixelFormat format = PixelFormat::Undefined;
    AttachmentLoad load = AttachmentLoad::Load;
    AttachmentStore store = AttachmentStore::Store;
    std::array<float, 4> clear{};
};

struct RasterDepthAttachment
{
    TextureViewHandle view;
    PixelFormat format = PixelFormat::Undefined;
    AttachmentLoad load = AttachmentLoad::Load;
    AttachmentStore store = AttachmentStore::Store;
    float clear = 0.0f;
    uint32_t clearStencil = 0;
    bool readOnly = false;
};

/// Attachment contract compiled once with the graph. Backend adapters translate
/// it into their cached native descriptors; recording does not rebuild it.
/// A read-only depth/stencil attachment never performs a clear or store.
struct RasterPassDescription
{
    static constexpr size_t MaxColorTargets = GraphicsPipelineDesc::MaxColorTargets;
    std::array<RasterColorAttachment, MaxColorTargets> colors{};
    uint32_t colorCount = 0;
    RasterDepthAttachment depth;
    uint32_t width = 0;
    uint32_t height = 0;
};

} // namespace infernux::rhi
