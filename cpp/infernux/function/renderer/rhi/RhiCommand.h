#pragma once

#include "RhiDescriptors.h"

#include <cstddef>
#include <cstdint>

namespace infernux::rhi
{

class GraphicsCommandEncoder
{
  public:
    struct Dispatch
    {
        void (*bindPipeline)(void *, GraphicsPipelineHandle) = nullptr;
        void (*bindGroup)(void *, GraphicsPipelineHandle, uint32_t, BindGroupHandle) = nullptr;
        void (*pushConstants)(void *, GraphicsPipelineHandle, ShaderStage, uint32_t, const void *) = nullptr;
        void (*draw)(void *, uint32_t, uint32_t, uint32_t, uint32_t) = nullptr;
        void (*drawIndirect)(void *, BufferHandle, uint64_t, uint32_t, uint32_t) = nullptr;
        void (*bindVertexBuffer)(void *, uint32_t, BufferHandle, uint64_t, uint64_t) = nullptr;
        void (*bindIndexBuffer)(void *, BufferHandle, IndexFormat, uint64_t, uint64_t) = nullptr;
        void (*drawIndexed)(void *, uint32_t, uint32_t, uint32_t, int32_t, uint32_t) = nullptr;
        void (*drawIndexedIndirect)(void *, BufferHandle, uint64_t, uint32_t, uint32_t) = nullptr;

        [[nodiscard]] constexpr bool IsComplete() const noexcept
        {
            return bindPipeline && bindGroup && pushConstants && draw && drawIndirect && bindVertexBuffer &&
                   bindIndexBuffer && drawIndexed && drawIndexedIndirect;
        }
    };

    constexpr GraphicsCommandEncoder() noexcept = default;
    constexpr GraphicsCommandEncoder(void *context, const Dispatch *dispatch) noexcept
        : m_context(context), m_dispatch(dispatch)
    {
    }

    [[nodiscard]] constexpr bool IsValid() const noexcept
    {
        return m_context != nullptr && m_dispatch != nullptr;
    }

    /// A valid context can still expose an incomplete optional table. Callers
    /// that require the full graphics contract must check this explicitly;
    /// incomplete tables are never treated as a successful command stream.
    [[nodiscard]] constexpr bool HasCompleteDispatch() const noexcept
    {
        return IsValid() && m_dispatch->IsComplete();
    }

    void BindPipeline(GraphicsPipelineHandle pipeline) const
    {
        if (IsValid() && m_dispatch->bindPipeline)
            m_dispatch->bindPipeline(m_context, pipeline);
    }

    void BindGroup(GraphicsPipelineHandle pipeline, uint32_t setIndex, BindGroupHandle group) const
    {
        if (IsValid() && m_dispatch->bindGroup)
            m_dispatch->bindGroup(m_context, pipeline, setIndex, group);
    }

    void PushConstants(GraphicsPipelineHandle pipeline, ShaderStage stages, uint32_t byteSize, const void *data) const
    {
        if (IsValid() && m_dispatch->pushConstants && byteSize > 0 && data)
            m_dispatch->pushConstants(m_context, pipeline, stages, byteSize, data);
    }

    void Draw(uint32_t vertexCount, uint32_t instanceCount = 1, uint32_t firstVertex = 0,
              uint32_t firstInstance = 0) const
    {
        if (IsValid() && m_dispatch->draw)
            m_dispatch->draw(m_context, vertexCount, instanceCount, firstVertex, firstInstance);
    }

    void DrawIndirect(BufferHandle arguments, uint64_t offset = 0, uint32_t drawCount = 1, uint32_t stride = 16) const
    {
        if (IsValid() && m_dispatch->drawIndirect && arguments.IsValid() && drawCount > 0)
            m_dispatch->drawIndirect(m_context, arguments, offset, drawCount, stride);
    }

    void BindVertexBuffer(uint32_t slot, BufferHandle buffer, uint64_t offset = 0, uint64_t byteSize = 0) const
    {
        m_dispatch->bindVertexBuffer(m_context, slot, buffer, offset, byteSize);
    }

    void BindIndexBuffer(BufferHandle buffer, IndexFormat format, uint64_t offset = 0, uint64_t byteSize = 0) const
    {
        m_dispatch->bindIndexBuffer(m_context, buffer, format, offset, byteSize);
    }

    void DrawIndexed(uint32_t indexCount, uint32_t instanceCount = 1, uint32_t firstIndex = 0, int32_t baseVertex = 0,
                     uint32_t firstInstance = 0) const
    {
        m_dispatch->drawIndexed(m_context, indexCount, instanceCount, firstIndex, baseVertex, firstInstance);
    }

    void DrawIndexedIndirect(BufferHandle arguments, uint64_t offset = 0, uint32_t drawCount = 1,
                             uint32_t stride = 20) const
    {
        m_dispatch->drawIndexedIndirect(m_context, arguments, offset, drawCount, stride);
    }

  private:
    void *m_context = nullptr;
    const Dispatch *m_dispatch = nullptr;
};

class ComputeCommandEncoder
{
  public:
    struct DispatchTable
    {
        void (*bindPipeline)(void *, ComputePipelineHandle) = nullptr;
        void (*bindGroup)(void *, ComputePipelineHandle, uint32_t, BindGroupHandle) = nullptr;
        void (*pushConstants)(void *, ComputePipelineHandle, uint32_t, const void *) = nullptr;
        void (*dispatch)(void *, uint32_t, uint32_t, uint32_t) = nullptr;
        void (*dispatchIndirect)(void *, BufferHandle, uint64_t) = nullptr;
        bool (*canRecord)(void *, ComputePipelineHandle, BindGroupHandle) = nullptr;

        [[nodiscard]] constexpr bool IsComplete() const noexcept
        {
            return bindPipeline && bindGroup && pushConstants && dispatch && dispatchIndirect && canRecord;
        }
    };

    constexpr ComputeCommandEncoder() noexcept = default;
    constexpr ComputeCommandEncoder(void *context, const DispatchTable *dispatch) noexcept
        : m_context(context), m_dispatch(dispatch)
    {
    }

    [[nodiscard]] constexpr bool IsValid() const noexcept
    {
        return m_context != nullptr && m_dispatch != nullptr;
    }

    [[nodiscard]] constexpr bool HasCompleteDispatch() const noexcept
    {
        return IsValid() && m_dispatch->IsComplete();
    }

    /// Check the actual recording context and the set-zero pipeline/group
    /// contract before emitting any commands. A dispatch table alone does not
    /// prove that a native command buffer or matching device exists.
    [[nodiscard]] bool CanRecord(ComputePipelineHandle pipeline, BindGroupHandle group) const
    {
        return HasCompleteDispatch() && pipeline.IsValid() && group.IsValid() &&
               m_dispatch->canRecord(m_context, pipeline, group);
    }

    void BindPipeline(ComputePipelineHandle pipeline) const
    {
        if (IsValid() && m_dispatch->bindPipeline)
            m_dispatch->bindPipeline(m_context, pipeline);
    }

    void BindGroup(ComputePipelineHandle pipeline, uint32_t setIndex, BindGroupHandle group) const
    {
        if (IsValid() && m_dispatch->bindGroup)
            m_dispatch->bindGroup(m_context, pipeline, setIndex, group);
    }

    void PushConstants(ComputePipelineHandle pipeline, uint32_t byteSize, const void *data) const
    {
        if (IsValid() && m_dispatch->pushConstants && byteSize > 0 && data)
            m_dispatch->pushConstants(m_context, pipeline, byteSize, data);
    }

    void Dispatch(uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1) const
    {
        if (IsValid() && m_dispatch->dispatch && groupCountX > 0 && groupCountY > 0 && groupCountZ > 0)
            m_dispatch->dispatch(m_context, groupCountX, groupCountY, groupCountZ);
    }

    void DispatchIndirect(BufferHandle arguments, uint64_t offset = 0) const
    {
        if (IsValid() && m_dispatch->dispatchIndirect && arguments.IsValid())
            m_dispatch->dispatchIndirect(m_context, arguments, offset);
    }

  private:
    void *m_context = nullptr;
    const DispatchTable *m_dispatch = nullptr;
};

struct BufferCopyRegion
{
    uint64_t sourceOffset = 0;
    uint64_t destinationOffset = 0;
    uint64_t byteSize = 0;
};

struct TextureCopyRegion
{
    TextureAspect aspect = TextureAspect::Color;
    uint32_t sourceMip = 0;
    uint32_t sourceLayer = 0;
    uint32_t destinationMip = 0;
    uint32_t destinationLayer = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
    uint32_t sourceX = 0;
    uint32_t sourceY = 0;
    uint32_t sourceZ = 0;
    uint32_t destinationX = 0;
    uint32_t destinationY = 0;
    uint32_t destinationZ = 0;
};

/// Single-aspect texels at the origin of one 2D mip/layer.
struct TextureBufferCopyRegion
{
    TextureAspect aspect = TextureAspect::Color;
    uint32_t textureMip = 0;
    uint32_t textureLayer = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint64_t bufferOffset = 0;
    /// Zero is tightly packed. WebGPU multi-row command copies require 256-byte rows.
    uint32_t bytesPerRow = 0;

    [[nodiscard]] static constexpr uint32_t TexelBytes(PixelFormat format) noexcept
    {
        const auto layout = GetPixelFormatLayout(format);
        return layout.blockWidth == 1 && layout.blockHeight == 1 && !IsStencilFormat(format) ? layout.bytesPerBlock : 0;
    }

    [[nodiscard]] static constexpr TextureAspect AspectFor(PixelFormat format) noexcept
    {
        return IsDepthFormat(format) ? TextureAspect::Depth : TextureAspect::Color;
    }

    [[nodiscard]] static constexpr uint64_t ByteSize(PixelFormat format, uint32_t width, uint32_t height) noexcept
    {
        const auto rowBytes = uint64_t{width} * TexelBytes(format);
        return rowBytes && height <= UINT64_MAX / rowBytes ? rowBytes * height : 0;
    }

    [[nodiscard]] constexpr bool Fits(PixelFormat format, uint64_t bytes) const noexcept
    {
        auto size = ByteSize(format, width, height);
        const auto texelBytes = TexelBytes(format);
        if (size && bytesPerRow) {
            const auto rowBytes = uint64_t{width} * texelBytes;
            if (bytesPerRow < rowBytes || bytesPerRow % texelBytes != 0)
                return false;
            size = uint64_t{height - 1} * bytesPerRow + rowBytes;
        }
        return aspect == AspectFor(format) && size && size <= bytes &&
               bufferOffset % (texelBytes > 4u ? texelBytes : 4u) == 0;
    }
};

struct TextureResolveRegion
{
    TextureAspect aspect = TextureAspect::Color;
    uint32_t sourceMip = 0;
    uint32_t sourceLayer = 0;
    uint32_t destinationMip = 0;
    uint32_t destinationLayer = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
};

class TransferCommandEncoder
{
  public:
    struct DispatchTable
    {
        void (*copyBuffer)(void *, BufferHandle, BufferHandle, const BufferCopyRegion &) = nullptr;
        void (*copyTexture)(void *, TextureHandle, TextureHandle, const TextureCopyRegion &) = nullptr;
        void (*resolveTexture)(void *, TextureHandle, TextureHandle, const TextureResolveRegion &) = nullptr;
        bool (*fillBuffer)(void *, BufferHandle, uint64_t, uint64_t, uint32_t) = nullptr;
        bool (*updateBuffer)(void *, BufferHandle, uint64_t, const void *, uint64_t) = nullptr;
        bool (*copyTextureToBuffer)(void *, TextureHandle, BufferHandle, const TextureBufferCopyRegion &) = nullptr;
        bool (*copyBufferToTexture)(void *, BufferHandle, TextureHandle, const TextureBufferCopyRegion &) = nullptr;

        [[nodiscard]] constexpr bool IsComplete() const noexcept
        {
            return copyBuffer && copyTexture && resolveTexture && fillBuffer && updateBuffer && copyTextureToBuffer &&
                   copyBufferToTexture;
        }
    };

    constexpr TransferCommandEncoder() noexcept = default;
    constexpr TransferCommandEncoder(void *context, const DispatchTable *dispatch) noexcept
        : m_context(context), m_dispatch(dispatch)
    {
    }

    [[nodiscard]] constexpr bool IsValid() const noexcept
    {
        return m_context != nullptr && m_dispatch != nullptr;
    }

    [[nodiscard]] constexpr bool HasCompleteDispatch() const noexcept
    {
        return IsValid() && m_dispatch->IsComplete();
    }

    void CopyBuffer(BufferHandle source, BufferHandle destination, const BufferCopyRegion &region) const
    {
        if (IsValid() && m_dispatch->copyBuffer && source.IsValid() && destination.IsValid() && region.byteSize > 0)
            m_dispatch->copyBuffer(m_context, source, destination, region);
    }

    /// Record a repeated 32-bit pattern, not a byte memset. Destination must
    /// have TransferDestination usage. Offset and size are explicit multiples
    /// of four; caller owns transfer-write barriers and completion dependencies.
    /// Returns false without recording if the encoder, handle or range is invalid.
    [[nodiscard]] bool FillBuffer(BufferHandle destination, uint64_t offset, uint64_t byteSize,
                                  uint32_t value = 0) const
    {
        return IsValid() && m_dispatch->fillBuffer && destination.IsValid() && byteSize > 0 && offset % 4 == 0 &&
               byteSize % 4 == 0 && m_dispatch->fillBuffer(m_context, destination, offset, byteSize, value);
    }

    /// Capture host bytes into this recording's immutable upload payload. The
    /// caller may change or release data immediately after successful recording.
    /// The destination changes only when the transfer executes. Callers own
    /// transfer-write barriers and completion dependencies. Requires
    /// TransferDestination usage and nonempty, four-byte-aligned ranges. Large
    /// updates require submission-owned staging storage in the native recording
    /// scope. This command must be outside a render pass.
    [[nodiscard]] bool UpdateBuffer(BufferHandle destination, uint64_t offset, const void *data,
                                    uint64_t byteSize) const
    {
        return IsValid() && m_dispatch->updateBuffer && destination.IsValid() && data && byteSize > 0 &&
               offset % 4 == 0 && byteSize % 4 == 0 &&
               m_dispatch->updateBuffer(m_context, destination, offset, data, byteSize);
    }

    void CopyTexture(TextureHandle source, TextureHandle destination, const TextureCopyRegion &region) const
    {
        if (IsValid() && m_dispatch->copyTexture && source.IsValid() && destination.IsValid() && region.width > 0 &&
            region.height > 0 && region.depth > 0)
            m_dispatch->copyTexture(m_context, source, destination, region);
    }

    /// Record image-to-buffer transfer without submission, host mapping or
    /// implicit barriers. Source requires a single-sample image with a tightly
    /// packed color or depth aspect and TransferSource usage. The buffer needs
    /// TransferDestination usage. Depth copies require a graphics queue. The caller
    /// chooses buffer memory, including explicit Readback, and owns any host
    /// visibility barrier and completion wait before mapping it.
    /// The caller owns both resources through actual GPU completion and must
    /// establish the transfer layouts/accesses outside a render pass.
    /// Returns false without recording when any resource or range is invalid.
    [[nodiscard]] bool CopyTextureToBuffer(TextureHandle source, BufferHandle destination,
                                           const TextureBufferCopyRegion &region) const
    {
        return CanCopyTextureBuffer(source, destination, region) && m_dispatch->copyTextureToBuffer &&
               m_dispatch->copyTextureToBuffer(m_context, source, destination, region);
    }

    /// Record the reverse transfer using the same texel layout. The source is
    /// TransferSource and the texture must be in its transfer-destination layout.
    [[nodiscard]] bool CopyBufferToTexture(BufferHandle source, TextureHandle destination,
                                           const TextureBufferCopyRegion &region) const
    {
        return CanCopyTextureBuffer(destination, source, region) && m_dispatch->copyBufferToTexture &&
               m_dispatch->copyBufferToTexture(m_context, source, destination, region);
    }

    void ResolveTexture(TextureHandle source, TextureHandle destination, const TextureResolveRegion &region) const
    {
        if (IsValid() && m_dispatch->resolveTexture && source.IsValid() && destination.IsValid() &&
            source != destination && region.width > 0 && region.height > 0 && region.depth > 0)
            m_dispatch->resolveTexture(m_context, source, destination, region);
    }

  private:
    [[nodiscard]] bool CanCopyTextureBuffer(TextureHandle texture, BufferHandle buffer,
                                            const TextureBufferCopyRegion &region) const noexcept
    {
        return IsValid() && texture.IsValid() && buffer.IsValid() &&
               (region.aspect == TextureAspect::Color || region.aspect == TextureAspect::Depth) && region.width > 0 &&
               region.height > 0 && region.bufferOffset % 4u == 0;
    }

    void *m_context = nullptr;
    const DispatchTable *m_dispatch = nullptr;
};

static_assert(sizeof(GraphicsCommandEncoder) == sizeof(void *) * 2);
static_assert(sizeof(ComputeCommandEncoder) == sizeof(void *) * 2);
static_assert(sizeof(TransferCommandEncoder) == sizeof(void *) * 2);

} // namespace infernux::rhi
