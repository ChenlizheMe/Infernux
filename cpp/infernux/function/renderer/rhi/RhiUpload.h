#pragma once

#include "RhiDescriptors.h"

#include <cstddef>
#include <cstdint>

namespace infernux::rhi
{

inline constexpr uint64_t InlineBufferUpdateBytes = 64 * 1024;
inline constexpr uint64_t MinUploadStagingClass = 4 * 1024;
inline constexpr uint64_t MaxPooledUploadStagingClass = 16 * 1024 * 1024;
inline constexpr uint64_t UploadStagingPoolBytes = 128ULL * 1024 * 1024;

[[nodiscard]] constexpr uint64_t UploadStagingClass(uint64_t bytes) noexcept
{
    if (!bytes || bytes > MaxPooledUploadStagingClass)
        return bytes;
    uint64_t size = MinUploadStagingClass;
    while (size < bytes)
        size *= 2;
    return size;
}

struct BufferUploadRequest
{
    const void *data = nullptr;
    size_t byteSize = 0;
    BufferUsageFlags usage = BufferUsageFlags::Storage;
    /// Queues that consume the uploaded buffer; the uploader adds its own queue.
    QueueAccessFlags queueAccess = QueueAccessFlags::Graphics;
};

struct TextureSubresourceUpload
{
    const void *data = nullptr;
    size_t byteSize = 0;
    uint32_t mipLevel = 0;
    uint32_t baseLayer = 0;
    uint32_t layerCount = 1;
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t depth = 1;
    size_t rowPitch = 0;
    size_t slicePitch = 0;
};

struct TextureUploadRequest
{
    TextureDesc texture;
    TextureViewDesc view;
    SamplerDesc sampler;
    const TextureSubresourceUpload *subresources = nullptr;
    uint32_t subresourceCount = 0;
};

} // namespace infernux::rhi
