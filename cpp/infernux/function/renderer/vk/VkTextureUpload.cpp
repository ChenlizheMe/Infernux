#include "RhiVulkanTypes.h"
#include "VkDeviceContext.h"
#include "VkResourceManager.h"
#include "VulkanRhiDevice.h"
#include <algorithm>
#include <core/threading/FrameWorkOwner.h>
#include <limits>
#include <stdexcept>

namespace infernux::vk
{
namespace
{
VkDeviceSize AlignUploadOffset(VkDeviceSize value)
{
    constexpr VkDeviceSize Alignment = 16;
    if (value > std::numeric_limits<VkDeviceSize>::max() - (Alignment - 1))
        throw std::overflow_error("RHI texture upload staging size overflow");
    return (value + Alignment - 1) & ~(Alignment - 1);
}

struct TextureUploadPlan
{
    struct Copy
    {
        uint32_t sourceIndex;
        VkDeviceSize sourceOffset, inputOffset, bytes;
    };
    std::vector<Copy> copies;
    std::vector<VkBufferImageCopy> regions;
    VkDeviceSize stagingBytes = 0;
};

bool ValidTextureUploadBlock(const TextureUploadTicket &ticket, VkDeviceSize bytes, rhi::PixelFormatLayout format)
{
    return bytes && bytes % format.bytesPerBlock == 0 &&
           bytes <= ticket.GetResidentBytes() - ticket.GetSubmittedBytes();
}

TextureUploadPlan PlanTextureUpload(const std::vector<rhi::TextureSubresourceUpload> &layout,
                                    rhi::PixelFormatLayout format, VkDeviceSize submitted, VkDeviceSize bytes)
{
    TextureUploadPlan plan;
    VkDeviceSize inputOffset = 0, remaining = bytes, offset = submitted;
    for (uint32_t index = 0; index < layout.size() && remaining; ++index) {
        const auto &part = layout[index];
        if (offset >= part.byteSize) {
            offset -= part.byteSize;
            continue;
        }
        while (offset < part.byteSize && remaining) {
            const auto row = static_cast<uint32_t>((offset % part.slicePitch) / part.rowPitch);
            const auto column = static_cast<uint32_t>((offset % part.rowPitch) / format.bytesPerBlock);
            const auto slice = offset / part.slicePitch;
            const auto blockRows = format.RowCount(part.height);
            VkDeviceSize count;
            uint32_t width, height;
            if (column == 0 && remaining >= part.rowPitch) {
                const auto rows = static_cast<uint32_t>((std::min)(blockRows - row, remaining / part.rowPitch));
                count = part.rowPitch * rows;
                width = part.width;
                height = (std::min)(rows * format.blockHeight, part.height - row * format.blockHeight);
            } else {
                count = (std::min)(VkDeviceSize(part.rowPitch - column * format.bytesPerBlock), remaining);
                width = (std::min)(static_cast<uint32_t>(count / format.bytesPerBlock) * format.blockWidth,
                                   part.width - column * format.blockWidth);
                height = (std::min)(format.blockHeight, part.height - row * format.blockHeight);
            }
            plan.stagingBytes = AlignUploadOffset(plan.stagingBytes);
            plan.copies.push_back({index, offset, inputOffset, count});
            VkBufferImageCopy region{};
            region.bufferOffset = plan.stagingBytes;
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, part.mipLevel,
                                       part.baseLayer + static_cast<uint32_t>(slice / part.depth), 1};
            region.imageOffset = {static_cast<int32_t>(column * format.blockWidth),
                                  static_cast<int32_t>(row * format.blockHeight),
                                  static_cast<int32_t>(slice % part.depth)};
            region.imageExtent = {width, height, 1};
            plan.regions.push_back(region);
            plan.stagingBytes += count;
            offset += count;
            inputOffset += count;
            remaining -= count;
        }
        offset = 0;
    }
    return plan;
}

} // namespace

std::shared_ptr<TextureUploadTicket> VkResourceManager::BeginTextureUpload(const rhi::TextureUploadRequest &request)
{
    for (uint32_t index = 0; index < request.subresourceCount; ++index)
        if (!request.subresources || !request.subresources[index].data)
            throw std::invalid_argument("RHI texture upload requires source data");
    auto ticket = CreateTextureUpload(request);
    (void)AppendTextureUpload(ticket, ticket->GetResidentBytes(),
                              [&](uint32_t index, VkDeviceSize offset, VkDeviceSize) {
                                  return static_cast<const uint8_t *>(request.subresources[index].data) + offset;
                              });
    return ticket;
}

std::shared_ptr<TextureUploadTicket> VkResourceManager::CreateTextureUpload(const rhi::TextureUploadRequest &request)
{
    AssertReadbackThread();
    if (m_device == VK_NULL_HANDLE || !m_rhiDevice)
        throw std::logic_error("RHI texture upload requires an initialized resource manager");
    if (!rhi::HasTextureUsage(request.texture.usage, rhi::TextureUsageFlags::Sampled) ||
        !rhi::HasTextureUsage(request.texture.usage, rhi::TextureUsageFlags::TransferDestination) ||
        request.texture.samples != rhi::SampleCount::One)
        throw std::invalid_argument("RHI texture upload requires a single-sample sampled transfer destination");
    if (!request.subresources || request.subresourceCount == 0 || request.view.texture.IsValid())
        throw std::invalid_argument("RHI texture upload requires source subresources and an unbound view descriptor");

    const auto layout = rhi::GetPixelFormatLayout(request.texture.format);
    if (!layout.bytesPerBlock || rhi::IsDepthFormat(request.texture.format))
        throw std::invalid_argument("RHI texture upload requires a supported color format");
    const uint32_t arrayLayers =
        request.texture.dimension == rhi::TextureDimension::Texture3D ? 1U : request.texture.depthOrLayers;
    if (arrayLayers == 0 || request.texture.mipLevels == 0 || request.texture.mipLevels > 32)
        throw std::invalid_argument("RHI texture upload dimensions are empty");

    std::vector<bool> uploaded(static_cast<size_t>(request.texture.mipLevels) * arrayLayers, false);
    VkDeviceSize residentBytes = 0;

    for (uint32_t index = 0; index < request.subresourceCount; ++index) {
        const auto &source = request.subresources[index];
        if (source.byteSize == 0 || source.mipLevel >= request.texture.mipLevels || source.layerCount == 0)
            throw std::invalid_argument("RHI texture upload contains an empty or out-of-range subresource");

        const uint32_t mipWidth = (std::max)(1U, request.texture.width >> source.mipLevel);
        const uint32_t mipHeight = request.texture.dimension == rhi::TextureDimension::Texture1D
                                       ? 1U
                                       : (std::max)(1U, request.texture.height >> source.mipLevel);
        const uint32_t mipDepth = request.texture.dimension == rhi::TextureDimension::Texture3D
                                      ? (std::max)(1U, request.texture.depthOrLayers >> source.mipLevel)
                                      : 1U;
        if (source.width != mipWidth || source.height != mipHeight || source.depth != mipDepth)
            throw std::invalid_argument("RHI texture upload subresource extent does not match its mip level");
        if (request.texture.dimension == rhi::TextureDimension::Texture3D) {
            if (source.baseLayer != 0 || source.layerCount != 1)
                throw std::invalid_argument("RHI Texture3D uploads cannot address array layers");
        } else if (source.baseLayer >= arrayLayers || source.layerCount > arrayLayers - source.baseLayer) {
            throw std::invalid_argument("RHI texture upload array layer range is out of bounds");
        }

        const uint64_t rowPitch = layout.RowBytes(mipWidth);
        const uint64_t slicePitch = rowPitch * layout.RowCount(mipHeight);
        const uint64_t expectedBytes = slicePitch * mipDepth * source.layerCount;
        if (source.rowPitch != rowPitch || source.slicePitch != slicePitch || source.byteSize != expectedBytes)
            throw std::invalid_argument("RHI texture upload subresource byte layout is not tightly packed");

        for (uint32_t layer = 0; layer < source.layerCount; ++layer) {
            const size_t coverage = static_cast<size_t>(source.mipLevel) * arrayLayers + source.baseLayer + layer;
            if (uploaded[coverage])
                throw std::invalid_argument("RHI texture upload contains overlapping subresources");
            uploaded[coverage] = true;
        }

        if (source.byteSize > std::numeric_limits<VkDeviceSize>::max() - residentBytes)
            throw std::overflow_error("RHI texture upload byte size overflow");
        residentBytes += source.byteSize;
    }
    if (std::find(uploaded.begin(), uploaded.end(), false) != uploaded.end())
        throw std::invalid_argument("RHI texture upload must initialize every mip and array layer");

    auto ticket = std::make_shared<TextureUploadTicket>();
    ticket->m_manager = this;
    ticket->m_residentBytes = residentBytes;
    ticket->m_description = request.texture;
    ticket->m_view = request.view;
    ticket->m_sampler = request.sampler;
    if (m_asyncTransfer && m_asyncTransfer->GetTimelineSemaphore() != VK_NULL_HANDLE) {
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &count, nullptr);
        std::vector<VkQueueFamilyProperties> properties(count);
        vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &count, properties.data());
        const auto granularity = properties.at(m_asyncTransfer->GetQueueFamily()).minImageTransferGranularity;
        // This cursor can split any row/slice. A coarser transfer family cannot
        // execute that contract; choose the asynchronous graphics queue up front.
        ticket->m_useTransfer = granularity.width == 1 && granularity.height == 1 && granularity.depth == 1;
    }
    ticket->m_layout.assign(request.subresources, request.subresources + request.subresourceCount);
    for (auto &source : ticket->m_layout)
        source.data = nullptr; // The caller owns CPU input between appends.
    return ticket;
}

bool VkResourceManager::PrepareTextureUploadDestination(TextureUploadTicket &ticket, bool managed)
{
    if (ticket.m_texture)
        return true;
    const auto candidate = m_rhiDevice->CreateUploadTexture(ticket.m_description, managed);
    if (!candidate)
        return false;
    const auto texture = *candidate;
    if (!texture.IsValid())
        throw std::runtime_error("failed to allocate RHI texture");
    rhi::TextureViewDesc viewDesc = ticket.m_view;
    viewDesc.texture = texture;
    const rhi::TextureViewHandle view = m_rhiDevice->CreateTextureView(viewDesc);
    if (!view.IsValid()) {
        m_rhiDevice->Release(texture);
        throw std::runtime_error("failed to allocate RHI texture view");
    }
    const rhi::SamplerHandle sampler = m_rhiDevice->CreateSampler(ticket.m_sampler);
    if (!sampler.IsValid()) {
        m_rhiDevice->Release(view);
        m_rhiDevice->Release(texture);
        throw std::runtime_error("failed to allocate RHI texture sampler");
    }
    ticket.m_texture = std::make_shared<rhi::TextureResource>(
        *m_rhiDevice, texture, view, sampler, ticket.m_residentBytes, ticket.m_description.format, viewDesc);
    return true;
}

bool VkResourceManager::TryAppendTextureUpload(const std::shared_ptr<TextureUploadTicket> &ticket, const void *data,
                                               VkDeviceSize bytes, FrameWorkOwner *owner)
{
    if (!data)
        throw std::invalid_argument("GPU texture upload requires source data");
    return AppendTextureUpload(
        ticket, bytes,
        [data](uint32_t, VkDeviceSize, VkDeviceSize offset) { return static_cast<const uint8_t *>(data) + offset; },
        owner);
}

struct VkResourceManager::TextureUploadBlock
{
    VkDeviceSize bytes;
    double nanosecondsPerByte = 0.0;

    uint64_t Estimate(VkDeviceSize count) const
    {
        return nanosecondsPerByte > 0.0
                   ? (std::max<uint64_t>)(1, static_cast<uint64_t>(nanosecondsPerByte * static_cast<double>(count)))
                   : 0;
    }
};

VkResourceManager::TextureUploadBlock VkResourceManager::SelectTextureUploadBlock(const TextureUploadTicket &ticket,
                                                                                  VkDeviceSize bytes,
                                                                                  const FrameWorkOwner *owner) const
{
    if (!owner)
        return {bytes};
    const auto layout = rhi::GetPixelFormatLayout(ticket.m_description.format);
    const auto aligned = [unit = VkDeviceSize(layout.bytesPerBlock)](VkDeviceSize count) {
        return (std::max)(unit, count - count % unit);
    };
    TextureUploadBlock block{aligned((std::min)(bytes, owner->ResourceRemaining(FrameWorkResource::Transfer)))};
    const auto rowBytes = layout.RowBytes(ticket.m_description.width);
    // Reuse completed transfers on this uploader instead of spending a fresh
    // host-frame calibration on every small icon or newly imported texture.
    const auto profiledBytes = ticket.m_profiledBytes ? ticket.m_profiledBytes : m_textureUploadProfiledBytes;
    const auto profiledNanoseconds =
        ticket.m_profiledBytes ? ticket.m_profiledNanoseconds : m_textureUploadProfiledNanoseconds;
    if (profiledBytes >= (std::min)(block.bytes, rowBytes)) {
        block.nanosecondsPerByte = static_cast<double>(profiledNanoseconds) / static_cast<double>(profiledBytes);
        const double gpuBytes =
            static_cast<double>(owner->ResourceRemaining(FrameWorkResource::Gpu)) / block.nanosecondsPerByte;
        block.bytes = aligned(static_cast<VkDeviceSize>((std::min)(static_cast<double>(block.bytes), gpuBytes)));
    } else if (profiledBytes) {
        // A tiny first receipt only calibrates tiny uploads. Measure a row
        // when more input becomes available before scaling to large ranges.
        block.bytes = (std::min)(block.bytes, rowBytes);
    }
    if (!CanPrepareGpuWork(owner, block.Estimate(block.bytes), block.bytes, block.bytes == layout.bytesPerBlock))
        block.bytes = 0;
    return block;
}

bool VkResourceManager::CanAppendTextureUpload(const TextureUploadTicket &ticket, VkDeviceSize bytes,
                                               const FrameWorkOwner &owner) const
{
    if (ticket.m_manager != this || ticket.IsPending())
        return false;
    return ValidTextureUploadBlock(ticket, bytes, rhi::GetPixelFormatLayout(ticket.m_description.format)) &&
           SelectTextureUploadBlock(ticket, bytes, &owner).bytes != 0;
}

bool VkResourceManager::AppendTextureUpload(
    const std::shared_ptr<TextureUploadTicket> &ticket, VkDeviceSize bytes,
    const std::function<const void *(uint32_t, VkDeviceSize, VkDeviceSize)> &source, FrameWorkOwner *owner)
{
    AssertReadbackThread();
    if (!ticket || ticket->m_manager != this)
        throw std::invalid_argument("GPU texture upload ticket belongs to another resource manager");
    if (ticket->IsPending())
        return false;
    const auto layout = rhi::GetPixelFormatLayout(ticket->m_description.format);
    if (!ValidTextureUploadBlock(*ticket, bytes, layout))
        throw std::invalid_argument("GPU texture upload block must fit the remaining payload and format blocks");
    const auto block = SelectTextureUploadBlock(*ticket, bytes, owner);
    bytes = block.bytes;
    if (!bytes)
        return false;
    const auto planBlock = [&](VkDeviceSize count) {
        return PlanTextureUpload(ticket->m_layout, layout, ticket->m_submittedBytes, count);
    };
    auto plan = planBlock(bytes);
    auto staging = m_rhiDevice->AcquireStagingBuffer(plan.stagingBytes, owner != nullptr);
    while (!staging || !PrepareTextureUploadDestination(*ticket, owner != nullptr)) {
        m_rhiDevice->RecycleStagingBuffer(std::move(staging));
        if (!owner)
            throw std::runtime_error("failed to allocate RHI texture staging buffer");
        bytes = SmallerStagingBlock(bytes, plan.stagingBytes, layout.bytesPerBlock);
        if (!bytes)
            return false;
        plan = planBlock(bytes);
        staging = m_rhiDevice->AcquireStagingBuffer(plan.stagingBytes, true);
    }
    if (m_pendingTextureUploads.size() == m_pendingTextureUploads.capacity())
        m_pendingTextureUploads.reserve((std::max)(size_t{1}, m_pendingTextureUploads.capacity() * 2));
    PrepareGpuWork(ticket->m_work, owner, bytes, ticket->m_useTransfer, block.Estimate(bytes));
    for (size_t index = 0; index < plan.copies.size(); ++index) {
        const auto &copy = plan.copies[index];
        staging->CopyFrom(source(copy.sourceIndex, copy.sourceOffset, copy.inputOffset), copy.bytes,
                          plan.regions[index].bufferOffset);
    }
    const bool useTransfer = ticket->m_useTransfer;
    const bool separateLane = useTransfer && m_asyncTransfer->GetQueueFamily() != m_graphicsQueueFamily;
    const auto commandBuffer = useTransfer ? m_asyncTransfer->Begin() : BeginSingleTimeCommands();
    if (commandBuffer == VK_NULL_HANDLE)
        throw std::runtime_error("failed to begin RHI texture upload");
    BeginGpuWork(ticket->m_work, commandBuffer, "texture.upload");
    const auto image = m_rhiDevice->Resolve(ticket->m_texture->GetTexture());
    const bool first = ticket->m_submittedBytes == 0;
    const bool last = ticket->m_submittedBytes + bytes == ticket->m_residentBytes;
    const auto &description = ticket->m_description;
    const uint32_t layers = description.dimension == rhi::TextureDimension::Texture3D ? 1 : description.depthOrLayers;

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, description.mipLevels, 0, layers};
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    if (first)
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &barrier);
    vkCmdCopyBufferToImage(commandBuffer, staging->GetBuffer(), image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           static_cast<uint32_t>(plan.regions.size()), plan.regions.data());

    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = separateLane ? VK_ACCESS_TRANSFER_READ_BIT : VK_ACCESS_MEMORY_READ_BIT;
    if (last)
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             separateLane ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &barrier);

    FinishGpuWork(ticket->m_work, commandBuffer);
    if (useTransfer) {
        ticket->m_upload = m_asyncTransfer->EndAsync(commandBuffer);
        if (!ticket->m_upload.IsValid())
            throw std::runtime_error("failed to submit asynchronous RHI texture upload");
    } else {
        ticket->m_graphicsSubmission =
            EndSingleTimeCommandsAsync(commandBuffer, [staging, texture = ticket->m_texture] {});
    }
    if (last) {
        const auto producer = useTransfer ? ticket->m_upload : ticket->m_graphicsSubmission->m_submission;
        ticket->m_texture->SetUploadSubmission(producer);
        m_rhiDevice->SetUploadSubmission(ticket->m_texture->GetTexture(), producer);
    }
    ticket->m_async = true;
    ticket->m_staging = std::move(staging);
    ticket->m_submittedBytes += bytes;
    AdmitGpuWork(ticket->m_work);
    m_pendingTextureUploads.push_back(ticket);
    return true;
}

bool VkResourceManager::TryPublishTextureUpload(const std::shared_ptr<TextureUploadTicket> &ticket)
{
    if (!ticket || ticket->m_manager != this)
        throw std::invalid_argument("GPU texture upload ticket belongs to another resource manager");
    if (ticket->m_published)
        return true;
    if (ticket->m_submittedBytes != ticket->m_residentBytes)
        return false;
    PollGpuUploads();
    ticket->m_published = RecordUploadPublication(ticket->m_work, ticket->m_complete);
    return ticket->m_published;
}

} // namespace infernux::vk
