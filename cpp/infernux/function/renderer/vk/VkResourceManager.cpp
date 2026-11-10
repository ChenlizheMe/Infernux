/**
 * @file VkResourceManager.cpp
 * @brief Implementation of Vulkan resource management
 */

#include "VkResourceManager.h"
#include "AsyncTransferContext.h"
#include "DescriptorBindTrace.h"
#include "RhiVulkanTypes.h"
#include "VkDeviceContext.h"
#include "VulkanQueueManager.h"
#include "VulkanRhiDevice.h"
#include <SDL3/SDL.h>
#include <core/error/InxError.h>
#include <core/threading/FrameWorkOwner.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>

namespace infernux
{
namespace vk
{

namespace
{
using StagingAccess = VulkanRhiDevice::StagingAccess;

struct ReadbackFormatInfo
{
    uint32_t channels;
    uint32_t bytesPerPixel;
    const char *elementType;
    bool bgra = false;
};

ReadbackFormatInfo GetReadbackFormatInfo(VkFormat format)
{
    const auto bytes = rhi::GetPixelFormatLayout(rhi::FromVkFormat(format)).bytesPerBlock;
    switch (format) {
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SRGB:
        return {4, bytes, "uint8"};
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_SRGB:
        return {4, bytes, "uint8", true};
    case VK_FORMAT_R16G16B16A16_SFLOAT:
        return {4, bytes, "float16"};
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return {4, bytes, "float32"};
    case VK_FORMAT_R32_SFLOAT:
        return {1, bytes, "float32"};
    case VK_FORMAT_R32G32_UINT:
        return {2, bytes, "uint32"};
    default:
        throw std::invalid_argument("Image format is not supported by GPU readback");
    }
}

void RecordImageReadbackCopy(VkCommandBuffer commands, VkImage image, VkImageLayout layout, VkImageAspectFlags aspect,
                             VkPipelineStageFlags sourceStage, VkAccessFlags sourceAccess, VkExtent2D extent,
                             VkBuffer staging, VkPipelineStageFlags restoreStage, VkAccessFlags restoreAccess)
{
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = layout;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {aspect, 0, 1, 0, 1};
    barrier.srcAccessMask = sourceAccess;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(commands, sourceStage, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    VkBufferImageCopy region{};
    region.imageSubresource = {aspect, 0, 0, 1};
    region.imageExtent = {extent.width, extent.height, 1};
    vkCmdCopyImageToBuffer(commands, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging, 1, &region);

    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = layout;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = restoreAccess;
    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, restoreStage, 0, 0, nullptr, 0, nullptr, 1,
                         &barrier);
}

void RecordReadbackHostBarrier(VkCommandBuffer commands, VkBuffer staging, VkDeviceSize bytes)
{
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = staging;
    barrier.size = bytes;
    // Fence completion orders execution. This dependency additionally makes
    // transfer writes visible to the host, including on HOST_COHERENT memory.
    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                         &barrier, 0, nullptr);
}

VkResult WaitForFencePumpingEvents(VkDevice device, VkFence fence)
{
    constexpr uint64_t kPollTimeoutNs = 50'000'000; // 50 ms
    // A healthy submission signals in milliseconds. A fence that stays
    // unsignaled for this long means the device is lost or the driver is
    // wedged; waiting forever turns that into an unkillable shutdown hang.
    constexpr int kMaxPolls = 200; // 10 s total
    for (int poll = 0; poll < kMaxPolls; ++poll) {
        VkResult result = vkWaitForFences(device, 1, &fence, VK_TRUE, kPollTimeoutNs);
        if (result == VK_SUCCESS) {
            return result;
        }
        if (result != VK_TIMEOUT) {
            INXLOG_ERROR("VkResourceManager::EndSingleTimeCommands fence wait failed: ", result);
            return result;
        }
        SDL_PumpEvents();
    }
    INXLOG_ERROR("Fence wait failed after 10 s; GPU completion has not been proven");
    return VK_TIMEOUT;
}

} // namespace

// ============================================================================
// Constructor / Destructor / Move
// ============================================================================

VkResourceManager::~VkResourceManager()
{
    Destroy();
}

const std::shared_ptr<VkBufferHandle> &BufferUploadTicket::GetBuffer() const
{
    if (!m_published || !m_destination)
        throw std::logic_error("GPU buffer upload has not been published");
    return m_destination;
}

std::shared_ptr<rhi::BufferResource> BufferUploadTicket::GetRhiBuffer() const
{
    auto resource = m_rhiBuffer.lock();
    if (!m_published || !resource)
        throw std::logic_error("RHI buffer upload has not been published");
    return resource;
}

const std::shared_ptr<rhi::TextureResource> &TextureUploadTicket::GetTexture() const
{
    if (!m_published || !m_texture)
        throw std::logic_error("GPU texture upload has not been published");
    return m_texture;
}

const std::vector<uint8_t> &ImageReadbackTicket::GetData() const
{
    const ImageReadbackStatus status = GetStatus();
    if (status == ImageReadbackStatus::Failed)
        throw std::runtime_error(m_error);
    if (status != ImageReadbackStatus::Completed || m_output != ImageReadbackOutput::Pixels)
        throw std::logic_error("GPU image readback has not completed");
    return m_data;
}

std::string_view ImageReadbackTicket::GetMappedData() const
{
    if (GetStatus() != ImageReadbackStatus::Completed || m_output != ImageReadbackOutput::Mapped)
        throw std::logic_error("Mapped image readback has not completed");
    return {reinterpret_cast<const char *>(m_mappedData), m_byteSize};
}

void ImageReadbackTicket::Cancel() noexcept
{
    ImageReadbackStatus expected = ImageReadbackStatus::Pending;
    m_status.compare_exchange_strong(expected, ImageReadbackStatus::Cancelled, std::memory_order_acq_rel);
}

GraphicsImageReadbackRecorder::~GraphicsImageReadbackRecorder()
{
    Reset();
}

GraphicsImageReadbackRecorder::GraphicsImageReadbackRecorder(GraphicsImageReadbackRecorder &&other) noexcept
    : m_manager(other.m_manager), m_ticket(std::move(other.m_ticket)), m_commandBuffer(other.m_commandBuffer)
{
    other.m_manager = nullptr;
    other.m_commandBuffer = VK_NULL_HANDLE;
}

GraphicsImageReadbackRecorder &GraphicsImageReadbackRecorder::operator=(GraphicsImageReadbackRecorder &&other) noexcept
{
    if (this == &other)
        return *this;
    Reset();
    m_manager = other.m_manager;
    m_ticket = std::move(other.m_ticket);
    m_commandBuffer = other.m_commandBuffer;
    other.m_manager = nullptr;
    other.m_commandBuffer = VK_NULL_HANDLE;
    return *this;
}

std::shared_ptr<ImageReadbackTicket>
GraphicsImageReadbackRecorder::Submit(std::function<void()> releaseResources,
                                      const std::vector<SubmissionWait> &dependencies)
{
    if (!m_manager)
        throw std::logic_error("Graphics image readback recorder is no longer active");
    return m_manager->SubmitGraphicsImageReadback(*this, std::move(releaseResources), dependencies);
}

void GraphicsImageReadbackRecorder::Reset() noexcept
{
    if (m_manager)
        m_manager->AbandonGraphicsImageReadback(*this);
}

// ============================================================================
// Initialization
// ============================================================================

bool VkResourceManager::Initialize(VkDeviceContext &context, VulkanQueueManager *queueManager)
{
    m_context = &context;
    m_ownerThread = std::this_thread::get_id();
    m_device = context.GetDevice();
    m_physicalDevice = context.GetPhysicalDevice();
    m_vmaAllocator = context.GetVmaAllocator();
    m_graphicsQueue = context.GetGraphicsQueue();
    m_rhiDevice = &context.GetRhiDevice();
    m_deviceLifetime = m_rhiDevice->GetLifetime();
    m_queueManager = queueManager;
    m_graphicsQueueFamily = context.GetQueueIndices().graphicsFamily.value();
    m_computeQueueFamily = context.GetQueueIndices().computeFamily.value();
    m_transferQueueFamily = context.GetQueueIndices().transferFamily.value();

    // Create command pool
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = context.GetQueueIndices().graphicsFamily.value();
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPool) != VK_SUCCESS) {
        INXLOG_ERROR("Failed to create command pool");
        return false;
    }
    INXLOG_INFO("VkResourceManager initialized");
    return true;
}

void VkResourceManager::Destroy() noexcept
{
    if (m_device == VK_NULL_HANDLE) {
        return;
    }

    if (m_readbackOwner)
        m_readbackOwner->RemoveSource(m_readbackSource);
    if (!m_skipWaitIdle) {
        const VkResult result = vkDeviceWaitIdle(m_device);
        if (result != VK_SUCCESS) {
            INXLOG_ERROR("VkResourceManager shutdown idle wait failed with VkResult ", result);
            if (result != VK_ERROR_DEVICE_LOST)
                std::terminate();
        }
    }
    DrainBufferUploads();
    DrainAsyncGraphicsSubmissions();
    DrainImageReadbacks();
    // Device idle or loss proves resources are no longer executing, even if a
    // failed fence poll could not publish successful completion to a ticket.
    for (const auto &ticket : m_pendingBufferUploads) {
        ticket->m_work.timestamps.reset();
        ticket->m_work.owner = nullptr;
    }
    m_pendingBufferUploads.clear();
    for (const auto &ticket : m_pendingTextureUploads) {
        ticket->m_work.timestamps.reset();
        ticket->m_work.owner = nullptr;
    }
    m_pendingTextureUploads.clear();
    m_graphicsWorkQueries.clear();
    m_transferWorkQueries.clear();
    for (const auto &ticket : m_pendingAsyncGraphicsSubmissions) {
        ticket->m_work.timestamps.reset();
        ticket->m_work.owner = nullptr;
    }
    m_pendingAsyncGraphicsSubmissions.clear();
    for (const auto &ticket : m_pendingImageReadbacks) {
        ticket->Cancel();
        RetireImageReadback(*ticket);
    }
    m_pendingImageReadbacks.clear();
    m_readbackOwner = nullptr;
    m_readbackTimingBytes = m_readbackTimingNanoseconds = 0;

    // Destroy samplers
    if (m_linearSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_linearSampler, nullptr);
        m_linearSampler = VK_NULL_HANDLE;
    }

    if (m_nearestSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_nearestSampler, nullptr);
        m_nearestSampler = VK_NULL_HANDLE;
    }

    // Tear down the single-time-command pools.
    // m_allSingleTimeCmdBuffers is owned by m_commandPool — destroying the
    // pool below frees them implicitly, so we only need to clear the lists.
    std::vector<rhi::SubmissionSerial> abandonedCompletionEpochs;
    {
        std::lock_guard<std::mutex> guard(m_singleTimeMutex);
        abandonedCompletionEpochs.reserve(m_singleTimeCompletionEpochs.size());
        for (const auto &[commandBuffer, epoch] : m_singleTimeCompletionEpochs) {
            (void)commandBuffer;
            abandonedCompletionEpochs.push_back(epoch);
        }
        m_singleTimeCompletionEpochs.clear();
        for (VkFence fence : m_allSingleTimeFences) {
            if (fence != VK_NULL_HANDLE) {
                vkDestroyFence(m_device, fence, nullptr);
            }
        }
        m_allSingleTimeFences.clear();
        m_freeSingleTimeFences.clear();
        m_allSingleTimeCmdBuffers.clear();
        m_freeSingleTimeCmdBuffers.clear();
    }
    for (const auto epoch : abandonedCompletionEpochs) {
        if (m_queueManager)
            m_queueManager->CompleteCompletionEpoch(epoch);
    }
    if (!abandonedCompletionEpochs.empty())
        vkdebug::ClearDescriptorRecordingContext();

    // Destroy command pool
    if (m_commandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(m_device, m_commandPool, nullptr);
        m_commandPool = VK_NULL_HANDLE;
    }

    m_device = VK_NULL_HANDLE;
    m_context = nullptr;
    m_physicalDevice = VK_NULL_HANDLE;
    m_graphicsQueue = VK_NULL_HANDLE;
    m_rhiDevice = nullptr;
    m_deviceLifetime.reset();
    m_queueManager = nullptr;
    m_asyncTransfer = nullptr;
}

// ============================================================================
// Buffer Management
// ============================================================================

std::shared_ptr<BufferUploadTicket> VkResourceManager::BeginBufferUpload(const rhi::BufferUploadRequest &request)
{
    if (!request.data)
        throw std::invalid_argument("GPU buffer upload requires source data");
    auto ticket = CreateBufferUpload(request.byteSize, request.usage, request.queueAccess);
    // Whole uploads still consume the borrowed pointer before returning. The
    // incremental entry point instead borrows only each explicitly supplied block.
    (void)TryAppendBufferUpload(ticket, request.data, request.byteSize);
    return ticket;
}

std::shared_ptr<BufferUploadTicket>
VkResourceManager::CreateBufferUpload(VkDeviceSize size, rhi::BufferUsageFlags usage, rhi::QueueAccessFlags queueAccess)
{
    AssertReadbackThread();
    if (m_device == VK_NULL_HANDLE || m_vmaAllocator == VK_NULL_HANDLE)
        throw std::logic_error("GPU buffer upload requires an initialized resource manager");
    if (!size)
        throw std::invalid_argument("GPU buffer upload requires a non-empty destination");
    const auto finalUsage = rhi::ToVkBufferUsage(usage);
    if (!finalUsage)
        throw std::invalid_argument("GPU buffer upload has no supported destination usage");
    constexpr auto validQueueAccess =
        rhi::QueueAccessFlags::Graphics | rhi::QueueAccessFlags::Compute | rhi::QueueAccessFlags::Transfer;
    if (queueAccess == rhi::QueueAccessFlags::None ||
        (static_cast<uint8_t>(queueAccess) & ~static_cast<uint8_t>(validQueueAccess)) != 0)
        throw std::invalid_argument("GPU buffer upload requires valid consumer queue access");
    auto ticket = std::make_shared<BufferUploadTicket>();
    ticket->m_manager = this;
    ticket->m_size = size;
    auto &queueFamilies = ticket->m_queueFamilies;
    const auto appendFamily = [&](uint32_t family) {
        if (std::find(queueFamilies.begin(), queueFamilies.end(), family) == queueFamilies.end())
            queueFamilies.push_back(family);
    };
    // An alias queue is still an asynchronous submission path. A timeline
    // permits early dependency publication; it does not determine nonblocking IO.
    appendFamily(m_asyncTransfer ? m_asyncTransfer->GetQueueFamily() : m_graphicsQueueFamily);
    if (rhi::HasQueueAccess(queueAccess, rhi::QueueAccessFlags::Graphics))
        appendFamily(m_graphicsQueueFamily);
    if (rhi::HasQueueAccess(queueAccess, rhi::QueueAccessFlags::Compute))
        appendFamily(m_computeQueueFamily);
    if (rhi::HasQueueAccess(queueAccess, rhi::QueueAccessFlags::Transfer))
        appendFamily(m_transferQueueFamily);
    ticket->m_destinationUsage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | finalUsage;
    ticket->m_concurrentQueueSharing = queueFamilies.size() > 1;
    return ticket;
}

bool VkResourceManager::PrepareBufferUploadDestination(BufferUploadTicket &ticket, bool managed)
{
    if (ticket.m_destination)
        return true;
    constexpr auto properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    if (managed) {
        auto destination = std::make_shared<VkBufferHandle>();
        if (!destination->TryCreateManaged(
                m_vmaAllocator, m_device, ticket.m_size, ticket.m_destinationUsage, properties, ticket.m_queueFamilies,
                m_deviceLifetime,
                [this](uint32_t type, VkDeviceSize bytes) { return m_rhiDevice->AdmitAllocation(type, bytes); }))
            return false;
        ticket.m_destination = std::move(destination);
    } else {
        ticket.m_destination =
            CreateBufferInternal(ticket.m_size, ticket.m_destinationUsage, properties, ticket.m_queueFamilies);
        if (!ticket.m_destination)
            throw std::runtime_error("failed to allocate GPU upload destination buffer");
    }
    return true;
}

bool VkResourceManager::PrepareBufferUploadDestinations(BufferUploadTicket &ticket,
                                                        std::initializer_list<BufferUploadTicket *> pairedDestinations,
                                                        bool managed)
{
    if (ticket.m_destination)
        return true;
    if (!PrepareBufferUploadDestination(ticket, managed))
        return false;
    for (auto it = pairedDestinations.begin(); it != pairedDestinations.end(); ++it) {
        if (!*it || PrepareBufferUploadDestination(**it, managed))
            continue;
        // Every paired ticket is private and belongs to this upload bundle.
        // Avoid retaining a partial reservation under heap pressure.
        ticket.m_destination.reset();
        for (auto earlier = pairedDestinations.begin(); earlier != it; ++earlier)
            if (*earlier)
                (*earlier)->m_destination.reset();
        return false;
    }
    return true;
}

VkDeviceSize VkResourceManager::GetUploadStagingBytes(VkDeviceSize bytes) noexcept
{
    return rhi::UploadStagingClass(bytes);
}

bool VkResourceManager::TryAppendBufferUpload(const std::shared_ptr<BufferUploadTicket> &ticket, const void *data,
                                              VkDeviceSize bytes, FrameWorkOwner *owner, bool indivisible,
                                              VkDeviceSize splitUnit,
                                              std::initializer_list<BufferUploadTicket *> pairedDestinations)
{
    return AppendBufferUpload(ticket, data, bytes, owner, indivisible, splitUnit, false, pairedDestinations);
}

bool VkResourceManager::TryAppendIndex16Upload(const std::shared_ptr<BufferUploadTicket> &ticket,
                                               const uint32_t *indices, VkDeviceSize bytes, FrameWorkOwner *owner)
{
    constexpr auto unit = sizeof(uint16_t);
    return AppendBufferUpload(ticket, indices, bytes, owner, bytes == unit, unit, true, {});
}

VkDeviceSize VkResourceManager::PlanBufferUploadBlock(const BufferUploadTicket &ticket, const FrameWorkOwner &owner,
                                                      VkDeviceSize unit) const
{
    if (ticket.IsPending() || ticket.m_submittedBytes == ticket.m_size)
        return 0;
    auto bytes = (std::min)({ticket.m_size - ticket.m_submittedBytes, rhi::MaxPooledUploadStagingClass,
                             owner.ResourceRemaining(FrameWorkResource::Transfer)});
    bytes -= bytes % unit;
    if (!bytes)
        bytes = unit;
    return CanPrepareGpuWork(&owner, ticket.m_work.lastExecutionNanoseconds, bytes, bytes == unit) ? bytes : 0;
}

bool VkResourceManager::AppendBufferUpload(const std::shared_ptr<BufferUploadTicket> &ticket, const void *data,
                                           VkDeviceSize bytes, FrameWorkOwner *owner, bool indivisible,
                                           VkDeviceSize splitUnit, bool packIndex16,
                                           std::initializer_list<BufferUploadTicket *> pairedDestinations)
{
    AssertReadbackThread();
    if (!ticket || ticket->m_manager != this)
        throw std::invalid_argument("GPU buffer upload ticket belongs to another resource manager");
    if (!data || !bytes || bytes > ticket->m_size - ticket->m_submittedBytes)
        throw std::invalid_argument("GPU buffer upload block exceeds its remaining destination");
    const auto gpuNanoseconds = ticket->m_work.lastExecutionNanoseconds;
    if (ticket->IsPending() || !CanPrepareGpuWork(owner, gpuNanoseconds, bytes, indivisible))
        return false;
    auto staging = m_rhiDevice->AcquireStagingBuffer(bytes, owner != nullptr);
    while (!staging || !PrepareBufferUploadDestinations(*ticket, pairedDestinations, owner != nullptr)) {
        m_rhiDevice->RecycleStagingBuffer(std::move(staging));
        if (!owner)
            throw std::runtime_error("failed to allocate GPU upload staging buffer");
        if (!splitUnit)
            return false;
        bytes = SmallerStagingBlock(bytes, bytes, splitUnit);
        if (!bytes)
            return false;
        staging = m_rhiDevice->AcquireStagingBuffer(bytes, true);
    }
    if (m_pendingBufferUploads.size() == m_pendingBufferUploads.capacity())
        m_pendingBufferUploads.reserve((std::max)(size_t{1}, m_pendingBufferUploads.capacity() * 2));
    PrepareGpuWork(ticket->m_work, owner, bytes, bool(m_asyncTransfer), gpuNanoseconds);
    if (packIndex16) {
        auto *mapped = static_cast<uint16_t *>(staging->Map());
        if (!mapped)
            throw std::runtime_error("failed to map GPU index upload staging buffer");
        const auto *indices = static_cast<const uint32_t *>(data);
        std::transform(indices, indices + bytes / sizeof(uint16_t), mapped,
                       [](uint32_t index) { return static_cast<uint16_t>(index); });
        staging->Unmap();
    } else {
        staging->CopyFrom(data, bytes, 0);
    }
    const auto commands = m_asyncTransfer ? m_asyncTransfer->Begin() : BeginSingleTimeCommands();
    if (commands == VK_NULL_HANDLE)
        throw std::runtime_error("failed to begin GPU buffer upload");
    BeginGpuWork(ticket->m_work, commands, "buffer.upload");
    const VkBufferCopy copy{0, ticket->m_submittedBytes, bytes};
    vkCmdCopyBuffer(commands, staging->GetBuffer(), ticket->m_destination->GetBuffer(), 1, &copy);
    if (ticket->m_submittedBytes + bytes == ticket->m_size) {
        const bool transferOnly = m_asyncTransfer && m_asyncTransfer->GetQueueFamily() != m_graphicsQueueFamily;
        VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = transferOnly ? VK_ACCESS_TRANSFER_READ_BIT : VK_ACCESS_MEMORY_READ_BIT;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = ticket->m_destination->GetBuffer();
        barrier.size = ticket->m_size;
        vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             transferOnly ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                             nullptr, 1, &barrier, 0, nullptr);
    }
    FinishGpuWork(ticket->m_work, commands);
    if (m_asyncTransfer) {
        const auto upload = m_asyncTransfer->EndAsync(commands);
        if (!upload.IsValid())
            throw std::runtime_error("failed to submit GPU buffer upload");
        ticket->m_upload = upload;
        ticket->m_destination->m_uploadSubmission = upload;
    } else {
        ticket->m_graphicsSubmission =
            EndSingleTimeCommandsAsync(commands, [source = staging, destination = ticket->m_destination]() mutable {
                source.reset();
                destination.reset();
            });
        ticket->m_destination->m_uploadSubmission = ticket->m_graphicsSubmission->m_submission;
    }
    ticket->m_staging = std::move(staging);
    ticket->m_submittedBytes += bytes;
    ticket->m_async = true;
    AdmitGpuWork(ticket->m_work);
    m_pendingBufferUploads.push_back(ticket);
    ++m_bufferUploadSubmissionCount;
    return true;
}

bool VkResourceManager::TryPublishBufferUpload(const std::shared_ptr<BufferUploadTicket> &ticket)
{
    if (!ticket || ticket->m_manager != this)
        throw std::invalid_argument("GPU buffer upload ticket belongs to another resource manager");
    if (ticket->m_published)
        return true;
    if (ticket->m_submittedBytes != ticket->m_size)
        return false;
    PollGpuUploads();
    ticket->m_published = RecordUploadPublication(ticket->m_work, ticket->m_complete);
    return ticket->m_published;
}

bool VkResourceManager::RecordUploadPublication(const GpuWorkReceipt &work, bool complete)
{
    const bool publishInFlight = work.useTransfer && IsUploadTimelineEnabled();
    if (publishInFlight)
        ++m_timelineUploadPublicationCount;
    return complete || publishInFlight;
}

std::shared_ptr<rhi::BufferResource>
VkResourceManager::GetPublishedRhiBuffer(const std::shared_ptr<BufferUploadTicket> &ticket)
{
    if (!ticket || ticket->m_manager != this)
        throw std::invalid_argument("GPU buffer upload ticket belongs to another resource manager");
    if (!ticket->m_published || !ticket->m_destination)
        throw std::logic_error("GPU buffer upload has not been published");
    auto resource = ticket->m_rhiBuffer.lock();
    if (!resource) {
        if (!m_rhiDevice)
            throw std::logic_error("GPU buffer upload has no RHI device");
        const auto handle = m_rhiDevice->RegisterBuffer(ticket->m_destination->GetBuffer(), ticket->m_size,
                                                        ticket->m_concurrentQueueSharing);
        if (!handle.IsValid())
            throw std::runtime_error("failed to register uploaded GPU buffer with the RHI device");
        m_rhiDevice->SetUploadSubmission(handle, ticket->m_destination->GetUploadSubmission());
        resource = std::make_shared<rhi::BufferResource>(*m_rhiDevice, handle, ticket->m_size, ticket->m_destination);
        ticket->m_rhiBuffer = resource;
    }
    return resource;
}

void VkResourceManager::DrainBufferUploads() noexcept
{
    DrainAsyncGraphicsSubmissions();
    for (const auto &ticket : m_pendingBufferUploads) {
        if (!ticket || !ticket->IsPending())
            continue;
        try {
            if (ticket->m_upload.IsValid()) {
                if (!m_asyncTransfer)
                    continue;
                m_asyncTransfer->Wait(ticket->m_upload);
                ticket->m_upload = {};
            }
            if (ticket->m_graphicsSubmission && !ticket->m_graphicsSubmission->IsComplete())
                continue;
            ticket->m_graphicsSubmission.reset();
            RetireBufferUpload(*ticket);
        } catch (const std::exception &error) {
            INXLOG_ERROR("Failed while draining a pending GPU buffer upload: ", error.what());
        }
    }
    m_pendingBufferUploads.erase(std::remove_if(m_pendingBufferUploads.begin(), m_pendingBufferUploads.end(),
                                                [](const auto &ticket) { return !ticket || !ticket->IsPending(); }),
                                 m_pendingBufferUploads.end());

    for (const auto &ticket : m_pendingTextureUploads) {
        if (!ticket || !ticket->IsPending())
            continue;
        try {
            if (ticket->m_upload.IsValid()) {
                if (!m_asyncTransfer)
                    continue;
                m_asyncTransfer->Wait(ticket->m_upload);
                ticket->m_upload = {};
            }
            if (ticket->m_graphicsSubmission && !ticket->m_graphicsSubmission->IsComplete())
                continue;
            ticket->m_graphicsSubmission.reset();
            RetireTextureUpload(*ticket);
        } catch (const std::exception &error) {
            INXLOG_ERROR("Failed while draining a pending GPU texture upload: ", error.what());
        }
    }
    m_pendingTextureUploads.erase(std::remove_if(m_pendingTextureUploads.begin(), m_pendingTextureUploads.end(),
                                                 [](const auto &ticket) { return !ticket || !ticket->IsPending(); }),
                                  m_pendingTextureUploads.end());
}

void VkResourceManager::PollGpuUploads()
{
    PollAsyncGraphicsSubmissions();
    size_t bufferWriteIndex = 0;
    for (size_t index = 0; index < m_pendingBufferUploads.size(); ++index) {
        auto &ticket = m_pendingBufferUploads[index];
        const bool completed =
            ticket && (ticket->m_graphicsSubmission ? ticket->m_graphicsSubmission->IsComplete()
                                                    : m_asyncTransfer && m_asyncTransfer->IsComplete(ticket->m_upload));
        if (completed) {
            ticket->m_upload = {};
            ticket->m_graphicsSubmission.reset();
            RetireBufferUpload(*ticket);
            continue;
        }
        if (bufferWriteIndex != index)
            m_pendingBufferUploads[bufferWriteIndex] = std::move(ticket);
        ++bufferWriteIndex;
    }
    m_pendingBufferUploads.resize(bufferWriteIndex);

    size_t textureWriteIndex = 0;
    for (size_t index = 0; index < m_pendingTextureUploads.size(); ++index) {
        auto &ticket = m_pendingTextureUploads[index];
        const bool completed =
            ticket && (ticket->m_graphicsSubmission ? ticket->m_graphicsSubmission->IsComplete()
                                                    : m_asyncTransfer && m_asyncTransfer->IsComplete(ticket->m_upload));
        if (completed) {
            ticket->m_upload = {};
            ticket->m_graphicsSubmission.reset();
            RetireTextureUpload(*ticket);
            continue;
        }
        if (textureWriteIndex != index)
            m_pendingTextureUploads[textureWriteIndex] = std::move(ticket);
        ++textureWriteIndex;
    }
    m_pendingTextureUploads.resize(textureWriteIndex);
}

void VkResourceManager::SetAsyncTransferContext(AsyncTransferContext *transfer)
{
    m_asyncTransfer = transfer;
    m_transferTimestampsSupported =
        transfer && m_context->QueryTimestampSupport(transfer->GetQueueFamily()).validBits > 0;
}

bool VkResourceManager::CanPrepareGpuWork(const FrameWorkOwner *owner, uint64_t gpuNanoseconds, VkDeviceSize bytes,
                                          bool indivisible)
{
    return !owner || (owner->CanAdmitResource(FrameWorkResource::Gpu, gpuNanoseconds, true) &&
                      (!bytes || owner->CanAdmitResource(FrameWorkResource::Transfer, bytes, indivisible)));
}

void VkResourceManager::PrepareGpuWork(GpuWorkReceipt &work, FrameWorkOwner *owner, VkDeviceSize bytes,
                                       bool useTransfer, uint64_t gpuNanoseconds)
{
    work.owner = owner;
    work.transferBytes = bytes;
    work.useTransfer = useTransfer;
    work.predictedExecutionNanoseconds = gpuNanoseconds;
    const bool timingSupported =
        useTransfer ? m_transferTimestampsSupported : m_context->GetCapabilities().timestampQueries.supported;
    // Allocate queries before opening the command buffer. A supported device
    // failing allocation is an execution error, not an alternate path.
    if (owner && timingSupported) {
        auto &pool = useTransfer ? m_transferWorkQueries : m_graphicsWorkQueries;
        if (pool.empty()) {
            work.timestamps = std::make_unique<GpuTimestampQueries>();
            if (!work.timestamps->Initialize(*m_context, 1, 1,
                                             useTransfer ? m_asyncTransfer->GetQueueFamily() : m_graphicsQueueFamily))
                throw std::runtime_error("Failed to create managed GPU timestamp queries");
        } else {
            work.timestamps = std::move(pool.back());
            pool.pop_back();
        }
    }
}

void VkResourceManager::BeginGpuWork(GpuWorkReceipt &work, VkCommandBuffer commands, const char *name)
{
    if (!work.timestamps)
        return;
    work.timestamps->BeginFrame(commands, 0, false);
    work.region = work.timestamps->BeginRegion(commands, name, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
}

void VkResourceManager::FinishGpuWork(GpuWorkReceipt &work, VkCommandBuffer commands)
{
    if (!work.timestamps)
        return;
    work.timestamps->EndRegion(commands, work.region, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    work.timestamps->FinishFrame(commands, 0);
}

void VkResourceManager::AdmitGpuWork(GpuWorkReceipt &work)
{
    if (!work.owner)
        return;
    if (work.owner->IsActive()) {
        work.frame = work.owner->Opportunity();
        work.gpuGrant = work.owner->RecordAcceptedResource(FrameWorkResource::Gpu, work.predictedExecutionNanoseconds);
        work.transferGrant = work.transferBytes
                                 ? work.owner->RecordAcceptedResource(FrameWorkResource::Transfer, work.transferBytes)
                                 : 0;
    }
    if (work.timestamps)
        work.timestamps->MarkSubmitted(0);
}

uint64_t VkResourceManager::RetireGpuWork(GpuWorkReceipt &work)
{
    // The transfer/graphics fence has completed. Collect before recycling the
    // query pool or releasing a ticket abandoned by its original consumer.
    uint64_t elapsed = 0;
    if (work.timestamps) {
        const auto measured = work.timestamps->CollectCompletedFrame(0);
        if (measured && measured->sampleCount == 1) {
            elapsed = static_cast<uint64_t>(std::ceil(measured->samples[0].milliseconds * 1.0e6));
            work.executionNanoseconds += elapsed;
            work.lastExecutionNanoseconds = elapsed;
            work.hasExecutionTiming = true;
            work.owner->ObserveResource(FrameWorkResource::Gpu, elapsed, work.frame, work.gpuGrant);
        }
        // Completion and measurement are distinct. A missing timing receipt
        // leaves the GPU grant unknown; it cannot prevent real IO retirement.
        auto &pool = work.useTransfer ? m_transferWorkQueries : m_graphicsWorkQueries;
        pool.push_back(std::move(work.timestamps));
    }
    if (work.owner && work.transferBytes)
        work.owner->ObserveResource(FrameWorkResource::Transfer, work.transferBytes, work.frame, work.transferGrant);
    work.owner = nullptr;
    work.frame = work.gpuGrant = work.transferGrant = work.transferBytes = 0;
    return elapsed;
}

void VkResourceManager::RetireBufferUpload(BufferUploadTicket &ticket)
{
    RetireGpuWork(ticket.m_work);
    m_rhiDevice->RecycleStagingBuffer(std::move(ticket.m_staging));
    ticket.m_completedBytes = ticket.m_submittedBytes;
    ticket.m_complete = ticket.m_completedBytes == ticket.m_size;
}

void VkResourceManager::RetireTextureUpload(TextureUploadTicket &ticket)
{
    const auto elapsed = RetireGpuWork(ticket.m_work);
    const auto bytes = ticket.m_submittedBytes - ticket.m_completedBytes;
    // All receipts are charged above. Keep the bandwidth estimate anchored to
    // a full-size sample, so a short tail does not become the transfer rate.
    if (elapsed && bytes >= ticket.m_profiledBytes) {
        ticket.m_profiledBytes = bytes;
        ticket.m_profiledNanoseconds = elapsed;
    }
    m_rhiDevice->RecycleStagingBuffer(std::move(ticket.m_staging));
    ticket.m_completedBytes = ticket.m_submittedBytes;
    ticket.m_complete = ticket.m_completedBytes == ticket.m_residentBytes;
}

SubmissionWait VkResourceManager::UploadReadDependency(rhi::SubmissionTicket upload, rhi::QueueRole consumer,
                                                       VkPipelineStageFlags stages) const
{
    if (!upload.IsValid())
        return {};
    // The uploader records visibility for consumers on its own lane.
    const auto point = m_queueManager->DependencyFor(upload, consumer);
    return {point.semaphore, point.value, stages, consumer};
}

std::shared_ptr<ImageReadbackTicket>
VkResourceManager::BeginImageReadback(VkImage image, VkImageLayout layout, VkImageAspectFlags aspect,
                                      VkPipelineStageFlags sourceStage, VkAccessFlags sourceAccess, uint32_t width,
                                      uint32_t height, VkFormat format, std::function<void()> releaseResources,
                                      rhi::SubmissionTicket upload, ImageReadbackOutput output)
{
    AssertReadbackThread();
    if (image == VK_NULL_HANDLE || width == 0 || height == 0)
        throw std::invalid_argument("GPU image readback requires a live image and non-zero dimensions");

    std::vector<SubmissionWait> dependencies;
    const auto wait = UploadReadDependency(upload, rhi::QueueRole::Graphics, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    if (wait.timeline != VK_NULL_HANDLE)
        dependencies.push_back(wait);
    auto recorder = BeginGraphicsImageReadback(width, height, format, output);
    RecordImageReadbackCopy(recorder.GetCommandBuffer(), image, layout, aspect, sourceStage, sourceAccess,
                            {width, height}, recorder.GetStagingBuffer(), sourceStage, sourceAccess);

    return recorder.Submit(std::move(releaseResources), dependencies);
}

std::shared_ptr<ImageReadbackTicket> VkResourceManager::RecordFrameImageReadback(
    VkCommandBuffer commandBuffer, VkImage image, VkImageLayout layout, VkImageAspectFlags aspect,
    VkPipelineStageFlags sourceStage, VkAccessFlags sourceAccess, uint32_t width, uint32_t height, VkFormat format,
    rhi::SubmissionSerial completionEpoch, ImageReadbackOutput output)
{
    AssertReadbackThread();
    if (commandBuffer == VK_NULL_HANDLE || image == VK_NULL_HANDLE || width == 0 || height == 0 ||
        completionEpoch == rhi::InvalidSubmissionSerial)
        throw std::invalid_argument("Frame image readback requires a live frame submission and image");

    ReserveReadbackSlot();
    auto ticket = CreateImageReadback(width, height, format, output);
    ticket->m_frameCompletionEpoch = completionEpoch;
    BeginGpuWork(ticket->m_work, commandBuffer, "Frame image readback");
    RecordImageReadbackCopy(commandBuffer, image, layout, aspect, sourceStage, sourceAccess, {width, height},
                            ticket->m_staging->GetBuffer(), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
    RecordReadbackHostBarrier(commandBuffer, ticket->m_staging->GetBuffer(), ticket->m_byteSize);
    FinishGpuWork(ticket->m_work, commandBuffer);
    QueueImageReadback(ticket);
    return ticket;
}

std::shared_ptr<ImageReadbackTicket> VkResourceManager::CreateImageReadback(uint32_t width, uint32_t height,
                                                                            VkFormat format, ImageReadbackOutput output)
{
    if (width == 0 || height == 0)
        throw std::invalid_argument("Image readback requires non-zero dimensions");

    const ReadbackFormatInfo formatInfo = GetReadbackFormatInfo(format);
    const uint64_t pixelCount = static_cast<uint64_t>(width) * height;
    if (pixelCount > std::numeric_limits<size_t>::max() / formatInfo.bytesPerPixel)
        throw std::overflow_error("Image readback byte size overflow");
    const size_t bytes = static_cast<size_t>(pixelCount * formatInfo.bytesPerPixel);
    if (output == ImageReadbackOutput::Pixels && m_readbackOwner && bytes > m_readbackOwner->CpuStorageLimit())
        throw std::runtime_error("Image readback exceeds shared CPU storage capacity");

    auto ticket = std::make_shared<ImageReadbackTicket>();
    ticket->m_width = width;
    ticket->m_height = height;
    ticket->m_channelCount = formatInfo.channels;
    ticket->m_elementType = formatInfo.elementType;
    ticket->m_bgra = formatInfo.bgra;
    ticket->m_byteSize = bytes;
    ticket->m_output = output;
    const auto prediction = bytes == m_readbackTimingBytes ? m_readbackTimingNanoseconds : 0;
    PrepareGpuWork(ticket->m_work, m_readbackOwner, bytes, false, prediction);
    ticket->m_staging = m_rhiDevice->AcquireStagingBuffer(bytes, true, StagingAccess::Readback);
    if (!ticket->m_staging)
        throw std::runtime_error("Image readback cannot reserve staging in the shared Vulkan heap budget");
    return ticket;
}

GraphicsImageReadbackRecorder VkResourceManager::BeginGraphicsImageReadback(uint32_t width, uint32_t height,
                                                                            VkFormat format, ImageReadbackOutput output)
{
    AssertReadbackThread();
    auto ticket = CreateImageReadback(width, height, format, output);
    VkCommandBuffer commandBuffer = BeginSingleTimeCommands();
    if (commandBuffer == VK_NULL_HANDLE) {
        m_rhiDevice->RecycleStagingBuffer(std::move(ticket->m_staging), StagingAccess::Readback);
        throw std::runtime_error("Failed to begin graphics image readback command buffer");
    }
    BeginGpuWork(ticket->m_work, commandBuffer, "Graphics image readback");

    GraphicsImageReadbackRecorder recorder;
    recorder.m_manager = this;
    recorder.m_ticket = std::move(ticket);
    recorder.m_commandBuffer = commandBuffer;
    return recorder;
}

void VkResourceManager::ReserveReadbackSlot()
{
    // Both recording paths reserve before a command can use the staging allocation.
    if (m_pendingImageReadbacks.size() == m_pendingImageReadbacks.capacity())
        m_pendingImageReadbacks.reserve((std::max)(size_t{1}, m_pendingImageReadbacks.capacity() * 2));
}

void VkResourceManager::QueueImageReadback(const std::shared_ptr<ImageReadbackTicket> &ticket)
{
    m_pendingImageReadbacks.push_back(ticket);
    if (m_readbackOwner)
        m_readbackOwner->Wake(m_readbackSource);
}

std::shared_ptr<ImageReadbackTicket>
VkResourceManager::SubmitGraphicsImageReadback(GraphicsImageReadbackRecorder &recorder,
                                               std::function<void()> releaseResources,
                                               const std::vector<SubmissionWait> &dependencies)
{
    AssertReadbackThread();
    if (recorder.m_manager != this || !recorder.m_ticket || recorder.m_commandBuffer == VK_NULL_HANDLE)
        throw std::invalid_argument("Graphics image readback recorder belongs to another resource manager");
    ReserveReadbackSlot();

    auto ticket = std::move(recorder.m_ticket);
    const VkCommandBuffer commandBuffer = recorder.m_commandBuffer;
    recorder.m_manager = nullptr;
    recorder.m_commandBuffer = VK_NULL_HANDLE;

    RecordReadbackHostBarrier(commandBuffer, ticket->m_staging->GetBuffer(), ticket->m_byteSize);
    try {
        ticket->m_graphicsSubmission = EndSingleTimeCommandsAsync(commandBuffer, std::move(releaseResources),
                                                                  std::move(ticket->m_work), dependencies);
    } catch (...) {
        m_rhiDevice->RecycleStagingBuffer(std::move(ticket->m_staging), StagingAccess::Readback);
        throw;
    }
    QueueImageReadback(ticket);
    return ticket;
}

void VkResourceManager::AbandonGraphicsImageReadback(GraphicsImageReadbackRecorder &recorder) noexcept
{
    auto ticket = std::move(recorder.m_ticket);
    const VkCommandBuffer commandBuffer = recorder.m_commandBuffer;
    recorder.m_manager = nullptr;
    recorder.m_commandBuffer = VK_NULL_HANDLE;

    DiscardSingleTimeCommands(commandBuffer);
    if (ticket)
        m_rhiDevice->RecycleStagingBuffer(std::move(ticket->m_staging), StagingAccess::Readback);
}

void VkResourceManager::DiscardSingleTimeCommands(VkCommandBuffer commandBuffer) noexcept
{
    if (commandBuffer != VK_NULL_HANDLE) {
        rhi::SubmissionSerial completionEpoch = rhi::InvalidSubmissionSerial;
        {
            std::lock_guard<std::mutex> guard(m_singleTimeMutex);
            const auto found = m_singleTimeCompletionEpochs.find(commandBuffer);
            if (found != m_singleTimeCompletionEpochs.end()) {
                completionEpoch = found->second;
                m_singleTimeCompletionEpochs.erase(found);
            }
        }
        if (completionEpoch != rhi::InvalidSubmissionSerial) {
            vkdebug::PopDescriptorRecordingContext();
            if (m_queueManager)
                m_queueManager->CompleteCompletionEpoch(completionEpoch);
        }
        if (vkResetCommandBuffer(commandBuffer, 0) == VK_SUCCESS) {
            std::lock_guard<std::mutex> guard(m_singleTimeMutex);
            m_freeSingleTimeCmdBuffers.push_back(commandBuffer);
        } else {
            INXLOG_ERROR("Abandoned graphics command buffer could not be reset");
        }
    }
}

void VkResourceManager::RegisterImageReadbackWork(FrameWorkOwner &owner)
{
    AssertReadbackThread();
    FrameWorkSource source;
    source.observe = [this] { PollImageReadbacks(); };
    source.refresh = source.observe;
    source.describe = [this, &owner] {
        FrameWorkFacts facts;
        facts.pending = !m_pendingImageReadbacks.empty();
        for (const auto &ticket : m_pendingImageReadbacks) {
            if (ticket->m_output != ImageReadbackOutput::Pixels || !ticket->m_gpuComplete || ticket->IsDone())
                continue;
            if (ticket->m_hostStorage || owner.CanReserveCpuStorage(ticket->m_byteSize)) {
                facts.host = {reinterpret_cast<std::uintptr_t>(ticket.get()), 0, true, false};
                break;
            }
        }
        return facts;
    };
    source.advance = [this, &owner](bool, uint64_t token) {
        constexpr size_t copyChunkBytes = 1024 * 1024;
        const auto found =
            std::find_if(m_pendingImageReadbacks.begin(), m_pendingImageReadbacks.end(), [token](const auto &ticket) {
                return reinterpret_cast<std::uintptr_t>(ticket.get()) == token;
            });
        return found != m_pendingImageReadbacks.end() && CopyImageReadback(**found, copyChunkBytes, &owner);
    };
    m_readbackSource = owner.AddSource(std::move(source));
    m_readbackOwner = &owner;
}

bool VkResourceManager::CopyImageReadback(ImageReadbackTicket &ticket, size_t maxBytes, FrameWorkOwner *owner)
{
    if (ticket.IsDone() || !ticket.m_gpuComplete)
        return false;
    auto expected = ImageReadbackStatus::Pending;
    try {
        if (!ticket.m_data.capacity()) {
            std::shared_ptr<FrameCpuStorageClaim> storage;
            if (owner) {
                storage = owner->IsAccepting() ? owner->TryReserveCpuStorage(ticket.m_byteSize)
                                               : owner->TryReserveCompletionCpuStorage(ticket.m_byteSize);
                if (!storage)
                    return false;
            }
            std::vector<uint8_t> data;
            data.reserve(ticket.m_byteSize);
            if (storage && !owner->TryGrowCpuStorage(*storage, data.capacity()))
                return false;
            ticket.m_data = std::move(data);
            ticket.m_hostStorage = std::move(storage);
        }
        const size_t bytes = (std::min)(maxBytes, ticket.m_byteSize - ticket.m_copiedBytes);
        ticket.m_data.resize(ticket.m_copiedBytes + bytes);
        const auto *mapped = static_cast<const uint8_t *>(ticket.m_staging->Map());
        if (!mapped)
            throw std::runtime_error("Failed to map GPU image readback staging buffer");
        std::memcpy(ticket.m_data.data() + ticket.m_copiedBytes, mapped + ticket.m_copiedBytes, bytes);
        ticket.m_staging->Unmap();
        ticket.m_copiedBytes += bytes;
        if (ticket.m_copiedBytes == ticket.m_byteSize)
            ticket.m_status.compare_exchange_strong(expected, ImageReadbackStatus::Completed,
                                                    std::memory_order_acq_rel);
    } catch (const std::exception &error) {
        ticket.m_error = error.what();
        ticket.m_status.compare_exchange_strong(expected, ImageReadbackStatus::Failed, std::memory_order_acq_rel);
    }
    return true;
}

void VkResourceManager::RetireImageReadback(ImageReadbackTicket &ticket) noexcept
{
    if (ticket.m_mappedData) {
        ticket.m_staging->Unmap();
        ticket.m_mappedData = nullptr;
    }
    m_rhiDevice->RecycleStagingBuffer(std::move(ticket.m_staging), StagingAccess::Readback);
    ticket.m_graphicsSubmission.reset();
    ticket.m_work = {};
    ticket.m_frameCompletionEpoch = rhi::InvalidSubmissionSerial;
    if (ticket.GetStatus() != ImageReadbackStatus::Completed) {
        std::vector<uint8_t>().swap(ticket.m_data);
        ticket.m_hostStorage.reset();
    }
}

void VkResourceManager::AcceptFrameImageReadback(ImageReadbackTicket &ticket, rhi::SubmissionSerial epoch) noexcept
{
    if (epoch == rhi::InvalidSubmissionSerial || ticket.m_frameCompletionEpoch != epoch || ticket.m_frameAccepted)
        return;
    ticket.m_frameAccepted = true;
    AdmitGpuWork(ticket.m_work);
}

void VkResourceManager::CancelFrameImageReadbacks(rhi::SubmissionSerial epoch) noexcept
{
    for (const auto &ticket : m_pendingImageReadbacks)
        if (ticket->m_frameCompletionEpoch == epoch)
            ticket->Cancel();
}

bool VkResourceManager::ObserveImageReadback(ImageReadbackTicket &ticket)
{
    if (ticket.m_gpuComplete)
        return true;

    uint64_t elapsed;
    if (ticket.m_graphicsSubmission) {
        if (!ticket.m_graphicsSubmission->IsComplete())
            return false;
        elapsed = ticket.m_graphicsSubmission->GetExecutionNanoseconds();
    } else {
        if (ticket.m_frameCompletionEpoch == rhi::InvalidSubmissionSerial || !m_queueManager ||
            !m_queueManager->IsCompletionEpochComplete(ticket.m_frameCompletionEpoch))
            return false;
        // An abandoned frame also retires its epoch so allocations can be released.
        if (!ticket.m_frameAccepted) {
            ticket.Cancel();
            return true;
        }
        elapsed = RetireGpuWork(ticket.m_work);
    }
    if (elapsed) {
        m_readbackTimingBytes = ticket.m_byteSize;
        m_readbackTimingNanoseconds = elapsed;
    }
    ticket.m_gpuComplete = true;
    return true;
}

void VkResourceManager::PollImageReadbacks()
{
    AssertReadbackThread();
    size_t writeIndex = 0;
    bool progressed = false;
    for (size_t index = 0; index < m_pendingImageReadbacks.size(); ++index) {
        auto &ticket = m_pendingImageReadbacks[index];
        const bool wasGpuComplete = ticket->m_gpuComplete;
        const bool retired = ObserveImageReadback(*ticket);
        progressed |= retired && !wasGpuComplete;
        const bool mapped = ticket->m_output == ImageReadbackOutput::Mapped;
        const bool retained = mapped && ticket.use_count() > 1;
        if (retired && mapped && !ticket->IsDone()) {
            if (retained) {
                ticket->m_mappedData = static_cast<const uint8_t *>(ticket->m_staging->Map());
                if (!ticket->m_mappedData)
                    ticket->m_error = "Failed to map GPU image readback staging buffer";
                ticket->m_status.store(ticket->m_mappedData ? ImageReadbackStatus::Completed
                                                            : ImageReadbackStatus::Failed,
                                       std::memory_order_release);
            } else {
                ticket->Cancel();
            }
        }
        if (retired && ticket->IsDone() && !(retained && ticket->m_mappedData)) {
            RetireImageReadback(*ticket);
            progressed = true;
            continue;
        }
        if (writeIndex != index)
            m_pendingImageReadbacks[writeIndex] = std::move(ticket);
        ++writeIndex;
    }
    m_pendingImageReadbacks.resize(writeIndex);
    if (progressed && m_readbackOwner)
        m_readbackOwner->ReportProgress();
}

void VkResourceManager::DrainImageReadbacks() noexcept
{
    if (std::any_of(m_pendingImageReadbacks.begin(), m_pendingImageReadbacks.end(), [](const auto &ticket) {
            return ticket && ticket->m_graphicsSubmission && !ticket->m_graphicsSubmission->IsComplete();
        }))
        DrainAsyncGraphicsSubmissions();

    for (const auto &ticket : m_pendingImageReadbacks) {
        if (!ticket || !ObserveImageReadback(*ticket))
            continue;
        // An accepted standalone capture can finish within the CPU storage
        // limit during explicit drain. Frame capture belongs to its window.
        if (ticket->m_output == ImageReadbackOutput::Mapped || !ticket->m_graphicsSubmission ||
            !CopyImageReadback(*ticket, ticket->m_byteSize, m_readbackOwner))
            ticket->Cancel();
        RetireImageReadback(*ticket);
    }
    m_pendingImageReadbacks.erase(std::remove_if(m_pendingImageReadbacks.begin(), m_pendingImageReadbacks.end(),
                                                 [](const auto &ticket) { return !ticket || !ticket->m_staging; }),
                                  m_pendingImageReadbacks.end());
}

uint64_t VkResourceManager::GetPendingImageReadbackBytes() const noexcept
{
    uint64_t bytes = 0;
    for (const auto &ticket : m_pendingImageReadbacks) {
        if (ticket)
            bytes += ticket->m_byteSize;
    }
    return bytes;
}

void VkResourceManager::AssertReadbackThread() const
{
    if (std::this_thread::get_id() != m_ownerThread)
        throw std::logic_error("GPU image readback submission and polling require the renderer owner thread");
}

std::unique_ptr<VkBufferHandle> VkResourceManager::CreateUniformBuffer(VkDeviceSize size)
{
    // TRANSFER_DST_BIT is required for vkCmdUpdateBuffer (multi-camera UBO updates)
    return CreateBufferInternal(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

std::unique_ptr<VkBufferHandle> VkResourceManager::CreateHostDrawBuffer(const void *data, VkDeviceSize size,
                                                                        VkBufferUsageFlags usage)
{
    auto buffer =
        CreateBufferInternal(size, usage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (buffer)
        buffer->CopyFrom(data, size);
    return buffer;
}

std::unique_ptr<VkBufferHandle> VkResourceManager::CreateStorageBuffer(VkDeviceSize size, bool deviceLocal)
{
    VkMemoryPropertyFlags properties =
        deviceLocal ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                    : (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (deviceLocal) {
        usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    }

    return CreateBufferInternal(size, usage, properties);
}

std::unique_ptr<VkBufferHandle> VkResourceManager::CreateStagingBuffer(VkDeviceSize size)
{
    return CreateBufferInternal(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

VkDeviceSize VkResourceManager::SmallerStagingBlock(VkDeviceSize bytes, VkDeviceSize stagingBytes, VkDeviceSize unit)
{
    const auto sizeClass = rhi::UploadStagingClass(stagingBytes);
    if (sizeClass <= rhi::MinUploadStagingClass || bytes <= unit)
        return 0;
    const auto smaller = (std::min)(bytes - unit, sizeClass / 2);
    return smaller - smaller % unit;
}

VkDeviceSize VkResourceManager::GetStagingPoolBytes() const noexcept
{
    return m_rhiDevice ? m_rhiDevice->GetStagingPoolStatistics().bytes : 0;
}

size_t VkResourceManager::GetStagingPoolBufferCount() const noexcept
{
    return m_rhiDevice ? m_rhiDevice->GetStagingPoolStatistics().buffers : 0;
}

uint64_t VkResourceManager::GetStagingAllocationCount() const noexcept
{
    return m_rhiDevice ? m_rhiDevice->GetStagingPoolStatistics().allocations : 0;
}

uint64_t VkResourceManager::GetStagingReuseCount() const noexcept
{
    return m_rhiDevice ? m_rhiDevice->GetStagingPoolStatistics().reuses : 0;
}

uint64_t VkResourceManager::GetStagingDiscardCount() const noexcept
{
    return m_rhiDevice ? m_rhiDevice->GetStagingPoolStatistics().discards : 0;
}

std::unique_ptr<VkBufferHandle> VkResourceManager::CreateBufferInternal(VkDeviceSize size, VkBufferUsageFlags usage,
                                                                        VkMemoryPropertyFlags properties,
                                                                        const std::vector<uint32_t> &queueFamilies)
{
    auto buffer = std::make_unique<VkBufferHandle>();
    if (!buffer->Create(m_vmaAllocator, m_device, size, usage, properties, queueFamilies, m_deviceLifetime)) {
        return nullptr;
    }
    return buffer;
}

// ============================================================================
// Image and Texture Management
// ============================================================================

std::unique_ptr<VkImageHandle> VkResourceManager::CreateImage(uint32_t width, uint32_t height, VkFormat format,
                                                              VkImageUsageFlags usage, VkMemoryPropertyFlags properties)
{
    auto image = std::make_unique<VkImageHandle>();
    if (!image->Create(m_vmaAllocator, m_device, width, height, format, VK_IMAGE_TILING_OPTIMAL, usage, properties,
                       VK_SAMPLE_COUNT_1_BIT, 1, m_deviceLifetime)) {
        return nullptr;
    }
    return image;
}

std::unique_ptr<VkImageHandle> VkResourceManager::CreateDepthBuffer(uint32_t width, uint32_t height, VkFormat format)
{
    if (format == VK_FORMAT_UNDEFINED) {
        format = FindDepthFormat();
    }

    auto depthImage = CreateImage(width, height, format, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (!depthImage) {
        return nullptr;
    }

    // Create image view
    if (!depthImage->CreateView(format, rhi::ToVkImageAspectMask(format), 1)) {
        return nullptr;
    }

    return depthImage;
}

VkFormat VkResourceManager::FindDepthFormat() const
{
    std::vector<VkFormat> candidates = {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
                                        VK_FORMAT_D24_UNORM_S8_UINT};

    for (VkFormat format : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(m_physicalDevice, format, &props);

        if (props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
            return format;
        }
    }

    INXLOG_ERROR("Failed to find supported depth format");
    return VK_FORMAT_D32_SFLOAT;
}

bool VkResourceManager::HasStencilComponent(VkFormat format)
{
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

// ============================================================================
// Command Buffer Management
// ============================================================================

CommandBufferAllocation VkResourceManager::AllocatePrimaryCommandBuffer()
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    CommandBufferAllocation allocation;
    allocation.pool = m_commandPool;

    if (vkAllocateCommandBuffers(m_device, &allocInfo, &allocation.cmdBuffer) != VK_SUCCESS) {
        INXLOG_ERROR("Failed to allocate primary command buffer");
        return {};
    }

    return allocation;
}

CommandBufferAllocation VkResourceManager::AllocateSecondaryCommandBuffer()
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
    allocInfo.commandBufferCount = 1;

    CommandBufferAllocation allocation;
    allocation.pool = m_commandPool;

    if (vkAllocateCommandBuffers(m_device, &allocInfo, &allocation.cmdBuffer) != VK_SUCCESS) {
        INXLOG_ERROR("Failed to allocate secondary command buffer");
        return {};
    }

    return allocation;
}

void VkResourceManager::FreeCommandBuffer(const CommandBufferAllocation &allocation)
{
    if (allocation.cmdBuffer != VK_NULL_HANDLE && allocation.pool != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(m_device, allocation.pool, 1, &allocation.cmdBuffer);
    }
}

VkCommandBuffer VkResourceManager::BeginSingleTimeCommands()
{
    // ────────────────────────────────────────────────────────────────────
    // Phase 5b: pool command buffers and fences instead of churning them
    // per upload. Hot path now hits zero kernel allocations after the
    // first few warm-up uploads — see VkResourceManager.h for rationale.
    // ────────────────────────────────────────────────────────────────────
    VkCommandBuffer cmdBuffer = VK_NULL_HANDLE;
    {
        std::lock_guard<std::mutex> guard(m_singleTimeMutex);
        if (!m_freeSingleTimeCmdBuffers.empty()) {
            cmdBuffer = m_freeSingleTimeCmdBuffers.back();
            m_freeSingleTimeCmdBuffers.pop_back();
        }
    }

    if (cmdBuffer == VK_NULL_HANDLE) {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = m_commandPool;
        allocInfo.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(m_device, &allocInfo, &cmdBuffer) != VK_SUCCESS)
            return VK_NULL_HANDLE;

        std::lock_guard<std::mutex> guard(m_singleTimeMutex);
        m_allSingleTimeCmdBuffers.push_back(cmdBuffer);
    } else {
        // Recycled buffer carries left-over recording state — reset before reuse.
        if (vkResetCommandBuffer(cmdBuffer, 0) != VK_SUCCESS)
            return VK_NULL_HANDLE;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(cmdBuffer, &beginInfo) != VK_SUCCESS) {
        std::lock_guard<std::mutex> guard(m_singleTimeMutex);
        m_freeSingleTimeCmdBuffers.push_back(cmdBuffer);
        return VK_NULL_HANDLE;
    }

    if (m_queueManager) {
        const auto epoch = m_queueManager->ReserveCompletionEpoch();
        if (epoch == rhi::InvalidSubmissionSerial) {
            (void)vkEndCommandBuffer(cmdBuffer);
            std::lock_guard<std::mutex> guard(m_singleTimeMutex);
            m_freeSingleTimeCmdBuffers.push_back(cmdBuffer);
            return VK_NULL_HANDLE;
        }
        {
            std::lock_guard<std::mutex> guard(m_singleTimeMutex);
            m_singleTimeCompletionEpochs[cmdBuffer] = epoch;
        }
        vkdebug::PushDescriptorRecordingContext(&m_rhiDevice->GetDescriptorManager(), epoch);
    }

    return cmdBuffer;
}

void VkResourceManager::EndSingleTimeCommands(VkCommandBuffer cmdBuffer)
{
    rhi::SubmissionSerial completionEpoch = rhi::InvalidSubmissionSerial;
    {
        std::lock_guard<std::mutex> guard(m_singleTimeMutex);
        const auto found = m_singleTimeCompletionEpochs.find(cmdBuffer);
        if (found != m_singleTimeCompletionEpochs.end()) {
            completionEpoch = found->second;
            m_singleTimeCompletionEpochs.erase(found);
        }
    }
    if (completionEpoch != rhi::InvalidSubmissionSerial)
        vkdebug::PopDescriptorRecordingContext();
    const auto finishEpoch = [&] {
        if (m_queueManager && completionEpoch != rhi::InvalidSubmissionSerial)
            m_queueManager->CompleteCompletionEpoch(completionEpoch);
    };
    if (vkEndCommandBuffer(cmdBuffer) != VK_SUCCESS) {
        finishEpoch();
        std::lock_guard<std::mutex> guard(m_singleTimeMutex);
        m_freeSingleTimeCmdBuffers.push_back(cmdBuffer);
        throw std::runtime_error("Failed to end single-time graphics command buffer");
    }

    // Acquire (or lazily create) a recycled fence.
    VkFence submitFence = VK_NULL_HANDLE;
    {
        std::lock_guard<std::mutex> guard(m_singleTimeMutex);
        if (!m_freeSingleTimeFences.empty()) {
            submitFence = m_freeSingleTimeFences.back();
            m_freeSingleTimeFences.pop_back();
        }
    }
    if (submitFence == VK_NULL_HANDLE) {
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(m_device, &fenceInfo, nullptr, &submitFence) != VK_SUCCESS) {
            INXLOG_ERROR("VkResourceManager::EndSingleTimeCommands: vkCreateFence failed");
            finishEpoch();
            std::lock_guard<std::mutex> guard(m_singleTimeMutex);
            m_freeSingleTimeCmdBuffers.push_back(cmdBuffer);
            throw std::runtime_error("Failed to create single-time graphics submission fence");
        }
        std::lock_guard<std::mutex> guard(m_singleTimeMutex);
        m_allSingleTimeFences.push_back(submitFence);
    } else if (vkResetFences(m_device, 1, &submitFence) != VK_SUCCESS) {
        finishEpoch();
        std::lock_guard<std::mutex> guard(m_singleTimeMutex);
        m_freeSingleTimeCmdBuffers.push_back(cmdBuffer);
        throw std::runtime_error("Single-time graphics fence reset failed");
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuffer;

    const auto submission =
        m_queueManager ? m_queueManager->Reserve(rhi::QueueRole::Graphics) : rhi::SubmissionTicket{};
    const VkResult submitResult = m_queueManager ? m_queueManager->SubmitReserved(submission, submitInfo, submitFence)
                                                 : vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, submitFence);
    if (submitResult != VK_SUCCESS) {
        INXLOG_ERROR("VkResourceManager::EndSingleTimeCommands: graphics submit failed with VkResult ",
                     static_cast<int>(submitResult));
        {
            std::lock_guard<std::mutex> guard(m_singleTimeMutex);
            m_freeSingleTimeFences.push_back(submitFence);
            m_freeSingleTimeCmdBuffers.push_back(cmdBuffer);
        }
        finishEpoch();
        throw std::runtime_error("Failed to submit single-time graphics command buffer");
    }
    const VkResult waitResult = WaitForFencePumpingEvents(m_device, submitFence);
    if (waitResult != VK_SUCCESS) {
        if (waitResult != VK_ERROR_DEVICE_LOST)
            std::terminate(); // Synchronous callers may own stack-scoped GPU resources.
        auto pending = std::shared_ptr<GraphicsSubmissionTicket>(new GraphicsSubmissionTicket());
        pending->m_fence = submitFence;
        pending->m_commandBuffer = cmdBuffer;
        pending->m_submission = submission;
        pending->m_completionEpoch = completionEpoch;
        std::lock_guard<std::mutex> guard(m_singleTimeMutex);
        m_pendingAsyncGraphicsSubmissions.push_back(std::move(pending));
        throw std::runtime_error("Single-time graphics wait failed; submission remains pending");
    }
    if (m_queueManager && submission.IsValid())
        m_queueManager->MarkCompleted(submission);
    finishEpoch();

    // Return both objects to the free list — the fence is reusable after
    // vkResetFences (above) and the command buffer is reusable after the
    // GPU has signalled (which is what WaitForFencePumpingEvents proved).
    {
        std::lock_guard<std::mutex> guard(m_singleTimeMutex);
        m_freeSingleTimeFences.push_back(submitFence);
        m_freeSingleTimeCmdBuffers.push_back(cmdBuffer);
    }
}

std::shared_ptr<GraphicsSubmissionTicket>
VkResourceManager::EndSingleTimeCommandsAsync(VkCommandBuffer cmdBuffer, std::function<void()> releaseResources,
                                              GpuWorkReceipt work, const std::vector<SubmissionWait> &dependencies)
{
    AssertReadbackThread();
    if (cmdBuffer == VK_NULL_HANDLE)
        throw std::invalid_argument("Asynchronous graphics submission requires a live command buffer");

    rhi::SubmissionSerial completionEpoch = rhi::InvalidSubmissionSerial;
    {
        std::lock_guard<std::mutex> guard(m_singleTimeMutex);
        const auto found = m_singleTimeCompletionEpochs.find(cmdBuffer);
        if (found != m_singleTimeCompletionEpochs.end()) {
            completionEpoch = found->second;
            m_singleTimeCompletionEpochs.erase(found);
        }
    }
    if (completionEpoch != rhi::InvalidSubmissionSerial)
        vkdebug::PopDescriptorRecordingContext();
    auto recycleUnsubmitted = [&](VkFence fence = VK_NULL_HANDLE) {
        {
            std::lock_guard<std::mutex> guard(m_singleTimeMutex);
            if (fence != VK_NULL_HANDLE)
                m_freeSingleTimeFences.push_back(fence);
            m_freeSingleTimeCmdBuffers.push_back(cmdBuffer);
        }
        if (m_queueManager && completionEpoch != rhi::InvalidSubmissionSerial)
            m_queueManager->CompleteCompletionEpoch(completionEpoch);
    };
    FinishGpuWork(work, cmdBuffer);
    if (vkEndCommandBuffer(cmdBuffer) != VK_SUCCESS) {
        recycleUnsubmitted();
        throw std::runtime_error("Failed to end asynchronous graphics command buffer");
    }

    VkFence submitFence = VK_NULL_HANDLE;
    {
        std::lock_guard<std::mutex> guard(m_singleTimeMutex);
        if (!m_freeSingleTimeFences.empty()) {
            submitFence = m_freeSingleTimeFences.back();
            m_freeSingleTimeFences.pop_back();
        }
    }
    if (submitFence == VK_NULL_HANDLE) {
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(m_device, &fenceInfo, nullptr, &submitFence) != VK_SUCCESS) {
            recycleUnsubmitted();
            throw std::runtime_error("Failed to create asynchronous graphics submission fence");
        }
        std::lock_guard<std::mutex> guard(m_singleTimeMutex);
        m_allSingleTimeFences.push_back(submitFence);
    } else if (vkResetFences(m_device, 1, &submitFence) != VK_SUCCESS) {
        recycleUnsubmitted();
        throw std::runtime_error("Failed to reset asynchronous graphics submission fence");
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuffer;

    VkTimelineSemaphoreSubmitInfo timelineSubmit{};
    SubmissionWaits waits;

    std::shared_ptr<GraphicsSubmissionTicket> ticket;
    try {
        for (const auto &dependency : dependencies)
            waits.Add(dependency);
        if (!waits.semaphores.empty()) {
            timelineSubmit.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            timelineSubmit.waitSemaphoreValueCount = static_cast<uint32_t>(waits.values.size());
            timelineSubmit.pWaitSemaphoreValues = waits.values.data();
            submitInfo.pNext = &timelineSubmit;
            submitInfo.waitSemaphoreCount = static_cast<uint32_t>(waits.semaphores.size());
            submitInfo.pWaitSemaphores = waits.semaphores.data();
            submitInfo.pWaitDstStageMask = waits.stages.data();
        }
        ticket = std::make_shared<GraphicsSubmissionTicket>();
        ticket->m_commandBuffer = cmdBuffer;
        ticket->m_fence = submitFence;
        ticket->m_completionEpoch = completionEpoch;
        ticket->m_releaseResources = std::move(releaseResources);
        ticket->m_work = std::move(work);
        std::lock_guard<std::mutex> guard(m_singleTimeMutex);
        if (m_pendingAsyncGraphicsSubmissions.size() == m_pendingAsyncGraphicsSubmissions.capacity())
            m_pendingAsyncGraphicsSubmissions.reserve(
                (std::max)(size_t{1}, m_pendingAsyncGraphicsSubmissions.capacity() * 2));
    } catch (...) {
        recycleUnsubmitted(submitFence);
        throw;
    }
    const auto submission =
        m_queueManager ? m_queueManager->Reserve(rhi::QueueRole::Graphics) : rhi::SubmissionTicket{};
    ticket->m_submission = submission;
    const VkResult submitResult = m_queueManager ? m_queueManager->SubmitReserved(submission, submitInfo, submitFence)
                                                 : vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, submitFence);
    if (submitResult != VK_SUCCESS) {
        recycleUnsubmitted(submitFence);
        throw std::runtime_error("Failed to submit asynchronous graphics command buffer");
    }
    AdmitGpuWork(ticket->m_work);
    {
        std::lock_guard<std::mutex> guard(m_singleTimeMutex);
        m_pendingAsyncGraphicsSubmissions.push_back(ticket);
    }
    ++m_asyncGraphicsSubmissionCount;
    return ticket;
}

VkCommandBuffer VkResourceManager::TryBeginManagedGraphics(GpuWorkReceipt &work, FrameWorkOwner &owner,
                                                           VkDeviceSize transferBytes)
{
    AssertReadbackThread();
    const auto gpuNanoseconds = work.lastExecutionNanoseconds;
    if (!CanPrepareGpuWork(&owner, gpuNanoseconds, transferBytes, true))
        return VK_NULL_HANDLE;
    PrepareGpuWork(work, &owner, transferBytes, false, gpuNanoseconds);
    const auto commands = BeginSingleTimeCommands();
    if (commands == VK_NULL_HANDLE)
        throw std::runtime_error("Failed to begin managed graphics commands");
    BeginGpuWork(work, commands, "preview.draw");
    return commands;
}

void VkResourceManager::PollAsyncGraphicsSubmissions()
{
    AssertReadbackThread();
    std::vector<std::shared_ptr<GraphicsSubmissionTicket>> completed;
    {
        std::lock_guard<std::mutex> guard(m_singleTimeMutex);
        size_t writeIndex = 0;
        for (size_t index = 0; index < m_pendingAsyncGraphicsSubmissions.size(); ++index) {
            auto &submission = m_pendingAsyncGraphicsSubmissions[index];
            const VkResult status = vkGetFenceStatus(m_device, submission->m_fence);
            if (status == VK_SUCCESS) {
                m_freeSingleTimeFences.push_back(submission->m_fence);
                m_freeSingleTimeCmdBuffers.push_back(submission->m_commandBuffer);
                submission->m_fence = VK_NULL_HANDLE;
                submission->m_commandBuffer = VK_NULL_HANDLE;
                completed.push_back(std::move(submission));
                continue;
            }
            if (status != VK_NOT_READY)
                throw std::runtime_error("Failed to poll asynchronous graphics submission fence");
            if (writeIndex != index)
                m_pendingAsyncGraphicsSubmissions[writeIndex] = std::move(submission);
            ++writeIndex;
        }
        m_pendingAsyncGraphicsSubmissions.resize(writeIndex);
    }

    for (const auto &submission : completed) {
        RetireGpuWork(submission->m_work);
        if (m_queueManager && submission->m_submission.IsValid())
            m_queueManager->MarkCompleted(submission->m_submission);
        if (m_queueManager && submission->m_completionEpoch != rhi::InvalidSubmissionSerial)
            m_queueManager->CompleteCompletionEpoch(submission->m_completionEpoch);
        try {
            if (submission->m_releaseResources)
                submission->m_releaseResources();
        } catch (const std::exception &error) {
            INXLOG_ERROR("Asynchronous graphics resource release failed: ", error.what());
        } catch (...) {
            INXLOG_ERROR("Asynchronous graphics resource release failed with an unknown exception");
        }
        submission->m_releaseResources = {};
        submission->m_complete.store(true, std::memory_order_release);
    }
}

void VkResourceManager::DrainAsyncGraphicsSubmissions() noexcept
{
    std::vector<std::shared_ptr<GraphicsSubmissionTicket>> completed;
    {
        std::lock_guard<std::mutex> guard(m_singleTimeMutex);
        for (auto &submission : m_pendingAsyncGraphicsSubmissions) {
            if (!submission || submission->m_fence == VK_NULL_HANDLE)
                continue;
            if (WaitForFencePumpingEvents(m_device, submission->m_fence) != VK_SUCCESS)
                continue;
            m_freeSingleTimeFences.push_back(submission->m_fence);
            m_freeSingleTimeCmdBuffers.push_back(submission->m_commandBuffer);
            submission->m_fence = VK_NULL_HANDLE;
            submission->m_commandBuffer = VK_NULL_HANDLE;
            completed.push_back(std::move(submission));
        }
        m_pendingAsyncGraphicsSubmissions.erase(
            std::remove(m_pendingAsyncGraphicsSubmissions.begin(), m_pendingAsyncGraphicsSubmissions.end(), nullptr),
            m_pendingAsyncGraphicsSubmissions.end());
    }

    for (const auto &submission : completed) {
        if (m_queueManager && submission->m_submission.IsValid())
            m_queueManager->MarkCompleted(submission->m_submission);
        if (m_queueManager && submission->m_completionEpoch != rhi::InvalidSubmissionSerial)
            m_queueManager->CompleteCompletionEpoch(submission->m_completionEpoch);
        try {
            RetireGpuWork(submission->m_work);
            if (submission->m_releaseResources)
                submission->m_releaseResources();
        } catch (...) {
            INXLOG_ERROR("Asynchronous graphics resource release failed during shutdown");
        }
        submission->m_releaseResources = {};
        submission->m_complete.store(true, std::memory_order_release);
    }
}

// ============================================================================
// Sampler Management
// ============================================================================

VkSampler VkResourceManager::GetLinearSampler()
{
    if (m_linearSampler != VK_NULL_HANDLE) {
        return m_linearSampler;
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(m_physicalDevice, &properties);

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    if (vkCreateSampler(m_device, &samplerInfo, nullptr, &m_linearSampler) != VK_SUCCESS) {
        INXLOG_ERROR("Failed to create linear sampler");
        return VK_NULL_HANDLE;
    }

    return m_linearSampler;
}

VkSampler VkResourceManager::GetNearestSampler()
{
    if (m_nearestSampler != VK_NULL_HANDLE) {
        return m_nearestSampler;
    }

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

    if (vkCreateSampler(m_device, &samplerInfo, nullptr, &m_nearestSampler) != VK_SUCCESS) {
        INXLOG_ERROR("Failed to create nearest sampler");
        return VK_NULL_HANDLE;
    }

    return m_nearestSampler;
}

VkCommandPool VkResourceManager::GetCommandPool() const
{
    return m_commandPool;
}

std::unique_ptr<VkSamplerHandle> VkResourceManager::CreateSampler(VkFilter filter, VkSamplerAddressMode addressMode)
{
    auto sampler = std::make_unique<VkSamplerHandle>();
    if (!sampler->Create(m_device, m_physicalDevice, filter, addressMode)) {
        return nullptr;
    }
    return sampler;
}

} // namespace vk
} // namespace infernux
