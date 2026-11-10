/**
 * @file VkResourceManager.h
 * @brief Vulkan resource management - buffers, images, textures, and command pools
 *
 * This class manages GPU resource allocation and provides:
 * - Staging buffer pooling for efficient uploads
 * - Backend-neutral RHI texture uploads
 * - Depth buffer management
 * - Command pool and command buffer management
 * - Queue-manager-backed uploads and readbacks
 *
 * Architecture Notes:
 * - Resources are reference counted or managed via unique ownership
 * - Staging uploads use double buffering for async transfers (future)
 * - Designed for easy integration with RenderGraph
 *
 * Usage:
 *   VkResourceManager resources;
 *   resources.Initialize(deviceContext, &queueManager);
 *
 *   auto upload = resources.BeginBufferUpload(
 *       {vertices.data(), vertices.size() * sizeof(Vertex), rhi::BufferUsageFlags::Vertex});
 */

#pragma once

#include "VulkanBackendApi.h"
#include "VulkanSubmissionWaits.h"

#include <function/renderer/rhi/RhiSubmission.h>

#include "../rhi/RhiBuffer.h"
#include "../rhi/RhiTexture.h"
#include "../rhi/RhiUpload.h"
#include "AsyncTransferContext.h"
#include "GpuTimestampQueries.h"
#include "VkHandle.h"
#include "VkTypes.h"
#include <atomic>
#include <functional>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace infernux
{
class FrameCpuStorageClaim;
class FrameWorkOwner;
namespace vk
{

// Forward declarations
class VkDeviceContext;
class VulkanQueueManager;
class VkResourceManager;
class GraphicsSubmissionTicket;
class GraphicsImageReadbackRecorder;
class VulkanRhiDevice;

// Retained by the original submission until its fence completes. Shared by
// uploads, image readbacks and offscreen graphics.
struct GpuWorkReceipt
{
    std::unique_ptr<GpuTimestampQueries> timestamps;
    rhi::TimestampRegionHandle region;
    FrameWorkOwner *owner = nullptr;
    uint64_t frame = 0, gpuGrant = 0, transferGrant = 0;
    uint64_t executionNanoseconds = 0, lastExecutionNanoseconds = 0;
    uint64_t predictedExecutionNanoseconds = 0;
    VkDeviceSize transferBytes = 0;
    bool hasExecutionTiming = false;
    bool useTransfer = false;
};

class BufferUploadTicket final
{
  public:
    [[nodiscard]] bool IsComplete() const noexcept
    {
        return m_complete;
    }
    [[nodiscard]] bool IsPublished() const noexcept
    {
        return m_published;
    }
    [[nodiscard]] bool IsAsync() const noexcept
    {
        return m_async;
    }
    [[nodiscard]] VkDeviceSize GetSize() const noexcept
    {
        return m_size;
    }
    [[nodiscard]] VkDeviceSize GetSubmittedBytes() const noexcept
    {
        return m_submittedBytes;
    }
    [[nodiscard]] VkDeviceSize GetCompletedBytes() const noexcept
    {
        return m_completedBytes;
    }
    [[nodiscard]] VkDeviceSize GetDestinationBytes() const noexcept
    {
        return m_destination ? m_size : 0;
    }
    [[nodiscard]] uint64_t GetExecutionNanoseconds() const noexcept
    {
        return m_work.executionNanoseconds;
    }
    [[nodiscard]] bool HasExecutionTiming() const noexcept
    {
        return m_work.hasExecutionTiming;
    }
    [[nodiscard]] VkDeviceSize GetStagingBytes() const noexcept
    {
        return m_staging ? m_staging->GetSize() : 0;
    }
    [[nodiscard]] bool IsPending() const noexcept
    {
        return m_submittedBytes != m_completedBytes;
    }
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API const std::shared_ptr<VkBufferHandle> &GetBuffer() const;
    [[nodiscard]] std::shared_ptr<rhi::BufferResource> GetRhiBuffer() const;

  private:
    friend class VkResourceManager;
    VkResourceManager *m_manager = nullptr;
    std::shared_ptr<VkBufferHandle> m_staging;
    std::shared_ptr<VkBufferHandle> m_destination;
    // Publication must not make the upload queue an owner of the public RHI
    // resource. m_destination independently keeps the Vulkan allocation alive
    // until transfer completion.
    std::weak_ptr<rhi::BufferResource> m_rhiBuffer;
    rhi::SubmissionTicket m_upload;
    std::shared_ptr<GraphicsSubmissionTicket> m_graphicsSubmission;
    VkDeviceSize m_size = 0;
    VkBufferUsageFlags m_destinationUsage = 0;
    std::vector<uint32_t> m_queueFamilies;
    VkDeviceSize m_submittedBytes = 0;
    VkDeviceSize m_completedBytes = 0;
    bool m_complete = false;
    bool m_published = false;
    bool m_async = false;
    bool m_concurrentQueueSharing = false;
    GpuWorkReceipt m_work;
};

class TextureUploadTicket final
{
  public:
    [[nodiscard]] bool IsComplete() const noexcept
    {
        return m_complete;
    }
    [[nodiscard]] bool IsPublished() const noexcept
    {
        return m_published;
    }
    [[nodiscard]] bool IsAsync() const noexcept
    {
        return m_async;
    }
    [[nodiscard]] VkDeviceSize GetResidentBytes() const noexcept
    {
        return m_residentBytes;
    }
    [[nodiscard]] VkDeviceSize GetSubmittedBytes() const noexcept
    {
        return m_submittedBytes;
    }
    [[nodiscard]] VkDeviceSize GetCompletedBytes() const noexcept
    {
        return m_completedBytes;
    }
    [[nodiscard]] uint64_t GetExecutionNanoseconds() const noexcept
    {
        return m_work.executionNanoseconds;
    }
    [[nodiscard]] bool HasExecutionTiming() const noexcept
    {
        return m_work.hasExecutionTiming;
    }
    [[nodiscard]] VkDeviceSize GetStagingBytes() const noexcept
    {
        return m_staging ? m_staging->GetSize() : 0;
    }
    [[nodiscard]] bool IsPending() const noexcept
    {
        return m_submittedBytes != m_completedBytes;
    }
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API const std::shared_ptr<rhi::TextureResource> &GetTexture() const;

  private:
    friend class VkResourceManager;
    VkResourceManager *m_manager = nullptr;
    std::shared_ptr<VkBufferHandle> m_staging;
    std::shared_ptr<rhi::TextureResource> m_texture;
    rhi::SubmissionTicket m_upload;
    std::shared_ptr<GraphicsSubmissionTicket> m_graphicsSubmission;
    rhi::TextureDesc m_description;
    rhi::TextureViewDesc m_view;
    rhi::SamplerDesc m_sampler;
    std::vector<rhi::TextureSubresourceUpload> m_layout;
    VkDeviceSize m_residentBytes = 0;
    VkDeviceSize m_submittedBytes = 0;
    VkDeviceSize m_completedBytes = 0;
    VkDeviceSize m_profiledBytes = 0;
    uint64_t m_profiledNanoseconds = 0;
    bool m_complete = false;
    bool m_published = false;
    bool m_async = false;
    bool m_useTransfer = false;
    GpuWorkReceipt m_work;
};

enum class ImageReadbackStatus : uint8_t
{
    Pending,
    Completed,
    Failed,
    Cancelled,
};

enum class ImageReadbackOutput : uint8_t
{
    Pixels,
    Mapped
};

class ImageReadbackTicket final
{
  public:
    [[nodiscard]] ImageReadbackStatus GetStatus() const noexcept
    {
        return m_status.load(std::memory_order_acquire);
    }
    [[nodiscard]] bool IsDone() const noexcept
    {
        return GetStatus() != ImageReadbackStatus::Pending;
    }
    [[nodiscard]] uint32_t GetWidth() const noexcept
    {
        return m_width;
    }
    [[nodiscard]] uint32_t GetHeight() const noexcept
    {
        return m_height;
    }
    [[nodiscard]] uint32_t GetChannelCount() const noexcept
    {
        return m_channelCount;
    }
    [[nodiscard]] const std::string &GetElementType() const noexcept
    {
        return m_elementType;
    }
    [[nodiscard]] bool IsBgra() const noexcept
    {
        return m_bgra;
    }
    [[nodiscard]] size_t GetByteSize() const noexcept
    {
        return m_byteSize;
    }
    [[nodiscard]] bool IsGpuComplete() const noexcept
    {
        return m_gpuComplete;
    }
    [[nodiscard]] size_t GetCopiedBytes() const noexcept
    {
        return m_copiedBytes;
    }
    [[nodiscard]] size_t GetHostCapacityBytes() const noexcept
    {
        return m_data.capacity();
    }
    [[nodiscard]] VkDeviceSize GetStagingBytes() const noexcept
    {
        return m_staging ? m_staging->GetSize() : 0;
    }
    [[nodiscard]] const std::string &GetError() const noexcept
    {
        return m_error;
    }
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API const std::vector<uint8_t> &GetData() const;
    /// Mapped bytes stay valid while the capture retains this ticket.
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API std::string_view GetMappedData() const;
    INFERNUX_VULKAN_BACKEND_API void Cancel() noexcept;

  private:
    friend class VkResourceManager;
    friend class GraphicsImageReadbackRecorder;
    std::shared_ptr<VkBufferHandle> m_staging;
    std::shared_ptr<GraphicsSubmissionTicket> m_graphicsSubmission;
    GpuWorkReceipt m_work;
    std::shared_ptr<FrameCpuStorageClaim> m_hostStorage;
    rhi::SubmissionSerial m_frameCompletionEpoch = rhi::InvalidSubmissionSerial;
    std::atomic<ImageReadbackStatus> m_status{ImageReadbackStatus::Pending};
    std::vector<uint8_t> m_data;
    const uint8_t *m_mappedData = nullptr;
    ImageReadbackOutput m_output = ImageReadbackOutput::Pixels;
    std::string m_elementType;
    std::string m_error;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    uint32_t m_channelCount = 0;
    size_t m_byteSize = 0;
    size_t m_copiedBytes = 0;
    bool m_frameAccepted = false;
    bool m_gpuComplete = false;
    bool m_bgra = false;
};

class GraphicsSubmissionTicket final
{
  public:
    [[nodiscard]] bool IsComplete() const noexcept
    {
        return m_complete.load(std::memory_order_acquire);
    }
    [[nodiscard]] uint64_t GetExecutionNanoseconds() const noexcept
    {
        return m_work.executionNanoseconds;
    }
    [[nodiscard]] bool HasExecutionTiming() const noexcept
    {
        return m_work.hasExecutionTiming;
    }

  private:
    friend class VkResourceManager;
    VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
    VkFence m_fence = VK_NULL_HANDLE;
    rhi::SubmissionTicket m_submission;
    rhi::SubmissionSerial m_completionEpoch = rhi::InvalidSubmissionSerial;
    std::function<void()> m_releaseResources;
    std::atomic<bool> m_complete{false};
    GpuWorkReceipt m_work;
};

class GraphicsImageReadbackRecorder final
{
  public:
    GraphicsImageReadbackRecorder() = default;
    INFERNUX_VULKAN_BACKEND_API ~GraphicsImageReadbackRecorder();

    GraphicsImageReadbackRecorder(const GraphicsImageReadbackRecorder &) = delete;
    GraphicsImageReadbackRecorder &operator=(const GraphicsImageReadbackRecorder &) = delete;
    GraphicsImageReadbackRecorder(GraphicsImageReadbackRecorder &&other) noexcept;
    GraphicsImageReadbackRecorder &operator=(GraphicsImageReadbackRecorder &&other) noexcept;

    [[nodiscard]] VkCommandBuffer GetCommandBuffer() const noexcept
    {
        return m_commandBuffer;
    }
    [[nodiscard]] VkBuffer GetStagingBuffer() const noexcept
    {
        return m_ticket && m_ticket->m_staging ? m_ticket->m_staging->GetBuffer() : VK_NULL_HANDLE;
    }
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API std::shared_ptr<ImageReadbackTicket>
    Submit(std::function<void()> releaseResources = {}, const std::vector<SubmissionWait> &dependencies = {});

  private:
    friend class VkResourceManager;
    void Reset() noexcept;

    VkResourceManager *m_manager = nullptr;
    std::shared_ptr<ImageReadbackTicket> m_ticket;
    VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
};

/**
 * @brief Descriptor binding information
 */
struct DescriptorBindingInfo
{
    uint32_t binding = 0;
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    VkShaderStageFlags stageFlags = VK_SHADER_STAGE_ALL;
    uint32_t count = 1;
};

/**
 * @brief Command buffer allocation result
 */
struct CommandBufferAllocation
{
    VkCommandBuffer cmdBuffer = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
};

/**
 * @brief Manages Vulkan resources - buffers, images, descriptors, etc.
 */
class VkResourceManager
{
  public:
    VkResourceManager() = default;
    INFERNUX_VULKAN_BACKEND_API ~VkResourceManager();

    // Bound to one device/transfer context for its full lifetime.
    VkResourceManager(const VkResourceManager &) = delete;
    VkResourceManager &operator=(const VkResourceManager &) = delete;
    VkResourceManager(VkResourceManager &&other) = delete;
    VkResourceManager &operator=(VkResourceManager &&other) = delete;

    // ========================================================================
    // Initialization
    // ========================================================================

    /**
     * @brief Initialize the resource manager
     *
     * @param context Device context for Vulkan access
     * @return true if initialization succeeded
     */
    INFERNUX_VULKAN_BACKEND_API bool Initialize(VkDeviceContext &context, VulkanQueueManager *queueManager = nullptr);

    /**
     * @brief Cleanup all managed resources
     */
    INFERNUX_VULKAN_BACKEND_API void Destroy() noexcept;

    /// @brief Check if initialized
    [[nodiscard]] bool IsValid() const
    {
        return m_device != VK_NULL_HANDLE;
    }

    /// @brief Get the internal command pool
    [[nodiscard]] VkCommandPool GetCommandPool() const;

    // ========================================================================
    // Buffer Management
    // ========================================================================

    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API std::shared_ptr<BufferUploadTicket>
    BeginBufferUpload(const rhi::BufferUploadRequest &request);
    /// Allocate the private destination only. The caller retains immutable
    /// source data between bounded appends; this method borrows no CPU pointer.
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API std::shared_ptr<BufferUploadTicket>
    CreateBufferUpload(VkDeviceSize size, rhi::BufferUsageFlags usage,
                       rhi::QueueAccessFlags queueAccess = rhi::QueueAccessFlags::Graphics);
    /// Copy and submit a block without waiting. A managed caller can permit a
    /// smaller prefix in whole splitUnit elements. The first block of a
    /// bundle reserves all of its private destinations together.
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API bool
    TryAppendBufferUpload(const std::shared_ptr<BufferUploadTicket> &ticket, const void *data, VkDeviceSize bytes,
                          FrameWorkOwner *owner = nullptr, bool indivisible = false, VkDeviceSize splitUnit = 0,
                          std::initializer_list<BufferUploadTicket *> pairedDestinations = {});
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API bool
    TryAppendIndex16Upload(const std::shared_ptr<BufferUploadTicket> &ticket, const uint32_t *indices,
                           VkDeviceSize bytes, FrameWorkOwner *owner);
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API VkDeviceSize PlanBufferUploadBlock(const BufferUploadTicket &ticket,
                                                                                 const FrameWorkOwner &owner,
                                                                                 VkDeviceSize unit) const;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API static VkDeviceSize GetUploadStagingBytes(VkDeviceSize bytes) noexcept;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API bool
    TryPublishBufferUpload(const std::shared_ptr<BufferUploadTicket> &ticket);
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API std::shared_ptr<rhi::BufferResource>
    GetPublishedRhiBuffer(const std::shared_ptr<BufferUploadTicket> &ticket);
    INFERNUX_VULKAN_BACKEND_API void DrainBufferUploads() noexcept;

    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API std::shared_ptr<TextureUploadTicket>
    BeginTextureUpload(const rhi::TextureUploadRequest &request);
    /// Prepare the private image and layout without borrowing any CPU data.
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API std::shared_ptr<TextureUploadTicket>
    CreateTextureUpload(const rhi::TextureUploadRequest &request);
    /// Append a contiguous block in the declared subresource order. Managed
    /// pressure may reduce the block. Read the ticket cursor for bytes consumed.
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API bool
    TryAppendTextureUpload(const std::shared_ptr<TextureUploadTicket> &ticket, const void *data, VkDeviceSize bytes,
                           FrameWorkOwner *owner = nullptr);
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API bool
    CanAppendTextureUpload(const TextureUploadTicket &ticket, VkDeviceSize bytes, const FrameWorkOwner &owner) const;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API bool
    TryPublishTextureUpload(const std::shared_ptr<TextureUploadTicket> &ticket);
    INFERNUX_VULKAN_BACKEND_API void PollGpuUploads();

    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API std::shared_ptr<ImageReadbackTicket>
    BeginImageReadback(VkImage image, VkImageLayout layout, VkImageAspectFlags aspect, VkPipelineStageFlags sourceStage,
                       VkAccessFlags sourceAccess, uint32_t width, uint32_t height, VkFormat format,
                       std::function<void()> releaseResources = {}, rhi::SubmissionTicket upload = {},
                       ImageReadbackOutput output = ImageReadbackOutput::Pixels);
    /// Record a readback into an already-open frame command buffer. Completion
    /// follows the frame fence identified by completionEpoch; no second queue
    /// submission is created.
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API std::shared_ptr<ImageReadbackTicket>
    RecordFrameImageReadback(VkCommandBuffer commandBuffer, VkImage image, VkImageLayout layout,
                             VkImageAspectFlags aspect, VkPipelineStageFlags sourceStage, VkAccessFlags sourceAccess,
                             uint32_t width, uint32_t height, VkFormat format, rhi::SubmissionSerial completionEpoch,
                             ImageReadbackOutput output = ImageReadbackOutput::Pixels);
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API GraphicsImageReadbackRecorder BeginGraphicsImageReadback(
        uint32_t width, uint32_t height, VkFormat format, ImageReadbackOutput output = ImageReadbackOutput::Pixels);
    /// The host owns destination admission and copying. Polling only observes
    /// receipts and retires staging for terminal requests.
    INFERNUX_VULKAN_BACKEND_API void RegisterImageReadbackWork(FrameWorkOwner &owner);
    INFERNUX_VULKAN_BACKEND_API void AcceptFrameImageReadback(ImageReadbackTicket &ticket,
                                                              rhi::SubmissionSerial epoch) noexcept;
    INFERNUX_VULKAN_BACKEND_API void CancelFrameImageReadbacks(rhi::SubmissionSerial epoch) noexcept;
    INFERNUX_VULKAN_BACKEND_API void PollImageReadbacks();
    /// Frame command buffers must already be idle or discarded before draining.
    /// Standalone graphics readbacks are waited here; frame readbacks are cancelled.
    INFERNUX_VULKAN_BACKEND_API void DrainImageReadbacks() noexcept;
    [[nodiscard]] size_t GetPendingImageReadbackCount() const noexcept
    {
        return m_pendingImageReadbacks.size();
    }
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API uint64_t GetPendingImageReadbackBytes() const noexcept;
    [[nodiscard]] size_t GetPendingGpuTransferCount() const noexcept
    {
        return m_pendingBufferUploads.size() + m_pendingTextureUploads.size();
    }
    [[nodiscard]] uint64_t GetTimelineUploadPublicationCount() const noexcept
    {
        return m_timelineUploadPublicationCount;
    }
    [[nodiscard]] uint64_t GetBufferUploadSubmissionCount() const noexcept
    {
        return m_bufferUploadSubmissionCount;
    }
    [[nodiscard]] bool IsUploadTimelineEnabled() const noexcept
    {
        return m_asyncTransfer && m_asyncTransfer->GetTimelineSemaphore() != VK_NULL_HANDLE;
    }
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API SubmissionWait UploadReadDependency(rhi::SubmissionTicket upload,
                                                                                  rhi::QueueRole consumer,
                                                                                  VkPipelineStageFlags stages) const;

    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API VkDeviceSize GetStagingPoolBytes() const noexcept;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API size_t GetStagingPoolBufferCount() const noexcept;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API uint64_t GetStagingAllocationCount() const noexcept;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API uint64_t GetStagingReuseCount() const noexcept;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API uint64_t GetStagingDiscardCount() const noexcept;

    /**
     * @brief Create a uniform buffer
     *
     * @param size Size in bytes
     * @return Unique pointer to buffer handle (host-visible for frequent updates)
     */
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API std::unique_ptr<VkBufferHandle> CreateUniformBuffer(VkDeviceSize size);

    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API std::unique_ptr<VkBufferHandle>
    CreateHostDrawBuffer(const void *data, VkDeviceSize size, VkBufferUsageFlags usage);

    /**
     * @brief Create a storage buffer
     *
     * @param size Size in bytes
     * @param deviceLocal If true, create device-local buffer
     * @return Unique pointer to buffer handle
     */
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API std::unique_ptr<VkBufferHandle>
    CreateStorageBuffer(VkDeviceSize size, bool deviceLocal = true);

    /**
     * @brief Create a staging buffer for data uploads
     *
     * @param size Size in bytes
     * @return Unique pointer to buffer handle (host-visible)
     */
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API std::unique_ptr<VkBufferHandle> CreateStagingBuffer(VkDeviceSize size);

    // ========================================================================
    // Image and Texture Management
    // ========================================================================

    /**
     * @brief Create an image handle
     *
     * @param width Image width
     * @param height Image height
     * @param format Image format
     * @param usage Usage flags
     * @param properties Memory properties
     * @return Unique pointer to image handle
     */
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API std::unique_ptr<VkImageHandle>
    CreateImage(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage,
                VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    /**
     * @brief Create a depth buffer
     *
     * @param width Width
     * @param height Height
     * @param format Depth format (VK_FORMAT_UNDEFINED = auto-select)
     * @return Unique pointer to image handle with view
     */
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API std::unique_ptr<VkImageHandle>
    CreateDepthBuffer(uint32_t width, uint32_t height, VkFormat format = VK_FORMAT_UNDEFINED);

    // ========================================================================
    // Command Buffer Management
    // ========================================================================

    /**
     * @brief Allocate a primary command buffer
     * @return Allocated command buffer
     */
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API CommandBufferAllocation AllocatePrimaryCommandBuffer();

    /**
     * @brief Allocate a secondary command buffer
     * @return Allocated command buffer
     */
    [[nodiscard]] CommandBufferAllocation AllocateSecondaryCommandBuffer();

    /**
     * @brief Free a command buffer
     * @param allocation Command buffer allocation to free
     */
    INFERNUX_VULKAN_BACKEND_API void FreeCommandBuffer(const CommandBufferAllocation &allocation);

    /**
     * @brief Begin a one-shot command buffer for immediate submission
     * @return Command buffer ready for recording
     */
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API VkCommandBuffer BeginSingleTimeCommands();
    /// Return an unsubmitted recording and its completion epoch to this manager.
    INFERNUX_VULKAN_BACKEND_API void DiscardSingleTimeCommands(VkCommandBuffer commandBuffer) noexcept;

    /**
     * @brief End and submit a one-shot command buffer
     * @param cmdBuffer Command buffer to submit
     */
    INFERNUX_VULKAN_BACKEND_API void EndSingleTimeCommands(VkCommandBuffer cmdBuffer);
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API std::shared_ptr<GraphicsSubmissionTicket>
    EndSingleTimeCommandsAsync(VkCommandBuffer cmdBuffer, std::function<void()> releaseResources = {},
                               GpuWorkReceipt work = {}, const std::vector<SubmissionWait> &dependencies = {});
    /// Returns no commands when this host cannot admit the whole draw packet.
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API VkCommandBuffer TryBeginManagedGraphics(GpuWorkReceipt &work,
                                                                                      FrameWorkOwner &owner,
                                                                                      VkDeviceSize transferBytes);
    INFERNUX_VULKAN_BACKEND_API void PollAsyncGraphicsSubmissions();
    INFERNUX_VULKAN_BACKEND_API void DrainAsyncGraphicsSubmissions() noexcept;
    [[nodiscard]] size_t GetPendingAsyncGraphicsSubmissionCount() const noexcept
    {
        return m_pendingAsyncGraphicsSubmissions.size();
    }
    [[nodiscard]] uint64_t GetAsyncGraphicsSubmissionCount() const noexcept
    {
        return m_asyncGraphicsSubmissionCount;
    }

    // ========================================================================
    // Sampler Management
    // ========================================================================

    /**
     * @brief Get or create a standard linear sampler
     */
    [[nodiscard]] VkSampler GetLinearSampler();

    /**
     * @brief Get or create a standard nearest sampler
     */
    [[nodiscard]] VkSampler GetNearestSampler();

    /**
     * @brief Create a custom sampler
     */
    [[nodiscard]] std::unique_ptr<VkSamplerHandle>
    CreateSampler(VkFilter filter = VK_FILTER_LINEAR,
                  VkSamplerAddressMode addressMode = VK_SAMPLER_ADDRESS_MODE_REPEAT);

    // ========================================================================
    // Accessors
    // ========================================================================

    [[nodiscard]] VkDevice GetDevice() const
    {
        return m_device;
    }
    [[nodiscard]] VkPhysicalDevice GetPhysicalDevice() const
    {
        return m_physicalDevice;
    }
    [[nodiscard]] VkQueue GetGraphicsQueue() const
    {
        return m_graphicsQueue;
    }

  private:
    struct TextureUploadBlock;
    TextureUploadBlock SelectTextureUploadBlock(const TextureUploadTicket &ticket, VkDeviceSize bytes,
                                                const FrameWorkOwner *owner) const;
    bool AppendTextureUpload(const std::shared_ptr<TextureUploadTicket> &ticket, VkDeviceSize bytes,
                             const std::function<const void *(uint32_t, VkDeviceSize, VkDeviceSize)> &source,
                             FrameWorkOwner *owner = nullptr);
    friend class GraphicsImageReadbackRecorder;
    // ========================================================================
    // Internal Methods
    // ========================================================================

    /// @brief Create a buffer with given usage and properties
    std::unique_ptr<VkBufferHandle> CreateBufferInternal(VkDeviceSize size, VkBufferUsageFlags usage,
                                                         VkMemoryPropertyFlags properties,
                                                         const std::vector<uint32_t> &queueFamilies = {});
    bool PrepareBufferUploadDestination(BufferUploadTicket &ticket, bool managed);
    bool PrepareTextureUploadDestination(TextureUploadTicket &ticket, bool managed);
    static VkDeviceSize SmallerStagingBlock(VkDeviceSize bytes, VkDeviceSize stagingBytes, VkDeviceSize unit);
    std::shared_ptr<ImageReadbackTicket> CreateImageReadback(uint32_t width, uint32_t height, VkFormat format,
                                                             ImageReadbackOutput output);
    void ReserveReadbackSlot();
    void QueueImageReadback(const std::shared_ptr<ImageReadbackTicket> &ticket);
    bool CopyImageReadback(ImageReadbackTicket &ticket, size_t maxBytes, FrameWorkOwner *owner);
    bool ObserveImageReadback(ImageReadbackTicket &ticket);
    void RetireImageReadback(ImageReadbackTicket &ticket) noexcept;
    std::shared_ptr<ImageReadbackTicket> SubmitGraphicsImageReadback(GraphicsImageReadbackRecorder &recorder,
                                                                     std::function<void()> releaseResources,
                                                                     const std::vector<SubmissionWait> &dependencies);
    void AbandonGraphicsImageReadback(GraphicsImageReadbackRecorder &recorder) noexcept;
    void AssertReadbackThread() const;
    bool AppendBufferUpload(const std::shared_ptr<BufferUploadTicket> &ticket, const void *data, VkDeviceSize bytes,
                            FrameWorkOwner *owner, bool indivisible, VkDeviceSize splitUnit, bool packIndex16,
                            std::initializer_list<BufferUploadTicket *> pairedDestinations);
    bool PrepareBufferUploadDestinations(BufferUploadTicket &ticket,
                                         std::initializer_list<BufferUploadTicket *> pairedDestinations, bool managed);
    static bool CanPrepareGpuWork(const FrameWorkOwner *owner, uint64_t gpuNanoseconds, VkDeviceSize bytes,
                                  bool indivisible);
    void PrepareGpuWork(GpuWorkReceipt &work, FrameWorkOwner *owner, VkDeviceSize bytes, bool useTransfer,
                        uint64_t gpuNanoseconds);
    void BeginGpuWork(GpuWorkReceipt &work, VkCommandBuffer commands, const char *name);
    void FinishGpuWork(GpuWorkReceipt &work, VkCommandBuffer commands);
    void AdmitGpuWork(GpuWorkReceipt &work);
    uint64_t RetireGpuWork(GpuWorkReceipt &work);
    bool RecordUploadPublication(const GpuWorkReceipt &work, bool complete);
    void RetireBufferUpload(BufferUploadTicket &ticket);

    /// @brief Get best depth format for the device
    VkFormat FindDepthFormat() const;

    /// @brief Check if format has stencil component
    static bool HasStencilComponent(VkFormat format);

  public:
    void SetSkipWaitIdle(bool v)
    {
        m_skipWaitIdle = v;
    }

    /// Configure the transfer context used by explicit Buffer/TextureUploadTicket submissions.
    INFERNUX_VULKAN_BACKEND_API void SetAsyncTransferContext(class AsyncTransferContext *transfer);

  private:
    bool m_skipWaitIdle = false;
    VkDeviceContext *m_context = nullptr;
    VmaAllocator m_vmaAllocator = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkQueue m_graphicsQueue = VK_NULL_HANDLE;
    VulkanRhiDevice *m_rhiDevice = nullptr;
    std::shared_ptr<rhi::DeviceLifetime> m_deviceLifetime;
    VulkanQueueManager *m_queueManager = nullptr;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    std::thread::id m_ownerThread;

    // Optional plug-in async-transfer context. Lifetime is owned externally
    // (typically InxVkCoreModular::m_asyncTransferContext) — VkResourceManager
    // never destroys it. nullptr means "always use the synchronous path".
    class AsyncTransferContext *m_asyncTransfer = nullptr;
    bool m_transferTimestampsSupported = false;
    uint32_t m_graphicsQueueFamily = 0;
    uint32_t m_computeQueueFamily = 0;
    uint32_t m_transferQueueFamily = 0;
    std::vector<std::shared_ptr<BufferUploadTicket>> m_pendingBufferUploads;
    std::vector<std::shared_ptr<TextureUploadTicket>> m_pendingTextureUploads;
    VkDeviceSize m_textureUploadProfiledBytes = 0;
    uint64_t m_textureUploadProfiledNanoseconds = 0;
    std::vector<std::unique_ptr<GpuTimestampQueries>> m_graphicsWorkQueries, m_transferWorkQueries;
    void RetireTextureUpload(TextureUploadTicket &ticket);
    std::vector<std::shared_ptr<ImageReadbackTicket>> m_pendingImageReadbacks;
    FrameWorkOwner *m_readbackOwner = nullptr;
    uint64_t m_readbackSource = 0;
    size_t m_readbackTimingBytes = 0;
    uint64_t m_readbackTimingNanoseconds = 0;
    uint64_t m_timelineUploadPublicationCount = 0;
    uint64_t m_bufferUploadSubmissionCount = 0;

    // Cached samplers
    VkSampler m_linearSampler = VK_NULL_HANDLE;
    VkSampler m_nearestSampler = VK_NULL_HANDLE;

    // Recycled single-time command buffers and fences are published under
    // m_singleTimeMutex and returned to free lists only after GPU completion.
    std::vector<VkCommandBuffer> m_freeSingleTimeCmdBuffers;
    std::vector<VkFence> m_freeSingleTimeFences;
    std::vector<VkFence> m_allSingleTimeFences;             // owned, destroyed in Destroy()
    std::vector<VkCommandBuffer> m_allSingleTimeCmdBuffers; // owned via m_commandPool
    std::unordered_map<VkCommandBuffer, rhi::SubmissionSerial> m_singleTimeCompletionEpochs;
    std::vector<std::shared_ptr<GraphicsSubmissionTicket>> m_pendingAsyncGraphicsSubmissions;
    uint64_t m_asyncGraphicsSubmissionCount = 0;
    std::mutex m_singleTimeMutex;
};

} // namespace vk
} // namespace infernux
