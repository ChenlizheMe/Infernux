#pragma once

#include "VulkanBackendApi.h"

#include <function/renderer/ProfileConfig.h>
#include <function/renderer/rhi/GpuRetirementQueue.h>
#include <function/renderer/rhi/RhiCapabilities.h>
#include <function/renderer/rhi/RhiCommand.h>
#include <function/renderer/rhi/RhiDevice.h>

#include "VkDescriptorManager.h"
#include "VkHandle.h"
#include "VulkanFeatureState.h"

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

namespace infernux::vk
{

class VulkanRhiDevice;

struct VulkanGraphicsCommandContext
{
    static constexpr size_t BindingSlotCount = rhi::GraphicsPipelineDesc::MaxBindingLayouts;

    VulkanRhiDevice *device = nullptr;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    rhi::GraphicsPipelineHandle boundPipeline;
    std::array<rhi::BindGroupHandle, BindingSlotCount> boundGroups{};
    // These fields are deliberately unconditional. Command contexts cross
    // DLL boundaries through RenderContext, so profile flags must never
    // change their binary layout.
    uint64_t pipelineBinds = 0;
    uint64_t pipelineBindSkips = 0;
    uint64_t groupBinds = 0;
    uint64_t groupBindSkips = 0;

    void ResetBindingState() noexcept
    {
        boundPipeline = {};
        boundGroups.fill({});
    }

    void ResetMetrics() noexcept
    {
        pipelineBinds = 0;
        pipelineBindSkips = 0;
        groupBinds = 0;
        groupBindSkips = 0;
    }
};

struct VulkanComputeCommandContext
{
    static constexpr size_t BindingSlotCount = rhi::ComputePipelineDesc::MaxBindingLayouts;

    VulkanRhiDevice *device = nullptr;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    rhi::ComputePipelineHandle boundPipeline;
    std::array<rhi::BindGroupHandle, BindingSlotCount> boundGroups{};
    // Keep the layout identical in every native module; only counter updates
    // are compiled out when frame profiling is disabled.
    uint64_t pipelineBinds = 0;
    uint64_t pipelineBindSkips = 0;
    uint64_t groupBinds = 0;
    uint64_t groupBindSkips = 0;

    void ResetBindingState() noexcept
    {
        boundPipeline = {};
        boundGroups.fill({});
    }

    void ResetMetrics() noexcept
    {
        pipelineBinds = 0;
        pipelineBindSkips = 0;
        groupBinds = 0;
        groupBindSkips = 0;
    }
};

struct VulkanTransferCommandContext
{
    VulkanRhiDevice *device = nullptr;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
};

/// Inputs captured from one physical device. Keeping this as a value type
/// makes capability policy testable without requiring a live VkDevice.
struct VulkanCapabilityProbeData final
{
    uint32_t apiVersion = VK_API_VERSION_1_0;
    VkPhysicalDeviceFeatures coreFeatures{};
    VkPhysicalDeviceVulkan12Features vulkan12Features{};
    VkPhysicalDeviceVulkan13Features vulkan13Features{};
    VkPhysicalDeviceDescriptorIndexingFeaturesEXT descriptorIndexingFeaturesEXT{};
    VkPhysicalDeviceTimelineSemaphoreFeaturesKHR timelineSemaphoreFeaturesKHR{};
    VkPhysicalDeviceDynamicRenderingFeaturesKHR dynamicRenderingFeaturesKHR{};
    VkPhysicalDeviceSynchronization2FeaturesKHR synchronization2FeaturesKHR{};

    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceVulkan12Properties vulkan12Properties{};
    VkPhysicalDeviceVulkan13Properties vulkan13Properties{};

    bool descriptorIndexingExtension = false;
    bool timelineSemaphoreExtension = false;
    bool dynamicRenderingExtension = false;
    bool synchronization2Extension = false;
};

struct VulkanCapabilitySnapshot final
{
    /// These are the physical-device properties captured by QueryProbe. They
    /// stay with the feature result so policy code can inspect device limits
    /// without querying Vulkan again.
    uint32_t apiVersion = VK_API_VERSION_1_0;
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceVulkan12Properties vulkan12Properties{};
    VkPhysicalDeviceVulkan13Properties vulkan13Properties{};
    bool descriptorIndexingExtension = false;
    bool timelineSemaphoreExtension = false;
    bool dynamicRenderingExtension = false;
    bool synchronization2Extension = false;
    vk::VulkanFeatureState supported;

    [[nodiscard]] static VulkanCapabilityProbeData QueryProbe(VkPhysicalDevice physicalDevice,
                                                              uint32_t apiVersionLimit = UINT32_MAX);
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API static VulkanCapabilitySnapshot
    FromProbe(const VulkanCapabilityProbeData &probe) noexcept;
};

/// Required features shared by adapter selection and logical-device creation.
[[nodiscard]] INFERNUX_VULKAN_BACKEND_API bool MeetsVulkanDeviceRequirements(const VulkanCapabilityProbeData &probe,
                                                                             bool rendering = true) noexcept;

/// Candidates have already passed the surface, queue and swapchain checks.
struct VulkanPhysicalDeviceCandidate final
{
    VkPhysicalDevice device = VK_NULL_HANDLE;
    int score = 0;
    VulkanCapabilityProbeData probe;
};

/// Returns a candidate owned by the input vector, or nullptr when none qualify.
/// Equal scores retain enumeration order; unsupported devices never participate.
[[nodiscard]] INFERNUX_VULKAN_BACKEND_API const VulkanPhysicalDeviceCandidate *
SelectVulkanPhysicalDevice(const std::vector<VulkanPhysicalDeviceCandidate> &candidates,
                           bool rendering = true) noexcept;

/// Builds only the capability feature chain requested by the caller. The
/// object owns every pNext node, so the returned VkPhysicalDeviceFeatures2
/// remains valid until this builder is destroyed or reset.
class VulkanDeviceFeatureChain final
{
  public:
    INFERNUX_VULKAN_BACKEND_API explicit VulkanDeviceFeatureChain(const VulkanCapabilitySnapshot &supported) noexcept;

    VulkanDeviceFeatureChain(const VulkanDeviceFeatureChain &) = delete;
    VulkanDeviceFeatureChain &operator=(const VulkanDeviceFeatureChain &) = delete;

    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API bool Enable(const vk::DeviceCapabilityRequest &request) noexcept;
    [[nodiscard]] const VkPhysicalDeviceFeatures2 &GetFeatures2() const noexcept
    {
        return m_features2;
    }
    [[nodiscard]] const vk::VulkanFeatureState &GetEnabledState() const noexcept
    {
        return m_enabled;
    }
    [[nodiscard]] const vk::DeviceCapabilityCheck &GetFailure() const noexcept
    {
        return m_failure;
    }
    [[nodiscard]] std::string_view GetFailureMessage() const noexcept
    {
        return m_failure.Message();
    }

  private:
    void ResetChain() noexcept;
    void Link(void *feature) noexcept;
    [[nodiscard]] bool EnableDescriptorIndexing() noexcept;
    [[nodiscard]] bool EnableTimelineSemaphore() noexcept;
    [[nodiscard]] bool EnableDynamicRendering() noexcept;
    [[nodiscard]] bool EnableSynchronization2() noexcept;

    VulkanCapabilitySnapshot m_supported;
    vk::VulkanFeatureState m_enabled{};
    vk::DeviceCapabilityCheck m_failure{};
    VkPhysicalDeviceFeatures2 m_features2{};
    VkPhysicalDeviceVulkan12Features m_vulkan12{};
    VkPhysicalDeviceVulkan13Features m_vulkan13{};
    VkPhysicalDeviceDescriptorIndexingFeaturesEXT m_descriptorIndexingEXT{};
    VkPhysicalDeviceTimelineSemaphoreFeaturesKHR m_timelineSemaphoreKHR{};
    VkPhysicalDeviceDynamicRenderingFeaturesKHR m_dynamicRenderingKHR{};
    VkPhysicalDeviceSynchronization2FeaturesKHR m_synchronization2KHR{};
    void *m_lastFeature = nullptr;
    bool m_vulkan12Linked = false;
    bool m_vulkan13Linked = false;
    bool m_descriptorIndexingEXTLinked = false;
    bool m_timelineSemaphoreKHRLinked = false;
    bool m_dynamicRenderingKHRLinked = false;
    bool m_synchronization2KHRLinked = false;
};

class VulkanRhiDevice final : public rhi::Device
{
  public:
    INFERNUX_VULKAN_BACKEND_API VulkanRhiDevice();
    INFERNUX_VULKAN_BACKEND_API explicit VulkanRhiDevice(VkDevice device, VmaAllocator allocator = VK_NULL_HANDLE,
                                                         const rhi::DeviceCaps &capabilities = {},
                                                         uint32_t graphicsQueueFamily = 0,
                                                         uint32_t computeQueueFamily = 0,
                                                         uint32_t transferQueueFamily = 0,
                                                         const vk::VulkanFeatureState &capabilityState = {});

    VulkanRhiDevice(const VulkanRhiDevice &) = delete;
    VulkanRhiDevice &operator=(const VulkanRhiDevice &) = delete;
    VulkanRhiDevice(VulkanRhiDevice &&) = delete;
    VulkanRhiDevice &operator=(VulkanRhiDevice &&) = delete;

    INFERNUX_VULKAN_BACKEND_API ~VulkanRhiDevice();

    [[nodiscard]] rhi::DeviceId GetDeviceId() const noexcept override
    {
        return m_deviceId;
    }
    [[nodiscard]] const rhi::DeviceCaps &GetCapabilities() const noexcept override
    {
        return m_capabilities;
    }
    [[nodiscard]] const vk::VulkanFeatureState &GetVulkanFeatures() const noexcept
    {
        return m_capabilityState;
    }
    [[nodiscard]] std::shared_ptr<rhi::DeviceLifetime> GetLifetime() const noexcept override
    {
        return m_lifetime;
    }

    INFERNUX_VULKAN_BACKEND_API void Reset(VkDevice device = VK_NULL_HANDLE, VmaAllocator allocator = VK_NULL_HANDLE,
                                           const rhi::DeviceCaps &capabilities = {}, uint32_t graphicsQueueFamily = 0,
                                           uint32_t computeQueueFamily = 0, uint32_t transferQueueFamily = 0,
                                           const vk::VulkanFeatureState &capabilityState = {}) noexcept;

    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::BufferHandle RegisterBuffer(VkBuffer buffer, uint64_t byteSize = 0,
                                                                               bool concurrentQueueSharing = false);
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::BufferHandle
    RegisterBuffer(VkBuffer buffer, const VkBufferCreateInfo &info, rhi::BufferMemory memory);
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::BufferHandle RegisterBuffer(rhi::BufferHandle source);
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::TextureHandle RegisterTexture(VkImage image);
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::TextureHandle RegisterTexture(VkImage image,
                                                                                 const VkImageCreateInfo &info);
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::TextureHandle RegisterTexture(rhi::TextureHandle source);
    // Registrations are non-owning aliases. Their original native owner must
    // outlive all graph uses; releasing an alias never releases that owner.
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API bool GetBufferDesc(rhi::BufferHandle handle,
                                                                 rhi::BufferDesc &desc) const noexcept;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API uint64_t GetBufferResidentBytes(rhi::BufferHandle handle) const noexcept;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API bool GetTextureDesc(rhi::TextureHandle handle,
                                                                  rhi::TextureDesc &desc) const noexcept;
    [[nodiscard]] uint64_t GetTextureResidentBytes(rhi::TextureHandle handle) const noexcept;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::TextureViewHandle
    RegisterTextureView(VkImageView view, rhi::TextureHandle texture = {});
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::SamplerHandle RegisterSampler(VkSampler sampler);
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::ShaderModuleHandle RegisterShaderModule(VkShaderModule module);
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::BindingLayoutHandle
    RegisterBindingLayout(VkDescriptorSetLayout layout);
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::BindGroupHandle RegisterBindGroup(VkDescriptorSet set);
    using BindlessTexturePublisher =
        std::function<rhi::ResourceIndex(const std::shared_ptr<const rhi::TextureGpuView> &)>;
    using BindlessTextureUsageMarker = std::function<void(const rhi::ResourceIndex *, size_t, rhi::SubmissionSerial)>;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API bool
    ConfigureBindlessTextureTable(VkDescriptorSetLayout layout, VkDescriptorSet set, BindlessTexturePublisher publisher,
                                  BindlessTextureUsageMarker usageMarker);
    INFERNUX_VULKAN_BACKEND_API void ClearBindlessTextureTable() noexcept;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::BindlessTextureTableBinding
    GetBindlessTextureTableBinding() const noexcept override;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::ResourceIndex
    PublishBindlessTexture(const std::shared_ptr<const rhi::TextureGpuView> &texture) noexcept override;
    INFERNUX_VULKAN_BACKEND_API void MarkBindlessTexturesUsed(const rhi::ResourceIndex *resources,
                                                              size_t count) noexcept override;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::GraphicsPipelineHandle
    RegisterGraphicsPipeline(VkPipeline pipeline, VkPipelineLayout layout);
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::ComputePipelineHandle
    RegisterComputePipeline(VkPipeline pipeline, VkPipelineLayout layout);
    /// Create a compute pipeline from backend-neutral RHI handles. The
    /// returned pipeline and its layout are owned by this adapter.
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::BufferHandle CreateBuffer(const rhi::BufferDesc &desc) override;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::TextureHandle CreateTexture(const rhi::TextureDesc &desc) override;
    [[nodiscard]] std::optional<rhi::TextureHandle> CreateUploadTexture(const rhi::TextureDesc &desc, bool managed);
    struct TextureAllocation
    {
        rhi::TextureHandle texture;
        rhi::TextureViewHandle view;
    };
    // The graph supplies its exact Vulkan format, sample count and queue sharing.
    [[nodiscard]] TextureAllocation CreateTransientTexture2D(const VkImageCreateInfo &info);
    void SetUploadSubmission(rhi::BufferHandle buffer, rhi::SubmissionTicket ticket) noexcept;
    void SetUploadSubmission(rhi::TextureHandle texture, rhi::SubmissionTicket ticket) noexcept;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::SubmissionTicket
    GetUploadSubmission(rhi::BufferHandle buffer) const noexcept;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::SubmissionTicket
    GetUploadSubmission(rhi::TextureHandle texture) const noexcept;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API std::optional<rhi::BufferHandle>
    CreatePrivateBuffer(const rhi::BufferDesc &desc, bool managed) override;
    void PublishPrivateBuffer(rhi::BufferHandle handle) noexcept override;
    void SetManagedHeapLimits(std::function<VkDeviceSize()> deviceLocal, VkDeviceSize otherHeap = 0)
    {
        m_managedDeviceLocalLimit = std::move(deviceLocal);
        m_managedOtherHeapLimit = otherHeap;
    }
    [[nodiscard]] bool HasDeviceLocalAllocationPressure() const noexcept
    {
        return m_deviceLocalAllocationPressure != 0;
    }
    [[nodiscard]] VkDeviceSize TakeDeviceLocalAllocationPressure() noexcept
    {
        const auto bytes = m_deviceLocalAllocationPressure;
        m_deviceLocalAllocationPressure = 0;
        return bytes;
    }
    struct StagingPoolStatistics
    {
        VkDeviceSize bytes = 0;
        size_t buffers = 0;
        uint64_t allocations = 0;
        uint64_t reuses = 0;
        uint64_t discards = 0;
    };
    enum class StagingAccess : uint8_t
    {
        Upload,
        Readback
    };
    [[nodiscard]] const StagingPoolStatistics &GetStagingPoolStatistics() const noexcept
    {
        return m_stagingStats;
    }
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API std::shared_ptr<VkBufferHandle>
    AcquireStagingBuffer(VkDeviceSize bytes, bool managed, StagingAccess access = StagingAccess::Upload);
    INFERNUX_VULKAN_BACKEND_API void RecycleStagingBuffer(std::shared_ptr<VkBufferHandle> buffer,
                                                          StagingAccess access = StagingAccess::Upload) noexcept;
    [[nodiscard]] bool AdmitAllocation(uint32_t memoryType, VkDeviceSize bytes);
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API bool
    CanRetainGraphBuffers(const std::vector<rhi::BufferHandle> &distinctBuffers);
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API static bool IsValidTextureDesc(const rhi::TextureDesc &desc) noexcept;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::TextureViewHandle
    CreateTextureView(const rhi::TextureViewDesc &desc) override;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::SamplerHandle CreateSampler(const rhi::SamplerDesc &desc) override;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::ShaderModuleHandle
    CreateShaderModule(const rhi::ShaderModuleDesc &desc) override;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::BindingLayoutHandle
    CreateBindingLayout(const rhi::BindingLayoutDesc &desc) override;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::BindGroupHandle
    CreateBindGroup(const rhi::BindGroupDesc &desc) override;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::GraphicsPipelineHandle
    CreateGraphicsPipeline(const rhi::GraphicsPipelineDesc &desc) override;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::ComputePipelineHandle
    CreateComputePipeline(const rhi::ComputePipelineDesc &desc) override;
    INFERNUX_VULKAN_BACKEND_API bool WriteBuffer(rhi::BufferHandle handle, uint64_t offset, const void *data,
                                                 uint64_t byteSize) override;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API void *MapBuffer(rhi::BufferHandle handle, uint64_t offset,
                                                              uint64_t byteSize, rhi::BufferMapAccess access) override;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API bool UnmapBuffer(rhi::BufferHandle handle, uint64_t offset,
                                                               uint64_t byteSize, rhi::BufferMapAccess access) override;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API bool ReadBuffer(rhi::BufferHandle handle, uint64_t offset, void *data,
                                                              uint64_t byteSize) override;
    [[nodiscard]] rhi::RenderTargetLayoutHandle RegisterRenderTargetLayout(VkRenderPass renderPass);

    INFERNUX_VULKAN_BACKEND_API void
    UseSubmissionSerials(std::function<rhi::SubmissionSerial()> retirementSerialSource);
    [[nodiscard]] bool UsesSubmissionSerials() const noexcept
    {
        return m_resourceRetirementQueue.HasSerialSource();
    }
    size_t CollectDescriptorRetirements(rhi::SubmissionSerial completedSerial) noexcept
    {
        return m_descriptorManager.Collect(completedSerial);
    }
    size_t CollectResourceRetirements(rhi::SubmissionSerial completedSerial) noexcept
    {
        return m_resourceRetirementQueue.Collect(completedSerial);
    }
    [[nodiscard]] GpuRetirementQueue::Stats GetResourceRetirementStats() const noexcept
    {
        return m_resourceRetirementQueue.GetStats();
    }
    [[nodiscard]] VkDescriptorManager::Stats GetDescriptorStats() const noexcept
    {
        return m_descriptorManager.GetStats();
    }
    [[nodiscard]] VkDescriptorManager &GetDescriptorManager() noexcept
    {
        return m_descriptorManager;
    }

    INFERNUX_VULKAN_BACKEND_API void Release(rhi::BufferHandle handle) noexcept override;
    INFERNUX_VULKAN_BACKEND_API void Release(rhi::BufferHandle handle, GpuRetirementQueue *completionDomain) noexcept;
    INFERNUX_VULKAN_BACKEND_API void Release(rhi::TextureHandle handle) noexcept override;
    INFERNUX_VULKAN_BACKEND_API void Release(rhi::TextureViewHandle handle) noexcept override;
    INFERNUX_VULKAN_BACKEND_API void Release(TextureAllocation allocation, GpuRetirementQueue *completionDomain,
                                             bool deviceIdle = false) noexcept;
    INFERNUX_VULKAN_BACKEND_API void Release(rhi::SamplerHandle handle) noexcept override;
    INFERNUX_VULKAN_BACKEND_API void Release(rhi::ShaderModuleHandle handle) noexcept override;
    INFERNUX_VULKAN_BACKEND_API void Release(rhi::BindingLayoutHandle handle) noexcept override;
    INFERNUX_VULKAN_BACKEND_API void Release(rhi::BindGroupHandle handle) noexcept override;
    INFERNUX_VULKAN_BACKEND_API void Release(rhi::GraphicsPipelineHandle handle) noexcept override;
    INFERNUX_VULKAN_BACKEND_API void Release(rhi::ComputePipelineHandle handle) noexcept override;
    INFERNUX_VULKAN_BACKEND_API void Release(rhi::RenderTargetLayoutHandle handle) noexcept;

    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API VkBuffer Resolve(rhi::BufferHandle handle) const noexcept;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API bool UsesConcurrentQueueSharing(rhi::BufferHandle handle) const noexcept;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API VkImage Resolve(rhi::TextureHandle handle) const noexcept;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API bool UsesConcurrentQueueSharing(rhi::TextureHandle handle) const noexcept;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API VkImageView Resolve(rhi::TextureViewHandle handle) const noexcept;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API VkSampler Resolve(rhi::SamplerHandle handle) const noexcept;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API VkShaderModule Resolve(rhi::ShaderModuleHandle handle) const noexcept;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API VkDescriptorSetLayout
    Resolve(rhi::BindingLayoutHandle handle) const noexcept;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API VkDescriptorSet Resolve(rhi::BindGroupHandle handle) const noexcept;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API VkRenderPass Resolve(rhi::RenderTargetLayoutHandle handle) const noexcept;

    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::GraphicsCommandEncoder
    MakeGraphicsCommandEncoder(VulkanGraphicsCommandContext &context, VkCommandBuffer commandBuffer) noexcept;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::ComputeCommandEncoder
    MakeComputeCommandEncoder(VulkanComputeCommandContext &context, VkCommandBuffer commandBuffer) noexcept;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::TransferCommandEncoder
    MakeTransferCommandEncoder(VulkanTransferCommandContext &context, VkCommandBuffer commandBuffer) noexcept;

  private:
    std::optional<rhi::BufferHandle> CreateBufferInternal(const rhi::BufferDesc &desc, bool managed,
                                                          bool submissionOwned = false);
    std::optional<rhi::TextureHandle>
    CreateTextureInternal(const rhi::TextureDesc &desc, const AllocationAdmission *admit, bool sampledAsset = false);
    std::optional<rhi::TextureHandle> AllocateTexture(const rhi::TextureDesc &desc, const VkImageCreateInfo &info,
                                                      const AllocationAdmission *admit);
    rhi::TextureViewHandle CreateTextureViewNative(rhi::TextureHandle texture, const VkImageViewCreateInfo &info);
    void CreatePipelineCache() noexcept;
    void SaveAndDestroyPipelineCache() noexcept;

    struct BufferPayload
    {
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        void *mappedData = nullptr;
        uint64_t byteSize = 0;
        bool owned = false;
        rhi::BufferMemory memory = rhi::BufferMemory::DeviceLocal;
        bool concurrentQueueSharing = false;
        rhi::BufferUsageFlags usage = rhi::BufferUsageFlags::None;
        bool hasCreationInfo = false;
        bool submissionOwned = false;
        rhi::SubmissionTicket uploadSubmission{};
    };

    struct TexturePayload
    {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        rhi::TextureDesc desc{};
        bool owned = false;
        bool concurrentQueueSharing = false;
        rhi::SubmissionTicket uploadSubmission{};
    };

    struct TextureViewPayload
    {
        VkImageView view = VK_NULL_HANDLE;
        bool owned = false;
        rhi::TextureHandle texture{};
    };

    struct SamplerPayload
    {
        VkSampler sampler = VK_NULL_HANDLE;
        bool owned = false;
    };

    struct ShaderModulePayload
    {
        VkShaderModule module = VK_NULL_HANDLE;
        bool owned = false;
    };

    struct BindingLayoutPayload
    {
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        bool owned = false;
    };

    struct BindGroupPayload
    {
        DescriptorLease lease;
        bool owned = false;
    };

    struct GraphicsPipelinePayload
    {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        bool ownsPipeline = false;
        bool ownsLayout = false;
        rhi::BindingLayoutHandle computeSetZeroLayout;
    };

    template <typename Payload> struct Slot
    {
        Payload payload{};
        uint16_t generation = 1; // Zero permanently retires this slot.
        uint32_t nextFree = UINT32_MAX;
        bool occupied = false;
    };

    template <typename HandleType, typename Payload>
    [[nodiscard]] HandleType Register(std::vector<Slot<Payload>> &slots, uint32_t &freeHead, const Payload &payload);
    template <typename HandleType, typename Payload>
    void Release(std::vector<Slot<Payload>> &slots, uint32_t &freeHead, HandleType handle) noexcept;
    template <typename Payload> static void ResetSlots(std::vector<Slot<Payload>> &slots, uint32_t &freeHead) noexcept;
    template <typename HandleType, typename Payload>
    [[nodiscard]] const Payload *Resolve(const std::vector<Slot<Payload>> &slots, HandleType handle) const noexcept;

    [[nodiscard]] const GraphicsPipelinePayload *ResolvePipeline(rhi::GraphicsPipelineHandle handle) const noexcept;
    [[nodiscard]] const GraphicsPipelinePayload *ResolvePipeline(rhi::ComputePipelineHandle handle) const noexcept;
    void RetireNativeResource(std::function<void()> deleter, GpuRetirementQueue *completionDomain = nullptr) noexcept;
    void DestroyOwnedResources() noexcept;
    [[nodiscard]] bool SupportsPipelineLayout(uint32_t bindingLayoutCount, uint32_t pushConstantBytes,
                                              const char *pipelineKind) const;

    static void BindPipeline(void *context, rhi::GraphicsPipelineHandle pipeline);
    static void BindGroup(void *context, rhi::GraphicsPipelineHandle pipeline, uint32_t setIndex,
                          rhi::BindGroupHandle group);
    static void PushConstants(void *context, rhi::GraphicsPipelineHandle pipeline, rhi::ShaderStage stages,
                              uint32_t byteSize, const void *data);
    static void Draw(void *context, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex,
                     uint32_t firstInstance);
    static void DrawIndirect(void *context, rhi::BufferHandle arguments, uint64_t offset, uint32_t drawCount,
                             uint32_t stride);

    static bool CanRecordCompute(void *context, rhi::ComputePipelineHandle pipeline, rhi::BindGroupHandle group);
    static void BindComputePipeline(void *context, rhi::ComputePipelineHandle pipeline);
    static void BindComputeGroup(void *context, rhi::ComputePipelineHandle pipeline, uint32_t setIndex,
                                 rhi::BindGroupHandle group);
    static void PushComputeConstants(void *context, rhi::ComputePipelineHandle pipeline, uint32_t byteSize,
                                     const void *data);
    static void Dispatch(void *context, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ);
    static void DispatchIndirect(void *context, rhi::BufferHandle arguments, uint64_t offset);

    static void CopyBuffer(void *context, rhi::BufferHandle source, rhi::BufferHandle destination,
                           const rhi::BufferCopyRegion &region);
    static bool FillBuffer(void *context, rhi::BufferHandle destination, uint64_t offset, uint64_t byteSize,
                           uint32_t value);
    static bool UpdateBuffer(void *context, rhi::BufferHandle destination, uint64_t offset, const void *data,
                             uint64_t byteSize);
    static void CopyTexture(void *context, rhi::TextureHandle source, rhi::TextureHandle destination,
                            const rhi::TextureCopyRegion &region);
    static bool CopyTextureToBuffer(void *context, rhi::TextureHandle source, rhi::BufferHandle destination,
                                    const rhi::TextureBufferCopyRegion &region);
    static bool CopyBufferToTexture(void *context, rhi::BufferHandle source, rhi::TextureHandle destination,
                                    const rhi::TextureBufferCopyRegion &region);
    static bool CopyTextureBuffer(void *context, rhi::TextureHandle texture, rhi::BufferHandle buffer,
                                  const rhi::TextureBufferCopyRegion &region, bool toBuffer);
    static void ResolveTexture(void *context, rhi::TextureHandle source, rhi::TextureHandle destination,
                               const rhi::TextureResolveRegion &region);

    static const rhi::GraphicsCommandEncoder::Dispatch s_graphicsDispatch;
    static const rhi::ComputeCommandEncoder::DispatchTable s_computeDispatch;
    static const rhi::TransferCommandEncoder::DispatchTable s_transferDispatch;

    rhi::DeviceId m_deviceId = rhi::InvalidDeviceId;
    VkDevice m_device = VK_NULL_HANDLE;
    VkPipelineCache m_pipelineCache = VK_NULL_HANDLE;
    std::string m_pipelineCachePath;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    std::function<VkDeviceSize()> m_managedDeviceLocalLimit;
    using HeapBytes = std::array<VkDeviceSize, VK_MAX_MEMORY_HEAPS>;
    bool AdmitHeapAllocations(const VkPhysicalDeviceMemoryProperties &memory, const HeapBytes &requested);
    bool ReclaimStaging(const VkPhysicalDeviceMemoryProperties &memory, HeapBytes needed, VkDeviceSize localNeeded);
    std::map<std::pair<VkDeviceSize, StagingAccess>, std::vector<std::shared_ptr<VkBufferHandle>>> m_stagingPool;
    StagingPoolStatistics m_stagingStats;
    VkDeviceSize m_managedOtherHeapLimit = 0;
    VkDeviceSize m_deviceLocalAllocationPressure = 0;
    rhi::DeviceCaps m_capabilities{};
    vk::VulkanFeatureState m_capabilityState{};
    uint32_t m_graphicsQueueFamily = 0;
    uint32_t m_computeQueueFamily = 0;
    uint32_t m_transferQueueFamily = 0;
    std::vector<Slot<BufferPayload>> m_buffers;
    std::vector<Slot<TexturePayload>> m_textures;
    std::vector<Slot<TextureViewPayload>> m_textureViews;
    std::vector<Slot<SamplerPayload>> m_samplers;
    std::vector<Slot<ShaderModulePayload>> m_shaderModules;
    std::vector<Slot<BindingLayoutPayload>> m_bindingLayouts;
    std::vector<Slot<BindGroupPayload>> m_bindGroups;
    std::vector<Slot<GraphicsPipelinePayload>> m_graphicsPipelines;
    std::vector<Slot<GraphicsPipelinePayload>> m_computePipelines;
    std::vector<Slot<VkRenderPass>> m_renderTargetLayouts;
    VkDescriptorManager m_descriptorManager;
    GpuRetirementQueue m_resourceRetirementQueue;
    std::shared_ptr<rhi::DeviceLifetime> m_lifetime = std::make_shared<rhi::DeviceLifetime>();
    rhi::BindlessTextureTableBinding m_bindlessTextureTableBinding{};
    BindlessTexturePublisher m_bindlessTexturePublisher;
    BindlessTextureUsageMarker m_bindlessTextureUsageMarker;

    uint32_t m_freeBuffer = UINT32_MAX;
    uint32_t m_freeTexture = UINT32_MAX;
    uint32_t m_freeTextureView = UINT32_MAX;
    uint32_t m_freeSampler = UINT32_MAX;
    uint32_t m_freeShaderModule = UINT32_MAX;
    uint32_t m_freeBindingLayout = UINT32_MAX;
    uint32_t m_freeBindGroup = UINT32_MAX;
    uint32_t m_freeGraphicsPipeline = UINT32_MAX;
    uint32_t m_freeComputePipeline = UINT32_MAX;
    uint32_t m_freeRenderTargetLayout = UINT32_MAX;
};

} // namespace infernux::vk
