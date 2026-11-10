#pragma once

#include <function/renderer/rhi/RenderViewContext.h>
#include <function/renderer/vk/RhiVulkanTypes.h>
#include <function/renderer/vk/VkDescriptorManager.h>
#include <function/renderer/vk/VkHandle.h>
#include <memory>
#include <vector>

namespace infernux
{

namespace vk
{
class GraphicsSubmissionTicket;
class BufferUploadTicket;
} // namespace vk

class InxVkCoreModular;
class InxMaterial;
class PreviewDisplay;
struct PreviewImage;

/// @brief GPU-based material preview renderer.
/// Uses the real material pipeline (vertex + fragment shaders) to render a
/// lit sphere into a private display texture, without a CPU pixel roundtrip.
class GPUMaterialPreview
{
  public:
    explicit GPUMaterialPreview(InxVkCoreModular *vkCore);
    ~GPUMaterialPreview();

    GPUMaterialPreview(const GPUMaterialPreview &) = delete;
    GPUMaterialPreview &operator=(const GPUMaterialPreview &) = delete;

    [[nodiscard]] std::shared_ptr<PreviewImage> BeginRender(InxMaterial &material, int size, bool &progressed,
                                                            bool *texturePending = nullptr);

    [[nodiscard]] const rhi::RenderViewContext &GetRenderViewContext() const noexcept
    {
        return m_renderView;
    }

  private:
    bool EnsureResources(int size, bool &progressed);
    bool EnsureViewResources(bool &progressed);
    void DestroyViewResources();
    void CreateAttachments(int size);
    bool PrepareSphereBuffers(bool &progressed);
    void DestroyAttachments();
    void PublishRenderView();
    void UnpublishRenderView();

    InxVkCoreModular *m_vkCore = nullptr;
    rhi::RenderViewContext m_renderView;
    int m_currentSize = 0;

    rhi::DynamicRenderingCommands m_dynamicRenderingCommands;

    // MSAA color attachment
    vk::VkImageHandle m_msaaColor;
    // Resolved (1x) input to the display pass
    vk::VkImageHandle m_resolveColor;
    // MSAA depth attachment
    vk::VkImageHandle m_depth;
    VkImageLayout m_msaaColorLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout m_resolveColorLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout m_depthLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    // Default per-view shadow descriptor used when no active scene descriptor
    // is available but the shader statically uses set 1.
    VkDescriptorSet m_fallbackShadowDescSet = VK_NULL_HANDLE;
    vk::DescriptorLease m_fallbackShadowDescLease;

    // Sphere geometry
    std::shared_ptr<vk::VkBufferHandle> m_sphereVBO;
    std::shared_ptr<vk::BufferUploadTicket> m_sphereVertexUpload, m_sphereIndexUpload;
    std::shared_ptr<vk::VkBufferHandle> m_sphereIBO;
    uint32_t m_sphereIndexCount = 0;

    std::unique_ptr<vk::VkBufferHandle> m_previewSceneUbo;
    std::unique_ptr<vk::VkBufferHandle> m_previewLightingUbo;
    std::unique_ptr<vk::VkBufferHandle> m_previewGlobalsUbo;
    std::unique_ptr<vk::VkBufferHandle> m_previewInstanceBuffer;
    std::unique_ptr<vk::VkBufferHandle> m_previewSkinInstanceBuffer;
    std::unique_ptr<vk::VkBufferHandle> m_previewSkinPaletteBuffer;
    std::unique_ptr<vk::VkBufferHandle> m_previewInstanceAuxBuffer;
    VkDescriptorSet m_previewGlobalsSet = VK_NULL_HANDLE;
    vk::DescriptorLease m_previewGlobalsLease;
    std::shared_ptr<vk::GraphicsSubmissionTicket> m_activeSubmission;
    uint64_t m_gpuNanoseconds = 0;
    std::unique_ptr<PreviewDisplay> m_display;

    // Cached format info
    VkFormat m_colorFormat = VK_FORMAT_UNDEFINED;
    VkFormat m_depthFormat = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits m_sampleCount = VK_SAMPLE_COUNT_1_BIT;
};

} // namespace infernux
