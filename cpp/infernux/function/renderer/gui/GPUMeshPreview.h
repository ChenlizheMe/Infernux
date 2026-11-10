#pragma once

#include "MeshPreviewPreparation.h"
#include <function/renderer/rhi/RenderViewContext.h>
#include <function/renderer/vk/RhiVulkanTypes.h>
#include <function/renderer/vk/VkDescriptorManager.h>
#include <function/renderer/vk/VkHandle.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxMesh/InxMesh.h>
#include <glm/glm.hpp>
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
class AssetDatabase;
class PreviewDisplay;
struct PreviewImage;

/// An individual request retains its source and byte cursors across advances.
/// Cancelling it drops unpublished buffers; accepted blocks retire in the RM.
struct PreviewGeometryUpload
{
    std::shared_ptr<const MeshGeometry> geometry;
    std::shared_ptr<const InxSkinnedMesh> skin;
    MeshIndexFormat format = MeshIndexFormat::Auto;
    VkIndexType indexType = VK_INDEX_TYPE_UINT32;
    std::shared_ptr<vk::BufferUploadTicket> vertices;
    std::shared_ptr<vk::BufferUploadTicket> indices;
};

/// @brief GPU-based mesh preview renderer.
///
/// Renders an arbitrary InxMesh (with per-submesh materials) into a small
/// offscreen attachment set and publishes independent GPU editor thumbnails.
/// Camera is auto-fitted to the mesh AABB.
///
/// Reuses the same attachment format as GPUMaterialPreview so material
/// pipelines created for either previewer are compatible.
class GPUMeshPreview
{
  public:
    explicit GPUMeshPreview(InxVkCoreModular *vkCore);
    ~GPUMeshPreview();

    GPUMeshPreview(const GPUMeshPreview &) = delete;
    GPUMeshPreview &operator=(const GPUMeshPreview &) = delete;

    bool PrepareMaterials(const std::vector<std::shared_ptr<InxMaterial>> &sources,
                          std::vector<std::shared_ptr<InxMaterial>> &prepared, bool &progressed);

    [[nodiscard]] std::shared_ptr<PreviewImage> BeginRender(const InxMesh &mesh,
                                                            const std::vector<std::shared_ptr<InxMaterial>> &materials,
                                                            int size, std::shared_ptr<PreviewGeometryUpload> &geometry,
                                                            bool &progressed, bool &materialsChanged);

    /// @brief Live editor preview: render directly into a GPU image and return an
    ///        ImGui texture id.  Avoids CPU readback + re-upload every frame
    ///        (same idea as SceneRenderTarget).  Uses scene MSAA/format settings.
    uint64_t RenderToImGuiTextureCamera(const InxMesh &mesh, const std::vector<std::shared_ptr<InxMaterial>> &materials,
                                        int size, const glm::mat4 &view, const glm::mat4 &proj,
                                        const glm::vec3 &cameraPos, std::shared_ptr<PreviewGeometryUpload> &geometry,
                                        bool &progressed, bool cloneMaterials = false,
                                        const std::vector<glm::mat4> *bonePalette = nullptr);

    uint64_t RenderAnimation(const std::shared_ptr<InxMesh> &mesh, const std::string &take, float seconds, int size,
                             uint64_t dependencyRevision);

    /// @brief Currently-published ImGui display descriptor, 0 when absent.
    ///
    /// The display target is destroyed and recreated on size/format changes,
    /// so any texture id previously returned by RenderToImGuiTextureCamera
    /// must be validated against this before being drawn again.
    [[nodiscard]] uint64_t GetDisplayTextureId() const
    {
        return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(m_displayDescriptorSet));
    }

    [[nodiscard]] const rhi::RenderViewContext &GetRenderViewContext() const noexcept
    {
        return m_renderView;
    }

  private:
    PreviewAssetLoads m_animationMaterialLoads;
    bool m_animationMaterialsReady = false;
    bool EnsureResources(int size, bool &progressed);
    bool PrepareGeometry(const InxMesh &mesh, std::shared_ptr<PreviewGeometryUpload> &upload, bool &progressed,
                         const std::shared_ptr<const InxSkinnedMesh> &skin = {});
    bool EnsureViewResources(bool &progressed);
    void DestroyViewResources();
    void EnsureImGuiDisplayDescriptor();
    void DestroyImGuiDisplayDescriptor();
    void CreateAttachments(int size);
    void DestroyAttachments();
    void PublishRenderView();
    void UnpublishRenderView();

    InxVkCoreModular *m_vkCore = nullptr;
    rhi::RenderViewContext m_renderView;
    int m_currentSize = 0;

    rhi::DynamicRenderingCommands m_dynamicRenderingCommands;

    vk::VkImageHandle m_msaaColor;
    vk::VkImageHandle m_resolveColor;
    vk::VkImageHandle m_depth;
    VkImageLayout m_msaaColorLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout m_resolveColorLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout m_depthLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkDescriptorSet m_fallbackShadowDescSet = VK_NULL_HANDLE;
    vk::DescriptorLease m_fallbackShadowDescLease;

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

    // Isolated animation preview retains one published source and GPU geometry.
    std::shared_ptr<InxMesh> m_animationMesh;
    uint64_t m_animationGeneration = 0;
    uint64_t m_animationRevision = 0;
    std::vector<std::shared_ptr<InxMaterial>> m_animationMaterials;
    std::shared_ptr<PreviewGeometryUpload> m_animationGeometry;
    std::string m_framedTake;
    glm::vec3 m_animationFrameMin{0.0f};
    glm::vec3 m_animationFrameMax{0.0f};
    std::string m_renderedTake;
    float m_renderedSeconds = -1.0f;
    int m_renderedSize = 0;

    VkSampler m_displaySampler = VK_NULL_HANDLE;
    VkDescriptorSet m_displayDescriptorSet = VK_NULL_HANDLE;
    bool m_displayImageShaderReady = false;

    VkFormat m_colorFormat = VK_FORMAT_UNDEFINED;
    VkFormat m_depthFormat = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits m_sampleCount = VK_SAMPLE_COUNT_1_BIT;
};

} // namespace infernux
