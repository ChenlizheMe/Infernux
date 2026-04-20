#pragma once

#include <function/renderer/vk/VkHandle.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxMesh/InxMesh.h>
#include <memory>
#include <vector>

namespace infernux
{

class InxVkCoreModular;
class AssetDatabase;

/// @brief GPU-based mesh preview renderer.
///
/// Renders an arbitrary InxMesh (with per-submesh materials) into a small
/// offscreen framebuffer and reads back RGBA8 pixels for editor thumbnails.
/// Camera is auto-fitted to the mesh AABB.
///
/// Reuses the same render-pass format as GPUMaterialPreview so material
/// pipelines created for either previewer are compatible.
class GPUMeshPreview
{
  public:
    explicit GPUMeshPreview(InxVkCoreModular *vkCore);
    ~GPUMeshPreview();

    GPUMeshPreview(const GPUMeshPreview &) = delete;
    GPUMeshPreview &operator=(const GPUMeshPreview &) = delete;

    /// @brief Render a mesh with its materials and return RGBA8 pixels.
    /// @param mesh       The mesh to render (must have valid vertices/indices).
    /// @param materials  Per-submesh materials. Index matches SubMesh::materialSlot.
    ///                   Use nullptr entries for missing materials (will use default).
    /// @param size       Output image width and height (square).
    /// @param outPixels  Receives size*size*4 bytes of RGBA8 pixel data.
    /// @return true on success.
    bool RenderToPixels(const InxMesh &mesh, const std::vector<std::shared_ptr<InxMaterial>> &materials, int size,
                        std::vector<unsigned char> &outPixels);

  private:
    bool EnsureResources(int size);
    void CreateRenderPass();
    void CreateFramebuffer(int size);
    void DestroyFramebuffer();

    InxVkCoreModular *m_vkCore = nullptr;
    int m_currentSize = 0;

    VkRenderPass m_renderPass = VK_NULL_HANDLE;

    vk::VkImageHandle m_msaaColor;
    vk::VkImageHandle m_resolveColor;
    vk::VkImageHandle m_depth;

    VkFramebuffer m_framebuffer = VK_NULL_HANDLE;
    vk::VkBufferHandle m_staging;

    VkDescriptorSet m_fallbackShadowDescSet = VK_NULL_HANDLE;

    VkFormat m_colorFormat = VK_FORMAT_UNDEFINED;
    VkFormat m_depthFormat = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits m_sampleCount = VK_SAMPLE_COUNT_1_BIT;
};

} // namespace infernux
