#pragma once

#include "PreviewImage.h"
#include <function/renderer/FullscreenRenderer.h>
#include <function/renderer/vk/RhiVulkanTypes.h>

namespace infernux
{
class InxVkCoreModular;

// Each producer owns its prepared parameters and transient bindings; immutable
// pipelines share the device cache. Prepare follows completion of its previous submission.
class PreviewDisplay
{
  public:
    PreviewDisplay(InxVkCoreModular &core, uint64_t viewId);
    std::shared_ptr<PreviewImage> Prepare(VkImageView source, VkFormat format, int outputSize);
    void Record(VkCommandBuffer commands, const PreviewImage &output, VkImage source, VkImageLayout &sourceLayout,
                int outputSize, float sphereRadius = 0.0f);

  private:
    InxVkCoreModular &m_core;
    FullscreenRenderer m_renderer;
    FullscreenPipelineEntry m_pipeline;
    std::shared_ptr<const FullscreenProgram> m_program;
    FullscreenParameters m_parameters;
    VkImageView m_sourceView = VK_NULL_HANDLE;
    rhi::PixelFormat m_sourceFormat = rhi::PixelFormat::Undefined;
    uint64_t m_viewId;
    uint64_t m_sequence = 0;
    bool m_hdr = false;
};
} // namespace infernux
