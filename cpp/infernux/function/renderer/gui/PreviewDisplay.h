#pragma once

#include "PreviewImage.h"
#include <function/renderer/FullscreenRenderer.h>
#include <function/renderer/vk/RhiVulkanTypes.h>

namespace infernux
{
class InxVkCoreModular;

// One descriptor pool per preview producer; immutable pipelines remain in the
// existing device cache. Prepare is called only after its previous submission.
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
    rhi::BindGroupHandle m_inputs;
    uint64_t m_viewId;
    uint64_t m_sequence = 0;
    bool m_hdr = false;
};
} // namespace infernux
