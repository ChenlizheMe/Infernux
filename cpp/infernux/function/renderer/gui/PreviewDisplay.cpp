#include "PreviewDisplay.h"

#include <function/renderer/InxVkCoreModular.h>
#include <function/renderer/rhi/RhiRenderTexture.h>
#include <function/renderer/vk/VkRenderUtils.h>
#include <function/renderer/vk/VkResourceManager.h>
#include <function/renderer/vk/VulkanRhiDevice.h>
#include <stdexcept>

namespace infernux
{
bool PreviewImage::IsComplete() const noexcept
{
    return submission && submission->IsComplete();
}

PreviewDisplay::PreviewDisplay(InxVkCoreModular &core, uint64_t viewId) : m_core(core), m_viewId(viewId)
{
    m_renderer.Initialize(&core);
}

std::shared_ptr<PreviewImage> PreviewDisplay::Prepare(VkImageView source, VkFormat format, int outputSize)
{
    // A restored asset catalog may still be discovering a newly installed
    // builtin shader. Do not cache an empty pipeline as a completed product.
    const auto program = m_renderer.AcquireProgram("infernux/internal/preview_display");
    if (!program)
        return nullptr;
    if (!m_program || program->code != m_program->code)
        m_parameters = FullscreenRenderer::PrepareParameters(*program, {{"sphere_radius", 0.0f}, {"hdr", false}});
    m_program = program;
    m_renderer.ResetPool();
    FullscreenPipelineKey key;
    key.shaderName = program->shaderId;
    key.program = program;
    key.colorFormat = rhi::PixelFormat::RGBA8Srgb;
    key.useDynamicRendering = true;
    key.inputResourceCount = 1;
    m_pipeline = m_renderer.EnsurePipeline(key);
    if (!m_pipeline.pipeline.IsValid())
        throw std::runtime_error("Preview display pipeline is unavailable");

    auto &device = m_core.GetDeviceContext().GetRhiDevice();
    m_sourceView = source;
    m_sourceFormat = rhi::FromVkFormat(format);

    rhi::RenderTextureDesc description;
    description.width = description.height = static_cast<uint32_t>(outputSize);
    description.colorFormat = key.colorFormat;
    const std::string sourceId = "Preview/" + std::to_string(m_viewId) + '/' + std::to_string(++m_sequence);
    rhi::RenderTexture target(device, sourceId, description);
    auto output = std::make_shared<PreviewImage>();
    output->texture = target.Acquire()->sampledColor;
    m_hdr = format == VK_FORMAT_R16G16B16A16_SFLOAT || format == VK_FORMAT_R32G32B32A32_SFLOAT;
    return output;
}

void PreviewDisplay::Record(VkCommandBuffer commands, const PreviewImage &output, VkImage source,
                            VkImageLayout &sourceLayout, int outputSize, float sphereRadius)
{
    auto &device = m_core.GetDeviceContext().GetRhiDevice();
    vkrender::TransitionTrackedImageLayout(commands, source, VK_IMAGE_ASPECT_COLOR_BIT, sourceLayout,
                                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    VkImageLayout targetLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    const VkImage target = device.Resolve(output.texture->GetTexture());
    vkrender::TransitionTrackedImageLayout(commands, target, VK_IMAGE_ASPECT_COLOR_BIT, targetLayout,
                                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    color.imageView = device.Resolve(output.texture->GetView());
    color.imageLayout = targetLayout;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
    rendering.renderArea.extent = {static_cast<uint32_t>(outputSize), static_cast<uint32_t>(outputSize)};
    rendering.layerCount = 1;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments = &color;
    const auto dynamic = rhi::ResolveDynamicRenderingCommands(m_core.GetDevice());
    dynamic.begin(commands, &rendering);
    const VkViewport viewport{0.0f, 0.0f, static_cast<float>(outputSize), static_cast<float>(outputSize), 0.0f, 1.0f};
    const VkRect2D scissor{{0, 0}, rendering.renderArea.extent};
    vkCmdSetViewport(commands, 0, 1, &viewport);
    vkCmdSetScissor(commands, 0, 1, &scissor);
    vk::VulkanGraphicsCommandContext context;
    auto encoder = device.MakeGraphicsCommandEncoder(context, commands);
    FullscreenResourceInput input;
    input.view = device.RegisterTextureView(m_sourceView);
    input.format = m_sourceFormat;
    const ShaderParameterValue values[] = {sphereRadius, m_hdr};
    try {
        m_renderer.DrawProgram(encoder, m_pipeline, &input, m_parameters, values);
    } catch (...) {
        device.Release(input.view);
        dynamic.end(commands);
        throw;
    }
    // Only the temporary RHI registration is released here. The producer owns
    // the scratch VkImageView through completion of this preview submission.
    device.Release(input.view);
    dynamic.end(commands);
    vkrender::TransitionTrackedImageLayout(commands, target, VK_IMAGE_ASPECT_COLOR_BIT, targetLayout,
                                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}
} // namespace infernux
