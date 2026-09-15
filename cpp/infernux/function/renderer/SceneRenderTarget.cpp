#include "SceneRenderTarget.h"
#include "InxVkCoreModular.h"
#include "rhi/GpuRetirementQueue.h"
#include "rhi/RhiRenderTexture.h"
#include "vk/RhiVulkanTypes.h"
#include "vk/VkDeviceContext.h"
#include "vk/VkRenderUtils.h"
#include "vk/VulkanRhiDevice.h"
#include <array>
#include <atomic>
#include <backends/imgui_impl_vulkan.h>
#include <core/log/InxLog.h>

namespace infernux
{
SceneRenderTarget::SceneRenderTarget(InxVkCoreModular *vkCore) : m_vkCore(vkCore)
{
}

SceneRenderTarget::~SceneRenderTarget()
{
    Cleanup();
}

bool SceneRenderTarget::Initialize(uint32_t width, uint32_t height)
{
    if (width == 0 || height == 0 || HasOwnedResources()) {
        INXLOG_ERROR("SceneRenderTarget requires an empty target and positive dimensions");
        return false;
    }
    m_width = width;
    m_height = height;
    try {
        auto &device = m_vkCore->GetDeviceContext().GetRhiDevice();
        static std::atomic<uint64_t> nextIdentity{1};
        const auto identity = "view-target/" + std::to_string(nextIdentity.fetch_add(1));
        rhi::RenderTextureDesc description;
        description.width = width;
        description.height = height;
        description.colorFormat = rhi::PixelFormat::RGBA16SFloat;
        const auto depthFormat = m_vkCore->GetDeviceContext().FindSampledDepthFormat();
        if (depthFormat == VK_FORMAT_UNDEFINED)
            throw std::runtime_error("No depth format supports attachment and sampled-image usage");
        description.depthFormat = rhi::FromVkFormat(depthFormat);
        description.sampledDepth = true;
        description.samples = rhi::FromVkSampleCount(m_msaaSampleCount);
        m_attachments = rhi::RenderTexture(device, identity, description).Acquire();

        rhi::RenderTextureDesc outline;
        outline.width = width;
        outline.height = height;
        m_outlineAttachments = rhi::RenderTexture(device, identity + "/outline", outline).Acquire();

        m_colorImage = device.Resolve(m_attachments->color->GetTexture());
        m_colorImageView = device.Resolve(m_attachments->color->GetView());
        m_sampler = device.Resolve(m_attachments->color->GetSampler());
        m_depthImage = device.Resolve(m_attachments->depth->GetTexture());
        m_depthImageView = device.Resolve(m_attachments->depth->GetView());
        if (m_attachments->multisampleColor) {
            m_msaaColorImage = device.Resolve(m_attachments->multisampleColor->GetTexture());
            m_msaaColorImageView = device.Resolve(m_attachments->multisampleColor->GetView());
        }
        const auto &mask = m_outlineAttachments->color;
        m_outlineMaskImage = device.Resolve(mask->GetTexture());
        m_outlineMaskImageView = device.Resolve(mask->GetView());
        m_outlineMaskSampler = device.Resolve(mask->GetSampler());

        // Imported Scene resources keep their existing initial-layout contract.
        // Establish all attachment layouts in one submission, not one per image.
        InitializeAttachmentLayouts();
        CreateImGuiDescriptor();
        m_isInitialized = true;
        return true;
    } catch (const std::exception &error) {
        INXLOG_ERROR("SceneRenderTarget initialization failed: ", error.what());
        CleanupResources();
        return false;
    }
}

void SceneRenderTarget::BindAttachments(std::shared_ptr<const rhi::RenderTextureGeneration> attachments)
{
    auto &device = m_vkCore->GetDeviceContext().GetRhiDevice();
    if (m_imguiDescriptorSet || m_outlineAttachments || !attachments || !attachments->depth ||
        attachments->color->GetTexture().Device() != device.GetDeviceId())
        throw std::invalid_argument("Camera output must be a depth-equipped RenderTexture on the view device");
    ClearBorrowedHandles();
    m_attachments = std::move(attachments);
    m_width = m_attachments->width;
    m_height = m_attachments->height;
    m_msaaSampleCount = rhi::ToVkSampleCount(m_attachments->description.samples);
    m_colorImage = device.Resolve(m_attachments->color->GetTexture());
    m_colorImageView = device.Resolve(m_attachments->color->GetView());
    m_sampler = device.Resolve(m_attachments->color->GetSampler());
    m_depthImage = device.Resolve(m_attachments->depth->GetTexture());
    m_depthImageView = device.Resolve(m_attachments->depth->GetView());
    if (m_attachments->multisampleColor) {
        m_msaaColorImage = device.Resolve(m_attachments->multisampleColor->GetTexture());
        m_msaaColorImageView = device.Resolve(m_attachments->multisampleColor->GetView());
    }
    m_isInitialized = true;
}

VkFormat SceneRenderTarget::GetColorFormat() const
{
    return m_attachments ? rhi::ToVkFormat(m_attachments->description.colorFormat) : VK_FORMAT_R16G16B16A16_SFLOAT;
}

VkFormat SceneRenderTarget::GetDepthFormat() const
{
    return m_attachments ? rhi::ToVkFormat(m_attachments->description.depthFormat)
                         : (m_vkCore ? m_vkCore->GetDeviceContext().FindSampledDepthFormat() : VK_FORMAT_D32_SFLOAT);
}

uint64_t SceneRenderTarget::GetResidentBytes() const
{
    return (m_attachments ? m_attachments->GetResidentBytes() : 0) +
           (m_outlineAttachments ? m_outlineAttachments->GetResidentBytes() : 0);
}

uint64_t SceneRenderTarget::GetMsaaColorResidentBytes() const
{
    return m_attachments && m_attachments->multisampleColor ? m_attachments->multisampleColor->GetResidentBytes() : 0;
}

void SceneRenderTarget::InitializeAttachmentLayouts()
{
    std::array<VkImageMemoryBarrier, 4> barriers;
    uint32_t count = 0;
    barriers[count++] =
        vkrender::MakeImageBarrier(m_colorImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                   VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_ACCESS_SHADER_READ_BIT);
    barriers[count++] = vkrender::MakeImageBarrier(
        m_depthImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        rhi::ToVkImageAspectMask(GetDepthFormat()), 0, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    barriers[count++] = vkrender::MakeImageBarrier(m_outlineMaskImage, VK_IMAGE_LAYOUT_UNDEFINED,
                                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT,
                                                   0, VK_ACCESS_SHADER_READ_BIT);
    if (m_msaaColorImage != VK_NULL_HANDLE) {
        barriers[count++] = vkrender::MakeImageBarrier(
            m_msaaColorImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
    }
    const auto command = m_vkCore->BeginSingleTimeCommands();
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         0, 0, nullptr, 0, nullptr, count, barriers.data());
    m_vkCore->EndSingleTimeCommands(command);
}

void SceneRenderTarget::CreateImGuiDescriptor()
{
    m_imguiDescriptorSet =
        ImGui_ImplVulkan_AddTexture(m_sampler, m_colorImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    if (m_imguiDescriptorSet == VK_NULL_HANDLE)
        throw std::runtime_error("Failed to create ImGui descriptor set for scene texture");
}

void SceneRenderTarget::ClearBorrowedHandles()
{
    m_imguiDescriptorSet = VK_NULL_HANDLE;
    m_sampler = VK_NULL_HANDLE;
    m_outlineMaskSampler = VK_NULL_HANDLE;
    m_outlineMaskImageView = VK_NULL_HANDLE;
    m_outlineMaskImage = VK_NULL_HANDLE;
    m_msaaColorImageView = VK_NULL_HANDLE;
    m_msaaColorImage = VK_NULL_HANDLE;
    m_depthImageView = VK_NULL_HANDLE;
    m_depthImage = VK_NULL_HANDLE;
    m_colorImageView = VK_NULL_HANDLE;
    m_colorImage = VK_NULL_HANDLE;
    m_width = m_height = 0;
    m_isInitialized = false;
}

void SceneRenderTarget::CleanupResources()
{
    // ImGui owns only its descriptor. RHI generations own every attachment and
    // retire their handles through the existing device completion mechanism.
    if (m_imguiDescriptorSet != VK_NULL_HANDLE)
        ImGui_ImplVulkan_RemoveTexture(m_imguiDescriptorSet);
    m_outlineAttachments.reset();
    m_attachments.reset();
    ClearBorrowedHandles();
}

bool SceneRenderTarget::HasOwnedResources() const noexcept
{
    return m_imguiDescriptorSet != VK_NULL_HANDLE || m_attachments || m_outlineAttachments;
}

void SceneRenderTarget::RetireResourcesAfter(GpuRetirementQueue &retirementQueue,
                                             rhi::SubmissionSerial retirementSerial)
{
    if (HasOwnedResources()) {
        retirementQueue.RetireAfter(retirementSerial,
                                    [descriptor = m_imguiDescriptorSet, attachments = std::move(m_attachments),
                                     outline = std::move(m_outlineAttachments)]() mutable {
                                        if (descriptor != VK_NULL_HANDLE)
                                            ImGui_ImplVulkan_RemoveTexture(descriptor);
                                        outline.reset();
                                        attachments.reset();
                                    });
    }
    ClearBorrowedHandles();
}

void SceneRenderTarget::Cleanup()
{
    if (m_vkCore && m_vkCore->GetDevice() != VK_NULL_HANDLE && HasOwnedResources()) {
        if (m_imguiDescriptorSet && !m_vkCore->IsShuttingDown())
            m_vkCore->GetDeviceContext().WaitIdle();
        CleanupResources();
    }
}
} // namespace infernux
