#include "FullscreenRenderer.h"

#include "InxVkCoreModular.h"
#include "shader/ShaderReflection.h"
#include "vk/VulkanRhiDevice.h"

#include <algorithm>
#include <core/error/InxError.h>
#include <memory>
#include <string>
#include <vector>

namespace infernux
{

namespace
{

class VulkanFullscreenRendererHost final : public FullscreenRendererHost
{
  public:
    explicit VulkanFullscreenRendererHost(InxVkCoreModular &core)
        : m_core(core), m_device(core.GetDeviceContext().GetRhiDevice())
    {
        m_perViewLayout = m_device.RegisterBindingLayout(core.GetPerViewDescSetLayout());
        m_globalsLayout = m_device.RegisterBindingLayout(core.GetGlobalsDescSetLayout());
        m_globalsGroups.resize(std::max(1u, core.GetMaxFramesInFlight()));
        m_nativeGlobals.resize(m_globalsGroups.size(), VK_NULL_HANDLE);
    }

    ~VulkanFullscreenRendererHost() override
    {
        for (const auto group : m_globalsGroups)
            m_device.Release(group);
        m_device.Release(m_perViewLayout);
        m_device.Release(m_globalsLayout);
    }

    [[nodiscard]] rhi::Device &GetRhiDevice() noexcept override
    {
        return m_device;
    }

    [[nodiscard]] uint32_t GetFrameCount() const noexcept override
    {
        return static_cast<uint32_t>(m_globalsGroups.size());
    }

    [[nodiscard]] uint32_t GetCurrentFrame() const noexcept override
    {
        return m_core.GetCurrentFrameSlot() % static_cast<uint32_t>(m_globalsGroups.size());
    }

    [[nodiscard]] rhi::ShaderModuleHandle AcquireShaderModule(const std::string &name, rhi::ShaderStage stage,
                                                              uint32_t inputCount, uint32_t inputBufferMask) override
    {
        const char *type = stage == rhi::ShaderStage::Vertex     ? "vertex"
                           : stage == rhi::ShaderStage::Fragment ? "fragment"
                                                                 : nullptr;
        if (!type || !m_core.EnsureShaderAvailable(name, type))
            return {};
        if (stage == rhi::ShaderStage::Fragment) {
            // Once per pipeline creation/publication, not per frame. A hot-edited
            // shader must not read descriptors absent from its still-live graph.
            const auto *code = m_core.GetShaderCache().FindFragCode(name);
            ShaderReflection reflection;
            if (!code || !reflection.Reflect(*code, VK_SHADER_STAGE_FRAGMENT_BIT))
                return {};
            if (reflection.RequiresSampleRateShading() &&
                !m_core.GetDeviceContext().GetDeviceFeatures().sampleRateShading) {
                ReportError("Fullscreen shader '" + name + "' requires sample-rate shading unsupported by this device");
                return {};
            }
            for (const auto &binding : reflection.GetDescriptorSetLayoutBindings(0)) {
                const bool storageBuffer = binding.binding < 32 && (inputBufferMask & (1u << binding.binding)) != 0;
                const VkDescriptorType expected =
                    storageBuffer ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                if (binding.descriptorType != expected || binding.descriptorCount != 1 ||
                    binding.binding >= inputCount) {
                    ReportError("Fullscreen shader '" + name + "' has an incompatible input layout at binding " +
                                std::to_string(binding.binding) + "; update the graph's resource inputs");
                    return {};
                }
            }
        }
        return m_device.RegisterShaderModule(m_core.GetShaderModule(name, type));
    }

    [[nodiscard]] rhi::BindingLayoutHandle GetPerViewLayout() const noexcept override
    {
        return m_perViewLayout;
    }

    [[nodiscard]] rhi::BindingLayoutHandle GetGlobalsLayout() const noexcept override
    {
        return m_globalsLayout;
    }

    [[nodiscard]] rhi::BindGroupHandle GetCurrentGlobalsGroup() override
    {
        const uint32_t frame = GetCurrentFrame();
        const VkDescriptorSet native = m_core.GetCurrentGlobalsDescSet();
        if (m_nativeGlobals[frame] != native) {
            m_device.Release(m_globalsGroups[frame]);
            m_globalsGroups[frame] = m_device.RegisterBindGroup(native);
            m_nativeGlobals[frame] = native;
        }
        return m_globalsGroups[frame];
    }

    void ReportError(const std::string &message) override
    {
        INXLOG_ERROR(message);
    }

    void ReportInfo(const std::string &message) override
    {
        INXLOG_INFO(message);
    }

  private:
    InxVkCoreModular &m_core;
    vk::VulkanRhiDevice &m_device;
    rhi::BindingLayoutHandle m_perViewLayout;
    rhi::BindingLayoutHandle m_globalsLayout;
    std::vector<rhi::BindGroupHandle> m_globalsGroups;
    std::vector<VkDescriptorSet> m_nativeGlobals;
};

} // namespace

void FullscreenRenderer::Initialize(InxVkCoreModular *vkCore)
{
    if (!vkCore) {
        Destroy();
        return;
    }
    Initialize(std::make_shared<VulkanFullscreenRendererHost>(*vkCore));
}

} // namespace infernux
