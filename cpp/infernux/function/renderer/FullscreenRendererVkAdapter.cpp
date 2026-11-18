#include "FullscreenRenderer.h"

#include "InxVkCoreModular.h"
#include "vk/VulkanRhiDevice.h"

#include <function/resources/ShaderAsset/FullscreenProgramAsset.h>

#include <core/error/InxError.h>
#include <memory>
#include <string>

namespace infernux
{

std::shared_ptr<const FullscreenProgram> InxVkCoreModular::AcquireFullscreenProgram(const std::string &name)
{
    return LoadFullscreenProgram(name, ShaderCodeTarget::SpirV);
}

namespace
{

class VulkanFullscreenRendererHost final : public FullscreenRendererHost
{
  public:
    explicit VulkanFullscreenRendererHost(InxVkCoreModular &core)
        : m_core(core), m_device(core.GetDeviceContext().GetRhiDevice())
    {
    }

    [[nodiscard]] rhi::Device &GetRhiDevice() noexcept override
    {
        return m_device;
    }

    [[nodiscard]] uint32_t GetFrameCount() const noexcept override
    {
        return m_core.GetMaxFramesInFlight();
    }

    [[nodiscard]] uint32_t GetCurrentFrame() const noexcept override
    {
        return m_core.GetCurrentFrameSlot();
    }

    [[nodiscard]] std::shared_ptr<const FullscreenProgram> AcquireProgram(const std::string &name) override
    {
        return m_core.AcquireFullscreenProgram(name);
    }

    void ReportInfo(const std::string &message) override
    {
        INXLOG_INFO(message);
    }

  private:
    InxVkCoreModular &m_core;
    vk::VulkanRhiDevice &m_device;
};

} // namespace

void FullscreenRenderer::Initialize(InxVkCoreModular *vkCore)
{
    if (!vkCore) {
        Destroy();
        return;
    }
    Initialize(std::make_shared<VulkanFullscreenRendererHost>(*vkCore), vkCore->GetFullscreenPipelineCache());
}

std::shared_ptr<FullscreenPipelineCache> InxVkCoreModular::GetFullscreenPipelineCache()
{
    if (!m_fullscreenPipelines)
        m_fullscreenPipelines =
            std::make_shared<FullscreenPipelineCache>(std::make_shared<VulkanFullscreenRendererHost>(*this));
    return m_fullscreenPipelines;
}

} // namespace infernux
