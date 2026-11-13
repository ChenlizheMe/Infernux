#include "FullscreenRenderer.h"

#include "InxVkCoreModular.h"
#include "vk/VulkanRhiDevice.h"

#include <algorithm>
#include <core/error/InxError.h>
#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/AssetFormatRegistry.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/ShaderAsset/ShaderAsset.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace infernux
{

std::shared_ptr<const FullscreenProgram> InxVkCoreModular::AcquireFullscreenProgram(const std::string &name)
{
    auto &registry = AssetRegistry::Instance();
    auto *database = registry.GetAssetDatabase();
    auto path = database->GetPathFromGuid(name);
    if (path.empty())
        path = asset_formats::FindShaderSourceKind(name) ? name : database->FindShaderPathById(name, "fragment");
    if (!asset_formats::FindShaderSourceKind(path))
        return {};
    const auto asset = registry.LoadAssetByPath<ShaderAsset>(path, ResourceType::Shader);
    if (!asset || !asset->program)
        throw std::runtime_error("Fullscreen author program is unavailable: " + path);
    const auto target = std::find_if(asset->program->targets.begin(), asset->program->targets.end(),
                                     [](const auto &code) { return code.target == ShaderCodeTarget::SpirV; });
    if (target == asset->program->targets.end())
        throw std::logic_error("Vulkan shader asset has no SPIR-V target: " + path);
    auto result = std::make_shared<FullscreenProgram>();
    result->assetGuid = database->GetGuidFromPath(path);
    result->shaderId = asset->shaderId;
    result->code = std::shared_ptr<const ShaderTargetCode>(asset->program, &*target);
    result->schema = asset->authoringSchema;
    return result;
}

namespace
{

class VulkanFullscreenRendererHost final : public FullscreenRendererHost
{
  public:
    explicit VulkanFullscreenRendererHost(InxVkCoreModular &core)
        : m_core(core), m_device(core.GetDeviceContext().GetRhiDevice())
    {
        m_perViewLayout = m_device.RegisterBindingLayout(core.GetPerViewDescSetLayout());
    }

    ~VulkanFullscreenRendererHost() override
    {
        m_device.Release(m_perViewLayout);
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

    [[nodiscard]] rhi::BindingLayoutHandle GetPerViewLayout() const noexcept override
    {
        return m_perViewLayout;
    }

    void ReportInfo(const std::string &message) override
    {
        INXLOG_INFO(message);
    }

  private:
    InxVkCoreModular &m_core;
    vk::VulkanRhiDevice &m_device;
    rhi::BindingLayoutHandle m_perViewLayout;
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
