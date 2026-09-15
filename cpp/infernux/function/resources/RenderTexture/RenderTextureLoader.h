#pragma once

#include <function/renderer/rhi/RhiRenderTexture.h>
#include <function/resources/AssetRegistry/IAssetLoader.h>
#include <functional>

namespace infernux
{
// Imported CPU description only. GPU target ownership belongs to the renderer,
// Discovery and cooking never allocate a graphics device. Reimport can update
// an already-resident GPU owner before publishing the CPU description.
class RenderTextureLoader final : public IAssetLoader
{
  public:
    using BeforePublish = std::function<void(const std::string &, const rhi::RenderTextureDesc &)>;
    explicit RenderTextureLoader(BeforePublish beforePublish = {}) : m_beforePublish(std::move(beforePublish))
    {
    }
    RuntimeAssetPayload Load(const std::string &filePath, const std::string &guid, AssetDatabase *adb) override;
    [[nodiscard]] bool SupportsWorkerLoad() const noexcept override
    {
        return true;
    }
    bool Reload(const RuntimeAssetPayload &existing, const std::string &filePath, const std::string &guid,
                AssetDatabase *adb) override;
    [[nodiscard]] size_t EstimateRuntimeBytes(const RuntimeAssetPayload &payload) const override;
    std::set<std::string> ScanDependencies(const std::string &, AssetDatabase *) override
    {
        return {};
    }
    void CreateMeta(const char *content, size_t size, const std::string &path, InxResourceMeta &meta) const override;

  private:
    BeforePublish m_beforePublish;
};
} // namespace infernux
