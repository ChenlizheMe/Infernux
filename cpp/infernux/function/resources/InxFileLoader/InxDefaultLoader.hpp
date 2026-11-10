#pragma once

#include <function/resources/AssetRegistry/IAssetLoader.h>

namespace infernux
{
class InxDefaultLoader final : public IAssetLoader
{
  public:
    explicit InxDefaultLoader(ResourceType resourceType = ResourceType::DefaultText);
    void CreateMeta(AssetSource &source, InxResourceMeta &metadata) const override;
    RuntimeAssetPayload Load(const std::string &path, const std::string &guid, AssetDatabase *database) override;
    [[nodiscard]] bool SupportsWorkerLoad() const noexcept override;
    WorkerAssetLoadPlan PlanWorkerLoad(const std::string &path, const std::string &, AssetDatabase *) override;
    [[nodiscard]] AssetReloadMode GetReloadMode() const noexcept override
    {
        return AssetReloadMode::Invalidate;
    }
    bool Reload(const RuntimeAssetPayload &, const std::string &, const std::string &, AssetDatabase *) override
    {
        return false; // The registry invalidates these immutable revisions instead.
    }
    [[nodiscard]] size_t EstimateRuntimeBytes(const RuntimeAssetPayload &payload) const override;
    std::set<std::string> ScanDependencies(const std::string &, AssetDatabase *) override
    {
        return {};
    }

  private:
    ResourceType m_resourceType;
};
} // namespace infernux
