#pragma once

#include <function/resources/AssetRegistry/IAssetLoader.h>

namespace infernux
{

class AudioClipLoader final : public IAssetLoader
{
  public:
    RuntimeAssetPayload Load(const std::string &filePath, const std::string &guid, AssetDatabase *adb) override;
    void OnRelocated(const RuntimeAssetPayload &payload, const std::string &filePath) override;
    [[nodiscard]] bool SupportsWorkerLoad() const noexcept override
    {
        return true;
    }
    bool Reload(const RuntimeAssetPayload &existing, const std::string &filePath, const std::string &guid,
                AssetDatabase *adb) override;
    [[nodiscard]] size_t EstimateRuntimeBytes(const RuntimeAssetPayload &payload) const override;
    std::set<std::string> ScanDependencies(const std::string &filePath, AssetDatabase *adb) override;

    void CreateMeta(AssetSource &source, InxResourceMeta &metaData) const override;
};

} // namespace infernux
