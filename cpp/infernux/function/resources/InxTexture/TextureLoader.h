#pragma once

#include <function/resources/AssetRegistry/IAssetLoader.h>

namespace infernux
{
struct TextureCpuData;

class TextureLoader final : public IAssetLoader
{
  public:
    RuntimeAssetPayload Load(const std::string &filePath, const std::string &guid, AssetDatabase *adb) override;
    std::shared_ptr<const TextureCpuData> ReadPixels(const std::string &filePath, const std::string &guid,
                                                     AssetDatabase *adb);
    [[nodiscard]] bool SupportsWorkerLoad() const noexcept override
    {
        return true;
    }
    void OnRelocated(const RuntimeAssetPayload &payload, const std::string &filePath) override;
    bool Reload(const RuntimeAssetPayload &existing, const std::string &filePath, const std::string &guid,
                AssetDatabase *adb) override;
    [[nodiscard]] size_t EstimateRuntimeBytes(const RuntimeAssetPayload &payload) const override;
    std::set<std::string> ScanDependencies(const std::string &filePath, AssetDatabase *adb) override;

    void CreateMeta(AssetSource &source, InxResourceMeta &metaData) const override;
    static void CreateSourceMeta(AssetSource &source, InxResourceMeta &metaData, bool identityOnly = false);
};

} // namespace infernux
