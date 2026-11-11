#pragma once

#include <function/resources/AssetRegistry/AssetRegistry.h>

namespace infernux
{

/**
 * @brief Runtime preparation and publication for GPU author assets.
 *
 * Static Python authors produce a plain source/field snapshot before native
 * Slang compilation. The asset retains both targets and its logical schema.
 *
 * Reload prepares a replacement before updating the published object. Device
 * programs and graph resources are published by the renderer's owner.
 */
class ShaderLoader final : public IAssetLoader
{
  public:
    RuntimeAssetPayload Load(const std::string &filePath, const std::string &guid, AssetDatabase *adb) override;
    void OnRelocated(const RuntimeAssetPayload &payload, const std::string &filePath) override;
    void OnPublished(const RuntimeAssetPayload &payload, const std::string &guid, AssetDatabase *adb) override;
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
