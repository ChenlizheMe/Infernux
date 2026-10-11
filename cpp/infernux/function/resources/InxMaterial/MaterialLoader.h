#pragma once

#include <function/resources/AssetRegistry/IAssetLoader.h>

namespace infernux
{

class MaterialLoader final : public IAssetLoader
{
  public:
    RuntimeAssetPayload Load(const std::string &filePath, const std::string &guid, AssetDatabase *adb) override;
    [[nodiscard]] bool SupportsWorkerLoad() const noexcept override
    {
        return true;
    }
    bool Reload(const RuntimeAssetPayload &existing, const std::string &filePath, const std::string &guid,
                AssetDatabase *adb) override;
    [[nodiscard]] size_t EstimateRuntimeBytes(const RuntimeAssetPayload &payload) const override;
    std::set<std::string> ScanDependencies(const std::string &filePath, AssetDatabase *adb) override;

    void CreateMeta(const char *content, size_t contentSize, const std::string &filePath,
                    InxResourceMeta &metaData) const override;

  private:
    static bool PrepareDocument(class InxMaterial &staged, const std::string &filePath, AssetDatabase *adb);
    static void RegisterDependencies(const std::string &materialGuid, const class InxMaterial &mat, AssetDatabase *adb);
};

} // namespace infernux
