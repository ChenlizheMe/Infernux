#pragma once

#include <function/resources/AssetRegistry/AssetRegistry.h>

namespace infernux
{

struct ShaderAsset;
struct ShaderMaterialPreparation;
struct ShaderStagePair;
struct ShaderSourceModule;
struct ShaderAuthorSource;
struct ShaderAuthorProgramRecord;
enum class ShaderCodeTarget : uint8_t;
class InxMaterial;

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
    /// Resolve the selected author pair and publish its immutable program and
    /// defaults before scene extraction. Device creation is a separate step.
    static void PrepareMaterial(InxMaterial &material);
    /// Worker-safe construction from an already frozen frontend snapshot.
    [[nodiscard]] static std::shared_ptr<ShaderAsset> CompileAuthorAsset(const ShaderSourceModule &source,
                                                                         const ShaderAuthorSource &prepared,
                                                                         const std::string &guid,
                                                                         std::vector<std::string> dependencies);
    [[nodiscard]] static std::shared_ptr<ShaderAsset>
    RestoreAuthorAsset(const ShaderSourceModule &source, const std::string &guid, ShaderAuthorProgramRecord record);
    static void CookAuthorProgram(const AssetDatabase &database, const std::string &authorGuid,
                                  const std::string &vertexGuid, ShaderCodeTarget target, const std::string &buildCache,
                                  const std::string &library);
    [[nodiscard]] static std::shared_ptr<const ShaderMaterialPreparation>
    CompileMaterialProgram(const std::shared_ptr<ShaderAsset> &fragment, const std::shared_ptr<ShaderAsset> &vertex,
                           const ShaderStagePair &stages, AssetDatabase *database);
    [[nodiscard]] static std::shared_ptr<const ShaderMaterialPreparation>
    PrepareMaterialProgram(const std::shared_ptr<ShaderAsset> &fragment, const std::shared_ptr<ShaderAsset> &vertex,
                           const ShaderStagePair &stages, AssetDatabase *database);
    RuntimeAssetPayload Load(const std::string &filePath, const std::string &guid, AssetDatabase *adb) override;
    [[nodiscard]] AssetReloadMode GetReloadMode() const noexcept override
    {
        return AssetReloadMode::Grouped;
    }
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
