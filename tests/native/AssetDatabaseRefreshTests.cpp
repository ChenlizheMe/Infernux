#include <core/threading/JobSystem.h>
#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/AssetDatabase/AssetIndex.h>
#include <function/resources/AssetDependencyGraph.h>
#include <function/resources/AssetImporter/ImporterRegistry.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxFileLoader/InxDefaultLoader.hpp>
#include <function/resources/InxFileLoader/InxPythonScriptLoader.hpp>
#include <function/resources/InxMesh/MeshLoader.h>
#include <function/resources/InxTexture/TextureLoader.h>
#include <platform/filesystem/InxPath.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
void Require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void WriteText(const std::filesystem::path &path, const std::string &text)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    Require(output.good(), "failed to write asset database refresh fixture");
}

class MixedCaseImporter final : public infernux::AssetImporter
{
  public:
    explicit MixedCaseImporter(std::string extension) : m_extension(std::move(extension))
    {
    }

    [[nodiscard]] infernux::ResourceType GetResourceType() const override
    {
        return infernux::ResourceType::Mesh;
    }

    [[nodiscard]] std::vector<std::string> GetSupportedExtensions() const override
    {
        return {m_extension};
    }

    [[nodiscard]] infernux::ImportArtifact Import(const infernux::ImportRequest &request) const override
    {
        return infernux::ImportArtifact(request.metadata);
    }

  private:
    std::string m_extension;
};

void TestImporterExtensionsAreCaseInsensitive()
{
    infernux::ImporterRegistry registry;
    registry.Register(std::make_unique<MixedCaseImporter>(".FbX"));
    auto *const importer = registry.GetImporterForExtension(".fbx");
    Require(importer != nullptr, "lowercase extension did not resolve a mixed-case importer registration");
    Require(registry.GetImporterForExtension(".FBX") == importer,
            "uppercase source extension did not resolve the registered importer");

    bool duplicateRejected = false;
    try {
        registry.Register(std::make_unique<MixedCaseImporter>(".FBX"));
    } catch (const std::logic_error &) {
        duplicateRejected = true;
    }
    Require(duplicateRejected, "case-only duplicate importer registration was accepted");
}

void TestNonGuidRenderEffectFieldsDoNotCreateDependenciesOnInitialRefresh()
{
    const auto root = std::filesystem::temp_directory_path() / "infernux-asset-refresh-relative-dependency";
    std::filesystem::remove_all(root);
    const auto rendering = root / "Assets" / "Rendering";
    const auto bloom = rendering / "Bloom.effect";
    const auto group = rendering / "Default Post Processing.effectgroup";

    WriteText(bloom, R"({
  "$schema": "infernux.render_effect",
  "dependencies": [],
  "feature_type": "infernux.post.bloom",
  "parameters": {}
})");
    WriteText(group, R"({
  "$schema": "infernux.render_effect_group",
  "legacy_version": 1,
  "entries": [
    {
      "asset": {
        "guid": "",
        "path_hint": "Assets/Rendering/Bloom.effect",
        "path": "Assets/Rendering/Bloom.effect"
      },
      "enabled": true,
      "entry_id": "bloom",
      "overrides": {},
      "scope": "final"
    }
  ]
})");

    infernux::JobSystem::Initialize(2);
    try {
        {
            auto database = std::make_unique<infernux::AssetDatabase>();
            database->Initialize(infernux::FromFsPath(root));
            auto &registry = infernux::AssetRegistry::Instance();
            registry.Initialize(std::move(database));
            registry.RegisterLoader(
                infernux::ResourceType::RenderEffect,
                std::make_unique<infernux::InxDefaultTextLoader>(infernux::ResourceType::RenderEffect));
            registry.PopulateAssetDatabaseLoaders();
            auto *assetDatabase = registry.GetAssetDatabase();
            assetDatabase->Refresh();

            const std::string bloomGuid = assetDatabase->GetGuidFromPath(infernux::FromFsPath(bloom));
            const std::string groupGuid = assetDatabase->GetGuidFromPath(infernux::FromFsPath(group));
            Require(!bloomGuid.empty(), "initial refresh did not register the referenced effect");
            Require(!groupGuid.empty(), "initial refresh did not register the effect group");

            infernux::AssetIndex index;
            const auto indexPath = root / "Library" / "AssetIndex.json";
            Require(
                index.Load(infernux::FromFsPath(indexPath), infernux::FilesystemPathKey(infernux::FromFsPath(root))),
                "initial refresh did not persist the asset index");
            const auto *entry = index.Find(infernux::FilesystemPathKey(infernux::FromFsPath(group)));
            Require(entry != nullptr, "effect group is absent from the initial asset index");
            Require(entry->importSucceeded, "non-GUID render effect fields should not invalidate the asset");
            Require(entry->dependencies.empty(), "path-only render effect field published a dependency");
            registry.Shutdown();
        }
        infernux::JobSystem::Shutdown();
    } catch (...) {
        if (infernux::AssetRegistry::Instance().IsInitialized())
            infernux::AssetRegistry::Instance().Shutdown();
        infernux::JobSystem::Shutdown();
        std::filesystem::remove_all(root);
        throw;
    }
    std::filesystem::remove_all(root);
}

void TestScriptReimportRefreshesContentHashAndPreservesGuid()
{
    const auto root = std::filesystem::temp_directory_path() / "infernux-script-reimport-content-hash";
    const auto clone = std::filesystem::temp_directory_path() / "infernux-script-reimport-clone";
    std::filesystem::remove_all(root);
    std::filesystem::remove_all(clone);
    const auto script = root / "Assets" / "Scripts" / "Gameplay.py";
    WriteText(script, "VALUE = 1\n");

    infernux::JobSystem::Initialize(2);
    try {
        auto database = std::make_unique<infernux::AssetDatabase>();
        database->Initialize(infernux::FromFsPath(root));
        auto &registry = infernux::AssetRegistry::Instance();
        registry.Initialize(std::move(database));
        registry.RegisterLoader(infernux::ResourceType::Script, std::make_unique<infernux::InxPythonScriptLoader>());
        registry.PopulateAssetDatabaseLoaders();
        auto *assetDatabase = registry.GetAssetDatabase();
        assetDatabase->Refresh();

        const std::string scriptPath = infernux::FromFsPath(script);
        const std::string originalGuid = assetDatabase->GetGuidFromPath(scriptPath);
        const auto originalMetadata = assetDatabase->GetMetaByGuid(originalGuid);
        Require(!originalGuid.empty() && originalMetadata != nullptr,
                "initial script refresh did not publish metadata");
        const std::string originalHash = originalMetadata->GetDataAs<std::string>("content_hash");

        // Preserve byte count so the regression cannot be hidden by a size-only check.
        WriteText(script, "VALUE = 2\n");
        const auto result = assetDatabase->ReimportAsset(scriptPath);
        Require(result.succeeded, "script reimport failed");
        Require(result.guid == originalGuid, "script reimport changed its GUID");
        const auto rebuiltMetadata = assetDatabase->GetMetaByGuid(originalGuid);
        Require(rebuiltMetadata != nullptr, "script reimport removed its metadata");
        Require(rebuiltMetadata->GetDataAs<std::string>("content_hash") != originalHash,
                "script reimport retained the stale source hash");
        Require(rebuiltMetadata->GetDataAs<size_t>("file_size") == std::string("VALUE = 2\n").size(),
                "script reimport retained the stale source size");

        assetDatabase->FlushDerivedIndex();
        infernux::AssetIndex index;
        Require(index.Load(infernux::FromFsPath(root / "Library" / "AssetIndex.json"),
                           infernux::FilesystemPathKey(infernux::FromFsPath(root))),
                "script reimport did not persist the derived index");
        const auto *entry = index.Find(infernux::FilesystemPathKey(scriptPath));
        Require(entry != nullptr && entry->guid == originalGuid, "script reimport index lost the original identity");
        Require(entry->contentHash == rebuiltMetadata->GetDataAs<std::string>("content_hash"),
                "script reimport index did not publish the rebuilt source hash");
        const auto readMetadata = [](const std::filesystem::path &path) {
            std::ifstream stream(path);
            return nlohmann::json::parse(stream);
        };
        const auto sharedMetadata = readMetadata(infernux::ToFsPath(infernux::FromFsPath(script) + ".meta"));
        Require(sharedMetadata.at("metadata").at("file_path").at("value") == "Assets/Scripts/Gameplay.py",
                "reimport persisted an absolute path");

        registry.Shutdown();
        std::filesystem::create_directories(clone);
        std::filesystem::copy(root / "Assets", clone / "Assets", std::filesystem::copy_options::recursive);
        auto clonedDatabase = std::make_unique<infernux::AssetDatabase>();
        clonedDatabase->Initialize(infernux::FromFsPath(clone));
        registry.Initialize(std::move(clonedDatabase));
        registry.RegisterLoader(infernux::ResourceType::Script, std::make_unique<infernux::InxPythonScriptLoader>());
        registry.PopulateAssetDatabaseLoaders();
        registry.GetAssetDatabase()->Refresh();
        const auto clonedScript = clone / "Assets" / "Scripts" / "Gameplay.py";
        Require(registry.GetAssetDatabase()->GetGuidFromPath(infernux::FromFsPath(clonedScript)) == originalGuid,
                "clone changed the shared asset GUID");
        Require(readMetadata(infernux::ToFsPath(infernux::FromFsPath(clonedScript) + ".meta")) == sharedMetadata,
                "opening a clone changed its shared sidecar");
        registry.Shutdown();
        infernux::JobSystem::Shutdown();
    } catch (...) {
        if (infernux::AssetRegistry::Instance().IsInitialized())
            infernux::AssetRegistry::Instance().Shutdown();
        infernux::JobSystem::Shutdown();
        std::filesystem::remove_all(root);
        std::filesystem::remove_all(clone);
        throw;
    }
    std::filesystem::remove_all(root);
    std::filesystem::remove_all(clone);
}

void TestValidButStaleSidecarIsRebuiltFromCurrentSource()
{
    const auto root = std::filesystem::temp_directory_path() / "infernux-asset-refresh-stale-sidecar";
    std::filesystem::remove_all(root);
    const auto script = root / "Assets" / "Scripts" / "Migrated.py";
    const auto sidecar = root / "Assets" / "Scripts" / "Migrated.py.meta";
    WriteText(script, "class Migrated:\n    VALUE = 2\n");
    WriteText(sidecar, R"({
  "metadata": {
    "content_hash": {"type": "string", "value": "0123456789abcdef"},
    "file_extension": {"type": "string", "value": ".py"},
    "file_path": {"type": "string", "value": "D:/OldProject/Assets/Scripts/Migrated.py"},
    "file_size": {"type": "size_t", "value": 12},
    "file_type": {"type": "string", "value": "script"},
    "guid": {"type": "string", "value": "fedcba0987654321fedcba0987654321"},
    "language": {"type": "string", "value": "python"},
    "resource_type": {"type": "enum infernux::ResourceType", "value": "Script"}
  }
})");

    infernux::JobSystem::Initialize(2);
    try {
        auto database = std::make_unique<infernux::AssetDatabase>();
        database->Initialize(infernux::FromFsPath(root));
        auto &registry = infernux::AssetRegistry::Instance();
        registry.Initialize(std::move(database));
        registry.RegisterLoader(infernux::ResourceType::Script, std::make_unique<infernux::InxPythonScriptLoader>());
        registry.PopulateAssetDatabaseLoaders();
        auto *assetDatabase = registry.GetAssetDatabase();
        assetDatabase->Refresh();

        const std::string scriptPath = infernux::FromFsPath(script);
        const std::string guid = assetDatabase->GetGuidFromPath(scriptPath);
        const auto metadata = assetDatabase->GetMetaByGuid(guid);
        Require(guid == "fedcba0987654321fedcba0987654321", "stale sidecar rebuild did not preserve its GUID");
        Require(metadata != nullptr, "stale sidecar rebuild did not publish metadata");
        Require(metadata->GetDataAs<std::string>("content_hash") != "0123456789abcdef",
                "stale sidecar rebuild retained the old content hash");
        Require(metadata->GetDataAs<size_t>("file_size") == std::filesystem::file_size(script),
                "stale sidecar rebuild retained the old source size");
        Require(metadata->GetDataAs<std::string>("file_path") ==
                    infernux::InxResourceMeta::NormalizeFilePath(scriptPath),
                "stale sidecar rebuild retained the old source path");

        registry.Shutdown();
        infernux::JobSystem::Shutdown();
    } catch (...) {
        if (infernux::AssetRegistry::Instance().IsInitialized())
            infernux::AssetRegistry::Instance().Shutdown();
        infernux::JobSystem::Shutdown();
        std::filesystem::remove_all(root);
        throw;
    }
    std::filesystem::remove_all(root);
}

void TestProjectPackagesScanRootSharesTheGuidCatalog()
{
    const auto root = std::filesystem::temp_directory_path() / "infernux-asset-refresh-packages-root";
    std::filesystem::remove_all(root);
    const auto assetScript = root / "Assets" / "Scripts" / "Gameplay.py";
    const auto packageScript = root / "Packages" / "vendor" / "tool" / "Runtime" / "Lifecycle.py";
    WriteText(assetScript, "class Gameplay:\n    pass\n");
    WriteText(packageScript, "class Lifecycle:\n    pass\n");

    infernux::JobSystem::Initialize(2);
    try {
        auto database = std::make_unique<infernux::AssetDatabase>();
        database->Initialize(infernux::FromFsPath(root));
        auto &registry = infernux::AssetRegistry::Instance();
        registry.Initialize(std::move(database));
        registry.RegisterLoader(infernux::ResourceType::Script, std::make_unique<infernux::InxPythonScriptLoader>());
        registry.PopulateAssetDatabaseLoaders();
        auto *assetDatabase = registry.GetAssetDatabase();
        assetDatabase->AddScanRoot(infernux::FromFsPath(root / "Packages"));
        assetDatabase->Refresh();

        const std::string assetGuid = assetDatabase->GetGuidFromPath(infernux::FromFsPath(assetScript));
        const std::string packageGuid = assetDatabase->GetGuidFromPath(infernux::FromFsPath(packageScript));
        Require(!assetGuid.empty(), "Assets script was not registered in the shared catalog");
        Require(!packageGuid.empty(), "Packages script was not registered in the shared catalog");
        Require(assetGuid != packageGuid, "Assets and Packages scripts received the same GUID");
        Require(assetDatabase->GetPathFromGuid(packageGuid) == infernux::FromFsPath(packageScript),
                "Packages GUID did not resolve to its current path");
        Require(std::filesystem::is_regular_file(infernux::ToFsPath(infernux::FromFsPath(packageScript) + ".meta")),
                "Packages scan root did not persist stable GUID metadata");

        registry.Shutdown();
        infernux::JobSystem::Shutdown();
    } catch (...) {
        if (infernux::AssetRegistry::Instance().IsInitialized())
            infernux::AssetRegistry::Instance().Shutdown();
        infernux::JobSystem::Shutdown();
        std::filesystem::remove_all(root);
        throw;
    }
    std::filesystem::remove_all(root);
}

void TestRegisteredScriptKeepsGuidWhenSidecarAndDerivedIndexAreMissing(bool keepIndex, bool changeSource,
                                                                     bool packageScript)
{
    const auto root = std::filesystem::temp_directory_path() /
                      (std::string("infernux-live-script-sidecar-") + (keepIndex ? "indexed-" : "unindexed-") +
                       (changeSource ? "changed-" : "unchanged-") + (packageScript ? "package" : "asset"));
    std::filesystem::remove_all(root);
    const auto script = packageScript ? root / "Packages" / "vendor" / "tool" / "editor" / "__init__.py"
                                      : root / "Assets" / "Scripts" / "Identity.py";
    WriteText(script, "VALUE = 1\n");
    infernux::JobSystem::Initialize(2);
    try {
        auto database = std::make_unique<infernux::AssetDatabase>();
        database->Initialize(infernux::FromFsPath(root));
        auto &registry = infernux::AssetRegistry::Instance();
        registry.Initialize(std::move(database));
        registry.RegisterLoader(infernux::ResourceType::Script, std::make_unique<infernux::InxPythonScriptLoader>());
        registry.PopulateAssetDatabaseLoaders();
        auto *assetDatabase = registry.GetAssetDatabase();
        if (packageScript)
            assetDatabase->AddScanRoot(infernux::FromFsPath(root / "Packages"));
        assetDatabase->Refresh();
        const std::string scriptPath = infernux::FromFsPath(script);
        const std::string originalGuid = assetDatabase->GetGuidFromPath(scriptPath);
        const auto originalMetadata = assetDatabase->GetMetaByGuid(originalGuid);
        Require(!originalGuid.empty() && originalMetadata != nullptr,
                "initial refresh did not register the identity fixture");
        const std::string originalHash =
            originalMetadata->GetDataAs<std::string>("content_hash");
        assetDatabase->FlushDerivedIndex();
        const auto sidecar = infernux::ToFsPath(scriptPath + ".meta");
        Require(std::filesystem::remove(sidecar), "identity fixture sidecar was not removed");
        if (!keepIndex)
            Require(std::filesystem::remove(root / "Library" / "AssetIndex.json"),
                    "identity fixture derived index was not removed");
        if (changeSource)
            WriteText(script, "VALUE = 2\n");

        assetDatabase->Refresh();
        Require(assetDatabase->GetGuidFromPath(scriptPath) == originalGuid,
                "refresh replaced a registered script GUID after its sidecar/index was removed");
        Require(assetDatabase->GetPathFromGuid(originalGuid) == scriptPath,
                "refresh retired the original GUID-to-path mapping");
        const auto rebuilt = assetDatabase->GetMetaByGuid(originalGuid);
        Require(rebuilt != nullptr && rebuilt->GetGuid() == originalGuid,
                "refresh did not publish the original script metadata identity");
        Require((rebuilt->GetDataAs<std::string>("content_hash") != originalHash) == changeSource,
                "identity preservation reused the wrong source revision");
        std::ifstream input(sidecar);
        Require(nlohmann::json::parse(input).at("metadata").at("guid").at("value") == originalGuid,
                "regenerated sidecar did not preserve the registered GUID");
        registry.Shutdown();
        infernux::JobSystem::Shutdown();
    } catch (...) {
        if (infernux::AssetRegistry::Instance().IsInitialized())
            infernux::AssetRegistry::Instance().Shutdown();
        infernux::JobSystem::Shutdown();
        std::filesystem::remove_all(root);
        throw;
    }
    std::filesystem::remove_all(root);
}

void TestStartupCatalogSurvivesLiveIndexInvalidation()
{
    const auto root = std::filesystem::temp_directory_path() / "infernux-asset-startup-catalog";
    std::filesystem::remove_all(root);
    const auto script = root / "Assets" / "Scripts" / "Startup.py";
    WriteText(script, "class Startup:\n    pass\n");

    infernux::JobSystem::Initialize(2);
    try {
        std::string scriptGuid;
        {
            auto database = std::make_unique<infernux::AssetDatabase>();
            database->Initialize(infernux::FromFsPath(root));
            auto &registry = infernux::AssetRegistry::Instance();
            registry.Initialize(std::move(database));
            registry.RegisterLoader(infernux::ResourceType::Script,
                                    std::make_unique<infernux::InxPythonScriptLoader>());
            registry.PopulateAssetDatabaseLoaders();
            auto *assetDatabase = registry.GetAssetDatabase();
            assetDatabase->Refresh();
            scriptGuid = assetDatabase->GetGuidFromPath(infernux::FromFsPath(script));
            Require(!scriptGuid.empty(), "initial refresh did not register the startup fixture");
            registry.Shutdown();
        }

        const auto liveIndex = root / "Library" / "AssetIndex.json";
        const auto startupIndex = root / "Library" / "AssetIndex.startup-cache.json";
        Require(std::filesystem::is_regular_file(liveIndex), "initial refresh did not publish the live index");
        Require(std::filesystem::is_regular_file(startupIndex), "initial refresh did not publish the startup index");
        std::filesystem::remove(liveIndex);

        {
            infernux::AssetDatabase restored;
            restored.Initialize(infernux::FromFsPath(root));
            Require(restored.RestoreCachedCatalog(), "startup catalog did not restore after live index invalidation");
            Require(restored.GetGuidFromPath(infernux::FromFsPath(script)) == scriptGuid,
                    "startup catalog did not preserve the committed asset identity");
        }
        infernux::JobSystem::Shutdown();
    } catch (...) {
        if (infernux::AssetRegistry::Instance().IsInitialized())
            infernux::AssetRegistry::Instance().Shutdown();
        infernux::JobSystem::Shutdown();
        std::filesystem::remove_all(root);
        throw;
    }
    std::filesystem::remove_all(root);
}

void TestRuntimeAssetCatalogInstallsStableIdentityWithoutSidecar()
{
    const auto root = std::filesystem::temp_directory_path() / "infernux-runtime-asset-catalog";
    std::filesystem::remove_all(root);
    const auto material = root / "Assets" / "Materials" / "Runtime.mat";
    const auto catalog = root / "Library" / "RuntimeAssetRecords.json";
    WriteText(material, "{}\n");
    WriteText(catalog, R"({
  "$schema": "infernux.runtime_asset_records",
  "entries": [
    {
      "guid": "runtime-material-guid",
      "runtime_path": "Assets/Materials/Runtime.mat",
      "metadata": {
        "metadata": {
          "content_hash": {"type": "string", "value": "runtime-hash"},
          "file_path": {"type": "string", "value": "Assets/Materials/Runtime.mat"},
          "guid": {"type": "string", "value": "runtime-material-guid"},
          "resource_type": {"type": "enum infernux::ResourceType", "value": "Material"}
        }
      }
    }
  ]
})");

    infernux::AssetDatabase database;
    database.Initialize(infernux::FromFsPath(root));
    database.InstallRuntimeAssetCatalog(infernux::FromFsPath(catalog));

    const std::string materialPath = infernux::FromFsPath(material);
    Require(database.GetGuidFromPath(materialPath) == "runtime-material-guid",
            "runtime asset catalog did not install the stable path mapping");
    Require(database.GetPathFromGuid("runtime-material-guid") == materialPath,
            "runtime asset catalog did not install the stable GUID mapping");
    const auto metadata = database.GetMetaByGuid("runtime-material-guid");
    Require(metadata != nullptr, "runtime asset catalog did not install metadata");
    Require(metadata->GetResourceType() == infernux::ResourceType::Material,
            "runtime asset catalog changed the resource type");
    Require(metadata->HasKey("read_only") && metadata->GetDataAs<bool>("read_only"),
            "runtime asset catalog identity is not read-only");
    std::filesystem::remove_all(root);
}

void TestCompositeModelPublishesExternalTextureGuidDependencies()
{
    const auto root = std::filesystem::temp_directory_path() / "infernux-asset-model-dependencies";
    std::filesystem::remove_all(root);
    const auto sourceRoot =
        std::filesystem::path(INFERNUX_SOURCE_DIR) / "external" / "assimp" / "test" / "models" / "OBJ";
    const auto model = root / "Assets" / "Models" / "spider.obj";
    const auto material = root / "Assets" / "Models" / "spider.mtl";
    std::filesystem::create_directories(model.parent_path());
    std::filesystem::copy_file(sourceRoot / "spider.obj", model, std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file(sourceRoot / "spider.mtl", material, std::filesystem::copy_options::overwrite_existing);
    for (const char *name :
         {"wal67ar_small.jpg", "wal69ar_small.jpg", "SpiderTex.jpg", "drkwood2.jpg", "engineflare1.jpg"}) {
        std::filesystem::copy_file(sourceRoot / name, model.parent_path() / name,
                                   std::filesystem::copy_options::overwrite_existing);
    }

    infernux::AssetDependencyGraph::Instance().Clear();
    infernux::JobSystem::Initialize(2);
    try {
        auto database = std::make_unique<infernux::AssetDatabase>();
        database->Initialize(infernux::FromFsPath(root));
        auto &registry = infernux::AssetRegistry::Instance();
        registry.Initialize(std::move(database));
        registry.RegisterLoader(infernux::ResourceType::DefaultText,
                                std::make_unique<infernux::InxDefaultTextLoader>(infernux::ResourceType::DefaultText));
        registry.RegisterLoader(infernux::ResourceType::Mesh, std::make_unique<infernux::MeshLoader>());
        registry.RegisterLoader(infernux::ResourceType::Texture, std::make_unique<infernux::TextureLoader>());
        registry.PopulateAssetDatabaseLoaders();
        auto *assetDatabase = registry.GetAssetDatabase();
        assetDatabase->Refresh();

        const auto modelGuid = assetDatabase->GetGuidFromPath(infernux::FromFsPath(model));
        Require(!modelGuid.empty(), "composite model fixture was not registered");
        const auto dependencies = infernux::AssetDependencyGraph::Instance().GetDependencies(modelGuid);
        Require(dependencies.size() == 5, "composite model did not publish five texture GUID dependencies");
        for (const char *name :
             {"wal67ar_small.jpg", "wal69ar_small.jpg", "SpiderTex.jpg", "drkwood2.jpg", "engineflare1.jpg"}) {
            const auto textureGuid = assetDatabase->GetGuidFromPath(infernux::FromFsPath(model.parent_path() / name));
            Require(!textureGuid.empty() && dependencies.count(textureGuid) == 1,
                    "composite model texture dependency was not published by GUID");
        }
        registry.Shutdown();
        infernux::JobSystem::Shutdown();
    } catch (...) {
        if (infernux::AssetRegistry::Instance().IsInitialized())
            infernux::AssetRegistry::Instance().Shutdown();
        infernux::JobSystem::Shutdown();
        infernux::AssetDependencyGraph::Instance().Clear();
        std::filesystem::remove_all(root);
        throw;
    }
    infernux::AssetDependencyGraph::Instance().Clear();
    std::filesystem::remove_all(root);
}

void TestRuntimeAssetCatalogResolvesBuiltInArchiveResources()
{
    const auto root = std::filesystem::temp_directory_path() / "infernux-runtime-builtin-catalog";
    std::filesystem::remove_all(root);
    const auto resources = root / "runtime" / "infernux" / "resources";
    const auto shader = resources / "shaders" / "standard.vert";
    const auto catalog = root / "content" / "Library" / "RuntimeAssetRecords.json";
    WriteText(shader, "#version 450\nvoid main() {}\n");
    WriteText(catalog, R"({
  "$schema": "infernux.runtime_asset_records",
  "entries": [
    {
      "guid": "runtime-standard-vertex-guid",
      "runtime_path": "Library/Resources/shaders/standard.vert",
      "runtime_artifacts": [
        {
          "package": "Runtime.inxrt",
          "runtime_path": "infernux/resources/shaders/standard.vert"
        }
      ],
      "metadata": {
        "metadata": {
          "content_hash": {"type": "string", "value": "runtime-shader-hash"},
          "file_path": {"type": "string", "value": "Library/Resources/shaders/standard.vert"},
          "guid": {"type": "string", "value": "runtime-standard-vertex-guid"},
          "resource_type": {"type": "enum infernux::ResourceType", "value": "Shader"},
          "shader_id": {"type": "string", "value": "Standard"},
          "type": {"type": "string", "value": "vertex"}
        }
      }
    }
  ]
})");

    infernux::AssetDatabase database;
    database.Initialize(infernux::FromFsPath(root / "content"));
    database.AddReadOnlyScanRoot(infernux::FromFsPath(resources));
    database.InstallRuntimeAssetCatalog(infernux::FromFsPath(catalog));

    const std::string shaderPath = infernux::FromFsPath(shader);
    Require(database.GetPathFromGuid("runtime-standard-vertex-guid") == shaderPath,
            "runtime built-in identity did not resolve to the extracted resource path");
    Require(database.GetGuidFromPath(shaderPath) == "runtime-standard-vertex-guid",
            "runtime built-in extracted path did not preserve its cooked GUID");
    Require(database.FindShaderPathById("Standard", "vertex") == shaderPath,
            "runtime shader lookup returned the non-existent authoring cache path");
    std::filesystem::remove_all(root);
}

void TestModelSettingsPublishOnlyAfterSuccessfulImport()
{
    const auto root = std::filesystem::temp_directory_path() /
                      ("infernux-model-settings-publication-" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto model = root / "Assets" / "Model.obj";
    const std::string geometry = "o Triangle\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
    WriteText(model, geometry);
    const auto readBytes = [](const std::filesystem::path &path) {
        std::ifstream input(path, std::ios::binary);
        Require(input.good(), "missing committed import file");
        return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    };
    infernux::JobSystem::Initialize(2);
    try {
        auto database = std::make_unique<infernux::AssetDatabase>();
        database->Initialize(infernux::FromFsPath(root));
        auto &registry = infernux::AssetRegistry::Instance();
        registry.Initialize(std::move(database));
        registry.RegisterLoader(infernux::ResourceType::Mesh, std::make_unique<infernux::MeshLoader>());
        registry.PopulateAssetDatabaseLoaders();
        auto *db = registry.GetAssetDatabase();
        db->Refresh();
        const std::string path = infernux::FromFsPath(model);
        const auto guid = db->GetGuidFromPath(path);
        const auto metaPath = infernux::InxResourceMeta::GetMetaFilePath(path);
        const auto artifactPath = db->GetRuntimeArtifactPath(guid, infernux::ResourceType::Mesh);
        const auto originalMeta = readBytes(infernux::ToFsPath(metaPath));
        const auto originalArtifact = readBytes(infernux::ToFsPath(artifactPath));
        const auto originalSnapshot = db->GetMetaByGuid(guid);
        const auto generation = db->GetQueryGeneration();
        WriteText(model, "not a model\n");
        const auto failure = db->ReimportAsset(path, {{"scale_factor", 3.0}, {"weld_vertices", false}});
        Require(!failure && !failure.databaseCommitted, "invalid source was published");
        Require(readBytes(infernux::ToFsPath(metaPath)) == originalMeta, "failed Apply changed disk settings");
        Require(readBytes(infernux::ToFsPath(artifactPath)) == originalArtifact, "failed Apply changed artifact");
        Require(db->GetMetaByGuid(guid) == originalSnapshot, "failed Apply replaced live metadata");
        Require(db->GetQueryGeneration() == generation, "failed Apply advanced published catalog");
        WriteText(model, geometry);
        const auto success = db->ReimportAsset(path, {{"scale_factor", 3.0}, {"weld_vertices", false}});
        Require(success && success.databaseCommitted, "valid model settings were not published");
        const auto published = db->GetMetaByGuid(guid);
        Require(published->GetDataAs<float>("scale_factor") == 3.0f, "scale draft was lost");
        Require(!published->GetDataAs<bool>("weld_vertices"), "welding draft was lost");
        Require(readBytes(infernux::ToFsPath(artifactPath)) != originalArtifact, "settings did not change geometry");
        infernux::InxResourceMeta disk;
        Require(disk.LoadFromFile(metaPath), "could not read committed settings");
        Require(disk.SerializeDocument() == published->SerializeDocumentPortable(infernux::FromFsPath(root)),
                "disk and live authored settings differ");
        Require(db->GetGuidFromPath(path) == guid, "Apply replaced source identity");
        Require(db->ReimportAsset(path).succeeded, "source reimport failed after Apply");
        Require(db->GetMetaByGuid(guid)->GetDataAs<float>("scale_factor") == 3.0f,
                "source reimport discarded applied settings");
        const auto committedMeta = readBytes(infernux::ToFsPath(metaPath));
        const auto committedArtifact = readBytes(infernux::ToFsPath(artifactPath));
        bool invalidRejected = false;
        try {
            (void)db->ReimportAsset(path, {{"guid", "replacement-guid"}});
        } catch (const std::invalid_argument &) {
            invalidRejected = true;
        }
        Require(invalidRejected, "settings were allowed to replace asset identity");
        Require(readBytes(infernux::ToFsPath(metaPath)) == committedMeta, "invalid settings changed metadata");
        Require(readBytes(infernux::ToFsPath(artifactPath)) == committedArtifact, "invalid settings changed artifact");
        registry.Shutdown();
        infernux::JobSystem::Shutdown();
        std::filesystem::remove_all(root);
    } catch (...) {
        if (infernux::AssetRegistry::Instance().IsInitialized())
            infernux::AssetRegistry::Instance().Shutdown();
        infernux::JobSystem::Shutdown();
        throw;
    }
}

void TestCopiedSkeletonRefreshWithLimitedWorkers(bool inlineJobs)
{
    const auto root =
        std::filesystem::temp_directory_path() /
        ("infernux-copied-skeleton-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto authored = root / "author";
    const auto clone = root / "clone";
    const auto fixture =
        std::filesystem::path(INFERNUX_SOURCE_DIR) / "external/assimp/test/models/FBX/animation_with_skeleton.fbx";
    std::filesystem::create_directories(authored / "Assets");
    std::filesystem::copy_file(fixture, authored / "Assets/Z_Definition.fbx");
    std::filesystem::copy_file(fixture, authored / "Assets/A_Copy.fbx");
    if (inlineJobs)
        infernux::JobSystem::InitializeInline();
    else
        infernux::JobSystem::Initialize(1);
    auto &registry = infernux::AssetRegistry::Instance();
    const auto initialize = [&](const std::filesystem::path &project) {
        auto database = std::make_unique<infernux::AssetDatabase>();
        database->Initialize(infernux::FromFsPath(project));
        registry.Initialize(std::move(database));
        registry.RegisterLoader(infernux::ResourceType::Mesh, std::make_unique<infernux::MeshLoader>());
        registry.RegisterLoader(infernux::ResourceType::Texture, std::make_unique<infernux::TextureLoader>());
        registry.PopulateAssetDatabaseLoaders();
        return registry.GetAssetDatabase();
    };
    try {
        auto *database = initialize(authored);
        const auto definitionPath = authored / "Assets/Z_Definition.fbx";
        const auto copyPath = authored / "Assets/A_Copy.fbx";
        const auto definition = database->ImportAsset(infernux::FromFsPath(definitionPath));
        const auto copy = database->ImportAsset(infernux::FromFsPath(copyPath));
        Require(definition.succeeded && copy.succeeded, "could not author copied skeleton fixtures");
        Require(database
                    ->ReimportAsset(infernux::FromFsPath(copyPath), {{"skeleton_definition_mode", "copy"},
                                                                     {"skeleton_definition_guid", definition.guid},
                                                                     {"skeleton_definition_id", "skeleton"},
                                                                     {"rig_root_node", ""}})
                    .succeeded,
                "could not author copied skeleton settings");
        const auto verify = [&](const std::filesystem::path &project) {
            std::ifstream input(infernux::ToFsPath(database->GetAssetIndexPath()));
            const auto index = nlohmann::json::parse(input);
            for (const auto &identity : {definition.guid, copy.guid}) {
                bool found = false;
                for (const auto &entry : index.at("entries")) {
                    if (entry.at("guid") == identity) {
                        Require(entry.at("import_succeeded").get<bool>(),
                                "copied skeleton batch import failed with limited workers");
                        found = true;
                    }
                }
                Require(found, "copied skeleton asset is missing from imported catalog");
                const auto metadata = database->GetMetaByGuid(identity);
                Require(metadata && metadata->GetStringData("published_skeleton_definition_guid") == definition.guid,
                        "copied skeleton batch changed published skeleton identity");
            }
            Require(database->GetGuidFromPath(infernux::FromFsPath(project / "Assets/A_Copy.fbx")) == copy.guid,
                    "copied skeleton batch changed source GUID");
        };
        // Force a definition and its lexically earlier dependent into the same batch.
        for (const auto &path : {definitionPath, copyPath})
            std::filesystem::last_write_time(path, std::filesystem::last_write_time(path) + std::chrono::seconds(1));
        database->Refresh();
        verify(authored);
        registry.Shutdown();
        std::filesystem::create_directories(clone);
        std::filesystem::copy(authored / "Assets", clone / "Assets", std::filesystem::copy_options::recursive);
        Require(!std::filesystem::exists(clone / "Library"), "clone unexpectedly has local import artifacts");
        database = initialize(clone);
        database->Refresh();
        verify(clone);
        std::cout << "Copied skeleton warm/clean refresh passed: " << (inlineJobs ? "inline" : "one worker")
                  << std::endl;
        registry.Shutdown();
        infernux::JobSystem::Shutdown();
    } catch (...) {
        if (registry.IsInitialized())
            registry.Shutdown();
        infernux::JobSystem::Shutdown();
        std::filesystem::remove_all(root);
        throw;
    }
    std::filesystem::remove_all(root);
}

void TestRuntimeAssetCatalogResolvesPrimaryContentArtifact()
{
    const auto root = std::filesystem::temp_directory_path() / "infernux-runtime-content-catalog";
    std::filesystem::remove_all(root);
    const auto cookedMaterial = root / "Library" / "Artifacts" / "Document" / "runtime-material-guid.mat";
    const auto catalog = root / "Library" / "RuntimeAssetRecords.json";
    WriteText(cookedMaterial, "{}\n");
    WriteText(catalog, R"({
  "$schema": "infernux.runtime_asset_records",
  "entries": [
    {
      "guid": "runtime-material-guid",
      "runtime_path": "Assets/Materials/Runtime.mat",
      "primary_runtime_artifact_id": "content-material-artifact",
      "runtime_artifacts": [
        {
          "runtime_artifact_id": "content-material-artifact",
          "package": "Content.inxpkg",
          "runtime_path": "Library/Artifacts/Document/runtime-material-guid.mat"
        }
      ],
      "metadata": {
        "metadata": {
          "content_hash": {"type": "string", "value": "runtime-hash"},
          "file_path": {"type": "string", "value": "Assets/Materials/Runtime.mat"},
          "guid": {"type": "string", "value": "runtime-material-guid"},
          "resource_type": {"type": "enum infernux::ResourceType", "value": "Material"}
        }
      }
    }
  ]
})");

    infernux::AssetDatabase database;
    database.Initialize(infernux::FromFsPath(root));
    database.InstallRuntimeAssetCatalog(infernux::FromFsPath(catalog));

    const std::string cookedPath = infernux::FromFsPath(cookedMaterial);
    Require(database.GetPathFromGuid("runtime-material-guid") == cookedPath,
            "runtime identity did not resolve to its primary Content artifact");
    Require(database.GetGuidFromPath(cookedPath) == "runtime-material-guid",
            "primary Content artifact did not preserve its cooked GUID");
    Require(database.GetGuidFromPath(infernux::FromFsPath(root / "Assets" / "Materials" / "Runtime.mat")).empty(),
            "runtime identity retained a missing authoring path");
    std::filesystem::remove_all(root);
}

void TestCookedModelTexturesKeepArtifactPaths()
{
    const auto root = std::filesystem::temp_directory_path() / "infernux-cooked-model-textures";
    infernux::InxResourceMeta model, texture;
    model.Init("model", 5, "Assets/Composite.glb", infernux::ResourceType::Mesh);
    texture.Init("model", 5, "Assets/Composite.glb::subtex:image", infernux::ResourceType::Texture);
    texture.AddMetadata("import_owner_guid", model.GetGuid());
    model.AddMetadata("model_textures", nlohmann::json::array({{{"key", "image/color"},
                                                                {"guid", texture.GetGuid()},
                                                                {"name", "Image"},
                                                                {"metadata", texture.SerializeDocument()}}})
                                            .dump());
    const auto modelPath = "Library/Artifacts/Mesh/" + model.GetGuid() + ".inxmesh";
    const auto texturePath = "Library/Artifacts/Texture/" + texture.GetGuid() + ".inxtex";
    WriteText(root / modelPath, "cooked model");
    WriteText(root / texturePath, "cooked texture");
    const auto catalog = root / "RuntimeAssetRecords.json";
    WriteText(catalog, nlohmann::json{{"$schema", "infernux.runtime_asset_records"},
                                      {"entries", nlohmann::json::array({{{"guid", model.GetGuid()},
                                                                          {"runtime_path", modelPath},
                                                                          {"metadata", model.SerializeDocument()}},
                                                                         {{"guid", texture.GetGuid()},
                                                                          {"runtime_path", texturePath},
                                                                          {"metadata", texture.SerializeDocument()}}})}}
                           .dump());
    infernux::AssetDatabase database;
    database.InitializeRuntime(infernux::FromFsPath(root));
    database.InstallRuntimeAssetCatalog(infernux::FromFsPath(catalog));
    Require(database.GetPathFromGuid(texture.GetGuid()) == infernux::FromFsPath(root / texturePath),
            "Player expanded editor-only model children over cooked Texture paths");
    Require(database.GetMetaByGuid(texture.GetGuid())->GetResourceType() == infernux::ResourceType::Texture,
            "Player lost imported Texture type");
    std::filesystem::remove_all(root);
}

void TestMoveRequiresRegisteredGuidIdentity()
{
    const auto root = std::filesystem::current_path() / "infernux-asset-move-guid-contract";
    std::filesystem::remove_all(root);
    const auto source = root / "Assets" / "Source.txt";
    const auto destination = root / "Assets" / "Destination.txt";
    WriteText(source, "identity\n");

    infernux::InxResourceMeta metadata;
    metadata.Init("identity\n", 9, infernux::FromFsPath(source), infernux::ResourceType::DefaultText);
    const auto sourceMetadata = infernux::InxResourceMeta::GetMetaFilePath(infernux::FromFsPath(source));
    Require(metadata.SaveToFile("Assets/Source.txt.meta", infernux::FromFsPath(root)),
            "failed to write relocation metadata fixture");
    infernux::InxResourceMeta storedMetadata;
    Require(storedMetadata.LoadFromFile(sourceMetadata), "failed to read portable relocation metadata fixture");
    Require(storedMetadata.GetStringData("file_path") == "Assets/Source.txt",
            "metadata save persisted a machine-local absolute path");
    Require(!storedMetadata.HasKey("content_hash") && !storedMetadata.HasKey("file_size"),
            "metadata save persisted local import statistics");
    Require(storedMetadata.GetGuid() == metadata.GetGuid(), "portable metadata save changed the asset GUID");
    std::filesystem::rename(source, destination);

    infernux::AssetDatabase database;
    database.Initialize(infernux::FromFsPath(root));
    const infernux::AssetMutationResult result =
        database.MoveAsset(infernux::FromFsPath(source), infernux::FromFsPath(destination));

    Require(!result, "unregistered asset relocation recovered identity from a sidecar");
    Require(result.errorCode == infernux::AssetMutationErrorCode::NotFound,
            "unregistered asset relocation returned the wrong error");
    Require(std::filesystem::is_regular_file(sourceMetadata), "failed relocation moved the source metadata sidecar");
    Require(!std::filesystem::exists(infernux::InxResourceMeta::GetMetaFilePath(infernux::FromFsPath(destination))),
            "failed relocation created destination metadata");
    Require(database.GetGuidFromPath(infernux::FromFsPath(destination)).empty(),
            "failed relocation imported a replacement GUID");
    std::filesystem::remove_all(root);
}

void TestBlenderNumberedBackupsAreNotAssets()
{
    const auto root = std::filesystem::temp_directory_path() / "infernux-blender-backup-scan";
    std::filesystem::remove_all(root);
    const auto source = root / "Assets" / "Notes.txt";
    const auto backup = root / "Assets" / "Assembly.blend1";
    WriteText(source, "asset\n");
    WriteText(backup, "Blender backup generation\n");

    infernux::JobSystem::Initialize(2);
    try {
        auto database = std::make_unique<infernux::AssetDatabase>();
        database->Initialize(infernux::FromFsPath(root));
        auto &registry = infernux::AssetRegistry::Instance();
        registry.Initialize(std::move(database));
        registry.RegisterLoader(infernux::ResourceType::DefaultText,
                                std::make_unique<infernux::InxDefaultTextLoader>(infernux::ResourceType::DefaultText));
        registry.PopulateAssetDatabaseLoaders();
        auto *assetDatabase = registry.GetAssetDatabase();
        assetDatabase->Refresh();

        Require(!assetDatabase->GetGuidFromPath(infernux::FromFsPath(source)).empty(),
                "ordinary source asset was not scanned");
        Require(assetDatabase->GetGuidFromPath(infernux::FromFsPath(backup)).empty(),
                "Blender numbered backup entered the asset catalog");
        Require(!std::filesystem::exists(infernux::ToFsPath(infernux::FromFsPath(backup) + ".meta")),
                "Blender numbered backup received a metadata sidecar");

        registry.Shutdown();
        infernux::JobSystem::Shutdown();
    } catch (...) {
        if (infernux::AssetRegistry::Instance().IsInitialized())
            infernux::AssetRegistry::Instance().Shutdown();
        infernux::JobSystem::Shutdown();
        std::filesystem::remove_all(root);
        throw;
    }
    std::filesystem::remove_all(root);
}

void TestRefreshPublishesDeterministicAssetEvents()
{
    using namespace infernux;
    const auto root =
        std::filesystem::temp_directory_path() /
        ("infernux-refresh-events-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto first = root / "Assets" / "First.txt";
    const auto second = root / "Assets" / "Second.txt";
    WriteText(first, "first");
    WriteText(second, "second");
    JobSystem::Initialize(2);
    auto &registry = AssetRegistry::Instance();
    auto &graph = AssetDependencyGraph::Instance();
    graph.Clear();
    try {
        auto database = std::make_unique<AssetDatabase>();
        database->Initialize(FromFsPath(root));
        registry.Initialize(std::move(database));
        registry.RegisterLoader(ResourceType::DefaultText,
                                std::make_unique<InxDefaultTextLoader>(ResourceType::DefaultText));
        registry.PopulateAssetDatabaseLoaders();
        auto *db = registry.GetAssetDatabase();
        db->Refresh();
        const auto firstGuid = db->GetGuidFromPath(FromFsPath(first));
        const auto secondGuid = db->GetGuidFromPath(FromFsPath(second));
        graph.AddRuntimeDependency("refresh-consumer", firstGuid);
        graph.AddRuntimeDependency("refresh-consumer", secondGuid);
        std::vector<std::pair<std::string, AssetEvent>> events;
        graph.RegisterCallback(ResourceType::DefaultText,
                               [&](const std::string &dependent, const std::string &guid, AssetEvent event) {
                                   Require(dependent == "refresh-consumer", "unexpected refresh consumer");
                                   Require(db->GetPathFromGuid(guid).empty() == (event == AssetEvent::Deleted),
                                           "consumer observed the old catalog during refresh publication");
                                   events.emplace_back(guid, event);
                               });
        WriteText(first, "first modified");
        WriteText(second, "second modified");
        db->Refresh();
        Require(events.size() == 2 && events[0].first < events[1].first && events[0].second == AssetEvent::Modified &&
                    events[1].second == AssetEvent::Modified,
                "refresh did not publish deterministic content events");
        events.clear();
        db->Refresh();
        Require(events.empty(), "unchanged refresh emitted a content event");
        const auto moved = root / "Assets" / "Moved.txt";
        const auto movedMeta = ToFsPath(FromFsPath(moved) + ".meta");
        std::filesystem::rename(first, moved);
        std::filesystem::rename(ToFsPath(FromFsPath(first) + ".meta"), movedMeta);
        db->Refresh();
        Require(events == std::vector<std::pair<std::string, AssetEvent>>{{firstGuid, AssetEvent::Moved}},
                "external move did not publish one Moved event");
        std::ifstream metadataInput(movedMeta, std::ios::binary);
        const std::string sidecar((std::istreambuf_iterator<char>(metadataInput)), std::istreambuf_iterator<char>());
        metadataInput.close();
        events.clear();
        std::filesystem::remove(moved);
        std::filesystem::remove(movedMeta);
        db->Refresh();
        Require(events == std::vector<std::pair<std::string, AssetEvent>>{{firstGuid, AssetEvent::Deleted}},
                "external deletion did not publish one Deleted event");
        Require(graph.HasDependency("refresh-consumer", firstGuid), "deletion erased authored runtime identity");
        events.clear();
        WriteText(moved, "restored original identity");
        WriteText(movedMeta, sidecar);
        db->Refresh();
        Require(events == std::vector<std::pair<std::string, AssetEvent>>{{firstGuid, AssetEvent::Modified}},
                "restored GUID did not notify its retained consumer");
        graph.Clear();
        registry.Shutdown();
        JobSystem::Shutdown();
        std::filesystem::remove_all(root);
    } catch (...) {
        graph.Clear();
        if (registry.IsInitialized())
            registry.Shutdown();
        JobSystem::Shutdown();
        throw;
    }
}

void TestVersionControlFilesAreNotAssets()
{
    const auto root = std::filesystem::temp_directory_path() / "infernux-vcs-control-scan";
    std::filesystem::remove_all(root);
    const auto source = root / "Assets" / "Vendor" / "Notes.txt";
    WriteText(source, "authored source\n");
    const std::vector<std::filesystem::path> controls = {
        root / "Assets" / "Vendor" / ".git" / "HEAD",
        root / "Assets" / "Vendor" / ".git" / "objects" / "Source.py",
        root / "Assets" / "Vendor" / ".hg" / "state.txt",
        root / "Assets" / "Vendor" / ".svn" / "state.txt",
        root / "Assets" / "Vendor" / ".gitignore",
        root / "Assets" / "Vendor" / ".gitattributes",
        root / "Assets" / "Vendor" / ".gitmodules",
        root / "Assets" / "Vendor" / ".gitkeep",
        root / "Assets" / "Worktree" / ".git",
    };
    for (const auto &control : controls)
        WriteText(control, "version control state\n");

    infernux::JobSystem::Initialize(2);
    try {
        auto database = std::make_unique<infernux::AssetDatabase>();
        database->Initialize(infernux::FromFsPath(root));
        auto &registry = infernux::AssetRegistry::Instance();
        registry.Initialize(std::move(database));
        registry.RegisterLoader(infernux::ResourceType::DefaultText,
                                std::make_unique<infernux::InxDefaultTextLoader>(infernux::ResourceType::DefaultText));
        registry.PopulateAssetDatabaseLoaders();
        auto *assetDatabase = registry.GetAssetDatabase();
        assetDatabase->Refresh();
        Require(!assetDatabase->GetGuidFromPath(infernux::FromFsPath(source)).empty(),
                "VCS filtering excluded an ordinary vendor asset");
        for (const auto &control : controls) {
            const std::string path = infernux::FromFsPath(control);
            Require(assetDatabase->GetGuidFromPath(path).empty(), "VCS control file entered the asset catalog");
            Require(!std::filesystem::exists(infernux::ToFsPath(path + ".meta")),
                    "VCS control file received a metadata sidecar");
            const auto imported = assetDatabase->ImportAsset(path);
            const auto reimported = assetDatabase->ReimportAsset(path);
            Require(!imported && imported.errorCode == infernux::AssetMutationErrorCode::UnsupportedType,
                    "explicit import accepted a VCS control file");
            Require(!reimported && reimported.errorCode == infernux::AssetMutationErrorCode::UnsupportedType,
                    "explicit reimport accepted a VCS control file");
        }
        registry.Shutdown();
        infernux::JobSystem::Shutdown();
    } catch (...) {
        if (infernux::AssetRegistry::Instance().IsInitialized())
            infernux::AssetRegistry::Instance().Shutdown();
        infernux::JobSystem::Shutdown();
        std::filesystem::remove_all(root);
        throw;
    }
    std::filesystem::remove_all(root);
}
} // namespace

int main()
{
    try {
        TestImporterExtensionsAreCaseInsensitive();
        TestNonGuidRenderEffectFieldsDoNotCreateDependenciesOnInitialRefresh();
        TestScriptReimportRefreshesContentHashAndPreservesGuid();
        TestValidButStaleSidecarIsRebuiltFromCurrentSource();
        TestProjectPackagesScanRootSharesTheGuidCatalog();
        for (const bool keepIndex : {true, false})
            for (const bool changeSource : {false, true})
                for (const bool packageScript : {false, true})
                    TestRegisteredScriptKeepsGuidWhenSidecarAndDerivedIndexAreMissing(keepIndex, changeSource,
                                                                                     packageScript);
        TestStartupCatalogSurvivesLiveIndexInvalidation();
        TestRuntimeAssetCatalogInstallsStableIdentityWithoutSidecar();
        TestCompositeModelPublishesExternalTextureGuidDependencies();
        TestModelSettingsPublishOnlyAfterSuccessfulImport();
        TestCopiedSkeletonRefreshWithLimitedWorkers(false);
        TestCopiedSkeletonRefreshWithLimitedWorkers(true);
        TestRuntimeAssetCatalogResolvesBuiltInArchiveResources();
        TestRuntimeAssetCatalogResolvesPrimaryContentArtifact();
        TestCookedModelTexturesKeepArtifactPaths();
        TestMoveRequiresRegisteredGuidIdentity();
        TestBlenderNumberedBackupsAreNotAssets();
        TestVersionControlFilesAreNotAssets();
        TestRefreshPublishesDeterministicAssetEvents();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Asset database refresh test failed: " << error.what() << '\n';
        return 1;
    }
}
