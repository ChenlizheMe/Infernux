#include <chrono>
#include <core/threading/JobSystem.h>
#include <filesystem>
#include <fstream>
#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/RenderTexture/RenderTextureArtifact.h>
#include <function/resources/RenderTexture/RenderTextureLoader.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <platform/filesystem/InxPath.h>
#include <stdexcept>

namespace
{
using namespace infernux;
void Require(bool ok, const char *message)
{
    if (!ok)
        throw std::runtime_error(message);
}
void Write(const std::filesystem::path &path, const std::string &bytes)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary);
    stream.write(bytes.data(), bytes.size());
    Require(stream.good(), "fixture write failed");
}
std::string Read(const std::filesystem::path &path)
{
    std::ifstream stream(path, std::ios::binary);
    Require(stream.good(), "fixture read failed");
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}
template <class F> void Reject(F action)
{
    bool rejected = false;
    try {
        action();
    } catch (const std::exception &) {
        rejected = true;
    }
    Require(rejected, "invalid RenderTexture input was accepted");
}
void TestCodec()
{
    rhi::RenderTextureDesc desc;
    desc.width = 641;
    desc.height = 401;
    auto doc = RenderTextureArtifact::SerializeDocument(desc);
    auto bytes = RenderTextureArtifact::Encode(desc, "0123456789abcdef");
    Require(RenderTextureArtifact::Decode(bytes).width == 641, "absolute descriptor roundtrip failed");
    Require(bytes.substr(16, 16) == "0123456789abcdef", "imported hash envelope changed");
    desc.sizeMode = rhi::RenderTextureSizeMode::Relative;
    desc.widthScale = 0.5f;
    desc.heightScale = 0.25f;
    desc.colorFormat = rhi::PixelFormat::RGBA16SFloat;
    desc.depthFormat = rhi::PixelFormat::D32SFloat;
    desc.samples = rhi::SampleCount::Four;
    desc.sampledDepth = true;
    desc.storage = true;
    auto relative = RenderTextureArtifact::Decode(RenderTextureArtifact::Encode(desc, "fedcba9876543210"));
    Require(relative.sizeMode == rhi::RenderTextureSizeMode::Relative && relative.widthScale == 0.5f &&
                relative.heightScale == 0.25f && relative.sampledDepth && relative.storage &&
                relative.samples == rhi::SampleCount::Four && relative.colorFormat == desc.colorFormat &&
                relative.depthFormat == desc.depthFormat,
            "relative HDR/MSAA descriptor roundtrip failed");
    for (size_t i = 0; i < bytes.size(); ++i)
        Reject([&] { RenderTextureArtifact::Decode(std::string_view(bytes).substr(0, i)); });
    Reject([&] { RenderTextureArtifact::Decode(bytes + '\0'); });
    Reject([&] { RenderTextureArtifact::Encode(desc, ""); });
    for (const auto &patch : std::vector<nlohmann::json>{{{"samples", 256}},
                                                         {{"samples", true}},
                                                         {{"format", "d32_sfloat"}},
                                                         {{"sampled_depth", true}},
                                                         {{"filter", "unknown"}},
                                                         {{"storage", 1}},
                                                         {{"size", {{"width", 0}, {"height", 3}}}},
                                                         {{"size", {{"width", 4294967296ULL}, {"height", 3}}}},
                                                         {{"size", {{"scale", {0.0, 0.5}}}}},
                                                         {{"size", {{"scale", {true, 0.5}}}}}}) {
        auto invalid = doc;
        invalid.update(patch);
        Reject([&] { RenderTextureArtifact::ParseDocument(invalid); });
    }
    auto obsolete = doc;
    obsolete["schema_version"] = 99;
    obsolete["old_path"] = "Assets/Old.rendertexture";
    obsolete["size"]["obsolete_unit"] = "pixels";
    Require(RenderTextureArtifact::ParseDocument(obsolete).width == desc.width,
            "obsolete RenderTexture fields were not ignored");
}

void TestImportAndReload()
{
    const auto root =
        std::filesystem::temp_directory_path() /
        ("infernux-render-texture-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto source = root / "Assets" / "Rendering" / "Monitor.rendertexture";
    auto description = RenderTextureArtifact::SerializeDocument({});
    Write(source, description.dump());
    JobSystem::Initialize(2);
    auto &registry = AssetRegistry::Instance();
    try {
        auto database = std::make_unique<AssetDatabase>();
        database->Initialize(FromFsPath(root));
        registry.Initialize(std::move(database));
        registry.RegisterLoader(ResourceType::RenderTexture, std::make_unique<RenderTextureLoader>());
        registry.PopulateAssetDatabaseLoaders();
        auto *adb = registry.GetAssetDatabase();
        adb->Refresh();
        const auto guid = adb->GetGuidFromPath(FromFsPath(source));
        Require(!guid.empty(), "import did not publish GUID");
        const auto artifact = ToFsPath(adb->GetRuntimeArtifactPath(guid, ResourceType::RenderTexture));
        Require(artifact.filename() == guid + ".inxrtex", "artifact is not GUID-addressed");
        auto loaded = registry.LoadAsset<rhi::RenderTextureDesc>(guid, ResourceType::RenderTexture);
        Require(loaded && loaded->width == 256, "headless imported description load failed");
        description["size"] = {{"scale", {0.5, 0.25}}};
        Write(source, description.dump());
        const auto imported = adb->ReimportAsset(FromFsPath(source));
        Require(imported.succeeded && imported.guid == guid, "reimport lost identity");
        registry.ReloadAsset(guid);
        Require(loaded->sizeMode == rhi::RenderTextureSizeMode::Relative && loaded->widthScale == 0.5f,
                "live CPU descriptor was not updated in place");
        const auto previous = Read(artifact);
        description["samples"] = 3;
        Write(source, description.dump());
        Require(!adb->ReimportAsset(FromFsPath(source)).succeeded, "invalid reimport committed");
        Require(Read(artifact) == previous && loaded->widthScale == 0.5f, "invalid edit replaced committed artifact");
        registry.Shutdown();
        JobSystem::Shutdown();
        std::filesystem::remove_all(root);
    } catch (...) {
        if (registry.IsInitialized())
            registry.Shutdown();
        JobSystem::Shutdown();
        std::filesystem::remove_all(root);
        throw;
    }
}
} // namespace
int main()
{
    try {
        TestCodec();
        TestImportAndReload();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
