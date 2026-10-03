#include "function/resources/AssetDatabase/AssetIndex.h"
#include "platform/filesystem/DocumentStore.h"
#include "platform/filesystem/InxPath.h"

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using infernux::AssetIndex;
using infernux::AssetIndexEntry;
using infernux::DocumentStore;
using infernux::ResourceType;

namespace
{

using Clock = std::chrono::steady_clock;

void Require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

AssetIndexEntry MakeEntry(size_t index)
{
    AssetIndexEntry entry;
    entry.normalizedPath = "c:/project/assets/item-" + std::to_string(index) + ".txt";
    entry.guid = "guid-" + std::to_string(index);
    entry.resourceType = ResourceType::DefaultText;
    entry.source = {100 + index, static_cast<int64_t>(1000 + index)};
    entry.meta = {200 + index, static_cast<int64_t>(2000 + index)};
    entry.contentHash = "hash-" + std::to_string(index);
    if (index > 1)
        entry.dependencies = {"guid-0", "guid-1"};
    entry.metadata.AddMetadata("guid", entry.guid);
    entry.metadata.AddMetadata("resource_type", entry.resourceType);
    entry.metadata.AddMetadata("content_hash", entry.contentHash);
    return entry;
}

double Milliseconds(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void TestResourceTypeMetadataRoundTrip()
{
    const std::pair<ResourceType, const char *> cases[] = {
        {ResourceType::Meta, "Meta"},
        {ResourceType::Shader, "Shader"},
        {ResourceType::Texture, "Texture"},
        {ResourceType::Mesh, "Mesh"},
        {ResourceType::Material, "Material"},
        {ResourceType::Script, "Script"},
        {ResourceType::Audio, "Audio"},
        {ResourceType::DefaultText, "DefaultText"},
        {ResourceType::DefaultBinary, "DefaultBinary"},
        {ResourceType::PhysicMaterial, "PhysicMaterial"},
        {ResourceType::RenderEffect, "RenderEffect"},
        {ResourceType::ParticleGraph, "ParticleGraph"},
        {ResourceType::DataAsset, "DataAsset"},
        {ResourceType::RenderTexture, "RenderTexture"},
    };

    for (const auto &[type, name] : cases) {
        infernux::InxResourceMeta metadata;
        metadata.AddMetadata("resource_type", type);
        const auto document = metadata.SerializeDocument();
        Require(document["metadata"]["resource_type"]["value"] == name,
                "ResourceType metadata serialized to the wrong name");

        infernux::InxResourceMeta restored;
        restored.DeserializeDocument(document);
        Require(restored.GetResourceType() == type, "ResourceType metadata failed strict round-trip");
    }
}

void TestSpriteFramesStructuredMetadata()
{
    infernux::InxResourceMeta metadata;
    nlohmann::json current = {
        {"metadata", {{"sprite_frames", {{"type", "json_array"}, {"value", nlohmann::json::array()}}}}},
    };
    metadata.DeserializeDocument(current);
    Require(metadata.SerializeDocument() == current, "Structured sprite_frames failed strict round-trip");

    infernux::InxResourceMeta rebuilt;
    rebuilt.CopyMetadataIfMissing(metadata, "sprite_frames");
    Require(rebuilt.SerializeDocument() == current, "Metadata rebuild changed the structured sprite_frames type tag");
}

void TestMetadataFilePathCanonicalization()
{
    const auto tempRoot = std::filesystem::temp_directory_path() / "infernux-meta-long-path-stability";
    std::filesystem::create_directories(tempRoot);
    const auto sourcePath = tempRoot / "Race.scene";
    {
        std::ofstream source(sourcePath, std::ios::binary | std::ios::trunc);
        source << "{}\n";
    }

    std::filesystem::path aliasPath = sourcePath;
#ifdef INX_PLATFORM_WINDOWS
    const std::wstring native = sourcePath.native();
    const DWORD required = GetShortPathNameW(native.c_str(), nullptr, 0);
    Require(required > 0, "Windows failed to query a short path for metadata canonicalization");
    std::wstring shortPath(static_cast<size_t>(required), L'\0');
    const DWORD written = GetShortPathNameW(native.c_str(), shortPath.data(), required);
    Require(written > 0 && written < required, "Windows failed to produce a short path for metadata canonicalization");
    shortPath.resize(static_cast<size_t>(written));
    aliasPath = std::filesystem::path(std::move(shortPath));
#endif

    infernux::InxResourceMeta metadata;
    const std::string content = "{}\n";
    metadata.Init(content.data(), content.size(), infernux::FromFsPath(aliasPath), ResourceType::DefaultText);
    const auto canonical = std::filesystem::weakly_canonical(sourcePath).lexically_normal();
    Require(metadata.GetDataAs<std::string>("file_path") == infernux::FromFsPath(canonical),
            "Resource metadata retained a filesystem alias instead of the canonical long path");
    std::filesystem::remove_all(tempRoot);
}

void TestPortableMetadataPath()
{
    const auto root = std::filesystem::temp_directory_path() / "infernux-portable-metadata";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "Assets" / "Scripts");
    const auto source = root / "Assets" / "Scripts" / "Player.py";
    {
        std::ofstream output(source);
        output << "class Player: pass\n";
    }

    infernux::InxResourceMeta metadata;
    const std::string sourcePath = infernux::FromFsPath(source);
    const std::string content = "class Player: pass\n";
    metadata.Init(content.data(), content.size(), sourcePath, ResourceType::Script);
    const auto document = metadata.SerializeDocumentPortable(infernux::FromFsPath(root));
    Require(document.at("metadata").at("file_path").at("value") == "Assets/Scripts/Player.py",
            "portable metadata retained the checkout's absolute path");
    Require(!document.at("metadata").contains("last_modified"), "portable metadata retained a local timestamp");
    metadata.AddMetadata("last_modified", "different checkout timestamp");
    Require(document == metadata.SerializeDocumentPortable(infernux::FromFsPath(root)),
            "local observation changed portable metadata");
    infernux::InxResourceMeta child;
    child.Init(content.data(), content.size(), sourcePath, ResourceType::Script);
    child.AddMetadata("file_path", sourcePath + "::subanim:idle");
    metadata.AddMetadata("model_animations", nlohmann::json::array({{{"metadata", child.SerializeDocument()}}}).dump());
    const auto nested = nlohmann::json::parse(metadata.SerializeDocumentPortable(infernux::FromFsPath(root))
                                                  .at("metadata")
                                                  .at("model_animations")
                                                  .at("value")
                                                  .get<std::string>());
    Require(nested.at(0).at("metadata").at("metadata").at("file_path").at("value") ==
                "Assets/Scripts/Player.py::subanim:idle",
            "model sub-asset retained a local absolute path");
    metadata.AddMetadata("file_path", infernux::FromFsPath(root.parent_path() / "outside.py"));
    Require(!metadata.SerializeDocumentPortable(infernux::FromFsPath(root)).at("metadata").contains("file_path"),
            "external sidecar persisted a machine-specific path hint");

    std::filesystem::remove_all(root);
}

void TestScaleAndStrictRoundTrip()
{
    constexpr size_t entryCount = 10'000;
    constexpr size_t queryCount = 100'000;
    AssetIndex index;
    index.Reset("c:/project");
    for (size_t i = 0; i < entryCount; ++i)
        index.Upsert(MakeEntry(i));

    auto start = Clock::now();
    const auto document = index.SerializeDocument();
    const double serializeMs = Milliseconds(start);
    Require(document["entries"].size() == entryCount, "AssetIndex serialized entry count mismatch");

    AssetIndex restored;
    start = Clock::now();
    restored.DeserializeDocument(document, "c:/project");
    const double deserializeMs = Milliseconds(start);
    Require(restored.Size() == entryCount, "AssetIndex round-trip entry count mismatch");

    start = Clock::now();
    for (size_t i = 0; i < queryCount; ++i) {
        const size_t key = (i * 7919) % entryCount;
        const auto *entry = restored.Find("c:/project/assets/item-" + std::to_string(key) + ".txt");
        Require(entry != nullptr && entry->guid == "guid-" + std::to_string(key), "AssetIndex query mismatch");
    }
    const double queryMs = Milliseconds(start);

    const auto tempRoot = std::filesystem::temp_directory_path() / "infernux-asset-index-tests";
    std::filesystem::create_directories(tempRoot);
    const auto indexPath = tempRoot / "AssetIndex.json";
    start = Clock::now();
    restored.Save(infernux::FromFsPath(indexPath));
    AssetIndex loaded;
    Require(loaded.Load(infernux::FromFsPath(indexPath), "c:/project"), "AssetIndex file load returned a miss");
    const double fileRoundTripMs = Milliseconds(start);
    Require(loaded.Size() == entryCount, "AssetIndex file round-trip entry count mismatch");

    auto invalid = document;
    invalid["entries"][0]["unexpected"] = true;
    bool rejected = false;
    try {
        loaded.DeserializeDocument(invalid, "c:/project");
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    Require(rejected, "AssetIndex accepted an unknown entry field");
    Require(loaded.Size() == entryCount, "AssetIndex invalid document partially mutated live state");

    auto legacy = document;
    legacy.erase("import_revision");
    DocumentStore::Instance().WriteAndWait(infernux::FromFsPath(indexPath), legacy.dump());
    Require(!loaded.Load(infernux::FromFsPath(indexPath), "c:/project"),
            "AssetIndex reused a catalog without authoritative Prefab dependencies");
    Require(loaded.Size() == 0, "AssetIndex retained stale entries after a revision change");
    restored.Save(infernux::FromFsPath(indexPath));
    Require(loaded.Load(infernux::FromFsPath(indexPath), "c:/project"),
            "AssetIndex failed to reuse the rebuilt current revision");

    Require(serializeMs < 10'000.0, "AssetIndex 10k serialization exceeded 10 seconds");
    Require(deserializeMs < 10'000.0, "AssetIndex 10k deserialization exceeded 10 seconds");
    Require(queryMs < 2'000.0, "AssetIndex 100k queries exceeded 2 seconds");
    Require(fileRoundTripMs < 15'000.0, "AssetIndex 10k file round-trip exceeded 15 seconds");

    std::cout << "AssetIndex profile: serialize_10k_ms=" << serializeMs << " deserialize_10k_ms=" << deserializeMs
              << " query_100k_ms=" << queryMs << " file_round_trip_10k_ms=" << fileRoundTripMs << '\n';
    DocumentStore::Instance().Shutdown();
    std::filesystem::remove_all(tempRoot);
}

} // namespace

int main()
{
    try {
        TestResourceTypeMetadataRoundTrip();
        TestSpriteFramesStructuredMetadata();
        TestMetadataFilePathCanonicalization();
        TestPortableMetadataPath();
        TestScaleAndStrictRoundTrip();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "AssetIndex test failed: " << error.what() << '\n';
        return 1;
    }
}
