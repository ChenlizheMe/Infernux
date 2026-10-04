#include "InxResourceMeta.h"

#include <core/log/InxLog.h>
#include <nlohmann/json.hpp>
#include <platform/filesystem/DocumentStore.h>
#include <platform/filesystem/InxPath.h>

#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <string_view>

namespace infernux
{

namespace
{
ResourceType ParseResourceTypeName(std::string_view value)
{
    if (value == "Meta")
        return ResourceType::Meta;
    if (value == "Shader")
        return ResourceType::Shader;
    if (value == "Texture")
        return ResourceType::Texture;
    if (value == "Mesh")
        return ResourceType::Mesh;
    if (value == "Material")
        return ResourceType::Material;
    if (value == "Script")
        return ResourceType::Script;
    if (value == "Audio")
        return ResourceType::Audio;
    if (value == "DefaultText")
        return ResourceType::DefaultText;
    if (value == "DefaultBinary")
        return ResourceType::DefaultBinary;
    if (value == "PhysicMaterial")
        return ResourceType::PhysicMaterial;
    if (value == "RenderEffect")
        return ResourceType::RenderEffect;
    if (value == "ParticleGraph")
        return ResourceType::ParticleGraph;
    if (value == "DataAsset")
        return ResourceType::DataAsset;
    if (value == "RenderTexture")
        return ResourceType::RenderTexture;
    throw std::invalid_argument("unknown ResourceType metadata value: " + std::string(value));
}

std::string_view ResourceTypeName(ResourceType value)
{
    switch (value) {
    case ResourceType::Meta:
        return "Meta";
    case ResourceType::Shader:
        return "Shader";
    case ResourceType::Texture:
        return "Texture";
    case ResourceType::Mesh:
        return "Mesh";
    case ResourceType::Material:
        return "Material";
    case ResourceType::Script:
        return "Script";
    case ResourceType::Audio:
        return "Audio";
    case ResourceType::DefaultText:
        return "DefaultText";
    case ResourceType::DefaultBinary:
        return "DefaultBinary";
    case ResourceType::PhysicMaterial:
        return "PhysicMaterial";
    case ResourceType::RenderEffect:
        return "RenderEffect";
    case ResourceType::ParticleGraph:
        return "ParticleGraph";
    case ResourceType::DataAsset:
        return "DataAsset";
    case ResourceType::RenderTexture:
        return "RenderTexture";
    }
    throw std::invalid_argument("invalid ResourceType metadata value");
}

std::string ComputeContentHashHex(const char *content, size_t contentSize)
{
    // Stable FNV-1a 64-bit hash
    const uint64_t fnvOffset = 14695981039346656037ull;
    const uint64_t fnvPrime = 1099511628211ull;
    uint64_t hash = fnvOffset;

    if (content && contentSize > 0) {
        const unsigned char *ptr = reinterpret_cast<const unsigned char *>(content);
        for (size_t i = 0; i < contentSize; ++i) {
            hash ^= static_cast<uint64_t>(ptr[i]);
            hash *= fnvPrime;
        }
    }

    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(16) << hash;
    return ss.str();
}

std::string GenerateGuid()
{
    static std::mutex mutex;
    static std::mt19937_64 generator = [] {
        std::random_device device;
        std::array<uint32_t, 10> seedData{};
        for (auto &value : seedData)
            value = device();

        const auto timestamp =
            static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        seedData[8] ^= static_cast<uint32_t>(timestamp);
        seedData[9] ^= static_cast<uint32_t>(timestamp >> 32u);
        std::seed_seq seed(seedData.begin(), seedData.end());
        return std::mt19937_64(seed);
    }();

    uint64_t hi = 0;
    uint64_t lo = 0;
    {
        std::lock_guard lock(mutex);
        hi = generator();
        lo = generator();
    }

    std::stringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16) << hi << std::setw(16) << lo;
    return stream.str();
}

std::string NormalizeMetadataFilePath(const std::string &filePath)
{
    return ResolveFilesystemPath(filePath);
}

std::optional<std::string> PortableMetadataFilePath(const std::string &filePath, const std::string &projectRoot)
{
    const auto virtualSuffix = filePath.find("::");
    const std::string filesystemPath = filePath.substr(0, virtualSuffix);
    const std::string suffix = virtualSuffix == std::string::npos ? "" : filePath.substr(virtualSuffix);

    const auto resolved = ToFsPath(filesystemPath).is_absolute()
                              ? filesystemPath
                              : FromFsPath(ToFsPath(projectRoot) / ToFsPath(filesystemPath));
    std::string relative;
    if (TryMakeRelativeFilesystemPath(resolved, projectRoot, relative))
        return relative + suffix;

    // Tools may explicitly import an external source for temporary authoring.
    // Its sidecar owns the GUID; its physical caller path supplies the location.
    // There is no project-relative path hint to persist for that source.
    return std::nullopt;
}

void MakeMetadataPortable(nlohmann::json &document, const std::string &projectRoot)
{
    if (document.is_array()) {
        for (auto &item : document)
            MakeMetadataPortable(item, projectRoot);
        return;
    }
    if (!document.is_object())
        return;
    auto fields = document.find("metadata");
    if (fields != document.end() && fields->is_object()) {
        // This is a local observation, not an authored importer setting.
        fields->erase("last_modified");
        fields->erase("content_hash");
        for (const auto *key : {"file_path"}) {
            auto entry = fields->find(key);
            if (entry != fields->end() && entry->at("type") == "string") {
                const auto portable = PortableMetadataFilePath(entry->at("value").get<std::string>(), projectRoot);
                if (portable)
                    (*entry)["value"] = *portable;
                else
                    fields->erase(entry);
            }
        }
        // Model tables contain serialized metadata documents inside strings.
        for (const auto *key : {"model_textures", "model_animations"}) {
            auto entry = fields->find(key);
            if (entry != fields->end()) {
                auto table = nlohmann::json::parse(entry->at("value").get<std::string>());
                MakeMetadataPortable(table, projectRoot);
                (*entry)["value"] = table.dump();
            }
        }
    }
    for (auto &value : document)
        MakeMetadataPortable(value, projectRoot);
}
} // namespace

// ----------------------------------
// InxResourceMeta Implementation
// ----------------------------------
void InxResourceMeta::Init(const char *content, size_t contentSize, const std::string &filePath, ResourceType type)
{
    // Store resource path in metadata (always forward-slash UTF-8 for stable lookups)
    AddMetadata("file_path", NormalizeFilePath(filePath));
    // Set resource type
    AddMetadata("resource_type", type);

    // Calculate content hash (for change detection)
    AddMetadata("content_hash", ComputeContentHashHex(content, contentSize));

    // Generate a random GUID (stable once stored in .meta)
    // This remains unchanged across moves/renames because the meta is preserved.
    AddMetadata("guid", GenerateGuid());

    // Get file modification time
    std::string modTimeStr;
    try {
        if (std::filesystem::exists(ToFsPath(filePath))) {
            auto fileTime = std::filesystem::last_write_time(ToFsPath(filePath));
            auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                fileTime - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now());
            auto time_t = std::chrono::system_clock::to_time_t(sctp);
            modTimeStr = std::to_string(time_t);
        } else {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            modTimeStr = std::to_string(time_t);
        }
    } catch (const std::exception &e) {
        INXLOG_WARN("Failed to get file time: ", e.what());
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        modTimeStr = std::to_string(time_t);
    }
    AddMetadata("last_modified", modTimeStr);
}

void InxResourceMeta::AddMetadata(const std::string &key, const std::string &value)
{
    m_metadata[key] = std::make_pair("string", MetadataValue(value));
}

void InxResourceMeta::AddMetadata(const std::string &key, const char *value)
{
    if (!value)
        throw std::invalid_argument("metadata string value cannot be null: " + key);
    AddMetadata(key, std::string(value));
}

void InxResourceMeta::AddMetadata(const std::string &key, int value)
{
    m_metadata[key] = std::make_pair("int", MetadataValue(value));
}

void InxResourceMeta::AddMetadata(const std::string &key, bool value)
{
    m_metadata[key] = std::make_pair("bool", MetadataValue(value));
}

void InxResourceMeta::AddMetadata(const std::string &key, size_t value)
{
    m_metadata[key] = std::make_pair("size_t", MetadataValue(value));
}

void InxResourceMeta::AddMetadata(const std::string &key, float value)
{
    m_metadata[key] = std::make_pair("float", MetadataValue(value));
}

void InxResourceMeta::AddMetadata(const std::string &key, ResourceType value)
{
    m_metadata[key] = std::make_pair("enum infernux::ResourceType", MetadataValue(value));
}

void InxResourceMeta::AddMetadata(const std::string &key, const nlohmann::json &value)
{
    if (value.is_array()) {
        m_metadata[key] = std::make_pair("json_array", MetadataValue(value.dump()));
        return;
    }
    if (value.is_object()) {
        m_metadata[key] = std::make_pair("json_object", MetadataValue(value.dump()));
        return;
    }
    throw std::invalid_argument("metadata JSON value must be an array or object: " + key);
}

bool InxResourceMeta::CopyMetadataIfMissing(const InxResourceMeta &source, const std::string &key)
{
    if (HasKey(key))
        return false;
    const auto sourceEntry = source.m_metadata.find(key);
    if (sourceEntry == source.m_metadata.end())
        return false;
    m_metadata.emplace(key, sourceEntry->second);
    return true;
}

const std::string &InxResourceMeta::GetResourceName() const
{
    static const std::string empty;
    auto it = m_metadata.find("resource_name");
    if (it != m_metadata.end()) {
        return std::get<std::string>(it->second.second);
    }
    return empty;
}

const std::string &InxResourceMeta::GetHashCode() const
{
    static const std::string empty;
    auto it = m_metadata.find("hash");
    if (it != m_metadata.end()) {
        return std::get<std::string>(it->second.second);
    }
    return empty;
}

const std::string &InxResourceMeta::GetGuid() const
{
    static const std::string empty;
    auto it = m_metadata.find("guid");
    if (it != m_metadata.end()) {
        return std::get<std::string>(it->second.second);
    }
    return empty;
}

bool InxResourceMeta::HasKey(const std::string &key) const
{
    return m_metadata.find(key) != m_metadata.end();
}

namespace
{
template <typename T>
T ReadMetadataValue(const InxResourceMeta::MetadataMap &metadata, const std::string &key, std::string_view expectedType)
{
    const auto it = metadata.find(key);
    if (it == metadata.end())
        throw std::invalid_argument("metadata key is missing: " + key);
    if (it->second.first != expectedType)
        throw std::invalid_argument("metadata type mismatch for '" + key + "': expected " + std::string(expectedType) +
                                    ", got " + it->second.first);
    return std::get<T>(it->second.second);
}
} // namespace

std::string InxResourceMeta::GetStringData(const std::string &key) const
{
    return ReadMetadataValue<std::string>(m_metadata, key, "string");
}

int InxResourceMeta::GetIntData(const std::string &key) const
{
    return ReadMetadataValue<int>(m_metadata, key, "int");
}

bool InxResourceMeta::GetBoolData(const std::string &key) const
{
    return ReadMetadataValue<bool>(m_metadata, key, "bool");
}

size_t InxResourceMeta::GetSizeData(const std::string &key) const
{
    return ReadMetadataValue<size_t>(m_metadata, key, "size_t");
}

float InxResourceMeta::GetFloatData(const std::string &key) const
{
    return ReadMetadataValue<float>(m_metadata, key, "float");
}

nlohmann::json InxResourceMeta::GetJsonData(const std::string &key) const
{
    const auto it = m_metadata.find(key);
    if (it == m_metadata.end())
        throw std::invalid_argument("metadata key is missing: " + key);
    const std::string &typeName = it->second.first;
    if (typeName != "json_array" && typeName != "json_object")
        throw std::invalid_argument("metadata type mismatch for '" + key + "': expected JSON, got " + typeName);
    const nlohmann::json value = nlohmann::json::parse(std::get<std::string>(it->second.second));
    if ((typeName == "json_array" && !value.is_array()) || (typeName == "json_object" && !value.is_object()))
        throw std::invalid_argument("metadata JSON shape mismatch for '" + key + "'");
    return value;
}

void InxResourceMeta::UpdateFilePath(const std::string &newFilePath)
{
    // Update file_path but keep the same GUID
    // This is used for move/rename operations
    AddMetadata("file_path", NormalizeFilePath(newFilePath));

    // Update last_modified time
    try {
        if (std::filesystem::exists(ToFsPath(newFilePath))) {
            auto fileTime = std::filesystem::last_write_time(ToFsPath(newFilePath));
            auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                fileTime - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now());
            auto time_t = std::chrono::system_clock::to_time_t(sctp);
            AddMetadata("last_modified", std::to_string(time_t));
        }
    } catch (const std::exception &e) {
        INXLOG_WARN("Failed to update modification time: ", e.what());
    }
}

const InxResourceMeta::MetadataMap &InxResourceMeta::GetMetadata() const
{
    return m_metadata;
}

const ResourceType &InxResourceMeta::GetResourceType() const
{
    static const ResourceType defaultType = ResourceType::DefaultText;
    auto it = m_metadata.find("resource_type");
    if (it != m_metadata.end()) {
        return std::get<ResourceType>(it->second.second);
    }
    return defaultType;
}

std::string InxResourceMeta::GetMetaFilePath(const std::string &resourceFilePath)
{
    return resourceFilePath + ".meta";
}

std::string InxResourceMeta::NormalizeFilePath(const std::string &filePath)
{
    return NormalizeMetadataFilePath(filePath);
}

nlohmann::json InxResourceMeta::SerializeDocument() const
{
    nlohmann::json root;
    nlohmann::json entries = nlohmann::json::object();
    for (const auto &[key, metaPair] : m_metadata) {
        const std::string &typeName = metaPair.first;
        const MetadataValue &value = metaPair.second;

        nlohmann::json entry;
        entry["type"] = typeName;

        if (typeName == "string") {
            entry["value"] = std::get<std::string>(value);
        } else if (typeName == "int") {
            entry["value"] = std::get<int>(value);
        } else if (typeName == "bool") {
            entry["value"] = std::get<bool>(value);
        } else if (typeName == "size_t") {
            entry["value"] = std::get<size_t>(value);
        } else if (typeName == "float") {
            const float number = std::get<float>(value);
            if (!std::isfinite(number))
                throw std::invalid_argument("metadata float must be finite: " + key);
            entry["value"] = number;
        } else if (typeName == "enum infernux::ResourceType") {
            entry["value"] = ResourceTypeName(std::get<ResourceType>(value));
        } else if (typeName == "json_array" || typeName == "json_object") {
            entry["value"] = nlohmann::json::parse(std::get<std::string>(value));
        } else {
            throw std::invalid_argument("unsupported metadata type for '" + key + "': " + typeName);
        }
        entries[key] = std::move(entry);
    }
    root["metadata"] = std::move(entries);
    return root;
}

nlohmann::json InxResourceMeta::SerializeDocumentPortable(const std::string &projectRoot) const
{
    if (projectRoot.empty())
        throw std::invalid_argument("portable metadata requires a project root");
    auto root = SerializeDocument();
    MakeMetadataPortable(root, projectRoot);
    return root;
}

void InxResourceMeta::DeserializeDocument(const nlohmann::json &document)
{
    if (!document.is_object() || document.size() != 1 || !document.contains("metadata"))
        throw std::invalid_argument("metadata document must contain exactly metadata");
    if (!document["metadata"].is_object())
        throw std::invalid_argument("metadata must be an object");

    InxResourceMeta staged;
    for (const auto &[key, entry] : document["metadata"].items()) {
        if (key.empty() || !entry.is_object() || entry.size() != 2 || !entry.contains("type") ||
            !entry.contains("value") || !entry["type"].is_string())
            throw std::invalid_argument("invalid metadata entry: " + key);

        const std::string typeName = entry["type"].get<std::string>();
        const auto &value = entry["value"];
        if (key == "sprite_frames" && typeName != "json_array")
            throw std::invalid_argument("metadata sprite_frames must use json_array");
        if (typeName == "string") {
            if (!value.is_string())
                throw std::invalid_argument("metadata string value expected: " + key);
            // The persisted type tag is the canonical metadata contract.
            staged.m_metadata[key] = std::make_pair(typeName, MetadataValue(value.get<std::string>()));
        } else if (typeName == "int") {
            if (!value.is_number_integer())
                throw std::invalid_argument("metadata int value expected: " + key);
            staged.m_metadata[key] = std::make_pair(typeName, MetadataValue(value.get<int>()));
        } else if (typeName == "bool") {
            if (!value.is_boolean())
                throw std::invalid_argument("metadata bool value expected: " + key);
            staged.m_metadata[key] = std::make_pair(typeName, MetadataValue(value.get<bool>()));
        } else if (typeName == "size_t") {
            if (!value.is_number_unsigned())
                throw std::invalid_argument("metadata unsigned value expected: " + key);
            staged.m_metadata[key] = std::make_pair(typeName, MetadataValue(value.get<size_t>()));
        } else if (typeName == "float") {
            if (!value.is_number())
                throw std::invalid_argument("metadata float value expected: " + key);
            const double number = value.get<double>();
            if (!std::isfinite(number) || number < -std::numeric_limits<float>::max() ||
                number > std::numeric_limits<float>::max())
                throw std::invalid_argument("metadata float must be finite: " + key);
            staged.m_metadata[key] = std::make_pair(typeName, MetadataValue(static_cast<float>(number)));
        } else if (typeName == "enum infernux::ResourceType") {
            if (!value.is_string())
                throw std::invalid_argument("metadata ResourceType string expected: " + key);
            staged.m_metadata[key] =
                std::make_pair(typeName, MetadataValue(ParseResourceTypeName(value.get<std::string>())));
        } else if (typeName == "json_array") {
            if (!value.is_array())
                throw std::invalid_argument("metadata JSON array expected: " + key);
            staged.m_metadata[key] = std::make_pair(typeName, MetadataValue(value.dump()));
        } else if (typeName == "json_object") {
            if (!value.is_object())
                throw std::invalid_argument("metadata JSON object expected: " + key);
            staged.m_metadata[key] = std::make_pair(typeName, MetadataValue(value.dump()));
        } else {
            throw std::invalid_argument("unsupported metadata type for '" + key + "': " + typeName);
        }
    }
    m_metadata = std::move(staged.m_metadata);
}

bool InxResourceMeta::SaveToFile(const std::string &metaFilePath) const
{
    try {
        DocumentStore::Instance().WriteAndWait(metaFilePath, SerializeDocument().dump(4) + "\n");

        return true;
    } catch (const std::exception &e) {
        INXLOG_ERROR("Exception while saving meta file: ", metaFilePath, " - ", e.what());
        return false;
    }
}

bool InxResourceMeta::LoadFromFile(const std::string &metaFilePath)
{
    std::ifstream file(ToFsPath(metaFilePath));
    if (!file.is_open()) {
        return false;
    }

    try {
        const nlohmann::json root = nlohmann::json::parse(file);
        DeserializeDocument(root);
        return true;
    } catch (const std::exception &e) {
        INXLOG_ERROR("Exception while loading meta file: ", metaFilePath, " - ", e.what());
        throw std::runtime_error("Invalid metadata file '" + metaFilePath + "': " + e.what());
    }
}

} // namespace infernux
