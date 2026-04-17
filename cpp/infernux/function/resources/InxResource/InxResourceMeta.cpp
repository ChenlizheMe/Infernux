#include "InxResourceMeta.h"

#include <core/log/InxLog.h>
#include <nlohmann/json.hpp>
#include <platform/filesystem/InxPath.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <random>
#include <sstream>

namespace infernux
{

namespace
{
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
} // namespace

// ----------------------------------
// InxResourceMeta Implementation
// ----------------------------------
void InxResourceMeta::Init(const char *content, size_t contentSize, const std::string &filePath, ResourceType type)
{
    // Store resource path in metadata
    AddMetadata("file_path", filePath);
    // Set resource type
    AddMetadata("resource_type", type);

    // Calculate content hash (for change detection)
    AddMetadata("content_hash", ComputeContentHashHex(content, contentSize));

    // Generate a random GUID (stable once stored in .meta)
    // This remains unchanged across moves/renames because the meta is preserved.
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;
    uint64_t hi = dist(gen);
    uint64_t lo = dist(gen);
    std::stringstream guidSs;
    guidSs << std::hex << std::setfill('0') << std::setw(16) << hi << std::setw(16) << lo;
    std::string guid = guidSs.str();

    AddMetadata("guid", guid);

    // Add importer version for future migration support
    AddMetadata("importer_version", 1);

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

void InxResourceMeta::AddMetadata(const std::string &key, const std::any &value)
{
    m_metadata[key] = std::make_pair(InxTypeRegistry::GetInstance().GetTypeName(value.type()), value);
}

const std::string &InxResourceMeta::GetResourceName() const
{
    static const std::string empty;
    auto it = m_metadata.find("resource_name");
    if (it != m_metadata.end()) {
        return std::any_cast<const std::string &>(it->second.second);
    }
    return empty;
}

const std::string &InxResourceMeta::GetHashCode() const
{
    static const std::string empty;
    auto it = m_metadata.find("hash");
    if (it != m_metadata.end()) {
        return std::any_cast<const std::string &>(it->second.second);
    }
    return empty;
}

const std::string &InxResourceMeta::GetGuid() const
{
    static const std::string empty;
    auto it = m_metadata.find("guid");
    if (it != m_metadata.end()) {
        return std::any_cast<const std::string &>(it->second.second);
    }
    return empty;
}

bool InxResourceMeta::HasKey(const std::string &key) const
{
    return m_metadata.find(key) != m_metadata.end();
}

void InxResourceMeta::UpdateFilePath(const std::string &newFilePath)
{
    // Update file_path but keep the same GUID
    // This is used for move/rename operations
    AddMetadata("file_path", newFilePath);

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
        return std::any_cast<const ResourceType &>(it->second.second);
    }
    return defaultType;
}

std::string InxResourceMeta::GetMetaFilePath(const std::string &resourceFilePath)
{
    return resourceFilePath + ".meta";
}

bool InxResourceMeta::SaveToFile(const std::string &metaFilePath) const
{
    const std::string tempPath = metaFilePath + ".tmp";
    std::ofstream file(ToFsPath(tempPath), std::ios::trunc);
    if (!file.is_open()) {
        INXLOG_ERROR("Failed to open meta temp file for writing: ", tempPath);
        return false;
    }

    try {
        nlohmann::json root;
        root["meta_version"] = 2;

        nlohmann::json entries = nlohmann::json::object();
        for (const auto &[key, metaPair] : m_metadata) {
            const std::string &typeName = metaPair.first;
            const std::any &value = metaPair.second;

            nlohmann::json entry;
            entry["type"] = typeName;

            if (typeName == "string") {
                entry["value"] = std::any_cast<std::string>(value);
            } else if (typeName == "int") {
                entry["value"] = std::any_cast<int>(value);
            } else if (typeName == "bool") {
                entry["value"] = std::any_cast<bool>(value);
            } else if (typeName == "size_t") {
                entry["value"] = std::any_cast<size_t>(value);
            } else if (typeName == "float") {
                entry["value"] = std::any_cast<float>(value);
            } else if (typeName == "enum infernux::ResourceType") {
                entry["value"] = InxTypeRegistry::GetInstance().ToString(typeName, value);
            } else if (typeName == "json_array" || typeName == "json_object") {
                // Stored as a raw JSON string — parse it back so the .meta
                // file contains the original structured JSON, not an escaped string.
                entry["value"] = nlohmann::json::parse(std::any_cast<std::string>(value));
            } else {
                INXLOG_WARN("Unknown type for JSON serialization: ", typeName);
                continue;
            }
            entries[key] = entry;
        }
        root["metadata"] = entries;

        file << root.dump(4) << "\n";
        file.flush();
        file.close();

        std::error_code ec;
        std::filesystem::rename(ToFsPath(tempPath), ToFsPath(metaFilePath), ec);
        if (ec) {
            std::filesystem::remove(ToFsPath(metaFilePath), ec);
            ec.clear();
            std::filesystem::rename(ToFsPath(tempPath), ToFsPath(metaFilePath), ec);
        }
        if (ec) {
            INXLOG_ERROR("Failed to finalize meta file: ", metaFilePath, " error: ", ec.message());
            // Remove orphaned temp file so it doesn't litter the project
            std::error_code removeEc;
            std::filesystem::remove(ToFsPath(tempPath), removeEc);
            return false;
        }

        INXLOG_DEBUG("Meta file saved: ", metaFilePath);
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
        INXLOG_DEBUG("Meta file not found: ", metaFilePath);
        return false;
    }

    try {
        nlohmann::json root = nlohmann::json::parse(file);
        file.close();

        m_metadata.clear();

        if (!root.contains("metadata") || !root["metadata"].is_object())
            return false;

        for (auto &[key, entry] : root["metadata"].items()) {
            if (!entry.contains("type") || !entry.contains("value"))
                continue;

            std::string typeName = entry["type"].get<std::string>();

            try {
                if (typeName == "string") {
                    if (entry["value"].is_string()) {
                        AddMetadata(key, entry["value"].get<std::string>());
                    } else {
                        // Legacy: Python wrote a list/dict with type "string".
                        // Treat it as json_array/json_object to preserve data.
                        typeName = entry["value"].is_array() ? "json_array" : "json_object";
                        m_metadata[key] = std::make_pair(typeName, std::any(entry["value"].dump()));
                    }
                } else if (typeName == "int") {
                    AddMetadata(key, entry["value"].get<int>());
                } else if (typeName == "bool") {
                    AddMetadata(key, entry["value"].get<bool>());
                } else if (typeName == "size_t") {
                    AddMetadata(key, entry["value"].get<size_t>());
                } else if (typeName == "float") {
                    AddMetadata(key, entry["value"].get<float>());
                } else if (typeName == "enum infernux::ResourceType") {
                    std::string valStr = entry["value"].get<std::string>();
                    std::any resourceType = InxTypeRegistry::GetInstance().FromString(typeName, valStr);
                    AddMetadata(key, std::any_cast<ResourceType>(resourceType));
                } else if (typeName == "json_array" || typeName == "json_object") {
                    // Python-only structured data (e.g. sprite_frames) — store
                    // the raw JSON string so the round-trip through C++ SaveToFile
                    // preserves it, but C++ code never needs to interpret it.
                    AddMetadata(key, entry["value"].dump());
                } else {
                    INXLOG_WARN("Unknown type in JSON meta: ", typeName);
                }
            } catch (const std::exception &entryEx) {
                INXLOG_WARN("Skipping meta entry '", key, "' (type ", typeName, "): ", entryEx.what());
            }
        }
        INXLOG_DEBUG("Meta file loaded: ", metaFilePath);
        return true;
    } catch (const std::exception &e) {
        INXLOG_ERROR("Exception while loading meta file: ", metaFilePath, " - ", e.what());
        return false;
    }
}

} // namespace infernux
