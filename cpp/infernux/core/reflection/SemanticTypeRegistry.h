#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace infernux
{

// Same value-only field document consumed by Python FieldSchema. Attributes
// contain codec-encoded defaults and declaration metadata, never callbacks.
struct SemanticFieldDescriptor
{
    std::string propertyPath;
    std::string valueType;
    bool readOnly = false;
    nlohmann::json attributes = nlohmann::json::object();
};

struct SemanticTypeDescriptor
{
    std::string typeGuid;
    std::string readableId;
    std::string owner;
    std::string origin;
    uint32_t schemaVersion = 1;
    uint64_t revision = 0; // Assigned by publication, not by the producer.
    std::string displayName;
    std::string baseTypeGuid;
    bool constructible = true;
    bool serializable = true;
    bool runtimeAvailable = true;
    std::vector<std::string> runtimeProfiles;
    std::vector<std::string> lifecycle;
    std::vector<SemanticFieldDescriptor> fields;
};

struct SemanticCatalogSnapshot
{
    static constexpr uint32_t Format = 1;
    uint64_t revision = 0;
    std::unordered_map<std::string, std::shared_ptr<const SemanticTypeDescriptor>> types;
    std::unordered_map<std::string, std::string> readableIds;
};

// Replaces this owner's complete type set. Empty types retires the owner.
struct SemanticOwnerEdit
{
    std::string owner;
    std::vector<SemanticTypeDescriptor> types;
};

class SemanticTypeRegistry
{
  public:
    using Snapshot = std::shared_ptr<const SemanticCatalogSnapshot>;

    class PreparedPublication
    {
      public:
        [[nodiscard]] Snapshot Candidate() const
        {
            return m_candidate;
        }

      private:
        friend class SemanticTypeRegistry;
        const SemanticTypeRegistry *m_registry = nullptr;
        Snapshot m_base;
        Snapshot m_candidate;
    };

    SemanticTypeRegistry();
    static SemanticTypeRegistry &Instance();
    [[nodiscard]] Snapshot Read() const;
    [[nodiscard]] PreparedPublication Prepare(const std::vector<SemanticOwnerEdit> &edits) const;
    // Final owner commit edge. Stale preparations are rejected, never replayed.
    uint64_t Publish(const PreparedPublication &publication);

  private:
    mutable std::mutex m_writer;
    Snapshot m_snapshot;
};

} // namespace infernux
