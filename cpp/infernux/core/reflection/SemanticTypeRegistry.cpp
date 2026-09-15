#include "SemanticTypeRegistry.h"

#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace infernux
{
namespace
{
void RequireData(const nlohmann::json &value)
{
    if (value.is_number_float() && !std::isfinite(value.get<double>()))
        throw std::invalid_argument("semantic field metadata contains a non-finite number");
    if (value.is_binary() || value.is_discarded())
        throw std::invalid_argument("semantic field metadata must be a value document");
    if (value.is_structured()) {
        for (const auto &item : value)
            RequireData(item);
    }
}

void ValidateType(const SemanticTypeDescriptor &type, const std::string &owner)
{
    if (type.typeGuid.empty() || type.readableId.empty() || type.owner != owner || type.schemaVersion == 0)
        throw std::invalid_argument("semantic type requires an identity, matching owner and schema version");
    if (type.origin != "native" && type.origin != "python")
        throw std::invalid_argument("semantic type origin must be native or python");
    std::unordered_set<std::string> fields;
    for (const auto &field : type.fields) {
        if (field.propertyPath.empty() || field.valueType.empty() || field.valueType == "FieldType.UNKNOWN" ||
            field.valueType == "Any")
            throw std::invalid_argument("semantic field requires a declared value type and property path");
        if (!field.attributes.is_object() || !field.attributes.contains("field_id") ||
            !field.attributes.at("field_id").is_string() || !field.attributes.contains("default"))
            throw std::invalid_argument("semantic field requires a field_id and encoded default");
        const auto id = field.attributes.at("field_id").get<std::string>();
        if (id.empty() || !fields.insert(id).second)
            throw std::invalid_argument("semantic type has an empty or duplicate field_id");
        RequireData(field.attributes);
    }
}
} // namespace

SemanticTypeRegistry::SemanticTypeRegistry() : m_snapshot(std::make_shared<const SemanticCatalogSnapshot>())
{
}

SemanticTypeRegistry &SemanticTypeRegistry::Instance()
{
    static SemanticTypeRegistry registry;
    return registry;
}

SemanticTypeRegistry::Snapshot SemanticTypeRegistry::Read() const
{
    return std::atomic_load(&m_snapshot);
}

SemanticTypeRegistry::PreparedPublication
SemanticTypeRegistry::Prepare(const std::vector<SemanticOwnerEdit> &edits) const
{
    if (edits.empty())
        throw std::invalid_argument("semantic publication requires an owner edit");
    const auto base = Read();
    if (base->revision == std::numeric_limits<uint64_t>::max())
        throw std::overflow_error("semantic catalog revision exhausted");
    auto next = std::make_shared<SemanticCatalogSnapshot>(*base);
    next->revision = base->revision + 1;
    std::unordered_set<std::string> owners;
    for (const auto &edit : edits) {
        if (edit.owner.empty() || !owners.insert(edit.owner).second)
            throw std::invalid_argument("semantic publication has an empty or duplicate owner");
    }
    for (auto iterator = next->types.begin(); iterator != next->types.end();) {
        if (owners.count(iterator->second->owner))
            iterator = next->types.erase(iterator);
        else
            ++iterator;
    }
    for (const auto &edit : edits) {
        for (auto type : edit.types) {
            ValidateType(type, edit.owner);
            const auto previous = base->types.find(type.typeGuid);
            if (previous != base->types.end() && previous->second->owner != edit.owner)
                throw std::invalid_argument("semantic type GUID belongs to another owner");
            type.revision = next->revision;
            auto descriptor = std::make_shared<const SemanticTypeDescriptor>(std::move(type));
            if (!next->types.emplace(descriptor->typeGuid, descriptor).second)
                throw std::invalid_argument("semantic publication contains a duplicate type GUID");
        }
    }
    next->readableIds.clear();
    for (const auto &[guid, type] : next->types) {
        if (!next->readableIds.emplace(type->readableId, guid).second)
            throw std::invalid_argument("semantic publication contains a duplicate readable ID");
        std::unordered_set<std::string> chain{guid};
        auto parent = type->baseTypeGuid;
        while (!parent.empty()) {
            if (!chain.insert(parent).second)
                throw std::invalid_argument("semantic type inheritance contains a cycle");
            const auto found = next->types.find(parent);
            if (found == next->types.end())
                throw std::invalid_argument("semantic base type is absent from the candidate catalog");
            parent = found->second->baseTypeGuid;
        }
    }
    PreparedPublication publication;
    publication.m_registry = this;
    publication.m_base = base;
    publication.m_candidate = std::move(next);
    return publication;
}

uint64_t SemanticTypeRegistry::Publish(const PreparedPublication &publication)
{
    std::lock_guard<std::mutex> lock(m_writer);
    if (publication.m_registry != this || !publication.m_candidate || Read() != publication.m_base)
        throw std::logic_error("semantic publication is foreign, empty or stale");
    std::atomic_store(&m_snapshot, publication.m_candidate);
    return publication.m_candidate->revision;
}
} // namespace infernux
