#include "SceneAuthoringIdentity.h"

#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unordered_set>

namespace infernux
{
namespace
{
using Json = nlohmann::json;
using IdentityTable = std::unordered_map<uint64_t, std::string>;

void RequireGuid(const std::string &guid)
{
    bool nonzero = false;
    if (guid.size() != 32)
        throw std::invalid_argument("Scene author identity must be a canonical 32-character GUID");
    for (const char digit : guid) {
        if (!((digit >= '0' && digit <= '9') || (digit >= 'a' && digit <= 'f')))
            throw std::invalid_argument("Scene author identity must be a lowercase hexadecimal GUID");
        nonzero = nonzero || digit != '0';
    }
    if (!nonzero)
        throw std::invalid_argument("Scene author identity must not be the zero GUID");
}

uint64_t ReadRuntimeId(const Json &value, bool nullable)
{
    if (!value.is_number_unsigned())
        throw std::invalid_argument("Scene runtime document identity must be an unsigned integer");
    const auto id = value.get<uint64_t>();
    if (id == 0 && !nullable)
        throw std::invalid_argument("Scene object/component identity must not be null");
    return id;
}

template <class ObjectId, class ComponentId>
void RewriteFields(Json &value, ObjectId &objectId, ComponentId &componentId)
{
    if (value.is_array()) {
        for (auto &item : value)
            RewriteFields(item, objectId, componentId);
    } else if (value.is_object()) {
        const auto type = value.find("$type");
        if (type != value.end() && *type == "game_object_ref") {
            value.at("object_id") = objectId(value.at("object_id"), true, false);
            return;
        }
        if (type != value.end() && *type == "component_ref") {
            value.at("game_object_id") = objectId(value.at("game_object_id"), true, false);
            if (value.contains("component_id"))
                value["component_id"] = componentId(value.at("component_id"), true, false);
            return;
        }
        for (auto &item : value)
            RewriteFields(item, objectId, componentId);
    }
}

template <class ObjectId, class ComponentId>
void RewriteDeclarations(Json &object, ObjectId &objectId, ComponentId &componentId)
{
    object.at("id") = objectId(object.at("id"), false, true);
    auto &transform = object.at("transform");
    transform.at("component_id") = componentId(transform.at("component_id"), false, true);
    for (auto &component : object.at("components"))
        component.at("component_id") = componentId(component.at("component_id"), false, true);
    for (auto &child : object.at("children"))
        RewriteDeclarations(child, objectId, componentId);
}

template <class ObjectId, class ComponentId>
void RewriteReferences(Json &object, ObjectId &objectId, ComponentId &componentId)
{
    for (auto &component : object.at("components")) {
        const std::string type = component.at("type_id").get<std::string>();
        auto &fields = component.at("data");
        if (type == "native:infernux.HingeJoint" || type == "native:infernux.SliderJoint")
            fields.at("connected_body_component_id") =
                componentId(fields.at("connected_body_component_id"), true, false);
        if (type.rfind("python:", 0) == 0)
            RewriteFields(fields, objectId, componentId);
    }
    // Prefab baselines/provenance and arbitrary native property dictionaries
    // belong to other namespaces. Never rewrite integers merely by field name.
    for (auto &child : object.at("children"))
        RewriteReferences(child, objectId, componentId);
}

template <class ObjectId, class ComponentId>
void RewriteDocument(Json &document, ObjectId objectId, ComponentId componentId)
{
    if (!document.is_object() || !document.contains("objects") || !document["objects"].is_array())
        throw std::invalid_argument("Scene authoring document requires an objects array");
    // Reserve all declarations before assigning missing-reference placeholders.
    for (auto &object : document["objects"])
        RewriteDeclarations(object, objectId, componentId);
    for (auto &object : document["objects"])
        RewriteReferences(object, objectId, componentId);
    if (document.contains("mainCameraComponentId"))
        document["mainCameraComponentId"] = componentId(document.at("mainCameraComponentId"), false, false);
}

class Encoder
{
  public:
    explicit Encoder(const IdentityTable &identities) : m_identities(identities)
    {
        std::unordered_set<std::string> guids;
        for (const auto &[id, guid] : identities) {
            if (id == 0)
                throw std::invalid_argument("Scene author identity table contains a null runtime ID");
            RequireGuid(guid);
            if (!guids.insert(guid).second)
                throw std::invalid_argument("Scene author identity table aliases two runtime IDs to one GUID");
        }
    }

    Json operator()(const Json &value, bool nullable, bool declaration)
    {
        const auto id = ReadRuntimeId(value, nullable);
        if (id == 0)
            return "";
        if (declaration && !m_declared.insert(id).second)
            throw std::invalid_argument("Scene authoring document contains a duplicate runtime declaration");
        const auto found = m_identities.find(id);
        if (found == m_identities.end())
            throw std::invalid_argument("Scene authoring document has an unallocated persistent identity");
        return found->second;
    }

  private:
    const IdentityTable &m_identities;
    std::unordered_set<uint64_t> m_declared;
};

class Decoder
{
  public:
    explicit Decoder(IdentityTable &identities) : m_identities(identities)
    {
    }

    Json operator()(const Json &value, bool nullable, bool declaration)
    {
        if (!value.is_string())
            throw std::invalid_argument("Scene authoring document identities must be GUID strings");
        const auto &guid = value.get_ref<const std::string &>();
        if (nullable && guid.empty())
            return uint64_t{0};
        RequireGuid(guid);
        if (declaration && !m_declared.insert(guid).second)
            throw std::invalid_argument("Scene authoring document contains a duplicate GUID declaration");
        const auto found = m_runtimeIds.find(guid);
        if (found != m_runtimeIds.end())
            return found->second;
        if (m_nextId == std::numeric_limits<uint64_t>::max())
            throw std::overflow_error("Scene authoring document exhausted runtime identities");
        const auto id = m_nextId++;
        m_runtimeIds.emplace(guid, id);
        m_identities.emplace(id, guid);
        return id;
    }

  private:
    IdentityTable &m_identities;
    std::unordered_map<std::string, uint64_t> m_runtimeIds;
    std::unordered_set<std::string> m_declared;
    uint64_t m_nextId = 1;
};
} // namespace

nlohmann::json EncodeSceneAuthoringDocument(const nlohmann::json &runtimeDocument,
                                            const SceneAuthoringIdentity &identities)
{
    if (runtimeDocument.contains("identity_format"))
        throw std::invalid_argument("Expected a runtime scene snapshot, not an authoring file");
    Json result = runtimeDocument;
    RewriteDocument(result, Encoder(identities.objects), Encoder(identities.components));
    result.erase("nextObjectId");
    result.erase("nextComponentId");
    result["identity_format"] = "guid-v1";
    return result;
}

nlohmann::json DecodeSceneAuthoringDocument(const nlohmann::json &assetDocument, SceneAuthoringIdentity &identities)
{
    if (!assetDocument.is_object() || assetDocument.value("identity_format", Json{}) != "guid-v1" ||
        assetDocument.contains("nextObjectId") || assetDocument.contains("nextComponentId"))
        throw std::invalid_argument("Expected the GUID scene authoring format without runtime watermarks");
    Json result = assetDocument;
    SceneAuthoringIdentity candidate;
    RewriteDocument(result, Decoder(candidate.objects), Decoder(candidate.components));
    result.erase("identity_format");
    result["nextObjectId"] = static_cast<uint64_t>(candidate.objects.size()) + 1;
    result["nextComponentId"] = static_cast<uint64_t>(candidate.components.size()) + 1;
    identities = std::move(candidate);
    return result;
}
} // namespace infernux
