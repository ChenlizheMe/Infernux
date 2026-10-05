#pragma once

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

namespace infernux
{

/// The file identity and the current document's compact numeric ID are distinct.
/// Tables also retain missing reference targets; deletion must not retarget them.
struct SceneAuthoringIdentity
{
    std::unordered_map<uint64_t, std::string> objects;
    std::unordered_map<uint64_t, std::string> components;
};

/// Encode the current authoring file schema. Every non-null identity must have
/// been allocated by the authoring owner; this boundary never invents an ID.
[[nodiscard]] nlohmann::json EncodeSceneAuthoringDocument(const nlohmann::json &runtimeDocument,
                                                          const SceneAuthoringIdentity &identities);

/// Decode GUIDs to document-local integers in traversal order. Native Scene
/// publication subsequently resolves any occupied live runtime IDs as usual.
/// The returned tables preserve both present and missing reference identities.
[[nodiscard]] nlohmann::json DecodeSceneAuthoringDocument(const nlohmann::json &assetDocument,
                                                          SceneAuthoringIdentity &identities);

} // namespace infernux
