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

/// Snapshot-only tables. They retain tombstones across Undo and Play restore;
/// authored asset files carry GUIDs directly and must not contain these tables.
[[nodiscard]] nlohmann::json SerializeSceneAuthoringIdentity(const SceneAuthoringIdentity &identities);
[[nodiscard]] SceneAuthoringIdentity DeserializeSceneAuthoringIdentity(const nlohmann::json &document);
[[nodiscard]] SceneAuthoringIdentity
RemapSceneAuthoringIdentity(const SceneAuthoringIdentity &identities,
                            const std::unordered_map<uint64_t, uint64_t> &objectIdRemap,
                            const std::unordered_map<uint64_t, uint64_t> &componentIdRemap);

/// Encode the current authoring file schema. Every non-null identity must have
/// been allocated by the authoring owner; this boundary never invents an ID.
[[nodiscard]] nlohmann::json EncodeSceneAuthoringDocument(const nlohmann::json &runtimeDocument,
                                                          const SceneAuthoringIdentity &identities);

/// Decode GUIDs to document-local integers in traversal order. Native Scene
/// publication subsequently resolves any occupied live runtime IDs as usual.
/// The returned tables preserve both present and missing reference identities.
[[nodiscard]] nlohmann::json DecodeSceneAuthoringDocument(const nlohmann::json &assetDocument,
                                                          SceneAuthoringIdentity &identities);

/// Cooked Player scenes have an explicit runtime-v1 discriminator and compact
/// local IDs. Missing references remain reserved when published additively.
[[nodiscard]] nlohmann::json DecodeSceneRuntimeArtifact(const nlohmann::json &artifactDocument);

} // namespace infernux
