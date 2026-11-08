#pragma once

#include <core/types/InxFwdType.h>
#include <functional>
#include <nlohmann/json.hpp>
#include <string>

namespace infernux
{

// Direct authored references only: no catalog access, resource loading or
// graph mutation. Import workers and owner-side Scene preflight share rules.
using SerializedAssetVisitor =
    std::function<void(const std::string &guid, ResourceType type, const std::string &location)>;

void VisitSceneResourceReferences(const nlohmann::json &document, const SerializedAssetVisitor &visit);
void VisitPrefabResourceReferences(const nlohmann::json &document, const SerializedAssetVisitor &visit);

} // namespace infernux
