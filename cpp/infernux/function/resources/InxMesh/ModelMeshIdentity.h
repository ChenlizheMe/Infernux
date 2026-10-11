#pragma once

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace infernux
{

// Resolve the identity against one published manifest. A serialized path is
// deliberately absent: it cannot rescue missing or ambiguous identities.
inline std::vector<std::string> ResolveModelMeshIdentityPath(const nlohmann::json &manifest,
                                                             const std::string &subresourceId)
{
    if (!manifest.is_array())
        throw std::invalid_argument("Model mesh identity manifest must be an array");

    std::vector<std::string> resolvedPath;
    size_t matches = 0;
    for (const auto &entry : manifest) {
        if (!entry.is_object() || entry.value("subresource_id", std::string{}) != subresourceId)
            continue;
        if (!entry.contains("path") || !entry["path"].is_array())
            throw std::invalid_argument("Model mesh identity has no node path");
        resolvedPath = entry["path"].get<std::vector<std::string>>();
        ++matches;
    }
    if (matches != 1)
        throw std::invalid_argument(matches == 0 ? "Model mesh identity no longer exists"
                                                 : "Model mesh identity is ambiguous");
    if (resolvedPath.empty())
        throw std::invalid_argument("Model mesh identity has no node path");
    return resolvedPath;
}

} // namespace infernux
