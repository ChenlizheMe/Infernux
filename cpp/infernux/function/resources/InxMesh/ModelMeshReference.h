#pragma once

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace infernux
{
inline constexpr const char *ModelMeshToken = "::submesh:";

// The suffix encodes the source node path, not a transient import index. Hex
// keeps DCC names (including slashes/Unicode) out of filesystem normalization.
inline std::string MakeModelMeshReference(const std::string &base, const std::vector<std::string> &path)
{
    if (base.empty() || path.empty())
        throw std::invalid_argument("A model mesh reference needs a source and node path");
    constexpr char digits[] = "0123456789abcdef";
    std::string result = base + ModelMeshToken;
    for (unsigned char byte : nlohmann::json(path).dump()) {
        result += digits[byte >> 4];
        result += digits[byte & 15];
    }
    return result;
}

inline std::pair<std::string, std::vector<std::string>> SplitModelMeshReference(const std::string &reference)
{
    const auto position = reference.find(ModelMeshToken);
    if (position == std::string::npos)
        return {reference, {}};
    const std::string suffix = reference.substr(position + std::char_traits<char>::length(ModelMeshToken));
    if (position == 0 || suffix.empty() || suffix.size() % 2)
        throw std::invalid_argument("Malformed model mesh reference");
    const auto nibble = [](char c) -> unsigned {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        throw std::invalid_argument("Malformed model mesh reference encoding");
    };
    std::string decoded;
    for (size_t i = 0; i < suffix.size(); i += 2)
        decoded += static_cast<char>((nibble(suffix[i]) << 4) | nibble(suffix[i + 1]));
    auto path = nlohmann::json::parse(decoded).get<std::vector<std::string>>();
    if (path.empty())
        throw std::invalid_argument("Empty model mesh node path");
    return {reference.substr(0, position), std::move(path)};
}
} // namespace infernux
