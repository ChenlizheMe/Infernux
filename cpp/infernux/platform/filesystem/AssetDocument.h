#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <string_view>

namespace infernux
{
// Cooked, GUID-addressed asset documents use CBOR. JSON remains the authoring
// format; arbitrary project payloads never pass through this codec.
std::string EncodeAssetDocument(const nlohmann::json &document);
nlohmann::json DecodeAssetDocument(std::string_view bytes);
nlohmann::json ReadAssetDocument(const std::string &path);
bool IsCookedAssetDocument(const std::string &path);
} // namespace infernux
