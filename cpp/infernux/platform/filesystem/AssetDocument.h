#pragma once

#include "core/FoundationApi.h"

#include <nlohmann/json.hpp>
#include <string>
#include <string_view>

namespace infernux
{
// Cooked, GUID-addressed asset documents use CBOR. JSON remains the authoring
// format; arbitrary project payloads never pass through this codec.
INFERNUX_FOUNDATION_API std::string EncodeAssetDocument(const nlohmann::json &document);
INFERNUX_FOUNDATION_API nlohmann::json DecodeAssetDocument(std::string_view bytes);
nlohmann::json ParseAssetDocument(const std::string &path, std::string_view bytes);
INFERNUX_FOUNDATION_API nlohmann::json ReadAssetDocument(const std::string &path);
INFERNUX_FOUNDATION_API bool IsCookedAssetDocument(const std::string &path);
} // namespace infernux
