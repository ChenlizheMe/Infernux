#include "RenderTextureArtifact.h"

#include <algorithm>
#include <array>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace infernux
{
namespace
{
using namespace rhi;
constexpr std::string_view Header = "INXRTEX1";
constexpr std::string_view EndianMarker = "\x04\x03\x02\x01";

bool IsSourceHash(std::string_view value)
{
    return value.size() == 16 && std::all_of(value.begin(), value.end(), [](char c) {
               return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
           });
}
// Stable file names, independent of Vulkan/backend enum values.
constexpr std::pair<std::string_view, PixelFormat> Formats[] = {
    {"undefined", PixelFormat::Undefined},
    {"r8_unorm", PixelFormat::R8UNorm},
    {"rg8_unorm", PixelFormat::RG8UNorm},
    {"r16_sfloat", PixelFormat::R16SFloat},
    {"rgba4_unorm_pack16", PixelFormat::RGBA4UNormPack16},
    {"rgba8_unorm", PixelFormat::RGBA8UNorm},
    {"rgba8_srgb", PixelFormat::RGBA8Srgb},
    {"bgra8_unorm", PixelFormat::BGRA8UNorm},
    {"bgra8_srgb", PixelFormat::BGRA8Srgb},
    {"rg16_sfloat", PixelFormat::RG16SFloat},
    {"r32_sfloat", PixelFormat::R32SFloat},
    {"rgb10a2_unorm", PixelFormat::RGB10A2UNorm},
    {"d32_sfloat", PixelFormat::D32SFloat},
    {"d24_unorm_s8_uint", PixelFormat::D24UNormS8UInt},
    {"rgba16_sfloat", PixelFormat::RGBA16SFloat},
    {"rgba16_unorm", PixelFormat::RGBA16UNorm},
    {"rg32_uint", PixelFormat::RG32UInt},
    {"rgba32_sfloat", PixelFormat::RGBA32SFloat},
};

PixelFormat ParseFormat(const nlohmann::json &value)
{
    if (value.is_string())
        for (const auto &[name, format] : Formats)
            if (value.get_ref<const std::string &>() == name)
                return format;
    throw std::invalid_argument("RenderTexture has an unknown attachment format");
}

std::string_view FormatName(PixelFormat value)
{
    for (const auto &[name, format] : Formats)
        if (format == value)
            return name;
    throw std::invalid_argument("RenderTexture has an unknown attachment format");
}

uint32_t PositiveInteger(const nlohmann::json &value)
{
    if (!value.is_number_integer() || value.get<double>() < 1 ||
        value.get<double>() > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("RenderTexture requires a positive uint32 integer");
    return value.get<uint32_t>();
}
} // namespace

std::vector<std::string> RenderTextureArtifact::FormatNames(bool depth)
{
    std::vector<std::string> names;
    for (const auto &[name, format] : Formats)
        if ((format == PixelFormat::Undefined && depth) ||
            (format != PixelFormat::Undefined && IsDepthFormat(format) == depth))
            names.emplace_back(name);
    return names;
}

rhi::RenderTextureDesc RenderTextureArtifact::ParseDocument(const nlohmann::json &doc)
{
    constexpr std::array<std::string_view, 9> fields = {
        "$type", "schema_version", "size", "format", "depth_format", "samples", "filter", "storage", "sampled_depth"};
    if (!doc.is_object() || doc.size() != fields.size())
        throw std::invalid_argument("RenderTexture document has an invalid field set");
    for (const auto field : fields)
        if (!doc.contains(field))
            throw std::invalid_argument("RenderTexture document is missing " + std::string(field));
    if (doc["$type"] != "render_texture" || !doc["schema_version"].is_number_integer() || doc["schema_version"] != 1)
        throw std::invalid_argument("Unsupported RenderTexture document type/version");
    rhi::RenderTextureDesc desc;
    const auto &size = doc["size"];
    if (size.is_object() && size.size() == 2 && size.contains("width") && size.contains("height")) {
        desc.width = PositiveInteger(size["width"]);
        desc.height = PositiveInteger(size["height"]);
    } else if (size.is_object() && size.size() == 1 && size.contains("scale")) {
        const auto &scale = size["scale"];
        if (!scale.is_array() || scale.size() != 2 || !scale[0].is_number() || !scale[1].is_number())
            throw std::invalid_argument("RenderTexture scale requires two numbers");
        desc.sizeMode = rhi::RenderTextureSizeMode::Relative;
        desc.widthScale = scale[0].get<float>();
        desc.heightScale = scale[1].get<float>();
    } else {
        throw std::invalid_argument("RenderTexture size requires pixels or scale, not both");
    }
    desc.colorFormat = ParseFormat(doc["format"]);
    desc.depthFormat = ParseFormat(doc["depth_format"]);
    const auto samples = PositiveInteger(doc["samples"]);
    if (samples != 1 && samples != 2 && samples != 4 && samples != 8)
        throw std::invalid_argument("RenderTexture samples must be 1, 2, 4, or 8");
    desc.samples = static_cast<rhi::SampleCount>(samples);
    if (doc["filter"] == "linear")
        desc.filter = rhi::FilterMode::Linear;
    else if (doc["filter"] == "nearest")
        desc.filter = rhi::FilterMode::Nearest;
    else
        throw std::invalid_argument("RenderTexture filter requires linear or nearest");
    if (!doc["storage"].is_boolean() || !doc["sampled_depth"].is_boolean())
        throw std::invalid_argument("RenderTexture usage flags must be boolean");
    desc.storage = doc["storage"].get<bool>();
    desc.sampledDepth = doc["sampled_depth"].get<bool>();
    rhi::ValidateRenderTextureDescription(desc);
    return desc;
}

nlohmann::json RenderTextureArtifact::SerializeDocument(const rhi::RenderTextureDesc &desc)
{
    rhi::ValidateRenderTextureDescription(desc);
    nlohmann::json size;
    if (desc.sizeMode == rhi::RenderTextureSizeMode::Relative)
        size = {{"scale", {desc.widthScale, desc.heightScale}}};
    else
        size = {{"width", desc.width}, {"height", desc.height}};
    return {{"$type", "render_texture"},
            {"schema_version", 1},
            {"size", std::move(size)},
            {"format", FormatName(desc.colorFormat)},
            {"depth_format", FormatName(desc.depthFormat)},
            {"samples", static_cast<unsigned>(desc.samples)},
            {"filter", desc.filter == rhi::FilterMode::Linear ? "linear" : "nearest"},
            {"storage", desc.storage},
            {"sampled_depth", desc.sampledDepth}};
}

std::string RenderTextureArtifact::Encode(const rhi::RenderTextureDesc &desc, std::string_view sourceHash)
{
    if (!IsSourceHash(sourceHash))
        throw std::invalid_argument("RenderTexture artifact requires the imported source content hash");
    const auto payload = nlohmann::json::to_cbor(SerializeDocument(desc));
    std::string result(Header);
    // Same source-version envelope as the other Library CPU artifacts. The
    // importer already computed this identity; no additional hash is made.
    result.append(EndianMarker);
    result.append("\x10\0\0\0", 4);
    result.append(sourceHash);
    result.append(reinterpret_cast<const char *>(payload.data()), payload.size());
    return result;
}

rhi::RenderTextureDesc RenderTextureArtifact::Decode(std::string_view bytes)
{
    constexpr size_t payloadOffset = 32;
    if (!HasCurrentHeader(bytes) || bytes.size() <= payloadOffset || bytes.substr(8, 4) != EndianMarker ||
        bytes.substr(12, 4) != std::string_view("\x10\0\0\0", 4) || !IsSourceHash(bytes.substr(16, 16)))
        throw std::invalid_argument("Unsupported RenderTexture artifact header");
    return ParseDocument(nlohmann::json::from_cbor(bytes.begin() + payloadOffset, bytes.end()));
}

bool RenderTextureArtifact::HasCurrentHeader(std::string_view bytes) noexcept
{
    return bytes.size() >= Header.size() && bytes.substr(0, Header.size()) == Header;
}
} // namespace infernux
