#include "AssetDocument.h"
#include "InxPath.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>

namespace infernux
{
namespace
{
constexpr std::string_view Magic = "INXDOCUMENT";
}

std::string EncodeAssetDocument(const nlohmann::json &document)
{
    const auto payload = nlohmann::json::to_cbor(document);
    std::string result(Magic);
    result.append(reinterpret_cast<const char *>(payload.data()), payload.size());
    return result;
}

nlohmann::json DecodeAssetDocument(std::string_view bytes)
{
    if (bytes.substr(0, Magic.size()) != Magic)
        throw std::runtime_error("Invalid cooked asset document header");
    try {
        return nlohmann::json::from_cbor(bytes.begin() + Magic.size(), bytes.end());
    } catch (const nlohmann::json::exception &error) {
        throw std::runtime_error(std::string("Invalid cooked asset document: ") + error.what());
    }
}

bool IsCookedAssetDocument(const std::string &path)
{
    auto extension = FromFsPath(ToFsPath(path).extension());
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return extension == ".inxdoc";
}

nlohmann::json ReadAssetDocument(const std::string &path)
{
    std::ifstream file(ToFsPath(path), std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open asset document: " + path);
    if (IsCookedAssetDocument(path)) {
        const std::string bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        return DecodeAssetDocument(bytes);
    }
    return nlohmann::json::parse(file);
}
} // namespace infernux
