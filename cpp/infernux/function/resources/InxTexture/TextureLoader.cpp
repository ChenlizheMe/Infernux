#include "TextureLoader.h"

#include <core/log/InxLog.h>
#include <function/resources/InxFileLoader/InxTextureLoader.hpp>
#include <function/resources/InxTexture/InxTexture.h>

#include <platform/filesystem/InxPath.h>
#include <stb_image.h>

#include <algorithm>
#include <filesystem>

namespace infernux
{

// =============================================================================
// LoadMeta — try to load existing .meta from disk
// =============================================================================

bool TextureLoader::LoadMeta(const char * /*content*/, const std::string &filePath, InxResourceMeta &metaData)
{
    std::string metaPath = InxResourceMeta::GetMetaFilePath(filePath);
    if (std::filesystem::exists(ToFsPath(metaPath))) {
        return metaData.LoadFromFile(metaPath);
    }
    return false;
}

// =============================================================================
// CreateMeta — texture-specific .meta creation (dimensions, format)
// =============================================================================

void TextureLoader::CreateMeta(const char *content, size_t contentSize, const std::string &filePath,
                               InxResourceMeta &metaData)
{
    metaData.Init(content, contentSize, filePath, ResourceType::Texture);

    std::filesystem::path path = ToFsPath(filePath);
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    // Get image dimensions without fully loading the pixel data
    int width = 0, height = 0, channels = 0;
    std::vector<unsigned char> fileBytes;
    if (ReadFileBytes(filePath, fileBytes) && !fileBytes.empty() &&
        stbi_info_from_memory(fileBytes.data(), static_cast<int>(fileBytes.size()), &width, &height, &channels)) {
        metaData.AddMetadata("width", width);
        metaData.AddMetadata("height", height);
        metaData.AddMetadata("channels", channels);
    } else if (!fileBytes.empty()) {
        InxTextureData pnmInfo = InxTextureLoader::LoadFromMemory(fileBytes.data(), fileBytes.size(), filePath);
        if (pnmInfo.IsValid()) {
            metaData.AddMetadata("width", pnmInfo.width);
            metaData.AddMetadata("height", pnmInfo.height);
            metaData.AddMetadata("channels", pnmInfo.channels);
        }
    }

    metaData.AddMetadata("file_type", std::string("texture"));
    metaData.AddMetadata("file_extension", extension);

    static const std::unordered_map<std::string, std::string> formatMap = {
        {".png", "PNG"}, {".jpg", "JPEG"}, {".jpeg", "JPEG"}, {".bmp", "BMP"}, {".tga", "TGA"}, {".gif", "GIF"},
        {".psd", "PSD"}, {".hdr", "HDR"},  {".pic", "PIC"},   {".pnm", "PNM"}, {".pgm", "PGM"}, {".ppm", "PPM"},
    };
    auto fmtIt = formatMap.find(extension);
    metaData.AddMetadata("texture_format", fmtIt != formatMap.end() ? fmtIt->second : std::string("Unknown"));
    metaData.AddMetadata("is_binary", true);

    try {
        if (std::filesystem::exists(path)) {
            metaData.AddMetadata("file_size", static_cast<size_t>(std::filesystem::file_size(path)));
        }
    } catch (const std::filesystem::filesystem_error &) {
    }
}

// =============================================================================
// Load — create an InxTexture with import settings from .meta
// =============================================================================

std::shared_ptr<void> TextureLoader::Load(const std::string &filePath, const std::string &guid, AssetDatabase * /*adb*/)
{
    if (filePath.empty() || guid.empty()) {
        INXLOG_WARN("TextureLoader::Load: empty filePath or guid");
        return nullptr;
    }

    auto texture = std::make_shared<InxTexture>();
    texture->SetGuid(guid);
    texture->SetFilePath(filePath);
    texture->SetName(FromFsPath(ToFsPath(filePath).stem()));

    // Read import settings from .meta (sRGB, mipmaps, texture_type).
    // Missing .meta is not an error — defaults (sRGB=true, mipmaps=true) apply.
    texture->LoadImportSettings(filePath);

    return texture;
}

// =============================================================================
// Reload — refresh import settings in-place (pointer identity preserved)
// =============================================================================

bool TextureLoader::Reload(std::shared_ptr<void> existing, const std::string &filePath, const std::string &guid,
                           AssetDatabase * /*adb*/)
{
    auto tex = std::static_pointer_cast<InxTexture>(existing);
    if (!tex) {
        INXLOG_WARN("TextureLoader::Reload: null existing instance");
        return false;
    }

    // Reload import settings from .meta
    tex->LoadImportSettings(filePath);

    INXLOG_INFO("TextureLoader: reloaded '", tex->GetName(), "' in-place (GUID: ", guid, ")");
    return true;
}

// =============================================================================
// ScanDependencies — textures have no outgoing dependencies
// =============================================================================

std::set<std::string> TextureLoader::ScanDependencies(const std::string & /*filePath*/, AssetDatabase * /*adb*/)
{
    return {};
}

} // namespace infernux
