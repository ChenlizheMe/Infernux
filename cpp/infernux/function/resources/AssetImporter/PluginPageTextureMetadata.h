#pragma once

#include <function/resources/InxResource/InxResourceMeta.h>
#include <platform/filesystem/InxPath.h>

#include <algorithm>
#include <cctype>
#include <string>

namespace infernux
{

inline bool IsPluginPageTexture(const InxResourceMeta &metadata, const std::string &sourcePath,
                                const std::string &projectRoot)
{
    if (projectRoot.empty() || metadata.GetResourceType() != ResourceType::Texture)
        return false;
    const auto path = ToFsPath(FilesystemPathKey(sourcePath));
    std::string relativePath;
    if (!TryMakeRelativeFilesystemPath(sourcePath, projectRoot, relativePath))
        return false;
    const auto relative = ToFsPath(relativePath);
    const auto lower = [](const std::filesystem::path &part) {
        auto value = FromFsPath(part);
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
        return value;
    };
    // Volume data is not a document image, even when stored next to a page.
    const auto extension = lower(path.extension());
    if (extension == ".inxvfield" || extension == ".inxsdf")
        return false;
    auto part = relative.begin();
    if (part == relative.end() || lower(*part) != "packages")
        return false;
    const auto parent = relative.parent_path();
    for (const auto &directory : parent)
        if (lower(directory) == "plugin_pages")
            return true;
    return false;
}

inline void ApplyPluginPageTextureMetadata(InxResourceMeta &metadata, const std::string &sourcePath,
                                           const std::string &projectRoot)
{
    if (!IsPluginPageTexture(metadata, sourcePath, projectRoot))
        return;
    // This is a directory policy, not an editable preset: documentation must
    // retain source dimensions, alpha and precision after every reimport.
    metadata.AddMetadata("texture_type", std::string("ui"));
    metadata.AddMetadata("texture_compression", std::string("none"));
    metadata.AddMetadata("texture_format", std::string("auto"));
    metadata.AddMetadata("max_size", 0);
    metadata.AddMetadata("generate_mipmaps", false);
    metadata.AddMetadata("srgb", true);
    metadata.AddMetadata("wrap_mode", std::string("clamp"));
}

inline bool HasCurrentPluginPageTextureMetadata(const InxResourceMeta &metadata, const std::string &sourcePath,
                                                const std::string &projectRoot)
{
    if (!IsPluginPageTexture(metadata, sourcePath, projectRoot))
        return true;
    return metadata.HasKey("texture_type") && metadata.GetDataAs<std::string>("texture_type") == "ui" &&
           metadata.HasKey("texture_compression") && metadata.GetDataAs<std::string>("texture_compression") == "none" &&
           metadata.HasKey("texture_format") && metadata.GetDataAs<std::string>("texture_format") == "auto" &&
           metadata.HasKey("max_size") && metadata.GetDataAs<int>("max_size") == 0 &&
           metadata.HasKey("generate_mipmaps") && !metadata.GetDataAs<bool>("generate_mipmaps") &&
           metadata.HasKey("srgb") && metadata.GetDataAs<bool>("srgb") && metadata.HasKey("wrap_mode") &&
           metadata.GetDataAs<std::string>("wrap_mode") == "clamp";
}

} // namespace infernux
