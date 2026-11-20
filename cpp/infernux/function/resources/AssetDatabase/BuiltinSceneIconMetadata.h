#pragma once

#include <function/resources/InxResource/InxResourceMeta.h>
#include <platform/filesystem/InxPath.h>

#include <string>

namespace infernux
{

// Import policy for the editor's immutable packaged Scene billboard assets.
// These package resources have no editable .meta sidecars. Ordinary project
// textures keep their own authored/default import settings.
inline bool IsBuiltinSceneIcon(const std::string &path, bool readOnly)
{
    if (!readOnly)
        return false;
    // Keep filesystem canonicalization in the InxPath owner layer.  Scene
    // icon metadata is queried from several resource backends; duplicating
    // lexical normalization here bypasses the platform path policy.
    const auto source = ToFsPath(NormalizeFilesystemPathLexically(path));
    const auto name = FromFsPath(source.filename());
    const auto icons = source.parent_path();
    const auto resources = icons.parent_path();
    const auto owner = resources.parent_path();
    const bool packagedSource =
        icons.filename() == "icons" && resources.filename() == "resources" && owner.filename() == "infernux";
    const bool projectLibraryMirror =
        icons.filename() == "icons" && resources.filename() == "Resources" && owner.filename() == "Library";
    return (packagedSource || projectLibraryMirror) &&
           (name == "gizmo_camera.png" || name == "gizmo_light.png" || name == "gizmo_particle.png");
}

inline void ApplyBuiltinSceneIconMetadata(InxResourceMeta &metadata, const std::string &path, bool readOnly)
{
    if (!IsBuiltinSceneIcon(path, readOnly))
        return;
    metadata.AddMetadata("texture_type", std::string("ui"));
    metadata.AddMetadata("texture_compression", std::string("none"));
    metadata.AddMetadata("wrap_mode", std::string("clamp"));
}

inline bool HasCurrentBuiltinSceneIconMetadata(const InxResourceMeta &metadata, const std::string &path, bool readOnly)
{
    if (!IsBuiltinSceneIcon(path, readOnly))
        return true;
    return metadata.HasKey("texture_type") && metadata.GetDataAs<std::string>("texture_type") == "ui" &&
           metadata.HasKey("texture_compression") && metadata.GetDataAs<std::string>("texture_compression") == "none" &&
           metadata.HasKey("wrap_mode") && metadata.GetDataAs<std::string>("wrap_mode") == "clamp";
}

inline bool IsBuiltinBlueNoise(const std::string &path, bool readOnly)
{
    if (!readOnly)
        return false;
    const auto file = ToFsPath(path);
    return file.filename() == "sh_blue_noise.png" && file.parent_path().filename() == "noise" &&
           file.parent_path().parent_path().filename() == "textures";
}

inline void ApplyBuiltinTextureMetadata(InxResourceMeta &metadata, const std::string &path, bool readOnly)
{
    ApplyBuiltinSceneIconMetadata(metadata, path, readOnly);
    if (IsBuiltinBlueNoise(path, readOnly)) {
        metadata.AddMetadata("texture_type", std::string("data"));
        metadata.AddMetadata("texture_compression", std::string("none"));
        metadata.AddMetadata("generate_mipmaps", false);
        metadata.AddMetadata("srgb", false);
    }
}

inline bool HasCurrentBuiltinTextureMetadata(const InxResourceMeta &metadata, const std::string &path, bool readOnly)
{
    if (IsBuiltinBlueNoise(path, readOnly))
        return metadata.HasKey("texture_type") && metadata.GetDataAs<std::string>("texture_type") == "data" &&
               metadata.HasKey("texture_compression") &&
               metadata.GetDataAs<std::string>("texture_compression") == "none" &&
               metadata.HasKey("generate_mipmaps") && !metadata.GetDataAs<bool>("generate_mipmaps") &&
               metadata.HasKey("srgb") && !metadata.GetDataAs<bool>("srgb");
    return HasCurrentBuiltinSceneIconMetadata(metadata, path, readOnly);
}

} // namespace infernux
