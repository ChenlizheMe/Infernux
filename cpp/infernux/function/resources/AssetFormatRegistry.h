#pragma once

// One native source of truth for source formats that the current runtime can
// actually decode/import.  Keeping this list next to the importer prevents
// the AssetDatabase and Project panel from slowly growing different opinions
// about which files are real assets.

#include <algorithm>
#include <array>
#include <cctype>
#include <string>
#include <string_view>
#include <vector>

namespace infernux::asset_formats
{

struct ShaderSourceKind
{
    std::string_view suffix;
    std::string_view stage;
    std::string_view icon;
    std::string_view dragPayload;
};

// Author files are assets, never host Python modules. All discovery, import,
// UI and cook consumers use this table before looking at the final suffix.
inline constexpr std::array<ShaderSourceKind, 5> kShaderSourceKinds = {{
    {".shadingmodel.py", "shadingmodel", "shadingmodel", "SHADING_MODEL_FILE"},
    {".shaderlib.py", "library", "shader_glsl", "SHADER_LIBRARY_FILE"},
    {".compute.py", "compute", "shader_hlsl", "COMPUTE_SHADER_FILE"},
    {".vert.py", "vertex", "shader_vert", "SHADER_FILE"},
    {".frag.py", "fragment", "shader_frag", "SHADER_FILE"},
}};

inline const ShaderSourceKind *FindShaderSourceKind(std::string_view path)
{
    for (const auto &kind : kShaderSourceKinds) {
        if (path.size() >= kind.suffix.size() &&
            std::equal(kind.suffix.begin(), kind.suffix.end(), path.end() - kind.suffix.size(),
                       [](char suffix, unsigned char value) { return suffix == std::tolower(value); }))
            return &kind;
    }
    return nullptr;
}

inline std::string SourceExtension(std::string_view path)
{
    if (const auto *kind = FindShaderSourceKind(path))
        return std::string(kind->suffix);
    const auto dot = path.find_last_of('.');
    const auto slash = path.find_last_of("/\\");
    if (dot == std::string_view::npos || (slash != std::string_view::npos && dot < slash))
        return {};
    std::string suffix(path.substr(dot));
    std::transform(suffix.begin(), suffix.end(), suffix.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return suffix;
}

inline constexpr std::array<std::string_view, 16> kTextureExtensions = {
    ".png", ".jpg", ".jpeg", ".jpe", ".bmp", ".tga",       ".gif",    ".psd",
    ".hdr", ".pic", ".pnm",  ".pgm", ".ppm", ".inxvfield", ".inxsdf", ".svg",
};

// Native .inxmesh sources use MeshArtifact; the other formats use Assimp.
// The list intentionally contains
// formats implemented by the vendored Assimp build, not merely file suffixes
// that a UI could display.
inline constexpr std::array<std::string_view, 36> kMeshExtensions = {
    ".fbx", ".obj", ".gltf", ".glb",     ".dae", ".3ds", ".ply",  ".stl", ".x",       ".b3d",  ".ase", ".blend",
    ".bvh", ".cob", ".c4d",  ".csm",     ".dxf", ".hmp", ".ifc",  ".iqm", ".irrmesh", ".lwo",  ".lws", ".m3d",
    ".md2", ".md3", ".md4",  ".md5mesh", ".mdc", ".mmd", ".ms3d", ".nff", ".off",     ".ogex", ".x3d", ".inxmesh",
};

// WAV/MP3/FLAC use dr_libs; OGG/Vorbis uses stb_vorbis. Resident and streaming
// playback share the same file-backed decoders.
// Every consumer (database, editor icons and drag payloads) derives from this
// list so a visible AudioClip is always a format the runtime can decode.
inline constexpr std::array<std::string_view, 4> kAudioExtensions = {".wav", ".ogg", ".mp3", ".flac"};

// Opaque containers and files. Their metadata does not interpret the payload as text.
inline constexpr std::array<std::string_view, 22> kBinaryExtensions = {
    ".exe", ".dll", ".so",  ".dylib", ".bin", ".dat", ".mp4", ".avi",  ".mkv",   ".mov", ".zip",
    ".rar", ".7z",  ".tar", ".gz",    ".pdf", ".ttf", ".otf", ".woff", ".woff2", ".eot", ".inm",
};

template <size_t N> inline bool Contains(const std::array<std::string_view, N> &extensions, std::string_view ext)
{
    for (const auto candidate : extensions) {
        if (candidate == ext)
            return true;
    }
    return false;
}

template <size_t N> inline std::vector<std::string> ToVector(const std::array<std::string_view, N> &extensions)
{
    std::vector<std::string> result;
    result.reserve(N);
    for (const auto ext : extensions)
        result.emplace_back(ext);
    return result;
}

} // namespace infernux::asset_formats
