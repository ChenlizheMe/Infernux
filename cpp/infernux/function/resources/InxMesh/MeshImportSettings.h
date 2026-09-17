#pragma once

#include <function/resources/InxResource/InxResourceMeta.h>

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace infernux
{
/// One declaration for source import, metadata defaults and authoring clients.
/// This is the current Model section; later sections extend this contract.
struct MeshImportSettings
{
    float scaleFactor = 1.0f;
    bool generateNormals = true;
    bool generateTangents = true;
    bool flipUVs = true;
    bool swapUVChannels = false;
    bool optimizeMesh = true;
    bool weldVertices = true;

    struct Flag
    {
        const char *name;
        bool MeshImportSettings::*member;
    };
    inline static constexpr std::array Flags = {
        Flag{"generate_normals", &MeshImportSettings::generateNormals},
        Flag{"generate_tangents", &MeshImportSettings::generateTangents},
        Flag{"flip_uvs", &MeshImportSettings::flipUVs},
        Flag{"swap_uv_channels", &MeshImportSettings::swapUVChannels},
        Flag{"weld_vertices", &MeshImportSettings::weldVertices},
        Flag{"optimize_mesh", &MeshImportSettings::optimizeMesh},
    };

    static void RequireScale(float scale)
    {
        if (!std::isfinite(scale) || scale <= 0.0f)
            throw std::invalid_argument("model scale_factor must be finite and positive");
    }

    static MeshImportSettings Read(const InxResourceMeta &metadata)
    {
        MeshImportSettings settings;
        if (metadata.HasKey("scale_factor"))
            settings.scaleFactor = metadata.GetDataAs<float>("scale_factor");
        RequireScale(settings.scaleFactor);
        for (const auto &flag : Flags)
            if (metadata.HasKey(flag.name))
                settings.*(flag.member) = metadata.GetDataAs<bool>(flag.name);
        return settings;
    }

    static void EnsureDefaults(InxResourceMeta &metadata)
    {
        const MeshImportSettings defaults;
        if (!metadata.HasKey("scale_factor"))
            metadata.AddMetadata("scale_factor", defaults.scaleFactor);
        for (const auto &flag : Flags)
            if (!metadata.HasKey(flag.name))
                metadata.AddMetadata(flag.name, defaults.*(flag.member));
    }

    static void ApplyPatch(InxResourceMeta &metadata, const nlohmann::json &patch)
    {
        if (!patch.is_object())
            throw std::invalid_argument("model import settings require an object");
        // Validate the entire authoring request before modifying its candidate.
        for (const auto &[key, value] : patch.items()) {
            if (key == "scale_factor") {
                if (!value.is_number())
                    throw std::invalid_argument("model scale_factor must be a number");
                RequireScale(value.get<float>());
                continue;
            }
            bool known = false;
            for (const auto &flag : Flags)
                known |= key == flag.name;
            if (!known)
                throw std::invalid_argument("unknown model import setting: " + key);
            if (!value.is_boolean())
                throw std::invalid_argument("model import flags must be booleans");
        }
        for (const auto &[key, value] : patch.items()) {
            if (key == "scale_factor")
                metadata.AddMetadata(key, value.get<float>());
            else
                metadata.AddMetadata(key, value.get<bool>());
        }
    }

    static nlohmann::json Schema()
    {
        const MeshImportSettings defaults;
        auto fields = nlohmann::json::array({{
            {"name", "scale_factor"},
            {"type", "float"},
            {"default", defaults.scaleFactor},
            {"page", "model"},
            {"label", "asset.scale_factor"},
            {"minimum_exclusive", 0.0},
            {"display_range", {0.0001, 1000.0}},
            {"step", 0.001},
        }});
        for (const auto &flag : Flags)
            fields.push_back({{"name", flag.name},
                              {"type", "bool"},
                              {"default", defaults.*(flag.member)},
                              {"page", "model"},
                              {"label", std::string("asset.") + flag.name},
                              {"legacy_optional", flag.member == &MeshImportSettings::weldVertices}});
        return {{"version", 1}, {"fields", std::move(fields)}};
    }
};
} // namespace infernux
