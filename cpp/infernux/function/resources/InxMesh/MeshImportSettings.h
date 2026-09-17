#pragma once

#include <function/resources/InxResource/InxResourceMeta.h>

#include <array>
#include <cmath>
#include <limits>
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
    float normalSmoothingAngle = 175.0f;
    bool generateNormals = true;
    bool generateTangents = true;
    bool flipUVs = true;
    bool swapUVChannels = false;
    bool optimizeMesh = true;
    bool weldVertices = true;

    struct Scalar
    {
        const char *name;
        float MeshImportSettings::*member;
        float minimum;
        float maximum;
        bool exclusiveMinimum;
        float displayMinimum;
        float displayMaximum;
        float step;
        bool legacyOptional;
    };
    inline static constexpr std::array Scalars = {
        Scalar{"scale_factor", &MeshImportSettings::scaleFactor, 0.0f, std::numeric_limits<float>::max(), true, 0.0001f,
               1000.0f, 0.001f, false},
        Scalar{"normal_smoothing_angle", &MeshImportSettings::normalSmoothingAngle, 0.0f, 175.0f, false, 0.0f, 175.0f,
               1.0f, true},
    };

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

    static void RequireScalar(const Scalar &field, float value)
    {
        if (!std::isfinite(value) || value > field.maximum || value < field.minimum ||
            (field.exclusiveMinimum && value == field.minimum))
            throw std::invalid_argument(std::string("invalid model ") + field.name + ": outside its finite range");
    }

    static MeshImportSettings Read(const InxResourceMeta &metadata)
    {
        MeshImportSettings settings;
        for (const auto &field : Scalars) {
            if (metadata.HasKey(field.name))
                settings.*(field.member) = metadata.GetDataAs<float>(field.name);
            RequireScalar(field, settings.*(field.member));
        }
        for (const auto &flag : Flags)
            if (metadata.HasKey(flag.name))
                settings.*(flag.member) = metadata.GetDataAs<bool>(flag.name);
        return settings;
    }

    static void EnsureDefaults(InxResourceMeta &metadata)
    {
        const MeshImportSettings defaults;
        for (const auto &field : Scalars)
            if (!metadata.HasKey(field.name))
                metadata.AddMetadata(field.name, defaults.*(field.member));
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
            bool scalar = false;
            for (const auto &field : Scalars) {
                if (key != field.name)
                    continue;
                if (!value.is_number())
                    throw std::invalid_argument("model " + key + " must be a number");
                RequireScalar(field, value.get<float>());
                scalar = true;
                break;
            }
            if (scalar)
                continue;
            bool known = false;
            for (const auto &flag : Flags)
                known |= key == flag.name;
            if (!known)
                throw std::invalid_argument("unknown model import setting: " + key);
            if (!value.is_boolean())
                throw std::invalid_argument("model import flags must be booleans");
        }
        for (const auto &[key, value] : patch.items()) {
            if (value.is_boolean())
                metadata.AddMetadata(key, value.get<bool>());
            else
                metadata.AddMetadata(key, value.get<float>());
        }
    }

    static nlohmann::json Schema()
    {
        const MeshImportSettings defaults;
        auto fields = nlohmann::json::array();
        for (const auto &field : Scalars)
            fields.push_back({{"name", field.name},
                              {"type", "float"},
                              {"default", defaults.*(field.member)},
                              {"page", "model"},
                              {"label", std::string("asset.") + field.name},
                              {field.exclusiveMinimum ? "minimum_exclusive" : "minimum", field.minimum},
                              {"maximum", field.maximum},
                              {"display_range", {field.displayMinimum, field.displayMaximum}},
                              {"step", field.step},
                              {"legacy_optional", field.legacyOptional}});
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
