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
/// Model, Rig, Animation and Materials share this import contract.
struct MeshImportSettings
{
    float scaleFactor = 1.0f;
    float normalSmoothingAngle = 175.0f;
    float minBoneWeight = 0.0f;
    int maxBonesPerVertex = 4;
    bool generateNormals = true;
    bool generateTangents = true;
    bool flipUVs = true;
    bool swapUVChannels = false;
    bool optimizeMesh = true;
    bool weldVertices = true;
    bool importAnimations = true;
    std::string rigType = "generic";
    std::string materialImportMode = "description";
    nlohmann::json materialRemaps = nlohmann::json::object();

    static void RequireMaterialRemaps(const nlohmann::json &value)
    {
        if (!value.is_object())
            throw std::invalid_argument("model material_remaps must be an object");
        for (const auto &[source, guid] : value.items())
            if (source.rfind("material/", 0) != 0 || source.size() == 9 || !guid.is_string() ||
                guid.get_ref<const std::string &>().empty())
                throw std::invalid_argument("model material_remaps require source material identifiers and GUIDs");
    }

    static void WriteMaterialRemaps(InxResourceMeta &metadata, const nlohmann::json &value)
    {
        auto document = metadata.SerializeDocument();
        document["metadata"]["material_remaps"] = {{"type", "json_object"}, {"value", value}};
        metadata.DeserializeDocument(document);
    }

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
        const char *page = "model";
    };
    inline static constexpr std::array Scalars = {
        Scalar{"scale_factor", &MeshImportSettings::scaleFactor, 0.0f, std::numeric_limits<float>::max(), true, 0.0001f,
               1000.0f, 0.001f, false},
        Scalar{"normal_smoothing_angle", &MeshImportSettings::normalSmoothingAngle, 0.0f, 175.0f, false, 0.0f, 175.0f,
               1.0f, true},
        Scalar{"min_bone_weight", &MeshImportSettings::minBoneWeight, 0.0f, 1.0f, false, 0.0f, 1.0f,
               0.001f, true, "rig"},
    };

    struct Flag
    {
        const char *name;
        bool MeshImportSettings::*member;
        const char *page = "model";
        bool legacyOptional = false;
    };
    inline static constexpr std::array Flags = {
        Flag{"generate_normals", &MeshImportSettings::generateNormals},
        Flag{"generate_tangents", &MeshImportSettings::generateTangents},
        Flag{"flip_uvs", &MeshImportSettings::flipUVs},
        Flag{"swap_uv_channels", &MeshImportSettings::swapUVChannels},
        Flag{"weld_vertices", &MeshImportSettings::weldVertices, "model", true},
        Flag{"optimize_mesh", &MeshImportSettings::optimizeMesh},
        Flag{"import_animations", &MeshImportSettings::importAnimations, "animation", true},
    };

    static void RequireRigType(const nlohmann::json &value)
    {
        if (!value.is_string() || (value != "none" && value != "generic"))
            throw std::invalid_argument("model rig_type must be none or generic");
    }

    static void RequireMaxBones(const nlohmann::json &value)
    {
        if (!value.is_number_integer() || value < 1 || value > 4)
            throw std::invalid_argument("model max_bones_per_vertex must be an integer between 1 and 4");
    }

    static void RequireMaterialImportMode(const nlohmann::json &value)
    {
        if (!value.is_string() || (value != "none" && value != "description"))
            throw std::invalid_argument("model material_import_mode must be none or description");
    }

    static void RequireScalar(const Scalar &field, float value)
    {
        if (!std::isfinite(value) || value > field.maximum || value < field.minimum ||
            (field.exclusiveMinimum && value == field.minimum))
            throw std::invalid_argument(std::string("invalid model ") + field.name + ": outside its finite range");
    }

    static MeshImportSettings Read(const InxResourceMeta &metadata)
    {
        MeshImportSettings settings;
        if (metadata.HasKey("max_bones_per_vertex"))
            settings.maxBonesPerVertex = metadata.GetDataAs<int>("max_bones_per_vertex");
        RequireMaxBones(settings.maxBonesPerVertex);
        for (const auto &field : Scalars) {
            if (metadata.HasKey(field.name))
                settings.*(field.member) = metadata.GetDataAs<float>(field.name);
            RequireScalar(field, settings.*(field.member));
        }
        for (const auto &flag : Flags)
            if (metadata.HasKey(flag.name))
                settings.*(flag.member) = metadata.GetDataAs<bool>(flag.name);
        if (metadata.HasKey("material_remaps")) {
            const auto &entry = metadata.GetMetadata().at("material_remaps");
            if (entry.first != "json_object")
                throw std::invalid_argument("model material_remaps metadata must use json_object");
            settings.materialRemaps = nlohmann::json::parse(std::any_cast<const std::string &>(entry.second));
        }
        RequireMaterialRemaps(settings.materialRemaps);
        if (metadata.HasKey("rig_type"))
            settings.rigType = metadata.GetDataAs<std::string>("rig_type");
        RequireRigType(settings.rigType);
        if (metadata.HasKey("material_import_mode"))
            settings.materialImportMode = metadata.GetDataAs<std::string>("material_import_mode");
        RequireMaterialImportMode(settings.materialImportMode);
        return settings;
    }

    static void EnsureDefaults(InxResourceMeta &metadata)
    {
        const MeshImportSettings defaults;
        if (!metadata.HasKey("max_bones_per_vertex"))
            metadata.AddMetadata("max_bones_per_vertex", defaults.maxBonesPerVertex);
        for (const auto &field : Scalars)
            if (!metadata.HasKey(field.name))
                metadata.AddMetadata(field.name, defaults.*(field.member));
        for (const auto &flag : Flags)
            if (!metadata.HasKey(flag.name))
                metadata.AddMetadata(flag.name, defaults.*(flag.member));
        if (!metadata.HasKey("material_remaps"))
            WriteMaterialRemaps(metadata, defaults.materialRemaps);
        if (!metadata.HasKey("rig_type"))
            metadata.AddMetadata("rig_type", defaults.rigType);
        if (!metadata.HasKey("material_import_mode"))
            metadata.AddMetadata("material_import_mode", defaults.materialImportMode);
    }

    static void ApplyPatch(InxResourceMeta &metadata, const nlohmann::json &patch)
    {
        if (!patch.is_object())
            throw std::invalid_argument("model import settings require an object");
        // Validate the entire authoring request before modifying its candidate.
        for (const auto &[key, value] : patch.items()) {
            if (key == "max_bones_per_vertex") {
                RequireMaxBones(value);
                continue;
            }
            if (key == "material_import_mode") {
                RequireMaterialImportMode(value);
                continue;
            }
            if (key == "rig_type") {
                RequireRigType(value);
                continue;
            }
            if (key == "material_remaps") {
                RequireMaterialRemaps(value);
                continue;
            }
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
            if (key == "material_remaps")
                WriteMaterialRemaps(metadata, value);
            else if (key == "max_bones_per_vertex")
                metadata.AddMetadata(key, value.get<int>());
            else if (key == "rig_type" || key == "material_import_mode")
                metadata.AddMetadata(key, value.get<std::string>());
            else if (value.is_boolean())
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
                              {"page", field.page},
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
                              {"page", flag.page},
                              {"label", std::string("asset.") + flag.name},
                              {"legacy_optional", flag.legacyOptional}});
        fields.push_back({{"name", "max_bones_per_vertex"}, {"type", "int"},
                          {"default", defaults.maxBonesPerVertex}, {"minimum", 1}, {"maximum", 4},
                          {"display_range", {1, 4}}, {"step", 1}, {"page", "rig"},
                          {"label", "asset.max_bones_per_vertex"}, {"legacy_optional", true}});
        fields.push_back({{"name", "rig_type"},
                          {"type", "enum"},
                          {"default", defaults.rigType},
                          {"choices",
                           {{{"value", "none"}, {"label", "asset.rig_none"}},
                            {{"value", "generic"}, {"label", "asset.rig_generic"}}}},
                          {"page", "rig"},
                          {"label", "asset.rig_type"},
                          {"legacy_optional", true}});
        fields.push_back({{"name", "material_import_mode"},
                          {"type", "enum"},
                          {"default", defaults.materialImportMode},
                          {"choices",
                           {{{"value", "none"}, {"label", "asset.material_import_none"}},
                            {{"value", "description"}, {"label", "asset.material_import_description"}}}},
                          {"page", "materials"},
                          {"label", "asset.material_import_mode"},
                          {"legacy_optional", true}});
        fields.push_back({{"name", "material_remaps"},
                          {"type", "material_remaps"},
                          {"default", nlohmann::json::object()},
                          {"page", "materials"},
                          {"label", "asset.material_remaps"},
                          {"legacy_optional", true}});
        return {{"version", 1}, {"fields", std::move(fields)}};
    }
};
} // namespace infernux
