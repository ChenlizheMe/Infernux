#pragma once

#include <function/resources/InxResource/InxResourceMeta.h>

#include <array>
#include <cmath>
#include <limits>
#include <set>
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
    std::string normalMode = "import";
    std::string tangentMode = "import";
    std::string normalWeighting = "unweighted";
    std::string tangentAlgorithm = "mikktspace";
    bool flipUVs = true;
    bool swapUVChannels = false;
    bool optimizeMesh = true;
    bool weldVertices = true;
    bool generateColliders = false;
    bool importAnimations = true;
    bool customAnimationClips = false;
    nlohmann::json animationClips = nlohmann::json::array();
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

    static void RequireAnimationClips(const nlohmann::json &value)
    {
        if (!value.is_array())
            throw std::invalid_argument("model animation_clips must be an array");
        std::set<std::string> ids, names;
        for (const auto &clip : value) {
            if (!clip.is_object() || clip.size() != 5 || !clip.contains("id") || !clip.contains("name") ||
                !clip.contains("source_take") || !clip.contains("start") || !clip.contains("end"))
                throw std::invalid_argument("animation_clips require id, name, source_take, start and end");
            for (const auto *key : {"id", "name", "source_take"})
                if (!clip.at(key).is_string() || clip.at(key).get_ref<const std::string &>().empty())
                    throw std::invalid_argument(std::string("animation clip requires a non-empty ") + key);
            const auto id = clip.at("id").get<std::string>();
            if (id.size() != 32 || id.find_first_not_of("0123456789abcdef") != std::string::npos ||
                !ids.insert(id).second || !names.insert(clip.at("name").get<std::string>()).second)
                throw std::invalid_argument("animation clips require unique lowercase GUID ids and unique names");
            if (!clip.at("start").is_number() || !clip.at("end").is_number())
                throw std::invalid_argument("animation clip start/end must be seconds");
            const double start = clip.at("start"), end = clip.at("end");
            if (!std::isfinite(start) || !std::isfinite(end) || start < 0.0 || end <= start)
                throw std::invalid_argument("animation clip requires finite 0 <= start < end");
        }
    }

    static void WriteAnimationClips(InxResourceMeta &metadata, const nlohmann::json &value)
    {
        auto document = metadata.SerializeDocument();
        document["metadata"]["animation_clips"] = {{"type", "json_array"}, {"value", value}};
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
        const char *page;
    };
    inline static constexpr std::array Scalars = {
        Scalar{"scale_factor", &MeshImportSettings::scaleFactor, 0.0f, std::numeric_limits<float>::max(), true, 0.0001f,
               1000.0f, 0.001f, false, "model"},
        Scalar{"normal_smoothing_angle", &MeshImportSettings::normalSmoothingAngle, 0.0f, 175.0f, false, 0.0f, 175.0f,
               1.0f, true, "model"},
        Scalar{"min_bone_weight", &MeshImportSettings::minBoneWeight, 0.0f, 1.0f, false, 0.0f, 1.0f,
               0.001f, true, "rig"},
    };

    struct Flag
    {
        const char *name;
        bool MeshImportSettings::*member;
        const char *page;
        bool legacyOptional;
    };
    inline static constexpr std::array Flags = {
        Flag{"flip_uvs", &MeshImportSettings::flipUVs, "model", false},
        Flag{"swap_uv_channels", &MeshImportSettings::swapUVChannels, "model", false},
        Flag{"weld_vertices", &MeshImportSettings::weldVertices, "model", true},
        Flag{"generate_colliders", &MeshImportSettings::generateColliders, "model", true},
        Flag{"optimize_mesh", &MeshImportSettings::optimizeMesh, "model", false},
        Flag{"import_animations", &MeshImportSettings::importAnimations, "animation", true},
        Flag{"custom_animation_clips", &MeshImportSettings::customAnimationClips, "animation", true},
    };

    struct BasisMode
    {
        const char *name;
        const char *legacyFlag;
        std::string MeshImportSettings::*member;
    };
    inline static constexpr std::array BasisModes = {
        BasisMode{"normal_mode", "generate_normals", &MeshImportSettings::normalMode},
        BasisMode{"tangent_mode", "generate_tangents", &MeshImportSettings::tangentMode},
    };

    static void RequireBasisMode(const nlohmann::json &value)
    {
        if (!value.is_string() ||
            (value != "import" && value != "calculate" && value != "none" && value != "source_only"))
            throw std::invalid_argument("model basis mode must be import, calculate, none or source_only");
    }

    static void RequireNormalWeighting(const nlohmann::json &value)
    {
        if (!value.is_string() || (value != "unweighted" && value != "area" && value != "angle" && value != "area_angle"))
            throw std::invalid_argument("model normal_weighting must be unweighted, area, angle or area_angle");
    }

    static void RequireTangentAlgorithm(const nlohmann::json &value)
    {
        if (!value.is_string() || (value != "mikktspace" && value != "assimp"))
            throw std::invalid_argument("model tangent_algorithm must be mikktspace or assimp");
    }

    static std::string ReadTangentAlgorithm(const InxResourceMeta &metadata)
    {
        if (metadata.HasKey("tangent_algorithm")) {
            auto value = metadata.GetDataAs<std::string>("tangent_algorithm");
            RequireTangentAlgorithm(value);
            return value;
        }
        // Existing projects used Assimp. New sources use MikkTSpace; upgrading
        // an old sidecar must not change its normal-map bake convention.
        return metadata.HasKey("tangent_mode") || metadata.HasKey("generate_tangents") ? "assimp" : "mikktspace";
    }

    static std::string ReadBasisMode(const InxResourceMeta &metadata, const BasisMode &field)
    {
        if (metadata.HasKey(field.name)) {
            auto value = metadata.GetDataAs<std::string>(field.name);
            RequireBasisMode(value);
            return value;
        }
        // Old unchecked Generate Missing preserved authored data; it did not
        // remove it. Keep that policy when upgrading old project metadata.
        return metadata.HasKey(field.legacyFlag) && !metadata.GetDataAs<bool>(field.legacyFlag) ? "source_only"
                                                                                            : "import";
    }

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
        settings.tangentAlgorithm = ReadTangentAlgorithm(metadata);
        if (metadata.HasKey("normal_weighting"))
            settings.normalWeighting = metadata.GetDataAs<std::string>("normal_weighting");
        RequireNormalWeighting(settings.normalWeighting);
        for (const auto &field : BasisModes)
            settings.*(field.member) = ReadBasisMode(metadata, field);
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
        if (metadata.HasKey("animation_clips")) {
            const auto &entry = metadata.GetMetadata().at("animation_clips");
            if (entry.first != "json_array")
                throw std::invalid_argument("model animation_clips metadata must use json_array");
            settings.animationClips = nlohmann::json::parse(std::any_cast<const std::string &>(entry.second));
        }
        RequireAnimationClips(settings.animationClips);
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
        metadata.AddMetadata("tangent_algorithm", ReadTangentAlgorithm(metadata));
        if (!metadata.HasKey("normal_weighting"))
            metadata.AddMetadata("normal_weighting", defaults.normalWeighting);
        for (const auto &field : BasisModes)
            metadata.AddMetadata(field.name, ReadBasisMode(metadata, field));
        if (metadata.HasKey("generate_normals") || metadata.HasKey("generate_tangents")) {
            auto document = metadata.SerializeDocument();
            for (const auto &field : BasisModes)
                document["metadata"].erase(field.legacyFlag);
            metadata.DeserializeDocument(document);
        }
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
        if (!metadata.HasKey("animation_clips"))
            WriteAnimationClips(metadata, defaults.animationClips);
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
            if (key == "normal_weighting") {
                RequireNormalWeighting(value);
                continue;
            }
            if (key == "tangent_algorithm") {
                RequireTangentAlgorithm(value);
                continue;
            }
            if (key == "normal_mode" || key == "tangent_mode") {
                RequireBasisMode(value);
                continue;
            }
            if (key == "animation_clips") {
                RequireAnimationClips(value);
                continue;
            }
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
            if (key == "animation_clips")
                WriteAnimationClips(metadata, value);
            else if (key == "material_remaps")
                WriteMaterialRemaps(metadata, value);
            else if (key == "max_bones_per_vertex")
                metadata.AddMetadata(key, value.get<int>());
            else if (key == "rig_type" || key == "material_import_mode" || key == "normal_mode" ||
                     key == "tangent_mode" || key == "normal_weighting" || key == "tangent_algorithm")
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
        for (const auto &field : BasisModes)
            fields.push_back({{"name", field.name}, {"type", "enum"}, {"default", defaults.*(field.member)},
                              {"page", "model"}, {"label", std::string("asset.") + field.name},
                              {"legacy_optional", true}, {"legacy_flag", field.legacyFlag},
                              {"choices", {{{"value", "import"}, {"label", "asset.basis_import"}},
                                           {{"value", "calculate"}, {"label", "asset.basis_calculate"}},
                                           {{"value", "none"}, {"label", "asset.basis_none"}},
                                           {{"value", "source_only"}, {"label", "asset.basis_source_only"}}}}});
        fields.push_back({{"name", "normal_weighting"}, {"type", "enum"}, {"default", defaults.normalWeighting},
                          {"page", "model"}, {"label", "asset.normal_weighting"}, {"legacy_optional", true},
                          {"choices", {{{"value", "unweighted"}, {"label", "asset.normal_unweighted"}},
                                       {{"value", "area"}, {"label", "asset.normal_area"}},
                                       {{"value", "angle"}, {"label", "asset.normal_angle"}},
                                       {{"value", "area_angle"}, {"label", "asset.normal_area_angle"}}}}});
        fields.push_back({{"name", "tangent_algorithm"}, {"type", "enum"}, {"default", defaults.tangentAlgorithm},
                          {"page", "model"}, {"label", "asset.tangent_algorithm"}, {"legacy_optional", true},
                          {"legacy_default", "assimp"},
                          {"choices", {{{"value", "mikktspace"}, {"label", "asset.tangent_mikktspace"}},
                                       {{"value", "assimp"}, {"label", "asset.tangent_assimp"}}}}});
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
        fields.push_back({{"name", "animation_clips"}, {"type", "animation_clips"},
                          {"default", nlohmann::json::array()}, {"page", "animation"},
                          {"label", "asset.animation_clips"}, {"legacy_optional", true}});
        return {{"version", 1}, {"fields", std::move(fields)}};
    }
};
} // namespace infernux
