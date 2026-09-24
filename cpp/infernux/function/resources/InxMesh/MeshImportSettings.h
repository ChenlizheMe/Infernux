#pragma once

#include <function/resources/InxResource/InxResourceMeta.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
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
    float animationSampleRate = 0.0f;
    float animationPositionError = 0.0f;
    float animationRotationError = 0.0f;
    float animationScaleError = 0.0f;
    int maxBonesPerVertex = 4;
    std::string normalMode = "import";
    std::string tangentMode = "import";
    std::string normalWeighting = "unweighted";
    // angle: derive smoothing from the authored threshold. source: preserve
    // source-authored hard/soft discontinuities and reject missing source data.
    std::string normalSmoothingSource = "angle";
    std::string tangentAlgorithm = "mikktspace";
    bool flipUVs = true;
    bool swapUVChannels = false;
    bool optimizeMesh = true;
    bool weldVertices = true;
    bool sortHierarchyByName = false;
    bool importVisibility = true;
    bool convertUnits = true;
    // Unity-compatible authoring switches. Both default to false until an
    // author explicitly requests hierarchy baking or CPU-readable geometry.
    bool bakeAxisConversion = false;
    bool isReadable = false;
    bool generateColliders = false;
    bool importBlendShapes = true;
    bool importAnimations = true;
    bool animationLoopTime = true;
    bool animationApplyRootMotion = false;
    bool customAnimationClips = false;
    nlohmann::json animationClips = nlohmann::json::array();
    nlohmann::json animationClipExtras = nlohmann::json::array();
    std::string rigType = "generic";
    std::string skeletonDefinitionMode = "create";
    std::string rigRootNode;
    std::string skeletonDefinitionGuid;
    std::string skeletonDefinitionId = "skeleton";
    bool optimizeBoneHierarchy = false;
    nlohmann::json exposedBones = nlohmann::json::array();
    // Sparse author overrides. The artifact carries the fully resolved map;
    // an absent slot is automatically mapped on every Apply.
    nlohmann::json humanoidBoneOverrides = nlohmann::json::object();
    std::string animationReferencePose = "bind_pose";
    std::string materialImportMode = "description";
    std::string meshCompression = "off";
    std::string indexFormat = "auto";
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

    static void RequireAnimationClipExtras(const nlohmann::json &value)
    {
        if (!value.is_array())
            throw std::invalid_argument("model animation_clip_extras must be an array");
        std::set<std::string> clipIds;
        for (const auto &extra : value) {
            if (!extra.is_object() || extra.size() != 4 || !extra.contains("clip_id") || !extra.contains("curves") ||
                !extra.contains("events") || !extra.contains("bone_mask"))
                throw std::invalid_argument("animation_clip_extras require clip_id, curves, events and bone_mask");
            if (!extra.at("clip_id").is_string() || extra.at("clip_id").get_ref<const std::string &>().empty() ||
                !clipIds.insert(extra.at("clip_id").get<std::string>()).second)
                throw std::invalid_argument("animation_clip_extras require unique non-empty clip ids");
            if (!extra.at("curves").is_array() || !extra.at("events").is_array() || !extra.at("bone_mask").is_array())
                throw std::invalid_argument("animation clip curves, events and bone_mask must be arrays");

            std::set<std::string> curveNames;
            for (const auto &curve : extra.at("curves")) {
                if (!curve.is_object() || curve.size() != 2 || !curve.contains("name") || !curve.contains("keys") ||
                    !curve.at("name").is_string() || curve.at("name").get_ref<const std::string &>().empty() ||
                    !curve.at("keys").is_array() || !curveNames.insert(curve.at("name").get<std::string>()).second)
                    throw std::invalid_argument("animation curves require unique non-empty names and key arrays");
                double previous = -1.0;
                for (const auto &key : curve.at("keys")) {
                    if (!key.is_object() || key.size() != 2 || !key.contains("time_normalized") ||
                        !key.contains("value") || !key.at("time_normalized").is_number() ||
                        !key.at("value").is_number())
                        throw std::invalid_argument("animation curve keys require time_normalized and value");
                    const double time = key.at("time_normalized").get<double>();
                    const double sample = key.at("value").get<double>();
                    if (!std::isfinite(time) || !std::isfinite(sample) || time < 0.0 || time > 1.0 || time <= previous)
                        throw std::invalid_argument(
                            "animation curve key times must be finite, strictly increasing and in [0, 1]");
                    previous = time;
                }
            }

            double previousEvent = -1.0;
            for (const auto &event : extra.at("events")) {
                if (!event.is_object() || event.size() != 4 || !event.contains("time_normalized") ||
                    !event.contains("function") || !event.contains("string_arg") || !event.contains("number_arg") ||
                    !event.at("time_normalized").is_number() || !event.at("number_arg").is_number() ||
                    !event.at("function").is_string() || event.at("function").get_ref<const std::string &>().empty() ||
                    !event.at("string_arg").is_string())
                    throw std::invalid_argument("animation events require time, function and typed arguments");
                const double time = event.at("time_normalized").get<double>();
                const double number = event.at("number_arg").get<double>();
                if (!std::isfinite(time) || !std::isfinite(number) || time < 0.0 || time > 1.0 || time < previousEvent)
                    throw std::invalid_argument(
                        "animation events must be finite, ordered and have normalized times in [0, 1]");
                previousEvent = time;
            }

            std::set<std::string> bones;
            for (const auto &bone : extra.at("bone_mask"))
                if (!bone.is_string() || bone.get_ref<const std::string &>().empty() ||
                    !bones.insert(bone.get<std::string>()).second)
                    throw std::invalid_argument("animation bone masks require unique non-empty bone names");
        }
    }

    static void WriteAnimationClipExtras(InxResourceMeta &metadata, const nlohmann::json &value)
    {
        auto document = metadata.SerializeDocument();
        document["metadata"]["animation_clip_extras"] = {{"type", "json_array"}, {"value", value}};
        metadata.DeserializeDocument(document);
    }

    static void RequireRigNodeName(const nlohmann::json &value, const char *field)
    {
        if (!value.is_string() || value.get_ref<const std::string &>().size() > 1024)
            throw std::invalid_argument(std::string("model ") + field + " must be a bounded string");
    }

    static void RequireExposedBones(const nlohmann::json &value)
    {
        if (!value.is_array())
            throw std::invalid_argument("model exposed_bones must be an array");
        std::set<std::string> names;
        for (const auto &node : value)
            if (!node.is_string() || node.get_ref<const std::string &>().empty() ||
                node.get_ref<const std::string &>().size() > 1024 || !names.insert(node.get<std::string>()).second)
                throw std::invalid_argument("model exposed_bones require unique non-empty node identities");
    }

    static void WriteExposedBones(InxResourceMeta &metadata, const nlohmann::json &value)
    {
        auto document = metadata.SerializeDocument();
        document["metadata"]["exposed_bones"] = {{"type", "json_array"}, {"value", value}};
        metadata.DeserializeDocument(document);
    }

    inline static constexpr std::array HumanoidBoneNames = {
        "hips",           "spine",           "chest",           "upper_chest",     "neck",
        "head",           "left_shoulder",   "left_upper_arm",  "left_lower_arm",  "left_hand",
        "right_shoulder", "right_upper_arm", "right_lower_arm", "right_hand",      "left_upper_leg",
        "left_lower_leg", "left_foot",       "left_toes",       "right_upper_leg", "right_lower_leg",
        "right_foot",     "right_toes"};

    static bool IsHumanoidBoneName(std::string_view value)
    {
        return std::find(HumanoidBoneNames.begin(), HumanoidBoneNames.end(), value) != HumanoidBoneNames.end();
    }

    static void RequireHumanoidBoneOverrides(const nlohmann::json &value)
    {
        if (!value.is_object())
            throw std::invalid_argument("model humanoid_bone_overrides must be an object");
        std::set<std::string> nodes;
        for (const auto &[bone, node] : value.items()) {
            if (!IsHumanoidBoneName(bone) || !node.is_string() || node.get_ref<const std::string &>().empty() ||
                node.get_ref<const std::string &>().size() > 1024 || !nodes.insert(node.get<std::string>()).second)
                throw std::invalid_argument(
                    "model humanoid_bone_overrides require known slots and unique non-empty node identities");
        }
    }

    static void WriteHumanoidBoneOverrides(InxResourceMeta &metadata, const nlohmann::json &value)
    {
        auto document = metadata.SerializeDocument();
        document["metadata"]["humanoid_bone_overrides"] = {{"type", "json_object"}, {"value", value}};
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
        const char *page;
    };
    inline static constexpr std::array Scalars = {
        Scalar{"scale_factor", &MeshImportSettings::scaleFactor, 0.0f, std::numeric_limits<float>::max(), true, 0.0001f,
               1000.0f, 0.001f, "model"},
        Scalar{"normal_smoothing_angle", &MeshImportSettings::normalSmoothingAngle, 0.0f, 175.0f, false, 0.0f, 175.0f,
               1.0f, "model"},
        Scalar{"min_bone_weight", &MeshImportSettings::minBoneWeight, 0.0f, 1.0f, false, 0.0f, 1.0f, 0.001f, "rig"},
        Scalar{"animation_sample_rate", &MeshImportSettings::animationSampleRate, 0.0f, 240.0f, false, 0.0f, 120.0f,
               1.0f, "animation"},
        Scalar{"animation_position_error", &MeshImportSettings::animationPositionError, 0.0f, 1000000.0f, false, 0.0f,
               0.1f, 0.0001f, "animation"},
        Scalar{"animation_rotation_error", &MeshImportSettings::animationRotationError, 0.0f, 180.0f, false, 0.0f,
               10.0f, 0.01f, "animation"},
        Scalar{"animation_scale_error", &MeshImportSettings::animationScaleError, 0.0f, 1000000.0f, false, 0.0f, 0.1f,
               0.0001f, "animation"},
    };

    struct Flag
    {
        const char *name;
        bool MeshImportSettings::*member;
        const char *page;
    };
    inline static constexpr std::array Flags = {
        Flag{"flip_uvs", &MeshImportSettings::flipUVs, "model"},
        Flag{"swap_uv_channels", &MeshImportSettings::swapUVChannels, "model"},
        Flag{"weld_vertices", &MeshImportSettings::weldVertices, "model"},
        Flag{"sort_hierarchy_by_name", &MeshImportSettings::sortHierarchyByName, "model"},
        Flag{"import_visibility", &MeshImportSettings::importVisibility, "model"},
        Flag{"convert_units", &MeshImportSettings::convertUnits, "model"},
        Flag{"bake_axis_conversion", &MeshImportSettings::bakeAxisConversion, "model"},
        Flag{"is_readable", &MeshImportSettings::isReadable, "model"},
        Flag{"generate_colliders", &MeshImportSettings::generateColliders, "model"},
        Flag{"import_blend_shapes", &MeshImportSettings::importBlendShapes, "model"},
        Flag{"optimize_mesh", &MeshImportSettings::optimizeMesh, "model"},
        Flag{"import_animations", &MeshImportSettings::importAnimations, "animation"},
        Flag{"animation_loop_time", &MeshImportSettings::animationLoopTime, "animation"},
        Flag{"animation_apply_root_motion", &MeshImportSettings::animationApplyRootMotion, "animation"},
        Flag{"custom_animation_clips", &MeshImportSettings::customAnimationClips, "animation"},
        Flag{"optimize_bone_hierarchy", &MeshImportSettings::optimizeBoneHierarchy, "rig"},
    };

    struct BasisMode
    {
        const char *name;
        std::string MeshImportSettings::*member;
    };
    inline static constexpr std::array BasisModes = {
        BasisMode{"normal_mode", &MeshImportSettings::normalMode},
        BasisMode{"tangent_mode", &MeshImportSettings::tangentMode},
    };

    static void RequireBasisMode(const nlohmann::json &value)
    {
        if (!value.is_string() ||
            (value != "import" && value != "calculate" && value != "none" && value != "source_only"))
            throw std::invalid_argument("model basis mode must be import, calculate, none or source_only");
    }

    static void RequireNormalWeighting(const nlohmann::json &value)
    {
        if (!value.is_string() ||
            (value != "unweighted" && value != "area" && value != "angle" && value != "area_angle"))
            throw std::invalid_argument("model normal_weighting must be unweighted, area, angle or area_angle");
    }

    static void RequireNormalSmoothingSource(const nlohmann::json &value)
    {
        if (!value.is_string() || (value != "angle" && value != "source"))
            throw std::invalid_argument("model normal_smoothing_source must be angle or source");
    }

    static void RequireTangentAlgorithm(const nlohmann::json &value)
    {
        if (!value.is_string() || (value != "mikktspace" && value != "assimp"))
            throw std::invalid_argument("model tangent_algorithm must be mikktspace or assimp");
    }

    static std::string ReadTangentAlgorithm(const InxResourceMeta &metadata)
    {
        auto value = metadata.GetDataAs<std::string>("tangent_algorithm");
        RequireTangentAlgorithm(value);
        return value;
    }

    static std::string ReadBasisMode(const InxResourceMeta &metadata, const BasisMode &field)
    {
        auto value = metadata.GetDataAs<std::string>(field.name);
        RequireBasisMode(value);
        return value;
    }

    static void RequireRigType(const nlohmann::json &value)
    {
        if (!value.is_string() || (value != "none" && value != "generic" && value != "humanoid"))
            throw std::invalid_argument("model rig_type must be none, generic or humanoid");
    }

    static void RequireSkeletonDefinitionMode(const nlohmann::json &value)
    {
        if (!value.is_string() || (value != "create" && value != "copy"))
            throw std::invalid_argument("model skeleton_definition_mode must be create or copy");
    }

    static void RequireAnimationReferencePose(const nlohmann::json &value)
    {
        if (!value.is_string() || (value != "bind_pose" && value != "first_frame"))
            throw std::invalid_argument("model animation_reference_pose must be bind_pose or first_frame");
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

    static void RequireMeshCompression(const nlohmann::json &value)
    {
        if (!value.is_string() || (value != "off" && value != "low" && value != "medium" && value != "high"))
            throw std::invalid_argument("model mesh_compression must be off, low, medium or high");
    }

    static void RequireIndexFormat(const nlohmann::json &value)
    {
        if (!value.is_string() || (value != "auto" && value != "uint16" && value != "uint32"))
            throw std::invalid_argument("model index_format must be auto, uint16 or uint32");
    }

    static void RequireScalar(const Scalar &field, float value)
    {
        if (!std::isfinite(value) || value > field.maximum || value < field.minimum ||
            (field.exclusiveMinimum && value == field.minimum))
            throw std::invalid_argument(std::string("invalid model ") + field.name + ": outside its finite range");
    }

    static MeshImportSettings Read(const InxResourceMeta &metadata)
    {
        const auto require = [&metadata](const char *name) {
            if (!metadata.HasKey(name))
                throw std::invalid_argument(std::string("model import metadata is missing current field: ") + name);
        };
        const auto requireType = [&metadata](const char *name, const char *type) {
            const auto &entry = metadata.GetMetadata().at(name);
            if (entry.first != type)
                throw std::invalid_argument(std::string("model import metadata field '") + name +
                                            "' must use current type " + type);
        };
        for (const auto &field : Scalars)
            require(field.name);
        for (const auto &flag : Flags)
            require(flag.name);
        for (const auto &field : BasisModes)
            require(field.name);
        for (const char *name :
             {"normal_weighting", "normal_smoothing_source", "tangent_algorithm", "max_bones_per_vertex",
              "material_remaps", "animation_clips", "animation_clip_extras", "exposed_bones", "humanoid_bone_overrides",
              "rig_type", "skeleton_definition_mode", "rig_root_node", "skeleton_definition_guid",
              "skeleton_definition_id", "animation_reference_pose", "material_import_mode", "mesh_compression",
              "index_format"})
            require(name);

        for (const auto &field : Scalars)
            requireType(field.name, "float");
        for (const auto &flag : Flags)
            requireType(flag.name, "bool");
        for (const auto &field : BasisModes)
            requireType(field.name, "string");
        requireType("max_bones_per_vertex", "int");
        for (const char *name :
             {"normal_weighting", "normal_smoothing_source", "tangent_algorithm", "rig_type",
              "skeleton_definition_mode", "rig_root_node", "skeleton_definition_guid", "skeleton_definition_id",
              "animation_reference_pose", "material_import_mode", "mesh_compression", "index_format"})
            requireType(name, "string");

        MeshImportSettings settings;
        settings.tangentAlgorithm = ReadTangentAlgorithm(metadata);
        settings.normalWeighting = metadata.GetDataAs<std::string>("normal_weighting");
        RequireNormalWeighting(settings.normalWeighting);
        settings.normalSmoothingSource = metadata.GetDataAs<std::string>("normal_smoothing_source");
        RequireNormalSmoothingSource(settings.normalSmoothingSource);
        for (const auto &field : BasisModes)
            settings.*(field.member) = ReadBasisMode(metadata, field);
        settings.maxBonesPerVertex = metadata.GetDataAs<int>("max_bones_per_vertex");
        RequireMaxBones(settings.maxBonesPerVertex);
        for (const auto &field : Scalars) {
            settings.*(field.member) = metadata.GetDataAs<float>(field.name);
            RequireScalar(field, settings.*(field.member));
        }
        for (const auto &flag : Flags)
            settings.*(flag.member) = metadata.GetDataAs<bool>(flag.name);
        const auto &materialRemaps = metadata.GetMetadata().at("material_remaps");
        if (materialRemaps.first != "json_object")
            throw std::invalid_argument("model material_remaps metadata must use json_object");
        settings.materialRemaps = nlohmann::json::parse(std::any_cast<const std::string &>(materialRemaps.second));
        RequireMaterialRemaps(settings.materialRemaps);
        const auto &animationClips = metadata.GetMetadata().at("animation_clips");
        if (animationClips.first != "json_array")
            throw std::invalid_argument("model animation_clips metadata must use json_array");
        settings.animationClips = nlohmann::json::parse(std::any_cast<const std::string &>(animationClips.second));
        RequireAnimationClips(settings.animationClips);
        const auto &animationClipExtras = metadata.GetMetadata().at("animation_clip_extras");
        if (animationClipExtras.first != "json_array")
            throw std::invalid_argument("model animation_clip_extras metadata must use json_array");
        settings.animationClipExtras =
            nlohmann::json::parse(std::any_cast<const std::string &>(animationClipExtras.second));
        RequireAnimationClipExtras(settings.animationClipExtras);
        const auto &exposedBones = metadata.GetMetadata().at("exposed_bones");
        if (exposedBones.first != "json_array")
            throw std::invalid_argument("model exposed_bones metadata must use json_array");
        settings.exposedBones = nlohmann::json::parse(std::any_cast<const std::string &>(exposedBones.second));
        RequireExposedBones(settings.exposedBones);
        const auto &humanoidBoneOverrides = metadata.GetMetadata().at("humanoid_bone_overrides");
        if (humanoidBoneOverrides.first != "json_object")
            throw std::invalid_argument("model humanoid_bone_overrides metadata must use json_object");
        settings.humanoidBoneOverrides =
            nlohmann::json::parse(std::any_cast<const std::string &>(humanoidBoneOverrides.second));
        RequireHumanoidBoneOverrides(settings.humanoidBoneOverrides);
        settings.rigType = metadata.GetDataAs<std::string>("rig_type");
        RequireRigType(settings.rigType);
        settings.skeletonDefinitionMode = metadata.GetDataAs<std::string>("skeleton_definition_mode");
        RequireSkeletonDefinitionMode(settings.skeletonDefinitionMode);
        settings.rigRootNode = metadata.GetDataAs<std::string>("rig_root_node");
        RequireRigNodeName(settings.rigRootNode, "rig_root_node");
        settings.skeletonDefinitionGuid = metadata.GetDataAs<std::string>("skeleton_definition_guid");
        RequireRigNodeName(settings.skeletonDefinitionGuid, "skeleton_definition_guid");
        settings.skeletonDefinitionId = metadata.GetDataAs<std::string>("skeleton_definition_id");
        RequireRigNodeName(settings.skeletonDefinitionId, "skeleton_definition_id");
        if (settings.skeletonDefinitionId.empty())
            throw std::invalid_argument("model skeleton_definition_id must not be empty");
        settings.animationReferencePose = metadata.GetDataAs<std::string>("animation_reference_pose");
        RequireAnimationReferencePose(settings.animationReferencePose);
        settings.materialImportMode = metadata.GetDataAs<std::string>("material_import_mode");
        RequireMaterialImportMode(settings.materialImportMode);
        settings.meshCompression = metadata.GetDataAs<std::string>("mesh_compression");
        RequireMeshCompression(settings.meshCompression);
        settings.indexFormat = metadata.GetDataAs<std::string>("index_format");
        RequireIndexFormat(settings.indexFormat);
        return settings;
    }

    /// Populate a newly-created model metadata document with the complete
    /// current settings contract. Published metadata is read through Read()
    /// and is never completed or migrated by this function.
    static void InitializeDefaults(InxResourceMeta &metadata)
    {
        const MeshImportSettings defaults;
        metadata.AddMetadata("tangent_algorithm", defaults.tangentAlgorithm);
        metadata.AddMetadata("normal_weighting", defaults.normalWeighting);
        metadata.AddMetadata("normal_smoothing_source", defaults.normalSmoothingSource);
        for (const auto &field : BasisModes)
            metadata.AddMetadata(field.name, defaults.*(field.member));
        metadata.AddMetadata("max_bones_per_vertex", defaults.maxBonesPerVertex);
        for (const auto &field : Scalars)
            metadata.AddMetadata(field.name, defaults.*(field.member));
        for (const auto &flag : Flags)
            metadata.AddMetadata(flag.name, defaults.*(flag.member));
        WriteMaterialRemaps(metadata, defaults.materialRemaps);
        WriteAnimationClips(metadata, defaults.animationClips);
        WriteAnimationClipExtras(metadata, defaults.animationClipExtras);
        WriteExposedBones(metadata, defaults.exposedBones);
        WriteHumanoidBoneOverrides(metadata, defaults.humanoidBoneOverrides);
        metadata.AddMetadata("rig_type", defaults.rigType);
        metadata.AddMetadata("skeleton_definition_mode", defaults.skeletonDefinitionMode);
        metadata.AddMetadata("rig_root_node", defaults.rigRootNode);
        metadata.AddMetadata("skeleton_definition_guid", defaults.skeletonDefinitionGuid);
        metadata.AddMetadata("skeleton_definition_id", defaults.skeletonDefinitionId);
        metadata.AddMetadata("animation_reference_pose", defaults.animationReferencePose);
        metadata.AddMetadata("material_import_mode", defaults.materialImportMode);
        metadata.AddMetadata("mesh_compression", defaults.meshCompression);
        metadata.AddMetadata("index_format", defaults.indexFormat);
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
            if (key == "normal_smoothing_source") {
                RequireNormalSmoothingSource(value);
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
            if (key == "animation_clip_extras") {
                RequireAnimationClipExtras(value);
                continue;
            }
            if (key == "exposed_bones") {
                RequireExposedBones(value);
                continue;
            }
            if (key == "humanoid_bone_overrides") {
                RequireHumanoidBoneOverrides(value);
                continue;
            }
            if (key == "rig_root_node" || key == "skeleton_definition_guid" || key == "skeleton_definition_id") {
                RequireRigNodeName(value, key.c_str());
                if (key == "skeleton_definition_id" && value.get<std::string>().empty())
                    throw std::invalid_argument("model skeleton_definition_id must not be empty");
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
            if (key == "mesh_compression") {
                RequireMeshCompression(value);
                continue;
            }
            if (key == "index_format") {
                RequireIndexFormat(value);
                continue;
            }
            if (key == "rig_type") {
                RequireRigType(value);
                continue;
            }
            if (key == "skeleton_definition_mode") {
                RequireSkeletonDefinitionMode(value);
                continue;
            }
            if (key == "animation_reference_pose") {
                RequireAnimationReferencePose(value);
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
            else if (key == "animation_clip_extras")
                WriteAnimationClipExtras(metadata, value);
            else if (key == "exposed_bones")
                WriteExposedBones(metadata, value);
            else if (key == "humanoid_bone_overrides")
                WriteHumanoidBoneOverrides(metadata, value);
            else if (key == "material_remaps")
                WriteMaterialRemaps(metadata, value);
            else if (key == "max_bones_per_vertex")
                metadata.AddMetadata(key, value.get<int>());
            else if (key == "rig_type" || key == "skeleton_definition_mode" || key == "rig_root_node" ||
                     key == "skeleton_definition_guid" || key == "skeleton_definition_id" ||
                     key == "animation_reference_pose" || key == "material_import_mode" || key == "mesh_compression" ||
                     key == "index_format" || key == "normal_mode" || key == "tangent_mode" ||
                     key == "normal_weighting" || key == "normal_smoothing_source" || key == "tangent_algorithm")
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
                              {"step", field.step}});
        for (const auto &flag : Flags)
            fields.push_back({{"name", flag.name},
                              {"type", "bool"},
                              {"default", defaults.*(flag.member)},
                              {"page", flag.page},
                              {"label", std::string("asset.") + flag.name}});
        fields.push_back({{"name", "max_bones_per_vertex"},
                          {"type", "int"},
                          {"default", defaults.maxBonesPerVertex},
                          {"minimum", 1},
                          {"maximum", 4},
                          {"display_range", {1, 4}},
                          {"step", 1},
                          {"page", "rig"},
                          {"label", "asset.max_bones_per_vertex"}});
        for (const auto &field : BasisModes)
            fields.push_back({{"name", field.name},
                              {"type", "enum"},
                              {"default", defaults.*(field.member)},
                              {"page", "model"},
                              {"label", std::string("asset.") + field.name},
                              {"choices",
                               {{{"value", "import"}, {"label", "asset.basis_import"}},
                                {{"value", "calculate"}, {"label", "asset.basis_calculate"}},
                                {{"value", "none"}, {"label", "asset.basis_none"}},
                                {{"value", "source_only"}, {"label", "asset.basis_source_only"}}}}});
        fields.push_back({{"name", "normal_weighting"},
                          {"type", "enum"},
                          {"default", defaults.normalWeighting},
                          {"page", "model"},
                          {"label", "asset.normal_weighting"},
                          {"choices",
                           {{{"value", "unweighted"}, {"label", "asset.normal_unweighted"}},
                            {{"value", "area"}, {"label", "asset.normal_area"}},
                            {{"value", "angle"}, {"label", "asset.normal_angle"}},
                            {{"value", "area_angle"}, {"label", "asset.normal_area_angle"}}}}});
        fields.push_back({{"name", "normal_smoothing_source"},
                          {"type", "enum"},
                          {"default", defaults.normalSmoothingSource},
                          {"page", "model"},
                          {"label", "asset.normal_smoothing_source"},
                          {"choices",
                           {{{"value", "angle"}, {"label", "asset.normal_smoothing_angle_source"}},
                            {{"value", "source"}, {"label", "asset.normal_smoothing_authored_source"}}}}});
        fields.push_back({{"name", "tangent_algorithm"},
                          {"type", "enum"},
                          {"default", defaults.tangentAlgorithm},
                          {"page", "model"},
                          {"label", "asset.tangent_algorithm"},
                          {"choices",
                           {{{"value", "mikktspace"}, {"label", "asset.tangent_mikktspace"}},
                            {{"value", "assimp"}, {"label", "asset.tangent_assimp"}}}}});
        fields.push_back({{"name", "rig_type"},
                          {"type", "enum"},
                          {"default", defaults.rigType},
                          {"choices",
                           {{{"value", "none"}, {"label", "asset.rig_none"}},
                            {{"value", "generic"}, {"label", "asset.rig_generic"}},
                            {{"value", "humanoid"}, {"label", "asset.rig_humanoid"}}}},
                          {"page", "rig"},
                          {"label", "asset.rig_type"}});
        fields.push_back({{"name", "skeleton_definition_mode"},
                          {"type", "enum"},
                          {"default", defaults.skeletonDefinitionMode},
                          {"page", "rig"},
                          {"label", "asset.skeleton_definition_mode"},
                          {"choices",
                           {{{"value", "create"}, {"label", "asset.skeleton_definition_create"}},
                            {{"value", "copy"}, {"label", "asset.skeleton_definition_copy"}}}}});
        fields.push_back({{"name", "rig_root_node"},
                          {"type", "rig_root_node"},
                          {"default", defaults.rigRootNode},
                          {"page", "rig"},
                          {"label", "asset.rig_root_node"}});
        fields.push_back({{"name", "skeleton_definition_guid"},
                          {"type", "rig_definition_guid"},
                          {"default", defaults.skeletonDefinitionGuid},
                          {"page", "rig"},
                          {"label", "asset.skeleton_definition_guid"}});
        fields.push_back({{"name", "skeleton_definition_id"},
                          {"type", "rig_definition_id"},
                          {"default", defaults.skeletonDefinitionId},
                          {"page", "rig"},
                          {"label", "asset.skeleton_definition_id"}});
        fields.push_back({{"name", "exposed_bones"},
                          {"type", "exposed_bones"},
                          {"default", nlohmann::json::array()},
                          {"page", "rig"},
                          {"label", "asset.exposed_bones"}});
        fields.push_back({{"name", "humanoid_bone_overrides"},
                          {"type", "humanoid_bone_overrides"},
                          {"default", nlohmann::json::object()},
                          {"page", "rig"},
                          {"label", "asset.humanoid_bone_overrides"}});
        fields.push_back({{"name", "animation_reference_pose"},
                          {"type", "enum"},
                          {"default", defaults.animationReferencePose},
                          {"choices",
                           {{{"value", "bind_pose"}, {"label", "asset.animation_reference_bind_pose"}},
                            {{"value", "first_frame"}, {"label", "asset.animation_reference_first_frame"}}}},
                          {"page", "animation"},
                          {"label", "asset.animation_reference_pose"}});
        fields.push_back({{"name", "material_import_mode"},
                          {"type", "enum"},
                          {"default", defaults.materialImportMode},
                          {"choices",
                           {{{"value", "none"}, {"label", "asset.material_import_none"}},
                            {{"value", "description"}, {"label", "asset.material_import_description"}}}},
                          {"page", "materials"},
                          {"label", "asset.material_import_mode"}});
        fields.push_back({{"name", "mesh_compression"},
                          {"type", "enum"},
                          {"default", defaults.meshCompression},
                          {"choices",
                           {{{"value", "off"}, {"label", "asset.mesh_compression_off"}},
                            {{"value", "low"}, {"label", "asset.mesh_compression_low"}},
                            {{"value", "medium"}, {"label", "asset.mesh_compression_medium"}},
                            {{"value", "high"}, {"label", "asset.mesh_compression_high"}}}},
                          {"page", "model"},
                          {"label", "asset.mesh_compression"}});
        fields.push_back({{"name", "index_format"},
                          {"type", "enum"},
                          {"default", defaults.indexFormat},
                          {"choices",
                           {{{"value", "auto"}, {"label", "asset.index_format_auto"}},
                            {{"value", "uint16"}, {"label", "asset.index_format_uint16"}},
                            {{"value", "uint32"}, {"label", "asset.index_format_uint32"}}}},
                          {"page", "model"},
                          {"label", "asset.index_format"}});
        fields.push_back({{"name", "material_remaps"},
                          {"type", "material_remaps"},
                          {"default", nlohmann::json::object()},
                          {"page", "materials"},
                          {"label", "asset.material_remaps"}});
        fields.push_back({{"name", "animation_clips"},
                          {"type", "animation_clips"},
                          {"default", nlohmann::json::array()},
                          {"page", "animation"},
                          {"label", "asset.animation_clips"}});
        fields.push_back({{"name", "animation_clip_extras"},
                          {"type", "animation_clip_extras"},
                          {"default", nlohmann::json::array()},
                          {"page", "animation"},
                          {"label", "asset.animation_clip_extras"}});
        return {{"version", 1}, {"fields", std::move(fields)}};
    }
};
} // namespace infernux
