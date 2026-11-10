#include "SerializedResourceReferences.h"

#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxMaterial/MaterialDocumentValidation.h>
#include <function/scene/ComponentRecord.h>
#include <function/scene/SceneEnvironment.h>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace infernux
{
namespace
{
bool IsBuiltinTexture(const std::string &guid)
{
    return guid == "white" || guid == "black" || guid == "normal";
}

std::optional<ResourceType> ResourceTypeFromName(const std::string &name)
{
    static const std::unordered_map<std::string, ResourceType> types = {
        {"Shader", ResourceType::Shader},
        {"Texture", ResourceType::Texture},
        {"Mesh", ResourceType::Mesh},
        {"Material", ResourceType::Material},
        {"Script", ResourceType::Script},
        {"Audio", ResourceType::Audio},
        {"PhysicMaterial", ResourceType::PhysicMaterial},
        {"RenderEffect", ResourceType::RenderEffect},
        {"ParticleGraph", ResourceType::ParticleGraph},
        {"DataAsset", ResourceType::DataAsset},
        {"RenderTexture", ResourceType::RenderTexture},
    };
    const auto found = types.find(name);
    return found == types.end() ? std::nullopt : std::optional<ResourceType>(found->second);
}

class DirectResourceReferences
{
  public:
    explicit DirectResourceReferences(const SerializedAssetVisitor &visit) : m_visit(visit)
    {
    }

    void Scene(const nlohmann::json &document)
    {
        const auto &objects = document.at("objects");
        for (size_t index = 0; index < objects.size(); ++index)
            ValidateObject(objects[index], "Scene.objects[" + std::to_string(index) + "]");
        CollectSerializedAssetRefs(document, "Scene");
        if (const auto environment = document.find("environment");
            environment != document.end() && environment->is_object()) {
            const auto skybox = environment->find(SceneEnvironmentSettings::SkyboxMaterialGuidKey);
            if (skybox != environment->end() && skybox->is_string() && !skybox->get_ref<const std::string &>().empty())
                RequireAsset(skybox->get<std::string>(), ResourceType::Material,
                             std::string("Scene.environment.") + SceneEnvironmentSettings::SkyboxMaterialGuidKey);
        }
    }

    void Prefab(const nlohmann::json &document)
    {
        ValidateObject(document.at("root_object"), "Prefab.root_object", true);
        if (const auto variant = document.find("variant"); variant != document.end())
            RequireAsset(variant->at("guid").get<std::string>(), ResourceType::DefaultText, "Prefab.variant.guid");
    }

  private:
    void RequireAsset(const std::string &guid, ResourceType type, const std::string &path)
    {
        m_visit(guid, type, path);
    }

    void CollectSerializedAssetRefs(const nlohmann::json &value, const std::string &path)
    {
        if (value.is_array()) {
            for (size_t index = 0; index < value.size(); ++index)
                CollectSerializedAssetRefs(value[index], path + "[" + std::to_string(index) + "]");
            return;
        }
        if (!value.is_object())
            return;

        const auto marker = value.find("$type");
        if (marker != value.end() && marker->is_string() && marker->get_ref<const std::string &>() == "asset_ref") {
            const auto typeName = value.find("asset_type");
            const auto guid = value.find("guid");
            if (typeName != value.end() && typeName->is_string() && guid != value.end() && guid->is_string() &&
                !guid->get_ref<const std::string &>().empty()) {
                if (const auto type = ResourceTypeFromName(typeName->get<std::string>()); type.has_value())
                    RequireAsset(guid->get<std::string>(), *type, path + ".guid");
            }
        }
        for (const auto &[name, child] : value.items())
            CollectSerializedAssetRefs(child, path + "." + name);
    }

    void ValidateEmbeddedMaterial(const nlohmann::json &document, const std::string &path)
    {
        material_document_validation::ValidateMaterialDocument(document, path);
        if (document.contains("properties") && document["properties"].is_object()) {
            for (const auto &[name, property] : document["properties"].items()) {
                if (!property.is_object() || !property.contains("type") || !property["type"].is_number_integer())
                    continue;
                if (property["type"].get<int>() != static_cast<int>(MaterialPropertyType::Texture2D))
                    continue;
                if (!property.contains("guid") || !property["guid"].is_string())
                    throw std::invalid_argument(path + ".properties." + name + ".guid must be a string");
                const std::string guid = property["guid"].get<std::string>();
                if (!guid.empty() && !IsBuiltinTexture(guid))
                    RequireAsset(guid, ResourceType::Texture, path + ".properties." + name + ".guid");
            }
        }
    }

    void ValidateComponent(const nlohmann::json &component, const std::string &path)
    {
        const DecodedComponentRecord record = DecodeComponentRecord(component);
        if (record.kind == ComponentRecordKind::Python)
            return;
        const std::string &type = record.nativeTypeName;
        const nlohmann::json data = BuildNativeComponentDocument(record);
        if (type == "SkeletonInstance") {
            const std::string guid = data.at("source_model_guid").get<std::string>();
            if (!guid.empty())
                RequireAsset(guid, ResourceType::Mesh, path + ".source_model_guid");
            return;
        }
        if (type == "Camera") {
            const std::string guid = data.value("targetTextureGuid", std::string{});
            if (!guid.empty())
                RequireAsset(guid, ResourceType::RenderTexture, path + ".targetTextureGuid");
            return;
        }
        if (type == "BoxCollider" || type == "SphereCollider" || type == "CapsuleCollider" ||
            type == "CylinderCollider" || type == "MeshCollider") {
            const std::string guid = data.at("physic_material_guid").get<std::string>();
            if (!guid.empty())
                RequireAsset(guid, ResourceType::PhysicMaterial, path + ".physic_material_guid");
            return;
        }

        if (type == "AudioSource") {
            const auto &tracks = data.at("tracks");
            for (size_t index = 0; index < tracks.size(); ++index) {
                if (!tracks[index].contains("clip_guid"))
                    continue;
                RequireAsset(tracks[index]["clip_guid"].get<std::string>(), ResourceType::Audio,
                             path + ".tracks[" + std::to_string(index) + "].clip_guid");
            }
            return;
        }

        if (type != "MeshRenderer" && type != "SkinnedMeshRenderer" && type != "SpriteRenderer" &&
            type != "LineRenderer" && type != "TrailRenderer")
            return;

        if (data.contains("meshAssetGuid"))
            RequireAsset(data["meshAssetGuid"].get<std::string>(), ResourceType::Mesh, path + ".meshAssetGuid");
        const auto &materials = data.at("materials");
        for (size_t index = 0; index < materials.size(); ++index) {
            const auto &slot = materials[index];
            const std::string slotPath = path + ".materials[" + std::to_string(index) + "]";
            if (slot.is_string()) {
                RequireAsset(slot.get<std::string>(), ResourceType::Material, slotPath);
            } else if (slot.is_object()) {
                ValidateEmbeddedMaterial(slot.at("material"), slotPath + ".material");
            }
        }
        if (type == "SpriteRenderer" && data.contains("spriteGuid"))
            RequireAsset(data["spriteGuid"].get<std::string>(), ResourceType::Texture, path + ".spriteGuid");
    }

    void ValidateObject(const nlohmann::json &object, const std::string &path, bool prefab = false)
    {
        if (const auto source = object.find("model_source"); source != object.end()) {
            if (!source->is_object() || !source->contains("guid") || !(*source)["guid"].is_string())
                throw std::invalid_argument(path + ".model_source.guid must be a string");
            RequireAsset((*source)["guid"].get<std::string>(), ResourceType::Mesh, path + ".model_source.guid");
        }
        const auto &components = object.at("components");
        for (size_t index = 0; index < components.size(); ++index) {
            const auto componentPath = path + ".components[" + std::to_string(index) + "]";
            ValidateComponent(components[index], componentPath);
            if (prefab)
                CollectSerializedAssetRefs(components[index].at("data"), componentPath + ".data");
        }
        if (prefab) {
            if (const auto nested = object.find("nested_prefab"); nested != object.end())
                RequireAsset(nested->at("guid").get<std::string>(), ResourceType::DefaultText,
                             path + ".nested_prefab.guid");
        }
        const auto &children = object.at("children");
        for (size_t index = 0; index < children.size(); ++index)
            ValidateObject(children[index], path + ".children[" + std::to_string(index) + "]", prefab);
    }

    const SerializedAssetVisitor &m_visit;
};
} // namespace

void VisitSceneResourceReferences(const nlohmann::json &document, const SerializedAssetVisitor &visit)
{
    DirectResourceReferences{visit}.Scene(document);
}

void VisitPrefabResourceReferences(const nlohmann::json &document, const SerializedAssetVisitor &visit)
{
    DirectResourceReferences{visit}.Prefab(document);
}
} // namespace infernux
