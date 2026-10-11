#include "PhysicMaterialLoader.h"

#include "PhysicMaterial.h"
#include <core/log/InxLog.h>
#include <fstream>
#include <platform/filesystem/AssetDocument.h>
#include <platform/filesystem/InxPath.h>

namespace infernux
{

namespace
{
nlohmann::json ReadDocument(const std::string &filePath)
{
    return ReadAssetDocument(filePath);
}
} // namespace

RuntimeAssetPayload PhysicMaterialLoader::Load(const std::string &filePath, const std::string &guid, AssetDatabase *)
{
    if (filePath.empty() || guid.empty())
        throw std::invalid_argument("PhysicMaterialLoader requires a file path and GUID");
    try {
        auto material = std::make_shared<PhysicMaterial>();
        material->DeserializeDocument(ReadDocument(filePath));
        material->SetFilePath(filePath);
        material->SetName(FromFsPath(ToFsPath(filePath).stem()));
        material->SetGuid(guid);
        return material;
    } catch (const std::exception &exception) {
        INXLOG_ERROR("PhysicMaterialLoader failed for '", filePath, "': ", exception.what());
        return nullptr;
    }
}

bool PhysicMaterialLoader::Reload(const RuntimeAssetPayload &existing, const std::string &filePath,
                                  const std::string &guid, AssetDatabase *)
{
    auto material = existing.Get<PhysicMaterial>();
    if (!material)
        throw std::invalid_argument("PhysicMaterialLoader cannot reload a null instance");
    try {
        PhysicMaterial staged;
        staged.DeserializeDocument(ReadDocument(filePath));
        material->CopyValuesFrom(staged);
        material->SetFilePath(filePath);
        material->SetName(FromFsPath(ToFsPath(filePath).stem()));
        material->SetGuid(guid);
        return true;
    } catch (const std::exception &exception) {
        INXLOG_ERROR("PhysicMaterialLoader reload failed for '", filePath, "': ", exception.what());
        return false;
    }
}

size_t PhysicMaterialLoader::EstimateRuntimeBytes(const RuntimeAssetPayload &payload) const
{
    const auto material = payload.Get<PhysicMaterial>();
    if (!material)
        throw std::invalid_argument("PhysicMaterialLoader cannot estimate an empty runtime payload");
    return material->GetRuntimeMemoryBytes();
}

std::set<std::string> PhysicMaterialLoader::ScanDependencies(const std::string &, AssetDatabase *)
{
    return {};
}

void PhysicMaterialLoader::CreateMeta(const char *content, size_t contentSize, const std::string &filePath,
                                      InxResourceMeta &metaData) const
{
    if (!content)
        throw std::invalid_argument("PhysicMaterial metadata requires file content");
    metaData.Init(content, contentSize, filePath, ResourceType::PhysicMaterial);
    metaData.AddMetadata("physic_material_name", FromFsPath(ToFsPath(filePath).stem()));
}

} // namespace infernux
