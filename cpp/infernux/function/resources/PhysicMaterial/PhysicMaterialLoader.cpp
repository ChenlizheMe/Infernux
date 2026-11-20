#include "PhysicMaterialLoader.h"

#include "PhysicMaterial.h"
#include <core/log/InxLog.h>
#include <fstream>
#include <platform/filesystem/AssetDocument.h>
#include <platform/filesystem/InxPath.h>

namespace infernux
{
void PhysicMaterialLoader::OnRelocated(const RuntimeAssetPayload &payload, const std::string &filePath)
{
    const auto material = payload.Get<PhysicMaterial>();
    material->SetFilePath(filePath);
    material->SetName(FromFsPath(ToFsPath(filePath).stem()));
}

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
    auto material = std::make_shared<PhysicMaterial>();
    material->DeserializeDocument(ReadDocument(filePath));
    material->SetFilePath(filePath);
    material->SetName(FromFsPath(ToFsPath(filePath).stem()));
    material->SetGuid(guid);
    return material;
}

bool PhysicMaterialLoader::Reload(const RuntimeAssetPayload &existing, const std::string &filePath,
                                  const std::string &guid, AssetDatabase *)
{
    auto material = existing.Get<PhysicMaterial>();
    if (!material)
        throw std::invalid_argument("PhysicMaterialLoader cannot reload a null instance");
    PhysicMaterial staged;
    staged.DeserializeDocument(ReadDocument(filePath));
    material->CopyValuesFrom(staged);
    material->SetFilePath(filePath);
    material->SetName(FromFsPath(ToFsPath(filePath).stem()));
    material->SetGuid(guid);
    return true;
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

void PhysicMaterialLoader::CreateMeta(AssetSource &source, InxResourceMeta &metaData) const
{
    metaData.InitFromSource(source, ResourceType::PhysicMaterial);
    metaData.AddMetadata("physic_material_name", FromFsPath(ToFsPath(source.Path()).stem()));
}

} // namespace infernux
