#include "InfMaterialLoader.hpp"
#include <core/log/InfLog.h>
#include <filesystem>
#include <platform/filesystem/InfPath.h>

namespace infengine
{

bool InfMaterialLoader::LoadMeta(const char *content, const std::string &filePath, InfResourceMeta &metaData)
{
    INFLOG_DEBUG("Loading material with metadata from file: ", filePath);
    // not implemented yet - material meta loading
    return false;
}

void InfMaterialLoader::CreateMeta(const char *content, size_t contentSize, const std::string &filePath,
                                   InfResourceMeta &metaData)
{
    if (!content) {
        INFLOG_ERROR("Invalid material content for metadata creation");
        return;
    }

    metaData.Init(content, contentSize, filePath, ResourceType::Material);

    // Use filename stem as material name — authoritative name is set by
    // MaterialLoader::Load() at runtime, so parsing JSON here is unnecessary.
    std::filesystem::path path = ToFsPath(filePath);
    metaData.AddMetadata("material_name", FromFsPath(path.stem()));

    INFLOG_DEBUG("Material metadata created for file: ", filePath);
}

} // namespace infengine
