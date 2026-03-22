#pragma once

#include <function/resources/InfResource/InfResourceMeta.h>

namespace infengine
{

/**
 * @brief Legacy material meta creator — now superseded by MaterialLoader (IAssetLoader).
 * Retained temporarily for source compatibility; not instantiated by AssetDatabase.
 */
class InfMaterialLoader
{
  public:
    bool LoadMeta(const char *content, const std::string &filePath, InfResourceMeta &metaData);
    void CreateMeta(const char *content, size_t contentSize, const std::string &filePath, InfResourceMeta &metaData);
};

} // namespace infengine
