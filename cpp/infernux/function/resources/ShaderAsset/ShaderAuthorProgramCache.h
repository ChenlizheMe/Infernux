#pragma once

#include "ShaderProgramDiskCache.h"
#include <function/renderer/shader/ShaderProgramCode.h>
#include <memory>

namespace infernux
{
class AssetDatabase;

struct ShaderAuthorProgramRecord
{
    // In-memory paths remain usable by tools. Only the private persisted author
    // payload replaces known source paths with verified shader-input identities.
    std::string schema;
    std::shared_ptr<const ShaderCompilation> compilation;
    std::vector<std::string> dependencies;
};

ShaderCodeTarget RuntimeShaderAuthorTarget() noexcept;
std::string ShaderAuthorProgramSelection(std::string_view authorGuid, std::string_view vertexGuid,
                                         ShaderCodeTarget target);
std::string ShaderAuthorInputId(const AssetDatabase &database, const std::string &path);
std::optional<ShaderAuthorProgramRecord>
FindShaderAuthorProgram(const AssetDatabase &database, std::string_view selection,
                        const std::vector<ShaderCacheInput> &loadedInputs = {},
                        std::vector<ShaderCacheInput> *matchedInputs = nullptr, const std::string &buildCache = {});
bool StoreShaderAuthorProgram(const std::string &cacheDirectory, std::string_view selection,
                              const std::vector<ShaderCacheInput> &inputs, const ShaderAuthorProgramRecord &record);
} // namespace infernux
