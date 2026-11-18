#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace infernux
{

struct ShaderCacheInput
{
    std::string id;
    uint64_t revision;
};

using ShaderCacheRevision = std::function<std::optional<uint64_t>(const std::string &)>;

std::string ProjectCacheDirectory(const std::string &projectRoot);
ShaderCacheInput MakeShaderCacheInput(std::string id, std::string_view originalBytes);

// selection is the same program, target and compiler contract used by the cooker.
// inputs name the original author files, including transitive imports. A cache
// hit needs their revisions but never a front-end invocation.
std::optional<std::string> ReadShaderProgramCache(const std::string &cacheDirectory, std::string_view selection,
                                                  const ShaderCacheRevision &currentRevision);
bool WriteShaderProgramCache(const std::string &cacheDirectory, std::string_view selection,
                             const std::vector<ShaderCacheInput> &inputs, std::string_view compiledProgram);

} // namespace infernux
