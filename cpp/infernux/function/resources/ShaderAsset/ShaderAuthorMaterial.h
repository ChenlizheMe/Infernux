#pragma once

#include "ShaderCompilerApi.h"
#include <core/types/ShaderProgramArtifact.h>
#include <function/renderer/shader/ShaderProgramCode.h>

#include <memory>
#include <string>

namespace infernux
{

/// Link a prepared mesh author's logical fields to the compiler's physical
/// target. Runs once on asset preparation, independently of GPU submission.
[[nodiscard]] INFERNUX_SHADER_COMPILER_API ShaderProgramArtifact
CreateShaderAuthorMaterial(std::string guid, const std::shared_ptr<const ShaderCompilation> &compilation,
                           const std::string &schema, std::string vertexKey);

} // namespace infernux
