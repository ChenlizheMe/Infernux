#pragma once

#include <function/renderer/shader/ShaderProgramLayout.h>
#include <optional>

namespace slang
{
struct TypeLayoutReflection;
struct EntryPointReflection;
} // namespace slang

namespace infernux
{
struct ShaderEntryPointCode;
std::vector<ShaderResourceLayout> ReflectSlangResources(slang::TypeLayoutReflection *layout,
                                                        std::optional<ShaderValueLayout> &rootConstants);
void ReflectSlangStageIO(slang::EntryPointReflection *layout, ShaderEntryPointCode &entry);
} // namespace infernux
