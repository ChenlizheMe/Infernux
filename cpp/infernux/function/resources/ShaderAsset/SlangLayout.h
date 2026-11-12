#pragma once

#include <function/renderer/shader/ShaderProgramLayout.h>

namespace slang
{
struct TypeLayoutReflection;
struct EntryPointReflection;
} // namespace slang

namespace infernux
{
struct ShaderEntryPointCode;
std::vector<ShaderResourceLayout> ReflectSlangResources(slang::TypeLayoutReflection *layout);
void ReflectSlangStageIO(slang::EntryPointReflection *layout, ShaderEntryPointCode &entry);
} // namespace infernux
