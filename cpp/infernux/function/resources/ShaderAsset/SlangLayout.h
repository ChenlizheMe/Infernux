#pragma once

#include <function/renderer/shader/ShaderProgramLayout.h>
#include <slang.h>

namespace slang
{
struct TypeLayoutReflection;
struct EntryPointReflection;
} // namespace slang

namespace infernux
{
struct ShaderEntryPointCode;
// A compiler location before final entry usage selects its target layout.
struct SlangRootParameter
{
    SlangParameterCategory category;
    uint32_t space;
    uint32_t binding;
    ShaderValueLayout value;
};
std::vector<SlangRootParameter> ReflectSlangRootParameters(slang::TypeLayoutReflection *layout);
std::vector<ShaderResourceLayout> ReflectSlangResources(slang::TypeLayoutReflection *layout);
void ReflectSlangStageIO(slang::EntryPointReflection *layout, ShaderEntryPointCode &entry);
} // namespace infernux
