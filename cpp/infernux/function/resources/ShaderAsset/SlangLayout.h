#pragma once

#include <function/renderer/shader/ShaderProgramLayout.h>

namespace slang
{
struct TypeLayoutReflection;
}

namespace infernux
{
std::vector<ShaderResourceLayout> ReflectSlangResources(slang::TypeLayoutReflection *layout);
}
