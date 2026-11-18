#pragma once

#include <function/renderer/shader/ShaderTargetArchive.h>

extern "C" const void *InfernuxUIShaderArchive(unsigned program, size_t *size);

namespace infernux
{

enum class BuiltinUIProgram : unsigned
{
    Host,
    Screen,
    World,
    Count
};

inline ShaderTargetCode LoadBuiltinUIProgram(BuiltinUIProgram program)
{
    size_t size;
    const auto *bytes = static_cast<const char *>(InfernuxUIShaderArchive(static_cast<unsigned>(program), &size));
    return DecodeShaderTarget(std::string_view(bytes, size));
}

} // namespace infernux
