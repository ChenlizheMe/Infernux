#pragma once

#include "ShaderTargetArchive.h"

extern "C" const void *InfernuxSceneShaderArchive(unsigned program, size_t *size);

namespace infernux
{

enum class BuiltinSceneProgram : unsigned
{
    DepthResolve,
    ForwardPlusGrid,
    Count
};

inline ShaderTargetCode LoadBuiltinSceneProgram(BuiltinSceneProgram program)
{
    size_t size;
    const auto *bytes = static_cast<const char *>(InfernuxSceneShaderArchive(static_cast<unsigned>(program), &size));
    return DecodeShaderTarget({bytes, size});
}

} // namespace infernux
