#pragma once

#include "SlangCompiler.h"
#include <string_view>

namespace infernux
{

/// Native engine modules use the same compiler, diagnostics and target layout
/// as authored shaders. Only the declaration of RHI root values is target
/// dependent; the algorithm and its resource bindings are shared.
inline ShaderTargetCode CompileSlangNativeModule(ShaderSourceModule source, ShaderCodeTarget target)
{
    auto result = SlangCompiler::Get().CompileRootProgram(std::move(source), {}, {target});
    return std::move(result.targets.front());
}

} // namespace infernux
