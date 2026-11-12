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
    const std::string declaration = target == ShaderCodeTarget::SpirV
                                        ? "#define INX_ROOT_CONSTANTS [[vk::push_constant]]\n"
                                        : "#define INX_ROOT_CONSTANTS [[vk::binding(" +
                                              std::to_string(rhi::RootConstantBinding) + ", " +
                                              std::to_string(rhi::RootConstantGroup) + ")]]\n";
    source.source.insert(0, declaration + "#line 1\n");
    auto result = SlangCompiler::Get().Compile(source, {}, {target});
    return std::move(result.targets.front());
}

} // namespace infernux
