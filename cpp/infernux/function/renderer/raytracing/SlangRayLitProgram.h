#pragma once

#include <function/resources/ShaderAsset/SlangCompiler.h>

namespace infernux::raytracing
{

/// Compile the caller's entry importing inx_ray_lit from the frozen built-in Lit author.
[[nodiscard]] inline ShaderCompilation CompileRayLitProgram(const ShaderSourceModule &entry, ShaderSourceModule lit,
                                                            const ShaderTextReader &reader,
                                                            const std::vector<ShaderCodeTarget> &targets,
                                                            bool precise = false)
{
    auto &compiler = SlangCompiler::Get();
    lit.name = "inx_ray_lit";
    auto prepared = compiler.PrepareAuthorSource(lit, reader);
    prepared.libraries.push_back(std::move(prepared.source));
    return compiler.Compile(entry, prepared.libraries, targets, precise, prepared.sourceRanges);
}

} // namespace infernux::raytracing
