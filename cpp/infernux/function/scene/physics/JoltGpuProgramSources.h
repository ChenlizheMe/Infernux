#pragma once

#include <function/resources/ShaderAsset/SlangCompiler.h>

namespace infernux
{

enum class JoltGpuProgram
{
    Step,
    BodyMutations,
    BodyCommands,
    Queries,
    Joints,
    RenderPoses,
    Broadphase
};

/// Builtin native programs, embedded from the pinned Jolt fork at build time.
/// Names/virtual paths stay stable in compiler diagnostics; loading does not
/// access the engine source tree or the current project directory.
[[nodiscard]] ShaderSourceModule GetJoltGpuProgramSource(JoltGpuProgram program);

/// Complete frozen include closure for these programs. Pass it unchanged to
/// the shared SlangCompiler; no physics-specific compiler or cache is needed.
[[nodiscard]] std::vector<ShaderSourceModule> GetJoltGpuProgramLibraries();

} // namespace infernux
