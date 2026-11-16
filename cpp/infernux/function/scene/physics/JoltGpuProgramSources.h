#pragma once

#include <function/resources/ShaderAsset/SlangCompiler.h>

namespace infernux
{

namespace rhi
{
class ComputeKernel;
class ComputeHost;
struct ComputeBufferBinding;
} // namespace rhi

enum class JoltGpuProgram
{
    Step,
    BodyMutations,
    BodyCommands,
    Queries,
    Joints,
    RenderPoses,
    Broadphase,
    Narrowphase,
    Constraints,
    ContactReports,
    ArticulationCapture,
    ArticulationDynamics
};

/// Builtin native programs, embedded from the pinned Jolt fork at build time.
/// Names/virtual paths stay stable in compiler diagnostics; loading does not
/// access the engine source tree or the current project directory.
[[nodiscard]] ShaderSourceModule GetJoltGpuProgramSource(JoltGpuProgram program);

/// Complete frozen include closure for these programs. Pass it unchanged to
/// the shared SlangCompiler; no physics-specific compiler or cache is needed.
[[nodiscard]] std::vector<ShaderSourceModule> GetJoltGpuProgramLibraries();

/// Validate the same artifact at load while allocating a device pipeline only
/// when the selected physics stage first prepares a dispatch.
[[nodiscard]] std::shared_ptr<rhi::ComputeKernel> PrepareJoltGpuKernel(rhi::ComputeHost &host, ShaderCodeTarget target,
                                                                       const ShaderEntryPointCode &entry,
                                                                       std::vector<rhi::ComputeBufferBinding> bindings);

} // namespace infernux
