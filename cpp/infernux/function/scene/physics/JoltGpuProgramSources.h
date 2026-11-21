#pragma once

#include <function/renderer/shader/ShaderProgramCode.h>

#include <memory>

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
    Broadphase,
    Narrowphase,
    Constraints,
    ContactReports,
    ArticulationCapture,
    ArticulationDynamics,
    Count
};

[[nodiscard]] ShaderTargetCode LoadJoltGpuProgram(JoltGpuProgram program, ShaderCodeTarget target);

/// Validate the same artifact at load while allocating a device pipeline only
/// when the selected physics stage first prepares a dispatch.
[[nodiscard]] std::shared_ptr<rhi::ComputeKernel> PrepareJoltGpuKernel(rhi::ComputeHost &host, ShaderCodeTarget target,
                                                                       const ShaderEntryPointCode &entry,
                                                                       std::vector<rhi::ComputeBufferBinding> bindings);

} // namespace infernux
