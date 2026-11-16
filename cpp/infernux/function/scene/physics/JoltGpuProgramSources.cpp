#include "JoltGpuProgramSources.h"

#include <function/renderer/rhi/RhiComputeKernel.h>
#include <function/renderer/shader/ShaderTargetArchive.h>

extern "C" const void *InfernuxJoltGpuArchive(unsigned program, size_t *size);

namespace infernux
{

ShaderTargetCode LoadJoltGpuProgram(JoltGpuProgram program, ShaderCodeTarget target)
{
    size_t size;
    const auto *bytes = static_cast<const char *>(InfernuxJoltGpuArchive(static_cast<unsigned>(program), &size));
    auto archive = DecodeShaderTarget(std::string_view(bytes, size));
    if (archive.target != target)
        throw std::logic_error("Cooked GPU physics target does not match the compute device");
    return archive;
}

std::shared_ptr<rhi::ComputeKernel> PrepareJoltGpuKernel(rhi::ComputeHost &host, ShaderCodeTarget target,
                                                         const ShaderEntryPointCode &entry,
                                                         std::vector<rhi::ComputeBufferBinding> bindings)
{
    const bool spirv = target == ShaderCodeTarget::SpirV;
    if (spirv && entry.code.size() % sizeof(uint32_t) != 0)
        throw std::invalid_argument("Invalid SPIR-V extent for GPU physics entry " + entry.name);
    const rhi::ShaderModuleDesc shader{spirv ? rhi::ShaderSourceLanguage::SpirV : rhi::ShaderSourceLanguage::Wgsl,
                                       entry.code.data(), entry.code.size()};
    return std::make_shared<rhi::ComputeKernel>(host, shader, entry.name, std::move(bindings), 0,
                                                rhi::ComputeKernel::PipelineCreation::FirstDispatch);
}

} // namespace infernux
