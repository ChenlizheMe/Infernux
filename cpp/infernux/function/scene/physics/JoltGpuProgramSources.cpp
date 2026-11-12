#include "JoltGpuProgramSources.h"

#include <Jolt/GPU/PhysicsStages.h>
#include <infernux/physics/JoltGpuEmbeddedSources.h>

namespace infernux
{

ShaderSourceModule GetJoltGpuProgramSource(JoltGpuProgram program)
{
    switch (program) {
    case JoltGpuProgram::Step:
        return {JPH::GPU::StepModuleName, JPH::GPU::StepSourcePath, jolt_gpu_sources::Step};
    case JoltGpuProgram::BodyMutations:
        return {"jph_gpu_body_mutations", "Jolt/GPU/Shaders/rigid_body_mutations.slang",
                jolt_gpu_sources::BodyMutations};
    case JoltGpuProgram::BodyCommands:
        return {"jph_gpu_body_commands", "Jolt/GPU/Shaders/body_commands.slang", jolt_gpu_sources::BodyCommands};
    case JoltGpuProgram::Queries:
        return {"jph_gpu_queries", "Jolt/GPU/Shaders/rigid_queries.slang", jolt_gpu_sources::Queries};
    case JoltGpuProgram::RenderPoses:
        return {"jph_gpu_render_poses", "Jolt/GPU/Shaders/render_poses.slang", jolt_gpu_sources::RenderPoses};
    }
    throw std::invalid_argument("Unknown builtin Jolt GPU program");
}

std::vector<ShaderSourceModule> GetJoltGpuProgramLibraries()
{
    return {{"jph_gpu_rigid_math", "Jolt/GPU/Shaders/rigid_math.slang", jolt_gpu_sources::Math},
            {"jph_gpu_physics_types", "Jolt/GPU/PhysicsTypes.h", jolt_gpu_sources::Types}};
}

} // namespace infernux
