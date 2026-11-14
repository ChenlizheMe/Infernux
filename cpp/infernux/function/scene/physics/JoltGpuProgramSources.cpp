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
    case JoltGpuProgram::Broadphase:
        return {"jph_gpu_broadphase", "Jolt/GPU/Shaders/rigid_broadphase.slang", jolt_gpu_sources::Broadphase};
    case JoltGpuProgram::Constraints:
        return {"jph_gpu_constraints", "Jolt/GPU/Shaders/rigid_constraints.slang", jolt_gpu_sources::Constraints};
    case JoltGpuProgram::BodyMutations:
        return {"jph_gpu_body_mutations", "Jolt/GPU/Shaders/rigid_body_mutations.slang",
                jolt_gpu_sources::BodyMutations};
    case JoltGpuProgram::BodyCommands:
        return {"jph_gpu_body_commands", "Jolt/GPU/Shaders/body_commands.slang", jolt_gpu_sources::BodyCommands};
    case JoltGpuProgram::Queries:
        return {"jph_gpu_queries", "Jolt/GPU/Shaders/rigid_queries.slang",
                std::string(jolt_gpu_sources::Queries) +
                    "\n#include \"../../../infernux/renderer/particle/collision_export.slang\"\n"};
    case JoltGpuProgram::RenderPoses:
        return {"jph_gpu_render_poses", "Jolt/GPU/Shaders/render_poses.slang", jolt_gpu_sources::RenderPoses};
    case JoltGpuProgram::Joints:
        return {"jph_gpu_joints", "Jolt/GPU/Shaders/joint_commands.slang", jolt_gpu_sources::JointCommands};
    }
    throw std::invalid_argument("Unknown builtin Jolt GPU program");
}

std::vector<ShaderSourceModule> GetJoltGpuProgramLibraries()
{
    return {{"jph_gpu_rigid_math", "Jolt/GPU/Shaders/rigid_math.slang", jolt_gpu_sources::Math},
            {"jph_gpu_physics_types", "Jolt/GPU/PhysicsTypes.h", jolt_gpu_sources::Types},
            {"jph_gpu_broadphase_types", "Jolt/GPU/BroadphaseTypes.h", jolt_gpu_sources::BroadphaseTypes},
            {"jph_gpu_constraint_types", "Jolt/GPU/ConstraintTypes.h", jolt_gpu_sources::ConstraintTypes},
            {"jph_gpu_island_owner", "Jolt/GPU/Shaders/island_owner_schedule.slang", jolt_gpu_sources::IslandOwner},
            {"jph_gpu_rigid_joints", "Jolt/GPU/Shaders/rigid_joints.slang", jolt_gpu_sources::Joints},
            {"jph_gpu_swing_twist", "Jolt/GPU/Shaders/rigid_swing_twist.slang", jolt_gpu_sources::SwingTwist},
            {"inx_particle_collision_export", "infernux/renderer/particle/collision_export.slang",
             jolt_gpu_sources::ParticleExport}};
}

} // namespace infernux
