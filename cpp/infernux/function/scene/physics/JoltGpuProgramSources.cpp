#include "JoltGpuProgramSources.h"

#include <Jolt/GPU/PhysicsStages.h>
#include <function/renderer/rhi/RhiComputeKernel.h>
#include <infernux/physics/JoltGpuEmbeddedSources.h>

namespace infernux
{

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

ShaderSourceModule GetJoltGpuProgramSource(JoltGpuProgram program)
{
    switch (program) {
    case JoltGpuProgram::ArticulationCapture:
        return {"jph_gpu_articulation_capture", "Jolt/GPU/Shaders/articulation_normal_capture.slang",
                jolt_gpu_sources::ArticulationNormalCapture};
    case JoltGpuProgram::ArticulationDynamics:
        return {"jph_gpu_articulation_dynamics_owner", "Jolt/GPU/Shaders/articulation_normal_dynamics.slang",
                jolt_gpu_sources::ArticulationNormalDynamics};
    case JoltGpuProgram::Step:
        return {JPH::GPU::StepModuleName, JPH::GPU::StepSourcePath, jolt_gpu_sources::Step};
    case JoltGpuProgram::Broadphase:
        return {"jph_gpu_broadphase", "Jolt/GPU/Shaders/rigid_broadphase.slang", jolt_gpu_sources::Broadphase};
    case JoltGpuProgram::Narrowphase:
        return {"jph_gpu_narrowphase", "Jolt/GPU/Shaders/rigid_convex_narrowphase.slang",
                jolt_gpu_sources::Narrowphase};
    case JoltGpuProgram::ContactReports:
        return {"jph_gpu_contact_reports", "Jolt/GPU/Shaders/rigid_contact_reports.slang",
                jolt_gpu_sources::ContactReports};
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
    return {
        {"jph_gpu_rigid_math", "Jolt/GPU/Shaders/rigid_math.slang", jolt_gpu_sources::Math},
        {"jph_gpu_joint_fields", "Jolt/GPU/JointPayloadFields.inl", jolt_gpu_sources::JointFields},
        {"jph_gpu_physics_types", "Jolt/GPU/PhysicsTypes.h", jolt_gpu_sources::Types},
        {"jph_gpu_articulation_integration_types", "Jolt/GPU/ArticulationIntegrationTypes.h",
         jolt_gpu_sources::ArticulationIntegrationTypes},
        {"jph_gpu_state_fk_types", "Jolt/GPU/StateFkTypes.h", jolt_gpu_sources::StateFkTypes},
        {"jph_gpu_articulation_dynamics_types", "Jolt/GPU/ArticulationDynamicsTypes.h",
         jolt_gpu_sources::ArticulationDynamicsTypes},
        {"jph_gpu_articulation_aba", "Jolt/GPU/Shaders/articulation_aba.slang", jolt_gpu_sources::ArticulationAba},
        {"jph_gpu_articulation_joint_load", "Jolt/GPU/Shaders/articulation_joint_load.slang",
         jolt_gpu_sources::ArticulationJointLoad},
        {"jph_gpu_articulation_kinematics", "Jolt/GPU/Shaders/articulation_kinematics.slang",
         jolt_gpu_sources::ArticulationKinematics},
        {"jph_gpu_articulation_state_fk", "Jolt/GPU/Shaders/articulation_state_fk.slang",
         jolt_gpu_sources::ArticulationStateFk},
        {"jph_gpu_articulation_dynamics", "Jolt/GPU/Shaders/articulation_dynamics.slang",
         jolt_gpu_sources::ArticulationDynamics},
        {"jph_gpu_articulation_types", "Jolt/GPU/ArticulationTypes.h", jolt_gpu_sources::ArticulationTypes},
        {"jph_gpu_articulation_control", "Jolt/GPU/ArticulationControlTypes.h",
         jolt_gpu_sources::ArticulationControlTypes},
        {"jph_gpu_articulations", "Jolt/GPU/Shaders/rigid_articulations.slang", jolt_gpu_sources::Articulations},
        {"jph_gpu_sleep_types", "Jolt/GPU/SleepTypes.h", jolt_gpu_sources::SleepTypes},
        {"jph_gpu_sleep_math", "Jolt/GPU/Shaders/rigid_sleep_math.slang", jolt_gpu_sources::SleepMath},
        {"jph_gpu_sleep", "Jolt/GPU/Shaders/rigid_sleep.slang", jolt_gpu_sources::Sleep},
        {"jph_gpu_convex_types", "Jolt/GPU/ConvexGeometryTypes.h", jolt_gpu_sources::ConvexTypes},
        {"jph_gpu_convex", "Jolt/GPU/Shaders/rigid_convex.slang", jolt_gpu_sources::Convex},
        {"jph_gpu_convex_manifold", "Jolt/GPU/Shaders/rigid_convex_manifold.slang", jolt_gpu_sources::ConvexManifold},
        {"jph_gpu_convex_queries", "Jolt/GPU/Shaders/rigid_convex_queries.slang", jolt_gpu_sources::ConvexQueries},
        {"jph_gpu_contact_report_types", "Jolt/GPU/ContactReportTypes.h", jolt_gpu_sources::ContactReportTypes},
        {"jph_gpu_broadphase_types", "Jolt/GPU/BroadphaseTypes.h", jolt_gpu_sources::BroadphaseTypes},
        {"jph_gpu_constraint_types", "Jolt/GPU/ConstraintTypes.h", jolt_gpu_sources::ConstraintTypes},
        {"jph_gpu_island_owner", "Jolt/GPU/Shaders/island_owner_schedule.slang", jolt_gpu_sources::IslandOwner},
        {"jph_gpu_rigid_psd", "Jolt/GPU/Shaders/rigid_psd.slang", jolt_gpu_sources::Psd},
        {"jph_gpu_rigid_joints", "Jolt/GPU/Shaders/rigid_joints.slang", jolt_gpu_sources::Joints},
        {"jph_gpu_swing_twist", "Jolt/GPU/Shaders/rigid_swing_twist.slang", jolt_gpu_sources::SwingTwist},
        {"jph_gpu_six_dof", "Jolt/GPU/Shaders/rigid_six_dof.slang", jolt_gpu_sources::SixDOF},
        {"inx_particle_collision_export", "infernux/renderer/particle/collision_export.slang",
         jolt_gpu_sources::ParticleExport}};
}

} // namespace infernux
