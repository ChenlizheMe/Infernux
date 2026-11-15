# Embed the fork and engine interop sources into a target-private header.
function(infernux_embed_physics_sources target)
    get_filename_component(_engine_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
    set(_physics_root "${_engine_root}/external/joltphysics_for_infernux/Jolt/GPU")
    set(_particle_export "${_engine_root}/cpp/infernux/function/renderer/particle/shaders/collision_export.slang")
    set(_physics_sources
        "${_physics_root}/PhysicsTypes.h"
        "${_physics_root}/SleepTypes.h"
        "${_physics_root}/ArticulationIntegrationTypes.h"
        "${_physics_root}/StateFkTypes.h"
        "${_physics_root}/ArticulationDynamicsTypes.h"
        "${_physics_root}/Shaders/articulation_aba.slang"
        "${_physics_root}/Shaders/articulation_joint_load.slang"
        "${_physics_root}/Shaders/articulation_kinematics.slang"
        "${_physics_root}/Shaders/articulation_state_fk.slang"
        "${_physics_root}/Shaders/articulation_dynamics.slang"
        "${_physics_root}/Shaders/articulation_normal_capture.slang"
        "${_physics_root}/Shaders/articulation_normal_dynamics.slang"
        "${_physics_root}/ArticulationTypes.h"
        "${_physics_root}/ArticulationControlTypes.h"
        "${_physics_root}/Shaders/rigid_articulations.slang"
        "${_physics_root}/Shaders/rigid_sleep_math.slang"
        "${_physics_root}/Shaders/rigid_sleep.slang"
        "${_physics_root}/ConvexGeometryTypes.h"
        "${_physics_root}/Shaders/rigid_convex.slang"
        "${_physics_root}/Shaders/rigid_convex_narrowphase.slang"
        "${_physics_root}/Shaders/rigid_convex_manifold.slang"
        "${_physics_root}/Shaders/rigid_convex_queries.slang"
        "${_physics_root}/ContactReportTypes.h"
        "${_physics_root}/Shaders/rigid_contact_reports.slang"
        "${_physics_root}/JointPayloadFields.inl"
        "${_physics_root}/BroadphaseTypes.h"
        "${_physics_root}/ConstraintTypes.h"
        "${_physics_root}/Shaders/rigid_constraints.slang"
        "${_physics_root}/Shaders/island_owner_schedule.slang"
        "${_physics_root}/Shaders/rigid_broadphase.slang"
        "${_physics_root}/Shaders/rigid_math.slang"
        "${_physics_root}/Shaders/rigid_psd.slang"
        "${_physics_root}/Shaders/rigid_joints.slang"
        "${_physics_root}/Shaders/rigid_swing_twist.slang"
        "${_physics_root}/Shaders/rigid_six_dof.slang"
        "${_physics_root}/Shaders/joint_commands.slang"
        "${_physics_root}/Shaders/rigid_step.slang"
        "${_physics_root}/Shaders/rigid_body_mutations.slang"
        "${_physics_root}/Shaders/body_commands.slang"
        "${_physics_root}/Shaders/rigid_queries.slang"
        "${_physics_root}/Shaders/render_poses.slang"
        "${_particle_export}"
    )
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_physics_sources})

    file(READ "${_physics_root}/PhysicsTypes.h" JOLT_GPU_TYPES_SOURCE)
    file(READ "${_physics_root}/SleepTypes.h" JOLT_GPU_SLEEP_TYPES_SOURCE)
    file(READ "${_physics_root}/ArticulationIntegrationTypes.h" JOLT_GPU_ART_INTEGRATION_TYPES_SOURCE)
    file(READ "${_physics_root}/StateFkTypes.h" JOLT_GPU_STATE_FK_TYPES_SOURCE)
    file(READ "${_physics_root}/ArticulationDynamicsTypes.h" JOLT_GPU_ART_DYNAMICS_TYPES_SOURCE)
    file(READ "${_physics_root}/Shaders/articulation_aba.slang" JOLT_GPU_ART_ABA_SOURCE)
    file(READ "${_physics_root}/Shaders/articulation_joint_load.slang" JOLT_GPU_ART_JOINT_LOAD_SOURCE)
    file(READ "${_physics_root}/Shaders/articulation_kinematics.slang" JOLT_GPU_ART_KINEMATICS_SOURCE)
    file(READ "${_physics_root}/Shaders/articulation_state_fk.slang" JOLT_GPU_ART_STATE_FK_SOURCE)
    file(READ "${_physics_root}/Shaders/articulation_dynamics.slang" JOLT_GPU_ART_DYNAMICS_SOURCE)
    file(READ "${_physics_root}/Shaders/articulation_normal_capture.slang" JOLT_GPU_ART_NORMAL_CAPTURE_SOURCE)
    file(READ "${_physics_root}/Shaders/articulation_normal_dynamics.slang" JOLT_GPU_ART_NORMAL_DYNAMICS_SOURCE)
    file(READ "${_physics_root}/ArticulationTypes.h" JOLT_GPU_ARTICULATION_TYPES_SOURCE)
    file(READ "${_physics_root}/ArticulationControlTypes.h" JOLT_GPU_ARTICULATION_CONTROL_SOURCE)
    file(READ "${_physics_root}/Shaders/rigid_articulations.slang" JOLT_GPU_ARTICULATIONS_SOURCE)
    file(READ "${_physics_root}/Shaders/rigid_sleep_math.slang" JOLT_GPU_SLEEP_MATH_SOURCE)
    file(READ "${_physics_root}/Shaders/rigid_sleep.slang" JOLT_GPU_SLEEP_SOURCE)
    file(READ "${_physics_root}/ConvexGeometryTypes.h" JOLT_GPU_CONVEX_TYPES_SOURCE)
    file(READ "${_physics_root}/Shaders/rigid_convex_narrowphase.slang" JOLT_GPU_NARROWPHASE_SOURCE)
    file(READ "${_physics_root}/Shaders/rigid_convex.slang" JOLT_GPU_CONVEX_SOURCE)
    file(READ "${_physics_root}/Shaders/rigid_convex_manifold.slang" JOLT_GPU_CONVEX_MANIFOLD_SOURCE)
    file(READ "${_physics_root}/Shaders/rigid_convex_queries.slang" JOLT_GPU_CONVEX_QUERIES_SOURCE)
    file(READ "${_physics_root}/ContactReportTypes.h" JOLT_GPU_CONTACT_REPORT_TYPES_SOURCE)
    file(READ "${_physics_root}/Shaders/rigid_contact_reports.slang" JOLT_GPU_CONTACT_REPORT_SOURCE)
    file(READ "${_physics_root}/JointPayloadFields.inl" JOLT_GPU_JOINT_FIELDS_SOURCE)
    file(READ "${_physics_root}/BroadphaseTypes.h" JOLT_GPU_BROADPHASE_TYPES_SOURCE)
    file(READ "${_physics_root}/ConstraintTypes.h" JOLT_GPU_CONSTRAINT_TYPES_SOURCE)
    file(READ "${_physics_root}/Shaders/rigid_constraints.slang" JOLT_GPU_CONSTRAINT_SOURCE)
    file(READ "${_physics_root}/Shaders/island_owner_schedule.slang" JOLT_GPU_ISLAND_OWNER_SOURCE)
    file(READ "${_physics_root}/Shaders/rigid_broadphase.slang" JOLT_GPU_BROADPHASE_SOURCE)
    file(READ "${_physics_root}/Shaders/rigid_math.slang" JOLT_GPU_MATH_SOURCE)
    file(READ "${_physics_root}/Shaders/rigid_psd.slang" JOLT_GPU_PSD_SOURCE)
    file(READ "${_physics_root}/Shaders/rigid_joints.slang" JOLT_GPU_JOINTS_SOURCE)
    file(READ "${_physics_root}/Shaders/rigid_swing_twist.slang" JOLT_GPU_SWING_TWIST_SOURCE)
    file(READ "${_physics_root}/Shaders/rigid_six_dof.slang" JOLT_GPU_SIX_DOF_SOURCE)
    file(READ "${_physics_root}/Shaders/joint_commands.slang" JOLT_GPU_JOINT_COMMANDS_SOURCE)
    file(READ "${_physics_root}/Shaders/rigid_step.slang" JOLT_GPU_STEP_SOURCE)
    file(READ "${_physics_root}/Shaders/rigid_body_mutations.slang" JOLT_GPU_BODY_MUTATIONS_SOURCE)
    file(READ "${_physics_root}/Shaders/body_commands.slang" JOLT_GPU_BODY_COMMANDS_SOURCE)
    file(READ "${_physics_root}/Shaders/rigid_queries.slang" JOLT_GPU_QUERIES_SOURCE)
    file(READ "${_physics_root}/Shaders/render_poses.slang" JOLT_GPU_RENDER_POSES_SOURCE)
    file(READ "${_particle_export}" INFERNUX_PARTICLE_EXPORT_SOURCE)

    # Keep individual literals below MSVC's limit without changing the source
    # text or introducing runtime file access. Adjacent literals concatenate.
    foreach(_source IN ITEMS JOLT_GPU_ART_INTEGRATION_TYPES_SOURCE JOLT_GPU_STATE_FK_TYPES_SOURCE JOLT_GPU_ART_DYNAMICS_TYPES_SOURCE JOLT_GPU_ART_ABA_SOURCE JOLT_GPU_ART_JOINT_LOAD_SOURCE JOLT_GPU_ART_KINEMATICS_SOURCE JOLT_GPU_ART_STATE_FK_SOURCE JOLT_GPU_ART_DYNAMICS_SOURCE JOLT_GPU_ART_NORMAL_CAPTURE_SOURCE JOLT_GPU_ART_NORMAL_DYNAMICS_SOURCE JOLT_GPU_ARTICULATION_TYPES_SOURCE JOLT_GPU_ARTICULATION_CONTROL_SOURCE JOLT_GPU_ARTICULATIONS_SOURCE JOLT_GPU_SLEEP_SOURCE JOLT_GPU_SLEEP_TYPES_SOURCE JOLT_GPU_SLEEP_MATH_SOURCE JOLT_GPU_NARROWPHASE_SOURCE JOLT_GPU_CONVEX_TYPES_SOURCE JOLT_GPU_CONVEX_SOURCE JOLT_GPU_CONVEX_MANIFOLD_SOURCE JOLT_GPU_CONVEX_QUERIES_SOURCE JOLT_GPU_CONTACT_REPORT_TYPES_SOURCE JOLT_GPU_CONTACT_REPORT_SOURCE JOLT_GPU_TYPES_SOURCE JOLT_GPU_JOINT_FIELDS_SOURCE JOLT_GPU_MATH_SOURCE JOLT_GPU_PSD_SOURCE JOLT_GPU_STEP_SOURCE
            JOLT_GPU_BROADPHASE_TYPES_SOURCE JOLT_GPU_BROADPHASE_SOURCE
            JOLT_GPU_CONSTRAINT_TYPES_SOURCE JOLT_GPU_CONSTRAINT_SOURCE JOLT_GPU_ISLAND_OWNER_SOURCE
            JOLT_GPU_BODY_MUTATIONS_SOURCE JOLT_GPU_BODY_COMMANDS_SOURCE
            JOLT_GPU_JOINTS_SOURCE JOLT_GPU_SIX_DOF_SOURCE JOLT_GPU_SWING_TWIST_SOURCE JOLT_GPU_JOINT_COMMANDS_SOURCE
            JOLT_GPU_QUERIES_SOURCE JOLT_GPU_RENDER_POSES_SOURCE INFERNUX_PARTICLE_EXPORT_SOURCE)
        string(REPLACE "\n" "\n)INX_JPH_GPU\"\nR\"INX_JPH_GPU(" ${_source} "${${_source}}")
    endforeach()

    set(_generated_include_root "${CMAKE_CURRENT_BINARY_DIR}/generated")
    configure_file(
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/templates/JoltGpuProgramSources.h.in"
        "${_generated_include_root}/infernux/physics/JoltGpuEmbeddedSources.h"
        @ONLY
    )
    target_include_directories(${target} PRIVATE "${_generated_include_root}")
endfunction()
