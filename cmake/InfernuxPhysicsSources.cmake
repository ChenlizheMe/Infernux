# Embed the authoritative fork sources into a target-private runtime header.
function(infernux_embed_physics_sources target)
    get_filename_component(_engine_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
    set(_physics_root "${_engine_root}/external/joltphysics_for_infernux/Jolt/GPU")
    set(_physics_sources
        "${_physics_root}/PhysicsTypes.h"
        "${_physics_root}/Shaders/rigid_math.slang"
        "${_physics_root}/Shaders/rigid_step.slang"
        "${_physics_root}/Shaders/rigid_body_mutations.slang"
        "${_physics_root}/Shaders/body_commands.slang"
        "${_physics_root}/Shaders/rigid_queries.slang"
        "${_physics_root}/Shaders/render_poses.slang"
    )
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_physics_sources})

    file(READ "${_physics_root}/PhysicsTypes.h" JOLT_GPU_TYPES_SOURCE)
    file(READ "${_physics_root}/Shaders/rigid_math.slang" JOLT_GPU_MATH_SOURCE)
    file(READ "${_physics_root}/Shaders/rigid_step.slang" JOLT_GPU_STEP_SOURCE)
    file(READ "${_physics_root}/Shaders/rigid_body_mutations.slang" JOLT_GPU_BODY_MUTATIONS_SOURCE)
    file(READ "${_physics_root}/Shaders/body_commands.slang" JOLT_GPU_BODY_COMMANDS_SOURCE)
    file(READ "${_physics_root}/Shaders/rigid_queries.slang" JOLT_GPU_QUERIES_SOURCE)
    file(READ "${_physics_root}/Shaders/render_poses.slang" JOLT_GPU_RENDER_POSES_SOURCE)

    # Keep individual literals below MSVC's limit without changing the source
    # text or introducing runtime file access. Adjacent literals concatenate.
    foreach(_source IN ITEMS JOLT_GPU_TYPES_SOURCE JOLT_GPU_MATH_SOURCE JOLT_GPU_STEP_SOURCE
            JOLT_GPU_BODY_MUTATIONS_SOURCE JOLT_GPU_BODY_COMMANDS_SOURCE
            JOLT_GPU_QUERIES_SOURCE JOLT_GPU_RENDER_POSES_SOURCE)
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
