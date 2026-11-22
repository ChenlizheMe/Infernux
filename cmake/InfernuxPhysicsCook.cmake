include_guard(GLOBAL)

function(infernux_add_jolt_gpu_cooker)
    get_filename_component(_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
    add_executable(InfernuxJoltGpuCook "${_root}/cpp/apps/cook/JoltGpuCook.cpp")
    target_compile_features(InfernuxJoltGpuCook PRIVATE cxx_std_17)
    target_include_directories(InfernuxJoltGpuCook PRIVATE
        "${_root}/cpp/infernux" "${_root}/external")

    if(TARGET InfernuxShaderCompiler)
        target_link_libraries(InfernuxJoltGpuCook PRIVATE InfernuxShaderCompiler)
    else()
        # A cross build needs only the native compiler, not a second engine SDK.
        target_sources(InfernuxJoltGpuCook PRIVATE
            "${_root}/cpp/infernux/core/types/ShaderBuildStatistics.cpp"
            "${_root}/cpp/infernux/function/resources/ShaderAsset/SlangCompiler.cpp"
            "${_root}/cpp/infernux/function/resources/ShaderAsset/SlangLayout.cpp")
        target_compile_definitions(InfernuxJoltGpuCook PRIVATE
            INFERNUX_FOUNDATION_EXPORTS=1 INFERNUX_SHADER_COMPILER_EXPORTS=1)
        target_link_libraries(InfernuxJoltGpuCook PRIVATE ${INFERNUX_SLANG_TARGET})
        infernux_add_rt_slang_module(InfernuxJoltGpuCook)
    endif()
    if(MSVC)
        target_compile_options(InfernuxJoltGpuCook PRIVATE /utf-8)
    endif()
    if(WIN32)
        add_custom_command(TARGET InfernuxJoltGpuCook POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                $<TARGET_RUNTIME_DLLS:InfernuxJoltGpuCook> "$<TARGET_FILE_DIR:InfernuxJoltGpuCook>"
            COMMAND_EXPAND_LISTS VERBATIM)
    endif()
endfunction()

function(infernux_get_shader_cooker executable build_target)
    if(CMAKE_CROSSCOMPILING)
        include(ExternalProject)
        set(_host_build "${CMAKE_BINARY_DIR}/jolt-gpu-cooker")
        set(_cooker "${_host_build}/bin/InfernuxJoltGpuCook")
        if(CMAKE_HOST_WIN32)
            string(APPEND _cooker ".exe")
        endif()
        if(NOT TARGET infernux_jolt_gpu_host_cooker)
            ExternalProject_Add(infernux_jolt_gpu_host_cooker
                SOURCE_DIR "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/jolt-gpu-cook"
                BINARY_DIR "${_host_build}"
                DOWNLOAD_COMMAND "" UPDATE_COMMAND ""
                CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DFETCHCONTENT_FULLY_DISCONNECTED=ON
                BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR>
                    --config Release --target InfernuxJoltGpuCook --parallel 2
                BUILD_BYPRODUCTS "${_cooker}"
                BUILD_ALWAYS TRUE
                INSTALL_COMMAND "")
        endif()
        set(_cooker_target infernux_jolt_gpu_host_cooker)
    else()
        if(NOT TARGET InfernuxJoltGpuCook)
            infernux_add_jolt_gpu_cooker()
        endif()
        set(_cooker "$<TARGET_FILE:InfernuxJoltGpuCook>")
        set(_cooker_target InfernuxJoltGpuCook)
    endif()

    set(${executable} "${_cooker}" PARENT_SCOPE)
    set(${build_target} "${_cooker_target}" PARENT_SCOPE)
endfunction()

function(infernux_cook_program_archive target format group filename)
    get_filename_component(_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
    infernux_get_shader_cooker(_cooker _cooker_target)
    set(_directory "${CMAKE_CURRENT_BINARY_DIR}/generated/${group}/${format}/$<CONFIG>")
    set(_archive "${_directory}/${filename}")
    add_custom_command(OUTPUT "${_archive}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_directory}"
        COMMAND "${CMAKE_COMMAND}" -E chdir "${_directory}"
            "${_cooker}" "${group}" "${_root}" "${format}" "${filename}.tmp"
        COMMAND "${CMAKE_COMMAND}" -E rename "${_archive}.tmp" "${_archive}"
        DEPENDS ${_cooker_target} "${_cooker}" ${ARGN}
        COMMENT "Cook built-in ${group} programs to ${format}"
        VERBATIM)
    target_sources(${target} PRIVATE "${_archive}")
    target_compile_features(${target} PRIVATE c_std_11)
endfunction()

function(infernux_cook_jolt_gpu_programs target format)
    get_filename_component(_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
    set(_physics_gpu "${_root}/cpp/infernux/function/scene/physics/gpu")
    file(GLOB_RECURSE _shader_sources CONFIGURE_DEPENDS
        "${_physics_gpu}/*.slang" "${_physics_gpu}/*.h" "${_physics_gpu}/*.inl")
    infernux_cook_program_archive(${target} ${format} physics JoltGpuArchive.c ${_shader_sources})
endfunction()

function(infernux_cook_inxphygpu_programs target format)
    set(_inxphygpu "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../cpp/infernux/function/scene/physics/inxphygpu")
    file(GLOB _shader_sources CONFIGURE_DEPENDS "${_inxphygpu}/*.slang")
    infernux_cook_program_archive(${target} ${format} inxphygpu InxPhyGpuArchive.c ${_shader_sources}
        "${_inxphygpu}/InxPhyGpuProgramSources.h")
endfunction()

function(infernux_cook_ui_programs target format)
    infernux_cook_program_archive(${target} ${format} ui UIShaderArchive.c
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../cpp/infernux/function/resources/ShaderAsset/SlangUIModule.h")
endfunction()

function(infernux_cook_scene_programs target format)
    set(_shaders "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../cpp/infernux/function/renderer/shaders")
    infernux_cook_program_archive(${target} ${format} scene SceneShaderArchive.c
        "${_shaders}/scene_depth_resolve.slang" "${_shaders}/forward_plus_grid.slang")
endfunction()
