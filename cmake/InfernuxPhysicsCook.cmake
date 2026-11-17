include_guard(GLOBAL)

function(infernux_add_jolt_gpu_cooker)
    get_filename_component(_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
    add_executable(InfernuxJoltGpuCook "${_root}/cpp/apps/cook/JoltGpuCook.cpp")
    target_compile_features(InfernuxJoltGpuCook PRIVATE cxx_std_17)
    target_include_directories(InfernuxJoltGpuCook PRIVATE
        "${_root}/cpp/infernux" "${_root}/external" "${_root}/external/joltphysics_for_infernux")

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

function(infernux_cook_jolt_gpu_programs target format)
    get_filename_component(_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
    set(_jolt_gpu "${_root}/external/joltphysics_for_infernux/Jolt/GPU")
    file(GLOB_RECURSE _shader_sources CONFIGURE_DEPENDS
        "${_root}/cpp/infernux/function/scene/physics/shaders/*.slang"
        "${_jolt_gpu}/*.slang" "${_jolt_gpu}/*.h" "${_jolt_gpu}/*.inl")
    list(APPEND _shader_sources "${_root}/cpp/infernux/function/renderer/particle/shaders/collision_export.slang")
    if(CMAKE_CROSSCOMPILING)
        include(ExternalProject)
        set(INFERNUX_HOST_SLANG_ROOT "" CACHE PATH "Prepared native Slang SDK for cross-build shader cooking")
        set(_host_build "${CMAKE_BINARY_DIR}/jolt-gpu-cooker")
        set(_cooker "${_host_build}/bin/InfernuxJoltGpuCook")
        if(CMAKE_HOST_WIN32)
            string(APPEND _cooker ".exe")
        endif()
        ExternalProject_Add(infernux_jolt_gpu_host_cooker
            SOURCE_DIR "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/jolt-gpu-cook"
            BINARY_DIR "${_host_build}"
            CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release "-DINFERNUX_SLANG_ROOT=${INFERNUX_HOST_SLANG_ROOT}"
            BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR>
                --config Release --target InfernuxJoltGpuCook --parallel 2
            BUILD_BYPRODUCTS "${_cooker}"
            BUILD_ALWAYS TRUE
            INSTALL_COMMAND "")
        set(_cooker_target infernux_jolt_gpu_host_cooker)
    else()
        infernux_add_jolt_gpu_cooker()
        set(_cooker "$<TARGET_FILE:InfernuxJoltGpuCook>")
        set(_cooker_target InfernuxJoltGpuCook)
    endif()

    set(_directory "${CMAKE_CURRENT_BINARY_DIR}/generated/physics/${format}/$<CONFIG>")
    set(_archive "${_directory}/JoltGpuArchive.c")
    add_custom_command(OUTPUT "${_archive}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_directory}"
        COMMAND "${CMAKE_COMMAND}" -E chdir "${_directory}"
            "${_cooker}" "${_root}" "${format}" JoltGpuArchive.c.tmp
        COMMAND "${CMAKE_COMMAND}" -E rename "${_archive}.tmp" "${_archive}"
        DEPENDS ${_cooker_target} "${_cooker}" ${_shader_sources}
        COMMENT "Cook built-in GPU physics programs to ${format}"
        VERBATIM)
    target_sources(${target} PRIVATE "${_archive}")
    target_compile_features(${target} PRIVATE c_std_11)
endfunction()
