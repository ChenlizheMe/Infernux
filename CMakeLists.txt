cmake_minimum_required(VERSION 3.22)
project(Infernux LANGUAGES CXX)

#------------------------------------------------------------------------------
# Linux: force -fPIC for all static libraries (fixes imgui linking error)
# ------------------------------------------------------------------------------
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ------------------------------------------------------------------------------
# C++ Standard
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------------------
# Paths and Directories
# ------------------------------------------------------------------------------
set(PYTHON_TARGET_DIR "${CMAKE_SOURCE_DIR}/python/Infernux/lib")

# ------------------------------------------------------------------------------
# Source Files
# ------------------------------------------------------------------------------
file(GLOB_RECURSE ENGINE_SOURCES CONFIGURE_DEPENDS cpp/*.cpp cpp/*.h)
# Exclude standalone launcher source from the pybind11 module
list(FILTER ENGINE_SOURCES EXCLUDE REGEX "tools/launcher/")
file(GLOB_RECURSE BINDING_SOURCES CONFIGURE_DEPENDS cpp/infernux/tools/pybinding/*.cpp)

# ------------------------------------------------------------------------------
# Header Include Directories
# ------------------------------------------------------------------------------
file(GLOB_RECURSE INFERNUX_HEADERS "${CMAKE_SOURCE_DIR}/cpp/*.h")
foreach(_hdr ${INFERNUX_HEADERS})
    get_filename_component(_hdr_dir "${_hdr}" DIRECTORY)
    list(APPEND INFERNUX_INCLUDE_DIRS "${_hdr_dir}")
endforeach()
list(REMOVE_DUPLICATES INFERNUX_INCLUDE_DIRS)

# ------------------------------------------------------------------------------
# pybind11 Module Definition
# ------------------------------------------------------------------------------
set(PYBIND11_FINDPYTHON ON)
find_package(Python3 COMPONENTS Interpreter Development.Module Development.Embed REQUIRED)

# Auto-discover pip-installed pybind11 so find_package works out of the box
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE _pybind11_cmakedir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _pybind11_result
)
if(_pybind11_result EQUAL 0 AND _pybind11_cmakedir)
    list(APPEND CMAKE_PREFIX_PATH "${_pybind11_cmakedir}")
endif()

find_package(pybind11 CONFIG REQUIRED)
find_package(Vulkan REQUIRED)

if(WIN32)
    include(InstallRequiredSystemLibraries)
endif()

pybind11_add_module(_Infernux MODULE
    ${ENGINE_SOURCES}
    ${BINDING_SOURCES}
)

target_include_directories(_Infernux PRIVATE
    ${INFERNUX_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/external
    ${CMAKE_SOURCE_DIR}/cpp/infernux
)

# Set UTF-8 encoding for MSVC to avoid C4819 warnings
if(MSVC)
    target_compile_options(_Infernux PRIVATE
        /utf-8
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/Ob2>
        $<$<CONFIG:Release>:/Oi>
        $<$<CONFIG:Release>:/Ot>
        $<$<CONFIG:Release>:/Oy>
        $<$<CONFIG:Release>:/Gy>
        $<$<CONFIG:Release>:/Gw>
        $<$<CONFIG:Release>:/GL>
    )

    target_link_options(_Infernux PRIVATE
        $<$<CONFIG:Release>:/LTCG>
        $<$<CONFIG:Release>:/OPT:REF>
        $<$<CONFIG:Release>:/OPT:ICF>
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(_Infernux PRIVATE
        $<$<CONFIG:Release>:-O3>
    )

    target_link_options(_Infernux PRIVATE
        $<$<CONFIG:Release>:-flto>
    )
endif()

# Prefer whole-program optimization in Release when the toolchain supports it.
set_property(TARGET _Infernux PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

# ------------------------------------------------------------------------------
# SPIRV-Cross (from Vulkan SDK)
# ------------------------------------------------------------------------------
# SPIRV-Cross is included with the Vulkan SDK for shader reflection
if(DEFINED ENV{VULKAN_SDK})
    set(VULKAN_SDK_PATH $ENV{VULKAN_SDK})
    target_include_directories(_Infernux PRIVATE ${VULKAN_SDK_PATH}/Include)
    
    # Link SPIRV-Cross libraries for shader reflection
    if(WIN32)
        target_link_libraries(_Infernux PRIVATE
            ${VULKAN_SDK_PATH}/Lib/spirv-cross-core.lib
            ${VULKAN_SDK_PATH}/Lib/spirv-cross-glsl.lib
            ${VULKAN_SDK_PATH}/Lib/spirv-cross-cpp.lib
            ${VULKAN_SDK_PATH}/Lib/spirv-cross-reflect.lib
        )
    else()
        # macOS / Linux: libraries live under lib/ with .a extension
        set(_spirv_cross_lib_dir "${VULKAN_SDK_PATH}/lib")
        foreach(_sc_lib spirv-cross-core spirv-cross-glsl spirv-cross-cpp spirv-cross-reflect)
            find_library(_found_${_sc_lib} NAMES ${_sc_lib} PATHS ${_spirv_cross_lib_dir} NO_DEFAULT_PATH)
            if(_found_${_sc_lib})
                target_link_libraries(_Infernux PRIVATE ${_found_${_sc_lib}})
            else()
                message(WARNING "SPIRV-Cross library ${_sc_lib} not found in ${_spirv_cross_lib_dir}")
            endif()
        endforeach()
    endif()
else()
    # VULKAN_SDK not set — try system paths (e.g. Homebrew on macOS)
    set(_spirv_cross_found TRUE)
    foreach(_sc_lib spirv-cross-core spirv-cross-glsl spirv-cross-cpp spirv-cross-reflect)
        find_library(_sys_${_sc_lib} NAMES ${_sc_lib})
        if(_sys_${_sc_lib})
            target_link_libraries(_Infernux PRIVATE ${_sys_${_sc_lib}})
        else()
            set(_spirv_cross_found FALSE)
        endif()
    endforeach()
    if(NOT _spirv_cross_found)
        message(WARNING "VULKAN_SDK not set and SPIRV-Cross libraries not found in system paths. Shader reflection will not be available.")
    endif()
endif()

# GLM: Use Vulkan-compatible [0,1] depth range instead of OpenGL [-1,1]
# GLM: Use left-handed coordinate system (Unity convention: X+=right, Y+=up, Z+=forward)
target_compile_definitions(_Infernux PRIVATE GLM_FORCE_DEPTH_ZERO_TO_ONE GLM_FORCE_LEFT_HANDED)

# Enable frame profiling instrumentation only in RelWithDebInfo builds.
# Terminal output (stderr) is OFF by default — enable with -DINFERNUX_FRAME_PROFILE_TERMINAL=1
# at CMake configure time if you need per-window dumps in the console.
#
# File logging is independent from profiling:
# - Debug / RelWithDebInfo: truncate Logs/engine.log on startup and write through.
# - Release / MinSizeRel: keep only the last 100 log lines in memory and write them on exit.
target_compile_definitions(_Infernux PRIVATE
    $<$<CONFIG:RelWithDebInfo>:INFERNUX_FRAME_PROFILE=1>
    $<$<CONFIG:Debug>:INFERNUX_FILE_LOGGING=1>
    $<$<CONFIG:RelWithDebInfo>:INFERNUX_FILE_LOGGING=1>
    $<$<CONFIG:Release>:INFERNUX_FILE_LOGGING=1>
    $<$<CONFIG:MinSizeRel>:INFERNUX_FILE_LOGGING=1>
    $<$<CONFIG:Release>:INFERNUX_DEFERRED_FILE_LOGGING=1>
    $<$<CONFIG:MinSizeRel>:INFERNUX_DEFERRED_FILE_LOGGING=1>
    # Vulkan validation layers: on for Debug / RelWithDebInfo (preset "debug"),
    # off for Release / MinSizeRel (preset "release"). See InxVkCoreModular::Init.
    $<$<CONFIG:Debug>:INFERNUX_VULKAN_VALIDATION_LAYERS=1>
    $<$<CONFIG:RelWithDebInfo>:INFERNUX_VULKAN_VALIDATION_LAYERS=1>
)

target_link_libraries(_Infernux PRIVATE
    assimp::assimp
    glslang::glslang
    SDL3::SDL3
    glm::glm
    imgui
    stb
    Vulkan::Vulkan
    Jolt
    GPUOpen::VulkanMemoryAllocator
)
# ------------------------------------------------------------------------------
# External Dependencies
# ------------------------------------------------------------------------------
add_subdirectory(external)

# ------------------------------------------------------------------------------
# Pre-Build Cleanup: remove python __pycache__ directories recursively
# ------------------------------------------------------------------------------
add_custom_target(clean_python_pycache
    COMMAND ${CMAKE_COMMAND}
            -DPYTHON_DIR=${CMAKE_SOURCE_DIR}/python
            -P ${CMAKE_SOURCE_DIR}/cmake/clean_python_pycache.cmake
    COMMENT "Removing __pycache__ directories under python/ before build"
)

# ------------------------------------------------------------------------------
# Post-Build Steps
# ------------------------------------------------------------------------------

# Create a custom script to copy all DLL dependencies
configure_file(
    "${CMAKE_SOURCE_DIR}/copy_dependencies.cmake.in"
    "${CMAKE_BINARY_DIR}/copy_dependencies.cmake"
    @ONLY
)

add_custom_command(TARGET _Infernux POST_BUILD
    # Copy the native module (.pyd on Windows, .so on macOS/Linux)
    COMMAND ${CMAKE_COMMAND} -E copy
        "$<TARGET_FILE:_Infernux>"
        "${PYTHON_TARGET_DIR}/$<TARGET_FILE_NAME:_Infernux>"
    
    COMMENT "Copy _Infernux native module to python/Infernux/lib"
)

if(WIN32)
    # Windows: copy all DLL dependencies automatically
    add_custom_command(TARGET _Infernux POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -DINFERNUX_BUILD_CFG=$<CONFIG>
            -P "${CMAKE_BINARY_DIR}/copy_dependencies.cmake"
        COMMENT "Copy dependent DLLs to python/Infernux/lib"
    )
elseif(APPLE)
    # macOS: copy shared libraries (.dylib) and fix RPATH
    add_custom_command(TARGET _Infernux POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -DINFERNUX_BUILD_CFG=$<CONFIG>
            -P "${CMAKE_BINARY_DIR}/copy_dependencies.cmake"
        COMMENT "Copy dependent dylibs to python/Infernux/lib"
    )
    # Set RPATH so the module finds dylibs next to itself
    set_target_properties(_Infernux PROPERTIES
        BUILD_RPATH "@loader_path"
        INSTALL_RPATH "@loader_path"
    )
else()
    # Linux: copy shared libraries (.so) and set RPATH
    add_custom_command(TARGET _Infernux POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -DINFERNUX_BUILD_CFG=$<CONFIG>
            -P "${CMAKE_BINARY_DIR}/copy_dependencies.cmake"
        COMMENT "Copy dependent shared libs to python/Infernux/lib"
    )
    set_target_properties(_Infernux PROPERTIES
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
    )
endif()

# ------------------------------------------------------------------------------
# Infernux Hub — PyInstaller packaging
# ------------------------------------------------------------------------------
# Build the standalone Infernux Hub launcher.
#   cmake --build --preset release --target infernux_hub
# Output: ${CMAKE_SOURCE_DIR}/dist/Infernux Hub/
# ------------------------------------------------------------------------------
add_custom_target(prepare_bundled_python_runtime
    COMMAND ${CMAKE_COMMAND} -E echo "Ensuring bundled Python 3.12 runtime is available..."
    COMMAND ${Python3_EXECUTABLE}
        "${CMAKE_SOURCE_DIR}/packaging/stage_bundled_python_runtime.py"
        --dest-root "${CMAKE_SOURCE_DIR}/packaging/runtime/python312"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Stage bundled full Python runtime assets"
)

add_custom_target(infernux_hub
    COMMAND ${CMAKE_COMMAND} -E echo "Packaging Infernux Hub with PyInstaller..."
    COMMAND ${Python3_EXECUTABLE} -m PyInstaller
        "${CMAKE_SOURCE_DIR}/packaging/infernux_hub.spec"
        --distpath "${CMAKE_SOURCE_DIR}/dist"
        --workpath "${CMAKE_BINARY_DIR}/hub_build"
        --clean
        --noconfirm
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/packaging"
    DEPENDS prepare_bundled_python_runtime
    COMMENT "Build Infernux Hub standalone application → dist/Infernux Hub/"
)

add_custom_target(infernux_hub_installer
    COMMAND ${CMAKE_COMMAND} -E echo "Building graphical Infernux Hub installer..."
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_SOURCE_DIR}/dist/installer"
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_SOURCE_DIR}/dist/installer_helper"
    COMMAND ${Python3_EXECUTABLE} -m PyInstaller
        "${CMAKE_SOURCE_DIR}/packaging/infernux_hub_installer.spec"
        --distpath "${CMAKE_SOURCE_DIR}/dist/installer"
        --workpath "${CMAKE_BINARY_DIR}/installer_build"
        --clean
        --noconfirm
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/packaging"
    DEPENDS infernux_hub
    COMMENT "Build graphical Infernux Hub installer → dist/installer/"
)

# ------------------------------------------------------------------------------
# Code Formatting Target (clang-format)
# ------------------------------------------------------------------------------
file(GLOB_RECURSE ALL_CXX_SOURCE_FILES 
    "${CMAKE_SOURCE_DIR}/cpp/*.cpp"
    "${CMAKE_SOURCE_DIR}/cpp/*.h"
    "${CMAKE_SOURCE_DIR}/cpp/*.hpp"
)

add_custom_target(clang_format
    COMMAND clang-format -i ${ALL_CXX_SOURCE_FILES}
    COMMENT "Running clang-format on source files"
)

# ------------------------------------------------------------------------------
# Python Packaging and Installation
# ------------------------------------------------------------------------------
add_custom_target(package_and_install_python ALL
    COMMAND ${CMAKE_COMMAND} -E echo "Ensuring Python packaging tools are available..."
    COMMAND ${CMAKE_COMMAND}
        -DINFERNUX_SOURCE_DIR=${CMAKE_SOURCE_DIR}
        -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}
        -P "${CMAKE_SOURCE_DIR}/cmake/ensure_python_packaging_tools.cmake"

    COMMAND ${CMAKE_COMMAND} -E echo "Building Python package via '${Python3_EXECUTABLE} -m build'..."
    COMMAND "${Python3_EXECUTABLE}" -m build --no-isolation --outdir "${CMAKE_SOURCE_DIR}/dist"

    COMMAND ${CMAKE_COMMAND} -E echo "Installing built wheel via pip..."
    COMMAND ${CMAKE_COMMAND}
        -DINFERNUX_SOURCE_DIR=${CMAKE_SOURCE_DIR}
        -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}
        -P "${CMAKE_SOURCE_DIR}/install_wheel.cmake"

    COMMAND ${CMAKE_COMMAND} -E echo "Cleaning up .egg-info directory..."
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${CMAKE_SOURCE_DIR}/python/Infernux.egg-info"

    COMMAND ${CMAKE_COMMAND} -E echo "Cleaning __pycache__ directories after packaging..."
    COMMAND ${CMAKE_COMMAND}
            -DPYTHON_DIR=${CMAKE_SOURCE_DIR}/python
            -P ${CMAKE_SOURCE_DIR}/cmake/clean_python_pycache.cmake

    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Python packaging and installation after native module build"
)

add_dependencies(package_and_install_python _Infernux)

# ------------------------------------------------------------------------------
# API Reference Generation (auto-introspect Python package → Markdown)
# ------------------------------------------------------------------------------
add_custom_target(generate_api_docs
    COMMAND ${CMAKE_COMMAND} -E echo "Generating API reference docs..."
    COMMAND ${CMAKE_COMMAND} -E env PYTHONIOENCODING=utf-8
            python "${CMAKE_SOURCE_DIR}/docs/wiki/generate_api_docs.py"
    COMMENT "Auto-generate API reference Markdown from Python stubs"
)

# ------------------------------------------------------------------------------
# Wiki Build (MkDocs)
# ------------------------------------------------------------------------------
add_custom_target(build_wiki_html
    COMMAND ${CMAKE_COMMAND} -E echo "Building MkDocs wiki..."
    COMMAND ${CMAKE_COMMAND} -E env PYTHONIOENCODING=utf-8
            python -m mkdocs build --clean -f "${CMAKE_SOURCE_DIR}/docs/wiki/mkdocs.yml"
    COMMENT "Generate wiki HTML into docs/wiki/site"
)
add_dependencies(build_wiki_html generate_api_docs)

# Ensure cleanup runs before any local build target compiles.
function(infernux_add_preclean_dependency _dir _dep_target)
    get_property(_targets DIRECTORY "${_dir}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_target IN LISTS _targets)
        if(_target STREQUAL "${_dep_target}")
            continue()
        endif()

        get_target_property(_is_imported ${_target} IMPORTED)
        if(_is_imported)
            continue()
        endif()

        get_target_property(_target_type ${_target} TYPE)
        if(_target_type STREQUAL "INTERFACE_LIBRARY")
            continue()
        endif()

        add_dependencies(${_target} ${_dep_target})
    endforeach()

    get_property(_subdirs DIRECTORY "${_dir}" PROPERTY SUBDIRECTORIES)
    foreach(_subdir IN LISTS _subdirs)
        infernux_add_preclean_dependency("${_subdir}" "${_dep_target}")
    endforeach()
endfunction()

infernux_add_preclean_dependency("${CMAKE_SOURCE_DIR}" clean_python_pycache)
