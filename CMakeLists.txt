cmake_minimum_required(VERSION 3.25)

# Minimum only, not an upper bound: newer CMake (including 4.x) is supported.
# Preset schema 6 and workflow presets require 3.25; lowering this alone would
# advertise a toolchain that cannot read the checked-in build entry points.

# Resolve platform compiler companion tools before project() performs its IPO
# probe. Ubuntu installs versioned LLVM binaries without always publishing the
# unversioned llvm-ar/llvm-ranlib names expected by CMake's Clang integration.
include("${CMAKE_CURRENT_LIST_DIR}/cmake/InfernuxCompilerTools.cmake")

project(Infernux LANGUAGES C CXX)

# Root orchestration only. Target ownership and packaging policy live in the
# focused modules below so platform bring-up does not keep expanding this file.
include("${CMAKE_SOURCE_DIR}/cmake/InfernuxBuildOptions.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/InfernuxSources.cmake")

add_subdirectory(external)

include("${CMAKE_SOURCE_DIR}/cmake/InfernuxNativeTargets.cmake")
if(INFERNUX_BUILD_TESTS AND Python3_EXECUTABLE)
    add_custom_target(infernux_rhi_boundary_check
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/cmake/check_rhi_boundary.py"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        VERBATIM)
endif()
include("${CMAKE_SOURCE_DIR}/cmake/InfernuxGpuJit.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/InfernuxPlayerHost.cmake")
add_subdirectory(external/plugins)
include("${CMAKE_SOURCE_DIR}/tests/native/CMakeLists.txt")
include("${CMAKE_SOURCE_DIR}/cmake/InfernuxSceneIdentityTests.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/InfernuxInstall.cmake")
include("${CMAKE_SOURCE_DIR}/external/plugins/infernux_android/native/CMakeLists.txt")
include("${CMAKE_SOURCE_DIR}/cmake/InfernuxHub.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/InfernuxDeveloperTools.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/InfernuxPackaging.cmake")
