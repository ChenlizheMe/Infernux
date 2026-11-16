cmake_minimum_required(VERSION 3.25)
project(InfernuxJoltGpuCook LANGUAGES C CXX)

include("../InfernuxSlang.cmake")
include("../InfernuxPhysicsCook.cmake")
infernux_add_jolt_gpu_cooker()
set_target_properties(InfernuxJoltGpuCook PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/bin>")
