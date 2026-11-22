cmake_minimum_required(VERSION 3.25)
project(InfernuxBuiltinShaderCook LANGUAGES C CXX)

include("../InfernuxSlang.cmake")
include("../InfernuxPhysicsCook.cmake")
infernux_add_builtin_shader_cooker()
set_target_properties(InfernuxBuiltinShaderCook PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/bin>")
