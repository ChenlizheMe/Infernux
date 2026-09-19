# Engine-owned GPU JIT compiler build and wheel staging.
#
# The compiler fork is kept in an isolated CMake build because upstream Taichi
# still owns broad, process-global CMake variables.  It nevertheless remains a
# normal dependency of the engine wheel: no release script or manual copy step
# is involved.

include_guard(GLOBAL)
include(ExternalProject)

option(INFERNUX_BUILD_GPU_JIT_COMPILER
    "Build the internal Vulkan JIT compiler shipped in the Infernux wheel"
    ON
)

if(NOT INFERNUX_BUILD_GPU_JIT_COMPILER)
    return()
endif()

if(CMAKE_CROSSCOMPILING OR ANDROID OR EMSCRIPTEN)
    message(FATAL_ERROR
        "The editor GPU JIT compiler is a host-wheel dependency and cannot be built in a target cross compile")
endif()

set(INFERNUX_GPU_JIT_SOURCE_DIR
    "${CMAKE_SOURCE_DIR}/external/taichi_for_infernux")
if(NOT EXISTS "${INFERNUX_GPU_JIT_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "Missing external/taichi_for_infernux. Initialize the registered submodule before configuring Infernux")
endif()

set(INFERNUX_GPU_JIT_BINARY_DIR
    "${CMAKE_BINARY_DIR}/taichi-for-infernux")
set(INFERNUX_GPU_JIT_INSTALL_ROOT
    "${CMAKE_BINARY_DIR}/gpu-jit-wheel")

set(_infernux_gpu_jit_cmake_args
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_INSTALL_PREFIX=${INFERNUX_GPU_JIT_INSTALL_ROOT}"
    "-DPython_EXECUTABLE=${Python3_EXECUTABLE}"
    "-DTI_WITH_PYTHON=ON"
    "-DTI_WITH_C_API=OFF"
    "-DTI_WITH_STATIC_C_API=OFF"
    "-DTI_WITH_LLVM=OFF"
    "-DTI_WITH_CUDA=OFF"
    "-DTI_WITH_CUDA_TOOLKIT=OFF"
    "-DTI_WITH_AMDGPU=OFF"
    "-DTI_WITH_METAL=OFF"
    "-DTI_WITH_OPENGL=OFF"
    "-DTI_WITH_VULKAN=OFF"
    "-DTI_WITH_DX11=OFF"
    "-DTI_WITH_DX12=OFF"
    "-DTI_WITH_GGUI=OFF"
    "-DTI_BUILD_TESTS=OFF"
    "-DTI_BUILD_EXAMPLES=OFF"
    "-DTI_BUILD_RHI_EXAMPLES=OFF"
    "-DINFERNUX_BUILD_TESTS=${INFERNUX_BUILD_TESTS}"
)

# The compiler module is part of the engine wheel and must share the host
# toolchain selected by the root preset.  Leaving the isolated project to pick
# a compiler from PATH made Linux Clang builds silently configure it with GCC.
if(CMAKE_C_COMPILER)
    list(APPEND _infernux_gpu_jit_cmake_args
        "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
endif()
if(CMAKE_CXX_COMPILER)
    list(APPEND _infernux_gpu_jit_cmake_args
        "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
endif()

set(_infernux_gpu_jit_targets _infernux_gpu_compiler)
if(INFERNUX_BUILD_TESTS)
    list(APPEND _infernux_gpu_jit_targets infernux_compiler_contract_tests infernux_bit_contract_tests)
endif()

ExternalProject_Add(infernux_gpu_jit_compiler
    SOURCE_DIR "${INFERNUX_GPU_JIT_SOURCE_DIR}"
    BINARY_DIR "${INFERNUX_GPU_JIT_BINARY_DIR}"
    CMAKE_ARGS ${_infernux_gpu_jit_cmake_args}
    BUILD_COMMAND
        "${CMAKE_COMMAND}" --build <BINARY_DIR>
        --config Release --target ${_infernux_gpu_jit_targets} --parallel 4
    INSTALL_COMMAND
        "${CMAKE_COMMAND}" -E rm -rf "${INFERNUX_GPU_JIT_INSTALL_ROOT}"
    COMMAND "${CMAKE_COMMAND}" --install <BINARY_DIR>
        --config Release
        --prefix "${INFERNUX_GPU_JIT_INSTALL_ROOT}"
        --component infernux_compiler
    BUILD_ALWAYS TRUE
    USES_TERMINAL_BUILD TRUE
    USES_TERMINAL_INSTALL TRUE
)

set(INFERNUX_GPU_JIT_INSTALL_ROOT
    "${INFERNUX_GPU_JIT_INSTALL_ROOT}" CACHE INTERNAL
    "Staged GPU JIT compiler root consumed by the PythonWheel component")
