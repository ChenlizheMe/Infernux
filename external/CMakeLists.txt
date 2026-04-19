cmake_minimum_required(VERSION 3.17)

# ------------------------------------------------------------------------------
# Build Configuration
# ------------------------------------------------------------------------------
if(NOT CMAKE_CONFIGURATION_TYPES)
    set(_saved_build_type "${CMAKE_BUILD_TYPE}")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(_saved_shared "${BUILD_SHARED_LIBS}")
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)

# ------------------------------------------------------------------------------
# Global Options
# ------------------------------------------------------------------------------
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

# assimp configuration
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ASSIMP_NO_EXPORT ON CACHE BOOL "" FORCE)
# IFC importer triggers a spurious MSVC /MP parallel compilation error
# (C2065 "_Ty1" in <map> from IFCUtil.h) — disable it; IFC (building
# CAD) is not needed for game asset pipelines.
set(ASSIMP_BUILD_IFC_IMPORTER OFF CACHE BOOL "" FORCE)
if(APPLE OR UNIX)
    # Use system zlib on macOS/Linux — assimp's bundled old zlib has
    # compilation issues with modern Apple Clang system headers.
    set(ASSIMP_BUILD_ZLIB OFF CACHE BOOL "" FORCE)
else()
    set(ASSIMP_BUILD_ZLIB ON CACHE BOOL "" FORCE)
endif()
set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)

# glslang configuration
set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE)
set(GLSLANG_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(ENABLE_OPT OFF CACHE BOOL "" FORCE)
set(ENABLE_HLSL OFF CACHE BOOL "" FORCE)

# GLM configuration
set(GLM_BUILD_LIBRARY OFF CACHE BOOL "" FORCE)
set(GLM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLM_BUILD_INSTALL OFF CACHE BOOL "" FORCE)

# SDL configuration
set(SDL_SHARED ON CACHE BOOL "" FORCE)
set(SDL_STATIC OFF CACHE BOOL "" FORCE)

# Jolt Physics configuration
set(USE_STATIC_MSVC_RUNTIME_LIBRARY OFF CACHE BOOL "" FORCE)   # pybind11 requires /MD (dynamic CRT)
set(OVERRIDE_CXX_FLAGS OFF CACHE BOOL "" FORCE)                # Don't override parent project's CXX flags
set(INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "" FORCE)      # Avoid LTO issues with pybind11 module
set(DOUBLE_PRECISION OFF CACHE BOOL "" FORCE)                  # Single precision (matches GLM default)
set(GENERATE_DEBUG_SYMBOLS ON CACHE BOOL "" FORCE)
set(ENABLE_ALL_WARNINGS OFF CACHE BOOL "" FORCE)               # Avoid Jolt /Wall /WX breaking our build
set(ENABLE_OBJECT_STREAM ON CACHE BOOL "" FORCE)
set(ENABLE_INSTALL OFF CACHE BOOL "" FORCE)                    # We don't need install targets
set(CPP_EXCEPTIONS_ENABLED OFF CACHE BOOL "" FORCE)
set(CPP_RTTI_ENABLED OFF CACHE BOOL "" FORCE)
set(DEBUG_RENDERER_IN_DEBUG_AND_RELEASE OFF CACHE BOOL "" FORCE)  # Not needed for physics-only usage
set(PROFILER_IN_DEBUG_AND_RELEASE OFF CACHE BOOL "" FORCE)        # Not needed
set(JPH_USE_DX12 OFF CACHE BOOL "" FORCE)                     # We don't use Jolt's GPU compute
set(JPH_USE_VK OFF CACHE BOOL "" FORCE)                       # We have our own Vulkan renderer  
if(APPLE)
    set(JPH_USE_MTL ON CACHE BOOL "" FORCE)                     # macOS Metal backend for Jolt
else()
    set(JPH_USE_MTL OFF CACHE BOOL "" FORCE)
endif()
set(JPH_USE_CPU_COMPUTE ON CACHE BOOL "" FORCE)               # CPU physics only

# ------------------------------------------------------------------------------
# External Libraries
# ------------------------------------------------------------------------------

# assimp
add_subdirectory(assimp)
add_library(assimp::assimp ALIAS assimp)

# glslang
add_subdirectory(glslang)
add_library(glslang::glslang ALIAS glslang)

# glm (header-only library)
add_subdirectory(glm)
# GLM already provides glm::glm target, no need for alias

# stb (header-only)
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/stb)

# SDL3
add_subdirectory(SDL)

# Jolt Physics (CMakeLists.txt is in Build/ subdirectory)
add_subdirectory(JoltPhysics/Build JoltPhysics)

# Vulkan
find_package(Vulkan REQUIRED)

# ------------------------------------------------------------------------------
# VulkanMemoryAllocator (VMA) — GPU memory management (git submodule)
# ------------------------------------------------------------------------------
set(VMA_BUILD_DOCUMENTATION OFF CACHE BOOL "" FORCE)
set(VMA_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(VulkanMemoryAllocator)

# ------------------------------------------------------------------------------
# ImGui
# ------------------------------------------------------------------------------
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/imgui)

set(IMGUI_CORE_SRCS
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
)

set(IMGUI_BACKEND_SRCS
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdl3.cpp
)

set(IMGUI_ALL_SRCS
    ${IMGUI_CORE_SRCS}
    ${IMGUI_BACKEND_SRCS}
)

add_library(imgui STATIC ${IMGUI_ALL_SRCS})
target_include_directories(imgui PUBLIC
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)

target_link_libraries(imgui PUBLIC
    Vulkan::Vulkan
    SDL3::SDL3
)

set_target_properties(imgui PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ------------------------------------------------------------------------------
# Restore Build Type
# ------------------------------------------------------------------------------
if(NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "${_saved_build_type}" CACHE STRING "" FORCE)
endif()

set(BUILD_SHARED_LIBS "${_saved_shared}" CACHE BOOL "" FORCE)

# ------------------------------------------------------------------------------
# Export External Libraries
# ------------------------------------------------------------------------------
set(EXTERNAL_LIBS
    assimp::assimp
    glslang::glslang
    SDL3::SDL3
    glm::glm
    imgui
    stb
    Vulkan::Vulkan
    Jolt
    GPUOpen::VulkanMemoryAllocator
    PARENT_SCOPE
)
