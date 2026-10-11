# Canonical dependency layout for the engine build.
#
# Git submodules remain the source of truth for pinned dependency revisions.
# This file describes where those checkouts live and which ones are maintained
# forks versus official platform/editor plugins. A plugin is a packaging input,
# not a link-time dependency of the editor runtime.

include_guard(GLOBAL)

set(INFERNUX_SOURCE_ROOT "${CMAKE_SOURCE_DIR}" CACHE INTERNAL
    "Infernux source checkout root")
set(INFERNUX_EXTERNAL_ROOT "${INFERNUX_SOURCE_ROOT}/external" CACHE INTERNAL
    "Infernux external dependency root")
set(INFERNUX_PLUGIN_ROOT "${INFERNUX_EXTERNAL_ROOT}/plugins" CACHE INTERNAL
    "Infernux official plugin checkout root")

# These are the engine-maintained dependency forks. Their submodules track the
# single `infernux-support` branch; the superproject still builds the exact
# gitlink recorded in the checkout.
set(INFERNUX_MAINTAINED_FORKS
    imgui_for_infernux
    sdl_for_infernux
    joltphysics_for_infernux
    taichi_for_infernux
    llvmlite_for_infernux
)

# Official plugins are packaged inputs. Their platform-native export projects
# are entered explicitly by platform workflows, not through this dependency
# list.
set(INFERNUX_OFFICIAL_PLUGIN_NAMES
    infernux_mcp
    infernux_windows
    infernux_linux
    infernux_android
    infernux_web
)

set(INFERNUX_THIRD_PARTY_SUBMODULES
    assimp
    glslang
    glm
    imgui_for_infernux
    sdl_for_infernux
    stb
    joltphysics_for_infernux
    VulkanMemoryAllocator
    taichi_for_infernux
    MikkTSpace
)

# Source-only Python dependency fork. It has no CMake project and is consumed
# by the CPU-JIT packaging target instead of add_subdirectory().
set(INFERNUX_PYTHON_DEPENDENCY_FORKS
    llvmlite_for_infernux
)

function(infernux_dependency_path OUT_VAR DEPENDENCY_NAME)
    if(DEPENDENCY_NAME MATCHES "^infernux_(mcp|windows|linux|android|web)$")
        set(_path "${INFERNUX_PLUGIN_ROOT}/${DEPENDENCY_NAME}")
    else()
        set(_path "${INFERNUX_EXTERNAL_ROOT}/${DEPENDENCY_NAME}")
    endif()
    set(${OUT_VAR} "${_path}" PARENT_SCOPE)
endfunction()

function(infernux_require_dependency DEPENDENCY_NAME DESCRIPTION)
    infernux_dependency_path(_dependency_path "${DEPENDENCY_NAME}")
    if(NOT IS_DIRECTORY "${_dependency_path}")
        message(FATAL_ERROR
            "Missing ${DESCRIPTION} at '${_dependency_path}'. "
            "Initialize it with: git submodule update --init --recursive -- '${_dependency_path}'")
    endif()
endfunction()
