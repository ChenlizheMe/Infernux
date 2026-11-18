# Source checkouts are prepared by Git before configuration. This also applies
# when the Web Player or a native host generator includes the engine modules.
include_guard(GLOBAL)
get_filename_component(INFERNUX_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(INFERNUX_EXTERNAL_ROOT "${INFERNUX_SOURCE_ROOT}/external")
set(INFERNUX_PLUGIN_ROOT "${INFERNUX_EXTERNAL_ROOT}/plugins")
set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE INTERNAL "Build from the prepared source tree")

function(infernux_require_sources root)
    set(missing)
    foreach(source IN LISTS ARGN)
        if(NOT EXISTS "${root}/${source}")
            list(APPEND missing "${root}/${source}")
        endif()
    endforeach()
    if(missing)
        list(JOIN missing "\n  " paths)
        message(FATAL_ERROR
            "Missing external source files:\n  ${paths}\n"
            "From the engine checkout, run: git submodule update --init --recursive")
    endif()
endfunction()

function(infernux_require_slang_sources)
    infernux_require_sources("${INFERNUX_EXTERNAL_ROOT}/slang_for_infernux"
        CMakeLists.txt
        external/cmark/CMakeLists.txt
        external/fast_float/include/fast_float/fast_float.h
        external/lua/lua.h
        external/lz4/build/cmake/CMakeLists.txt
        external/mimalloc/CMakeLists.txt
        external/miniz/CMakeLists.txt
        external/spirv-headers/CMakeLists.txt
        external/spirv-tools/CMakeLists.txt
        external/unordered_dense/CMakeLists.txt
        external/vulkan/include/vulkan/vulkan.h)
endfunction()
