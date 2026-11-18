# Share one deflate implementation between asset import, capture and the Web runtime.
include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/InfernuxDependencyLayout.cmake")
infernux_require_sources("${INFERNUX_EXTERNAL_ROOT}" zlib/CMakeLists.txt)
set(ZLIB_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(ZLIB_BUILD_STATIC ON CACHE BOOL "" FORCE)
set(ZLIB_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ZLIB_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory("${INFERNUX_EXTERNAL_ROOT}/zlib" "${CMAKE_CURRENT_BINARY_DIR}/zlib" EXCLUDE_FROM_ALL)
