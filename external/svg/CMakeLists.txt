# Portable CPU rasterization, with no additional deployment DLL or GPU backend.
include(FetchContent)
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(LUNASVG_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(USE_SYSTEM_PLUTOVG OFF CACHE BOOL "" FORCE)
FetchContent_Declare(infernux_lunasvg
    GIT_REPOSITORY https://github.com/sammycage/lunasvg.git
    GIT_TAG 83c58df8103dc7dca423dfd824992af94d49bed6
    SOURCE_SUBDIR _infernux_fetch_only
)
FetchContent_MakeAvailable(infernux_lunasvg)
add_subdirectory("${infernux_lunasvg_SOURCE_DIR}" "${infernux_lunasvg_BINARY_DIR}" EXCLUDE_FROM_ALL)
if(EMSCRIPTEN)
    target_compile_options(lunasvg PRIVATE -fwasm-exceptions)
endif()

# SVG assets are self-contained. Keep embedded data images, but do not let
# image hrefs read arbitrary host paths. Do not modify the upstream checkout.
file(READ "${infernux_lunasvg_SOURCE_DIR}/source/svgelement.cpp" _svg_elements)
set(_svg_file_load "return plutovg_surface_load_from_image_file(href.data());")
string(FIND "${_svg_elements}" "${_svg_file_load}" _svg_file_load_offset)
if(_svg_file_load_offset EQUAL -1)
    message(FATAL_ERROR "LunaSVG image resource policy must be reviewed before updating the dependency")
endif()
string(REPLACE "${_svg_file_load}" "return Bitmap(); // External file images are not asset dependencies."
    _svg_elements "${_svg_elements}")
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/SvgElements.cpp" CONTENT "${_svg_elements}" @ONLY)
get_target_property(_svg_sources lunasvg SOURCES)
list(REMOVE_ITEM _svg_sources source/svgelement.cpp)
list(APPEND _svg_sources "${CMAKE_CURRENT_BINARY_DIR}/SvgElements.cpp")
set_property(TARGET lunasvg PROPERTY SOURCES "${_svg_sources}")
