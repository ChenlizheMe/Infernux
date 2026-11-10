# Toolchains stay in their platform builders; this entry point only orchestrates.
find_package(Python3 3.13 REQUIRED COMPONENTS Interpreter)
set(INFERNUX_PLATFORM_PLUGIN_TARGETS "host" CACHE STRING "host, all, or a semicolon-separated platform list")
set(INFERNUX_PLATFORM_PLUGIN_TOOLS "" CACHE PATH "Web shader tools for both supported editor hosts")
add_custom_target(update_platform_plugins
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/scripts/release/refresh_platform_plugins.py"
        --targets ${INFERNUX_PLATFORM_PLUGIN_TARGETS}
        --tools "${INFERNUX_PLATFORM_PLUGIN_TOOLS}"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    USES_TERMINAL VERBATIM)
