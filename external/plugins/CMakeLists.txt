# Only wheel-bundled archives are generated in the build tree. The default MCP
# package is copied into the source-checkout resources used by the editor; the
# platform packages remain release artifacts and are downloaded on demand.
# Wheel staging depends on this target, preserving the order native module →
# official plugins → wheel.
include("${CMAKE_SOURCE_DIR}/cmake/InfernuxDependencyLayout.cmake")

set(INFERNUX_OFFICIAL_PLUGIN_CATALOG
    "${CMAKE_CURRENT_SOURCE_DIR}/plugins.json")
set(INFERNUX_OFFICIAL_PLUGIN_BUILDER
    "${CMAKE_CURRENT_SOURCE_DIR}/build_official_plugins.py")

option(INFERNUX_REFRESH_EDITOR_PLUGIN_RESOURCES
    "Copy the generated default plugin catalog/package into the source checkout"
    ON
)

foreach(_infernux_plugin IN LISTS INFERNUX_OFFICIAL_PLUGIN_NAMES)
    infernux_require_dependency("${_infernux_plugin}" "official plugin source")
endforeach()

set(INFERNUX_OFFICIAL_PLUGIN_OUTPUT_DIR
    "${CMAKE_BINARY_DIR}/official-plugins"
)
set(INFERNUX_OFFICIAL_PLUGIN_OUTPUT_DIR
    "${INFERNUX_OFFICIAL_PLUGIN_OUTPUT_DIR}"
    PARENT_SCOPE
)

# This phony target rebuilds the packages on every invocation.  Do not attach a
# recursive CONFIGURE_DEPENDS glob here: generated Python cache files would
# otherwise force an unrelated whole-project CMake reconfigure even though the
# package exporter explicitly excludes them.

set(INFERNUX_OFFICIAL_PLUGIN_ARTIFACTS
    "${INFERNUX_OFFICIAL_PLUGIN_OUTPUT_DIR}/official-registry.json"
    "${INFERNUX_OFFICIAL_PLUGIN_OUTPUT_DIR}/default-libraries.json"
    "${INFERNUX_OFFICIAL_PLUGIN_OUTPUT_DIR}/infernux.mcp.inxpkg"
)

# The plugin builder imports the freshly linked native module.  On ELF hosts
# the module's third-party shared libraries live in sibling target directories
# in the build tree, so make that search path part of the build target itself;
# callers should not have to export a machine-specific LD_LIBRARY_PATH.
set(INFERNUX_OFFICIAL_PLUGIN_RUNTIME_ENV)
if(UNIX AND NOT APPLE)
    list(APPEND INFERNUX_OFFICIAL_PLUGIN_RUNTIME_ENV
        "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:_Infernux>:$<TARGET_FILE_DIR:assimp>:$<TARGET_FILE_DIR:SDL3-shared>:$<TARGET_FILE_DIR:Jolt>"
    )
endif()

set(_infernux_editor_plugin_refresh_commands)
if(INFERNUX_REFRESH_EDITOR_PLUGIN_RESOURCES)
    list(APPEND _infernux_editor_plugin_refresh_commands
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${INFERNUX_OFFICIAL_PLUGIN_OUTPUT_DIR}/infernux.mcp.inxpkg"
            "${CMAKE_SOURCE_DIR}/python/infernux/resources/infernux.mcp.inxpkg"
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "${CMAKE_SOURCE_DIR}/python/infernux/resources/official_packages"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${INFERNUX_OFFICIAL_PLUGIN_OUTPUT_DIR}/official-registry.json"
            "${INFERNUX_OFFICIAL_PLUGIN_OUTPUT_DIR}/default-libraries.json"
            "${CMAKE_SOURCE_DIR}/python/infernux/resources/official_packages"
    )
endif()

add_custom_target(infernux_official_plugins
    COMMAND ${CMAKE_COMMAND} -E env
        "PYTHONPATH=${CMAKE_SOURCE_DIR}/python"
        "INFERNUX_NATIVE_MODULE_DIR=$<TARGET_FILE_DIR:_Infernux>"
        ${INFERNUX_OFFICIAL_PLUGIN_RUNTIME_ENV}
        "${Python3_EXECUTABLE}"
        "${INFERNUX_OFFICIAL_PLUGIN_BUILDER}"
        --source-root "${CMAKE_CURRENT_SOURCE_DIR}"
        --output-root "${INFERNUX_OFFICIAL_PLUGIN_OUTPUT_DIR}"
        --catalog "${INFERNUX_OFFICIAL_PLUGIN_CATALOG}"
    ${_infernux_editor_plugin_refresh_commands}
    DEPENDS
        _Infernux
        "${INFERNUX_OFFICIAL_PLUGIN_BUILDER}"
        "${INFERNUX_OFFICIAL_PLUGIN_CATALOG}"
    BYPRODUCTS ${INFERNUX_OFFICIAL_PLUGIN_ARTIFACTS}
    COMMENT "Building official Infernux plugins"
    VERBATIM
)

# Platform plugin releases are built and published by each plugin repository.
# The engine owns only catalog generation and its bundled default packages.
