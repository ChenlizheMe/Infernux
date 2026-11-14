# Python wheel staging, packaging, and optional environment installation.

set(INFERNUX_PYTHON_STAGE_DIR "${INFERNUX_STAGE_DIR}/python-wheel-source")
set(INFERNUX_PYTHON_WHEEL_DIR "${INFERNUX_STAGE_DIR}/wheels")
option(INFERNUX_REQUIRE_CLEAN_WHEEL_SOURCE "Require a clean Git checkout for release wheels" OFF)
set(_infernux_player_build_source_args)
if(INFERNUX_REQUIRE_CLEAN_WHEEL_SOURCE)
    list(APPEND _infernux_player_build_source_args --require-clean)
endif()

# Developer/release dependency build, not part of a user's game export.
# LLVM's CMake package must already be provided through CMAKE_PREFIX_PATH.
# Keep this separate from package_python until every supported installation
# channel can resolve the patched dependency without a source build.
add_custom_target(package_cpu_jit_dependency
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/cmake/package_cpu_jit_dependency.py"
        --source-dir "${CMAKE_SOURCE_DIR}/external/llvmlite_for_infernux"
        --wheel-dir "${INFERNUX_STAGE_DIR}/cpu-jit-wheels"
    COMMENT "Building the pinned llvmlite dependency wheel (requires an LLVM build toolchain)"
    VERBATIM
)

set(_infernux_repair_linux_wheel)
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    set(_infernux_repair_linux_wheel
        COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/cmake/repair_linux_wheel.py"
            --wheel-dir "${INFERNUX_PYTHON_WHEEL_DIR}"
    )
endif()

add_custom_target(stage_python_package
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${INFERNUX_PYTHON_STAGE_DIR}"
    COMMAND ${CMAKE_COMMAND} --install "${CMAKE_BINARY_DIR}"
        --config "$<CONFIG>"
        --prefix "${INFERNUX_PYTHON_STAGE_DIR}"
        --component ${INFERNUX_PYTHON_INSTALL_COMPONENT}
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/cmake/write_player_build_source.py"
        --source "${CMAKE_SOURCE_DIR}"
        --output "${INFERNUX_PYTHON_STAGE_DIR}/python/infernux/resources/player-build-source.json"
        ${_infernux_player_build_source_args}
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/scripts/release/build_platform_plugins.py"
        catalog --output "${INFERNUX_PYTHON_STAGE_DIR}/python/infernux/resources/official_packages/official-registry.json"
    DEPENDS
        stage_python_sdk
        infernux_official_plugins
    COMMENT "Assembling the Python wheel source tree after native and official plugin builds"
    VERBATIM
)
if(TARGET InfernuxPlayerHost)
    add_dependencies(stage_python_package InfernuxPlayerHost)
endif()

# Editor progress remains independent of the application's render loop.
if(NOT INFERNUX_USE_TARGET_PYTHON AND NOT EMSCRIPTEN)
    add_executable(InfernuxEditorProgress
        cpp/apps/editor/EditorProgress.cpp
        cpp/infernux/function/editor/EditorThemeRegistry.cpp
        external/imgui_for_infernux/backends/imgui_impl_sdlrenderer3.cpp
    )
    target_include_directories(InfernuxEditorProgress PRIVATE cpp/infernux)
    target_link_libraries(InfernuxEditorProgress PRIVATE SDL3::SDL3 imgui)
    if(MSVC)
        target_compile_options(InfernuxEditorProgress PRIVATE /utf-8)
    endif()
    set_target_properties(InfernuxEditorProgress PROPERTIES
        BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
    add_custom_command(TARGET stage_python_sdk POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            "-DSOURCE=$<TARGET_FILE:InfernuxEditorProgress>"
            "-DDESTINATION=${PYTHON_TARGET_DIR}/$<TARGET_FILE_NAME:InfernuxEditorProgress>"
            -P "${CMAKE_SOURCE_DIR}/cmake/stage_build_file.cmake"
        COMMENT "Stage the Editor progress helper in the build tree"
        VERBATIM
    )
    install(TARGETS InfernuxEditorProgress RUNTIME DESTINATION python/infernux/lib
        COMPONENT ${INFERNUX_PYTHON_INSTALL_COMPONENT})
    add_dependencies(stage_python_sdk InfernuxEditorProgress)
endif()
add_dependencies(prebuild_player_runtime stage_python_package)

add_custom_target(update_native_platform_plugin
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/scripts/release/build_platform_plugins.py"
        package --targets "${_infernux_player_platform}"
        --payload "${INFERNUX_PLATFORM_PLAYER_OUTPUT_DIR}"
    DEPENDS prebuild_player_runtime
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    USES_TERMINAL VERBATIM)

add_custom_target(package_python
    COMMAND ${CMAKE_COMMAND} -E echo "Ensuring Python packaging tools are available..."
    COMMAND ${CMAKE_COMMAND}
        -DINFERNUX_SOURCE_DIR=${CMAKE_SOURCE_DIR}
        -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}
        -P "${CMAKE_SOURCE_DIR}/cmake/ensure_python_packaging_tools.cmake"

    COMMAND ${CMAKE_COMMAND} -E rm -rf "${INFERNUX_PYTHON_STAGE_DIR}/build"
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${INFERNUX_PYTHON_STAGE_DIR}/python/infernux.egg-info"
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${INFERNUX_PYTHON_WHEEL_DIR}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${INFERNUX_PYTHON_WHEEL_DIR}"
    COMMAND ${CMAKE_COMMAND} -E chdir "${INFERNUX_PYTHON_STAGE_DIR}"
        ${CMAKE_COMMAND} -E env
        "INFERNUX_SOURCE_DIR=${INFERNUX_PYTHON_STAGE_DIR}"
        "INFERNUX_STAGED_WHEEL_BUILD=1"
        "${Python3_EXECUTABLE}" -m build --wheel --no-isolation
        --outdir "${INFERNUX_PYTHON_WHEEL_DIR}"

    COMMAND ${CMAKE_COMMAND}
        -DINFERNUX_SOURCE_DIR=${INFERNUX_PYTHON_STAGE_DIR}
        -DINFERNUX_WHEEL_DIR=${INFERNUX_PYTHON_WHEEL_DIR}
        -P "${CMAKE_SOURCE_DIR}/cmake/verify_python_wheel.cmake"

    ${_infernux_repair_linux_wheel}

    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${INFERNUX_PYTHON_WHEEL_DIR}" "${INFERNUX_RELEASE_DIR}"

    DEPENDS stage_python_package "${CMAKE_SOURCE_DIR}/cmake/repair_linux_wheel.py"
    COMMENT "Building and verifying the Infernux Python wheel"
    VERBATIM
)

add_custom_target(install_python_wheel
    COMMAND ${CMAKE_COMMAND}
        -DINFERNUX_SOURCE_DIR=${CMAKE_SOURCE_DIR}
        -DINFERNUX_WHEEL_DIR=${INFERNUX_PYTHON_WHEEL_DIR}
        -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}
        -P "${CMAKE_SOURCE_DIR}/cmake/install_wheel.cmake"
    DEPENDS package_python
    COMMENT "Installing the verified Infernux wheel into the active Python environment"
    VERBATIM
)
