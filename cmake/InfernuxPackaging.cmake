# Python wheel staging, packaging, and optional environment installation.

set(INFERNUX_PYTHON_STAGE_DIR "${INFERNUX_STAGE_DIR}/python-wheel-source")
set(INFERNUX_PYTHON_WHEEL_DIR "${INFERNUX_STAGE_DIR}/wheels")

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
    DEPENDS
        _Infernux
        _InfernuxBootstrap
        refresh_player_native_contract
        infernux_official_plugins
    COMMENT "Assembling the Python wheel source tree after native and official plugin builds"
    VERBATIM
)
if(TARGET infernux_gpu_jit_compiler)
    add_dependencies(stage_python_package infernux_gpu_jit_compiler)
endif()
if(TARGET InfernuxPlayerHost)
    add_dependencies(stage_python_package InfernuxPlayerHost)
endif()

# Desktop extensions are built before both wheel and precompiled Player staging.
# Android/Web staging retains the same Python sources and excludes host binaries.
if(NOT INFERNUX_USE_TARGET_PYTHON AND NOT EMSCRIPTEN)
    add_executable(InfernuxEditorProgress cpp/apps/editor/EditorProgress.cpp)
    target_link_libraries(InfernuxEditorProgress PRIVATE SDL3::SDL3)
    set_target_properties(InfernuxEditorProgress PROPERTIES
        BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
    install(TARGETS InfernuxEditorProgress RUNTIME DESTINATION python/infernux/lib
        COMPONENT ${INFERNUX_PYTHON_INSTALL_COMPONENT})
    add_dependencies(stage_python_package InfernuxEditorProgress)

    foreach(_module IN ITEMS components/_component_native components/_component_lifecycle
            components/skeletal_animator ui/ui_render_dispatch)
        get_filename_component(_name "${_module}" NAME)
        get_filename_component(_package "${_module}" DIRECTORY)
        string(REPLACE "/" "." _import_name "${_module}")
        set(_target "infernux_python_${_name}")
        set(_generated "${CMAKE_CURRENT_BINARY_DIR}/python-aot/${_package}/${_name}.c")
        add_custom_command(OUTPUT "${_generated}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/python-aot/${_package}"
            COMMAND "${Python3_EXECUTABLE}" -m cython -3 -X annotation_typing=False
                --module-name "infernux.${_import_name}"
                --output-file "${_generated}" "python/infernux/${_module}.py"
            DEPENDS "${CMAKE_SOURCE_DIR}/python/infernux/${_module}.py"
                "${CMAKE_SOURCE_DIR}/requirements.txt"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            VERBATIM)
        Python3_add_library(${_target} MODULE WITH_SOABI "${_generated}")
        set_target_properties(${_target} PROPERTIES
            OUTPUT_NAME "${_name}"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python-aot/$<CONFIG>/infernux/${_package}"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python-aot/$<CONFIG>/infernux/${_package}"
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "$ORIGIN")
        install(TARGETS ${_target}
            LIBRARY DESTINATION "python/infernux/${_package}" COMPONENT ${INFERNUX_PYTHON_INSTALL_COMPONENT}
            RUNTIME DESTINATION "python/infernux/${_package}" COMPONENT ${INFERNUX_PYTHON_INSTALL_COMPONENT})
        add_dependencies(stage_python_package ${_target})
    endforeach()
endif()
add_dependencies(prebuild_player_runtime stage_python_package)

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
