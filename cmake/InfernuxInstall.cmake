# ------------------------------------------------------------------------------
# Post-Build Steps
# ------------------------------------------------------------------------------

set(_infernux_third_party_runtime_targets assimp SDL3-shared Jolt)
set(_infernux_runtime_dependency_targets
    ${INFERNUX_RUNTIME_DLL_TARGETS} ${_infernux_third_party_runtime_targets} ${INFERNUX_SLANG_TARGET})
if(TARGET InfernuxVulkanLoader)
    list(APPEND _infernux_runtime_dependency_targets InfernuxVulkanLoader)
endif()
set(_infernux_player_native_contract "${CMAKE_CURRENT_BINARY_DIR}/PlayerNativeContract.json")
file(CONFIGURE OUTPUT "${_infernux_player_native_contract}"
    CONTENT "{\"contract\":\"infernux.player-native\",\"runtime_linkage\":\"static\"}\n"
    NEWLINE_STYLE UNIX)
set(_infernux_runtime_files "${_infernux_player_native_contract}" ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS})
foreach(_dependency IN LISTS _infernux_runtime_dependency_targets)
    get_target_property(_dependency_type ${_dependency} TYPE)
    if(NOT _dependency_type STREQUAL "SHARED_LIBRARY")
        continue()
    endif()
    list(APPEND _infernux_runtime_files "$<TARGET_FILE:${_dependency}>")
    if(UNIX)
        list(APPEND _infernux_runtime_files
            "$<TARGET_SONAME_FILE:${_dependency}>" "$<TARGET_LINKER_FILE:${_dependency}>")
    endif()
endforeach()

add_custom_command(TARGET _Infernux POST_BUILD
    # Remove native modules from other Python ABIs/builds first so wheels can
    # never ship two _Infernux modules (GitHub issue #47).
    COMMAND ${CMAKE_COMMAND}
        "-DSOURCE=$<TARGET_FILE:_Infernux>"
        "-DTARGET_DIR=$<TARGET_FILE_DIR:_Infernux>"
        "-DKEEP_NAME=$<TARGET_FILE_NAME:_Infernux>"
        -P "${CMAKE_SOURCE_DIR}/cmake/stage_native_module.cmake"
    COMMAND ${CMAKE_COMMAND}
        "-DDEPENDENCY_FILES=${_infernux_runtime_files}"
        "-DTARGET_DIR=$<TARGET_FILE_DIR:_Infernux>"
        -P "${CMAKE_SOURCE_DIR}/cmake/copy_dependencies.cmake"
    COMMENT "Stage the native binding and its build target dependencies"
    VERBATIM
)

# Standard modules are versioned by Slang and must accompany its compiler.
# No LLVM, GLSL compiler wrapper, graphics runtime, or proxy DLL is staged.
add_custom_command(TARGET _Infernux POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${INFERNUX_SLANG_MODULE_DIR}"
        "$<TARGET_FILE_DIR:_Infernux>/slang-standard-module-${INFERNUX_SLANG_VERSION}"
    VERBATIM)

# A module or library target builds its own artifact. Publish the coupled
# Python SDK only after both bindings and all runtime dependencies succeeded.
# This explicit target also refreshes an up-to-date build without relinking.
add_custom_target(stage_python_sdk ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${INFERNUX_SLANG_MODULE_DIR}"
        "${PYTHON_TARGET_DIR}/slang-standard-module-${INFERNUX_SLANG_VERSION}"
    COMMAND ${CMAKE_COMMAND}
        "-DSOURCE=$<TARGET_FILE:_Infernux>"
        "-DTARGET_DIR=${PYTHON_TARGET_DIR}"
        "-DKEEP_NAME=$<TARGET_FILE_NAME:_Infernux>"
        -P "${CMAKE_SOURCE_DIR}/cmake/stage_native_module.cmake"
    COMMAND ${CMAKE_COMMAND}
        "-DSOURCE=$<TARGET_FILE:_InfernuxBootstrap>"
        "-DTARGET_DIR=${PYTHON_TARGET_DIR}"
        "-DKEEP_NAME=$<TARGET_FILE_NAME:_InfernuxBootstrap>"
        -P "${CMAKE_SOURCE_DIR}/cmake/stage_native_module.cmake"
    COMMAND ${CMAKE_COMMAND}
        "-DDEPENDENCY_FILES=${_infernux_runtime_files}"
        "-DTARGET_DIR=${PYTHON_TARGET_DIR}"
        -P "${CMAKE_SOURCE_DIR}/cmake/copy_dependencies.cmake"
    DEPENDS _Infernux _InfernuxBootstrap
    COMMENT "Publishing the complete Python native SDK"
    VERBATIM
)
if(TARGET infernux_slang_metadata)
    add_dependencies(stage_python_sdk infernux_slang_metadata)
endif()
if(APPLE)
    # Set RPATH so the module finds dylibs next to itself
    set_target_properties(_Infernux PROPERTIES
        BUILD_RPATH "@loader_path"
        INSTALL_RPATH "@loader_path"
    )
    set_target_properties(_InfernuxBootstrap PROPERTIES
        BUILD_RPATH "@loader_path"
        INSTALL_RPATH "@loader_path"
    )
    foreach(_infernux_dll ${INFERNUX_RUNTIME_DLL_TARGETS})
        set_target_properties(${_infernux_dll} PROPERTIES
            BUILD_RPATH "@loader_path"
            INSTALL_RPATH "@loader_path"
        )
    endforeach()
elseif(UNIX)
    # The extension and engine libraries are a self-contained sibling set.
    # Native Python extensions resolve interpreter symbols from the importing
    # process, so reaching back into a creator environment for libpython would
    # make a wheel fail after installation into an ordinary venv.
    set(_infernux_linux_rpath "$ORIGIN")
    set_target_properties(_Infernux PROPERTIES
        BUILD_RPATH "${_infernux_linux_rpath}"
        INSTALL_RPATH "${_infernux_linux_rpath}"
    )
    set_target_properties(_InfernuxBootstrap PROPERTIES
        BUILD_RPATH "${_infernux_linux_rpath}"
        INSTALL_RPATH "${_infernux_linux_rpath}"
    )
    foreach(_infernux_dll ${INFERNUX_RUNTIME_DLL_TARGETS})
        set_target_properties(${_infernux_dll} PROPERTIES
            BUILD_RPATH "${_infernux_linux_rpath}"
            INSTALL_RPATH "${_infernux_linux_rpath}"
        )
    endforeach()
endif()

# ------------------------------------------------------------------------------
# Platform-plugin Player Runtime Pack (release engineering only)
# ------------------------------------------------------------------------------
# Runtime publication is separate from the Editor wheel. The platform plugin
# owns these target-specific artifacts; ordinary game exports only assemble them.
set(INFERNUX_PREBUILT_RUNTIME_DIR "${CMAKE_BINARY_DIR}/platform-player/runtime-packs")
set(INFERNUX_PREBUILT_RUNTIME_MODULE_DIR "${CMAKE_BINARY_DIR}/platform-player/_runtime_modules")
string(TOLOWER "${CMAKE_SYSTEM_NAME}" _infernux_player_platform)
infernux_dependency_path(
    _infernux_player_plugin
    "infernux_${_infernux_player_platform}")
set(INFERNUX_PLATFORM_PLAYER_OUTPUT_DIR
    "${_infernux_player_plugin}/package/editor/infernux_${_infernux_player_platform}/player"
    CACHE PATH "Compatible desktop Player payload destination")
option(INFERNUX_PLAYER_PACKAGE_PLUGIN "Build a standalone development plugin after publishing its Player" OFF)
set(_infernux_package_player_plugin)
if(INFERNUX_PLAYER_PACKAGE_PLUGIN)
    set(_infernux_package_player_plugin COMMAND "${Python3_EXECUTABLE}" "${_infernux_player_plugin}/release.py")
endif()
add_custom_target(prebuild_player_runtime
    COMMAND ${CMAKE_COMMAND}
        "-DINFERNUX_BUILD_CONFIG=$<CONFIG>"
        "-DINFERNUX_SOURCE_DIR=${INFERNUX_STAGE_DIR}/python-wheel-source"
        "-DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}"
        # Fingerprint and compile the exact installed wheel payload, not a
        # source tree whose native layout differs from the installed package.
        "-DNATIVE_MODULE_DIR=${INFERNUX_STAGE_DIR}/python-wheel-source/python/infernux/lib"
        "-DPLAYER_HOST_PATH=${INFERNUX_PLAYER_HOST_BUILD_PATH}"
        "-DOUTPUT_ROOT=${INFERNUX_PREBUILT_RUNTIME_DIR}"
        "-DMODULE_OUTPUT_ROOT=${INFERNUX_PREBUILT_RUNTIME_MODULE_DIR}"
        "-DPLATFORM_PLAYER_OUTPUT=${INFERNUX_PLATFORM_PLAYER_OUTPUT_DIR}"
        "-DBUILD_CACHE_ROOT=${CMAKE_BINARY_DIR}/build-cache/player-runtime"
        "-DINFERNUX_STRIP_TOOL=${CMAKE_STRIP}"
        -P "${CMAKE_SOURCE_DIR}/cmake/prebuild_player_runtime.cmake"
    ${_infernux_package_player_plugin}
    DEPENDS _Infernux _InfernuxBootstrap
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Publishing the platform Player payload for engine-versioned packaging"
    VERBATIM
)
add_dependencies(prebuild_player_runtime stage_python_sdk)
if(TARGET InfernuxPlayerHost)
    add_dependencies(prebuild_player_runtime InfernuxPlayerHost)
endif()

add_custom_command(TARGET _InfernuxBootstrap POST_BUILD
    COMMAND ${CMAKE_COMMAND}
        "-DSOURCE=$<TARGET_FILE:_InfernuxBootstrap>"
        "-DTARGET_DIR=$<TARGET_FILE_DIR:_InfernuxBootstrap>"
        "-DKEEP_NAME=$<TARGET_FILE_NAME:_InfernuxBootstrap>"
        -P "${CMAKE_SOURCE_DIR}/cmake/stage_native_module.cmake"
    COMMENT "Stage _InfernuxBootstrap binding module in the build tree"
    VERBATIM
)

# ------------------------------------------------------------------------------
# Install components used as deterministic wheel staging inputs
# ------------------------------------------------------------------------------
set(INFERNUX_PYTHON_INSTALL_COMPONENT "PythonWheel")

install(DIRECTORY "${CMAKE_SOURCE_DIR}/licenses/svg/"
    DESTINATION "python/infernux/resources/licenses/svg"
    COMPONENT ${INFERNUX_PYTHON_INSTALL_COMPONENT})

install(DIRECTORY "${CMAKE_SOURCE_DIR}/licenses/utf8proc/"
    DESTINATION "licenses/utf8proc"
    COMPONENT ${INFERNUX_PYTHON_INSTALL_COMPONENT})

if(NOT INFERNUX_OFFICIAL_PLUGIN_OUTPUT_DIR)
    message(FATAL_ERROR
        "INFERNUX_OFFICIAL_PLUGIN_OUTPUT_DIR must be defined before install rules are declared")
endif()

install(
    DIRECTORY "${CMAKE_SOURCE_DIR}/python/infernux/"
    DESTINATION "python/infernux"
    COMPONENT ${INFERNUX_PYTHON_INSTALL_COMPONENT}
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*.pyc" EXCLUDE
    PATTERN "*.pyo" EXCLUDE
    PATTERN "*.dll" EXCLUDE
    PATTERN "*.pyd" EXCLUDE
    PATTERN "*.so" EXCLUDE
    PATTERN "*.dylib" EXCLUDE
    PATTERN "*.meta" EXCLUDE
    PATTERN "*.inxpkg" EXCLUDE
    PATTERN "_runtime_packs" EXCLUDE
    PATTERN "_runtime_modules" EXCLUDE
    PATTERN "official_packages" EXCLUDE
    PATTERN "player_runtime" EXCLUDE
)

install(
    FILES
        "${CMAKE_SOURCE_DIR}/python/infernux_project_lock.py"
        "${CMAKE_SOURCE_DIR}/python/infernux_editor.py"
    DESTINATION "python"
    COMPONENT ${INFERNUX_PYTHON_INSTALL_COMPONENT}
)

install(
    FILES
        "${CMAKE_SOURCE_DIR}/pyproject.toml"
        "${CMAKE_SOURCE_DIR}/setup.py"
        "${CMAKE_SOURCE_DIR}/MANIFEST.in"
        "${CMAKE_SOURCE_DIR}/README.md"
        "${CMAKE_SOURCE_DIR}/README-zh.md"
        "${CMAKE_SOURCE_DIR}/LICENSE"
    DESTINATION "."
    COMPONENT ${INFERNUX_PYTHON_INSTALL_COMPONENT}
)

install(
    TARGETS _Infernux _InfernuxBootstrap ${INFERNUX_RUNTIME_DLL_TARGETS}
    RUNTIME DESTINATION "python/infernux/lib"
        COMPONENT ${INFERNUX_PYTHON_INSTALL_COMPONENT}
    LIBRARY DESTINATION "python/infernux/lib"
        COMPONENT ${INFERNUX_PYTHON_INSTALL_COMPONENT}
)
install(
    FILES "${_infernux_player_native_contract}"
    DESTINATION "python/infernux/lib"
    COMPONENT ${INFERNUX_PYTHON_INSTALL_COMPONENT}
)
install(FILES "$<TARGET_FILE:${INFERNUX_SLANG_TARGET}>"
    DESTINATION "python/infernux/lib" COMPONENT ${INFERNUX_PYTHON_INSTALL_COMPONENT})
if(UNIX)
    install(FILES "$<TARGET_SONAME_FILE:${INFERNUX_SLANG_TARGET}>"
                  "$<TARGET_LINKER_FILE:${INFERNUX_SLANG_TARGET}>"
        DESTINATION "python/infernux/lib" COMPONENT ${INFERNUX_PYTHON_INSTALL_COMPONENT})
endif()
install(DIRECTORY "${INFERNUX_SLANG_MODULE_DIR}/"
    DESTINATION "python/infernux/lib/slang-standard-module-${INFERNUX_SLANG_VERSION}"
    COMPONENT ${INFERNUX_PYTHON_INSTALL_COMPONENT})
install(FILES "${CMAKE_SOURCE_DIR}/external/slang_for_infernux/LICENSE"
    DESTINATION "python/infernux/resources/licenses/slang"
    COMPONENT ${INFERNUX_PYTHON_INSTALL_COMPONENT})
install(DIRECTORY "${CMAKE_SOURCE_DIR}/external/slang_for_infernux/LICENSES/"
    DESTINATION "python/infernux/resources/licenses/slang/LICENSES"
    COMPONENT ${INFERNUX_PYTHON_INSTALL_COMPONENT})
install(DIRECTORY "${INFERNUX_SLANG_NOTICES_DIR}/"
    DESTINATION "python/infernux/resources/licenses/slang/third-party-notices"
    COMPONENT ${INFERNUX_PYTHON_INSTALL_COMPONENT})
install(CODE [[
    file(REMOVE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/infernux/lib/libInfernuxRuntime.so")
]] COMPONENT ${INFERNUX_PYTHON_INSTALL_COMPONENT})

install(
    TARGETS ${_infernux_third_party_runtime_targets}
    RUNTIME DESTINATION "python/infernux/lib"
        COMPONENT ${INFERNUX_PYTHON_INSTALL_COMPONENT}
    LIBRARY DESTINATION "python/infernux/lib"
        COMPONENT ${INFERNUX_PYTHON_INSTALL_COMPONENT}
        NAMELINK_COMPONENT ${INFERNUX_PYTHON_INSTALL_COMPONENT}
)
install(
    FILES ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS}
    DESTINATION "python/infernux/lib"
    COMPONENT ${INFERNUX_PYTHON_INSTALL_COMPONENT}
)

install(
    FILES
        "${INFERNUX_OFFICIAL_PLUGIN_OUTPUT_DIR}/official-registry.json"
        "${INFERNUX_OFFICIAL_PLUGIN_OUTPUT_DIR}/default-libraries.json"
    DESTINATION "python/infernux/resources/official_packages"
    COMPONENT ${INFERNUX_PYTHON_INSTALL_COMPONENT}
)

install(
    FILES
        "${INFERNUX_OFFICIAL_PLUGIN_OUTPUT_DIR}/infernux.mcp.inxpkg"
    DESTINATION "python/infernux/resources"
    COMPONENT ${INFERNUX_PYTHON_INSTALL_COMPONENT}
)
