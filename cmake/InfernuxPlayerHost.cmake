# Native packaged-game bootstrap shared by Windows and Linux desktop Players.

if(NOT INFERNUX_BUILD_PLAYER_HOST)
    return()
endif()

add_executable(InfernuxPlayerHost
    cpp/apps/player/InfernuxPlayerLauncher.cpp
    cpp/apps/player/PlayerHost.cpp
    cpp/apps/player/PlayerHost.h
)
if(WIN32)
    set(_player_version_file "${CMAKE_SOURCE_DIR}/python/infernux/version.py")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_player_version_file}")
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
            "import runpy; print(runpy.run_path(r'${_player_version_file}')['ENGINE_VERSION'])"
        OUTPUT_VARIABLE INFERNUX_PLAYER_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
    )
    string(REPLACE "." "," INFERNUX_PLAYER_VERSION_COMMAS "${INFERNUX_PLAYER_VERSION}")
    configure_file(cpp/apps/player/InfernuxPlayerLauncher.rc
        "${CMAKE_BINARY_DIR}/generated/InfernuxPlayerLauncher.rc" @ONLY)
    target_sources(InfernuxPlayerHost PRIVATE
        "${CMAKE_BINARY_DIR}/generated/InfernuxPlayerLauncher.rc"
        cpp/apps/player/InfernuxPlayerLauncher.manifest
    )
    set_target_properties(InfernuxPlayerHost PROPERTIES WIN32_EXECUTABLE TRUE)
endif()
set(INFERNUX_PLAYER_HOST_DIR
    "${CMAKE_BINARY_DIR}/player-runtime"
)
set_target_properties(InfernuxPlayerHost PROPERTIES
    OUTPUT_NAME "InfernuxPlayerHost"
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
    RUNTIME_OUTPUT_DIRECTORY "${INFERNUX_PLAYER_HOST_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${INFERNUX_PLAYER_HOST_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${INFERNUX_PLAYER_HOST_DIR}"
    PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/symbols"
    PDB_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/symbols"
    PDB_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/symbols"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/symbols"
    COMPILE_PDB_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/symbols"
    COMPILE_PDB_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/symbols"
)

# prebuild_player_runtime publishes this target with the complete Player payload
# into its platform plugin. Do not also copy it into the Editor's source resources.
if(MSVC)
    target_compile_options(InfernuxPlayerHost PRIVATE /utf-8 /O1)
    # Python's Windows headers inject the import library. PlayerHost resolves
    # every PEP 587 symbol from the Python DLL at runtime instead.
    target_link_options(InfernuxPlayerHost PRIVATE
        "/NODEFAULTLIB:python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}.lib"
    )
endif()
target_include_directories(InfernuxPlayerHost PRIVATE
    ${CMAKE_SOURCE_DIR}/cpp
    ${CMAKE_SOURCE_DIR}/cpp/infernux
    ${CMAKE_SOURCE_DIR}/external
    ${INFERNUX_PYTHON_INCLUDE_DIRS}
)
target_link_libraries(InfernuxPlayerHost PRIVATE ${CMAKE_DL_LIBS})
if(WIN32)
    target_link_libraries(InfernuxPlayerHost PRIVATE shell32)
else()
    set_target_properties(InfernuxPlayerHost PROPERTIES
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
    )
endif()
set(INFERNUX_PLAYER_HOST_BUILD_PATH "$<TARGET_FILE:InfernuxPlayerHost>")
