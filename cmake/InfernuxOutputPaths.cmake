# Shared ownership for disposable assembly trees and versioned release files.
include_guard(GLOBAL)

get_filename_component(_infernux_build_name "${CMAKE_BINARY_DIR}" NAME)
set(INFERNUX_STAGE_DIR "${CMAKE_SOURCE_DIR}/out/stage/${_infernux_build_name}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/pyproject.toml")
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
        "import pathlib,sys,tomllib; print(tomllib.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))['project']['version'])"
        "${CMAKE_SOURCE_DIR}/pyproject.toml"
    OUTPUT_VARIABLE INFERNUX_PACKAGE_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
set(INFERNUX_RELEASE_DIR "${CMAKE_SOURCE_DIR}/dist/releases/${INFERNUX_PACKAGE_VERSION}")
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
        "import runpy,sys; print(runpy.run_path(sys.argv[1])['ENGINE_RELEASE'])"
        "${CMAKE_SOURCE_DIR}/python/infernux/version.py"
    OUTPUT_VARIABLE INFERNUX_ENGINE_RELEASE OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/python/infernux/version.py")
execute_process(COMMAND git -C "${CMAKE_SOURCE_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE INFERNUX_SOURCE_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
foreach(_infernux_git_file IN ITEMS HEAD packed-refs)
    execute_process(COMMAND git -C "${CMAKE_SOURCE_DIR}" rev-parse --path-format=absolute --git-path "${_infernux_git_file}"
        OUTPUT_VARIABLE _infernux_git_path OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_infernux_git_path}")
endforeach()
execute_process(COMMAND git -C "${CMAKE_SOURCE_DIR}" symbolic-ref -q HEAD
    OUTPUT_VARIABLE _infernux_git_branch OUTPUT_STRIP_TRAILING_WHITESPACE)
if(_infernux_git_branch)
    execute_process(COMMAND git -C "${CMAKE_SOURCE_DIR}" rev-parse --path-format=absolute --git-path "${_infernux_git_branch}"
        OUTPUT_VARIABLE _infernux_git_path OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_infernux_git_path}")
endif()
