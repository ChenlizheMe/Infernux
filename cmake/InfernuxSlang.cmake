# The same compiler is linked into Editor and Player. A prepared SDK and the
# pinned source build expose the same target; neither uses the system PATH.
include_guard(GLOBAL)

set(INFERNUX_SLANG_ROOT "" CACHE PATH "Prepared Slang SDK matching the pinned compiler")
set(INFERNUX_SLANG_VERSION "2026.21")

if(INFERNUX_SLANG_ROOT)
    find_package(slang ${INFERNUX_SLANG_VERSION} EXACT CONFIG REQUIRED
        PATHS "${INFERNUX_SLANG_ROOT}/cmake" NO_DEFAULT_PATH)
    set(INFERNUX_SLANG_TARGET slang::slang)
    set(INFERNUX_SLANG_NOTICES_DIR "${INFERNUX_SLANG_ROOT}/third-party-notices")
else()
    # Slang owns its SPIR-V implementation and standard modules. The engine
    # does not build Slang's graphics runtime or unrelated device toolchains.
    foreach(_option IN ITEMS CUDA OPTIX NVAPI AFTERMATH DXIL GFX SLANG_RHI
            SLANGD SLANGC SLANGI SLANGRT SLANG_GLSLANG SLANG_PROXY
            TESTS EXAMPLES REPLAYER RECORD_REPLAY)
        set(SLANG_ENABLE_${_option} OFF CACHE BOOL "" FORCE)
    endforeach()
    set(SLANG_SLANG_LLVM_FLAVOR DISABLE CACHE STRING "" FORCE)
    set(SLANG_EXCLUDE_DAWN ON CACHE BOOL "" FORCE)
    set(SLANG_EXCLUDE_TINT ON CACHE BOOL "" FORCE)
    set(SLANG_INSTALL_USER_SKILLS OFF CACHE STRING "" FORCE)
    set(SLANG_EMBED_CORE_MODULE ON CACHE BOOL "" FORCE)
    set(SLANG_EMBED_CORE_MODULE_SOURCE ON CACHE BOOL "" FORCE)
    set(SLANG_STANDARD_MODULE_DEVELOP_BUILD OFF CACHE BOOL "" FORCE)
    set(SLANG_LIB_TYPE SHARED CACHE STRING "" FORCE)
    get_directory_property(_infernux_compile_options COMPILE_OPTIONS)
    if(MSVC)
        add_compile_options(/utf-8)
    endif()
    add_subdirectory("${CMAKE_SOURCE_DIR}/external/slang_for_infernux"
                     "${CMAKE_BINARY_DIR}/external/slang" EXCLUDE_FROM_ALL)
    set_property(DIRECTORY PROPERTY COMPILE_OPTIONS ${_infernux_compile_options})
    set(INFERNUX_SLANG_TARGET slang)
    set(INFERNUX_SLANG_MODULE_TARGETS ${SLANG_STANDARD_MODULE_TARGETS})
    set(INFERNUX_SLANG_NOTICES_DIR "${CMAKE_BINARY_DIR}/slang-metadata/third-party-notices")
    add_custom_target(infernux_slang_metadata
        COMMAND ${CMAKE_COMMAND} --install "${slang_BINARY_DIR}"
            --prefix "${CMAKE_BINARY_DIR}/slang-metadata"
            --component metadata --config $<CONFIG>
        COMMENT "Stage Slang's own dependency notices"
        VERBATIM)
endif()
set(INFERNUX_SLANG_MODULE_DIR
    "$<TARGET_FILE_DIR:${INFERNUX_SLANG_TARGET}>/slang-standard-module-${INFERNUX_SLANG_VERSION}")
