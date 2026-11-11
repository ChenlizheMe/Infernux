# The same compiler is linked into Editor and Player. A prepared SDK and the
# pinned source build expose the same target; neither uses the system PATH.
include_guard(GLOBAL)

set(INFERNUX_SLANG_ROOT "" CACHE PATH "Prepared Slang SDK matching the pinned compiler")
set(INFERNUX_SLANG_VERSION "2026.21")
set(INFERNUX_SLANG_MODULE_NAME "slang-standard-module-${INFERNUX_SLANG_VERSION}")
get_filename_component(_infernux_slang_engine_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

if(INFERNUX_SLANG_ROOT)
    if(EMSCRIPTEN)
        # Slang's prepared Wasm SDK exposes static archives, with the same
        # public C++ compilation API as the native shared library.
        add_library(infernux_slang_sdk INTERFACE)
        target_include_directories(infernux_slang_sdk INTERFACE "${INFERNUX_SLANG_ROOT}/include")
        target_link_libraries(infernux_slang_sdk INTERFACE "-Wl,--start-group"
            "${INFERNUX_SLANG_ROOT}/lib/libslang-compiler.a"
            "${INFERNUX_SLANG_ROOT}/lib/libcore.a"
            "${INFERNUX_SLANG_ROOT}/lib/libcompiler-core.a"
            "${INFERNUX_SLANG_ROOT}/lib/libcmark-gfm.a"
            "${INFERNUX_SLANG_ROOT}/lib/libminiz.a"
            "${INFERNUX_SLANG_ROOT}/lib/liblz4.a" "-Wl,--end-group")
        set(INFERNUX_SLANG_TARGET infernux_slang_sdk)
        set(INFERNUX_SLANG_MODULE_DIR "${INFERNUX_SLANG_ROOT}/lib/${INFERNUX_SLANG_MODULE_NAME}")
    else()
        find_package(slang ${INFERNUX_SLANG_VERSION} EXACT CONFIG REQUIRED
            PATHS "${INFERNUX_SLANG_ROOT}" "${INFERNUX_SLANG_ROOT}/cmake"
                  "${INFERNUX_SLANG_ROOT}/lib64/cmake/slang" NO_DEFAULT_PATH)
        set(INFERNUX_SLANG_TARGET slang::slang)
        set(INFERNUX_SLANG_MODULE_DIR
            "$<TARGET_FILE_DIR:${INFERNUX_SLANG_TARGET}>/${INFERNUX_SLANG_MODULE_NAME}")
    endif()
    set(INFERNUX_SLANG_NOTICES_DIR "${INFERNUX_SLANG_ROOT}/third-party-notices")
else()
    # Slang caches its git-derived version. Refresh it when reconfiguring the
    # pinned source so host generators and target modules keep one identity.
    unset(SLANG_VERSION_FULL CACHE)
    unset(SLANG_VERSION_NUMERIC CACHE)
    unset(SLANG_STANDARD_MODULE_DIR_NAME CACHE)
    unset(SLANG_STANDARD_MODULE_INSTALL_DIR CACHE)
    # Slang owns its SPIR-V implementation and standard modules. The engine
    # does not build Slang's graphics runtime or unrelated device toolchains.
    set(_infernux_slang_host_options)
    foreach(_option IN ITEMS CUDA OPTIX NVAPI AFTERMATH DXIL GFX SLANG_RHI
            SLANGD SLANGC SLANGI SLANGRT SLANG_GLSLANG SLANG_PROXY
            TESTS EXAMPLES REPLAYER RECORD_REPLAY)
        set(SLANG_ENABLE_${_option} OFF CACHE BOOL "" FORCE)
        list(APPEND _infernux_slang_host_options "-DSLANG_ENABLE_${_option}=OFF")
    endforeach()
    set(SLANG_SLANG_LLVM_FLAVOR DISABLE CACHE STRING "" FORCE)
    set(SLANG_EXCLUDE_DAWN ON CACHE BOOL "" FORCE)
    set(SLANG_EXCLUDE_TINT ON CACHE BOOL "" FORCE)
    set(SLANG_INSTALL_USER_SKILLS OFF CACHE STRING "" FORCE)
    set(SLANG_EMBED_CORE_MODULE ON CACHE BOOL "" FORCE)
    set(SLANG_EMBED_CORE_MODULE_SOURCE ON CACHE BOOL "" FORCE)
    set(SLANG_STANDARD_MODULE_DEVELOP_BUILD OFF CACHE BOOL "" FORCE)
    set(_infernux_slang_host_root "${CMAKE_BINARY_DIR}/slang-host")
    if(CMAKE_CROSSCOMPILING AND (NOT SLANG_GENERATORS_PATH OR
            SLANG_GENERATORS_PATH STREQUAL "${_infernux_slang_host_root}/install/bin"))
        # Code generators execute on the build host, never on the target. A
        # fresh Android/Web build prepares them from this same source pin.
        include(ExternalProject)
        set(SLANG_GENERATORS_PATH "${_infernux_slang_host_root}/install/bin"
            CACHE PATH "Slang host code generators" FORCE)
        set(_infernux_slang_generators slang-fiddle slang-embed slang-generate
            slang-generate-monolithic-standard-module-library slang-core-module-cache
            slang-lookup-generator slang-capability-generator slang-spirv-embed-generator slang-bootstrap)
        if(CMAKE_HOST_WIN32)
            set(_infernux_slang_host_suffix .exe)
        else()
            set(_infernux_slang_host_suffix "")
        endif()
        set(_infernux_slang_host_byproducts)
        if(GIT_EXECUTABLE)
            list(APPEND _infernux_slang_host_options "-DGIT_EXECUTABLE=${GIT_EXECUTABLE}")
        endif()
        foreach(_generator IN LISTS _infernux_slang_generators)
            list(APPEND _infernux_slang_host_byproducts
                "${SLANG_GENERATORS_PATH}/${_generator}${_infernux_slang_host_suffix}")
        endforeach()
        ExternalProject_Add(infernux_slang_generators
            SOURCE_DIR "${_infernux_slang_engine_root}/external/slang_for_infernux"
            BINARY_DIR "${_infernux_slang_host_root}/build"
            CMAKE_ARGS ${_infernux_slang_host_options}
                "-USLANG_VERSION_*"
                -DCMAKE_BUILD_TYPE=Release -DSLANG_LIB_TYPE=SHARED
                -DSLANG_SLANG_LLVM_FLAVOR=DISABLE
                -DSLANG_EXCLUDE_DAWN=ON -DSLANG_EXCLUDE_TINT=ON
                -DSLANG_INSTALL_USER_SKILLS=OFF
                -DSLANG_EMBED_CORE_MODULE=ON -DSLANG_EMBED_CORE_MODULE_SOURCE=ON
                -DSLANG_STANDARD_MODULE_DEVELOP_BUILD=OFF
            BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR>
                --config Release --target all-generators --parallel 2
            INSTALL_COMMAND "${CMAKE_COMMAND}" --install <BINARY_DIR>
                --config Release --component generators
                --prefix "${_infernux_slang_host_root}/install"
            BUILD_BYPRODUCTS ${_infernux_slang_host_byproducts}
            BUILD_ALWAYS TRUE)
    endif()
    if(EMSCRIPTEN)
        set(SLANG_LIB_TYPE STATIC CACHE STRING "" FORCE)
        set(SLANG_ENABLE_SPLIT_DEBUG_INFO OFF CACHE BOOL "" FORCE)
    else()
        set(SLANG_LIB_TYPE SHARED CACHE STRING "" FORCE)
    endif()
    get_directory_property(_infernux_compile_options COMPILE_OPTIONS)
    if(MSVC)
        add_compile_options(/utf-8)
    elseif(EMSCRIPTEN)
        add_compile_options(-fwasm-exceptions)
    endif()
    add_subdirectory("${_infernux_slang_engine_root}/external/slang_for_infernux"
                     "${CMAKE_BINARY_DIR}/external/slang" EXCLUDE_FROM_ALL)
    set_property(DIRECTORY PROPERTY COMPILE_OPTIONS ${_infernux_compile_options})
    if(TARGET infernux_slang_generators)
        foreach(_generator IN LISTS _infernux_slang_generators)
            add_dependencies(${_generator} infernux_slang_generators)
        endforeach()
    endif()
    set(INFERNUX_SLANG_TARGET slang)
    set(INFERNUX_SLANG_MODULE_DIR
        "$<TARGET_FILE_DIR:slang>/${INFERNUX_SLANG_MODULE_NAME}")
    set(INFERNUX_SLANG_MODULE_TARGETS ${SLANG_STANDARD_MODULE_TARGETS})
    set(INFERNUX_SLANG_NOTICES_DIR "${CMAKE_BINARY_DIR}/slang-metadata/third-party-notices")
    add_custom_target(infernux_slang_metadata
        COMMAND ${CMAKE_COMMAND} --install "${slang_BINARY_DIR}"
            --prefix "${CMAKE_BINARY_DIR}/slang-metadata"
            --component metadata --config $<CONFIG>
        COMMENT "Stage Slang's own dependency notices"
        VERBATIM)
endif()
