# The same compiler is linked into Editor and Player. A prepared SDK and the
# pinned source build expose the same target; neither uses the system PATH.
include_guard(GLOBAL)

set(INFERNUX_SLANG_ROOT "" CACHE PATH "Prepared Slang SDK matching the pinned compiler")
set(INFERNUX_SLANG_VERSION "2026.21")
get_filename_component(_infernux_slang_engine_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

# Keep each raw string below MSVC's literal limit while preserving the shader text.
function(_infernux_embed_ray_source path output)
    file(READ "${path}" _source)
    string(REPLACE "\r\n" "\n" _source "${_source}")
    string(PREPEND _source "\n")
    string(LENGTH "${_source}" _length)
    set(_statements "            source.reserve(${_length});\n")
    set(_offset 0)
    while(_offset LESS _length)
        math(EXPR _end "${_offset} + 8192")
        if(_end GREATER _length)
            set(_end "${_length}")
        endif()
        # Move the cut back to a UTF-8 character boundary.
        while(_end LESS _length)
            string(SUBSTRING "${_source}" "${_end}" 1 _character)
            string(HEX "${_character}" _byte)
            if(NOT _byte MATCHES "^[89ab]")
                break()
            endif()
            math(EXPR _end "${_end} - 1")
        endwhile()
        math(EXPR _size "${_end} - ${_offset}")
        string(SUBSTRING "${_source}" "${_offset}" "${_size}" _chunk)
        string(APPEND _statements "            source.append(R\"inxslang(${_chunk})inxslang\");\n")
        set(_offset "${_end}")
    endwhile()
    set("${output}" "${_statements}" PARENT_SCOPE)
endfunction()

# All resident compilers use the same ray modules and rebuild on authored source changes.
function(infernux_add_rt_slang_module compiler_target)
    get_filename_component(_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
    set(_shader_dir "${_root}/cpp/infernux/function/renderer/raytracing/shaders")
    set(_traversal "${_shader_dir}/ray_traversal.slang")
    set(_bsdf "${_shader_dir}/ray_bsdf.slang")
    set(_layout "${_shader_dir}/ray_layout.slang")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_traversal}" "${_bsdf}" "${_layout}")
    _infernux_embed_ray_source("${_traversal}" _ray_traversal_source)
    _infernux_embed_ray_source("${_bsdf}" _ray_bsdf_source)
    _infernux_embed_ray_source("${_layout}" _ray_layout_source)
    set(_deformation "${_shader_dir}/ray_deformation.slang")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_deformation}")
    _infernux_embed_ray_source("${_deformation}" _ray_deformation_source)
    set(_output "${CMAKE_CURRENT_BINARY_DIR}/infernux-slang-modules")
    set(_header "${_output}/SlangRayModules.h")
    configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/SlangRayModules.h.in" "${_header}" @ONLY)
    configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/RayDeformationSource.h.in"
        "${_output}/RayDeformationSource.h" @ONLY)
    set_property(SOURCE "${_root}/cpp/infernux/function/renderer/raytracing/RayWorldGeometry.cpp"
        APPEND PROPERTY OBJECT_DEPENDS "${_output}/RayDeformationSource.h")
    set_property(SOURCE "${_root}/cpp/infernux/function/renderer/raytracing/RayWorldGeometry.cpp"
        APPEND PROPERTY INCLUDE_DIRECTORIES "${_output}")
    set_property(SOURCE "${_root}/cpp/infernux/function/resources/ShaderAsset/SlangCompiler.cpp"
        APPEND PROPERTY OBJECT_DEPENDS "${_header}")
    target_include_directories("${compiler_target}" PRIVATE "${_output}")
endfunction()

if(INFERNUX_SLANG_ROOT)
    # Read the SDK's own compiler identity; never infer it from the consumer's checkout.
    unset(slang_DIR CACHE)
    find_package(slang ${INFERNUX_SLANG_VERSION} EXACT CONFIG REQUIRED
        PATHS "${INFERNUX_SLANG_ROOT}" "${INFERNUX_SLANG_ROOT}/cmake"
              "${INFERNUX_SLANG_ROOT}/lib/cmake/slang"
              "${INFERNUX_SLANG_ROOT}/lib64/cmake/slang"
        NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    if(EMSCRIPTEN)
        # Slang's prepared Wasm SDK exposes static archives, with the same
        # public C++ compilation API as the native shared library.
        add_library(infernux_slang_sdk INTERFACE)
        target_include_directories(infernux_slang_sdk INTERFACE "${INFERNUX_SLANG_ROOT}/include")
        target_compile_definitions(infernux_slang_sdk INTERFACE
            "SLANG_COMPILER_ID=\"${slang_COMPILER_ID}\"")
        target_link_libraries(infernux_slang_sdk INTERFACE "-Wl,--start-group"
            "${INFERNUX_SLANG_ROOT}/lib/libslang-compiler.a"
            "${INFERNUX_SLANG_ROOT}/lib/libcore.a"
            "${INFERNUX_SLANG_ROOT}/lib/libcompiler-core.a"
            "${INFERNUX_SLANG_ROOT}/lib/libSPIRV-Tools-opt.a"
            "${INFERNUX_SLANG_ROOT}/lib/libSPIRV-Tools-link.a"
            "${INFERNUX_SLANG_ROOT}/lib/libSPIRV-Tools.a"
            "${INFERNUX_SLANG_ROOT}/lib/libcmark-gfm.a"
            "${INFERNUX_SLANG_ROOT}/lib/libminiz.a"
            "${INFERNUX_SLANG_ROOT}/lib/liblz4.a" "-Wl,--end-group")
        set(INFERNUX_SLANG_TARGET infernux_slang_sdk)
    else()
        set(INFERNUX_SLANG_TARGET slang::slang)
    endif()
    set(INFERNUX_SLANG_NOTICES_DIR "${INFERNUX_SLANG_ROOT}/third-party-notices")
else()
    # The engine pin owns this release identity. Shallow submodules and source
    # archives have no upstream tags; git describe must not rename the compiler
    # and its standard modules to 0.0.0 on those otherwise identical builds.
    # The host generators and target compiler must embed the same version.
    set(SLANG_VERSION_FULL "${INFERNUX_SLANG_VERSION}" CACHE STRING "Pinned Slang release" FORCE)
    set(SLANG_VERSION_NUMERIC "${INFERNUX_SLANG_VERSION}" CACHE STRING "Pinned Slang numeric release" FORCE)
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
    set(SLANG_EMBED_STANDARD_MODULES ON CACHE BOOL "" FORCE)
    set(SLANG_ENABLE_BUILTIN_SPIRV_TOOLS ON CACHE BOOL "" FORCE)
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
                "-DSLANG_VERSION_FULL=${INFERNUX_SLANG_VERSION}"
                "-DSLANG_VERSION_NUMERIC=${INFERNUX_SLANG_VERSION}"
                -DCMAKE_BUILD_TYPE=Release -DSLANG_LIB_TYPE=SHARED
                -DSLANG_SLANG_LLVM_FLAVOR=DISABLE
                -DSLANG_EXCLUDE_DAWN=ON -DSLANG_EXCLUDE_TINT=ON
                -DSLANG_INSTALL_USER_SKILLS=OFF
                -DSLANG_EMBED_CORE_MODULE=ON -DSLANG_EMBED_CORE_MODULE_SOURCE=ON
                -DSLANG_EMBED_STANDARD_MODULES=OFF
                -DSLANG_ENABLE_BUILTIN_SPIRV_TOOLS=OFF
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
    if(EMSCRIPTEN)
        # SPIRV-Tools disables C++ exceptions locally. Keep its setjmp mode
        # consistent with the resident compiler's Wasm exception ABI.
        add_compile_options(-fwasm-exceptions -sSUPPORT_LONGJMP=wasm)
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
    set(INFERNUX_SLANG_NOTICES_DIR "${CMAKE_BINARY_DIR}/slang-metadata/third-party-notices")
    add_custom_target(infernux_slang_metadata
        COMMAND ${CMAKE_COMMAND} --install "${slang_BINARY_DIR}"
            --prefix "${CMAKE_BINARY_DIR}/slang-metadata"
            --component metadata --config $<CONFIG>
        COMMENT "Stage Slang's own dependency notices"
        VERBATIM)
endif()
