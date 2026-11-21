# Native engine libraries and Python extension modules.

# NO_EXTRAS keeps pybind11 from choosing its own LTO flags. Release IPO/LTO is
# enabled explicitly below where the target's export strategy supports it, so
# compile and link settings always agree.
add_library(InfernuxFoundation SHARED ${INFERNUX_FOUNDATION_SOURCES})
add_library(InfernuxAudioRuntime SHARED ${INFERNUX_AUDIO_RUNTIME_SOURCES})
add_library(InfernuxAssetRuntime SHARED ${INFERNUX_ASSET_RUNTIME_SOURCES})
add_library(InfernuxParticleRuntime SHARED ${INFERNUX_PARTICLE_RUNTIME_SOURCES})
add_library(InfernuxShaderCompiler SHARED ${INFERNUX_SHADER_COMPILER_SOURCES})
target_link_libraries(InfernuxShaderCompiler PRIVATE ${INFERNUX_SLANG_TARGET})
infernux_add_rt_slang_module(InfernuxShaderCompiler)
add_library(InfernuxRenderCore SHARED ${INFERNUX_RENDER_CORE_SOURCES})
add_library(InfernuxRendererRuntime SHARED ${INFERNUX_RENDERER_RUNTIME_SOURCES})
add_library(InfernuxVulkanBackend SHARED ${INFERNUX_VULKAN_BACKEND_SOURCES})

set(_infernux_vulkan_loader Vulkan::Vulkan)
if(INFERNUX_VULKAN_LOADER_TARGET)
    set(_infernux_vulkan_loader ${INFERNUX_VULKAN_LOADER_TARGET})
endif()

# The composition layer is private to its final executable or extension.
add_library(InfernuxRuntime STATIC ${INFERNUX_RUNTIME_SOURCES})
include("${CMAKE_CURRENT_LIST_DIR}/InfernuxPhysicsCook.cmake")
infernux_cook_jolt_gpu_programs(InfernuxRuntime spirv)
infernux_cook_inxphygpu_programs(InfernuxRuntime spirv)
infernux_cook_ui_programs(InfernuxRuntime spirv)
infernux_cook_scene_programs(InfernuxRuntime spirv)
set_target_properties(InfernuxRuntime PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

if(INFERNUX_USE_TARGET_PYTHON)
    add_library(_Infernux MODULE ${BINDING_SOURCES})
    add_library(_InfernuxBootstrap MODULE ${INFERNUX_BOOTSTRAP_BINDING_SOURCE})
    foreach(_infernux_python_module _Infernux _InfernuxBootstrap)
        set_target_properties(${_infernux_python_module} PROPERTIES PREFIX "")
        target_link_libraries(${_infernux_python_module} PRIVATE
            pybind11::headers
            InfernuxTargetPython
        )
    endforeach()
else()
    pybind11_add_module(_Infernux MODULE NO_EXTRAS ${BINDING_SOURCES})
    pybind11_add_module(_InfernuxBootstrap MODULE NO_EXTRAS ${INFERNUX_BOOTSTRAP_BINDING_SOURCE})
endif()

set(INFERNUX_RUNTIME_DLL_TARGETS
    InfernuxFoundation
    InfernuxAudioRuntime
    InfernuxAssetRuntime
    InfernuxParticleRuntime
    InfernuxShaderCompiler
    InfernuxRenderCore
    InfernuxRendererRuntime
    InfernuxVulkanBackend
)

set(INFERNUX_NATIVE_TARGETS
    InfernuxFoundation
    InfernuxAudioRuntime
    InfernuxAssetRuntime
    InfernuxParticleRuntime
    InfernuxShaderCompiler
    InfernuxRenderCore
    InfernuxRendererRuntime
    InfernuxVulkanBackend
    InfernuxRuntime
    _Infernux
    _InfernuxBootstrap
)
foreach(_infernux_target ${INFERNUX_NATIVE_TARGETS})
    target_include_directories(${_infernux_target} PRIVATE
        ${INFERNUX_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/cpp
        ${CMAKE_SOURCE_DIR}/external
        ${CMAKE_SOURCE_DIR}/cpp/infernux
    )
endforeach()
if(INFERNUX_VULKAN_LOADER_TARGET)
    foreach(_infernux_target IN ITEMS InfernuxVulkanBackend InfernuxRuntime _Infernux)
        target_compile_definitions(${_infernux_target} PRIVATE
            VK_NO_PROTOTYPES=1
            INFERNUX_USE_VOLK=1
        )
        if(MSVC)
            target_compile_options(${_infernux_target} PRIVATE /FIvolk.h)
        else()
            target_compile_options(${_infernux_target} PRIVATE
                "$<$<COMPILE_LANGUAGE:C,CXX>:-include>"
                "$<$<COMPILE_LANGUAGE:C,CXX>:volk.h>"
            )
        endif()
    endforeach()
endif()

target_link_libraries(InfernuxFoundation PRIVATE infernux_utf8proc ${CMAKE_DL_LIBS})
target_compile_definitions(InfernuxFoundation
    PRIVATE INFERNUX_FOUNDATION_EXPORTS=1
    INTERFACE "$<$<PLATFORM_ID:Windows>:INFERNUX_FOUNDATION_IMPORTS=1>"
)

target_link_libraries(InfernuxAudioRuntime PUBLIC InfernuxFoundation SDL3::SDL3)
target_link_libraries(InfernuxAudioRuntime PRIVATE dr_libs stb_vorbis)
target_compile_definitions(InfernuxAudioRuntime PRIVATE INFERNUX_AUDIO_RUNTIME_EXPORTS=1)
target_link_libraries(InfernuxAssetRuntime PUBLIC InfernuxFoundation)
target_compile_definitions(InfernuxAssetRuntime PRIVATE INFERNUX_ASSET_RUNTIME_EXPORTS=1)
target_link_libraries(InfernuxParticleRuntime PUBLIC InfernuxRenderCore)
target_compile_definitions(InfernuxParticleRuntime PRIVATE INFERNUX_PARTICLE_RUNTIME_EXPORTS=1)
target_link_libraries(InfernuxShaderCompiler PUBLIC InfernuxFoundation Vulkan::Headers)
target_compile_definitions(InfernuxShaderCompiler PRIVATE INFERNUX_SHADER_COMPILER_EXPORTS=1)
target_link_libraries(InfernuxRenderCore PUBLIC InfernuxFoundation)
target_compile_definitions(InfernuxRenderCore PRIVATE INFERNUX_RENDER_CORE_EXPORTS=1)
target_link_libraries(InfernuxRendererRuntime PUBLIC InfernuxFoundation)
target_compile_definitions(InfernuxRendererRuntime PRIVATE INFERNUX_RENDERER_RUNTIME_EXPORTS=1)
set_target_properties(
    InfernuxFoundation InfernuxAssetRuntime
    InfernuxAudioRuntime InfernuxParticleRuntime InfernuxShaderCompiler
    InfernuxRenderCore InfernuxRendererRuntime InfernuxVulkanBackend
    PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS OFF
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
set_target_properties(InfernuxAudioRuntime PROPERTIES C_VISIBILITY_PRESET hidden)
if(UNIX AND NOT APPLE)
    # Keep linked archives local to the module or library that owns them.
    foreach(_infernux_target IN ITEMS InfernuxFoundation InfernuxShaderCompiler _Infernux)
        target_link_options(${_infernux_target} PRIVATE "LINKER:--exclude-libs,ALL")
    endforeach()
endif()
target_link_libraries(InfernuxVulkanBackend PUBLIC InfernuxRenderCore)
target_compile_definitions(InfernuxVulkanBackend PRIVATE INFERNUX_VULKAN_BACKEND_EXPORTS=1)
target_link_libraries(InfernuxRuntime PUBLIC
    InfernuxFoundation
    InfernuxAudioRuntime
    InfernuxAssetRuntime
    InfernuxParticleRuntime
    InfernuxShaderCompiler
    InfernuxRenderCore
    InfernuxRendererRuntime
    InfernuxVulkanBackend
)
target_link_libraries(InfernuxRuntime PRIVATE lunasvg::lunasvg mikktspace stb)
if(INFERNUX_USE_TARGET_PYTHON)
    target_link_libraries(InfernuxRuntime PRIVATE pybind11::headers InfernuxTargetPython)
else()
    # InfernuxRuntime is ultimately loaded through the _Infernux extension.
    # Python3::Module links the import library where a platform requires it
    # (Windows), but deliberately leaves Python symbols for the interpreter on
    # ELF platforms. pybind11::embed would add a DT_NEEDED libpython entry to
    # the wheel and make an otherwise valid venv depend on its creator prefix.
    target_link_libraries(InfernuxRuntime PRIVATE pybind11::headers Python3::Module)
endif()
target_link_libraries(_Infernux PRIVATE InfernuxRuntime)
target_link_libraries(_InfernuxBootstrap PRIVATE InfernuxFoundation)
if(WIN32)
    target_link_libraries(_InfernuxBootstrap PRIVATE user32)
endif()

if(MSVC)
    # Public runtime headers need conforming C++ member lookup. Propagate the
    # mode to Python bindings and native tests that consume those headers.
    target_compile_options(InfernuxRuntime PUBLIC
        "$<$<COMPILE_LANGUAGE:CXX>:/permissive->"
    )
    # MSBuild already compiles independent projects concurrently. Bound /MP and
    # serialize writes to the shared compiler PDB.
    foreach(_infernux_target ${INFERNUX_NATIVE_TARGETS})
        target_compile_options(${_infernux_target} PRIVATE
            /utf-8
            /MP
            /FS
            /bigobj
            /W4
            "$<$<CONFIG:RelWithDebInfo>:/UNDEBUG>"
        )
    endforeach()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    foreach(_infernux_target ${INFERNUX_NATIVE_TARGETS})
        target_compile_options(${_infernux_target} PRIVATE
            -Wall -Wextra
            "$<$<CONFIG:RelWithDebInfo>:-UNDEBUG>"
        )
    endforeach()
endif()

if(INFERNUX_RELEASE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _infernux_ipo_supported OUTPUT _infernux_ipo_error)
    if(_infernux_ipo_supported)
        set_property(TARGET _Infernux PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
        set_property(TARGET _InfernuxBootstrap PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
        set_property(TARGET InfernuxRuntime PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            # Incremental LTCG reused an inlined pre-change class layout after
            # all source objects had been rebuilt. Use full LTO for final modules.
            foreach(_infernux_module _Infernux _InfernuxBootstrap)
                target_link_options(${_infernux_module} PRIVATE "$<$<CONFIG:Release>:/LTCG>")
            endforeach()
        endif()
        if(WIN32)
            # Auto-exported subsystem DLLs cannot consume MSVC /GL objects.
            foreach(_infernux_dll ${INFERNUX_RUNTIME_DLL_TARGETS})
                set_property(TARGET ${_infernux_dll} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE FALSE)
            endforeach()
        else()
            foreach(_infernux_dll ${INFERNUX_RUNTIME_DLL_TARGETS})
                set_property(TARGET ${_infernux_dll} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
            endforeach()
        endif()
    else()
        message(FATAL_ERROR "Release IPO/LTO is required but unsupported: ${_infernux_ipo_error}")
    endif()
endif()

# The Vulkan layout projection needs headers; Slang owns target reflection.
target_link_libraries(InfernuxShaderCompiler PRIVATE Vulkan::Headers)

foreach(_infernux_target ${INFERNUX_NATIVE_TARGETS})
    # ProfileConfig.h is included by public renderer headers whose class
    # layouts contain profiling state in RelWithDebInfo.  Consumers must see
    # the same configuration-specific value as the library or objects such as
    # SceneRenderExtractor are allocated with the wrong size.
    target_compile_definitions(${_infernux_target} PUBLIC
        $<$<CONFIG:RelWithDebInfo>:INFERNUX_FRAME_PROFILE=1>
    )
    target_compile_definitions(${_infernux_target} PRIVATE
        GLM_FORCE_DEPTH_ZERO_TO_ONE
        GLM_FORCE_LEFT_HANDED
        INFERNUX_FILE_LOGGING=1
        $<$<CONFIG:Release>:INFERNUX_DEFERRED_FILE_LOGGING=1>
        $<$<CONFIG:MinSizeRel>:INFERNUX_DEFERRED_FILE_LOGGING=1>
        $<$<OR:$<CONFIG:Release>,$<CONFIG:MinSizeRel>>:INFERNUX_COMPILE_OUT_DEBUG_LOGS=1>
        ${_infernux_vulkan_validation_definition}
    )
endforeach()

if(INFERNUX_FRAME_PROFILE_TERMINAL)
    target_compile_definitions(InfernuxRuntime PRIVATE INFERNUX_FRAME_PROFILE_TERMINAL=1)
endif()

target_link_libraries(InfernuxRuntime PUBLIC
    assimp::assimp
    SDL3::SDL3
    glm::glm
    imgui
    dr_libs
    ${_infernux_vulkan_loader}
    Jolt
    GPUOpen::VulkanMemoryAllocator
)
target_link_libraries(InfernuxRuntime PRIVATE ZLIB::ZLIBSTATIC)
target_link_libraries(InfernuxFoundation PUBLIC glm::glm)
target_link_libraries(InfernuxRenderCore PUBLIC glm::glm)
target_link_libraries(InfernuxRendererRuntime PUBLIC glm::glm)
target_link_libraries(InfernuxParticleRuntime PUBLIC glm::glm)
target_link_libraries(InfernuxVulkanBackend PUBLIC
    SDL3::SDL3
    glm::glm
    ${_infernux_vulkan_loader}
    GPUOpen::VulkanMemoryAllocator
)

if(NOT INFERNUX_ZSTD_TARGET)
    message(FATAL_ERROR
        "InfernuxFoundation requires the native InxPack Zstandard target; "
        "initialize external/zstd_for_infernux with git submodule update --init --recursive")
endif()
target_link_libraries(InfernuxFoundation PRIVATE ${INFERNUX_ZSTD_TARGET})
