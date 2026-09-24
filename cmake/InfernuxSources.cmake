# Source ownership for the native runtime and Python bindings.

file(GLOB_RECURSE INFERNUX_RUNTIME_SOURCES CONFIGURE_DEPENDS cpp/*.cpp cpp/*.h)
list(APPEND INFERNUX_RUNTIME_SOURCES "${CMAKE_SOURCE_DIR}/external/MikkTSpace/mikktspace.c")

# Runtime implementation and Python bindings are separate binary layers.
list(FILTER INFERNUX_RUNTIME_SOURCES EXCLUDE REGEX "tools/launcher/")
list(FILTER INFERNUX_RUNTIME_SOURCES EXCLUDE REGEX "tools/pybinding/")
list(FILTER INFERNUX_RUNTIME_SOURCES EXCLUDE REGEX "cpp/tests/")

file(GLOB_RECURSE BINDING_SOURCES CONFIGURE_DEPENDS cpp/infernux/tools/pybinding/*.cpp)
set(INFERNUX_BOOTSTRAP_BINDING_SOURCE
    "${CMAKE_SOURCE_DIR}/cpp/infernux/tools/pybinding/BindingInfernuxBootstrap.cpp"
)
list(REMOVE_ITEM BINDING_SOURCES ${INFERNUX_BOOTSTRAP_BINDING_SOURCE})

file(GLOB_RECURSE INFERNUX_FOUNDATION_SOURCES CONFIGURE_DEPENDS
    cpp/infernux/core/*.cpp
    cpp/infernux/core/*.h
    cpp/infernux/platform/filesystem/*.cpp
    cpp/infernux/platform/filesystem/*.h
)
list(APPEND INFERNUX_FOUNDATION_SOURCES
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/resources/InxTexture/StbImage.cpp"
)

# Decode and stream audio without pulling scene/component ownership into a DLL.
set(INFERNUX_AUDIO_RUNTIME_SOURCES
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/audio/AudioDecoders.cpp"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/audio/AudioStreamDecoder.cpp"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/audio/AudioStreamBuffer.cpp"
    "${CMAKE_SOURCE_DIR}/external/stb/stb_vorbis.c"
)

# Asset identity, metadata, and dependency publication form one native ABI.
# Keep this layer independent of scene/renderer ownership so the composition
# archive does not repeatedly compile and link the asset catalog core.
set(INFERNUX_ASSET_RUNTIME_SOURCES
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/resources/AssetDependencyGraph.cpp"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/resources/AssetDependencyGraph.h"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/resources/AssetRuntimeApi.h"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/resources/AssetDatabase/AssetIndex.cpp"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/resources/AssetDatabase/AssetIndex.h"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/resources/InxResource/InxResourceMeta.cpp"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/resources/InxResource/InxResourceMeta.h"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/resources/InxResource/InxResourceMeta.inl"
)

set(INFERNUX_PARTICLE_RUNTIME_SOURCES
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/renderer/particle/ParticleGpuBounds.cpp"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/renderer/particle/ParticleGpuContinuationRuntime.cpp"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/renderer/particle/ParticleGpuContactRuntime.cpp"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/renderer/particle/ParticleGpuCollisionScene.cpp"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/renderer/particle/ParticleGpuCuller.cpp"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/renderer/particle/ParticleGpuDrawRegistry.cpp"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/renderer/particle/ParticleGpuMigrator.cpp"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/renderer/particle/ParticleGpuRibbonTopology.cpp"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/renderer/particle/ParticleGpuRuntime.cpp"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/renderer/particle/ParticleGpuSorter.cpp"
)

set(INFERNUX_SHADER_COMPILER_SOURCES
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/renderer/shader/ShaderCache.cpp"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/renderer/shader/ShaderProgram.cpp"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/renderer/shader/ShaderReflection.cpp"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/resources/ShaderAsset/GlslStageInterfaceEmitter.cpp"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/resources/ShaderAsset/ShaderInfoSchema.cpp"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/resources/ShaderAsset/ShaderPassVariantPlanner.cpp"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/resources/ShaderAsset/ShaderStageLinker.cpp"
)

file(GLOB_RECURSE INFERNUX_RENDER_CORE_SOURCES CONFIGURE_DEPENDS
    cpp/infernux/function/renderer/rhi/*.cpp
    cpp/infernux/function/renderer/rhi/*.h
)
list(APPEND INFERNUX_RENDER_CORE_SOURCES
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/renderer/FullscreenRenderer.cpp"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/renderer/FullscreenRenderer.h"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/renderer/SceneDepthResolver.cpp"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/renderer/SceneDepthResolver.h"
)

set(INFERNUX_RENDERER_RUNTIME_SOURCES
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/scene/SceneRenderer.cpp"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/scene/SceneRenderer.h"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/renderer/Frustum.h"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/renderer/RenderWorld.h"
)

file(GLOB_RECURSE INFERNUX_VULKAN_BACKEND_SOURCES CONFIGURE_DEPENDS
    cpp/infernux/function/renderer/vk/*.cpp
    cpp/infernux/function/renderer/vk/*.h
)
list(APPEND INFERNUX_VULKAN_BACKEND_SOURCES
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/renderer/CaptureService.cpp"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/renderer/CaptureService.h"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/renderer/TransientResourcePool.cpp"
    "${CMAKE_SOURCE_DIR}/cpp/infernux/function/renderer/TransientResourcePool.h"
)

list(REMOVE_ITEM INFERNUX_RUNTIME_SOURCES
    ${INFERNUX_FOUNDATION_SOURCES}
    ${INFERNUX_AUDIO_RUNTIME_SOURCES}
    ${INFERNUX_ASSET_RUNTIME_SOURCES}
    ${INFERNUX_PARTICLE_RUNTIME_SOURCES}
    ${INFERNUX_SHADER_COMPILER_SOURCES}
    ${INFERNUX_RENDER_CORE_SOURCES}
    ${INFERNUX_RENDERER_RUNTIME_SOURCES}
    ${INFERNUX_VULKAN_BACKEND_SOURCES}
)

file(GLOB_RECURSE INFERNUX_HEADERS "${CMAKE_SOURCE_DIR}/cpp/*.h")
foreach(_hdr ${INFERNUX_HEADERS})
    get_filename_component(_hdr_dir "${_hdr}" DIRECTORY)
    list(APPEND INFERNUX_INCLUDE_DIRS "${_hdr_dir}")
endforeach()
list(REMOVE_DUPLICATES INFERNUX_INCLUDE_DIRS)
