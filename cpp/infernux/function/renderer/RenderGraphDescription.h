/**
 * @file RenderGraphDescription.h
 * @brief Data structures for render-graph topology defined from Python
 *
 * These POD structures capture the render graph topology defined in Python,
 * allowing C++ to receive, compile, and execute the graph with automatic
 * Vulkan barrier insertion and transient resource management.
 *
 * Architecture:
 *   Python has "definition authority" — defines pass topology, resource
 *   connections, and per-pass render actions.
 *   C++ has "compilation authority" — performs DAG compilation, dead-pass
 *   culling, barrier generation, and transient resource allocation.
 */

#pragma once

#include "rhi/RhiDescriptors.h"
#include <core/types/ShaderTypes.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace infernux
{
namespace rhi
{
class RenderTexture;
}
class RendererSelection;
namespace rhi
{
class ComputeBuffer;
}

/**
 * @brief Backend-neutral command recorded by a graph pass.
 *
 * Commands describe engine rendering operations
 * rather than Vulkan calls so
 * the same graph artifact can be compiled by another RHI backend later.
 * Compute work
 * is intentionally engine-owned and enters the native RHI
 * RenderGraph directly; Python pipelines do not expose raw
 * compute programs.
 */
enum class GraphCommandType
{
    DrawRenderers,
    DrawSkybox,
    DrawShadowCasters,
    DrawWorldUI,
    DrawScreenUI,
    FullscreenQuad,
    CopyTexture,
    CopyBuffer,
    Present
};

enum class GraphPassType
{
    Raster,
    Copy,
    Present
};

enum class GraphMaterialFilter : uint8_t
{
    All,
    DeferredCompatible,
    DeferredUnsupported,
};

enum class GraphBufferUsage : uint32_t
{
    None = 0,
    Storage = 1 << 0,
    Indirect = 1 << 1,
    TransferSource = 1 << 2,
    TransferDestination = 1 << 3
};

enum class GraphBufferAccessType
{
    StorageRead,
    StorageWrite,
    IndirectRead,
    TransferRead,
    TransferWrite
};

enum class GraphTextureRole : uint8_t
{
    Transient,
    TemporalRead,
    TemporalWrite,
    Persistent,
    Asset
};

enum class GraphTextureAttachment : uint8_t
{
    Color,
    Depth,
    Resolve
};

struct GraphCommandDesc
{
    GraphCommandType type = GraphCommandType::DrawRenderers;
    ShaderCompileTarget shaderTarget = ShaderCompileTarget::Forward;
    GraphMaterialFilter materialFilter = GraphMaterialFilter::All;

    int queueMin = 0;
    int queueMax = 5000;
    std::string sortMode;
    std::string passTag;
    std::string overrideMaterial;
    /// Stable owner; entry/parameter edits do not change graph topology.
    std::shared_ptr<RendererSelection> rendererSelection;

    int32_t lightIndex = 0;
    int screenUIList = 0;
    /// World UI pass filter, intersected with the rendering Camera's mask.
    uint32_t worldUILayerMask = 0xffffffffu;

    std::string shaderName;
    /// Fullscreen raster state. Renderer draws retain their material state.
    bool depthTest = false;
    bool depthWrite = false;
    rhi::CompareFunction depthCompare = rhi::CompareFunction::Always;
    bool alphaBlend = false;
    /// Stable pass-local runtime parameter block. When non-empty,
    /// pushConstants define the block layout and initial values rather than
    /// immutable topology. This payload does not override Material/Renderer,
    /// View or World values by name.
    std::string parameterBlock;
    std::vector<std::pair<std::string, float>> pushConstants;
    std::vector<std::pair<std::string, std::string>> inputBindings;

    std::string sourceResource;
    std::string destinationResource;
    uint64_t copyBytes = 0;
};

/**
 * @brief Revisioned values for one graph-owned runtime parameter block.
 *
 * Parameter updates are intentionally
 * separate from RenderGraphDescription so
 * ordinary effect edits do not rebuild or recompile graph topology.
 */
struct GraphParameterBlockUpdate
{
    std::string id;
    uint64_t revision = 0;
    std::vector<std::pair<std::string, float>> values;
};

// ============================================================================
// Texture Description
// ============================================================================

/**
 * @brief Description of a texture resource in the Python-defined graph
 */
struct GraphTextureDesc
{
    std::string name;                                       ///< Unique resource name
    rhi::PixelFormat format = rhi::PixelFormat::RGBA8UNorm; ///< Backend-neutral pixel format
    bool isBackbuffer = false;                              ///< If true, refers to the scene's main MSAA color target
    bool isDepth = false;                                   ///< If true, this is a depth/stencil texture
    uint32_t width = 0;                                     ///< Custom width (0 = use scene target size)
    uint32_t height = 0;                                    ///< Custom height (0 = use scene target size)
    uint32_t sizeDivisor = 0;                               ///< >0: actual = scene_size / divisor
    uint32_t samples = 1;                                   ///< 0 = inherit frame MSAA, otherwise 1/2/4/8
    GraphTextureRole role = GraphTextureRole::Transient;    ///< Frame-local or one side of a temporal history pair
    std::string temporalKey;                                ///< Stable per-view history identity for temporal resources
    std::shared_ptr<rhi::RenderTexture> renderTexture;      ///< Persistent owner; never a disk GUID
    GraphTextureAttachment attachment = GraphTextureAttachment::Color;
    std::string assetGuid;                                  ///< GUID of a read-only imported Texture asset
    uint32_t depth = 1;                                     ///< Imported Texture3D depth; 1 for ordinary textures
    bool isVolume = false;                                  ///< True when the imported asset is a Texture3D

    /// The root pipeline's conventional depth resource shares its Camera target.
    /// Other names denote distinct images, even at the same viewport dimensions.
    [[nodiscard]] bool IsViewDepth() const noexcept
    {
        return name == "depth" && isDepth && role == GraphTextureRole::Transient && width == 0 && height == 0 &&
               sizeDivisor == 0;
    }
};

struct GraphBufferDesc
{
    std::string name;
    uint64_t byteSize = 0;
    uint32_t usage = static_cast<uint32_t>(GraphBufferUsage::None);
    std::shared_ptr<rhi::ComputeBuffer> computeBuffer; ///< Live GPU owner for an imported buffer.
};

struct GraphBufferAccessDesc
{
    std::string resource;
    GraphBufferAccessType type = GraphBufferAccessType::StorageRead;
};

// ============================================================================
// Pass Description
// ============================================================================

/**
 * @brief Description of a single render pass in the Python-defined graph
 */
struct GraphPassDesc
{
    std::string name; ///< Pass name (must be unique within the graph)
    GraphPassType type = GraphPassType::Raster;

    // === Resource connections ===
    std::vector<std::string> readTextures; ///< Names of textures this pass reads
    /// MRT color outputs: list of (slot, texture_name) pairs.
    /// Slot 0 is the primary color output; higher slots enable deferred / GBuffer.
    std::vector<std::pair<int, std::string>> writeColors;
    std::string writeDepth;   ///< Name of depth output texture
    std::string resolveColor; ///< Optional 1x resolve target for color slot 0
    std::vector<GraphBufferAccessDesc> bufferAccesses;
    bool sideEffect = false;

    // === Clear settings ===
    bool clearColor = false;
    bool clearDepth = false;
    float clearColorR = 0.0f;
    float clearColorG = 0.0f;
    float clearColorB = 0.0f;
    float clearColorA = 1.0f;
    float clearDepthValue = 1.0f;

    // === Typed command IR ===
    // Empty is valid for a resource-only pass. The current executor accepts
    // one command while the IR is intentionally a list for the upcoming
    // raster/compute/copy command-list executor.
    std::vector<GraphCommandDesc> commands;
};

// ============================================================================
// RenderGraph Description (complete topology from Python)
// ============================================================================

/**
 * @brief Complete render graph topology defined by Python
 *
 * This structure is built by the Python RenderGraph API and sent to C++
 * via SceneRenderGraph::ApplyPythonGraph(). C++ uses it to configure
 * the SceneRenderGraph passes and underlying vk::RenderGraph.
 */
struct RenderGraphDescription
{
    std::string name; ///< Graph name for debugging

    /// Monotonic source artifact revision assigned when Python records the graph.
    /// Steady-state frames send only this value to reuse an already-applied graph.
    uint64_t sourceRevision = 0;

    std::vector<GraphTextureDesc> textures; ///< All texture resources
    std::vector<GraphBufferDesc> buffers;   ///< All buffer resources
    std::vector<GraphPassDesc> passes;      ///< All passes in declaration order
    std::string outputTexture;              ///< Name of the final output texture
    /// Terminal linear image before display encoding/overlay. Offscreen
    /// cameras consume this prefix, not a display-encoded screen image.
    std::string linearOutputTexture;
    uint32_t linearOutputPassCount = 0;

    /// MSAA sample count requested by the pipeline (0 = don't change, 1/2/4/8).
    int msaaSamples = 0;
    /// Explicit camera sampling policy, independent of double-buffered resources.
    bool temporalJitter = false;
};

} // namespace infernux
